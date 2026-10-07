$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
Set-Location $root

function Fail([string]$Message) { throw "OpenMP/Rayon interop gate: $Message" }
function Need([string]$Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) { Fail "$Name is required" }
    return $cmd.Source
}
function Run-Checked([string]$Description, [scriptblock]$Command) {
    Write-Host "-- $Description"
    & $Command
    if ($LASTEXITCODE -ne 0) { Fail "$Description failed with exit code $LASTEXITCODE" }
}

Write-Host "=== HYBIT 0.8.0 OPENMP / RAYON INTEROP GATE ==="

$gcc = Need "gcc.exe"
$gxx = Need "g++.exe"
$gfortran = Need "gfortran.exe"

if (-not (Test-Path ".\target\release\hybit.dll")) {
    Run-Checked "build hybit.dll" { cargo build --release -p hybit-ffi }
}
if (-not (Test-Path ".\build\libhybit.dll.a")) {
    & .\tools\build\build-examples.ps1
    if ($LASTEXITCODE -ne 0) { Fail "external-language build failed" }
}

$tmp = Join-Path $root "target\openmp-interop-gate"
New-Item -ItemType Directory -Force $tmp | Out-Null

$cPath = Join-Path $tmp "env_threads.c"
$cppPath = Join-Path $tmp "api_threads.cpp"
$fPath = Join-Path $tmp "api_threads.f90"

@'
#include "hybit.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    unsigned expected = (unsigned)strtoul(argv[1], NULL, 10);

    hybit_solver_t *solver = NULL;
    int32_t rc = hybit_solver_create(&solver);
    if (rc != HYBIT_OK) return 3;

    uint32_t got = hybit_num_threads();
    hybit_solver_destroy(solver);

    printf("C env thread count: expected=%u got=%u\n", expected, got);
    return got == expected ? 0 : 4;
}
'@ | Set-Content -LiteralPath $cPath -Encoding utf8

@'
#include "hybit.hpp"
#include <omp.h>
#include <cstdint>
#include <iostream>

int main() {
    omp_set_num_threads(5);
    hybit::sync_openmp_threads();

    std::uint32_t got = hybit_num_threads();
    std::cout << "C++ OpenMP API thread count: expected=5 got=" << got << "\n";
    return got == 5 ? 0 : 5;
}
'@ | Set-Content -LiteralPath $cppPath -Encoding utf8

@'
program hybit_openmp_threads
    use, intrinsic :: iso_c_binding
    use omp_lib
    use hybit
    implicit none
    integer(c_int) :: rc
    integer(c_int32_t) :: got

    call omp_set_num_threads(6)
    rc = hybit_set_num_threads(int(omp_get_max_threads(), c_int32_t))
    if (rc /= HYBIT_OK) stop 6

    got = hybit_num_threads()
    print '(A,I0)', 'Fortran OpenMP API thread count: expected=6 got=', got
    if (got /= 6_c_int32_t) stop 7
end program hybit_openmp_threads
'@ | Set-Content -LiteralPath $fPath -Encoding utf8

$cExe = Join-Path $tmp "hybit_omp_env_c.exe"
$cppExe = Join-Path $tmp "hybit_omp_api_cpp.exe"
$fExe = Join-Path $tmp "hybit_omp_api_fortran.exe"

Run-Checked "compile C environment-fallback probe" {
    & $gcc -O2 -I "$root\include" $cPath -L "$root\build" -lhybit -o $cExe
}
Run-Checked "compile C++ OpenMP API probe" {
    & $gxx -O2 -std=c++17 -fopenmp -I "$root\include" $cppPath -L "$root\build" -lhybit -static-libstdc++ -static-libgcc -o $cppExe
}
Run-Checked "compile Fortran OpenMP API probe" {
    & $gfortran -O2 -fopenmp -J $tmp -I $tmp "$root\fortran\hybit.f90" $fPath -L "$root\build" -lhybit -static-libgfortran -static-libgcc -o $fExe
}

Copy-Item -LiteralPath ".\target\release\hybit.dll" -Destination $tmp -Force

$oldOmp = $env:OMP_NUM_THREADS
$oldRayon = $env:RAYON_NUM_THREADS

try {
    $env:OMP_NUM_THREADS = "3"
    Remove-Item Env:\RAYON_NUM_THREADS -ErrorAction SilentlyContinue
    Run-Checked "OMP_NUM_THREADS fallback" { & $cExe 3 }

    $env:OMP_NUM_THREADS = "7"
    $env:RAYON_NUM_THREADS = "2"
    Run-Checked "RAYON_NUM_THREADS precedence" { & $cExe 2 }

    $env:OMP_NUM_THREADS = "9"
    Remove-Item Env:\RAYON_NUM_THREADS -ErrorAction SilentlyContinue
    Run-Checked "C++ omp_set_num_threads bridge" { & $cppExe }

    $env:OMP_NUM_THREADS = "9"
    Remove-Item Env:\RAYON_NUM_THREADS -ErrorAction SilentlyContinue
    Run-Checked "Fortran omp_get_max_threads bridge" { & $fExe }
}
finally {
    if ($null -eq $oldOmp) {
        Remove-Item Env:\OMP_NUM_THREADS -ErrorAction SilentlyContinue
    } else {
        $env:OMP_NUM_THREADS = $oldOmp
    }
    if ($null -eq $oldRayon) {
        Remove-Item Env:\RAYON_NUM_THREADS -ErrorAction SilentlyContinue
    } else {
        $env:RAYON_NUM_THREADS = $oldRayon
    }
}

Write-Host "=== HYBIT 0.8.0 OPENMP / RAYON INTEROP GATE PASS ==="
