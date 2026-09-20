$ErrorActionPreference = "Stop"

function Invoke-Checked([string]$Description, [scriptblock]$Command) {
    Write-Host "-- $Description"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

Write-Host "== HyBIT 0.6.0 external-language examples =="

if (-not (Test-Path "target/release/hybit.dll")) {
    Invoke-Checked "build Rust cdylib" { cargo build --release -p hybit-ffi }
}

if (Test-Path "build") {
    Remove-Item -Recurse -Force "build"
}

Invoke-Checked "configure CMake examples" { cmake -S examples -B build }
Invoke-Checked "build C/C++/Fortran examples" { cmake --build build --config Release }

Write-Host ""
Write-Host "Build complete. hybit.dll is copied beside each example executable."
Write-Host "Run:"
Write-Host "  .\build\hybit_c.exe"
Write-Host "  .\build\hybit_cpp.exe"
Write-Host "  .\build\hybit_fortran.exe"
