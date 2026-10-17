# Dot-source this file from the same PowerShell process that will run Cargo:
#   . .\tools\env\Enter-HybitMsMpi.ps1
# This initializes process environment for subsequent Cargo commands.
# Do not run it in a separate 'pwsh -File' child process.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'

$vsDeveloper = 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\Launch-VsDevShell.ps1'
$llvmBin = 'C:\Program Files\LLVM\bin'
$mpiBin = 'C:\Program Files\Microsoft MPI\Bin'
$mpiInclude = 'C:\Program Files (x86)\Microsoft SDKs\MPI\Include\'
$mpiLibrary = 'C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\'
$mpiexec = Join-Path $mpiBin 'mpiexec.exe'
$clang = Join-Path $llvmBin 'clang.exe'
$libclang = Join-Path $llvmBin 'libclang.dll'

foreach ($path in @($vsDeveloper, $clang, $libclang, $mpiexec)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "HyBIT MS-MPI environment missing file: $path"
    }
}
foreach ($path in @($mpiInclude, $mpiLibrary)) {
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        throw "HyBIT MS-MPI SDK missing directory: $path"
    }
}

# Visual Studio Developer PowerShell must initialize VCINSTALLDIR, INCLUDE, LIB.
# Merely putting cl.exe on PATH does not initialize MSVC's SDK header/library paths.
& $vsDeveloper -Arch amd64 -HostArch amd64
if (-not $env:VCINSTALLDIR -or -not $env:INCLUDE -or -not $env:LIB) {
    throw 'Visual Studio 2026 x64 developer environment initialization failed'
}

$env:CLANG_PATH = $clang
$env:LIBCLANG_PATH = $llvmBin
$env:MSMPI_INC = $mpiInclude
$env:MSMPI_LIB64 = $mpiLibrary

# Add LLVM and MS-MPI executable directories once; path lookup is separate from
# CLANG_PATH/LIBCLANG_PATH, and MPI build-time SDK paths are separate from mpiexec.
foreach ($path in @($llvmBin, $mpiBin)) {
    $parts = @($env:PATH -split ';' | Where-Object { $_ -ne '' })
    if (-not ($parts | Where-Object { $_.TrimEnd('\') -ieq $path.TrimEnd('\') })) {
        $env:PATH = "$path;$env:PATH"
    }
}

$cl = Get-Command cl.exe -CommandType Application -ErrorAction Stop
$clangCmd = Get-Command clang.exe -CommandType Application -ErrorAction Stop
if ([IO.Path]::GetFullPath($clangCmd.Source) -ine [IO.Path]::GetFullPath($clang)) {
    throw "Unexpected clang.exe resolved: $($clangCmd.Source)"
}
if (-not $cl.Source.StartsWith('C:\Program Files\Microsoft Visual Studio\18\Community\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "Unexpected cl.exe outside Visual Studio 2026 Community: $($cl.Source)"
}

Write-Host '=== HYBIT WINDOWS MSVC / LLVM / MS-MPI READY ==='
Write-Host "MSVC           : $($cl.Source)"
Write-Host "Clang          : $env:CLANG_PATH"
Write-Host "libclang       : $libclang"
Write-Host "MS-MPI include : $env:MSMPI_INC"
Write-Host "MS-MPI library : $env:MSMPI_LIB64"
Write-Host "mpiexec        : $mpiexec"
& $env:CLANG_PATH --version | Select-Object -First 1
Write-Host 'MPI backend    : MS-MPI ONLY (do not switch to MinGW/Open MPI)'
