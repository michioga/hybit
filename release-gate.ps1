$ErrorActionPreference = "Stop"

function Invoke-Checked([string]$Description, [scriptblock]$Command) {
    Write-Host "-- $Description"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

function Show-PeImports([string]$Path) {
    $objdump = Get-Command objdump.exe -ErrorAction SilentlyContinue
    if (-not $objdump) {
        $objdump = Get-Command x86_64-w64-mingw32-objdump.exe -ErrorAction SilentlyContinue
    }
    if ($objdump) {
        Write-Host "-- PE imports: $Path"
        & $objdump.Source -p $Path |
            Select-String -Pattern "DLL Name:" |
            ForEach-Object { Write-Host ("   " + $_.Line.Trim()) }
    }
}

Write-Host "=== HYBIT 0.5.0 RELEASE GATE ==="
& .\build.ps1
& .\build-examples.ps1

Show-PeImports ".\build\hybit_c.exe"
Show-PeImports ".\build\hybit_cpp.exe"
Show-PeImports ".\build\hybit_fortran.exe"

Invoke-Checked "C ABI runtime example" { .\build\hybit_c.exe }
Invoke-Checked "C++ runtime example" { .\build\hybit_cpp.exe }
Invoke-Checked "Fortran runtime example" { .\build\hybit_fortran.exe }

Write-Host ""
Write-Host "=== HYBIT 0.5.0 RELEASE GATE PASS ==="
