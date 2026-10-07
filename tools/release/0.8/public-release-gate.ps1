$ErrorActionPreference = "Stop"
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
Set-Location $RepoRoot

function Invoke-ScriptChecked([string]$Description, [string]$ScriptPath) {
    Write-Host ""
    Write-Host "== $Description =="
    & $ScriptPath
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

Write-Host "=== HYBIT 0.8.0 PUBLIC RELEASE GATE ==="
Invoke-ScriptChecked "runtime / ABI release gate" (Join-Path $PSScriptRoot "release-gate.ps1")
Invoke-ScriptChecked "crates.io package gate" (Join-Path $PSScriptRoot "crates-package-gate.ps1")
Write-Host ""
Write-Host "=== HYBIT 0.8.0 PUBLIC RELEASE GATE PASS ==="
