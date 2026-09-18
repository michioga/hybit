$ErrorActionPreference = "Stop"

function Invoke-ScriptChecked([string]$Description, [string]$ScriptPath) {
    Write-Host ""
    Write-Host "== $Description =="
    & $ScriptPath
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

Write-Host "=== HYBIT 0.5.0 PUBLIC RELEASE GATE ==="
Invoke-ScriptChecked "runtime / ABI release gate" ".\release-gate.ps1"
Invoke-ScriptChecked "crates.io package gate" ".\crates-package-gate.ps1"
Write-Host ""
Write-Host "=== HYBIT 0.5.0 PUBLIC RELEASE GATE PASS ==="
