$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Invoke-Step([string]$Description, [scriptblock]$Command) {
    Write-Host "-- $Description"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

Write-Host "=== HYBIT 0.5.0 CRATES.IO PACKAGE GATE ==="

Invoke-Step "cargo metadata" { cargo metadata --no-deps --format-version 1 | Out-Null }
Invoke-Step "workspace release tests" { cargo test --workspace --release }

$packages = @(
    "hybit-core",
    "hybit-matrix",
    "hybit-krylov",
    "hybit-precond",
    "hybit-auto",
    "hybit"
)

foreach ($pkg in $packages) {
    Invoke-Step "package $pkg" { cargo package -p $pkg --no-verify --allow-dirty }
    Write-Host "-- package contents: $pkg"
    cargo package -p $pkg --list --allow-dirty
    if ($LASTEXITCODE -ne 0) {
        throw "package list failed for $pkg"
    }
}

# hybit-core has no HyBIT registry dependencies, so its dry-run can be checked
# before any internal crate has been published. Higher crates must be dry-run
# immediately before their real publication, after dependencies are indexed.
Invoke-Step "crates.io dry-run for hybit-core" { cargo publish -p hybit-core --dry-run --allow-dirty }

Write-Host ""
Write-Host "Publish order after GitHub source is pushed:"
Write-Host "  1. hybit-core"
Write-Host "  2. hybit-matrix, hybit-krylov"
Write-Host "  3. hybit-precond"
Write-Host "  4. hybit-auto"
Write-Host "  5. hybit"
Write-Host ""
Write-Host "=== HYBIT 0.5.0 CRATES.IO PACKAGE GATE PASS ==="
