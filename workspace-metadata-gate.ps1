$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Fail([string]$Message) { throw "workspace metadata gate: $Message" }

Write-Host "=== HYBIT 0.6.0 WORKSPACE METADATA GATE ==="
$json = cargo metadata --no-deps --format-version 1
if ($LASTEXITCODE -ne 0) { Fail "cargo metadata failed with exit code $LASTEXITCODE" }
$metadata = $json | ConvertFrom-Json

$expectedPublished = @("hybit-core", "hybit-matrix", "hybit-krylov", "hybit-precond", "hybit-auto", "hybit")
$expectedAll = @($expectedPublished + "hybit-ffi")
$packages = @($metadata.packages)

foreach ($name in $expectedAll) {
    $pkg = @($packages | Where-Object { $_.name -eq $name })
    if ($pkg.Count -ne 1) { Fail "expected exactly one package named '$name', found $($pkg.Count)" }
    $p = $pkg[0]
    if ($p.version -ne "0.6.0") { Fail "$name version is $($p.version), expected 0.6.0" }
    if ($p.license -ne "MIT") { Fail "$name license is '$($p.license)', expected MIT" }
    if ($p.repository -ne "https://github.com/michioga/hybit") { Fail "$name repository metadata is '$($p.repository)'" }
    if ($p.rust_version -ne "1.73") { Fail "$name rust-version is '$($p.rust_version)', expected 1.73" }

    if ($name -in @("hybit-matrix", "hybit-krylov", "hybit-precond")) {
        $rayon = @($p.dependencies | Where-Object { $_.name -eq "rayon" })
        if ($rayon.Count -ne 1 -or $rayon[0].req -ne "=1.10.0") {
            Fail "$name must pin rayon exactly to =1.10.0 for the Rust 1.73 MSRV"
        }
        $rayonCore = @($p.dependencies | Where-Object { $_.name -eq "rayon-core" })
        if ($rayonCore.Count -ne 1 -or $rayonCore[0].req -ne "=1.12.1") {
            Fail "$name must pin rayon-core exactly to =1.12.1 for the Rust 1.73 MSRV"
        }
    }

    if ($name -eq "hybit-ffi") {
        if ($null -eq $p.publish -or @($p.publish).Count -ne 0) {
            Fail "hybit-ffi must have publish=false"
        }
    } else {
        if ($null -eq $p.publish -or -not (@($p.publish) -contains "crates-io")) {
            Fail "$name must publish only to crates-io"
        }
        if ([string]::IsNullOrWhiteSpace($p.description)) { Fail "$name has no package description" }
    }
}

$unexpected = @($packages | Where-Object { $expectedAll -notcontains $_.name } | ForEach-Object { $_.name })
if ($unexpected.Count -ne 0) { Fail ("unexpected workspace packages: " + ($unexpected -join ", ")) }

Write-Host "version             : 0.6.0"
Write-Host "license             : MIT"
Write-Host "rust-version        : 1.73"
Write-Host "rayon / rayon-core  : =1.10.0 / =1.12.1 (MSRV pin)"
Write-Host "repository          : https://github.com/michioga/hybit"
Write-Host "published crates    : $($expectedPublished -join ', ')"
Write-Host "repository-only     : hybit-ffi"
Write-Host "=== HYBIT 0.6.0 WORKSPACE METADATA GATE PASS ==="
