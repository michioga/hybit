param(
    [string]$Matrix,
    [string]$Coordinates,
    [string]$Rhs,
    [int]$RayonThreads = 8,
    [switch]$SkipRealFem,
    [switch]$SkipMsrv
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Fail([string]$Message) { throw "release candidate gate: $Message" }
function Invoke-Checked([string]$Description, [scriptblock]$Command) {
    Write-Host ""
    Write-Host "== $Description =="
    & $Command
    if ($LASTEXITCODE -ne 0) { Fail "$Description failed with exit code $LASTEXITCODE" }
}
function Invoke-ScriptChecked([string]$Description, [string]$Script, [object[]]$Arguments = @()) {
    Write-Host ""
    Write-Host "== $Description =="
    & $Script @Arguments
    if ($LASTEXITCODE -ne 0) { Fail "$Description failed with exit code $LASTEXITCODE" }
}

Write-Host "=== HYBIT 0.6.0 RELEASE CANDIDATE GATE ==="

$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) { Fail "git is required" }
$branch = (git branch --show-current).Trim()
if ($LASTEXITCODE -ne 0) { Fail "could not read current git branch" }
if ($branch -ne "develop/0.6.0" -and $branch -ne "main") {
    Fail "run the RC gate from develop/0.6.0 or main; current branch is '$branch'"
}
$dirty = @(git status --porcelain)
if ($LASTEXITCODE -ne 0) { Fail "git status failed" }
if ($dirty.Count -ne 0) {
    Write-Host ($dirty -join [Environment]::NewLine)
    Fail "working tree must be clean"
}
$commit = (git rev-parse HEAD).Trim()
Write-Host "branch              : $branch"
Write-Host "commit              : $commit"
Write-Host "working tree        : clean"

Invoke-ScriptChecked "source integrity" ".\source-integrity-gate.ps1"
Invoke-ScriptChecked "workspace metadata" ".\workspace-metadata-gate.ps1"
Invoke-Checked "cargo fmt --check" { cargo fmt --all -- --check }
Invoke-Checked "cargo clippy -D warnings" { cargo clippy --workspace --all-targets --all-features -- -D warnings }
Invoke-Checked "workspace release tests" { cargo test --workspace --release }
Invoke-Checked "build all workspace targets" { cargo build --workspace --release --all-targets }

if (-not $SkipMsrv) {
    $rustup = Get-Command rustup -ErrorAction SilentlyContinue
    if (-not $rustup) { Fail "rustup is required for the Rust 1.73 MSRV gate; use -SkipMsrv only for an intermediate local run" }
    $installed = @(rustup toolchain list)
    if ($LASTEXITCODE -ne 0) { Fail "rustup toolchain list failed" }
    if (-not ($installed | Where-Object { $_ -match '^1\.73\.0' })) {
        Fail "Rust 1.73.0 toolchain is not installed. Run: rustup toolchain install 1.73.0 --profile minimal"
    }
    Invoke-Checked "Rust 1.73.0 workspace tests" { rustup run 1.73.0 cargo test --workspace --release }
} else {
    Write-Host ""
    Write-Host "== Rust 1.73.0 MSRV gate SKIPPED by request =="
}

Invoke-ScriptChecked "runtime / ABI / language binding gate" ".\release-gate.ps1"
Invoke-ScriptChecked "crates.io package gate" ".\crates-package-gate.ps1"

if (-not $SkipRealFem) {
    if ([string]::IsNullOrWhiteSpace($Matrix) -or [string]::IsNullOrWhiteSpace($Coordinates) -or [string]::IsNullOrWhiteSpace($Rhs)) {
        Fail "real FEM gate requires -Matrix, -Coordinates, and -Rhs; use -SkipRealFem only for an intermediate local run"
    }
    Write-Host ""
    Write-Host "== real FEM structural regression =="
    & .\real-fem-release-gate.ps1 -Matrix $Matrix -Coordinates $Coordinates -Rhs $Rhs -RayonThreads $RayonThreads
    if ($LASTEXITCODE -ne 0) { Fail "real FEM structural regression failed with exit code $LASTEXITCODE" }
} else {
    Write-Host ""
    Write-Host "== real FEM structural regression SKIPPED by request =="
}

$dirtyAfter = @(git status --porcelain)
if ($LASTEXITCODE -ne 0) { Fail "final git status failed" }
if ($dirtyAfter.Count -ne 0) {
    Write-Host ($dirtyAfter -join [Environment]::NewLine)
    Fail "release gate modified the worktree; generated files such as Cargo.lock must be deliberately committed or ignored before release"
}
$commitAfter = (git rev-parse HEAD).Trim()
if ($commitAfter -ne $commit) { Fail "HEAD changed during the release gate" }

Write-Host ""
Write-Host "=== HYBIT 0.6.0 RELEASE CANDIDATE GATE PASS ==="
Write-Host "validated branch    : $branch"
Write-Host "validated commit    : $commit"
Write-Host "final worktree      : clean"
if ($SkipMsrv -or $SkipRealFem) {
    Write-Host "NOTE: one or more release-critical gates were explicitly skipped; do not tag/publish this run."
} else {
    Write-Host "This commit is eligible to become the 0.6.0 release candidate."
}
