param(
    [Parameter(Mandatory=$true)][string]$Matrix,
    [Parameter(Mandatory=$true)][string]$Coordinates,
    [Parameter(Mandatory=$true)][string]$Rhs,
    [int]$RayonThreads = 8,
    [int]$TargetCoarseDimension = 1536,
    [double]$Tolerance = 1.0e-8,
    [int]$MaxIterations = 3000,
    [int]$MaxReferenceIterations = 240,
    [int]$ExpectedRows = 358065,
    [int]$ExpectedNnz = 28239653,
    [int]$ExpectedAggregates = 233,
    [int]$ExpectedCoarseDimension = 1398
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Fail([string]$Message) { throw "real FEM release gate: $Message" }
function Require-File([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { Fail "$Label file not found: $Path" }
}
function Require-Match([string]$Text, [string]$Pattern, [string]$Label) {
    if ($Text -notmatch $Pattern) { Fail "$Label not found; expected /$Pattern/" }
}
function Extract-FirstDouble([string]$Text, [string]$Label) {
    $pattern = "(?m)^" + [regex]::Escape($Label) + "\s*:\s*([0-9eE+.-]+)\s*$"
    $match = [regex]::Match($Text, $pattern)
    if (-not $match.Success) { Fail "could not parse '$Label'" }
    return [double]::Parse($match.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
}
function Extract-FirstInt([string]$Text, [string]$Label) {
    $pattern = "(?m)^" + [regex]::Escape($Label) + "\s*:\s*([0-9]+)(?:\s|$)"
    $match = [regex]::Match($Text, $pattern)
    if (-not $match.Success) { Fail "could not parse '$Label'" }
    return [int]$match.Groups[1].Value
}
function Invoke-CapturedCargo([string[]]$Arguments) {
    $lines = @(& cargo @Arguments 2>&1)
    $exitCode = $LASTEXITCODE
    $text = $lines -join [Environment]::NewLine
    $lines | ForEach-Object { Write-Host $_ }
    if ($exitCode -ne 0) { Fail "cargo command failed with exit code $exitCode" }
    return $text
}

Require-File $Matrix "matrix"
Require-File $Coordinates "coordinates"
Require-File $Rhs "RHS"
if ($RayonThreads -lt 4) { Fail "RayonThreads must be >= 4 for the production parallel Auto reference" }
if ($Tolerance -le 0.0) { Fail "Tolerance must be > 0" }

$oldThreads = $env:RAYON_NUM_THREADS
$env:RAYON_NUM_THREADS = "$RayonThreads"
try {
    Write-Host "=== HYBIT 0.6.0 REAL FEM RELEASE GATE ==="
    Write-Host "matrix             : $Matrix"
    Write-Host "coordinates        : $Coordinates"
    Write-Host "RHS                : $Rhs"
    Write-Host "Rayon threads      : $RayonThreads"

    Write-Host ""
    Write-Host "== production Structural Auto regression =="
    $auto = Invoke-CapturedCargo @(
        "run", "--release", "-p", "hybit", "--example", "fem_structural_auto", "--",
        "--matrix", $Matrix, "--coords", $Coordinates, "--rhs", $Rhs,
        "--tol", $Tolerance.ToString("R", [Globalization.CultureInfo]::InvariantCulture),
        "--max-iters", "$MaxIterations", "--target-coarse-dim", "$TargetCoarseDimension",
        "--aggregation", "auto", "--spmv", "auto", "--precond", "auto", "--pcg-vectors", "auto"
    )

    Require-Match $auto "(?m)^dimensions\s*:\s*$ExpectedRows\s+x\s+$ExpectedRows\s*$" "reference dimensions"
    Require-Match $auto "(?m)^nnz\s*:\s*$ExpectedNnz\s*$" "reference nnz"
    Require-Match $auto "(?m)^aggregation\s*:\s*Graph\s*$" "Graph aggregation"
    Require-Match $auto "(?m)^SpMV policy\s*:\s*Parallel\s*$" "Parallel SpMV policy"
    Require-Match $auto "(?m)^precond policy\s*:\s*Parallel\s*$" "Parallel preconditioner policy"
    Require-Match $auto "(?m)^PCG vector policy\s*:\s*Parallel\s*$" "Parallel PCG-vector policy"
    Require-Match $auto "(?m)^status\s*:\s*Converged\s*$" "converged status"

    $aggregateCount = Extract-FirstInt $auto "aggregate count"
    $coarseDimension = Extract-FirstInt $auto "coarse dimension"
    $iterations = Extract-FirstInt $auto "iterations"
    $verified = Extract-FirstDouble $auto "verified residual"
    if ($aggregateCount -ne $ExpectedAggregates) { Fail "aggregate count $aggregateCount != expected $ExpectedAggregates" }
    if ($coarseDimension -ne $ExpectedCoarseDimension) { Fail "coarse dimension $coarseDimension != expected $ExpectedCoarseDimension" }
    if ($iterations -gt $MaxReferenceIterations) { Fail "iterations $iterations exceed reference guard $MaxReferenceIterations" }
    if (-not [double]::IsFinite($verified) -or $verified -gt $Tolerance) { Fail "verified residual $verified exceeds tolerance $Tolerance" }

    Write-Host ""
    Write-Host "== prepared solve-many regression =="
    $prepared = Invoke-CapturedCargo @(
        "run", "--release", "-p", "hybit", "--example", "fem_structural_prepared", "--",
        "--matrix", $Matrix, "--coords", $Coordinates, "--rhs", $Rhs,
        "--tol", $Tolerance.ToString("R", [Globalization.CultureInfo]::InvariantCulture),
        "--max-iters", "$MaxIterations", "--target-coarse-dim", "$TargetCoarseDimension",
        "--aggregation", "auto", "--spmv", "auto", "--precond", "auto", "--pcg-vectors", "auto",
        "--repeats", "2"
    )

    Require-Match $prepared "(?m)^aggregation\s*:\s*Graph\s*$" "prepared Graph aggregation"
    Require-Match $prepared "(?m)^SpMV policy\s*:\s*Parallel\s*$" "prepared Parallel SpMV"
    Require-Match $prepared "(?m)^precond policy\s*:\s*Parallel\s*$" "prepared Parallel preconditioner"
    Require-Match $prepared "(?m)^PCG vector policy\s*:\s*Parallel\s*$" "prepared Parallel PCG vectors"
    Require-Match $prepared "(?m)^Solve #1\s*$" "prepared solve #1"
    Require-Match $prepared "(?m)^Solve #2\s*$" "prepared solve #2"
    Require-Match $prepared "(?m)^reused\s*:\s*false\s*$" "first solve non-reused marker"
    Require-Match $prepared "(?m)^reused\s*:\s*true\s*$" "second solve reused marker"
    Require-Match $prepared "(?m)^sequence\s*:\s*1\s*$" "solve sequence 1"
    Require-Match $prepared "(?m)^sequence\s*:\s*2\s*$" "solve sequence 2"

    $statusMatches = [regex]::Matches($prepared, "(?m)^status\s*:\s*Converged\s*$")
    if ($statusMatches.Count -ne 2) { Fail "expected two Converged prepared solves, found $($statusMatches.Count)" }
    $residualMatches = [regex]::Matches($prepared, "(?m)^verified residual\s*:\s*([0-9eE+.-]+)\s*$")
    if ($residualMatches.Count -ne 2) { Fail "expected two prepared verified residuals, found $($residualMatches.Count)" }
    foreach ($m in $residualMatches) {
        $r = [double]::Parse($m.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
        if (-not [double]::IsFinite($r) -or $r -gt $Tolerance) { Fail "prepared verified residual $r exceeds tolerance $Tolerance" }
    }

    Write-Host ""
    Write-Host "reference iterations : $iterations (guard <= $MaxReferenceIterations)"
    Write-Host ("verified residual    : {0:E6}" -f $verified)
    Write-Host "prepared reuse       : PASS"
    Write-Host "performance timing   : recorded above, not a pass/fail criterion"
    Write-Host "=== HYBIT 0.6.0 REAL FEM RELEASE GATE PASS ==="
}
finally {
    $env:RAYON_NUM_THREADS = $oldThreads
}
