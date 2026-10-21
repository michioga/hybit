[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RepoRoot,
    [Parameter(Mandatory)][string]$Matrix,
    [int[]]$Ranks = @(4, 8),
    [string[]]$Policies = @('equal', 'region', 'multisource'),
    [ValidateRange(3,512)][int]$Repeats = 64,
    [string]$OutputDir = 'D:\Work',
    [string]$Prefix = 'k7c_sparse_owner'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = [IO.Path]::GetFullPath($RepoRoot)
if (-not (Test-Path -LiteralPath $Matrix -PathType Leaf)) { throw "Missing matrix: $Matrix" }
if (-not (Test-Path -LiteralPath $OutputDir -PathType Container)) { throw "Missing output directory: $OutputDir" }
if ($Prefix -cnotmatch '^[A-Za-z0-9_-]+$') { throw 'Prefix must contain only ASCII letters/digits/-/_' }
foreach ($p in $Policies) { if ($p -cnotin @('equal','region','multisource')) { throw "Invalid K7-C policy: $p" } }
foreach ($r in $Ranks) { if ($r -notin @(1,2,4,8)) { throw "Unsupported rank count: $r" } }
$mpiexec = 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe'
if (-not (Test-Path -LiteralPath $mpiexec)) { throw 'MS-MPI mpiexec is missing' }
$exe = Join-Path $root 'target\release\examples\mpi_k7c_sparse_owner_spmv.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw 'K7-C release example missing. Run the Apply script.' }
$summary = Join-Path $OutputDir "${Prefix}_summary.csv"
if (Test-Path -LiteralPath $summary) { throw "Output exists: $summary" }
$records = @()
Push-Location $root
try {
    & { Set-StrictMode -Off; . .\tools\env\Enter-HybitMsMpi.ps1 }
    foreach ($r in $Ranks) {
        foreach ($p in $Policies) {
            $csv = Join-Path $OutputDir "${Prefix}_${p}_r${r}.csv"
            if (Test-Path -LiteralPath $csv) { throw "Output exists: $csv" }
            Write-Host "=== K7-C root-only distribution: ranks=$r method=$p ==="
            & $mpiexec -n $r $exe $r $Matrix $p $Repeats $csv
            if ($LASTEXITCODE -ne 0) { throw "K7-C failed: ranks=$r method=$p exit=$LASTEXITCODE" }
            $rows = @(Import-Csv -LiteralPath $csv)
            if ($rows.Count -ne 1 -or $rows[0].method -ne $(switch ($p) { equal {'equal'} region {'abtm_region_growth'} multisource {'abtm_multisource'} })) {
                throw "Unexpected K7-C CSV: $csv"
            }
            if ($rows[0].exact_original_verified -ne 'true') { throw 'K7-C original CSR verification missing' }
            $records += $rows[0]
        }
    }
    $records | Export-Csv -LiteralPath $summary -NoTypeInformation -Encoding utf8
    Write-Host "=== K7-C ROOT-SCATTER BENCHMARK SUITE PASS: $($records.Count) records / $summary ==="
} finally {
    Pop-Location
}
