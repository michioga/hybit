# K5-B-O1: policy solve-order rotation diagnostic, Windows MS-MPI only.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RepoRoot,
    [Parameter(Mandatory)][string]$Matrix,
    [int[]]$Ranks = @(4,8),
    [string]$OutputDir = 'D:\Work',
    [string]$Prefix = 'k5b_o1_rotation'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = [IO.Path]::GetFullPath($RepoRoot)
$matrixPath = [IO.Path]::GetFullPath($Matrix)
if (-not (Test-Path -LiteralPath $matrixPath -PathType Leaf)) { throw "Missing matrix: $matrixPath" }
if (-not (Test-Path -LiteralPath $OutputDir -PathType Container)) { throw "Missing output directory: $OutputDir" }
if ($Ranks.Count -eq 0) { throw 'Ranks cannot be empty' }
foreach ($r in $Ranks) { if ($r -notin @(1,2,4,8)) { throw "Unsupported single-node rank count: $r" } }
$envScript = Join-Path $repo 'tools\env\Enter-HybitMsMpi.ps1'
if (-not (Test-Path -LiteralPath $envScript -PathType Leaf)) { throw 'MS-MPI environment helper missing' }
& { Set-StrictMode -Off; . $envScript }
$mpi = 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe'
$exe = Join-Path $repo 'target\release\examples\mpi_k5b_order_rotation.exe'
if (-not (Test-Path -LiteralPath $mpi -PathType Leaf)) { throw 'MS-MPI mpiexec missing' }
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw 'K5B-O1 example missing' }
$methods = @('equal','graph_contiguous','abtm_region_growth','abtm_multisource','abtm_multilevel')
$aggregatePath = Join-Path $OutputDir ("{0}_summary.csv" -f $Prefix)
if (Test-Path -LiteralPath $aggregatePath) { throw "Refusing overwrite: $aggregatePath" }
$records = [System.Collections.Generic.List[object]]::new()
foreach ($shift in 0..4) {
    foreach ($rank in $Ranks) {
        $csv = Join-Path $OutputDir ("{0}_rotate{1}_r{2}.csv" -f $Prefix,$shift,$rank)
        $raw = [IO.Path]::ChangeExtension($csv,'rhs.csv')
        if ((Test-Path -LiteralPath $csv) -or (Test-Path -LiteralPath $raw)) { throw "Refusing overwrite: $csv or $raw" }
        Write-Host "=== K5B-O1 ROTATE$shift / $rank ranks / one physical node ==="
        & $mpi -n $rank $exe $rank $matrixPath 16 3000 1e-10 0.01 $csv "rotate$shift"
        if ($LASTEXITCODE -ne 0) { throw "MS-MPI rotate$shift ranks=$rank failed (exit $LASTEXITCODE)" }
        if (-not (Test-Path -LiteralPath $csv) -or -not (Test-Path -LiteralPath $raw)) { throw "Missing output: $csv or $raw" }
        $rows = @(Import-Csv -LiteralPath $csv)
        $rhsRows = @(Import-Csv -LiteralPath $raw)
        if ($rows.Count -ne 25 -or $rhsRows.Count -ne 320) { throw "Unexpected rows: $($rows.Count)/$($rhsRows.Count)" }
        foreach ($item in $rows) {
            $index = [Array]::IndexOf($methods, [string]$item.method)
            if ($index -lt 0) { throw "Unknown method in CSV: $($item.method)" }
            if ([int]$item.ranks -ne $rank) { throw 'Unexpected ranks in CSV' }
            $rel = [double]::Parse($item.worst_true_rel, [Globalization.CultureInfo]::InvariantCulture)
            if ([double]::IsNaN($rel) -or [double]::IsInfinity($rel) -or $rel -gt 1e-10) { throw 'Residual failed validation' }
            $records.Add([pscustomobject]@{
                rotation = $shift
                ranks = $rank
                method = $item.method
                solve_slot = (($index - $shift + 5) % 5) + 1
                rhs_count = [int]$item.rhs_count
                prepared_ms = $item.prepared_total_approx_ms
                solve_many_ms = $item.solve_many_wall_max_ms
                true_rel = $item.worst_true_rel
            })
        }
        Write-Host "=== K5B-O1 PASS: rotate$shift ranks=$rank ==="
    }
}
$records | Export-Csv -LiteralPath $aggregatePath -NoTypeInformation -Encoding utf8
if ($records.Count -ne (125 * $Ranks.Count)) { throw 'Aggregate row count mismatch' }
Write-Host "=== K5B-O1 ROTATION SWEEP PASS: $($records.Count) records; $aggregatePath ==="
Write-Host 'Numerical checks only; evaluate order bias and prepared time before claiming performance advantage.'
