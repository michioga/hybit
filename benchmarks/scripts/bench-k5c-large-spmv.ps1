# K5-C single-node MS-MPI original-coefficient SpMV, three rotating policies.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RepoRoot,
    [Parameter(Mandatory)][string]$Matrix,
    [int[]]$Ranks = @(4,8),
    [int[]]$Rotations = @(0,1,2),
    [ValidateRange(1,20)][int]$Warmups = 3,
    [ValidateRange(3,256)][int]$Repeats = 64,
    [ValidateRange(1,1000000)][int]$ProjectedSpmvCalls = 8000,
    [string]$OutputDir = 'D:\Work',
    [string]$Prefix = 'k5c_large_spmv'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = [IO.Path]::GetFullPath($RepoRoot)
$matrixPath = [IO.Path]::GetFullPath($Matrix)
if (-not (Test-Path -LiteralPath $matrixPath -PathType Leaf)) { throw "Missing matrix: $matrixPath" }
if ((Get-Item -LiteralPath $matrixPath).Length -gt 384MB) { throw 'Matrix exceeds K5-C input size cap (384 MiB)' }
if (-not (Test-Path -LiteralPath $OutputDir -PathType Container)) { throw "Missing output dir: $OutputDir" }
if ($Ranks.Count -lt 1 -or $Rotations.Count -lt 1) { throw 'Empty Ranks or Rotations' }
foreach ($r in $Ranks) { if ($r -notin @(1,2,4,8)) { throw "Unsupported rank count $r" } }
foreach ($rotation in $Rotations) { if ($rotation -notin @(0,1,2)) { throw "Unsupported rotation $rotation" } }
$envScript = Join-Path $repo 'tools\env\Enter-HybitMsMpi.ps1'
if (-not (Test-Path -LiteralPath $envScript -PathType Leaf)) { throw 'MS-MPI environment helper missing' }
& { Set-StrictMode -Off; . $envScript }
$mpi = 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe'
$exe = Join-Path $repo 'target\release\examples\mpi_k5c_large_spmv.exe'
if (-not (Test-Path -LiteralPath $mpi -PathType Leaf)) { throw "Missing mpiexec: $mpi" }
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Missing K5-C release example: $exe" }
$summary = Join-Path $OutputDir ("{0}_summary.csv" -f $Prefix)
$platform = Join-Path $OutputDir ("{0}_platform.txt" -f $Prefix)
if (Test-Path -LiteralPath $summary) { throw "Refusing summary overwrite: $summary" }
if (Test-Path -LiteralPath $platform) { throw "Refusing platform overwrite: $platform" }
foreach ($rotation in $Rotations) {
    foreach ($r in $Ranks) {
        $p = Join-Path $OutputDir ("{0}_rotate{1}_r{2}.csv" -f $Prefix,$rotation,$r)
        if (Test-Path -LiteralPath $p) { throw "Refusing output overwrite: $p" }
    }
}
# This metadata is not proof of actual operating clock/temperature or CPU affinity.
$processor = @(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors)
$system = Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
@(
    'K5-C single physical node, Microsoft MS-MPI only',
    "matrix=$matrixPath",
    "input_bytes=$((Get-Item -LiteralPath $matrixPath).Length)",
    "ranks=$($Ranks -join ',')",
    "rotations=$($Rotations -join ',')",
    "warmups=$Warmups",
    "repeats=$Repeats",
    "projected_calls=$ProjectedSpmvCalls",
    "memory_bytes=$($system.TotalPhysicalMemory)",
    'CPU clock, actual core placement, package power, and temperature are NOT measured',
    ($processor | Out-String)
) | Set-Content -LiteralPath $platform -Encoding utf8
$records = [System.Collections.Generic.List[object]]::new()
$methods = @('equal','abtm_region_growth','abtm_multisource')
foreach ($rotation in $Rotations) {
    foreach ($r in $Ranks) {
        $output = Join-Path $OutputDir ("{0}_rotate{1}_r{2}.csv" -f $Prefix,$rotation,$r)
        Write-Host "=== K5-C ORIGINAL CSR: rotate$rotation / $r ranks ==="
        & $mpi -n $r $exe $r $matrixPath $Warmups $Repeats $ProjectedSpmvCalls "rotate$rotation" $output
        if ($LASTEXITCODE -ne 0) { throw "K5-C native run failed rotation=$rotation ranks=$r exit=$LASTEXITCODE" }
        if (-not (Test-Path -LiteralPath $output)) { throw "Missing K5-C output $output" }
        $rows = @(Import-Csv -LiteralPath $output)
        if ($rows.Count -ne 3) { throw "Expected 3 methods, got $($rows.Count)" }
        $seen = @{}
        foreach ($row in $rows) {
            $method = [string]$row.method
            if ($method -notin $methods -or $seen.ContainsKey($method)) { throw "Bad or repeated method: $method" }
            $seen[$method] = $true
            if ([int]$row.ranks -ne $r -or [int]$row.measured_calls -ne $Repeats) { throw "Mismatch in $output" }
            $rel = [double]::Parse($row.worst_rel_error, [Globalization.CultureInfo]::InvariantCulture)
            if ([double]::IsNaN($rel) -or [double]::IsInfinity($rel) -or $rel -gt 1e-11) { throw "SpMV cross-check failed: $rel" }
            $records.Add([pscustomobject]@{
                rotation = $rotation
                ranks = $r
                method = $method
                order_slot = $row.order_slot
                n = $row.n
                nnz = $row.csr_nnz
                global_csr_bytes_per_rank = $row.global_csr_bytes_per_rank
                partition_ms = $row.partition_max_ms
                extraction_ms = $row.extraction_max_ms
                halo_setup_ms = $row.halo_setup_max_ms
                median_spmv_ms = $row.median_max_spmv_ms
                median_halo_ms = $row.median_max_halo_ms
                median_numeric_ms = $row.median_max_numeric_ms
                measured_spmv_sum_ms = $row.measured_spmv_sum_ms
                projected_prepared_total_ms = $row.projected_prepared_total_ms
                ghosts = $row.ghosts
                cut_nnz = $row.cut_nnz
                worst_rel_error = $row.worst_rel_error
            })
        }
    }
}
$records | Export-Csv -LiteralPath $summary -NoTypeInformation -Encoding utf8
if ($records.Count -ne (3 * $Ranks.Count * $Rotations.Count)) { throw 'Aggregated row count mismatch' }
Write-Host "=== K5-C MATRIX SpMV REPEATABILITY PASS: $($records.Count) records, $summary ==="
Write-Host 'Caution: estimates use measured per-SpMV medians multiplied by projected iterations; NOT a solve-many measurement.'
