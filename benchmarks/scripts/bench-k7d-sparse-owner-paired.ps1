[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RepoRoot,
    [Parameter(Mandatory)][string]$Matrix,
    [int[]]$Ranks = @(4,8),
    [string[]]$Policies = @('equal','region','multisource'),
    [ValidateRange(3,512)][int]$Repeats = 32,
    [ValidateRange(1,10)][int]$Cycles = 3,
    [string]$OutputDir = 'D:\Work',
    [string]$Prefix = 'k7d_setup_pair'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = [IO.Path]::GetFullPath($RepoRoot)
if (-not (Test-Path -LiteralPath $Matrix -PathType Leaf)) { throw "Missing matrix: $Matrix" }
if (-not (Test-Path -LiteralPath $OutputDir -PathType Container)) { throw "Missing output dir: $OutputDir" }
if ($Prefix -cnotmatch '^[A-Za-z0-9_-]+$') { throw 'Unsafe output prefix' }
foreach ($r in $Ranks) { if ($r -notin @(1,2,4,8)) { throw "Unsupported rank count: $r" } }
foreach ($p in $Policies) { if ($p -cnotin @('equal','region','multisource')) { throw "Unsupported policy: $p" } }
$mpiexec = 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe'
$bin = Join-Path $root 'target\release\examples'
$executables = @{
    baseline = (Join-Path $bin 'mpi_k7c_sparse_owner_spmv_k7d_baseline.exe')
    optimized = (Join-Path $bin 'mpi_k7c_sparse_owner_spmv.exe')
}
foreach ($exe in $executables.Values) { if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Missing executable: $exe" } }
if (-not (Test-Path -LiteralPath $mpiexec -PathType Leaf)) { throw 'Microsoft mpiexec.exe missing' }
$summary = Join-Path $OutputDir "${Prefix}_pairs.csv"
if (Test-Path -LiteralPath $summary) { throw "Refusing overwrite: $summary" }
$records = @()
$inv = [Globalization.CultureInfo]::InvariantCulture
function Value([object]$record,[string]$field) {
    [double]::Parse($record.$field, $inv)
}
Push-Location $root
try {
    & { Set-StrictMode -Off; . .\tools\env\Enter-HybitMsMpi.ps1 }
    foreach ($r in $Ranks) {
        foreach ($p in $Policies) {
            for ($cycle = 1; $cycle -le $Cycles; $cycle++) {
                # Reverse launch order for every other pair to expose slot bias.
                $order = if ($cycle % 2 -eq 1) { @('baseline','optimized') } else { @('optimized','baseline') }
                $pair = @{}
                foreach ($kind in $order) {
                    $file = Join-Path $OutputDir "${Prefix}_${p}_r${r}_c${cycle}_${kind}.csv"
                    if (Test-Path -LiteralPath $file) { throw "Refusing overwrite: $file" }
                    Write-Host "=== K7-D PAIR: ranks=$r policy=$p cycle=$cycle executable=$kind ==="
                    & $mpiexec -n $r $executables[$kind] $r $Matrix $p $Repeats $file
                    if ($LASTEXITCODE -ne 0) { throw "MPI execution failed ($kind ranks=$r policy=$p cycle=$cycle): exit=$LASTEXITCODE" }
                    $csv = @(Import-Csv -LiteralPath $file)
                    if ($csv.Count -ne 1 -or $csv[0].exact_original_verified -ne 'true') { throw "Missing exact-original verification in $file" }
                    if ((Value $csv[0] 'worst_rel') -gt 1e-11) { throw "SpMV accuracy failure: $file" }
                    $pair[$kind] = $csv[0]
                }
                $a = $pair['baseline']; $b = $pair['optimized']
                foreach ($field in @('matrix','ranks','method','n','nnz','ghosts','verified_original_rows','verified_original_nnz','sparse_owner_entries','sparse_owner_bytes_sum','replicated_owner_baseline_bytes')) {
                    if ($a.$field -cne $b.$field) { throw "K7-D baseline/optimized result mismatch in $field, policy=$p ranks=$r" }
                }
                $setupA = (Value $a 'scatter_max_ms') + (Value $a 'halo_setup_max_ms')
                $setupB = (Value $b 'scatter_max_ms') + (Value $b 'halo_setup_max_ms')
                if ($setupA -le 0) { throw 'Invalid baseline setup latency' }
                $gain = 100.0 * ($setupA - $setupB) / $setupA
                $records += [pscustomobject]@{
                    matrix = $Matrix; ranks = $r; policy = $p; cycle = $cycle
                    launch_order = ($order -join '->')
                    baseline_scatter_ms = (Value $a 'scatter_max_ms')
                    optimized_scatter_ms = (Value $b 'scatter_max_ms')
                    baseline_halo_setup_ms = (Value $a 'halo_setup_max_ms')
                    optimized_halo_setup_ms = (Value $b 'halo_setup_max_ms')
                    baseline_setup_ms = $setupA; optimized_setup_ms = $setupB
                    setup_gain_percent = $gain
                    baseline_spmv_ms = (Value $a 'median_spmv_max_ms')
                    optimized_spmv_ms = (Value $b 'median_spmv_max_ms')
                    owner_bytes_sum = $b.sparse_owner_bytes_sum
                    exact_original = $b.exact_original_verified
                }
                Write-Host ('PAIR ranks={0} policy={1} cycle={2}: baseline={3:N3} ms optimized={4:N3} ms setup_gain={5:N2}%' -f $r,$p,$cycle,$setupA,$setupB,$gain)
            }
        }
    }
    $records | Export-Csv -LiteralPath $summary -NoTypeInformation -Encoding utf8
    Write-Host "=== K7-D PAIRED BENCHMARK PASS: $($records.Count) pairs / $summary ==="
} finally { Pop-Location }