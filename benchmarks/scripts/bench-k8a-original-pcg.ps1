[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RepoRoot,
    [Parameter(Mandatory)][string]$Matrix,
    [int[]]$Ranks=@(4),
    [string[]]$Policies=@('equal','region','multisource'),
    [ValidateRange(0,3)][int]$Warmups=1,
    [ValidateRange(1,12)][int]$Repeats=3,
    [ValidateRange(1,3000)][int]$MaxIters=1000,
    [ValidateRange(1.0e-12,1.0e-3)][double]$Tolerance=1e-8,
    [string]$OutputDir='D:\Work',
    [string]$Prefix='k8a_original_pcg',
    [switch]$KeepCaseCsv
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if (!(Test-Path -LiteralPath $Matrix -PathType Leaf)) { throw "Missing matrix $Matrix" }
if (!(Test-Path -LiteralPath $OutputDir -PathType Container)) { throw "Missing output dir $OutputDir" }
if ($Prefix -cnotmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid prefix' }
foreach ($r in $Ranks) { if ($r -notin @(1,2,4,8)) { throw "Invalid ranks: $r" } }
foreach ($p in $Policies) { if ($p -cnotin @('equal','region','multisource')) { throw "Invalid policy: $p" } }
$root=[IO.Path]::GetFullPath($RepoRoot)
$mpiexec='C:\Program Files\Microsoft MPI\Bin\mpiexec.exe'
$exe=Join-Path $root 'target\release\examples\mpi_k8a_sparse_pcg.exe'
if (!(Test-Path -LiteralPath $mpiexec -PathType Leaf)) { throw 'MS-MPI mpiexec missing' }
if (!(Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Missing K8-A release example: $exe" }
$summary=Join-Path $OutputDir "${Prefix}_summary.csv"
if (Test-Path -LiteralPath $summary) { throw "Refusing overwrite: $summary" }
# One durable summary CSV per suite by default. Native MPI still writes one CSV
# per case, but these are transient unless explicitly kept for debugging.
$caseDir=$OutputDir
$transientDir=$null
if (-not $KeepCaseCsv) {
    $transientDir=Join-Path ([IO.Path]::GetTempPath()) ('hybit_k8a_' + [guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($transientDir) | Out-Null
    $caseDir=$transientDir
}
$records=[Collections.Generic.List[object]]::new()
try {
    Push-Location $root
    try {
        & { Set-StrictMode -Off; . .\tools\env\Enter-HybitMsMpi.ps1 }
        foreach ($r in $Ranks) {
            foreach ($p in $Policies) {
                $file=Join-Path $caseDir "${Prefix}_${p}_r${r}.csv"
                if (Test-Path -LiteralPath $file) { throw "Refusing overwrite: $file" }
                Write-Host "=== K8-A ORIGINAL PCG: ranks=$r policy=$p ==="
                $invariant=[Globalization.CultureInfo]::InvariantCulture
                $tolString=$Tolerance.ToString('G17',$invariant)
                & $mpiexec -n $r $exe $r $Matrix $p $Warmups $Repeats $MaxIters $tolString $file
                if ($LASTEXITCODE -ne 0) { throw "K8-A MS-MPI PCG failed: r=$r p=$p exit=$LASTEXITCODE" }
                $csv=@(Import-Csv -LiteralPath $file)
                if ($csv.Count -ne 1 -or $csv[0].exact_original_verified -cne 'true') { throw "Root verification missing $file" }
                $rel=[double]::Parse($csv[0].root_original_true_rel, $invariant)
                if (-not [double]::IsFinite($rel) -or $rel -gt $Tolerance) { throw "Root residual failed in $file" }
                $records.Add($csv[0])
                # Keep an up-to-date aggregate even if a subsequent MPI case fails.
                $records | Export-Csv -LiteralPath $summary -NoTypeInformation -Encoding utf8
            }
        }
        Write-Host "=== K8-A ORIGINAL PCG SUITE PASS: $($records.Count) / $summary ==="
        if (-not $KeepCaseCsv) {
            Write-Host '=== Only summary CSV retained; temporary per-case CSV files removed ==='
        }
    } finally { Pop-Location }
} finally {
    if ($null -ne $transientDir -and (Test-Path -LiteralPath $transientDir -PathType Container)) {
        Remove-Item -LiteralPath $transientDir -Recurse -Force
    }
}
