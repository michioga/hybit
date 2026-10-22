[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RepoRoot,
    [Parameter(Mandatory)][string]$Matrix,
    [ValidateSet('equal','region','multisource')][string[]]$Policies=@('equal','region','multisource'),
    [int[]]$Ranks=@(4),
    [ValidateSet('jacobi','ic0','block')][string[]]$Preconditioners=@('jacobi','ic0','block'),
    [ValidateRange(2,512)][int]$BlockSize=8,
    [ValidateRange(0,3)][int]$Warmups=1,
    [ValidateRange(1,12)][int]$Repeats=3,
    [ValidateRange(1,3000)][int]$MaxIters=1000,
    [ValidateRange(1.0e-12,1.0e-3)][double]$Tolerance=1e-8,
    [string]$OutputDir='D:\Work',
    [string]$Prefix='k8b_fem_pcg',
    [switch]$KeepCaseCsv
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if (!(Test-Path -LiteralPath $Matrix -PathType Leaf)) { throw "Matrix missing: $Matrix" }
if (!(Test-Path -LiteralPath $OutputDir -PathType Container)) { throw "OutputDir missing: $OutputDir" }
if ($Prefix -cnotmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid Prefix' }
foreach ($r in $Ranks) { if ($r -notin @(1,2,4,8)) { throw "Invalid ranks: $r" } }
$root=[IO.Path]::GetFullPath($RepoRoot)
$exe=Join-Path $root 'target\release\examples\mpi_k8b_fem_pcg.exe'
$mpiexec='C:\Program Files\Microsoft MPI\Bin\mpiexec.exe'
if (!(Test-Path -LiteralPath $mpiexec -PathType Leaf)) { throw 'MS-MPI missing' }
if (!(Test-Path -LiteralPath $exe -PathType Leaf)) { throw "K8-B example not built: $exe" }
$summary=Join-Path $OutputDir "${Prefix}_summary.csv"
if (Test-Path -LiteralPath $summary) { throw "Summary exists: $summary" }
$caseDir=$OutputDir
$tempDir=$null
if (!$KeepCaseCsv) {
    $tempDir=Join-Path ([IO.Path]::GetTempPath()) ('hybit_k8b_' + [Guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($tempDir) | Out-Null
    $caseDir=$tempDir
}
$results=[Collections.Generic.List[object]]::new()
try {
    Push-Location $root
    try {
        & { Set-StrictMode -Off; . .\tools\env\Enter-HybitMsMpi.ps1 }
        $inv=[Globalization.CultureInfo]::InvariantCulture
        $tol=$Tolerance.ToString('G17',$inv)
        foreach ($r in $Ranks) { foreach ($p in $Policies) { foreach ($prec in $Preconditioners) {
            $out=Join-Path $caseDir "${Prefix}_${p}_${prec}_r${r}.csv"
            if (Test-Path -LiteralPath $out) { throw "Case CSV exists: $out" }
            Write-Host "=== K8-B FEM PCG ranks=$r policy=$p preconditioner=$prec block=$BlockSize ==="
            & $mpiexec -n $r $exe $r $Matrix $p $Warmups $Repeats $MaxIters $tol $out $prec $BlockSize
            if ($LASTEXITCODE -ne 0) { throw "K8-B native MPI failed ranks=$r policy=$p precond=$prec" }
            $row=@(Import-Csv -LiteralPath $out)
            if ($row.Count -ne 1 -or $row[0].exact_original_verified -cne 'true') { throw 'Root original CSR verification missing' }
            $rel=[double]::Parse($row[0].root_original_true_rel,$inv)
            if (![double]::IsFinite($rel) -or $rel -gt $Tolerance) { throw "Root true residual failed: $rel" }
            $results.Add($row[0])
            # Durable one-file summary, even if a later case fails.
            $results | Export-Csv -LiteralPath $summary -NoTypeInformation -Encoding utf8
        } } }
        Write-Host "=== K8-B FEM PCG SUITE PASS: $($results.Count) / $summary ==="
        if (!$KeepCaseCsv) { Write-Host '=== Summary only; case CSVs were temporary ===' }
    } finally { Pop-Location }
} finally {
    if ($null -ne $tempDir -and (Test-Path -LiteralPath $tempDir)) {
        Remove-Item -LiteralPath $tempDir -Recurse -Force
    }
}
