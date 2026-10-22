[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$OutputPath,
    [ValidateRange(2,12)][int]$Side=8,
    [ValidateRange(0.05,20)][double]$Shift=0.5
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if (Test-Path -LiteralPath $OutputPath) { throw "Refusing overwrite $OutputPath" }
$n=2*$Side*$Side
if ($n -gt 512) { throw 'K8-B numerical SPD screen bounded to 512 DOFs' }
$matrix=New-Object 'double[,]' $n,$n
# Each axial spring contributes w (c^T u)^2 / 2 to the total energy.
# Off-axis c yields mixed-sign, cross-component stiffness couplings.
$steps=@(@(1,0),@(0,1),@(1,1),@(-1,1))
for ($y=0; $y -lt $Side; $y++) {
    for ($x=0; $x -lt $Side; $x++) {
        $a=$y*$Side+$x
        foreach ($step in $steps) {
            $bx=$x+[int]$step[0]; $by=$y+[int]$step[1]
            if ($bx -lt 0 -or $bx -ge $Side -or $by -lt 0 -or $by -ge $Side) { continue }
            $b=$by*$Side+$bx
            [double]$dx=$bx-$x; [double]$dy=$by-$y
            $length=[Math]::Sqrt($dx*$dx+$dy*$dy)
            $c=@(($dx/$length),($dy/$length),(-$dx/$length),(-$dy/$length))
            $id=@((2*$a),(2*$a+1),(2*$b),(2*$b+1))
            for ($i=0; $i -lt 4; $i++) {
                for ($j=0; $j -lt 4; $j++) {
                    $matrix[$id[$i],$id[$j]] += $c[$i]*$c[$j]
                }
            }
        }
    }
}
# Uniform grounding term: positive definite by construction, and exercises
# finite-precision Cholesky for a non-diagonally-dominant FEM-like matrix.
for ($i=0; $i -lt $n; $i++) { $matrix[$i,$i]+=$Shift }
$nz=[Collections.Generic.List[string]]::new()
$inv=[Globalization.CultureInfo]::InvariantCulture
for ($i=0; $i -lt $n; $i++) {
    for ($j=0; $j -le $i; $j++) {
        $v=$matrix[$i,$j]
        if ($v -ne 0.0) {
            $nz.Add(('{0} {1} {2}' -f ($i+1),($j+1),$v.ToString('G17',$inv)))
        }
    }
}
$lines=[Collections.Generic.List[string]]::new()
$lines.Add('%%MatrixMarket matrix coordinate real symmetric')
$lines.Add('% K8-B two-component spring stiffness, plus positive grounding')
$lines.Add(('{0} {0} {1}' -f $n,$nz.Count))
foreach ($v in $nz) { $lines.Add($v) }
[IO.File]::WriteAllLines($OutputPath,[string[]]$lines,[Text.UTF8Encoding]::new($false))
Write-Host "K8-B FEM SPRING MATRIX READY: n=$n lower_nnz=$($nz.Count) shift=$Shift path=$OutputPath"
