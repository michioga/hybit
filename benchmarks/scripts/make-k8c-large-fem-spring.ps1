[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$OutputPath,
    [ValidateRange(17,60)][int]$Side=32,
    [ValidateRange(0.05,20)][double]$Shift=0.5
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if (Test-Path -LiteralPath $OutputPath) { throw "Refusing overwrite $OutputPath" }
$n=2*$Side*$Side
if ($n -gt 12000) { throw 'K8-C generator size cap exceeded' }
$coeff=[Collections.Generic.Dictionary[long,double]]::new()
function AddEntry([int]$row,[int]$col,[double]$v) {
    if ($row -lt $col) { $tmp=$row; $row=$col; $col=$tmp }
    $key=[long]$row*[long]$n + [long]$col
    $current=0.0
    if ($coeff.TryGetValue($key,[ref]$current)) { $coeff[$key]=$current+$v }
    else { $coeff.Add($key,$v) }
}
# Axial and diagonal two-component spring contributions, identical energy to K8-B.
$steps=@(@(1,0),@(0,1),@(1,1),@(-1,1))
for ($y=0; $y -lt $Side; $y++) {
    for ($x=0; $x -lt $Side; $x++) {
        $a=$y*$Side+$x
        foreach ($step in $steps) {
            $bx=$x+[int]$step[0]; $by=$y+[int]$step[1]
            if ($bx -lt 0 -or $bx -ge $Side -or $by -ge $Side) { continue }
            $b=$by*$Side+$bx
            [double]$dx=$bx-$x; [double]$dy=$by-$y
            $len=[Math]::Sqrt($dx*$dx+$dy*$dy)
            [double[]]$c=@(($dx/$len),($dy/$len),(-$dx/$len),(-$dy/$len))
            [int[]]$id=@((2*$a),(2*$a+1),(2*$b),(2*$b+1))
            for ($i=0; $i -lt 4; $i++) {
                for ($j=0; $j -le $i; $j++) {
                    AddEntry $id[$i] $id[$j] ($c[$i]*$c[$j])
                }
            }
        }
    }
}
for ($i=0; $i -lt $n; $i++) { AddEntry $i $i $Shift }
$keys=[long[]]@($coeff.Keys)
[Array]::Sort($keys)
$inv=[Globalization.CultureInfo]::InvariantCulture
# Format lines incrementally, keep only nonzero lower entries.
$entries=[Collections.Generic.List[string]]::new()
foreach ($key in $keys) {
    $v=$coeff[$key]
    if ($v -eq 0.0) { continue }
    [long]$row=[Math]::Floor($key / [double]$n)
    [long]$col=$key % [long]$n
    $entries.Add(('{0} {1} {2}' -f ($row+1),($col+1),$v.ToString('G17',$inv)))
}
$full=[IO.Path]::GetFullPath($OutputPath)
$lines=[Collections.Generic.List[string]]::new()
$lines.Add('%%MatrixMarket matrix coordinate real symmetric')
$lines.Add('% K8-C two-DOF spring: exact symmetric outer products + positive diagonal shift')
$lines.Add(('{0} {0} {1}' -f $n,$entries.Count))
foreach ($line in $entries) { $lines.Add($line) }
[IO.File]::WriteAllLines($full,[string[]]$lines,[Text.UTF8Encoding]::new($false))
Write-Host "K8-C FEM MATRIX READY: n=$n lower_nnz=$($entries.Count) shift=$Shift path=$full"
