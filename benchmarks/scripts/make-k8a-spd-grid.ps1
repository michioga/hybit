[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$OutputPath,
    [ValidateRange(4,512)][int]$Side=64,
    [ValidateRange(0.001,5.0)][double]$Shift=0.2
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if (Test-Path -LiteralPath $OutputPath) { throw "Refusing overwrite: $OutputPath" }
$n = $Side*$Side
$nnz = $n + 4*$Side*($Side-1)
$invariant = [Globalization.CultureInfo]::InvariantCulture
$encoding = [Text.UTF8Encoding]::new($false)
$writer = [IO.StreamWriter]::new([IO.Path]::GetFullPath($OutputPath), $false, $encoding)
try {
    $writer.WriteLine('%%MatrixMarket matrix coordinate real general')
    $writer.WriteLine('% K8-A certificate: symmetric, positive diagonal, strictly diagonally dominant')
    $writer.WriteLine("$n $n $nnz")
    for ($y=0; $y -lt $Side; $y++) {
        for ($x=0; $x -lt $Side; $x++) {
            $row = $y*$Side+$x+1
            $degree = 0
            if ($x -gt 0) {$degree++}
            if ($x -lt $Side-1) {$degree++}
            if ($y -gt 0) {$degree++}
            if ($y -lt $Side-1) {$degree++}
            $diagonal = ($degree + $Shift).ToString('G17', $invariant)
            $writer.WriteLine("$row $row $diagonal")
            if ($x -gt 0) { $col = $row-1; $writer.WriteLine("$row $col -1") }
            if ($x -lt $Side-1) { $col = $row+1; $writer.WriteLine("$row $col -1") }
            if ($y -gt 0) { $col = $row-$Side; $writer.WriteLine("$row $col -1") }
            if ($y -lt $Side-1) { $col = $row+$Side; $writer.WriteLine("$row $col -1") }
        }
    }
} finally { $writer.Dispose() }
Write-Host "K8-A ORIGINAL SPD GRID READY: n=$n nnz=$nnz shift=$Shift path=$OutputPath"
