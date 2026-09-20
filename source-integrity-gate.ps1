$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Fail([string]$Message) {
    throw "source integrity gate: $Message"
}

Write-Host "=== HYBIT 0.6.0 SOURCE INTEGRITY GATE ==="

if (-not (Test-Path .\MANIFEST.txt)) { Fail "MANIFEST.txt is missing" }
if (-not (Test-Path .\SOURCE_SHA256.txt)) { Fail "SOURCE_SHA256.txt is missing" }

$manifest = @(Get-Content .\MANIFEST.txt | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
$hashLines = @(Get-Content .\SOURCE_SHA256.txt | Where-Object { $_.Trim() -ne "" })
$hashMap = @{}
foreach ($line in $hashLines) {
    if ($line -notmatch '^([0-9a-fA-F]{64})\s{2}(.+)$') {
        Fail "malformed SOURCE_SHA256.txt line: $line"
    }
    $hashMap[$Matches[2].Replace('\\','/')] = $Matches[1].ToLowerInvariant()
}

$excluded = @("MANIFEST.txt", "SOURCE_SHA256.txt")
$expectedHashed = @($manifest | Where-Object { $excluded -notcontains $_ })

foreach ($path in $manifest) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        Fail "manifest entry is missing: $path"
    }
}

foreach ($path in $expectedHashed) {
    if (-not $hashMap.ContainsKey($path)) {
        Fail "missing SHA-256 entry: $path"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
    if ($actual -ne $hashMap[$path]) {
        Fail "SHA-256 mismatch: $path`n  expected $($hashMap[$path])`n  actual   $actual"
    }
}

$unexpectedHashes = @($hashMap.Keys | Where-Object { $expectedHashed -notcontains $_ })
if ($unexpectedHashes.Count -ne 0) {
    Fail ("hash list contains entries absent from MANIFEST.txt: " + ($unexpectedHashes -join ", "))
}

Write-Host ("manifest files      : {0}" -f $manifest.Count)
Write-Host ("verified SHA-256    : {0}" -f $expectedHashed.Count)
Write-Host "=== HYBIT 0.6.0 SOURCE INTEGRITY GATE PASS ==="
