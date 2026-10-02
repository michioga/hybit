$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

function Fail([string]$Message) {
    throw "source integrity gate: $Message"
}

Write-Host "=== HYBIT SOURCE INTEGRITY GATE ==="

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

# In a Git worktree, MANIFEST.txt is also the release-source allowlist.
# Reject stale experimental/generated files that may survive an archive overlay
# and accidentally become tracked before a release commit.
$git = Get-Command git -ErrorAction SilentlyContinue
if ($git -and (Test-Path .\.git)) {
    $tracked = @(git ls-files | ForEach-Object { $_.Trim().Replace('\\','/') } | Where-Object { $_ -ne "" })
    if ($LASTEXITCODE -ne 0) { Fail "git ls-files failed" }

    $unexpectedTracked = @($tracked | Where-Object { $manifest -notcontains $_ })
    if ($unexpectedTracked.Count -ne 0) {
        Fail ("tracked files are absent from MANIFEST.txt: " + ($unexpectedTracked -join ", "))
    }

    $untrackedManifest = @($manifest | Where-Object { $tracked -notcontains $_ })
    if ($untrackedManifest.Count -ne 0) {
        Fail ("MANIFEST.txt contains files that are not tracked by Git: " + ($untrackedManifest -join ", "))
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
Write-Host "=== HYBIT SOURCE INTEGRITY GATE PASS ==="
