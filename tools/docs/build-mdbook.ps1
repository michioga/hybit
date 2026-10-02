param(
    [switch]$Serve,
    [switch]$Open
)

$ErrorActionPreference = "Stop"

$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $RepoRoot

$Version = "0.5.4"
$ArchiveName = "mdbook-v$Version-x86_64-pc-windows-msvc.zip"
$ExpectedSha256 = "8a6b2421aa522de06d746871d2b8fff9c8d71773467e310a4516d2566e7f2de4"
$ToolDir = Join-Path $RepoRoot "target\tools\mdbook-$Version"
$ArchivePath = Join-Path $ToolDir $ArchiveName
$ExePath = Join-Path $ToolDir "mdbook.exe"

if (-not (Test-Path -LiteralPath $ExePath -PathType Leaf)) {
    New-Item -ItemType Directory -Force -Path $ToolDir | Out-Null

    $url = "https://github.com/rust-lang/mdBook/releases/download/v$Version/$ArchiveName"
    Write-Host "Downloading mdBook $Version from the official rust-lang/mdBook release..."
    Invoke-WebRequest -Uri $url -OutFile $ArchivePath

    $actual = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $ExpectedSha256) {
        Remove-Item -LiteralPath $ArchivePath -Force -ErrorAction SilentlyContinue
        throw "mdBook archive SHA-256 mismatch.`nexpected: $ExpectedSha256`nactual:   $actual"
    }

    Expand-Archive -LiteralPath $ArchivePath -DestinationPath $ToolDir -Force
    Remove-Item -LiteralPath $ArchivePath -Force
}

$versionText = (& $ExePath --version).Trim()
if ($LASTEXITCODE -ne 0 -or $versionText -ne "mdbook v$Version") {
    throw "Unexpected mdBook binary: '$versionText' (expected 'mdbook v$Version')"
}

Write-Host "mdBook: $versionText"

if ($Serve) {
    $argsList = @("serve")
    if ($Open) { $argsList += "--open" }
    & $ExePath @argsList
} else {
    & $ExePath build
}

if ($LASTEXITCODE -ne 0) {
    throw "mdBook command failed with exit code $LASTEXITCODE"
}