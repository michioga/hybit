$ErrorActionPreference = "Stop"

function Invoke-Checked([string]$Description, [scriptblock]$Command) {
    Write-Host "-- $Description"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

Write-Host "== HyBIT 0.5.0 build =="
Invoke-Checked "cargo test --release" { cargo test --release }
Invoke-Checked "build C ABI DLL" { cargo build --release -p hybit-ffi }
Invoke-Checked "run basic example" { cargo run --release -p hybit --example basic }
Invoke-Checked "run hybrid example" { cargo run --release -p hybit --example hybrid }
Invoke-Checked "run multiregion example" { cargo run --release -p hybit --example multiregion }
Invoke-Checked "run prepared solve-many example" { cargo run --release -p hybit --example prepared }

Write-Host ""
Write-Host "Built C ABI DLL under target/release and headers under include/."
Write-Host "To build C/C++/Fortran examples (MSVC Rust + MinGW is supported):"
Write-Host "  .\build-examples.ps1"
