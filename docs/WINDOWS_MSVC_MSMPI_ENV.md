# Windows development environment — MSVC / LLVM / Microsoft MPI

**HyBIT 0.9**, baseline `develop/0.9.0` HEAD `cbbcdfa` (H1-C3K1
committed and pushed on 2026-10-17). This is the authoritative Windows
build/verification setup for MPI-enabled Rust tasks H1-C3K2 onward.

## Fixed toolchain; no MinGW/Open MPI substitutions

| Component | Installed, verified on user's Windows 11 machine |
| --- | --- |
| Visual Studio | 2026 Community v18.11.0; MSVC 14.51.362x x64 |
| Developer PowerShell | `C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\Launch-VsDevShell.ps1` |
| Clang / LLVM | Clang 23.1.3; `C:\Program Files\LLVM\bin\clang.exe` |
| libclang | `C:\Program Files\LLVM\bin\libclang.dll` |
| MS-MPI runtime launcher | `C:\Program Files\Microsoft MPI\Bin\mpiexec.exe` |
| MS-MPI headers | `C:\Program Files (x86)\Microsoft SDKs\MPI\Include\` |
| MS-MPI link libraries | `C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\` |
| ABI | `x86_64-pc-windows-msvc`; Microsoft MPI; do **not** use MinGW |

## EVERY new PowerShell session: initialize before Cargo MPI builds

From repository root, **dot-source** the checked-in script:

```powershell
cd D:\Work\hybit\hybit
. .\tools\env\Enter-HybitMsMpi.ps1

cargo fmt --all -- --check
cargo clippy --locked -p hybit --features mpi --all-targets -- -D warnings
cargo test --release --locked -p hybit-distributed --features mpi --lib
```

The leading `. ` (dot followed by space) dot-sources the helper in
the current PowerShell session. Process environment changes also persist with
`&` in the same process, but NOT when launching a separate `pwsh -File` child.
The script calls Visual Studio Developer PowerShell with `-Arch amd64 -HostArch amd64`,
sets `CLANG_PATH`, `LIBCLANG_PATH`, `MSMPI_INC`, `MSMPI_LIB64`, prepends LLVM
and MS-MPI `bin` directories to `PATH` idempotently, and validates `cl.exe`,
`clang.exe`, `libclang.dll`, MPI include/library directories and `mpiexec.exe`.
It does NOT modify system-wide environment settings or install tools.

## Failures already diagnosed; avoid repeating them

- `mpi-sys v0.2.4` -> `bindgen v0.72.1` -> `libclang error` with
  `VCINSTALLDIR = None`, `INCLUDE = None`, `LIB = None`: first initialize
  VS Developer PowerShell; verify `cl.exe` and the x64 SDK variables.
- `Get-Command clang.exe` fails despite setting `CLANG_PATH`: add
  `C:\Program Files\LLVM\bin` to `PATH`. `CLANG_PATH` does not set `PATH`.
  `clang` and `clang.exe` resolve to the same executable on Windows.
- `libclang.dll` is **not** the `clang.exe` command: Rust `bindgen` loads
  `libclang.dll` via `LIBCLANG_PATH`. Verify both separately.
- `mpiexec` is **run-time** and distinct from MS-MPI **SDK** include/lib
  settings for compilation. Use the exact Microsoft MPI executable above.
- Any new terminal or editor terminal may lack initialization. Successful
  commands in a prior terminal do not prove the current terminal has the
  environment. Do not change the Rust target or MPI implementation to hide
  a missing environment initialization.

### Execution checkpoint

After compilation, the existing `benchmarks/scripts/bench-h1c3k1-real-fem-mpi.ps1`
exercises native MS-MPI and strict PCG numerical gates. `thermal1.mtx`
(82,654 DOF, 574,458 expanded nnz) Equal/Jacobi baseline at rtol `1e-11`
passed 12/12 independent FEM cases, plus 4/4 certified small controls.
1/2/4/8-rank median solver times were 1485.4827 / 834.7466 / 448.6422 /
297.4493 ms, respectively; this is **solver-only**, with peak RSS
unmeasured. The thermal1 matrix is **NOT SPD-certified** by its
`ExperimentalUnverified` input policy. H1-C3K2 will compare the same matrix,
RHS, tolerance and preconditioner with alternate partitions, without
changing production defaults.

When a future session resumes HyBIT 0.9 Windows MPI work, read this document
**before** suggesting Cargo commands or MPI alternatives.
