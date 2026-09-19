# Publishing HyBIT 0.6.0

The public repository is:

`https://github.com/michioga/hybit`

The user-facing Rust crate is:

`https://crates.io/crates/hybit`

HyBIT is a Cargo workspace. The facade crate depends on internal crates, so crates.io publication must follow dependency order. `hybit-ffi` is intentionally not published to crates.io in 0.6.0.

## Development branch and release candidate

Active 0.6 work is kept on `develop/0.6.0`. Keep `main` at the last validated public release until the 0.6 release candidate passes the full release gate. Once the exact candidate commit is validated, merge that commit to `main`, rerun the gate from the final source tree, and only then create/publish immutable release artifacts.

## Before publishing

Run the local release gate:

```powershell
.\build.ps1
.\release-gate.ps1
.\crates-package-gate.ps1
.\public-release-gate.ps1
```

Review `cargo package --list` output and make sure no build directories, local secrets, large generated files, or private data are included.

Create the GitHub repository and push the exact source that will be published before publishing crates. This makes the `repository` metadata immediately valid.

## crates.io authentication

Create an API token on crates.io and authenticate locally with `cargo login`. Treat the token as a secret. Do not commit Cargo credentials or paste the token into issue logs.

## Publish order

Publish each package only after the previous dependency level has appeared in the crates.io index:

```powershell
cargo publish -p hybit-core

cargo publish -p hybit-matrix
cargo publish -p hybit-krylov

cargo publish -p hybit-precond

cargo publish -p hybit-auto

cargo publish -p hybit
```

Use `cargo publish --dry-run -p <package>` immediately before each real publish. Higher-level dry-runs can only resolve after their internal dependencies are visible on crates.io.

Published crate versions are immutable. If a published package contains a serious problem, publish a new version; yanking is available for preventing new dependency resolution, but it is not a replacement for careful preflight checks.

## Git tag and release

After the source is committed and the crate set is published, tag the exact commit:

```powershell
git tag -a v0.6.0 -m "HyBIT 0.6.0"
git push origin v0.6.0
```

Use `RELEASE_NOTES_0.6.0.md` as the starting point for the GitHub Release text.
