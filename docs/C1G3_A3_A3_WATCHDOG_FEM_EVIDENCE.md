# C1-G3-A3-A3 — Verified real-FEM watchdog and rollback evidence

Status: **9/9 timed FEM arms passed and 2/2 watchdog decisions verified**.
C1-G3-A3-A3 Rust gate: `hybit-auto` 56/56 tests, release Clippy and
`fem_bench` release build passed on WSL Rust 1.99.1 (user run).
Baseline: `develop/0.10.0` at `bde0583` with uncommitted A3-A3 opt-in patch.

## Strict comparison (n144 / 20,736 DOFs, max 600 PCG iterations)

| FEM case | Arm | Decision | Status | Iterations | True relative residual | Relative x error | Total wall ms | Peak RSS KiB |
|---|---|---|---|---:|---:|---:|---:|---:|
| island_shell | observe | Baseline | MaxIterations | 600 | 7.749079e-09 | 1.721567e-07 | 65.758 | 8140 |
| island_shell | a2 | Accepted | MaxIterations | 600 | 1.131198e-06 | 5.219228e-01 | 95.866 | 10904 |
| island_shell | watchdog | RolledBack | MaxIterations | 600 | 7.749079e-09 | 1.721567e-07 | 113.495 | 11960 |
| multi_patch | observe | Baseline | MaxIterations | 600 | 2.528508e-08 | 4.011900e-05 | 64.590 | 8172 |
| multi_patch | a2 | Accepted | MaxIterations | 600 | 8.848107e-09 | 1.508957e-06 | 129.787 | 10584 |
| multi_patch | watchdog | Accepted | MaxIterations | 600 | 8.848107e-09 | 1.508957e-06 | 191.417 | 11576 |
| rotated_anisotropy | observe | Baseline | MaxIterations | 600 | 1.984328e-07 | 3.686524e-06 | 74.822 | 8224 |
| rotated_anisotropy | a2 | NotTriggered | MaxIterations | 600 | 1.984328e-07 | 3.686524e-06 | 75.814 | 7868 |
| rotated_anisotropy | watchdog | NotTriggered | MaxIterations | 600 | 1.984328e-07 | 3.686524e-06 | 77.602 | 8368 |

## Controller outcomes and iteration accounting

- `island_shell`: initial A2 trial at iteration 396 accepted 32-step
  candidate (`1.390873134479e-9`). A3-A3 checked 172 further candidate
  steps, independently replayed 204 original Jacobi steps, detected
  `WatchdogCandidateDeteriorated` (1.131197636162e-6 vs 7.749078519813e-9)
  and rolled back. Selected-path iteration count remains 600;
  escalation count returns to zero. Final CSR residual and x error
  reproduce the observe/Jacobi arm at printed precision.
- `multi_patch`: initial trial at iteration 12 accepted 64-step
  candidate. A3-A3 ran 524 further candidate steps, replayed 588
  Jacobi steps, and retained the stronger candidate (8.848107398002e-9 vs 2.528508380918e-8).
- `rotated_anisotropy`: no trigger, no speculative factor/replay.
- Peak duplicate PCG state: 1,990,656 bytes in triggered A3-A3 arms
  (`2 * 995,328`), excluding 602,624 bytes of candidate factors,
  ABTM, and transient allocations. The selected-path iteration count
  excludes speculative baseline replay; both trial and watchdog elapsed
  time include real speculative computation.

## Interpretation and limitations

The full-horizon watchdog recovers the exact Jacobi reference behavior
at the originally observed printed residual and x-error precision for
`island_shell`, while preserving the useful `multi_patch` candidate.
This validates experimental **selection safety for these three FEM
cases**, not a universal convergence guarantee or an optimized policy.
All arms end with `MaxIterations` rather than `Converged` at `1e-10`.
Full-horizon replay adds material CPU time and RSS. All runtime
numbers are single observations, not statistical speedup claims.
The watchdog is single-window; only a window spanning remaining
iterations protects the final horizon. Default shadow and watchdog
routing remain disabled. No production policy promotion is justified.

## Reproduction and provenance

- Original evidence directory (outside repository):
  `/home/michioga/c1h11_evidence/c1g3_a3_a3_watchdog_umbDmk`.
- Replay: `bash /mnt/c/Users/michi/Downloads/C1G3_A3_A3_replay.sh`.
- Common options: `--tol 1e-10 --max-iters 600 --backend csr
  --probe-iters 12 --monitor-iters 64 --monitor-rate-limit 0.99
  --max-region 128 --max-regions 8 --overlap 1
  --factor-budget-mib 64 --max-escalations 1 --skip-plain`.
- Shadow: `--shadow-trial-iters 32` (island/rotated) or `64`
  (multi), `--shadow-accept-ratio 0.8 --shadow-max-mib 16`.
- Guarded arm additionally: `--shadow-watchdog-iters 600`.
- Environment: `RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1`,
  `/usr/bin/time`, 120-second timeout per arm.
- Run directory records HEAD, tracked source diff, Git status,
  source and input SHA-256 manifests. Input and source hashes were
  independently rechecked during evidence acceptance.
  The input manifest covers every `c1g_p1_*` corpus member matched
  by the replay script, potentially more than the nine required n144
  files; all listed digests are checked, not just the selected triplets.

### WSL source SHA-256 (post-rustfmt)

```text
91c5262112c852e58a1ce70f34c9009d72eb53e7dc5d92bff8658c46f2d52fff  crates/hybit-auto/src/lib.rs
9fd71a8c5b5e4765ac466b9cfe2921789b8582dbc933d736068ee0c816c6ef8b  crates/hybit/examples/fem_bench.rs
```

### Evidence files SHA-256

```text
bdf91a2cc13d8a38ac3ad66cf16c6f333ef4f0a57ce47f03ba7c87ba00b9bf56  git_head.txt
67fe9484d63c6f53c70b9d0757331ee14b8754424cc6601b55439b98b1c08faf  git_status.txt
2c78a10e1c897263ff9de3d9241f84bf67a44ec7e5bab97c721990506ba22ee0  input_sha256.txt
7cec86f1747f0bdcb4b7f9bfac069692783a1884aca23cc53b592a2b7fe64755  island_shell_a2.log
ba7400c12ac1fed0b5269addae40589374f2deea0ee0f3076100acdb123360a3  island_shell_a2.time
71a0b8013b28bfa178e703600b0d97ac96bd37687c1915c12da0c877bc66c8de  island_shell_observe.log
c5cbc0e573fb6b8fa25d1725d0558920d6d0173aad2078358a43c0c3a8bdd2b5  island_shell_observe.time
e7723f232e5e164d251dc02f63d74308afe36220993d00521cbf74590614acfa  island_shell_watchdog.log
3cbe85f72fd629232b28e1a383b4efd77ff7a3c57b937ba808647ecf7f29a34d  island_shell_watchdog.time
3337055a655a87de4555ae5b95d2064f4805f5f9957f116519449445add9cd2f  multi_patch_a2.log
25630a0f2d4e149fde29809a9b8e0067baaf088b71c13b73a031eb4b005cdedc  multi_patch_a2.time
d515475329dcbf3576757c5d6954b64e15e613d26675c892aa72797bb90d87f8  multi_patch_observe.log
5470b4a39e9d92255e0e0ec72d3d5ea0e2279bc489144c42c5bafccbf052a002  multi_patch_observe.time
d656ece6d73aac0932a9474bfe04e84dae83b35720e7cb7a97ec14d2f6ef5edd  multi_patch_watchdog.log
3a73f0b06ca46c88934bd8e5ad316bf6c2a60a710a4b09a98c5d5f745042f37a  multi_patch_watchdog.time
a29c72620980e291c7cba8e8722f3ece4d0a1d2e06c07f80d5a6f2db7bb36c88  rotated_anisotropy_a2.log
97f720255c2c02d693a995aa9a3e7de5ec6df9eb18f2053c7a7e10cbac770e0e  rotated_anisotropy_a2.time
e0e61a0547d9cbd5bdcc0064c505cad93f378c8f4b7da45052f2a67abd20ae6b  rotated_anisotropy_observe.log
500d0004d8b32d8d3a7a6830305d13be4a7c86df5485612e8d9bcb609a4e0cc4  rotated_anisotropy_observe.time
477852c513c2a59df09474ed42f9cc0cfb3a40a412cc2eccffd5235f1bb66351  rotated_anisotropy_watchdog.log
a92613adbf5c038a45c487da7c9bcd2fd16a1282219ae39bcb9d9cfeab72988e  rotated_anisotropy_watchdog.time
b283ff87d453c7cccd4879438ab30ed18ee7bbf65ad895470c11f1f0bb23953d  source_diff_stat.txt
8dd4c647a1af458ba8406f73630d334fc646f98e7c477381260d99ee5a417e4f  source_sha256.txt
```
