# C1-G3-A4-A1 — Real-FEM phase-cost evidence (45 timed arms)

Status: **45/45 numerical/status/decision/cost checks PASS** on WSL
Rust 1.99.1 (user-run acceptance, 2026-11-08). Measurement-only
opt-in instrumentation at `develop/0.10.0` HEAD `f66f76c` with
uncommitted A4-A1 source changes. No controller policy change.

## External elapsed time (seconds, five runs per arm)

| FEM case (n144) | Observe | A2 shadow | A3 watchdog |
|---|---:|---:|---:|
| island_shell | 0.080 | 0.100 | 0.130 |
| multi_patch | 0.070 | 0.140 | 0.200 |
| rotated_anisotropy | 0.090 | 0.090 | 0.090 |

The external `/usr/bin/time` elapsed values are quantized to 0.01 s,
and cannot resolve small differences. They include the full solve,
not only the shadow trial. No statistical speedup claim is made.

## Instrumented phase medians (milliseconds, five runs)

| Case | A2 trial | Watchdog trial | Watchdog inclusive | Candidate continuation | Jacobi replay | CSR verification |
|---|---:|---:|---:|---:|---:|---:|
| island_shell | 10.148 | 10.117 | 53.662 | 32.838 | 20.814 | 0.156 |
| multi_patch | 19.013 | 19.563 | 156.477 | 98.584 | 57.121 | 0.151 |

- `island_shell`: watchdog **RolledBack** to the Jacobi reference
  at printed CSR residual and relative x-error precision; the A2-only
  candidate deteriorates over the remaining horizon.
- `multi_patch`: watchdog **Accepted** the useful candidate,
  preserving A2 final residual and x error.
- `rotated_anisotropy`: **NotTriggered** in A2/watchdog; no phase
  record or added work was observed.
- All 45 runs: **MaxIterations / 600 selected-path iterations**
  at tolerance `1e-10`, not `Converged`. Source instrumentation does
  not count speculative replay toward selected-path iterations.
- Cost is overwhelmingly candidate continuation plus exact Jacobi
  replay; independent CSR verification costs about 0.15 ms in each
  triggered watchdog case. Bounded early-warning and partial replay
  are **not** yet safety-certified optimizations.

## Evidence provenance and scope

- Original WSL run directory (external to repository): `/home/michioga/c1h11_evidence/c1g3_a4_a1_cost_cNF8nx`.
- 45 original `.log` and 45 `.time` files; all returned zero exit
  status and positive RSS. Fixed printed residual/error, status,
  escalation count, decision and full-horizon budget compared against
  A3-A3 `f66f76c` reference, not calibrated from the new measurements.
- `phase_cost_summary.csv`: exactly 45 rows compared field-for-field
  with the raw logs and time files, including 13 nonnegative finite
  phase-cost fields in each triggered A2/watchdog arm.
- All 12 recorded corpus input SHA-256 entries verified, including
  the nine required `n144` files; Rust source SHA-256 and repository
  source-integrity manifest independently verified.
- WSL Rust acceptance reported: `hybit-auto` 56/56 release tests
  PASS; Clippy `-D warnings` for auto and `fem_bench` PASS; release
  `fem_bench` build PASS; formatting and source integrity PASS.
- Shadow/watchdog remain opt-in; this is a bounded three-case
  numerical regression, not proof of a universal convergence bound,
  a memory reduction, or an A4-A2 production policy.

### Accepted post-rustfmt Rust source SHA-256

```text
95d5d3f16de205b0cf9035d79e17ed5c64514bdc01f187518ee41f2da1e5110f  crates/hybit-auto/src/lib.rs
734dcad77b4b0766ebbb39fbf74612a680e2210a14d00d8c6148944f4a4107cb  crates/hybit/examples/fem_bench.rs
```

### Evidence files SHA-256

```text
dfe4684fa5af3b8dcc9decc211958bad2e9994b9fbc617932b9ac1a89b54e43e  git_head.txt
360a324bc846c8b0179d4f89de899fa8ee9003ba6ae694009577b3535ea96dc9  git_status.txt
2c78a10e1c897263ff9de3d9241f84bf67a44ec7e5bab97c721990506ba22ee0  input_sha256.txt
d95f94ffb91e369ca9e2ae3333ef3b379a8bed5f661bcdf5e2a4326724939620  island_shell_a2_1.log
33c6aa1ef854859d3de6dd9be230d8ce5a8b5f4bdfbb8bf768698f36593f79fe  island_shell_a2_1.time
8ebef0adc148ecc6a061870875d3ae98825edacf6dd0cec1fc71de31cf4e2b04  island_shell_a2_2.log
d5af4b9cfaf91f1ee94101d8ada207aa8e90be68f07dfdaee6e87a7e3df514e1  island_shell_a2_2.time
9b851d4a0950f0d378f3292a9d2e110948e517c30d050c7e644d4a2937771f4b  island_shell_a2_3.log
84bbb982db6bf7534523b554ac36c0745be6a6335dae1a503d8d5c67455d3e5b  island_shell_a2_3.time
ee4db48c294458a3a3cba3ab6da1a84f7d101d1b7644da7ac53a1e1c325d7ee3  island_shell_a2_4.log
e75bc2eaccf6e41be201fbe43d5a8c54ce9e360e89878e71e3653d2cd3311c81  island_shell_a2_4.time
fbe0c2e984fffe60132cb79b98ba09016bd16bf232c0e3d32a0aeca05c6cef68  island_shell_a2_5.log
e3e13dfbd9b6f03c35c973f235d6d2c7eb95cf2409132eea53a9fc765cd923e0  island_shell_a2_5.time
23f06202e2fb7fe09242c2e2b6d1f123767298fe249ee0253a99d18563cb7619  island_shell_observe_1.log
bb0884e934302b087582b4b1ddf8b20bec941df655c1f48e5294aa0c82dbd987  island_shell_observe_1.time
34a654e17cfbf2a352f9002eab999d5b6d26f06bd20875cdedbd2d1c4a9941e4  island_shell_observe_2.log
162175da42e197742528967158e8350d00e98fd918398ed8eabe90949a0e06f1  island_shell_observe_2.time
6642c2df6d687598490f54117994dc421843d59e39570d2b94be5f81a8af65c5  island_shell_observe_3.log
e2998db986f6dccb74db9e11d945b2abfe4a81c6ace546f52fb7b2d9c1d13d58  island_shell_observe_3.time
653ef00481a0377cb6edc4bb0fee44037e7a3963b934625d00c95a4b92c15562  island_shell_observe_4.log
f1cbb270152919289d6b13c9265023d18f27b03dde5c9094b28753c0928c04fd  island_shell_observe_4.time
f50847941c504dc084c45d84ddea3e433717a07a07b030e560dc8f453742a70e  island_shell_observe_5.log
40a2373f4c528fc73baf088e53ec32eb478c77038b9b95277e566deee22e2a25  island_shell_observe_5.time
eb51dedf1896d1d99f1e62f84e1f2175ad354155c92863ed28d1fb06076378f0  island_shell_watchdog_1.log
2af09f80831143bae7723c973f791cc7458c539dcb9a6cd678229f9a5897c6ed  island_shell_watchdog_1.time
96b7b1109ca11ab14281edf7ed525d2b47a55f4072eec4e125550f64ceef9b43  island_shell_watchdog_2.log
57b982a14b42780c7387a2ccd0c0ded7efb762e1ec30a57db4bb2bc92243b40f  island_shell_watchdog_2.time
1c11fbd45bb7cb70f62e8a19fa77adb646d7a7ff95ef9e51b67138cbe4f7b5d9  island_shell_watchdog_3.log
f013dbc4a6129e1d8f1949463a3922a77d07c1f9fd940c439b36b488ea62e87e  island_shell_watchdog_3.time
388afc848167c8df6e12f80eaa543036ad837fd134a61656b4393f29ab60eaad  island_shell_watchdog_4.log
8f1bc4eca6e1bf8c10aad1b68648325ffd0b58c39fdeb4592665d15b062468bf  island_shell_watchdog_4.time
906643085122a3168ff67dfd55063c2ace0234f3e4ca6ce6252693213823530d  island_shell_watchdog_5.log
2e28a46ee4cf47eb54593397306881969aa209b41aed4e8d74c734a8fd8f3fb4  island_shell_watchdog_5.time
7af432e1e5853a2ddd373a2ffe1795909ce2d6f027c22e478bf9c80d18bd97ca  multi_patch_a2_1.log
dc43b100e1c2baca0375ac0fe1108eaeb926c1c5e6034e080eb3f0211da5e27b  multi_patch_a2_1.time
87d27099ac6e07b1597f60e7a5b2016f13a9e7ea3933d331c9756510957e0bb5  multi_patch_a2_2.log
8bfbd1dab061e48cccf6ffe731c10660bba2ba49aab28bd3642e0ad2b78cf05b  multi_patch_a2_2.time
5f9aab86b95da1c79e0653afa05844361968dc29a6fb02e15feb282121428879  multi_patch_a2_3.log
81cca65cce4c1852a29ece20d83b51e2e76884e94e08b9c6c98e6ea4e5f49edd  multi_patch_a2_3.time
b11fa8ba4732729fc3a5169c7973a8b4c199e163ea7e83cab905e928301f6956  multi_patch_a2_4.log
cfd6e867c6f66dc8ea99d2d85ebccf0ab0bdbbb4816750700313de94b4d1b7da  multi_patch_a2_4.time
b143d5aae0c295cd3a186b4630a68dee3df7294498e9078fa4bfe3a230c5914b  multi_patch_a2_5.log
8bfbd1dab061e48cccf6ffe731c10660bba2ba49aab28bd3642e0ad2b78cf05b  multi_patch_a2_5.time
8d14825a49423b37a0717711b4655026465753cc6e7691adc5b6a32861d66767  multi_patch_observe_1.log
901dc324647b72189874ff76f6f4e7b93d627e083d37e94bbe4425915f413224  multi_patch_observe_1.time
ece79197bdf6541296a3a0020cd2c7aab264570da5698cc3b24d35345998bf14  multi_patch_observe_2.log
312ae06b7a612d7b373119c945cccce4d60506efda4fe3389a5a1578bc05e93b  multi_patch_observe_2.time
feb389f325e32f1a6378ba3b6b91d837c01031500d3ebfe0c2836ce2163acc05  multi_patch_observe_3.log
e1ea8e3e83d2cb37bf1232b3386833e108d0a45cfd35b55e1060235502b81c5c  multi_patch_observe_3.time
c78c9e5f91bb40406f116505dbacae8ac7e83d096f563419e0095ad8c3cb71b2  multi_patch_observe_4.log
334bc9473d6492620600d2cb10494aba7ef349ce5bc0e8b4003b2bdb22d8e893  multi_patch_observe_4.time
98ca95fe104bbf9372b549525c64c2943a67bd43d8cab612f382d4ae075ef7a5  multi_patch_observe_5.log
45ceb5274e6fecd85608a234e1f2f11862b68b057c0f11669bbd88313210c5bc  multi_patch_observe_5.time
2d9594f8baaafd7d77d3da84bca8ef75869d2bcaed943bc9181ca5cb2d5f95d2  multi_patch_watchdog_1.log
83d397dc6435dc972263c6cd535e20332ab36a7585fbe69dc170bb5802e4d8c8  multi_patch_watchdog_1.time
8d9eaf8176c9044efe1c3c7acbd045b9ebe51a2d16b38af4eb0f5d2f92a701a2  multi_patch_watchdog_2.log
6b6af0d91c69525cd60303770c4c751d580433d32c1d76e2ff83ffd10ebdb073  multi_patch_watchdog_2.time
21c4902d410f18feda465a4182825f28bf2d4b24d55dae7f28fcdb20f819d719  multi_patch_watchdog_3.log
2a55d97db7fac02b447e133d0ea5a281cbe3ae9a8934b148134314b3f3614365  multi_patch_watchdog_3.time
12935a1fdd6d2ee6a4afdfac36c42e05ad4ef32183a6b51bd049be3862bcc3d7  multi_patch_watchdog_4.log
b9d8f0c36af8653ddd7a8f2648669bc4e8df4021771591255f5b0a546a76437c  multi_patch_watchdog_4.time
b4a01d15c28923fc3e2431696d153c084a0983318cba437f6dbcf42abc146909  multi_patch_watchdog_5.log
d9d24dc49a0f459173fcbe36e89ac29475825cd57401cbb86c930d327537885a  multi_patch_watchdog_5.time
2110906eededf87b333de18ab216cfbc0cc10f0a92cdbf831e6dbc3c81eb45f9  phase_cost_summary.csv
37d7fadd3047694146d74f93a0c06b53db870ff25ecc67c5194c3deb39f1f5cf  rotated_anisotropy_a2_1.log
d019cf906a6356badc2b138239b1d6a80b2fd1ee2823666fe6292180f681ee7f  rotated_anisotropy_a2_1.time
fbe7b2721489dde442431de43674e2b7540dbbda34594cb6d181030951966476  rotated_anisotropy_a2_2.log
da67fcddc62d9f35b3ecad8ae7978ed6e9314471d5a295bbccc13429b459a542  rotated_anisotropy_a2_2.time
d3a6e7b0a7f5246195283130a0e1fa7261c38bef3433a86a73edca6fae77abd2  rotated_anisotropy_a2_3.log
4c204f99b6103b7c7df39feed51134d077ef3527489372ebb5162ba9fb689f66  rotated_anisotropy_a2_3.time
8900f05cbd14bfdeb157959303e1dcdbe0f0b5030165a8a1942eb4a5d658cbc8  rotated_anisotropy_a2_4.log
df1a8ab565565449b9564d5f6bfc4f1e3eab0f891665a2c2b25bf393cd49f92d  rotated_anisotropy_a2_4.time
f22bd7201e98b2dadc64b368a70ffbd4cb1dc8afc8927b467926617ce9d46323  rotated_anisotropy_a2_5.log
4576fc9fc217d90bdeb2060b6efe5cb7150a6cab72d6c28d795246485bada5c3  rotated_anisotropy_a2_5.time
c4a17f28f89a620cd7b15d02ccccd816a62e052ff18974ff7492ef3b38d95e16  rotated_anisotropy_observe_1.log
500d0004d8b32d8d3a7a6830305d13be4a7c86df5485612e8d9bcb609a4e0cc4  rotated_anisotropy_observe_1.time
97c36901ed27d4ed85502d7be82685f4dbebeb875786b8550659231f76ca681b  rotated_anisotropy_observe_2.log
512ad4d24987337dd854889e99e10aaa67ef25f35f738dc2ded642c7a4c4181e  rotated_anisotropy_observe_2.time
2e145a4ba8cec9ce80fb35676027e9635528875204b983ec809a5419b05c27e4  rotated_anisotropy_observe_3.log
56ffca69c9222118ea2bbb3eb16e0ca5b1aea6c163bb22ea7822592f86084b67  rotated_anisotropy_observe_3.time
fb4ba76dd95869255e373020a910dfe539ee38a1b18c4c6fe8f109daec510d90  rotated_anisotropy_observe_4.log
184ddef5e63bceae14369452a02584a3e02dbccb2443e619a8c3a1431b985ace  rotated_anisotropy_observe_4.time
d3643f4462c9339636edb44f8db59b4a1a70621ed8be16ea44ef6c1c3ade61c5  rotated_anisotropy_observe_5.log
debe6b50729b371831d9080dc54fca88ea4a4302d69ea548fd065c062dedeb6f  rotated_anisotropy_observe_5.time
b4d08a7ac829cb4e93d4d5d4e2105c973ce09b605047092d0b9e902dc517a4bf  rotated_anisotropy_watchdog_1.log
3a4e6fd1085b6d055855279b7aa5a634e6e01fd751d0d9a287b54f952bfaa93c  rotated_anisotropy_watchdog_1.time
ce9a9ff1420ed1cedee2bab2e81a4ea8df35f665f42610b659fe4d8d8cbc10d5  rotated_anisotropy_watchdog_2.log
be8a7cd3337aa821157e539565d3ac6c6d8d211ec89aa23c2994356a57e123a8  rotated_anisotropy_watchdog_2.time
5af19989be85b19d1c71b306641afc4d1701fb73f20fff96f0125046297a68fc  rotated_anisotropy_watchdog_3.log
491dcd510f72853fbe184b8a34735111fcc913236d03a5c469b3f37e9e629be1  rotated_anisotropy_watchdog_3.time
b7ee3b4d43010f0c48a91d48f144e71175d24ad9ce7c0f6b1c7cc5796615df1a  rotated_anisotropy_watchdog_4.log
e6f289199761db0ea0fee55963e6d2036d9c80842b44babc7fd69861385dee0b  rotated_anisotropy_watchdog_4.time
3bf9f55c759ad59a42c1d5816ef33947ad6c2941d4442ff4233bb5590381a40a  rotated_anisotropy_watchdog_5.log
0c8a87db25772b9da0d60054c0e0285fa1ae7340fcd58ad924ea7d97186b005f  rotated_anisotropy_watchdog_5.time
36a341edf459f1912474c557c7f11cc036ebb245e3b9e0aef09eeb1c71e8457a  source_diff_stat.txt
950c1ae3156ec6a54a8d7eddbd673176d1879dad74dc82b06a1fa32e72576aba  source_sha256.txt
```
