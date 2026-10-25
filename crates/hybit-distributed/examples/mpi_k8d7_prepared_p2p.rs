//! H1-C3K8-D3 isolated MPI rankwise Halo and ABBA experiment. Original K8-C/D1/D2 preserved.
//! SPD checks: strict diagonal dominance, anchored weakly diagonal dominant M-matrix,
//! or bounded-size numerical Cholesky screening (not an unrestricted SPD proof).
//! Collective MPI errors are synchronized where possible. Experimental only.

use hybit_distributed::mpi_assignment::{
    AssignmentHaloPlan, AssignmentLocalCsr, AssignmentOwnedCsrRows,
    prepare_assignment_owned_rows_sparse, spmv_assignment, spmv_assignment_d2_profile,
    spmv_assignment_d7_p2p_profile, validate_assignment_d2_layout,
};
use hybit_distributed::mpi_assignment_sparse::scatter_sparse_owners_from_root;
use hybit_distributed::mpi_assignment_sparse_verify::verify_sparse_assignment_scatter_exact;
use hybit_distributed::mpi_backend::MpiRuntime;
use hybit_distributed::mpi_block_jacobi::LocalBlockCholesky;
use hybit_distributed::mpi_ic0::LocalBlockIc0;
use hybit_distributed::{
    AbtmMultilevelOptions, ContiguousPartition, PartitionAssignment,
    abtm_balanced_multisource_partition, abtm_multilevel_partition, abtm_region_grow_partition,
};
use hybit_matrix::{Csr32Matrix, read_matrix_market};
use mpi::collective::SystemOperation;
use mpi::traits::*;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

type BenchResult<T> = Result<T, Box<dyn Error>>;
const MAX_INPUT_BYTES: u64 = 384 * 1024 * 1024;
const MAX_DOFS: usize = 1_200_000;
const MAX_NNZ: usize = 16_000_000;

// H1-C3K8-D9B: compare native ABTM and externally prepared WSL labels on the
// SAME Prepared Halo / MS-MPI PCG path. The labels are read by rank zero only.
#[derive(Clone, Copy, Debug)]
enum Policy {
    Equal,
    Region,
    Multisource,
    Multilevel,
    External,
}
impl Policy {
    fn parse(s: &str) -> BenchResult<Self> {
        match s {
            "equal" => Ok(Self::Equal),
            "region" => Ok(Self::Region),
            "multisource" => Ok(Self::Multisource),
            "multilevel" => Ok(Self::Multilevel),
            "external" => Ok(Self::External),
            _ => Err("policy must be equal, region, multisource, multilevel, or external".into()),
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Equal => "equal",
            Self::Region => "abtm_region_growth",
            Self::Multisource => "abtm_multisource",
            Self::Multilevel => "abtm_multilevel",
            Self::External => "external_owner_labels",
        }
    }
}

fn parse_external_owner_labels(
    text: &str,
    expected_dofs: usize,
    ranks: u32,
) -> BenchResult<PartitionAssignment> {
    let owners: Vec<u32> = text
        .split_whitespace()
        .map(str::parse::<u32>)
        .collect::<Result<_, _>>()?;
    if owners.len() != expected_dofs {
        return Err(format!(
            "external owner labels contain {} entries, expected {}",
            owners.len(),
            expected_dofs
        )
        .into());
    }
    // Enforces all ranks nonempty and each owner in 0..ranks.
    Ok(PartitionAssignment::from_owners(ranks, owners)?)
}

fn partition(matrix: &Csr32Matrix, ranks: u32, policy: Policy) -> BenchResult<PartitionAssignment> {
    Ok(match policy {
        Policy::Equal => PartitionAssignment::from_contiguous(&ContiguousPartition::balanced(
            matrix.nrows() as u64,
            ranks,
        )?)?,
        Policy::Region => abtm_region_grow_partition(matrix, ranks)?.0,
        Policy::Multisource => abtm_balanced_multisource_partition(matrix, ranks)?.0,
        Policy::Multilevel => {
            abtm_multilevel_partition(matrix, ranks, AbtmMultilevelOptions::default())?.0
        }
        Policy::External => {
            let path = std::env::var_os("HYBIT_D9B_OWNER_LABELS")
                .ok_or("HYBIT_D9B_OWNER_LABELS is required for external partition")?;
            let max_bytes = (matrix.nrows() as u64)
                .checked_mul(16)
                .ok_or("external owner label size bound overflow")?;
            if std::fs::metadata(&path)?.len() > max_bytes {
                return Err("external owner labels file exceeds length bound".into());
            }
            let labels = std::fs::read_to_string(&path)?;
            parse_external_owner_labels(&labels, matrix.nrows(), ranks)?
        }
    })
}

fn world_max(value: f64) -> f64 {
    let world = mpi::topology::SimpleCommunicator::world();
    let mut out = 0.0f64;
    world.all_reduce_into(&value, &mut out, SystemOperation::max());
    out
}

fn x_exact(global: u64) -> f64 {
    let t = global as f64 + 1.0;
    (t * 0.017).sin() + 0.125 * (t * 0.0031).cos()
}

// D9-C8B: genuinely different, deterministic manufactured solutions on the
// ORIGINAL input matrix. RHS 0 is bit-for-bit the legacy D7 experiment.
fn x_exact_rhs(global: u64, rhs_id: usize) -> f64 {
    if rhs_id == 0 {
        return x_exact(global);
    }
    let t = global as f64 + 1.0;
    let phase = rhs_id as f64 * 0.137;
    (t * 0.017 + phase).sin() + 0.125 * (t * (0.0031 + rhs_id as f64 * 0.000017)).cos()
}

/// A sufficient (not necessary) SPD certificate: A=A^T exactly, finite,
/// positive diagonal and strict row diagonal dominance. It intentionally
/// rejects many valid SPD FEM matrices; never confuses PCG convergence with
/// certification of arbitrary input.
fn certify_strict_spd(a: &Csr32Matrix) -> BenchResult<()> {
    if a.nrows() == 0 || a.nrows() != a.ncols() {
        return Err("SPD certificate requires nonempty square matrix".into());
    }
    let n = a.nrows();
    for i in 0..n {
        let start = a.row_ptr()[i] as usize;
        let end = a.row_ptr()[i + 1] as usize;
        let cols = &a.col_idx()[start..end];
        let vals = &a.values()[start..end];
        if cols.is_empty() || cols.windows(2).any(|w| w[0] >= w[1]) {
            return Err("SPD certificate requires sorted unique CSR columns".into());
        }
        let mut diag = None;
        let mut offsum = 0.0f64;
        for (&j, &v) in cols.iter().zip(vals) {
            if !v.is_finite() || j as usize >= n {
                return Err("SPD certificate: invalid CSR coefficient".into());
            }
            if j as usize == i {
                diag = Some(v);
            } else {
                offsum += v.abs();
                let j0 = a.row_ptr()[j as usize] as usize;
                let j1 = a.row_ptr()[j as usize + 1] as usize;
                let reverse = a.col_idx()[j0..j1]
                    .binary_search(&(i as u32))
                    .map_err(|_| "SPD certificate: missing transpose coefficient")?;
                if a.values()[j0 + reverse].to_bits() != v.to_bits() {
                    return Err("SPD certificate: asymmetric coefficients".into());
                }
            }
        }
        let d = diag.ok_or("SPD certificate: diagonal missing")?;
        if !offsum.is_finite() || d <= 0.0 || d <= offsum + 1.0e-12 * (d + offsum) {
            return Err(
                "SPD certificate: not strictly diagonally dominant with positive diagonal".into(),
            );
        }
    }
    Ok(())
}

/// Return the condition under which the matrix was admitted to this prototype.
/// A Cholesky screening is a finite-precision test, not an analytic SPD proof.
fn classify_spd(a: &Csr32Matrix) -> BenchResult<&'static str> {
    if certify_strict_spd(a).is_ok() {
        return Ok("strict_diagonal_dominance");
    }
    let n = a.nrows();
    if n == 0 || n != a.ncols() {
        return Err("SPD requires a nonempty square matrix".into());
    }
    let mut anchored = vec![false; n];
    let mut weak_mmatrix = true;
    for (i, anchored_i) in anchored.iter_mut().enumerate() {
        let begin = a.row_ptr()[i] as usize;
        let end = a.row_ptr()[i + 1] as usize;
        let mut diag = None;
        let mut offsum = 0.0f64;
        if a.col_idx()[begin..end].windows(2).any(|x| x[0] >= x[1]) {
            return Err("SPD CSR columns must be sorted and unique".into());
        }
        for k in begin..end {
            let j = a.col_idx()[k] as usize;
            let v = a.values()[k];
            if j >= n || !v.is_finite() {
                return Err("invalid SPD coefficient".into());
            }
            if j == i {
                diag = Some(v);
            } else {
                let b = a.row_ptr()[j] as usize;
                let e = a.row_ptr()[j + 1] as usize;
                let reverse = a.col_idx()[b..e]
                    .binary_search(&(i as u32))
                    .map_err(|_| "missing symmetric transpose entry")?;
                if v.to_bits() != a.values()[b + reverse].to_bits() {
                    return Err("asymmetric matrix".into());
                }
                if v > 0.0 {
                    weak_mmatrix = false;
                }
                offsum += v.abs();
            }
        }
        let d = diag.ok_or("SPD diagonal missing")?;
        if d <= 0.0 || !offsum.is_finite() {
            return Err("SPD requires positive finite diagonal".into());
        }
        if d < offsum {
            weak_mmatrix = false;
        }
        if d > offsum + 1.0e-12 * (d + offsum).max(1.0) {
            *anchored_i = true;
        }
    }
    if weak_mmatrix {
        // Each connected component must contain a strictly dominant anchor.
        // Graph is symmetric, so a BFS from all anchors covers exactly those components.
        let mut reachable = anchored.clone();
        let mut stack = (0..n).filter(|&i| anchored[i]).collect::<Vec<_>>();
        while let Some(i) = stack.pop() {
            for k in a.row_ptr()[i] as usize..a.row_ptr()[i + 1] as usize {
                let j = a.col_idx()[k] as usize;
                if i != j && a.values()[k] < 0.0 && !reachable[j] {
                    reachable[j] = true;
                    stack.push(j);
                }
            }
        }
        if reachable.iter().all(|&v| v) {
            return Ok("anchored_weak_diagonal_dominance");
        }
    }
    // A bounded numerical Cholesky check admits FEM matrices with positive
    // cross-component couplings that are not M-matrices or diagonally dominant.
    // Never allocate a global dense matrix on nonroot ranks.
    const MAX_DENSE_CERT_DOFS: usize = 512;
    if n > MAX_DENSE_CERT_DOFS {
        return Err(
            "not certified SPD: non-M-matrix or non-DD matrix exceeds bounded Cholesky screen"
                .into(),
        );
    }
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for k in a.row_ptr()[i] as usize..a.row_ptr()[i + 1] as usize {
            l[i * n + a.col_idx()[k] as usize] = a.values()[k];
        }
    }
    let scale = (0..n).map(|i| l[i * n + i]).fold(0.0f64, f64::max);
    let floor = 1.0e-12 * scale.max(1.0);
    for i in 0..n {
        for j in 0..=i {
            let mut v = l[i * n + j];
            for k in 0..j {
                v -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !v.is_finite() || v <= floor {
                    return Err("Cholesky SPD screening: nonpositive/small pivot".into());
                }
                l[i * n + i] = v.sqrt();
            } else {
                l[i * n + j] = v / l[j * n + j];
            }
        }
    }
    Ok("bounded_numerical_cholesky")
}

/// Explicit experimental gate for non-diagonally-dominant large FEM matrices.
/// This is NOT an SPD certificate: the few deterministic Rayleigh probes may
/// miss a negative eigendirection. PCG curvature and true residual are also
/// monitored. No automatic fallback or production activation occurs.
fn screen_large_spd_candidate(a: &Csr32Matrix) -> BenchResult<()> {
    let n = a.nrows();
    if n <= 512 || n != a.ncols() {
        return Err("large SPD screen requires square n > 512".into());
    }
    for i in 0..n {
        let begin = a.row_ptr()[i] as usize;
        let end = a.row_ptr()[i + 1] as usize;
        let cols = &a.col_idx()[begin..end];
        let values = &a.values()[begin..end];
        if cols.is_empty() || cols.windows(2).any(|v| v[0] >= v[1]) {
            return Err("experimental SPD screen requires sorted unique CSR".into());
        }
        let mut diagonal = None;
        for (&col, &value) in cols.iter().zip(values) {
            let j = col as usize;
            if j >= n || !value.is_finite() {
                return Err("invalid experimental SPD coefficient".into());
            }
            if j == i {
                diagonal = Some(value);
            } else {
                let trans_begin = a.row_ptr()[j] as usize;
                let trans_end = a.row_ptr()[j + 1] as usize;
                let trans_idx = a.col_idx()[trans_begin..trans_end]
                    .binary_search(&(i as u32))
                    .map_err(|_| "experimental SPD missing transpose")?;
                if a.values()[trans_begin + trans_idx].to_bits() != value.to_bits() {
                    return Err("experimental SPD asymmetric coefficient".into());
                }
            }
        }
        if !diagonal.is_some_and(|v| v > 0.0 && v.is_finite()) {
            return Err("experimental SPD needs positive finite diagonal".into());
        }
    }
    // Four cheap necessary, never sufficient, positive-curvature probes.
    // Fail closed if a checked direction is not positive.
    let mut x = vec![0.0f64; n];
    let mut ax = vec![0.0f64; n];
    for probe in 0..4 {
        for (i, value) in x.iter_mut().enumerate() {
            let t = i as f64 + 1.0;
            *value = match probe {
                0 => {
                    if i % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                1 => 1.0,
                2 => (t * 0.137).sin() + 0.25 * (t * 0.071).cos(),
                _ => {
                    if (i / 2) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
            };
        }
        for (i, out) in ax.iter_mut().enumerate() {
            let first = a.row_ptr()[i] as usize;
            let last = a.row_ptr()[i + 1] as usize;
            *out = (first..last)
                .map(|p| a.values()[p] * x[a.col_idx()[p] as usize])
                .sum::<f64>();
        }
        let energy = x.iter().zip(&ax).map(|(&u, &v)| u * v).sum::<f64>();
        if !energy.is_finite() || energy <= 0.0 {
            return Err("experimental SPD Rayleigh probe has nonpositive curvature".into());
        }
    }
    Ok(())
}

fn classify_for_run(a: &Csr32Matrix, experimental_large_spd: bool) -> BenchResult<&'static str> {
    if let Ok(certificate) = classify_spd(a) {
        return Ok(certificate);
    }
    if !experimental_large_spd || a.nrows() <= 512 {
        return classify_spd(a);
    }
    screen_large_spd_candidate(a)?;
    Ok("experimental_symmetric_positive_diagonal_curvature_screen")
}

enum FixedPreconditioner {
    Jacobi(Vec<f64>),
    Ic0(LocalBlockIc0),
    Block(LocalBlockCholesky),
}
impl FixedPreconditioner {
    fn build(
        mode: &str,
        block_size: usize,
        rows: &AssignmentOwnedCsrRows,
        local: &AssignmentLocalCsr,
    ) -> BenchResult<Self> {
        match mode {
            "jacobi" => Ok(Self::Jacobi(inverse_diagonal(rows)?)),
            "ic0" => {
                let compatible = local.preconditioner_rank_local_csr()?;
                Ok(Self::Ic0(LocalBlockIc0::prepare(&compatible, block_size)?))
            }
            "block" => {
                let compatible = local.preconditioner_rank_local_csr()?;
                Ok(Self::Block(LocalBlockCholesky::prepare(
                    &compatible,
                    block_size,
                )?))
            }
            _ => Err("preconditioner must be jacobi, ic0, or block".into()),
        }
    }
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        match self {
            Self::Jacobi(diag) => {
                for ((v, &d), &ri) in z.iter_mut().zip(diag).zip(r) {
                    *v = d * ri;
                }
            }
            Self::Ic0(value) => value.apply(r, z),
            Self::Block(value) => value.apply(r, z),
        }
    }
    fn bytes(&self) -> usize {
        match self {
            Self::Jacobi(diag) => diag.len() * std::mem::size_of::<f64>(),
            Self::Ic0(value) => value.factor_bytes(),
            Self::Block(value) => value.factor_bytes(),
        }
    }
}

fn inverse_diagonal(rows: &AssignmentOwnedCsrRows) -> BenchResult<Vec<f64>> {
    if rows.row_ptr.len() != rows.owned_globals.len() + 1 {
        return Err("bad local row_ptr".into());
    }
    let mut inv = Vec::with_capacity(rows.owned_globals.len());
    for (i, &global) in rows.owned_globals.iter().enumerate() {
        let mut diagonal = None;
        for k in rows.row_ptr[i] as usize..rows.row_ptr[i + 1] as usize {
            if rows.global_col_idx[k] == global && diagonal.replace(rows.values[k]).is_some() {
                return Err("duplicate diagonal".into());
            }
        }
        let value = diagonal.ok_or("Jacobi diagonal missing")?;
        if !value.is_finite() || value <= 0.0 {
            return Err("Jacobi diagonal nonpositive".into());
        }
        inv.push(1.0 / value);
    }
    Ok(inv)
}

// K8-D2 rank-local phase timers; full A/B of SpMV validation. Per-phase *max across ranks* is not additive.
#[derive(Clone, Copy, Debug, Default)]
struct PhaseNs {
    spmv_halo: u128,
    spmv_numeric: u128,
    spmv_other: u128,
    spmv_validation: u128,
    spmv_pack: u128,
    spmv_exchange: u128,
    spmv_unpack: u128,
    skip_collective_validation: bool,
    use_p2p: bool,
    dot_local: u128,
    dot_collective: u128,
    precond: u128,
    vector_update: u128,
}
impl PhaseNs {
    fn accounted(self) -> u128 {
        self.spmv_halo
            + self.spmv_numeric
            + self.spmv_other
            + self.spmv_validation
            + self.dot_local
            + self.dot_collective
            + self.precond
            + self.vector_update
    }
    fn globalize(self, local_elapsed_ns: u128) -> PhaseMs {
        // Each metric is reduced separately; peaks can belong to distinct ranks.
        let ms = |n: u128| world_max(n as f64 * 1.0e-6);
        PhaseMs {
            spmv_halo_ms: ms(self.spmv_halo),
            spmv_numeric_ms: ms(self.spmv_numeric),
            spmv_other_ms: ms(self.spmv_other),
            spmv_validation_ms: ms(self.spmv_validation),
            spmv_pack_ms: ms(self.spmv_pack),
            spmv_exchange_ms: ms(self.spmv_exchange),
            spmv_unpack_ms: ms(self.spmv_unpack),
            dot_local_ms: ms(self.dot_local),
            dot_collective_ms: ms(self.dot_collective),
            precond_ms: ms(self.precond),
            vector_update_ms: ms(self.vector_update),
            unattributed_ms: ms(local_elapsed_ns.saturating_sub(self.accounted())),
        }
    }
}
#[derive(Clone, Copy, Debug, Default)]
struct PhaseMs {
    spmv_halo_ms: f64,
    spmv_numeric_ms: f64,
    spmv_other_ms: f64,
    spmv_validation_ms: f64,
    spmv_pack_ms: f64,
    spmv_exchange_ms: f64,
    spmv_unpack_ms: f64,
    dot_local_ms: f64,
    dot_collective_ms: f64,
    precond_ms: f64,
    vector_update_ms: f64,
    unattributed_ms: f64,
}

fn world_dot_profile(a: &[f64], b: &[f64], phases: &mut PhaseNs) -> f64 {
    let started = Instant::now();
    let local = a.iter().zip(b).map(|(&x, &y)| x * y).sum::<f64>();
    phases.dot_local += started.elapsed().as_nanos();
    let started = Instant::now();
    let mut result = 0.0f64;
    mpi::topology::SimpleCommunicator::world().all_reduce_into(
        &local,
        &mut result,
        SystemOperation::sum(),
    );
    phases.dot_collective += started.elapsed().as_nanos();
    result
}

fn spmv_profile(
    mpi: &MpiRuntime,
    halo: &AssignmentHaloPlan,
    op: &AssignmentLocalCsr,
    x: &[f64],
    scratch: &mut hybit_distributed::mpi_assignment::AssignmentSpmvWorkspace,
    y: &mut [f64],
    phases: &mut PhaseNs,
) -> BenchResult<()> {
    let started = Instant::now();
    let timing = if phases.use_p2p {
        spmv_assignment_d7_p2p_profile(halo, op, x, scratch, y)?
    } else {
        spmv_assignment_d2_profile(
            mpi,
            halo,
            op,
            x,
            scratch,
            y,
            phases.skip_collective_validation,
        )?
    };
    let elapsed_ns = started.elapsed().as_nanos();
    let halo_ns = timing.pack_ns + timing.exchange_ns + timing.unpack_ns;
    phases.spmv_halo += halo_ns;
    phases.spmv_numeric += timing.numeric_ns;
    phases.spmv_validation += timing.validation_ns;
    phases.spmv_pack += timing.pack_ns;
    phases.spmv_exchange += timing.exchange_ns;
    phases.spmv_unpack += timing.unpack_ns;
    phases.spmv_other +=
        elapsed_ns.saturating_sub(timing.validation_ns + halo_ns + timing.numeric_ns);
    Ok(())
}

/// Post-solve rankwise totals from a single solve trial. Reporting collectives
/// run after the timed PCG region, so they do not alter measured solve time.
#[derive(Clone, Debug, Default)]
struct RankTelemetry {
    exchange_ms: Vec<f64>,
    dot_collective_ms: Vec<f64>,
    // D9-C8A: rankwise local phase totals, gathered after the timed solve.
    spmv_numeric_ms: Vec<f64>,
    spmv_pack_ms: Vec<f64>,
    spmv_unpack_ms: Vec<f64>,
    spmv_validation_ms: Vec<f64>,
    spmv_other_ms: Vec<f64>,
    precond_ms: Vec<f64>,
    dot_local_ms: Vec<f64>,
    vector_update_ms: Vec<f64>,
    local_elapsed_ms: Vec<f64>,
    unattributed_ms: Vec<f64>,
    sent_values: Vec<u64>,
    received_values: Vec<u64>,
    sent_peers: Vec<u64>,
    received_peers: Vec<u64>,
}
impl RankTelemetry {
    fn slowest_exchange_rank(&self) -> usize {
        self.exchange_ms
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(rank, _)| rank)
    }
    fn exchange_skew_ms(&self) -> f64 {
        let low = self
            .exchange_ms
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let high = self.exchange_ms.iter().copied().fold(0.0_f64, f64::max);
        if low.is_finite() { high - low } else { 0.0 }
    }
    fn numeric_skew_ms(&self) -> f64 {
        let low = self
            .spmv_numeric_ms
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let high = self.spmv_numeric_ms.iter().copied().fold(0.0_f64, f64::max);
        if low.is_finite() { high - low } else { 0.0 }
    }
    fn slowest_numeric_rank(&self) -> usize {
        self.spmv_numeric_ms
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(rank, _)| rank)
    }
    fn f64_vector_csv(v: &[f64]) -> String {
        v.iter()
            .map(|n| format!("{n:.6}"))
            .collect::<Vec<_>>()
            .join("|")
    }
    fn u64_vector_csv(v: &[u64]) -> String {
        v.iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join("|")
    }
}

struct Solution {
    x: Vec<f64>,
    iterations: usize,
    true_rel: f64,
    max_error: f64,
    solve_ms: f64,
    spmv_calls: usize,
    dot_reductions: usize,
    true_residual_checkpoints: usize,
    phases: PhaseMs,
    rank_telemetry: RankTelemetry,
}

#[derive(Clone, Copy, Debug)]
struct PcgSolveOptions {
    max_iters: usize,
    tol: f64,
    experimental_large_spd: bool,
    skip_collective_validation: bool,
    use_p2p: bool,
}

fn solve_pcg(
    mpi: &MpiRuntime,
    halo: &AssignmentHaloPlan,
    op: &AssignmentLocalCsr,
    precond: &FixedPreconditioner,
    rhs: &[f64],
    rhs_id: usize,
    options: PcgSolveOptions,
) -> BenchResult<Solution> {
    let n = halo.owned_len();
    if mpi.all_reduce_sum_u64(u64::from(n == 0 || rhs.len() != n)) != 0 {
        return Err("incompatible PCG local vectors".into());
    }
    let mut x = vec![0.0; n];
    let mut r = rhs.to_vec();
    let mut z = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut ap = vec![0.0; n];
    let mut scratch = halo.workspace();
    // Collective prevalidation is deliberately outside the timed region.
    // Plan/operator/workspace layouts remain unchanged throughout this solve.
    if options.skip_collective_validation {
        validate_assignment_d2_layout(mpi, halo, op, &p, &scratch, &ap)?;
    }
    let mut dots = 0usize;
    let mut spmvs = 0usize;
    let mut checkpoints = 0usize;
    let mut phases = PhaseNs {
        skip_collective_validation: options.skip_collective_validation,
        use_p2p: options.use_p2p,
        ..PhaseNs::default()
    };
    mpi.barrier();
    let started = Instant::now();
    let norm_b2 = world_dot_profile(rhs, rhs, &mut phases);
    dots += 1;
    if !norm_b2.is_finite() || norm_b2 <= 0.0 {
        return Err("invalid manufactured RHS".into());
    }
    let norm_b = norm_b2.sqrt();
    {
        let started = Instant::now();
        precond.apply(&r, &mut z);
        phases.precond += started.elapsed().as_nanos();
    }
    p.copy_from_slice(&z);
    let mut rz = world_dot_profile(&r, &z, &mut phases);
    dots += 1;
    if !rz.is_finite() || rz <= 0.0 {
        return Err("invalid initial preconditioned norm".into());
    }
    let mut iters = 0usize;
    let mut converged = false;
    for it in 0..options.max_iters {
        spmv_profile(mpi, halo, op, &p, &mut scratch, &mut ap, &mut phases)?;
        spmvs += 1;
        let pap = world_dot_profile(&p, &ap, &mut phases);
        dots += 1;
        if !pap.is_finite() || pap <= 0.0 {
            return Err("PCG nonpositive curvature".into());
        }
        let alpha = rz / pap;
        if !alpha.is_finite() {
            return Err("PCG nonfinite alpha".into());
        }
        let vector_started = Instant::now();
        for (((xi, ri), &pi), &api) in x.iter_mut().zip(&mut r).zip(&p).zip(&ap) {
            *xi += alpha * pi;
            *ri -= alpha * api;
        }
        phases.vector_update += vector_started.elapsed().as_nanos();
        iters = it + 1;
        let rr = world_dot_profile(&r, &r, &mut phases);
        dots += 1;
        if !rr.is_finite() || rr < 0.0 {
            return Err("invalid PCG residual".into());
        }
        if rr.sqrt() <= options.tol * norm_b * 0.5 {
            converged = true;
            break;
        }
        if options.experimental_large_spd && (iters == 1 || iters.is_multiple_of(32)) {
            // Recompute b-Ax independently of recursive PCG residual.
            spmv_profile(mpi, halo, op, &x, &mut scratch, &mut ap, &mut phases)?;
            spmvs += 1;
            let true_residual = rhs
                .iter()
                .zip(&ap)
                .map(|(&b, &ax)| b - ax)
                .collect::<Vec<_>>();
            let true_sq = world_dot_profile(&true_residual, &true_residual, &mut phases);
            dots += 1;
            checkpoints += 1;
            if !true_sq.is_finite() || true_sq < 0.0 || true_sq > norm_b2 * 1.0e24 {
                return Err("experimental PCG true-residual checkpoint failed".into());
            }
        }
        {
            let started = Instant::now();
            precond.apply(&r, &mut z);
            phases.precond += started.elapsed().as_nanos();
        }
        let next = world_dot_profile(&r, &z, &mut phases);
        dots += 1;
        if !next.is_finite() || next <= 0.0 {
            return Err("invalid PCG recurrence".into());
        }
        let beta = next / rz;
        if !beta.is_finite() {
            return Err("invalid PCG beta".into());
        }
        let vector_started = Instant::now();
        for (pi, &zi) in p.iter_mut().zip(&z) {
            *pi = zi + beta * *pi;
        }
        phases.vector_update += vector_started.elapsed().as_nanos();
        rz = next;
    }
    if !converged {
        return Err("Jacobi-PCG failed to converge within max_iters".into());
    }
    spmv_profile(mpi, halo, op, &x, &mut scratch, &mut ap, &mut phases)?;
    spmvs += 1;
    let residual = rhs
        .iter()
        .zip(&ap)
        .map(|(&b, &ax)| b - ax)
        .collect::<Vec<_>>();
    let rel = (world_dot_profile(&residual, &residual, &mut phases) / norm_b2).sqrt();
    dots += 1;
    let local_err = x
        .iter()
        .zip(&halo.owned_globals)
        .map(|(&v, &g)| (v - x_exact_rhs(g, rhs_id)).abs())
        .fold(0.0f64, f64::max);
    let max_err = world_max(local_err);
    if !rel.is_finite() || rel > options.tol || !max_err.is_finite() || max_err > 1.0e-5 {
        return Err("PCG true residual or manufactured solution certification failed".into());
    }
    // Stop before diagnostic reductions so timing excludes instrumentation reporting.
    let local_elapsed_ns = started.elapsed().as_nanos();
    let solve_ms = world_max(local_elapsed_ns as f64 * 1.0e-6);
    let phase_report = phases.globalize(local_elapsed_ns);
    // Allgather is deliberately after the local timer was stopped AND after
    // D2's phase max reductions. This adds diagnostic collectives only to D3.
    let world = mpi::topology::SimpleCommunicator::world();
    let nranks = usize::try_from(mpi.size())?;
    let mut exchange_ms = vec![0.0_f64; nranks];
    let exchange_local = phases.spmv_exchange as f64 * 1e-6;
    world.all_gather_into(&exchange_local, &mut exchange_ms[..]);
    let mut dot_collective_ms = vec![0.0_f64; nranks];
    let dot_local = phases.dot_collective as f64 * 1e-6;
    world.all_gather_into(&dot_local, &mut dot_collective_ms[..]);
    // D9-C8A: one post-timer collective records ten local phases. These
    // values correspond to the SAME measured solve trial, not maxima from
    // potentially different ranks. No additional MPI call is on the PCG clock.
    let local_phase = [
        phases.spmv_numeric as f64 * 1e-6,
        phases.spmv_pack as f64 * 1e-6,
        phases.spmv_unpack as f64 * 1e-6,
        phases.spmv_validation as f64 * 1e-6,
        phases.spmv_other as f64 * 1e-6,
        phases.precond as f64 * 1e-6,
        phases.dot_local as f64 * 1e-6,
        phases.vector_update as f64 * 1e-6,
        local_elapsed_ns as f64 * 1e-6,
        local_elapsed_ns.saturating_sub(phases.accounted()) as f64 * 1e-6,
    ];
    let mut all_phases = vec![0.0_f64; nranks * local_phase.len()];
    world.all_gather_into(&local_phase[..], &mut all_phases[..]);
    let halo_counts = halo.d3_halo_counts();
    let local_counts = [
        halo_counts.sent_values,
        halo_counts.received_values,
        halo_counts.sent_peers,
        halo_counts.received_peers,
    ];
    let mut all_counts = vec![0_u64; nranks * 4];
    world.all_gather_into(&local_counts[..], &mut all_counts[..]);
    let mut rank_telemetry = RankTelemetry {
        exchange_ms,
        dot_collective_ms,
        ..RankTelemetry::default()
    };
    for values in all_phases.chunks_exact(local_phase.len()) {
        rank_telemetry.spmv_numeric_ms.push(values[0]);
        rank_telemetry.spmv_pack_ms.push(values[1]);
        rank_telemetry.spmv_unpack_ms.push(values[2]);
        rank_telemetry.spmv_validation_ms.push(values[3]);
        rank_telemetry.spmv_other_ms.push(values[4]);
        rank_telemetry.precond_ms.push(values[5]);
        rank_telemetry.dot_local_ms.push(values[6]);
        rank_telemetry.vector_update_ms.push(values[7]);
        rank_telemetry.local_elapsed_ms.push(values[8]);
        rank_telemetry.unattributed_ms.push(values[9]);
    }
    for values in all_counts.as_chunks::<4>().0 {
        rank_telemetry.sent_values.push(values[0]);
        rank_telemetry.received_values.push(values[1]);
        rank_telemetry.sent_peers.push(values[2]);
        rank_telemetry.received_peers.push(values[3]);
    }
    Ok(Solution {
        x,
        iterations: iters,
        true_rel: rel,
        max_error: max_err,
        solve_ms,
        spmv_calls: spmvs,
        dot_reductions: dots,
        true_residual_checkpoints: checkpoints,
        phases: phase_report,
        rank_telemetry,
    })
}

/// Collect each noncontiguous owned solution to root and evaluate the
/// ORIGINAL (pre-scatter) matrix. Unlike distributed residual checks, this
/// reference uses neither scattered values nor Halo numeric output.
fn verify_root_original(
    mpi: &MpiRuntime,
    root: Option<&Csr32Matrix>,
    owners: Option<&PartitionAssignment>,
    rows: &AssignmentOwnedCsrRows,
    x: &[f64],
    rhs_id: usize,
    tol: f64,
) -> BenchResult<(f64, f64)> {
    let world = mpi::topology::SimpleCommunicator::world();
    let count = rows.owned_globals.len() as u64;
    let mut counts = vec![0u64; mpi.size() as usize];
    world.all_gather_into(&count, &mut counts[..]);
    if mpi.all_reduce_sum_u64(u64::from(
        x.len() != count as usize || x.iter().any(|v| !v.is_finite()),
    )) != 0
    {
        return Err("invalid local solution before root reference gather".into());
    }
    let total = counts.iter().sum::<u64>();
    let expected_n = root.map_or(0, |m| m.nrows() as u64);
    let n = mpi.all_reduce_sum_u64(expected_n);
    if total != n {
        return Err("gathered solution global size mismatch".into());
    }
    let root_result: BenchResult<(f64, f64)> = if mpi.rank() == 0 {
        // Do not abort root before finishing the expected receives.
        let mut packets = vec![(Vec::<u64>::new(), Vec::<f64>::new()); mpi.size() as usize];
        packets[0] = (rows.owned_globals.clone(), x.to_vec());
        for rank in 1..mpi.size() {
            let len = counts[rank as usize] as usize;
            let mut ids = vec![0u64; len];
            let mut vals = vec![0f64; len];
            world.process_at_rank(rank).receive_into(&mut ids[..]);
            world.process_at_rank(rank).receive_into(&mut vals[..]);
            packets[rank as usize] = (ids, vals);
        }
        (|| -> BenchResult<(f64, f64)> {
            let matrix = root.ok_or("root matrix missing")?;
            let selection = owners.ok_or("root assignment missing")?;
            let mut full = vec![0.0f64; matrix.nrows()];
            let mut seen = vec![false; matrix.nrows()];
            for (rank, (ids, values)) in packets.iter().enumerate() {
                if ids.len() != values.len() {
                    return Err("gather row/value counts disagree".into());
                }
                for (&id, &v) in ids.iter().zip(values) {
                    let idx = usize::try_from(id)?;
                    if idx >= seen.len()
                        || seen[idx]
                        || selection.owner_of(id)? != rank as u32
                        || !v.is_finite()
                    {
                        return Err("gather invalid global row ownership".into());
                    }
                    seen[idx] = true;
                    full[idx] = v;
                }
            }
            if seen.iter().any(|&flag| !flag) {
                return Err("root missed a global solution row".into());
            }
            let mut numer = 0.0f64;
            let mut denom = 0.0f64;
            let mut reference_error = 0.0f64;
            for i in 0..matrix.nrows() {
                let mut ax = 0.0;
                let mut b = 0.0;
                for k in matrix.row_ptr()[i] as usize..matrix.row_ptr()[i + 1] as usize {
                    let j = matrix.col_idx()[k] as usize;
                    let aij = matrix.values()[k];
                    ax += aij * full[j];
                    b += aij * x_exact_rhs(j as u64, rhs_id);
                }
                numer += (b - ax) * (b - ax);
                denom += b * b;
                reference_error =
                    reference_error.max((full[i] - x_exact_rhs(i as u64, rhs_id)).abs());
            }
            if !denom.is_finite() || denom <= 0.0 {
                return Err("root manufactured RHS invalid".into());
            }
            let reference_rel = (numer / denom).sqrt();
            Ok((reference_rel, reference_error))
        })()
    } else {
        world.process_at_rank(0).send(&rows.owned_globals[..]);
        world.process_at_rank(0).send(x);
        Ok((0.0, 0.0))
    };
    if mpi.all_reduce_sum_u64(u64::from(root_result.is_err())) != 0 {
        if let Err(ref err) = root_result {
            eprintln!("K8-B root original residual verification error: {err}");
        }
        return Err("K8-B root original reference check failed on a rank".into());
    }
    let (mut reference_rel, mut reference_error) = root_result?;
    reference_rel = world_max(reference_rel);
    reference_error = world_max(reference_error);
    if reference_rel > tol
        || !reference_rel.is_finite()
        || !reference_error.is_finite()
        || reference_error > 1.0e-5
    {
        return Err("root-original CSR final residual/solution error failed".into());
    }
    Ok((reference_rel, reference_error))
}

fn parse_d7_transport(mode: &str) -> BenchResult<bool> {
    match mode {
        "alltoallv" => Ok(false),
        "p2p" => Ok(true),
        _ => Err("D7 transport must be alltoallv or p2p".into()),
    }
}

fn main() -> BenchResult<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 12 && args.len() != 14 {
        return Err("usage: mpi_k8d7_prepared_p2p <ranks> <input.mtx> <equal|region|multisource|multilevel|external> <warmups> <repeats> <max_iters> <tol> <output.csv> <jacobi|ic0|block> <block-size> --experimental-large-spd <alltoallv|p2p> [--solve-many 1|4|8|16]".into());
    }
    let solve_many = if args.len() == 14 {
        if args[12] != "--solve-many" {
            return Err("D9-C8B requires --solve-many flag".into());
        }
        let count: usize = args[13].parse()?;
        if !matches!(count, 1 | 4 | 8 | 16) {
            return Err("D9-C8B RHS count must be 1, 4, 8 or 16".into());
        }
        Some(count)
    } else {
        None
    };
    let experimental_large_spd = true;
    if args[10] != "--experimental-large-spd" {
        return Err("invalid opt-in flag for experimental large SPD path".into());
    }
    let mode = args[11].as_str();
    let use_p2p = parse_d7_transport(mode)?;
    // Both A/B paths use the same prevalidated layout, isolating transport cost.
    let skip_collective_validation = true;
    let ranks: i32 = args[0].parse()?;
    let path = Path::new(&args[1]);
    let policy = Policy::parse(&args[2])?;
    let warmups: usize = args[3].parse()?;
    let repeats: usize = args[4].parse()?;
    let max_iters: usize = args[5].parse()?;
    let tol: f64 = args[6].parse()?;
    let output = Path::new(&args[7]);
    let precond_mode = args[8].as_str();
    let block_size: usize = args[9].parse()?;
    if !matches!(ranks, 1 | 2 | 4 | 8)
        || !matches!(precond_mode, "jacobi" | "ic0" | "block")
        || !(2..=512).contains(&block_size)
        || warmups > 3
        || !(1..=12).contains(&repeats)
        || !(1..=3000).contains(&max_iters)
        || !(1.0e-12..=1.0e-3).contains(&tol)
    {
        return Err("K8-B guarded parameter bounds violated".into());
    }
    let mpi = MpiRuntime::initialize()?;
    if mpi.size() != ranks {
        return Err("unexpected MPI ranks".into());
    }
    if mpi.all_reduce_sum_u64(u64::from(mpi.rank() == 0 && output.exists())) != 0 {
        return Err("refuse to overwrite output CSV".into());
    }
    mpi.barrier();
    // Legacy 12-argument D7 path has no extra setup timer or collective.
    let prepared_setup_started = solve_many.map(|_| Instant::now());
    let stamp = Instant::now();
    let loaded = if mpi.rank() == 0 {
        Some((|| -> BenchResult<Csr32Matrix> {
            if std::fs::metadata(path)?.len() > MAX_INPUT_BYTES {
                return Err("input too large".into());
            }
            let (m, _) = read_matrix_market(path)?;
            if m.nrows() == 0
                || m.nrows() != m.ncols()
                || m.nrows() > MAX_DOFS
                || m.nrows() < ranks as usize
                || m.nnz() > MAX_NNZ
            {
                return Err("invalid root-only input matrix dimensions".into());
            }
            classify_for_run(&m, experimental_large_spd)?;
            Ok(m)
        })())
    } else {
        None
    };
    if mpi.all_reduce_sum_u64(u64::from(loaded.as_ref().is_some_and(Result::is_err))) != 0 {
        if let Some(Err(err)) = &loaded {
            eprintln!("K8-B SPD input rejected: {err}");
        }
        return Err("root SPD input validation failed".into());
    }
    let matrix = loaded.transpose()?;
    let certificate = if let Some(ref m) = matrix {
        classify_for_run(m, experimental_large_spd)?
    } else {
        "nonroot"
    };
    let read_ms = world_max(stamp.elapsed().as_secs_f64() * 1000.0);
    let n = mpi.all_reduce_sum_u64(matrix.as_ref().map_or(0, |m| m.nrows() as u64));
    let nnz = mpi.all_reduce_sum_u64(matrix.as_ref().map_or(0, |m| m.nnz() as u64));
    mpi.barrier();
    let started = Instant::now();
    let proposed = matrix.as_ref().map(|m| partition(m, ranks as u32, policy));
    if mpi.all_reduce_sum_u64(u64::from(proposed.as_ref().is_some_and(Result::is_err))) != 0 {
        return Err("root partition proposal failed".into());
    }
    let selection = proposed.transpose()?;
    let partition_ms = world_max(started.elapsed().as_secs_f64() * 1000.0);
    mpi.barrier();
    let (directory, rows, stats) =
        scatter_sparse_owners_from_root(&mpi, matrix.as_ref(), selection.as_ref())?;
    let scatter_ms = world_max(stats.distribution.scatter_ns as f64 * 1e-6);
    let verification = verify_sparse_assignment_scatter_exact(
        &mpi,
        matrix.as_ref(),
        selection.as_ref(),
        &directory,
        &rows,
    )?;
    if verification.verified_rows != n
        || verification.verified_nnz != nnz
        || !verification.bitwise_coefficients
    {
        return Err("root original distributed CSR exact check failed".into());
    }
    let verify_ms = world_max(verification.verify_ns as f64 * 1e-6);
    mpi.barrier();
    let started = Instant::now();
    let (halo, local, hstats) = prepare_assignment_owned_rows_sparse(&mpi, &directory, &rows)?;
    let halo_ms = world_max(started.elapsed().as_secs_f64() * 1000.0);
    mpi.barrier();
    let started_precond = Instant::now();
    let prepared = FixedPreconditioner::build(precond_mode, block_size, &rows, &local);
    if mpi.all_reduce_sum_u64(u64::from(prepared.is_err())) != 0 {
        if let Err(ref error) = prepared {
            eprintln!("K8-B preconditioner preparation error: {error}");
        }
        return Err("collective preconditioner preparation failed".into());
    }
    let precond = prepared?;
    let precond_ms = world_max(started_precond.elapsed().as_secs_f64() * 1000.0);
    let factor_bytes = mpi.all_reduce_sum_u64(precond.bytes() as u64);
    let ghosts = mpi.all_reduce_sum_u64(hstats.ghosts as u64);
    let sparse_owner_bytes = mpi.all_reduce_sum_u64(stats.owner_bytes as u64);
    // D9-C8A: compute-load proxies from the exact rank-owned original CSR.
    // This collective is performed once during setup, outside solve timing.
    let local_work = [rows.owned_globals.len() as u64, rows.values.len() as u64];
    let mut rank_work = vec![0_u64; ranks as usize * local_work.len()];
    mpi::topology::SimpleCommunicator::world().all_gather_into(&local_work[..], &mut rank_work[..]);
    let (rank_pairs, remainder) = rank_work.as_chunks::<2>();
    debug_assert!(remainder.is_empty());
    let rank_owned_dofs = rank_pairs.iter().map(|w| w[0]).collect::<Vec<_>>();
    let rank_owned_nnz = rank_pairs.iter().map(|w| w[1]).collect::<Vec<_>>();
    if rank_owned_dofs.iter().sum::<u64>() != n || rank_owned_nnz.iter().sum::<u64>() != nnz {
        return Err("D9-C8A rank-owned work sum does not equal original CSR".into());
    }
    if let Some(rhs_count) = solve_many {
        let prepared_setup_total_ms = world_max(
            prepared_setup_started
                .ok_or("D9-C8B missing setup timer")?
                .elapsed()
                .as_secs_f64()
                * 1000.0,
        );
        // D9-C8B: opt-in amortization experiment on the exact ORIGINAL CSR.
        // Partition, sparse scatter, Halo, and preconditioner above are built
        // ONCE and remain identical across independent manufactured RHS.
        mpi.barrier();
        let rhs_started = Instant::now();
        let mut rhs_vectors = Vec::with_capacity(rhs_count);
        let mut p2p_workspace = halo.workspace();
        for rhs_id in 0..rhs_count {
            let known = halo
                .owned_globals
                .iter()
                .map(|&g| x_exact_rhs(g, rhs_id))
                .collect::<Vec<_>>();
            let mut rhs = vec![0.0; known.len()];
            spmv_assignment(&mpi, &halo, &local, &known, &mut halo.workspace(), &mut rhs)?;
            // Both transports must construct bitwise-identical RHS for EVERY
            // independently generated exact solution, before timed batches.
            let mut p2p_rhs = vec![0.0; known.len()];
            validate_assignment_d2_layout(&mpi, &halo, &local, &known, &p2p_workspace, &p2p_rhs)?;
            spmv_assignment_d7_p2p_profile(
                &halo,
                &local,
                &known,
                &mut p2p_workspace,
                &mut p2p_rhs,
            )?;
            let mismatch = rhs
                .iter()
                .zip(&p2p_rhs)
                .any(|(a, b)| a.to_bits() != b.to_bits());
            if mpi.all_reduce_sum_u64(u64::from(mismatch)) != 0 {
                return Err("D9-C8B manufactured RHS differs between transports".into());
            }
            rhs_vectors.push(rhs);
        }
        let rhs_gen_verify_ms = world_max(rhs_started.elapsed().as_secs_f64() * 1000.0);
        let options = PcgSolveOptions {
            max_iters,
            tol,
            experimental_large_spd,
            skip_collective_validation,
            use_p2p,
        };
        let mut iterations = vec![None::<usize>; rhs_count];
        let mut measured_wall_ms = Vec::with_capacity(repeats);
        let mut measured_solver_ms = Vec::with_capacity(repeats);
        let mut last_solutions = Vec::new();
        let mut max_distributed_rel = 0.0_f64;
        let mut max_distributed_err = 0.0_f64;
        let mut batch_spmv_calls = 0_usize;
        let mut batch_dot_reductions = 0_usize;
        for trial in 0..warmups + repeats {
            mpi.barrier();
            let batch_started = Instant::now();
            let mut sum_solver_only_ms = 0.0_f64;
            let mut solutions = Vec::with_capacity(rhs_count);
            let mut trial_spmvs = 0_usize;
            let mut trial_dots = 0_usize;
            for (rhs_id, rhs) in rhs_vectors.iter().enumerate() {
                let result = solve_pcg(&mpi, &halo, &local, &precond, rhs, rhs_id, options)?;
                if iterations[rhs_id].is_some_and(|v| v != result.iterations) {
                    return Err("D9-C8B iteration count changed between batches".into());
                }
                iterations[rhs_id] = Some(result.iterations);
                sum_solver_only_ms += result.solve_ms;
                trial_spmvs += result.spmv_calls;
                trial_dots += result.dot_reductions;
                if trial >= warmups {
                    max_distributed_rel = max_distributed_rel.max(result.true_rel);
                    max_distributed_err = max_distributed_err.max(result.max_error);
                }
                solutions.push(result.x);
            }
            let batch_profiled_wall_ms = world_max(batch_started.elapsed().as_secs_f64() * 1000.0);
            if trial >= warmups {
                measured_wall_ms.push(batch_profiled_wall_ms);
                measured_solver_ms.push(sum_solver_only_ms);
                batch_spmv_calls = trial_spmvs;
                batch_dot_reductions = trial_dots;
            }
            if trial + 1 == warmups + repeats {
                last_solutions = solutions;
            }
        }
        mpi.barrier();
        let root_started = Instant::now();
        let mut root_true_rel_max = 0.0_f64;
        let mut root_solution_error_max = 0.0_f64;
        for (rhs_id, solution) in last_solutions.iter().enumerate() {
            let (true_rel, max_error) = verify_root_original(
                &mpi,
                matrix.as_ref(),
                selection.as_ref(),
                &rows,
                solution,
                rhs_id,
                tol,
            )?;
            root_true_rel_max = root_true_rel_max.max(true_rel);
            root_solution_error_max = root_solution_error_max.max(max_error);
        }
        let root_verify_ms = world_max(root_started.elapsed().as_secs_f64() * 1000.0);
        let batch_wall_median = median(&measured_wall_ms);
        let solver_only_sum_median = median(&measured_solver_ms);
        // A controlled estimate for one prepared batch, not the measured
        // full program elapsed time with benchmark warmups/repetitions.
        let estimated_prepared_total_ms = prepared_setup_total_ms + batch_wall_median;
        if mpi.rank() == 0 {
            println!(
                "D9-C8B ORIGINAL CSR SOLVE-MANY PASS ranks={} method={} transport={} rhs={} median_batch_wall_ms={:.6} median_sum_solver_only_ms={:.6} setup_ms={:.6} root_true_rel_max={:.3e} original_coefficients=true",
                ranks,
                policy.label(),
                mode,
                rhs_count,
                batch_wall_median,
                solver_only_sum_median,
                prepared_setup_total_ms,
                root_true_rel_max,
            );
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(output)?;
            writeln!(
                file,
                "matrix,ranks,method,d7_transport,n,nnz,rhs_count,warmups,repeats,max_iters,tol,preconditioner,block_size,spd_screen,iterations_per_rhs,median_batch_profiled_wall_ms,median_sum_solver_only_ms,prepared_setup_total_ms,estimated_prepared_time_to_solution_ms,estimated_prepared_per_rhs_ms,read_ms,partition_ms,scatter_ms,verify_csr_ms,halo_setup_ms,precond_setup_ms,rhs_gen_verify_ms,root_verify_ms,root_original_true_rel_max,root_solution_error_max,distributed_true_rel_max,distributed_solution_error_max,ghosts,factor_bytes,sparse_owner_bytes,rank_owned_dofs,rank_owned_nnz,spmv_calls_per_batch,dot_reductions_per_batch,exact_original_verified,d7_halo_bitwise,reference_scope"
            )?;
            let iteration_counts = iterations
                .iter()
                .map(|value| value.unwrap_or(0).to_string())
                .collect::<Vec<_>>()
                .join("|");
            let fields = [
                path.display().to_string(),
                ranks.to_string(),
                policy.label().to_string(),
                mode.to_string(),
                n.to_string(),
                nnz.to_string(),
                rhs_count.to_string(),
                warmups.to_string(),
                repeats.to_string(),
                max_iters.to_string(),
                format!("{tol:.3e}"),
                precond_mode.to_string(),
                block_size.to_string(),
                certificate.to_string(),
                iteration_counts,
                format!("{batch_wall_median:.6}"),
                format!("{solver_only_sum_median:.6}"),
                format!("{prepared_setup_total_ms:.6}"),
                format!("{estimated_prepared_total_ms:.6}"),
                format!("{:.6}", estimated_prepared_total_ms / rhs_count as f64),
                format!("{read_ms:.6}"),
                format!("{partition_ms:.6}"),
                format!("{scatter_ms:.6}"),
                format!("{verify_ms:.6}"),
                format!("{halo_ms:.6}"),
                format!("{precond_ms:.6}"),
                format!("{rhs_gen_verify_ms:.6}"),
                format!("{root_verify_ms:.6}"),
                format!("{root_true_rel_max:.12e}"),
                format!("{root_solution_error_max:.12e}"),
                format!("{max_distributed_rel:.12e}"),
                format!("{max_distributed_err:.12e}"),
                ghosts.to_string(),
                factor_bytes.to_string(),
                sparse_owner_bytes.to_string(),
                RankTelemetry::u64_vector_csv(&rank_owned_dofs),
                RankTelemetry::u64_vector_csv(&rank_owned_nnz),
                batch_spmv_calls.to_string(),
                batch_dot_reductions.to_string(),
                "true".to_string(),
                "true".to_string(),
                "original_matrix_coefficients".to_string(),
            ];
            writeln!(file, "{}", fields.join(","))?;
        }
        mpi.barrier();
        return Ok(());
    }
    mpi.barrier();
    let started = Instant::now();
    let known = halo
        .owned_globals
        .iter()
        .map(|&g| x_exact(g))
        .collect::<Vec<_>>();
    let mut rhs = vec![0.0; known.len()];
    spmv_assignment(&mpi, &halo, &local, &known, &mut halo.workspace(), &mut rhs)?;
    let rhs_ms = world_max(started.elapsed().as_secs_f64() * 1000.0);
    // Outside the timed solver: identical prepared Halo, data and numeric CSR.
    // The bitwise comparison is collective so no rank silently accepts bad data.
    let mut p2p_rhs = vec![0.0; known.len()];
    let mut p2p_scratch = halo.workspace();
    validate_assignment_d2_layout(&mpi, &halo, &local, &known, &p2p_scratch, &p2p_rhs)?;
    spmv_assignment_d7_p2p_profile(&halo, &local, &known, &mut p2p_scratch, &mut p2p_rhs)?;
    let different = rhs
        .iter()
        .zip(&p2p_rhs)
        .any(|(a, b)| a.to_bits() != b.to_bits());
    if mpi.all_reduce_sum_u64(u64::from(different)) != 0 {
        return Err("D7 Halo P2P failed bitwise original-CSR SpMV cross-check".into());
    }
    let mut solve_times = Vec::new();
    let mut measured_phases = Vec::<(f64, PhaseMs, RankTelemetry)>::new();
    let mut observed_iters = None;
    let mut last_solution = None;
    let mut worst_true_rel = 0.0f64;
    let mut worst_err = 0.0f64;
    let mut calls = 0usize;
    let mut reductions = 0usize;
    let mut true_checkpoints = 0usize;
    for trial in 0..warmups + repeats {
        let result = solve_pcg(
            &mpi,
            &halo,
            &local,
            &precond,
            &rhs,
            0,
            PcgSolveOptions {
                max_iters,
                tol,
                experimental_large_spd,
                skip_collective_validation,
                use_p2p,
            },
        )?;
        if observed_iters.is_some_and(|k| k != result.iterations) {
            return Err("inconsistent iteration count between repeats".into());
        }
        observed_iters = Some(result.iterations);
        if trial >= warmups {
            solve_times.push(result.solve_ms);
            measured_phases.push((result.solve_ms, result.phases, result.rank_telemetry));
            worst_true_rel = worst_true_rel.max(result.true_rel);
            worst_err = worst_err.max(result.max_error);
            calls = result.spmv_calls;
            reductions = result.dot_reductions;
            true_checkpoints = result.true_residual_checkpoints;
        }
        last_solution = Some(result.x);
    }
    let final_solution = last_solution.ok_or("missing PCG result")?;
    mpi.barrier();
    let root_started = Instant::now();
    let (root_rel, root_err) = verify_root_original(
        &mpi,
        matrix.as_ref(),
        selection.as_ref(),
        &rows,
        &final_solution,
        0,
        tol,
    )?;
    let root_verify_ms = world_max(root_started.elapsed().as_secs_f64() * 1000.0);
    let solve_median = median(&solve_times);
    measured_phases.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (selected_phases, selected_ranks) = measured_phases
        .get(measured_phases.len() / 2)
        .map(|r| (r.1, &r.2))
        .ok_or("missing measured K8-D7 phase sample")?;
    if mpi.rank() == 0 {
        println!(
            "K8-D7 PREPARED P2P PASS mode={} method={} ranks={} n={} nnz={} spd={} precond={} block={} iterations={} median_solve_ms={:.6} root_original_rel={:.3e} precond_setup_ms={:.3} factor_bytes={} exact_original=true",
            mode,
            policy.label(),
            ranks,
            n,
            nnz,
            certificate,
            precond_mode,
            block_size,
            observed_iters.unwrap_or(0),
            solve_median,
            root_rel,
            precond_ms,
            factor_bytes
        );
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output)?;
        writeln!(
            file,
            "matrix,ranks,method,n,nnz,spd_certificate,preconditioner,warmups,repeats,max_iters,tol,iterations,median_solve_ms,read_ms,partition_ms,scatter_ms,verify_csr_ms,halo_setup_ms,rhs_setup_ms,root_residual_verification_ms,distributed_true_rel,root_original_true_rel,max_solution_error,root_solution_error,sparse_owner_bytes_sum,ghosts,spmv_calls,dot_reductions,exact_original_verified,precond_setup_ms,precond_factor_bytes_sum,block_size,true_residual_checkpoints,experimental_opt_in,phase_spmv_halo_ms,phase_spmv_numeric_ms,phase_spmv_other_ms,phase_dot_local_ms,phase_dot_allreduce_ms,phase_precond_apply_ms,phase_vector_update_ms,phase_unattributed_ms,phase_scope,d2_mode,phase_spmv_validation_ms,phase_spmv_pack_ms,phase_spmv_exchange_ms,phase_spmv_unpack_ms,d3_rank_exchange_ms,d3_rank_dot_allreduce_ms,d3_rank_sent_values,d3_rank_received_values,d3_rank_sent_peers,d3_rank_received_peers,d3_exchange_skew_ms,d3_slowest_exchange_rank,d7_transport,d7_halo_bitwise,d8_rank_spmv_numeric_ms,d8_rank_spmv_pack_ms,d8_rank_spmv_unpack_ms,d8_rank_spmv_validation_ms,d8_rank_spmv_other_ms,d8_rank_precond_ms,d8_rank_dot_local_ms,d8_rank_vector_update_ms,d8_rank_local_elapsed_ms,d8_rank_unattributed_ms,d8_rank_owned_dofs,d8_rank_owned_nnz,d8_numeric_skew_ms,d8_slowest_numeric_rank"
        )?;
        write!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{:.3e},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.12e},{:.12e},{:.12e},{:.12e},{},{},{},{},true,{:.6},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},rankwise_max_not_additive,prevalidated,{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{},{:.6},{},{},true",
            path.display(),
            ranks,
            policy.label(),
            n,
            nnz,
            certificate,
            precond_mode,
            warmups,
            repeats,
            max_iters,
            tol,
            observed_iters.unwrap_or(0),
            solve_median,
            read_ms,
            partition_ms,
            scatter_ms,
            verify_ms,
            halo_ms,
            rhs_ms,
            root_verify_ms,
            worst_true_rel,
            root_rel,
            worst_err,
            root_err,
            sparse_owner_bytes,
            ghosts,
            calls,
            reductions,
            precond_ms,
            factor_bytes,
            block_size,
            true_checkpoints,
            experimental_large_spd,
            selected_phases.spmv_halo_ms,
            selected_phases.spmv_numeric_ms,
            selected_phases.spmv_other_ms,
            selected_phases.dot_local_ms,
            selected_phases.dot_collective_ms,
            selected_phases.precond_ms,
            selected_phases.vector_update_ms,
            selected_phases.unattributed_ms,
            selected_phases.spmv_validation_ms,
            selected_phases.spmv_pack_ms,
            selected_phases.spmv_exchange_ms,
            selected_phases.spmv_unpack_ms,
            RankTelemetry::f64_vector_csv(&selected_ranks.exchange_ms),
            RankTelemetry::f64_vector_csv(&selected_ranks.dot_collective_ms),
            RankTelemetry::u64_vector_csv(&selected_ranks.sent_values),
            RankTelemetry::u64_vector_csv(&selected_ranks.received_values),
            RankTelemetry::u64_vector_csv(&selected_ranks.sent_peers),
            RankTelemetry::u64_vector_csv(&selected_ranks.received_peers),
            selected_ranks.exchange_skew_ms(),
            selected_ranks.slowest_exchange_rank(),
            mode
        )?;
        writeln!(
            file,
            ",{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{}",
            RankTelemetry::f64_vector_csv(&selected_ranks.spmv_numeric_ms),
            RankTelemetry::f64_vector_csv(&selected_ranks.spmv_pack_ms),
            RankTelemetry::f64_vector_csv(&selected_ranks.spmv_unpack_ms),
            RankTelemetry::f64_vector_csv(&selected_ranks.spmv_validation_ms),
            RankTelemetry::f64_vector_csv(&selected_ranks.spmv_other_ms),
            RankTelemetry::f64_vector_csv(&selected_ranks.precond_ms),
            RankTelemetry::f64_vector_csv(&selected_ranks.dot_local_ms),
            RankTelemetry::f64_vector_csv(&selected_ranks.vector_update_ms),
            RankTelemetry::f64_vector_csv(&selected_ranks.local_elapsed_ms),
            RankTelemetry::f64_vector_csv(&selected_ranks.unattributed_ms),
            RankTelemetry::u64_vector_csv(&rank_owned_dofs),
            RankTelemetry::u64_vector_csv(&rank_owned_nnz),
            selected_ranks.numeric_skew_ms(),
            selected_ranks.slowest_numeric_rank()
        )?;
    }
    mpi.barrier();
    Ok(())
}

fn median(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(f64::total_cmp);
    let m = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[m - 1] + sorted[m])
    } else {
        sorted[m]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn d9c8b_rhs_zero_preserves_d7_manufactured_solution() {
        for global in [0, 1, 127, 1_000, 82_653] {
            assert_eq!(x_exact_rhs(global, 0).to_bits(), x_exact(global).to_bits());
        }
    }

    #[test]
    fn d9c8b_rhs_vectors_are_deterministic_and_distinct() {
        for rhs_id in [1, 4, 8, 15] {
            let value = x_exact_rhs(1_337, rhs_id);
            assert!(value.is_finite());
            assert_eq!(value.to_bits(), x_exact_rhs(1_337, rhs_id).to_bits());
            assert_ne!(value.to_bits(), x_exact_rhs(1_337, 0).to_bits());
        }
        assert_ne!(x_exact_rhs(17, 1).to_bits(), x_exact_rhs(17, 2).to_bits());
    }

    #[test]
    fn d9b_owner_labels_require_full_valid_rank_coverage() {
        let p = parse_external_owner_labels("0\n1\n0\n1\n", 4, 2).unwrap();
        assert_eq!(p.owners(), &[0, 1, 0, 1]);
        assert!(parse_external_owner_labels("0\n1\n0\n", 4, 2).is_err());
        assert!(parse_external_owner_labels("0\n1\n2\n1\n", 4, 2).is_err());
        assert!(parse_external_owner_labels("0\n0\n0\n0\n", 4, 2).is_err());
        assert!(parse_external_owner_labels("0\n1\n0\nx\n", 4, 2).is_err());
    }

    #[test]
    fn d9b_policy_parsing_is_explicit() {
        assert!(matches!(
            Policy::parse("multilevel").unwrap(),
            Policy::Multilevel
        ));
        assert!(matches!(
            Policy::parse("external").unwrap(),
            Policy::External
        ));
        assert!(Policy::parse("metis").is_err());
        assert!(Policy::parse("scotch").is_err());
    }

    #[test]
    fn d7_transport_modes_are_strict() {
        assert!(!parse_d7_transport("alltoallv").unwrap());
        assert!(parse_d7_transport("p2p").unwrap());
        assert!(parse_d7_transport("checked").is_err());
    }
    #[test]
    fn d3_rank_exchange_skew_and_slowest_rank() {
        let data = RankTelemetry {
            exchange_ms: vec![0.5, 2.5, 1.0],
            ..RankTelemetry::default()
        };
        assert!((data.exchange_skew_ms() - 2.0).abs() < 1.0e-12);
        assert_eq!(data.slowest_exchange_rank(), 1);
        assert_eq!(RankTelemetry::u64_vector_csv(&[2, 4, 1]), "2|4|1");
    }
    #[test]
    fn d8_rank_numeric_skew_and_slowest_rank() {
        let data = RankTelemetry {
            spmv_numeric_ms: vec![0.4, 1.1, 0.6],
            ..RankTelemetry::default()
        };
        assert!((data.numeric_skew_ms() - 0.7).abs() < 1.0e-12);
        assert_eq!(data.slowest_numeric_rank(), 1);
        assert_eq!(
            RankTelemetry::f64_vector_csv(&data.spmv_numeric_ms),
            "0.400000|1.100000|0.600000"
        );
    }
    fn make_matrix(d0: f64, d1: f64, a01: f64, a10: f64) -> Csr32Matrix {
        Csr32Matrix::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![d0, a01, a10, d1],
        )
        .unwrap()
    }
    #[test]
    fn anchored_weak_dominance_accepts_boundary_constrained_laplacian() {
        let m = Csr32Matrix::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0],
        )
        .unwrap();
        assert_eq!(
            classify_spd(&m).unwrap(),
            "anchored_weak_diagonal_dominance"
        );
    }
    #[test]
    fn nondominant_spd_uses_bounded_cholesky() {
        let m = make_matrix(1.0, 5.0, 2.0, 2.0);
        assert_eq!(classify_spd(&m).unwrap(), "bounded_numerical_cholesky");
    }
    #[test]
    fn indefinite_symmetric_rejected() {
        assert!(classify_spd(&make_matrix(1.0, 1.0, 2.0, 2.0)).is_err());
    }
    #[test]
    fn asymmetric_matrix_rejected_by_all_screeners() {
        assert!(classify_spd(&make_matrix(3.0, 4.0, -1.0, -0.9)).is_err());
    }
    #[test]
    fn strict_spd_certificate_accepts_diagonally_dominant_matrix() {
        assert!(certify_strict_spd(&make_matrix(3.0, 4.0, -1.0, -1.0)).is_ok());
    }
    #[test]
    fn strict_spd_certificate_rejects_asymmetry() {
        assert!(certify_strict_spd(&make_matrix(3.0, 4.0, -1.0, -0.9)).is_err());
    }
    #[test]
    fn strict_spd_certificate_rejects_insufficient_dominance() {
        assert!(certify_strict_spd(&make_matrix(1.0, 1.0, -1.0, -1.0)).is_err());
    }

    fn build_large_paired_matrix(off: f64, reverse: f64) -> Csr32Matrix {
        let n = 513usize;
        let mut rows = vec![0u32];
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            if i + 1 < n && i % 2 == 0 {
                cols.extend([i as u32, (i + 1) as u32]);
                vals.extend([1.0, off]);
            } else if i % 2 == 1 {
                cols.extend([(i - 1) as u32, i as u32]);
                vals.extend([reverse, 5.0]);
            } else {
                cols.push(i as u32);
                vals.push(1.0);
            }
            rows.push(vals.len() as u32);
        }
        Csr32Matrix::new(n, n, rows, cols, vals).unwrap()
    }
    #[test]
    fn large_nondominant_spd_requires_explicit_opt_in() {
        let matrix = build_large_paired_matrix(2.0, 2.0);
        assert!(classify_for_run(&matrix, false).is_err());
        assert_eq!(
            classify_for_run(&matrix, true).unwrap(),
            "experimental_symmetric_positive_diagonal_curvature_screen"
        );
    }
    #[test]
    fn large_asymmetric_input_rejected_even_with_opt_in() {
        let matrix = build_large_paired_matrix(2.0, 2.1);
        assert!(classify_for_run(&matrix, true).is_err());
    }
    #[test]
    fn large_negative_curvature_probe_rejected() {
        let matrix = build_large_paired_matrix(4.0, 4.0);
        assert!(classify_for_run(&matrix, true).is_err());
    }
    #[test]
    fn small_indefinite_matrix_remains_rejected_with_opt_in() {
        let matrix = make_matrix(1.0, 1.0, 2.0, 2.0);
        assert!(classify_for_run(&matrix, true).is_err());
    }
}
