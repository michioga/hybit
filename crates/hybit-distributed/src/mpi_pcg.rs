//! G8-C8: MPI rank-local PCG with fixed SPD-compatible local preconditioners.
//!
//! The numerical iteration holds only rank-owned vectors. Halo values are
//! acquired by the G8-B3 overlapped SpMV. Global scalar products use MPI
//! Allreduce. `prepare` and `solve` are collective MPI operations; all ranks
//! must use the same options, operator and call sequence.
//!
//! This is the first correct distributed Krylov baseline, not an automatic
//! SPD detector, multi-host benchmark or general nonsymmetric solver.

use crate::mpi_backend::MpiRuntime;
use crate::mpi_block_jacobi::LocalBlockCholesky;
use crate::mpi_ic0::LocalBlockIc0;
use crate::mpi_overlap::{OverlapSpmv, OverlapTiming};
use crate::mpi_two_level::RankConstantTwoLevelJacobi;
use crate::{HaloPlan, RankLocalCsr};
use std::io;
use std::time::Instant;

fn invalid(msg: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

#[derive(Clone, Copy, Debug)]
pub struct DistributedPcgOptions {
    pub relative_tolerance: f64,
    pub absolute_tolerance: f64,
    pub max_iterations: usize,
}

impl Default for DistributedPcgOptions {
    fn default() -> Self {
        Self {
            relative_tolerance: 1.0e-9,
            absolute_tolerance: 1.0e-13,
            max_iterations: 2000,
        }
    }
}

/// One bounded, collective promotion from rank-constant two-level Jacobi to
/// rank-constant coarse correction plus exact rank-local block Cholesky.
/// This is a guarded G8-C12d-3 experiment, not yet the residual-selected
/// arbitrary-region controller used by the serial HyBIT API.
/// G8-C12d-6B: global-norm residual sample, identical on all MPI ranks.
/// `true_relative_residual` is only available at the initial state, the
/// promotion restart, and the terminal true-residual verification. Interior
/// checkpoints use the recursive PCG residual without additional collectives.
#[derive(Clone, Debug)]
pub struct DistributedPcgResidualSample {
    pub iteration: usize,
    pub phase: &'static str,
    pub recursive_relative_residual: f64,
    pub true_relative_residual: Option<f64>,
}

/// G8-C12d-5: optional rank-local scoring heuristic for fixed contiguous blocks.
/// The chosen mask is frozen after promotion; standard PCG is then restarted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DistributedAdaptiveBlockScore {
    /// Original G8-C12d-4 policy. This path must retain its exact mask.
    ResidualEnergy,
    /// Penalize the L1 coupling across the selected principal block boundary.
    /// Experimental heuristic, not a mathematical convergence guarantee.
    BoundaryCoupling,
    /// D6-A: strictly controlled one-for-one mask intervention on one MPI rank.
    /// It is a sensitivity experiment, never an automatic production heuristic.
    SwapOne {
        target_rank: usize,
        remove: usize,
        add: usize,
    },
    /// G8-C12d-6C: pre-recorded rank-local block masks, for controlled
    /// promotion-timing experiments. Each entry uses bits 0..17 only.
    /// Not intended as an automatic selector or a production API.
    ReplayMask { masks: [u32; 8] },
    /// G8-C12d-6C2: same reference mask, but omit the rank-constant
    /// coarse correction after promotion; validation experiment only.
    ReplayMaskNoCoarse { masks: [u32; 8] },
    /// G8-C12d-6C4A: pooled global *direct factor* budget replay.
    /// The same mask as an existing experiment; only the validation cap changes.
    /// Experimental mode, not an automatic ABTM selector.
    ReplayPooled { masks: [u32; 8] },
}

/// One rank-local decision at the instant of adaptive escalation.
#[derive(Clone, Debug)]
pub struct DistributedAdaptiveBlockDecision {
    pub block_index: usize,
    pub first_owned: usize,
    pub size: usize,
    pub factor_bytes: usize,
    pub residual_energy: f64,
    pub internal_l1: f64,
    pub boundary_l1: f64,
    pub score: f64,
    pub selected: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct DistributedAdaptivePcgPolicy {
    /// Length of EACH global-residual progress window, in PCG iterations.
    /// Recheck at successive windows while the lightweight preconditioner remains active.
    pub probe_iterations: usize,
    /// Promote when ||r_current|| / ||r_previous_checkpoint|| exceeds this threshold.
    pub poor_progress_ratio: f64,
    /// Exact rank-local Cholesky block order (2..=512).
    pub block_size: usize,
    /// Global cap on dense Cholesky payload in bytes. Each MPI rank receives
    /// floor(cap / nranks): deterministic and globally safe, not globally optimal.
    pub factor_budget_bytes_global: usize,
    /// Rank-local block scoring policy; does not alter factorization algebra.
    pub selection_mode: DistributedAdaptiveBlockScore,
}

/// Configuration for a single adaptive MPI-PCG solve.
/// Bundling the PCG stopping options and promotion policy keeps the public
/// call boundary small without suppressing Clippy's argument-count lint.
#[derive(Clone, Copy, Debug)]
pub struct DistributedAdaptiveSolveOptions {
    pub pcg: DistributedPcgOptions,
    pub policy: DistributedAdaptivePcgPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistributedPcgStatus {
    Converged,
    MaxIterations,
    Breakdown,
}

#[derive(Clone, Copy, Debug)]
pub struct DistributedPcgReport {
    pub status: DistributedPcgStatus,
    pub iterations: usize,
    pub initial_residual_norm: f64,
    pub recursive_residual_norm: f64,
    pub true_residual_norm: f64,
    /// `||b-Ax||_2/||b||_2`; for a zero RHS, the absolute norm is reported.
    pub true_relative_residual: f64,
    pub spmv_calls: usize,
    pub allreduce_calls: usize,
    /// Of allreduce_calls, vector reductions for the coarse correction.
    pub coarse_allreduce_calls: usize,
    pub elapsed_ns: u128,
    /// Sum of durations of all rank-local overlapped SpMV calls (includes MPI halo wait).
    pub spmv_elapsed_ns: u128,
    /// Sum of durations of all global reductions issued inside solve (including validation).
    pub allreduce_elapsed_ns: u128,
    /// Subset of both allreduce_elapsed_ns and preconditioner_apply_ns.
    pub coarse_allreduce_elapsed_ns: u128,
    /// Subset of spmv_elapsed_ns spent waiting on MPI nonblocking requests.
    pub halo_wait_ns: u128,
    /// Subset of spmv_elapsed_ns spent computing interior CSR rows.
    pub interior_compute_ns: u128,
    /// Subset of spmv_elapsed_ns spent computing boundary CSR rows.
    pub boundary_compute_ns: u128,
    /// Inclusive time for M^-1 r; two-level includes its MPI reduction.
    pub preconditioner_apply_ns: u128,
    /// Zero means no dynamic promotion was performed.
    pub adaptive_escalation_iteration: usize,
    /// ||r_probe|| / ||r_initial||, or zero if the probe was never reached.
    pub adaptive_probe_ratio: f64,
    /// Collective direct/coarse factor preparation duration on this rank.
    /// This is included in `elapsed_ns`, not in `preconditioner_apply_ns`.
    pub adaptive_factor_setup_ns: u128,
}

/// Select a fixed SPD-compatible preconditioner during collective preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DistributedPcgPreconditioner {
    Jacobi,
    /// Non-overlapping Cholesky-factored contiguous principal diagonal blocks.
    /// A positive local factorization does not certify global SPD.
    LocalBlockCholesky {
        block_size: usize,
    },
    /// No-fill incomplete Cholesky in each contiguous locally owned principal block.
    /// A positive IC(0) factor is an SPD preconditioner, not a certificate for A.
    LocalBlockIc0 {
        block_size: usize,
    },
    /// G8-C12a: positive Jacobi plus replicated rank-constant Galerkin correction.
    /// Experimental input SPD remains unverified.
    RankConstantTwoLevelJacobi,
    /// G8-C12d-1: local exact block Cholesky + rank-constant coarse correction.
    /// The additive preconditioner is fixed throughout PCG. Block factorizations
    /// and coarse Cholesky must succeed; no fallback or SPD certification is implied.
    RankConstantTwoLevelBlockCholesky {
        block_size: usize,
    },
}

#[derive(Debug)]
enum PreparedPreconditioner {
    Jacobi(Vec<f64>),
    LocalBlockCholesky(LocalBlockCholesky),
    LocalBlockIc0(LocalBlockIc0),
    RankConstantTwoLevelJacobi(RankConstantTwoLevelJacobi),
    RankConstantTwoLevelBlockCholesky {
        blocks: LocalBlockCholesky,
        coarse: RankConstantTwoLevelJacobi,
    },
}

impl PreparedPreconditioner {
    fn owned_len(&self) -> usize {
        match self {
            Self::Jacobi(diag) => diag.len(),
            Self::LocalBlockCholesky(blocks) => blocks.owned_len(),
            Self::LocalBlockIc0(blocks) => blocks.owned_len(),
            Self::RankConstantTwoLevelJacobi(coarse) => coarse.owned_len(),
            Self::RankConstantTwoLevelBlockCholesky { blocks, .. } => blocks.owned_len(),
        }
    }

    /// Returns time spent in the additional coarse MPI reduction (if any).
    fn apply(&self, r: &[f64], z: &mut [f64]) -> u128 {
        match self {
            Self::Jacobi(diag) => {
                for ((zi, &ri), &di) in z.iter_mut().zip(r).zip(diag) {
                    *zi = di * ri;
                }
                0
            }
            Self::LocalBlockCholesky(blocks) => {
                blocks.apply(r, z);
                0
            }
            Self::LocalBlockIc0(blocks) => {
                blocks.apply(r, z);
                0
            }
            Self::RankConstantTwoLevelJacobi(coarse) => coarse.apply(r, z),
            Self::RankConstantTwoLevelBlockCholesky { blocks, coarse } => {
                blocks.apply(r, z);
                coarse.add_coarse_correction(r, z)
            }
        }
    }
    fn has_coarse_reduction(&self) -> bool {
        matches!(
            self,
            Self::RankConstantTwoLevelJacobi(_) | Self::RankConstantTwoLevelBlockCholesky { .. }
        )
    }
}

/// Reusable rank-local Krylov vectors, fixed preconditioner and Halo/SpMV buffers.
///
/// Each rank prepares once, then calls solve in the same order on every rank.
/// `local` must remain the exact same matrix as at preparation.
#[derive(Debug)]
pub struct DistributedPcg {
    overlap: OverlapSpmv,
    preconditioner: PreparedPreconditioner,
    ax: Vec<f64>,
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    ap: Vec<f64>,
    adaptive_block_trace: Vec<DistributedAdaptiveBlockDecision>,
    residual_trace_stride: usize,
    residual_trace: Vec<DistributedPcgResidualSample>,
}

/// Extract positive, finite rank-owned diagonal entries, with no dependence on
/// remote CSR columns. Jacobi is deliberately a fixed linear preconditioner.
fn inverse_diagonal(local: &RankLocalCsr) -> io::Result<Vec<f64>> {
    let mut inverse = Vec::with_capacity(local.owned_len());
    for row in 0..local.owned_len() {
        let start = local.row_ptr()[row] as usize;
        let end = local.row_ptr()[row + 1] as usize;
        let mut diagonal = 0.0;
        let mut found = false;
        for pos in start..end {
            if local.col_idx()[pos] as usize == row {
                // Duplicate diagonal entries, if any, are summed.
                diagonal += local.values()[pos];
                found = true;
            }
        }
        if !found || !diagonal.is_finite() || diagonal <= 0.0 {
            return Err(invalid(
                "rank-local Jacobi diagonal must be finite and positive",
            ));
        }
        inverse.push(1.0 / diagonal);
    }
    Ok(inverse)
}

fn local_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Scalar collective timing is inclusive: MPI synchronization and local call overhead.
fn timed_sum(mpi: &MpiRuntime, value: f64, elapsed: &mut u128) -> f64 {
    let started = Instant::now();
    let result = mpi.all_reduce_sum_f64(value);
    *elapsed += started.elapsed().as_nanos();
    result
}

fn accumulate_spmv(
    elapsed: OverlapTiming,
    total: &mut u128,
    wait: &mut u128,
    interior: &mut u128,
    boundary: &mut u128,
) {
    *total += elapsed.elapsed_ns;
    *wait += elapsed.wait_ns;
    *interior += elapsed.interior_ns;
    *boundary += elapsed.boundary_ns;
}

/// Select principal blocks by residual energy per dense-factor byte.
/// Tie breaks are deterministic. `budget` covers dense direct factors only.
/// No MPI communication here; a caller must allocate rank-local quotas.
fn select_residual_blocks(r: &[f64], block_size: usize, budget: usize) -> Vec<bool> {
    let count = r.len().div_ceil(block_size);
    let mut candidates = Vec::with_capacity(count);
    for (i, entries) in r.chunks(block_size).enumerate() {
        let bytes = entries.len() * entries.len() * std::mem::size_of::<f64>();
        let energy: f64 = entries.iter().map(|v| v * v).sum();
        candidates.push((i, energy / bytes as f64, bytes));
    }
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut mask = vec![false; count];
    let mut remaining = budget;
    for (i, _, bytes) in candidates {
        if bytes <= remaining {
            mask[i] = true;
            remaining -= bytes;
        }
    }
    mask
}

/// Swap one selected principal block for one previously unselected block.
/// Return false without mutation for invalid indices, costs or the budget.
/// In particular, a successful swap cannot increase the rank-local direct
/// factor payload beyond `budget`.
fn swap_selected_block(
    mask: &mut [bool],
    bytes: &[usize],
    remove: usize,
    add: usize,
    budget: usize,
) -> bool {
    if mask.len() != bytes.len()
        || remove >= mask.len()
        || add >= mask.len()
        || remove == add
        || !mask[remove]
        || mask[add]
    {
        return false;
    }
    let Some(used) = mask
        .iter()
        .zip(bytes)
        .try_fold(0usize, |sum, (&chosen, &cost)| {
            if chosen {
                sum.checked_add(cost)
            } else {
                Some(sum)
            }
        })
    else {
        return false;
    };
    let Some(after) = used
        .checked_sub(bytes[remove])
        .and_then(|v| v.checked_add(bytes[add]))
    else {
        return false;
    };
    if after > budget {
        return false;
    }
    mask[remove] = false;
    mask[add] = true;
    true
}

/// G8-C12d-6C4A: ABTM-inspired packed block mask (8 ranks x 18 blocks = 144 bits).
/// This is a *block selection* bitmap, not the ABTM SpMV value/tile encoding.
/// Logical set operations are wordwise and never change floating point arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AbtmPooledBlockMask([u64; 3]);

impl AbtmPooledBlockMask {
    const BLOCKS_PER_RANK: usize = 18;
    const RANKS: usize = 8;
    const VALID_BITS_PER_RANK: u32 = (1u32 << 18) - 1;

    fn from_rank_masks(rank_masks: [u32; 8]) -> Option<Self> {
        if rank_masks
            .iter()
            .any(|m| *m & !Self::VALID_BITS_PER_RANK != 0)
        {
            return None;
        }
        let mut words = [0u64; 3];
        for (rank, rank_mask) in rank_masks.into_iter().enumerate() {
            let mut remaining = rank_mask;
            while remaining != 0 {
                let local_index = remaining.trailing_zeros() as usize;
                let bit = rank * Self::BLOCKS_PER_RANK + local_index;
                words[bit / 64] |= 1u64 << (bit % 64);
                remaining &= remaining - 1;
            }
        }
        Some(Self(words))
    }

    fn rank_bits(self, rank: usize) -> u32 {
        assert!(rank < Self::RANKS);
        let first = rank * Self::BLOCKS_PER_RANK;
        let word = first / 64;
        let offset = first % 64;
        let mut bits = self.0[word] >> offset;
        if offset + Self::BLOCKS_PER_RANK > 64 {
            bits |= self.0[word + 1] << (64 - offset);
        }
        (bits as u32) & Self::VALID_BITS_PER_RANK
    }

    #[cfg(test)]
    fn intersection(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] & other.0[i]))
    }

    #[cfg(test)]
    fn union(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] | other.0[i]))
    }

    #[cfg(test)]
    fn symmetric_difference(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] ^ other.0[i]))
    }

    #[cfg(test)]
    fn count_selected(self) -> u32 {
        self.0.iter().map(|word| word.count_ones()).sum()
    }
}

/// Decode a reference mask without changing any selection until validation passes.
/// Only 8-rank replay traces (up to 18 blocks per rank) are supported here.
fn replay_reference_mask(
    mask_bits: u32,
    dense_factor_bytes: &[usize],
    rank_budget: usize,
) -> Option<Vec<bool>> {
    if dense_factor_bytes.is_empty()
        || dense_factor_bytes.len() > 18
        || (mask_bits >> dense_factor_bytes.len()) != 0
    {
        return None;
    }
    let mut bytes = 0usize;
    let mask: Vec<bool> = dense_factor_bytes
        .iter()
        .enumerate()
        .map(|(i, cost)| {
            let chosen = ((mask_bits >> i) & 1) != 0;
            if chosen {
                bytes = bytes.saturating_add(*cost);
            }
            chosen
        })
        .collect();
    if bytes > rank_budget || !mask.iter().any(|&b| b) {
        return None;
    }
    Some(mask)
}

/// Return both a bounded selection mask and the metrics explaining every choice.
/// No collective communication: each MPI rank uses its assigned local quota.
/// For an original-policy comparison, keep select_residual_blocks unchanged.
fn select_budgeted_blocks(
    local: &RankLocalCsr,
    residual: &[f64],
    block_size: usize,
    budget: usize,
    mode: DistributedAdaptiveBlockScore,
) -> (Vec<bool>, Vec<DistributedAdaptiveBlockDecision>) {
    let mut trace = Vec::with_capacity(residual.len().div_ceil(block_size));
    for (i, entries) in residual.chunks(block_size).enumerate() {
        let first = i * block_size;
        let end = first + entries.len();
        let mut internal_l1 = 0.0;
        let mut boundary_l1 = 0.0;
        for row in first..end {
            for p in local.row_ptr()[row] as usize..local.row_ptr()[row + 1] as usize {
                let col = local.col_idx()[p] as usize;
                let magnitude = local.values()[p].abs();
                if (first..end).contains(&col) {
                    internal_l1 += magnitude;
                } else {
                    // Includes both locally owned external blocks and MPI ghosts.
                    boundary_l1 += magnitude;
                }
            }
        }
        let residual_energy: f64 = entries.iter().map(|&v| v * v).sum();
        let bytes = entries.len() * entries.len() * std::mem::size_of::<f64>();
        let baseline_score = residual_energy / bytes as f64;
        let score = match mode {
            DistributedAdaptiveBlockScore::ResidualEnergy
            | DistributedAdaptiveBlockScore::SwapOne { .. }
            | DistributedAdaptiveBlockScore::ReplayMask { .. }
            | DistributedAdaptiveBlockScore::ReplayMaskNoCoarse { .. }
            | DistributedAdaptiveBlockScore::ReplayPooled { .. } => baseline_score,
            DistributedAdaptiveBlockScore::BoundaryCoupling => {
                // A dimensionless boundary penalty. This rewards blocks whose
                // inverse can address relatively more of their internal coupling.
                // The factor is strictly positive and bounded in (0,1].
                let proportion = boundary_l1 / (internal_l1 + boundary_l1).max(f64::MIN_POSITIVE);
                baseline_score / (1.0 + proportion)
            }
        };
        trace.push(DistributedAdaptiveBlockDecision {
            block_index: i,
            first_owned: first,
            size: entries.len(),
            factor_bytes: bytes,
            residual_energy,
            internal_l1,
            boundary_l1,
            score,
            selected: false,
        });
    }
    let mask = match mode {
        DistributedAdaptiveBlockScore::ResidualEnergy
        | DistributedAdaptiveBlockScore::SwapOne { .. }
        | DistributedAdaptiveBlockScore::ReplayMask { .. }
        | DistributedAdaptiveBlockScore::ReplayMaskNoCoarse { .. }
        | DistributedAdaptiveBlockScore::ReplayPooled { .. } => {
            // Preserve G8-C12d-4's exact tie-break and knapsack heuristic.
            select_residual_blocks(residual, block_size, budget)
        }
        DistributedAdaptiveBlockScore::BoundaryCoupling => {
            let mut order: Vec<usize> = (0..trace.len()).collect();
            order.sort_by(|&a, &b| {
                trace[b]
                    .score
                    .total_cmp(&trace[a].score)
                    .then_with(|| a.cmp(&b))
            });
            let mut remaining = budget;
            let mut selected = vec![false; trace.len()];
            for index in order {
                let bytes = trace[index].factor_bytes;
                if bytes <= remaining {
                    selected[index] = true;
                    remaining -= bytes;
                }
            }
            selected
        }
    };
    for (record, &chosen) in trace.iter_mut().zip(&mask) {
        record.selected = chosen;
    }
    (mask, trace)
}

/// Return progress relative to the previous checkpoint (not to the initial RHS).
/// No reductions or allocations are needed: residual norms are already global.
fn adaptive_window_ratio(
    step: usize,
    interval: usize,
    current_norm: f64,
    last_checkpoint_norm: &mut f64,
) -> Option<f64> {
    if interval == 0 || step % interval != 0 {
        return None;
    }
    let ratio = current_norm / *last_checkpoint_norm;
    *last_checkpoint_norm = current_norm;
    Some(ratio)
}

impl DistributedPcg {
    /// Backward-compatible G8-C1 Jacobi constructor.
    pub fn prepare(mpi: &MpiRuntime, plan: &HaloPlan, local: &RankLocalCsr) -> io::Result<Self> {
        Self::prepare_with_preconditioner(mpi, plan, local, DistributedPcgPreconditioner::Jacobi)
    }

    /// Collective preparation. Reject a factorization error on *all* ranks
    /// before any rank enters OverlapSpmv's collective setup.
    pub fn prepare_with_preconditioner(
        mpi: &MpiRuntime,
        plan: &HaloPlan,
        local: &RankLocalCsr,
        choice: DistributedPcgPreconditioner,
    ) -> io::Result<Self> {
        let prepared = match choice {
            DistributedPcgPreconditioner::Jacobi => {
                inverse_diagonal(local).map(PreparedPreconditioner::Jacobi)
            }
            DistributedPcgPreconditioner::LocalBlockCholesky { block_size } => {
                LocalBlockCholesky::prepare(local, block_size)
                    .map(PreparedPreconditioner::LocalBlockCholesky)
            }
            DistributedPcgPreconditioner::LocalBlockIc0 { block_size } => {
                LocalBlockIc0::prepare(local, block_size).map(PreparedPreconditioner::LocalBlockIc0)
            }
            DistributedPcgPreconditioner::RankConstantTwoLevelJacobi => {
                RankConstantTwoLevelJacobi::prepare(mpi, plan, local)
                    .map(PreparedPreconditioner::RankConstantTwoLevelJacobi)
            }
            DistributedPcgPreconditioner::RankConstantTwoLevelBlockCholesky { block_size } => {
                // Validate every local direct factor collectively before starting
                // coarse preparation, which itself contains MPI collectives.
                let blocks = LocalBlockCholesky::prepare(local, block_size);
                if mpi.all_reduce_sum_u64(u64::from(blocks.is_err())) != 0 {
                    Err(invalid(
                        "at least one MPI rank rejected local hybrid Cholesky",
                    ))
                } else {
                    blocks.and_then(|blocks| {
                        RankConstantTwoLevelJacobi::prepare(mpi, plan, local).map(|coarse| {
                            PreparedPreconditioner::RankConstantTwoLevelBlockCholesky {
                                blocks,
                                coarse,
                            }
                        })
                    })
                }
            }
        };
        let local_bad = u64::from(prepared.is_err() || plan.rank() != mpi.rank() as u32);
        if mpi.all_reduce_sum_u64(local_bad) != 0 {
            return Err(invalid(
                "at least one MPI rank has an invalid local PCG preconditioner",
            ));
        }
        let preconditioner = prepared?;
        let overlap = OverlapSpmv::prepare(mpi, plan, local)?;
        let n = local.owned_len();
        Ok(Self {
            overlap,
            preconditioner,
            ax: vec![0.0; n],
            r: vec![0.0; n],
            z: vec![0.0; n],
            p: vec![0.0; n],
            ap: vec![0.0; n],
            adaptive_block_trace: Vec::new(),
            residual_trace_stride: 0,
            residual_trace: Vec::new(),
        })
    }

    /// Opt-in sampling of the already globally reduced recursive residual.
    /// 0 disables history recording. Setting 64 stores one sample every 64 steps.
    /// This does NOT compute a fresh true residual at every checkpoint.
    pub fn set_residual_trace_stride(&mut self, stride: usize) {
        self.residual_trace_stride = stride;
    }

    pub fn residual_trace(&self) -> &[DistributedPcgResidualSample] {
        &self.residual_trace
    }

    /// Only populated if this solve triggered adaptive promotion.
    /// Ordered by increasing rank-local block index, not by selection priority.
    pub fn adaptive_block_trace(&self) -> &[DistributedAdaptiveBlockDecision] {
        &self.adaptive_block_trace
    }

    /// Rank-local factor storage payload only (not allocator/Vec overhead).
    /// Use MPI max/sum reductions in benchmark code to report across ranks.
    pub fn preconditioner_factor_bytes(&self) -> usize {
        match &self.preconditioner {
            PreparedPreconditioner::Jacobi(d) => d.len() * std::mem::size_of::<f64>(),
            PreparedPreconditioner::LocalBlockCholesky(p) => p.factor_bytes(),
            PreparedPreconditioner::LocalBlockIc0(p) => p.factor_bytes(),
            PreparedPreconditioner::RankConstantTwoLevelJacobi(p) => p.factor_bytes(),
            PreparedPreconditioner::RankConstantTwoLevelBlockCholesky { blocks, coarse } => {
                blocks.factor_bytes() + coarse.factor_bytes()
            }
        }
    }

    /// f64 factor entries. For dense blocks this counts stored n*n entries,
    /// including the unused upper triangle; IC(0) includes the diagonal.
    pub fn preconditioner_factor_values(&self) -> usize {
        match &self.preconditioner {
            PreparedPreconditioner::Jacobi(d) => d.len(),
            PreparedPreconditioner::LocalBlockCholesky(p) => p.factor_bytes() / 8,
            PreparedPreconditioner::LocalBlockIc0(p) => p.factor_values(),
            PreparedPreconditioner::RankConstantTwoLevelJacobi(p) => p.factor_values(),
            PreparedPreconditioner::RankConstantTwoLevelBlockCholesky { blocks, coarse } => {
                blocks.factor_bytes() / 8 + coarse.factor_values()
            }
        }
    }

    pub fn interior_rows(&self) -> usize {
        self.overlap.interior_len()
    }

    pub fn boundary_rows(&self) -> usize {
        self.overlap.boundary_len()
    }

    /// Solve `Ax=b` using a fixed SPD-compatible preconditioned distributed PCG.
    /// Convergence is ALWAYS checked against a fresh, true `b-Ax` SpMV.
    /// Non-positive curvature is returned as Breakdown, not convergence.
    pub fn solve(
        &mut self,
        mpi: &MpiRuntime,
        local: &RankLocalCsr,
        b: &[f64],
        x: &mut [f64],
        options: DistributedPcgOptions,
    ) -> io::Result<DistributedPcgReport> {
        self.solve_impl(mpi, local, b, x, options, None)
    }

    /// Keep PCG recurrence intact if the pilot is making acceptable progress.
    /// If the collective probe detects poor progress, factorize local principal
    /// blocks, add the pre-existing rank-constant coarse correction and restart
    /// PCG from a *true* residual with the new fixed SPD preconditioner.
    /// `prepare_with_preconditioner(...RankConstantTwoLevelJacobi)` must have
    /// initialized this solver. All ranks must pass identical policy/options.
    pub fn solve_adaptive(
        &mut self,
        mpi: &MpiRuntime,
        plan: &HaloPlan,
        local: &RankLocalCsr,
        b: &[f64],
        x: &mut [f64],
        config: DistributedAdaptiveSolveOptions,
    ) -> io::Result<DistributedPcgReport> {
        self.solve_impl(mpi, local, b, x, config.pcg, Some((plan, config.policy)))
    }

    fn solve_impl(
        &mut self,
        mpi: &MpiRuntime,
        local: &RankLocalCsr,
        b: &[f64],
        x: &mut [f64],
        options: DistributedPcgOptions,
        adaptive: Option<(&HaloPlan, DistributedAdaptivePcgPolicy)>,
    ) -> io::Result<DistributedPcgReport> {
        self.adaptive_block_trace.clear();
        self.residual_trace.clear();
        let start = Instant::now();
        let mut spmv_elapsed_ns = 0u128;
        let mut allreduce_elapsed_ns = 0u128;
        let mut halo_wait_ns = 0u128;
        let mut interior_compute_ns = 0u128;
        let mut boundary_compute_ns = 0u128;
        let mut preconditioner_apply_ns = 0u128;
        let mut coarse_allreduce_elapsed_ns = 0u128;
        let mut coarse_allreduce_calls = 0usize;
        let mut adaptive_escalation_iteration = 0usize;
        let mut adaptive_probe_ratio = 0.0_f64;
        let mut adaptive_factor_setup_ns = 0u128;
        let n = self.preconditioner.owned_len();
        let local_bad = b.len() != n
            || x.len() != n
            || local.owned_len() != n
            || local.rank() as i32 != mpi.rank()
            || !options.relative_tolerance.is_finite()
            || options.relative_tolerance <= 0.0
            || !options.absolute_tolerance.is_finite()
            || options.absolute_tolerance < 0.0
            || options.max_iterations == 0
            || b.iter().any(|v| !v.is_finite())
            || x.iter().any(|v| !v.is_finite())
            || adaptive.is_some_and(|(plan, policy)| {
                plan.rank() as i32 != mpi.rank()
                    || plan.owned_len() != n
                    || !matches!(
                        &self.preconditioner,
                        PreparedPreconditioner::RankConstantTwoLevelJacobi(_)
                    )
                    || policy.probe_iterations == 0
                    || policy.probe_iterations >= options.max_iterations
                    || !(2..=512).contains(&policy.block_size)
                    || policy.factor_budget_bytes_global == 0
                    || !policy.poor_progress_ratio.is_finite()
                    || policy.poor_progress_ratio <= 0.0
                    || policy.poor_progress_ratio >= 1.0
            });
        // The count collective ensures bad local inputs do not strand peers.
        let t_validation = Instant::now();
        let global_bad = mpi.all_reduce_sum_u64(u64::from(local_bad));
        allreduce_elapsed_ns += t_validation.elapsed().as_nanos();
        if global_bad != 0 {
            return Err(invalid(
                "invalid distributed PCG vector dimensions, options or values",
            ));
        }
        let mut reductions = 1usize;
        let mut spmv_calls = 0usize;
        let timing = self.overlap.spmv_overlap(mpi, local, x, &mut self.ax)?;
        accumulate_spmv(
            timing,
            &mut spmv_elapsed_ns,
            &mut halo_wait_ns,
            &mut interior_compute_ns,
            &mut boundary_compute_ns,
        );
        spmv_calls += 1;
        for (i, &bi) in b.iter().enumerate() {
            self.r[i] = bi - self.ax[i];
        }
        let t_precond = Instant::now();
        let coarse_ns = self.preconditioner.apply(&self.r, &mut self.z);
        preconditioner_apply_ns += t_precond.elapsed().as_nanos();
        coarse_allreduce_elapsed_ns += coarse_ns;
        if self.preconditioner.has_coarse_reduction() {
            coarse_allreduce_calls += 1;
        }
        self.p.copy_from_slice(&self.z);
        let b_sq = timed_sum(mpi, local_dot(b, b), &mut allreduce_elapsed_ns);
        let mut r_sq = timed_sum(mpi, local_dot(&self.r, &self.r), &mut allreduce_elapsed_ns);
        reductions += 2;
        let norm_b = b_sq.sqrt();
        let initial_norm = r_sq.sqrt();
        if self.residual_trace_stride != 0 {
            let relative = if norm_b > 0.0 {
                initial_norm / norm_b
            } else {
                initial_norm
            };
            self.residual_trace.push(DistributedPcgResidualSample {
                iteration: 0,
                phase: if adaptive.is_some() { "pre" } else { "fixed" },
                recursive_relative_residual: relative,
                true_relative_residual: Some(relative),
            });
        }
        let mut adaptive_checkpoint_norm = initial_norm;
        let threshold = options
            .absolute_tolerance
            .max(options.relative_tolerance * norm_b);
        let mut status = DistributedPcgStatus::MaxIterations;
        let mut iterations = 0usize;
        if initial_norm <= threshold {
            status = DistributedPcgStatus::Converged;
        } else {
            let mut rz = timed_sum(mpi, local_dot(&self.r, &self.z), &mut allreduce_elapsed_ns);
            reductions += 1;
            if !rz.is_finite() || rz <= 0.0 {
                status = DistributedPcgStatus::Breakdown;
            } else {
                for step in 1..=options.max_iterations {
                    let timing = self
                        .overlap
                        .spmv_overlap(mpi, local, &self.p, &mut self.ap)?;
                    accumulate_spmv(
                        timing,
                        &mut spmv_elapsed_ns,
                        &mut halo_wait_ns,
                        &mut interior_compute_ns,
                        &mut boundary_compute_ns,
                    );
                    spmv_calls += 1;
                    let p_ap =
                        timed_sum(mpi, local_dot(&self.p, &self.ap), &mut allreduce_elapsed_ns);
                    reductions += 1;
                    if !p_ap.is_finite() || p_ap <= 0.0 {
                        status = DistributedPcgStatus::Breakdown;
                        break;
                    }
                    let alpha = rz / p_ap;
                    if !alpha.is_finite() {
                        status = DistributedPcgStatus::Breakdown;
                        break;
                    }
                    for (i, xi) in x.iter_mut().enumerate() {
                        *xi += alpha * self.p[i];
                        self.r[i] -= alpha * self.ap[i];
                    }
                    iterations = step;
                    r_sq = timed_sum(mpi, local_dot(&self.r, &self.r), &mut allreduce_elapsed_ns);
                    reductions += 1;
                    if !r_sq.is_finite() || r_sq < 0.0 {
                        status = DistributedPcgStatus::Breakdown;
                        break;
                    }
                    if self.residual_trace_stride != 0 && step % self.residual_trace_stride == 0 {
                        let recursive = if norm_b > 0.0 {
                            r_sq.sqrt() / norm_b
                        } else {
                            r_sq.sqrt()
                        };
                        self.residual_trace.push(DistributedPcgResidualSample {
                            iteration: step,
                            phase: if adaptive.is_none() {
                                "fixed"
                            } else if adaptive_escalation_iteration == 0 {
                                "pre"
                            } else {
                                "post"
                            },
                            recursive_relative_residual: recursive,
                            true_relative_residual: None,
                        });
                    }
                    if r_sq.sqrt() <= threshold {
                        status = DistributedPcgStatus::Converged;
                        break;
                    }
                    if let Some((plan, policy)) = adaptive {
                        if adaptive_escalation_iteration == 0 {
                            if let Some(window_ratio) = adaptive_window_ratio(
                                step,
                                policy.probe_iterations,
                                r_sq.sqrt(),
                                &mut adaptive_checkpoint_norm,
                            ) {
                                adaptive_probe_ratio = window_ratio;
                                if adaptive_probe_ratio > policy.poor_progress_ratio {
                                    // Every rank makes the same choice: r_sq and initial_norm
                                    // are MPI global reductions. Reject local factor failures
                                    // collectively BEFORE coarse preparation enters collectives.
                                    let started = Instant::now();
                                    let rank_budget =
                                        policy.factor_budget_bytes_global / mpi.size() as usize;
                                    let (mut mask, mut decisions) = select_budgeted_blocks(
                                        local,
                                        &self.r,
                                        policy.block_size,
                                        rank_budget,
                                        policy.selection_mode,
                                    );
                                    if let DistributedAdaptiveBlockScore::SwapOne {
                                        target_rank,
                                        remove,
                                        add,
                                    } = policy.selection_mode
                                    {
                                        // Collective validation prevents an invalid intervention
                                        // on one rank from leaving other ranks inside MPI calls.
                                        let mut locally_valid = target_rank < mpi.size() as usize;
                                        if locally_valid && target_rank == mpi.rank() as usize {
                                            let costs: Vec<usize> =
                                                decisions.iter().map(|d| d.factor_bytes).collect();
                                            locally_valid = swap_selected_block(
                                                &mut mask,
                                                &costs,
                                                remove,
                                                add,
                                                rank_budget,
                                            );
                                            if locally_valid {
                                                for (record, &chosen) in
                                                    decisions.iter_mut().zip(&mask)
                                                {
                                                    record.selected = chosen;
                                                }
                                            }
                                        }
                                        let rejected =
                                            mpi.all_reduce_sum_u64(u64::from(!locally_valid));
                                        reductions += 1;
                                        if rejected != 0 {
                                            return Err(invalid(
                                                "invalid or over-budget adaptive block swap",
                                            ));
                                        }
                                    }
                                    if let DistributedAdaptiveBlockScore::ReplayMask { masks }
                                    | DistributedAdaptiveBlockScore::ReplayPooled { masks }
                                    | DistributedAdaptiveBlockScore::ReplayMaskNoCoarse {
                                        masks,
                                    } = policy.selection_mode
                                    {
                                        let rank = mpi.rank() as usize;
                                        let pooled = matches!(
                                            policy.selection_mode,
                                            DistributedAdaptiveBlockScore::ReplayPooled { .. }
                                        );
                                        let packed = AbtmPooledBlockMask::from_rank_masks(masks);
                                        let replayed = if mpi.size() == 8 && rank < masks.len() {
                                            let costs: Vec<usize> =
                                                decisions.iter().map(|d| d.factor_bytes).collect();
                                            packed.and_then(|bitmap| {
                                                replay_reference_mask(
                                                    bitmap.rank_bits(rank),
                                                    &costs,
                                                    if pooled { usize::MAX } else { rank_budget },
                                                )
                                            })
                                        } else {
                                            None
                                        };
                                        // All ranks validate before calling Cholesky preparation.
                                        let rejected =
                                            mpi.all_reduce_sum_u64(u64::from(replayed.is_none()));
                                        reductions += 1;
                                        if rejected != 0 {
                                            return Err(invalid("replay mask invalid, over budget, or not eight ranks"));
                                        }
                                        if pooled {
                                            // Every rank enters this reduction after the validation
                                            // collective. Only direct Cholesky bytes are budgeted;
                                            // coarse/Jacobi payload is reported separately.
                                            let local_bytes = decisions
                                                .iter()
                                                .zip(replayed.as_ref().ok_or_else(|| {
                                                    invalid("replay mask missing")
                                                })?)
                                                .filter(|(_, selected)| **selected)
                                                .try_fold(0u64, |sum, (d, _)| {
                                                    sum.checked_add(d.factor_bytes as u64)
                                                });
                                            // Overflow is collectively treated as an over-budget run.
                                            let local_bytes = local_bytes.unwrap_or(u64::MAX);
                                            let global_bytes = mpi.all_reduce_sum_u64(local_bytes);
                                            reductions += 1;
                                            if global_bytes
                                                > policy.factor_budget_bytes_global as u64
                                            {
                                                return Err(invalid("pooled replay exceeds global direct-factor budget"));
                                            }
                                        }
                                        mask = replayed
                                            .ok_or_else(|| invalid("replay mask missing"))?;
                                        for (record, &chosen) in decisions.iter_mut().zip(&mask) {
                                            record.selected = chosen;
                                        }
                                    }
                                    self.adaptive_block_trace = decisions;
                                    let any_selected = mpi.all_reduce_sum_u64(
                                        mask.iter().filter(|&&v| v).count() as u64,
                                    );
                                    reductions += 1;
                                    if any_selected != 0 {
                                        let blocks = LocalBlockCholesky::prepare_selected(
                                            local,
                                            policy.block_size,
                                            &mask,
                                        );
                                        if mpi.all_reduce_sum_u64(u64::from(blocks.is_err())) != 0 {
                                            return Err(invalid(
                                        "adaptive local Cholesky failed on at least one MPI rank",
                                    ));
                                        }
                                        let blocks = blocks?;
                                        // Both arms reuse the exact same collectively validated, replayed
                                        // selected local factors and Jacobi fallback.
                                        // Only the rank-constant correction is toggled.
                                        if matches!(
                                            policy.selection_mode,
                                            DistributedAdaptiveBlockScore::ReplayMaskNoCoarse { .. }
                                        ) {
                                            self.preconditioner =
                                                PreparedPreconditioner::LocalBlockCholesky(blocks);
                                        } else {
                                            let coarse = RankConstantTwoLevelJacobi::prepare(
                                                mpi, plan, local,
                                            )?;
                                            self.preconditioner =
                                        PreparedPreconditioner::RankConstantTwoLevelBlockCholesky {
                                            blocks,
                                            coarse,
                                        };
                                        }
                                        adaptive_factor_setup_ns = started.elapsed().as_nanos();
                                        adaptive_escalation_iteration = step;

                                        // Preconditioner changed: restart PCG with the actual
                                        // residual, not the recursively accumulated residual.
                                        let timing = self.overlap.spmv_overlap(
                                            mpi,
                                            local,
                                            x,
                                            &mut self.ax,
                                        )?;
                                        accumulate_spmv(
                                            timing,
                                            &mut spmv_elapsed_ns,
                                            &mut halo_wait_ns,
                                            &mut interior_compute_ns,
                                            &mut boundary_compute_ns,
                                        );
                                        spmv_calls += 1;
                                        for (i, &bi) in b.iter().enumerate() {
                                            self.r[i] = bi - self.ax[i];
                                        }
                                        r_sq = timed_sum(
                                            mpi,
                                            local_dot(&self.r, &self.r),
                                            &mut allreduce_elapsed_ns,
                                        );
                                        reductions += 1;
                                        // Sample the actual b-Ax residual following promotion.
                                        // The pre-existing numerical guard remains untouched.
                                        if self.residual_trace_stride != 0
                                            && r_sq.is_finite()
                                            && r_sq >= 0.0
                                        {
                                            let actual_norm = r_sq.sqrt();
                                            let relative = if norm_b > 0.0 {
                                                actual_norm / norm_b
                                            } else {
                                                actual_norm
                                            };
                                            self.residual_trace.push(
                                                DistributedPcgResidualSample {
                                                    iteration: step,
                                                    phase: "promotion_restart",
                                                    recursive_relative_residual: relative,
                                                    true_relative_residual: Some(relative),
                                                },
                                            );
                                        }
                                        if !r_sq.is_finite() || r_sq < 0.0 {
                                            status = DistributedPcgStatus::Breakdown;
                                            break;
                                        }
                                        if r_sq.sqrt() <= threshold {
                                            status = DistributedPcgStatus::Converged;
                                            break;
                                        }
                                        let t_precond = Instant::now();
                                        let coarse_ns =
                                            self.preconditioner.apply(&self.r, &mut self.z);
                                        preconditioner_apply_ns += t_precond.elapsed().as_nanos();
                                        coarse_allreduce_elapsed_ns += coarse_ns;
                                        if self.preconditioner.has_coarse_reduction() {
                                            coarse_allreduce_calls += 1;
                                        }
                                        let next_rz = timed_sum(
                                            mpi,
                                            local_dot(&self.r, &self.z),
                                            &mut allreduce_elapsed_ns,
                                        );
                                        reductions += 1;
                                        if !next_rz.is_finite() || next_rz <= 0.0 {
                                            status = DistributedPcgStatus::Breakdown;
                                            break;
                                        }
                                        self.p.copy_from_slice(&self.z);
                                        rz = next_rz;
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                    let t_precond = Instant::now();
                    let coarse_ns = self.preconditioner.apply(&self.r, &mut self.z);
                    preconditioner_apply_ns += t_precond.elapsed().as_nanos();
                    coarse_allreduce_elapsed_ns += coarse_ns;
                    if self.preconditioner.has_coarse_reduction() {
                        coarse_allreduce_calls += 1;
                    }
                    let next_rz =
                        timed_sum(mpi, local_dot(&self.r, &self.z), &mut allreduce_elapsed_ns);
                    reductions += 1;
                    if !next_rz.is_finite() || next_rz <= 0.0 {
                        status = DistributedPcgStatus::Breakdown;
                        break;
                    }
                    let beta = next_rz / rz;
                    if !beta.is_finite() {
                        status = DistributedPcgStatus::Breakdown;
                        break;
                    }
                    for (i, pi) in self.p.iter_mut().enumerate() {
                        *pi = self.z[i] + beta * *pi;
                    }
                    rz = next_rz;
                }
            }
        }
        // True-residual verification is independent of recursive residual.
        let timing = self.overlap.spmv_overlap(mpi, local, x, &mut self.ax)?;
        accumulate_spmv(
            timing,
            &mut spmv_elapsed_ns,
            &mut halo_wait_ns,
            &mut interior_compute_ns,
            &mut boundary_compute_ns,
        );
        spmv_calls += 1;
        let true_local_sq: f64 = b
            .iter()
            .zip(&self.ax)
            .map(|(&bi, &axi)| (bi - axi) * (bi - axi))
            .sum();
        let true_norm = timed_sum(mpi, true_local_sq, &mut allreduce_elapsed_ns).sqrt();
        reductions += 1;
        if status == DistributedPcgStatus::Converged && true_norm > threshold {
            status = DistributedPcgStatus::MaxIterations;
        }
        if self.residual_trace_stride != 0 {
            self.residual_trace.push(DistributedPcgResidualSample {
                iteration: iterations,
                phase: "final_true",
                recursive_relative_residual: if norm_b > 0.0 {
                    r_sq.sqrt() / norm_b
                } else {
                    r_sq.sqrt()
                },
                true_relative_residual: Some(if norm_b > 0.0 {
                    true_norm / norm_b
                } else {
                    true_norm
                }),
            });
        }
        Ok(DistributedPcgReport {
            status,
            iterations,
            initial_residual_norm: initial_norm,
            recursive_residual_norm: r_sq.sqrt(),
            true_residual_norm: true_norm,
            true_relative_residual: if norm_b > 0.0 {
                true_norm / norm_b
            } else {
                true_norm
            },
            spmv_calls,
            allreduce_calls: reductions + coarse_allreduce_calls,
            coarse_allreduce_calls,
            elapsed_ns: start.elapsed().as_nanos(),
            spmv_elapsed_ns,
            allreduce_elapsed_ns: allreduce_elapsed_ns + coarse_allreduce_elapsed_ns,
            coarse_allreduce_elapsed_ns,
            halo_wait_ns,
            interior_compute_ns,
            boundary_compute_ns,
            preconditioner_apply_ns,
            adaptive_escalation_iteration,
            adaptive_probe_ratio,
            adaptive_factor_setup_ns,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_contiguous_halo_plans, prepare_rank_local_csr, ContiguousPartition};
    use hybit_matrix::Csr32Matrix;

    #[test]
    fn adaptive_policy_guard_rejects_impossible_or_nonfinite_configurations() {
        let valid = DistributedAdaptivePcgPolicy {
            probe_iterations: 64,
            poor_progress_ratio: 0.5,
            block_size: 128,
            factor_budget_bytes_global: 4 * 1024 * 1024,
            selection_mode: DistributedAdaptiveBlockScore::ResidualEnergy,
        };
        assert!((2..=512).contains(&valid.block_size));
        assert!(valid.probe_iterations < 1000);
        assert!(valid.poor_progress_ratio.is_finite());
        assert!(valid.poor_progress_ratio > 0.0 && valid.poor_progress_ratio < 1.0);
        for bad in [-1.0, 0.0, 1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(!bad.is_finite() || bad <= 0.0 || bad >= 1.0);
        }
    }

    #[test]
    fn residual_trace_disabled_by_default() {
        let sample = DistributedPcgResidualSample {
            iteration: 256,
            phase: "promotion_restart",
            recursive_relative_residual: 1e-4,
            true_relative_residual: Some(1.01e-4),
        };
        assert_eq!(sample.iteration, 256);
        assert_eq!(sample.phase, "promotion_restart");
        assert!(sample.true_relative_residual.unwrap() > sample.recursive_relative_residual);
    }

    #[test]
    fn abtm_pooled_mask_set_algebra_matches_scalar_baseline() {
        let a = [
            0x3fe7f, 0x3faff, 0x3f7df, 0x2bfff, 0x3ff7d, 0x3deff, 0x3f3ff, 0x1bfff,
        ];
        let b = [
            0x3ff7f, 0x3fbff, 0x3ffdf, 0x3bfff, 0x3fffd, 0x3feff, 0x3fbff, 0x1ffff,
        ];
        let packed_a = AbtmPooledBlockMask::from_rank_masks(a).unwrap();
        let packed_b = AbtmPooledBlockMask::from_rank_masks(b).unwrap();
        for rank in 0..8 {
            assert_eq!(packed_a.rank_bits(rank), a[rank]);
            assert_eq!(packed_b.rank_bits(rank), b[rank]);
            assert_eq!(
                packed_a.intersection(packed_b).rank_bits(rank),
                a[rank] & b[rank]
            );
            assert_eq!(packed_a.union(packed_b).rank_bits(rank), a[rank] | b[rank]);
            assert_eq!(
                packed_a.symmetric_difference(packed_b).rank_bits(rank),
                a[rank] ^ b[rank]
            );
        }
        assert_eq!(
            packed_a.count_selected(),
            a.iter().map(|x| x.count_ones()).sum()
        );
        assert!(AbtmPooledBlockMask::from_rank_masks([0x40000; 8]).is_none());
    }

    #[test]
    fn abtm_pooled_mask_global_cap_differs_from_uniform_rank_cap() {
        let masks = [
            0x3ff7f, 0x3fbff, 0x3ffdf, 0x3bfff, 0x3fffd, 0x3feff, 0x3fbff, 0x1ffff,
        ];
        let packed = AbtmPooledBlockMask::from_rank_masks(masks).unwrap();
        let mut summed = 0usize;
        let mut exceeds_local_quota = false;
        // All eight ranks have 17 full-size 64-DOF blocks and one
        // 37-DOF end block; whether the end block is selected varies.
        for rank in 0..8 {
            let costs = (0..18)
                .map(|j| if j == 17 { 37 * 37 * 8 } else { 64 * 64 * 8 })
                .collect::<Vec<_>>();
            let mask = replay_reference_mask(packed.rank_bits(rank), &costs, usize::MAX).unwrap();
            let local_bytes: usize = costs
                .iter()
                .zip(mask)
                .filter(|(_, chosen)| *chosen)
                .map(|(cost, _)| cost)
                .sum();
            exceeds_local_quota |= local_bytes > (4_325_376 / 8);
            summed += local_bytes;
        }
        assert_eq!(summed, 4_303_736);
        assert!(summed <= 4_325_376);
        assert!(exceeds_local_quota);
    }

    #[test]
    fn replay_mask_preserves_exact_selection_and_budget() {
        let factors = [32, 32, 8];
        assert_eq!(
            replay_reference_mask(0b101, &factors, 40),
            Some(vec![true, false, true])
        );
        assert_eq!(replay_reference_mask(0b011, &factors, 40), None); // 64 > 40
        assert_eq!(replay_reference_mask(0b1000, &factors, 1024), None); // nonexistent bit
        assert_eq!(replay_reference_mask(0, &factors, 1024), None); // no direct factor
    }

    #[test]
    fn block_swap_only_changes_two_flags_and_preserves_budget() {
        let bytes = [32, 32, 32];
        let mut selected = [true, true, false];
        assert!(swap_selected_block(&mut selected, &bytes, 1, 2, 64));
        assert_eq!(selected, [true, false, true]);
        assert!(!swap_selected_block(&mut selected, &bytes, 1, 0, 64));
        assert_eq!(selected, [true, false, true]);
        // A larger replacement is rejected without any partial mutation.
        let mut short = [true, false];
        assert!(!swap_selected_block(&mut short, &[8, 32], 0, 1, 16));
        assert_eq!(short, [true, false]);
        assert!(!swap_selected_block(&mut short, &[8, 32], 0, 0, 64));
    }

    #[test]
    fn boundary_mode_changes_ranked_block_without_exceeding_budget() {
        use crate::{build_contiguous_halo_plans, prepare_rank_local_csr, ContiguousPartition};
        use hybit_matrix::Csr32Matrix;
        // Three two-DOF blocks, all with equal residual energy; the last
        // block has the least exterior coupling. This A is SPD by strict
        // diagonal dominance, so positive-definite factorization is valid.
        let matrix = Csr32Matrix::new(
            6,
            6,
            vec![0, 2, 5, 8, 11, 14, 16],
            vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5],
            vec![
                8.0, -1.0, -1.0, 8.0, -3.0, -3.0, 8.0, -1.0, -1.0, 8.0, -0.25, -0.25, 8.0, -1.0,
                -1.0, 8.0,
            ],
        )
        .unwrap();
        let partition = ContiguousPartition::balanced(6, 1).unwrap();
        let plans = build_contiguous_halo_plans(&matrix, &partition).unwrap();
        let local = prepare_rank_local_csr(&matrix, &plans[0]).unwrap();
        let residual = [1.0; 6];
        let (original, trace) = select_budgeted_blocks(
            &local,
            &residual,
            2,
            32,
            DistributedAdaptiveBlockScore::ResidualEnergy,
        );
        assert_eq!(original, [true, false, false]);
        assert_eq!(trace.iter().filter(|x| x.selected).count(), 1);
        assert_eq!(trace.iter().map(|x| x.factor_bytes).sum::<usize>(), 96);
        let (coupled, coupled_trace) = select_budgeted_blocks(
            &local,
            &residual,
            2,
            32,
            DistributedAdaptiveBlockScore::BoundaryCoupling,
        );
        assert_eq!(coupled, [false, false, true]);
        assert!(coupled_trace[2].score > coupled_trace[0].score);
        assert_eq!(
            coupled_trace
                .iter()
                .filter(|x| x.selected)
                .map(|x| x.factor_bytes)
                .sum::<usize>(),
            32
        );
    }

    #[test]
    fn residual_block_selector_never_overspends_and_uses_stable_ties() {
        let residual = [4.0, 4.0, 1.0, 1.0, 4.0, 4.0];
        assert_eq!(
            select_residual_blocks(&residual, 2, 0),
            [false, false, false]
        );
        assert_eq!(
            select_residual_blocks(&residual, 2, 32),
            [true, false, false]
        );
        assert_eq!(
            select_residual_blocks(&residual, 2, 64),
            [true, false, true]
        );
        assert_eq!(select_residual_blocks(&residual, 2, 96), [true, true, true]);
    }

    #[test]
    fn adaptive_window_detects_late_stagnation_after_good_initial_progress() {
        let mut checkpoint = 1.0_f64;
        assert!(adaptive_window_ratio(63, 64, 0.075, &mut checkpoint).is_none());
        assert_eq!(checkpoint, 1.0);
        let first = adaptive_window_ratio(64, 64, 0.0749759, &mut checkpoint).unwrap();
        assert!(first < 0.5, "good first window must not promote");
        let later = adaptive_window_ratio(128, 64, 0.062, &mut checkpoint).unwrap();
        assert!(later > 0.5, "stalled later window must promote");
        assert_eq!(checkpoint, 0.062);
        assert!(adaptive_window_ratio(129, 64, 0.061, &mut checkpoint).is_none());
        assert_eq!(checkpoint, 0.062);
        assert!(adaptive_window_ratio(64, 0, 0.05, &mut checkpoint).is_none());
    }

    #[test]
    fn jacobi_diagonal_is_extracted_in_owned_order() {
        let a = Csr32Matrix::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![4.0, -1.0, -1.0, 5.0, -1.0, -1.0, 6.0],
        )
        .unwrap();
        let partition = ContiguousPartition::balanced(3, 1).unwrap();
        let plans = build_contiguous_halo_plans(&a, &partition).unwrap();
        let local = prepare_rank_local_csr(&a, &plans[0]).unwrap();
        assert_eq!(
            inverse_diagonal(&local).unwrap(),
            vec![0.25, 0.2, 1.0 / 6.0]
        );
    }
}
