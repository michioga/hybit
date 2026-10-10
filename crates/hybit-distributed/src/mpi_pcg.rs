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
    pub elapsed_ns: u128,
    /// Sum of durations of all rank-local overlapped SpMV calls (includes MPI halo wait).
    pub spmv_elapsed_ns: u128,
    /// Sum of durations of all global reductions issued inside solve (including validation).
    pub allreduce_elapsed_ns: u128,
    /// Subset of spmv_elapsed_ns spent waiting on MPI nonblocking requests.
    pub halo_wait_ns: u128,
    /// Subset of spmv_elapsed_ns spent computing interior CSR rows.
    pub interior_compute_ns: u128,
    /// Subset of spmv_elapsed_ns spent computing boundary CSR rows.
    pub boundary_compute_ns: u128,
    /// Exclusive local time for M^-1 r applications (no MPI communication).
    pub preconditioner_apply_ns: u128,
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
}

#[derive(Debug)]
enum PreparedPreconditioner {
    Jacobi(Vec<f64>),
    LocalBlockCholesky(LocalBlockCholesky),
    LocalBlockIc0(LocalBlockIc0),
}

impl PreparedPreconditioner {
    fn owned_len(&self) -> usize {
        match self {
            Self::Jacobi(diag) => diag.len(),
            Self::LocalBlockCholesky(blocks) => blocks.owned_len(),
            Self::LocalBlockIc0(blocks) => blocks.owned_len(),
        }
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) {
        match self {
            Self::Jacobi(diag) => {
                for ((zi, &ri), &di) in z.iter_mut().zip(r).zip(diag) {
                    *zi = di * ri;
                }
            }
            Self::LocalBlockCholesky(blocks) => blocks.apply(r, z),
            Self::LocalBlockIc0(blocks) => blocks.apply(r, z),
        }
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
        })
    }

    /// Rank-local factor storage payload only (not allocator/Vec overhead).
    /// Use MPI max/sum reductions in benchmark code to report across ranks.
    pub fn preconditioner_factor_bytes(&self) -> usize {
        match &self.preconditioner {
            PreparedPreconditioner::Jacobi(d) => d.len() * std::mem::size_of::<f64>(),
            PreparedPreconditioner::LocalBlockCholesky(p) => p.factor_bytes(),
            PreparedPreconditioner::LocalBlockIc0(p) => p.factor_bytes(),
        }
    }

    /// f64 factor entries. For dense blocks this counts stored n*n entries,
    /// including the unused upper triangle; IC(0) includes the diagonal.
    pub fn preconditioner_factor_values(&self) -> usize {
        match &self.preconditioner {
            PreparedPreconditioner::Jacobi(d) => d.len(),
            PreparedPreconditioner::LocalBlockCholesky(p) => p.factor_bytes() / 8,
            PreparedPreconditioner::LocalBlockIc0(p) => p.factor_values(),
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
        let start = Instant::now();
        let mut spmv_elapsed_ns = 0u128;
        let mut allreduce_elapsed_ns = 0u128;
        let mut halo_wait_ns = 0u128;
        let mut interior_compute_ns = 0u128;
        let mut boundary_compute_ns = 0u128;
        let mut preconditioner_apply_ns = 0u128;
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
            || x.iter().any(|v| !v.is_finite());
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
        self.preconditioner.apply(&self.r, &mut self.z);
        preconditioner_apply_ns += t_precond.elapsed().as_nanos();
        self.p.copy_from_slice(&self.z);
        let b_sq = timed_sum(mpi, local_dot(b, b), &mut allreduce_elapsed_ns);
        let mut r_sq = timed_sum(mpi, local_dot(&self.r, &self.r), &mut allreduce_elapsed_ns);
        reductions += 2;
        let norm_b = b_sq.sqrt();
        let initial_norm = r_sq.sqrt();
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
                    if r_sq.sqrt() <= threshold {
                        status = DistributedPcgStatus::Converged;
                        break;
                    }
                    let t_precond = Instant::now();
                    self.preconditioner.apply(&self.r, &mut self.z);
                    preconditioner_apply_ns += t_precond.elapsed().as_nanos();
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
            allreduce_calls: reductions,
            elapsed_ns: start.elapsed().as_nanos(),
            spmv_elapsed_ns,
            allreduce_elapsed_ns,
            halo_wait_ns,
            interior_compute_ns,
            boundary_compute_ns,
            preconditioner_apply_ns,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_contiguous_halo_plans, prepare_rank_local_csr, ContiguousPartition};
    use hybit_matrix::Csr32Matrix;

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
