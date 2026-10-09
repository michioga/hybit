//! G8-C1: MPI rank-local Jacobi PCG for real symmetric positive-definite CSR.
//!
//! The numerical iteration holds only rank-owned vectors. Halo values are
//! acquired by the G8-B3 overlapped SpMV. Global scalar products use MPI
//! Allreduce. `prepare` and `solve` are collective MPI operations; all ranks
//! must use the same options, operator and call sequence.
//!
//! This is the first correct distributed Krylov baseline, not an automatic
//! SPD detector, multi-host benchmark or general nonsymmetric solver.

use crate::mpi_backend::MpiRuntime;
use crate::mpi_overlap::OverlapSpmv;
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
}

/// Reusable rank-local Krylov vectors, Jacobi diagonal and Halo/SpMV buffers.
///
/// Each rank prepares once, then calls solve in the same order on every rank.
/// `local` must remain the exact same matrix as at preparation.
#[derive(Debug)]
pub struct DistributedPcg {
    overlap: OverlapSpmv,
    diag_inverse: Vec<f64>,
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

impl DistributedPcg {
    /// Collective prepare. Invalid local diagonal is propagated to ALL ranks
    /// before entering OverlapSpmv's collective setup.
    pub fn prepare(mpi: &MpiRuntime, plan: &HaloPlan, local: &RankLocalCsr) -> io::Result<Self> {
        let diag = inverse_diagonal(local);
        let local_bad = u64::from(diag.is_err() || plan.rank() != mpi.rank() as u32);
        if mpi.all_reduce_sum_u64(local_bad) != 0 {
            return Err(invalid(
                "at least one MPI rank has an invalid local Jacobi diagonal or rank",
            ));
        }
        let diag_inverse = diag?;
        let overlap = OverlapSpmv::prepare(mpi, plan, local)?;
        let n = local.owned_len();
        Ok(Self {
            overlap,
            diag_inverse,
            ax: vec![0.0; n],
            r: vec![0.0; n],
            z: vec![0.0; n],
            p: vec![0.0; n],
            ap: vec![0.0; n],
        })
    }

    pub fn interior_rows(&self) -> usize {
        self.overlap.interior_len()
    }

    pub fn boundary_rows(&self) -> usize {
        self.overlap.boundary_len()
    }

    /// Solve `Ax=b` using a fixed Jacobi-preconditioned distributed PCG.
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
        let n = self.diag_inverse.len();
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
        if mpi.all_reduce_sum_u64(u64::from(local_bad)) != 0 {
            return Err(invalid(
                "invalid distributed PCG vector dimensions, options or values",
            ));
        }
        let mut reductions = 1usize;
        let mut spmv_calls = 0usize;
        self.overlap.spmv_overlap(mpi, local, x, &mut self.ax)?;
        spmv_calls += 1;
        for (i, &bi) in b.iter().enumerate() {
            self.r[i] = bi - self.ax[i];
            self.z[i] = self.diag_inverse[i] * self.r[i];
            self.p[i] = self.z[i];
        }
        let b_sq = mpi.all_reduce_sum_f64(local_dot(b, b));
        let mut r_sq = mpi.all_reduce_sum_f64(local_dot(&self.r, &self.r));
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
            let mut rz = mpi.all_reduce_sum_f64(local_dot(&self.r, &self.z));
            reductions += 1;
            if !rz.is_finite() || rz <= 0.0 {
                status = DistributedPcgStatus::Breakdown;
            } else {
                for step in 1..=options.max_iterations {
                    self.overlap
                        .spmv_overlap(mpi, local, &self.p, &mut self.ap)?;
                    spmv_calls += 1;
                    let p_ap = mpi.all_reduce_sum_f64(local_dot(&self.p, &self.ap));
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
                    r_sq = mpi.all_reduce_sum_f64(local_dot(&self.r, &self.r));
                    reductions += 1;
                    if !r_sq.is_finite() || r_sq < 0.0 {
                        status = DistributedPcgStatus::Breakdown;
                        break;
                    }
                    if r_sq.sqrt() <= threshold {
                        status = DistributedPcgStatus::Converged;
                        break;
                    }
                    for (i, zi) in self.z.iter_mut().enumerate() {
                        *zi = self.diag_inverse[i] * self.r[i];
                    }
                    let next_rz = mpi.all_reduce_sum_f64(local_dot(&self.r, &self.z));
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
        self.overlap.spmv_overlap(mpi, local, x, &mut self.ax)?;
        spmv_calls += 1;
        let true_local_sq: f64 = b
            .iter()
            .zip(&self.ax)
            .map(|(&bi, &axi)| (bi - axi) * (bi - axi))
            .sum();
        let true_norm = mpi.all_reduce_sum_f64(true_local_sq).sqrt();
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
