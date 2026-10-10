//! G8-C12a: bounded experimental rank-constant two-level Jacobi.
//!
//! M^{-1} = D^{-1} + Z (Z^T A Z)^{-1} Z^T, with one normalized
//! constant basis vector per MPI rank. If the global A is SPD and the
//! coarse Cholesky succeeds, this fixed additive preconditioner is SPD.
//! Neither a successful coarse factorization nor PCG convergence is a
//! certificate that an experimentally supplied matrix A is SPD.
//!
//! The p x p coarse matrix is replicated on every rank (p <= 8).
//! Each application has one length-p MPI Allreduce. This is a G8-C12a
//! experimental baseline, not a scalable AMG or adaptive coarse space.

use crate::mpi_backend::MpiRuntime;
use crate::{HaloPlan, RankLocalCsr};
use mpi::collective::SystemOperation;
use mpi::traits::*;
use std::io;
use std::time::Instant;

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[derive(Debug)]
pub struct RankConstantTwoLevelJacobi {
    inv_diag: Vec<f64>,
    coarse_l: Vec<f64>,
    rank: usize,
    ranks: usize,
    inv_sqrt_owned: f64,
}

/// Strictly positive Cholesky pivots are required. This factors only the
/// coarse matrix, NOT the original global operator.
#[allow(clippy::needless_range_loop)]
fn cholesky(mut a: Vec<f64>, n: usize) -> io::Result<Vec<f64>> {
    if a.len() != n * n || n == 0 {
        return Err(invalid("invalid coarse matrix shape"));
    }
    let scale = (0..n).map(|i| a[i * n + i].abs()).fold(0.0_f64, f64::max);
    if !scale.is_finite() || scale <= 0.0 {
        return Err(invalid("nonpositive or nonfinite coarse diagonal scale"));
    }
    let threshold = 1.0e-14 * scale;
    for i in 0..n {
        for j in 0..=i {
            let mut value = a[i * n + j];
            for k in 0..j {
                value -= a[i * n + k] * a[j * n + k];
            }
            if i == j {
                if !value.is_finite() || value <= threshold {
                    return Err(invalid("coarse Cholesky has nonpositive/unstable pivot"));
                }
                a[i * n + j] = value.sqrt();
            } else {
                let denominator = a[j * n + j];
                a[i * n + j] = value / denominator;
                if !a[i * n + j].is_finite() {
                    return Err(invalid("nonfinite coarse Cholesky factor"));
                }
            }
        }
        for j in (i + 1)..n {
            a[i * n + j] = 0.0;
        }
    }
    Ok(a)
}

#[allow(clippy::needless_range_loop)]
fn solve_cholesky(l: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut y = b[i];
        for j in 0..i {
            y -= l[i * n + j] * b[j];
        }
        b[i] = y / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut x = b[i];
        for j in (i + 1)..n {
            x -= l[j * n + i] * b[j];
        }
        b[i] = x / l[i * n + i];
    }
}

impl RankConstantTwoLevelJacobi {
    /// Collective, deterministic preparation from rank-owned operator entries.
    /// Ghost column owners are obtained from the same HaloPlan as the SpMV.
    pub fn prepare(mpi: &MpiRuntime, plan: &HaloPlan, a: &RankLocalCsr) -> io::Result<Self> {
        let rank = mpi.rank() as usize;
        let ranks = mpi.size() as usize;
        let mut inv_diag = Vec::with_capacity(a.owned_len());
        let mut local_ok = (1..=8).contains(&ranks)
            && rank < ranks
            && plan.rank() as usize == rank
            && a.rank() as usize == rank
            && plan.owned_len() == a.owned_len()
            && plan.ghost_len() + a.owned_len() == a.extended_len();
        if local_ok {
            for i in 0..a.owned_len() {
                let mut d = 0.0;
                let mut found = false;
                for p in a.row_ptr()[i] as usize..a.row_ptr()[i + 1] as usize {
                    let c = a.col_idx()[p] as usize;
                    if c == i {
                        d += a.values()[p];
                        found = true;
                    }
                    if c >= a.extended_len() || !a.values()[p].is_finite() {
                        local_ok = false;
                    }
                }
                if !found || !d.is_finite() || d <= 0.0 {
                    local_ok = false;
                }
                inv_diag.push(1.0 / d);
            }
        }
        // Synchronize validation before any rank enters subsequent collectives.
        if mpi.all_reduce_sum_u64(u64::from(!local_ok)) != 0 {
            return Err(invalid(
                "rank-constant coarse preparation: invalid owned matrix",
            ));
        }
        let world = mpi::topology::SimpleCommunicator::world();
        let owned_count = a.owned_len() as u64;
        let mut owned_counts = vec![0_u64; ranks];
        world.all_gather_into(&owned_count, &mut owned_counts[..]);
        if owned_counts.contains(&0) {
            return Err(invalid(
                "empty rank is not allowed in rank-constant coarse space",
            ));
        }
        let scales: Vec<f64> = owned_counts
            .iter()
            .map(|&v| 1.0 / (v as f64).sqrt())
            .collect();
        let mut local_e = vec![0.0_f64; ranks * ranks];
        for i in 0..a.owned_len() {
            for p in a.row_ptr()[i] as usize..a.row_ptr()[i + 1] as usize {
                let c = a.col_idx()[p] as usize;
                let owner = if c < a.owned_len() {
                    rank
                } else {
                    plan.ghost_owners()[c - a.owned_len()] as usize
                };
                local_e[rank * ranks + owner] += a.values()[p] * scales[rank] * scales[owner];
            }
        }
        let mut e = vec![0.0_f64; ranks * ranks];
        world.all_reduce_into(&local_e[..], &mut e[..], SystemOperation::sum());
        // Small reduction rounding differences should not introduce asymmetry.
        let mut symmetric = true;
        for i in 0..ranks {
            for j in 0..i {
                let aij = e[i * ranks + j];
                let aji = e[j * ranks + i];
                if !aij.is_finite()
                    || !aji.is_finite()
                    || (aij - aji).abs() > 1e-10 * (1.0 + aij.abs().max(aji.abs()))
                {
                    symmetric = false;
                }
                let mean = 0.5 * (aij + aji);
                e[i * ranks + j] = mean;
                e[j * ranks + i] = mean;
            }
        }
        if !symmetric {
            return Err(invalid("coarse operator is not numerically symmetric"));
        }
        let coarse_l = cholesky(e, ranks)?;
        Ok(Self {
            inv_diag,
            coarse_l,
            rank,
            ranks,
            inv_sqrt_owned: scales[rank],
        })
    }

    pub fn owned_len(&self) -> usize {
        self.inv_diag.len()
    }
    /// Global storage must sum this returned rank-local payload across ranks.
    /// The dense coarse Cholesky is replicated on every MPI process.
    pub fn factor_bytes(&self) -> usize {
        (self.inv_diag.len() + self.coarse_l.len()) * std::mem::size_of::<f64>()
    }
    pub fn factor_values(&self) -> usize {
        self.inv_diag.len() + self.coarse_l.len()
    }
    pub fn coarse_dofs(&self) -> usize {
        self.ranks
    }

    /// Returns MPI Allreduce duration; the caller includes it in both total
    /// preconditioner time and collective communication time.
    pub fn apply(&self, r: &[f64], z: &mut [f64]) -> u128 {
        debug_assert_eq!(r.len(), self.inv_diag.len());
        debug_assert_eq!(z.len(), r.len());
        for ((zi, &ri), &di) in z.iter_mut().zip(r).zip(&self.inv_diag) {
            *zi = ri * di;
        }
        let mut local = [0.0_f64; 8];
        let mut global = [0.0_f64; 8];
        local[self.rank] = self.inv_sqrt_owned * r.iter().sum::<f64>();
        let started = Instant::now();
        mpi::topology::SimpleCommunicator::world().all_reduce_into(
            &local[..self.ranks],
            &mut global[..self.ranks],
            SystemOperation::sum(),
        );
        let allreduce_ns = started.elapsed().as_nanos();
        solve_cholesky(&self.coarse_l, self.ranks, &mut global[..self.ranks]);
        let correction = global[self.rank] * self.inv_sqrt_owned;
        for zi in z.iter_mut() {
            *zi += correction;
        }
        allreduce_ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_spd_coarse_solve() {
        let l = cholesky(vec![4.0, 1.0, 1.0, 3.0], 2).unwrap();
        let mut b = [9.0, 8.0];
        solve_cholesky(&l, 2, &mut b);
        assert!((b[0] - 19.0 / 11.0).abs() < 1.0e-12);
        assert!((b[1] - 23.0 / 11.0).abs() < 1.0e-12);
    }

    #[test]
    fn singular_and_indefinite_coarse_rejected() {
        assert!(cholesky(vec![1.0, 1.0, 1.0, 1.0], 2).is_err());
        assert!(cholesky(vec![1.0, 2.0, 2.0, 1.0], 2).is_err());
    }
}
