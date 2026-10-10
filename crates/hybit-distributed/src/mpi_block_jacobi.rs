//! G8-C6: fixed, SPD-preserving rank-local block-Jacobi preconditioner.
//!
//! Each principal diagonal block is Cholesky-factored once.  Ghost columns
//! are intentionally ignored by the block-diagonal approximation M.  If A is
//! SPD, each extracted principal block is SPD and M is SPD.  A successful
//! factorization alone does NOT establish that the global A is SPD.
//!
//! No MPI calls occur in this module; callers must propagate preparation
//! errors collectively before entering any later MPI collective operations.

use crate::RankLocalCsr;
use std::io;

fn invalid(what: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, what)
}

#[derive(Debug)]
struct CholeskyBlock {
    first: usize,
    size: usize,
    /// Row-major dense storage: lower triangle of L; upper triangle unused.
    lower: Vec<f64>,
}

#[derive(Debug)]
pub struct LocalBlockCholesky {
    owned_len: usize,
    blocks: Vec<CholeskyBlock>,
    factor_bytes: usize,
}

impl LocalBlockCholesky {
    pub fn prepare(local: &RankLocalCsr, block_size: usize) -> io::Result<Self> {
        if !(2..=512).contains(&block_size) {
            return Err(invalid("local Cholesky block size must be within 2..=512"));
        }
        let owned_len = local.owned_len();
        if owned_len == 0 {
            return Err(invalid(
                "rank-local block Cholesky requires nonempty owned rows",
            ));
        }
        let mut blocks = Vec::new();
        let mut factor_bytes = 0usize;
        for first in (0..owned_len).step_by(block_size) {
            let size = block_size.min(owned_len - first);
            let mut lower = vec![0.0; size * size];
            for local_row in 0..size {
                let row = first + local_row;
                let begin = local.row_ptr()[row] as usize;
                let end = local.row_ptr()[row + 1] as usize;
                for p in begin..end {
                    let col = local.col_idx()[p] as usize;
                    if col >= first && col < first + size {
                        lower[local_row * size + (col - first)] += local.values()[p];
                    }
                }
            }
            // Reject local nonsymmetry explicitly rather than silently
            // factoring the lower triangle of a nonsymmetric block.
            for i in 0..size {
                for j in 0..i {
                    let a = lower[i * size + j];
                    let b = lower[j * size + i];
                    let tolerance = 1.0e-12 * (1.0 + a.abs().max(b.abs()));
                    if !a.is_finite() || !b.is_finite() || (a - b).abs() > tolerance {
                        return Err(invalid("local Cholesky principal block is not symmetric"));
                    }
                }
            }
            // In-place unpivoted lower Cholesky, A = L L^T.
            for i in 0..size {
                for j in 0..=i {
                    let mut value = lower[i * size + j];
                    for k in 0..j {
                        value -= lower[i * size + k] * lower[j * size + k];
                    }
                    if i == j {
                        if !value.is_finite() || value <= 0.0 {
                            return Err(invalid("non-positive Cholesky pivot in rank-local block"));
                        }
                        lower[i * size + i] = value.sqrt();
                    } else {
                        lower[i * size + j] = value / lower[j * size + j];
                        if !lower[i * size + j].is_finite() {
                            return Err(invalid("non-finite Cholesky factor coefficient"));
                        }
                    }
                }
            }
            factor_bytes += lower.len() * std::mem::size_of::<f64>();
            blocks.push(CholeskyBlock { first, size, lower });
        }
        Ok(Self {
            owned_len,
            blocks,
            factor_bytes,
        })
    }

    pub fn owned_len(&self) -> usize {
        self.owned_len
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn factor_bytes(&self) -> usize {
        self.factor_bytes
    }

    /// Compute z = M^-1 r with no allocation or MPI communication.
    /// Input and output must have rank-owned lengths (checked by PCG).
    pub fn apply(&self, r: &[f64], z: &mut [f64]) {
        debug_assert_eq!(r.len(), self.owned_len);
        debug_assert_eq!(z.len(), self.owned_len);
        z.copy_from_slice(r);
        for block in &self.blocks {
            let first = block.first;
            let n = block.size;
            let l = &block.lower;
            // Forward solve: L y = r.
            for i in 0..n {
                let mut v = z[first + i];
                for j in 0..i {
                    v -= l[i * n + j] * z[first + j];
                }
                z[first + i] = v / l[i * n + i];
            }
            // Backward solve: L^T z = y.
            for i in (0..n).rev() {
                let mut v = z[first + i];
                for j in i + 1..n {
                    v -= l[j * n + i] * z[first + j];
                }
                z[first + i] = v / l[i * n + i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_contiguous_halo_plans, prepare_rank_local_csr, ContiguousPartition};
    use hybit_matrix::Csr32Matrix;

    fn local_matrix(diag: [f64; 3]) -> RankLocalCsr {
        let a = Csr32Matrix::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![diag[0], -1.0, -1.0, diag[1], -1.0, -1.0, diag[2]],
        )
        .unwrap();
        let partition = ContiguousPartition::balanced(3, 1).unwrap();
        let plans = build_contiguous_halo_plans(&a, &partition).unwrap();
        prepare_rank_local_csr(&a, &plans[0]).unwrap()
    }

    #[test]
    fn local_cholesky_solves_each_principal_block() {
        let local = local_matrix([4.0, 5.0, 6.0]);
        let precond = LocalBlockCholesky::prepare(&local, 2).unwrap();
        assert_eq!(precond.block_count(), 2);
        assert_eq!(precond.factor_bytes(), (4 + 1) * 8);
        let r = [2.0, 1.0, 3.0];
        let mut z = [0.0; 3];
        precond.apply(&r, &mut z);
        assert!((4.0 * z[0] - z[1] - r[0]).abs() < 1e-12);
        assert!((-z[0] + 5.0 * z[1] - r[1]).abs() < 1e-12);
        assert!((6.0 * z[2] - r[2]).abs() < 1e-12);
    }

    #[test]
    fn invalid_size_and_nonpositive_local_pivot_rejected() {
        let local = local_matrix([1.0, 1.0, 1.0]);
        assert!(LocalBlockCholesky::prepare(&local, 1).is_err());
        // Principal [1,-1;-1,1] is singular, so factorization must refuse it.
        assert!(LocalBlockCholesky::prepare(&local, 2).is_err());
    }
}
