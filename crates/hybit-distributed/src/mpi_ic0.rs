//! G8-C8: rank-local, no-fill incomplete Cholesky (IC(0)) for PCG.
//!
//! Each contiguous principal block is factored with the ORIGINAL lower
//! sparsity pattern. No fill-in is inserted. Its positive diagonal L defines
//! an SPD preconditioner M = L L^T even when IC(0) is an approximation.
//! A positive IC(0) factor does NOT prove global SPD; failed pivots are
//! rejected, without shifting or silently switching to another algorithm.
//! Only locally owned CSR rows are inspected; ghost columns are ignored.
//! No MPI calls occur here. The PCG collective constructor propagates errors.

use crate::RankLocalCsr;
use std::collections::BTreeMap;
use std::io;
use std::mem::size_of;

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[derive(Debug)]
struct Ic0Block {
    first: usize,
    size: usize,
    /// Lower triangular off-diagonal L in row-compressed form.
    row_ptr: Vec<usize>,
    cols: Vec<u32>,
    vals: Vec<f64>,
    diag: Vec<f64>,
}

#[derive(Debug)]
pub struct LocalBlockIc0 {
    owned_len: usize,
    blocks: Vec<Ic0Block>,
    factor_bytes: usize,
    factor_values: usize,
}

impl LocalBlockIc0 {
    pub fn prepare(local: &RankLocalCsr, block_size: usize) -> io::Result<Self> {
        if !(2..=512).contains(&block_size) {
            return Err(invalid("IC(0) local block size must be within 2..=512"));
        }
        let owned_len = local.owned_len();
        if owned_len == 0 {
            return Err(invalid("IC(0) requires nonempty locally owned rows"));
        }
        let mut blocks = Vec::new();
        let mut factor_bytes = 0usize;
        let mut factor_values = 0usize;
        for first in (0..owned_len).step_by(block_size) {
            let size = block_size.min(owned_len - first);
            let mut a = vec![BTreeMap::<usize, f64>::new(); size];
            for (i, row) in a.iter_mut().enumerate() {
                let begin = local.row_ptr()[first + i] as usize;
                let end = local.row_ptr()[first + i + 1] as usize;
                for p in begin..end {
                    let column = local.col_idx()[p] as usize;
                    if column >= first && column < first + size {
                        *row.entry(column - first).or_insert(0.0) += local.values()[p];
                    }
                }
                if row.values().any(|value| !value.is_finite()) {
                    return Err(invalid("IC(0) has nonfinite coalesced coefficient"));
                }
            }
            // Check both triangles before discarding the upper triangle.
            for (i, row) in a.iter().enumerate() {
                for (&j, &value) in row {
                    if i == j {
                        continue;
                    }
                    let transposed = a[j].get(&i).copied().unwrap_or(0.0);
                    let tolerance = 1.0e-12 * (1.0 + value.abs().max(transposed.abs()));
                    if !transposed.is_finite() || (value - transposed).abs() > tolerance {
                        return Err(invalid("IC(0) principal block is not symmetric"));
                    }
                }
            }
            let mut diag = vec![0.0f64; size];
            let mut lower_rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(size);
            for i in 0..size {
                let mut current = Vec::<(usize, f64)>::new();
                // IC(0): use only strictly-lower structural nonzeros in A.
                for (&j, &a_ij) in a[i].range(..i) {
                    if a_ij == 0.0 {
                        continue;
                    }
                    let prev_row = &lower_rows[j];
                    // Intersection of already computed L(i,k) and L(j,k), k<j.
                    let (mut left, mut right) = (0usize, 0usize);
                    let mut correction = 0.0f64;
                    while left < current.len() && right < prev_row.len() {
                        let (col_i, value_i) = current[left];
                        let (col_j, value_j) = prev_row[right];
                        if col_i == col_j {
                            correction += value_i * value_j;
                            left += 1;
                            right += 1;
                        } else if col_i < col_j {
                            left += 1;
                        } else {
                            right += 1;
                        }
                    }
                    let lij = (a_ij - correction) / diag[j];
                    if !lij.is_finite() {
                        return Err(invalid("IC(0) computed nonfinite lower factor"));
                    }
                    current.push((j, lij));
                }
                let a_ii = a[i].get(&i).copied().unwrap_or(0.0);
                let update: f64 = current.iter().map(|&(_, val)| val * val).sum();
                let pivot = a_ii - update;
                if !pivot.is_finite() || pivot <= 0.0 {
                    return Err(invalid("IC(0) encountered nonpositive local pivot"));
                }
                diag[i] = pivot.sqrt();
                lower_rows.push(current);
            }
            let lower_nnz = lower_rows.iter().map(Vec::len).sum::<usize>();
            let mut row_ptr = Vec::with_capacity(size + 1);
            let mut cols = Vec::with_capacity(lower_nnz);
            let mut vals = Vec::with_capacity(lower_nnz);
            row_ptr.push(0);
            for row in lower_rows {
                for (col, value) in row {
                    cols.push(
                        u32::try_from(col).map_err(|_| invalid("IC(0) block index overflow"))?,
                    );
                    vals.push(value);
                }
                row_ptr.push(vals.len());
            }
            factor_values += diag.len() + vals.len();
            factor_bytes += row_ptr.len() * size_of::<usize>()
                + cols.len() * size_of::<u32>()
                + vals.len() * size_of::<f64>()
                + diag.len() * size_of::<f64>();
            blocks.push(Ic0Block {
                first,
                size,
                row_ptr,
                cols,
                vals,
                diag,
            });
        }
        Ok(Self {
            owned_len,
            blocks,
            factor_bytes,
            factor_values,
        })
    }

    pub fn owned_len(&self) -> usize {
        self.owned_len
    }
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
    /// Allocated factor array payload, excluding Vec capacities and metadata.
    pub fn factor_bytes(&self) -> usize {
        self.factor_bytes
    }
    /// Diagonal plus stored strictly-lower factor coefficients.
    pub fn factor_values(&self) -> usize {
        self.factor_values
    }

    /// z = M^-1 r through L and L^T; allocation-free and communication-free.
    pub fn apply(&self, r: &[f64], z: &mut [f64]) {
        debug_assert_eq!(r.len(), self.owned_len);
        debug_assert_eq!(z.len(), self.owned_len);
        z.copy_from_slice(r);
        for block in &self.blocks {
            let start = block.first;
            // Solve Ly=r.
            for i in 0..block.size {
                let mut value = z[start + i];
                for p in block.row_ptr[i]..block.row_ptr[i + 1] {
                    value -= block.vals[p] * z[start + block.cols[p] as usize];
                }
                z[start + i] = value / block.diag[i];
            }
            // Solve L^Tz=y: scatter from row i into earlier columns.
            for i in (0..block.size).rev() {
                let value = z[start + i] / block.diag[i];
                z[start + i] = value;
                for p in block.row_ptr[i]..block.row_ptr[i + 1] {
                    z[start + block.cols[p] as usize] -= block.vals[p] * value;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpi_block_jacobi::LocalBlockCholesky;
    use crate::{build_contiguous_halo_plans, prepare_rank_local_csr, ContiguousPartition};
    use hybit_matrix::Csr32Matrix;

    fn local_chain(n: usize, diagonal: f64) -> RankLocalCsr {
        let mut ptr = vec![0u32];
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            if i > 0 {
                cols.push((i - 1) as u32);
                vals.push(-1.0);
            }
            cols.push(i as u32);
            vals.push(diagonal);
            if i + 1 < n {
                cols.push((i + 1) as u32);
                vals.push(-1.0);
            }
            ptr.push(vals.len() as u32);
        }
        let matrix = Csr32Matrix::new(n, n, ptr, cols, vals).unwrap();
        let partition = ContiguousPartition::balanced(n as u64, 1).unwrap();
        let plans = build_contiguous_halo_plans(&matrix, &partition).unwrap();
        prepare_rank_local_csr(&matrix, &plans[0]).unwrap()
    }

    #[test]
    fn ic0_matches_dense_block_for_tridiagonal() {
        let local = local_chain(9, 4.0);
        let ic0 = LocalBlockIc0::prepare(&local, 8).unwrap();
        let dense = LocalBlockCholesky::prepare(&local, 8).unwrap();
        let r: Vec<f64> = (1..=9).map(f64::from).collect();
        let mut sparse_x = vec![0.0; 9];
        let mut dense_x = vec![0.0; 9];
        ic0.apply(&r, &mut sparse_x);
        dense.apply(&r, &mut dense_x);
        for (a, b) in sparse_x.iter().zip(&dense_x) {
            assert!((a - b).abs() < 1e-12);
        }
        assert_eq!(ic0.block_count(), 2);
        assert_eq!(ic0.factor_values(), 9 + 7);
    }

    #[test]
    fn ic0_sparse_storage_beats_dense_on_chain() {
        let local = local_chain(256, 4.0);
        let sparse = LocalBlockIc0::prepare(&local, 128).unwrap();
        let dense = LocalBlockCholesky::prepare(&local, 128).unwrap();
        assert_eq!(sparse.factor_values(), 256 + 254);
        assert!(sparse.factor_bytes() < dense.factor_bytes() / 8);
    }

    #[test]
    fn ic0_rejects_nonpositive_pivot_and_invalid_size() {
        let local = local_chain(4, 1.0);
        assert!(LocalBlockIc0::prepare(&local, 1).is_err());
        assert!(LocalBlockIc0::prepare(&local, 4).is_err());
    }
}
