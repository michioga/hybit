use std::cell::RefCell;
use std::collections::HashMap;

use hybit_core::{HybitError, Preconditioner};
use hybit_matrix::Csr32Matrix;

#[derive(Clone, Debug)]
pub struct IdentityPreconditioner {
    n: usize,
}

impl IdentityPreconditioner {
    pub fn new(n: usize) -> Self { Self { n } }
}

impl Preconditioner for IdentityPreconditioner {
    fn len(&self) -> usize { self.n }
    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        if r.len() != self.n { return Err(HybitError::DimensionMismatch { expected: self.n, actual: r.len() }); }
        if z.len() != self.n { return Err(HybitError::DimensionMismatch { expected: self.n, actual: z.len() }); }
        z.copy_from_slice(r);
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct JacobiPreconditioner {
    inv_diag: Vec<f64>,
}

impl JacobiPreconditioner {
    pub fn from_csr32(matrix: &Csr32Matrix) -> Result<Self, HybitError> {
        let diagonal = matrix.diagonal()?;
        let mut inv_diag = Vec::with_capacity(diagonal.len());
        for (row, d) in diagonal.into_iter().enumerate() {
            if d == 0.0 || !d.is_finite() {
                return Err(HybitError::ZeroDiagonal { row });
            }
            if d < 0.0 {
                return Err(HybitError::InvalidMatrix("Jacobi-PCG requires a positive diagonal"));
            }
            inv_diag.push(1.0 / d);
        }
        Ok(Self { inv_diag })
    }

    pub fn inv_diagonal(&self) -> &[f64] { &self.inv_diag }
}

impl Preconditioner for JacobiPreconditioner {
    fn len(&self) -> usize { self.inv_diag.len() }

    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        if r.len() != self.inv_diag.len() {
            return Err(HybitError::DimensionMismatch { expected: self.inv_diag.len(), actual: r.len() });
        }
        if z.len() != self.inv_diag.len() {
            return Err(HybitError::DimensionMismatch { expected: self.inv_diag.len(), actual: z.len() });
        }
        for ((out, &ri), &d) in z.iter_mut().zip(r).zip(&self.inv_diag) {
            *out = d * ri;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct LocalCholeskyRegion {
    indices: Vec<usize>,
    lower: Vec<f64>,
}

impl LocalCholeskyRegion {
    pub fn from_csr32(matrix: &Csr32Matrix, indices: &[usize]) -> Result<Self, HybitError> {
        if indices.is_empty() {
            return Err(HybitError::InvalidArgument("local Cholesky region may not be empty"));
        }
        let mut canonical = indices.to_vec();
        canonical.sort_unstable();
        canonical.dedup();
        if canonical.len() != indices.len() {
            return Err(HybitError::InvalidArgument("local Cholesky region contains duplicate DOFs"));
        }
        if canonical.iter().any(|&i| i >= matrix.nrows()) {
            return Err(HybitError::InvalidArgument("local Cholesky DOF is out of range"));
        }
        if matrix.nrows() != matrix.ncols() {
            return Err(HybitError::InvalidMatrix("local Cholesky requires a square matrix"));
        }

        let n = canonical.len();
        let local_of: HashMap<usize, usize> = canonical.iter().copied().enumerate().map(|(i, g)| (g, i)).collect();
        let mut dense = vec![0.0; n * n];
        for (li, &global_row) in canonical.iter().enumerate() {
            let start = matrix.row_ptr()[global_row] as usize;
            let end = matrix.row_ptr()[global_row + 1] as usize;
            for p in start..end {
                let global_col = matrix.col_idx()[p] as usize;
                if let Some(&lj) = local_of.get(&global_col) {
                    dense[li * n + lj] += matrix.values()[p];
                }
            }
        }

        // Local direct correction is currently an SPD path. Require the local
        // principal matrix to be numerically symmetric before factorization.
        let mut scale = 0.0f64;
        for &v in &dense { scale = scale.max(v.abs()); }
        let symmetry_tol = 1.0e-11 * scale.max(1.0);
        for i in 0..n {
            for j in 0..i {
                if (dense[i * n + j] - dense[j * n + i]).abs() > symmetry_tol {
                    return Err(HybitError::InvalidMatrix("local Cholesky region is not symmetric"));
                }
            }
        }

        let mut lower = vec![0.0; n * n];
        let pivot_tol = 1.0e-14 * scale.max(1.0);
        for i in 0..n {
            for j in 0..=i {
                let mut sum = dense[i * n + j];
                for k in 0..j {
                    sum -= lower[i * n + k] * lower[j * n + k];
                }
                if i == j {
                    if !sum.is_finite() || sum <= pivot_tol {
                        return Err(HybitError::NumericalBreakdown("local Cholesky encountered a non-positive pivot"));
                    }
                    lower[i * n + i] = sum.sqrt();
                } else {
                    lower[i * n + j] = sum / lower[j * n + j];
                }
            }
        }

        Ok(Self { indices: canonical, lower })
    }

    pub fn len(&self) -> usize { self.indices.len() }
    pub fn is_empty(&self) -> bool { self.indices.is_empty() }
    pub fn indices(&self) -> &[usize] { &self.indices }

    fn solve_local(&self, rhs: &[f64], out: &mut [f64]) {
        let n = self.indices.len();
        debug_assert_eq!(rhs.len(), n);
        debug_assert_eq!(out.len(), n);
        for i in 0..n {
            let mut sum = rhs[i];
            for k in 0..i {
                sum -= self.lower[i * n + k] * out[k];
            }
            out[i] = sum / self.lower[i * n + i];
        }
        for i in (0..n).rev() {
            let mut sum = out[i];
            for k in (i + 1)..n {
                sum -= self.lower[k * n + i] * out[k];
            }
            out[i] = sum / self.lower[i * n + i];
        }
    }

    pub fn factor_bytes(&self) -> usize {
        self.indices.len() * std::mem::size_of::<usize>()
            + self.lower.len() * std::mem::size_of::<f64>()
    }
}

#[derive(Clone, Debug)]
struct RegionScratch {
    rhs: Vec<f64>,
    sol: Vec<f64>,
}

#[derive(Clone, Debug)]
struct WeightedLocalRegion {
    factor: LocalCholeskyRegion,
    weights: Vec<f64>,
    scratch: RefCell<RegionScratch>,
}

#[derive(Clone, Debug)]
pub struct HybridPreconditioner {
    jacobi: JacobiPreconditioner,
    regions: Vec<WeightedLocalRegion>,
    multiplicity: Vec<u16>,
    largest_region: usize,
    unique_local_dofs: usize,
    factor_bytes: usize,
}

impl HybridPreconditioner {
    /// Build a symmetric weighted overlapping Schwarz preconditioner.
    ///
    /// For a DOF contained in m local regions each local restriction uses
    /// w_i = 1/sqrt(m).  The local term is therefore
    /// R^T W A_H^{-1} W R, which is symmetric positive semidefinite when
    /// A_H is SPD. Jacobi is retained only on DOFs not covered by a local
    /// factor, preventing double counting in the single-region case.
    pub fn from_csr32(matrix: &Csr32Matrix, regions: Vec<Vec<usize>>) -> Result<Self, HybitError> {
        let jacobi = JacobiPreconditioner::from_csr32(matrix)?;
        let mut canonical_regions = Vec::with_capacity(regions.len());
        let mut multiplicity = vec![0u16; matrix.nrows()];
        for mut region in regions {
            if region.is_empty() { continue; }
            region.sort_unstable();
            region.dedup();
            for &dof in &region {
                if dof >= matrix.nrows() {
                    return Err(HybitError::InvalidArgument("hybrid preconditioner DOF is out of range"));
                }
                multiplicity[dof] = multiplicity[dof]
                    .checked_add(1)
                    .ok_or(HybitError::SizeOverflow)?;
            }
            canonical_regions.push(region);
        }
        if canonical_regions.is_empty() {
            return Err(HybitError::InvalidArgument("hybrid preconditioner requires at least one local region"));
        }

        let unique_local_dofs = multiplicity.iter().filter(|&&m| m > 0).count();
        let mut factors = Vec::with_capacity(canonical_regions.len());
        let mut largest_region = 0usize;
        let mut factor_bytes = multiplicity.len() * std::mem::size_of::<u16>();
        for region in canonical_regions {
            let factor = LocalCholeskyRegion::from_csr32(matrix, &region)?;
            let weights: Vec<f64> = factor
                .indices()
                .iter()
                .map(|&dof| 1.0 / (multiplicity[dof] as f64).sqrt())
                .collect();
            largest_region = largest_region.max(factor.len());
            let n = factor.len();
            let scratch = RegionScratch { rhs: vec![0.0; n], sol: vec![0.0; n] };
            factor_bytes += factor.factor_bytes()
                + weights.len() * std::mem::size_of::<f64>()
                + 2 * n * std::mem::size_of::<f64>();
            factors.push(WeightedLocalRegion { factor, weights, scratch: RefCell::new(scratch) });
        }

        Ok(Self { jacobi, regions: factors, multiplicity, largest_region, unique_local_dofs, factor_bytes })
    }

    pub fn region_count(&self) -> usize { self.regions.len() }
    pub fn largest_region(&self) -> usize { self.largest_region }
    pub fn local_dofs(&self) -> usize { self.regions.iter().map(|r| r.factor.len()).sum() }
    pub fn unique_local_dofs(&self) -> usize { self.unique_local_dofs }
    pub fn factor_bytes(&self) -> usize { self.factor_bytes }
    pub fn regions(&self) -> impl Iterator<Item = &LocalCholeskyRegion> {
        self.regions.iter().map(|r| &r.factor)
    }
}

impl Preconditioner for HybridPreconditioner {
    fn len(&self) -> usize { self.jacobi.len() }

    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        if r.len() != self.len() {
            return Err(HybitError::DimensionMismatch { expected: self.len(), actual: r.len() });
        }
        if z.len() != self.len() {
            return Err(HybitError::DimensionMismatch { expected: self.len(), actual: z.len() });
        }

        // Base Jacobi acts only outside all selected local factors.
        for i in 0..z.len() {
            z[i] = if self.multiplicity[i] == 0 {
                self.jacobi.inv_diagonal()[i] * r[i]
            } else {
                0.0
            };
        }

        // Symmetrically weighted overlapping local corrections. Scratch storage
        // is allocated once when the preconditioner is built; apply() performs
        // no heap allocation in the Krylov iteration loop.
        for region in &self.regions {
            let mut scratch = region.scratch.borrow_mut();
            let RegionScratch { rhs, sol } = &mut *scratch;
            for (i, (&gi, &w)) in region.factor.indices().iter().zip(&region.weights).enumerate() {
                rhs[i] = w * r[gi];
            }
            region.factor.solve_local(rhs, sol);
            for (i, (&gi, &w)) in region.factor.indices().iter().zip(&region.weights).enumerate() {
                z[gi] += w * sol[i];
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn poisson_1d(n: usize) -> Csr32Matrix {
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        for i in 0..n {
            if i > 0 { col_idx.push((i - 1) as u32); values.push(-1.0); }
            col_idx.push(i as u32); values.push(2.0);
            if i + 1 < n { col_idx.push((i + 1) as u32); values.push(-1.0); }
            row_ptr.push(col_idx.len() as u32);
        }
        Csr32Matrix::new(n, n, row_ptr, col_idx, values).unwrap()
    }

    #[test]
    fn local_cholesky_solves_poisson_block() {
        let a = poisson_1d(4);
        let region = LocalCholeskyRegion::from_csr32(&a, &[0, 1, 2, 3]).unwrap();
        let r = vec![1.0, 0.0, 0.0, 1.0];
        let mut z = vec![0.0; 4];
        region.solve_local(&r, &mut z);
        let y = a.spmv(&z).unwrap();
        for (yi, ri) in y.iter().zip(&r) { assert!((yi - ri).abs() < 1.0e-12); }
    }

    #[test]
    fn single_region_matches_exact_local_solve() {
        let a = poisson_1d(4);
        let hybrid = HybridPreconditioner::from_csr32(&a, vec![vec![0, 1, 2, 3]]).unwrap();
        let r = vec![1.0, 0.0, 0.0, 1.0];
        let mut z = vec![0.0; 4];
        hybrid.apply(&r, &mut z).unwrap();
        let y = a.spmv(&z).unwrap();
        for (yi, ri) in y.iter().zip(&r) { assert!((yi - ri).abs() < 1.0e-12); }
    }

    #[test]
    fn overlapping_weighted_schwarz_is_positive() {
        let a = poisson_1d(8);
        let hybrid = HybridPreconditioner::from_csr32(
            &a,
            vec![vec![0, 1, 2, 3, 4], vec![3, 4, 5, 6, 7]],
        ).unwrap();
        let r = vec![1.0, -0.5, 0.25, 2.0, -1.0, 0.75, 1.5, -0.25];
        let mut z = vec![0.0; 8];
        hybrid.apply(&r, &mut z).unwrap();
        let rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        assert!(rz > 0.0);
        assert_eq!(hybrid.region_count(), 2);
        assert_eq!(hybrid.unique_local_dofs(), 8);
        assert!(hybrid.factor_bytes() > 0);
    }
}
