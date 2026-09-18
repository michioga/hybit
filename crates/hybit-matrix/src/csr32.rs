use hybit_core::{HybitError, LinearOperator};

#[derive(Clone, Debug)]
pub struct Csr32Matrix {
    nrows: usize,
    ncols: usize,
    row_ptr: Vec<u32>,
    col_idx: Vec<u32>,
    values: Vec<f64>,
}

impl Csr32Matrix {
    pub fn new(
        nrows: usize,
        ncols: usize,
        row_ptr: Vec<u32>,
        col_idx: Vec<u32>,
        values: Vec<f64>,
    ) -> Result<Self, HybitError> {
        let matrix = Self { nrows, ncols, row_ptr, col_idx, values };
        matrix.validate()?;
        Ok(matrix)
    }

    pub fn validate(&self) -> Result<(), HybitError> {
        if self.row_ptr.len() != self.nrows + 1 {
            return Err(HybitError::InvalidMatrix("row_ptr length must equal nrows + 1"));
        }
        if self.col_idx.len() != self.values.len() {
            return Err(HybitError::InvalidMatrix("col_idx and values lengths differ"));
        }
        if self.row_ptr.first().copied().unwrap_or(1) != 0 {
            return Err(HybitError::InvalidMatrix("row_ptr[0] must be zero"));
        }
        if self.values.len() > u32::MAX as usize {
            return Err(HybitError::SizeOverflow);
        }
        let nnz = self.values.len() as u32;
        if self.row_ptr.last().copied().unwrap_or(0) != nnz {
            return Err(HybitError::InvalidMatrix("row_ptr[nrows] must equal nnz"));
        }
        for pair in self.row_ptr.windows(2) {
            if pair[0] > pair[1] {
                return Err(HybitError::InvalidMatrix("row_ptr must be monotonically nondecreasing"));
            }
        }
        if self.col_idx.iter().any(|&c| c as usize >= self.ncols) {
            return Err(HybitError::InvalidMatrix("column index out of range"));
        }
        if self.values.iter().any(|v| !v.is_finite()) {
            return Err(HybitError::InvalidMatrix("matrix contains NaN or infinity"));
        }
        Ok(())
    }

    pub fn nrows(&self) -> usize { self.nrows }
    pub fn ncols(&self) -> usize { self.ncols }
    pub fn nnz(&self) -> usize { self.values.len() }
    pub fn row_ptr(&self) -> &[u32] { &self.row_ptr }
    pub fn col_idx(&self) -> &[u32] { &self.col_idx }
    pub fn values(&self) -> &[f64] { &self.values }

    pub fn storage_bytes(&self) -> usize {
        self.row_ptr.len() * std::mem::size_of::<u32>()
            + self.col_idx.len() * std::mem::size_of::<u32>()
            + self.values.len() * std::mem::size_of::<f64>()
    }

    pub fn metadata_bytes(&self) -> usize {
        self.row_ptr.len() * std::mem::size_of::<u32>()
            + self.col_idx.len() * std::mem::size_of::<u32>()
    }

    pub fn diagonal(&self) -> Result<Vec<f64>, HybitError> {
        if self.nrows != self.ncols {
            return Err(HybitError::InvalidMatrix("diagonal requires a square matrix"));
        }
        let mut diagonal = vec![0.0; self.nrows];
        let mut found = vec![false; self.nrows];
        for row in 0..self.nrows {
            let start = self.row_ptr[row] as usize;
            let end = self.row_ptr[row + 1] as usize;
            for p in start..end {
                if self.col_idx[p] as usize == row {
                    diagonal[row] += self.values[p];
                    found[row] = true;
                }
            }
        }
        for row in 0..self.nrows {
            if !found[row] {
                return Err(HybitError::MissingDiagonal { row });
            }
        }
        Ok(diagonal)
    }

    pub fn spmv(&self, x: &[f64]) -> Result<Vec<f64>, HybitError> {
        let mut y = vec![0.0; self.nrows];
        self.apply(x, &mut y)?;
        Ok(y)
    }
}

impl LinearOperator for Csr32Matrix {
    fn rows(&self) -> usize { self.nrows }
    fn cols(&self) -> usize { self.ncols }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), HybitError> {
        if x.len() != self.ncols {
            return Err(HybitError::DimensionMismatch { expected: self.ncols, actual: x.len() });
        }
        if y.len() != self.nrows {
            return Err(HybitError::DimensionMismatch { expected: self.nrows, actual: y.len() });
        }
        for (row, out) in y.iter_mut().enumerate() {
            let start = self.row_ptr[row] as usize;
            let end = self.row_ptr[row + 1] as usize;
            let mut sum = 0.0;
            for p in start..end {
                sum += self.values[p] * x[self.col_idx[p] as usize];
            }
            *out = sum;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_spmv() {
        let a = Csr32Matrix::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
        ).unwrap();
        let y = a.spmv(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(y, vec![0.0, 0.0, 4.0]);
    }
}
