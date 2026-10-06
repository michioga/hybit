use hybit_core::{HybitError, LinearOperator};

use crate::Csr32Matrix;

/// Fixed dense block width for the explicit block-CSR operator.
///
/// These widths are intentionally limited to the G5-validated FEM-oriented
/// cases. Automatic routing is not performed by this type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseBlockSize {
    B3,
    B6,
}

impl DenseBlockSize {
    #[inline]
    pub const fn width(self) -> usize {
        match self {
            Self::B3 => 3,
            Self::B6 => 6,
        }
    }
}

/// Explicit opt-in dense block-CSR operator.
///
/// `DenseBlockCsrOperator` converts a scalar `Csr32Matrix` into fixed-size
/// 3x3 or 6x6 dense numerical blocks. Duplicate scalar entries are accumulated
/// in the same block slot so the operator preserves ordinary CSR SpMV
/// semantics. Interior blocks use specialized fixed-size kernels; only the
/// final partial block row/column uses bounded tail handling.
///
/// This operator does not automatically replace `Csr32Matrix`. G5 selector
/// validation supports structural routing as a future policy, but automatic
/// routing remains deliberately separate from this explicit production path.
#[derive(Clone, Debug)]
pub struct DenseBlockCsrOperator {
    nrows: usize,
    ncols: usize,
    block_size: DenseBlockSize,
    row_ptr: Vec<u32>,
    col_idx: Vec<u32>,
    values: Vec<f64>,
    structural_slots: usize,
    valid_dense_slots: usize,
}

impl DenseBlockCsrOperator {
    pub fn from_csr32(
        matrix: &Csr32Matrix,
        block_size: DenseBlockSize,
    ) -> Result<Self, HybitError> {
        matrix.validate()?;

        let b = block_size.width();
        let block_rows = matrix.nrows().div_ceil(b);

        let mut row_ptr = Vec::with_capacity(block_rows.saturating_add(1));
        let mut col_idx = Vec::<u32>::new();
        let mut values = Vec::<f64>::new();
        let mut structural_slots = 0usize;
        let mut valid_dense_slots = 0usize;
        let mut entries = Vec::<(u32, usize, f64)>::new();

        row_ptr.push(0);

        for br in 0..block_rows {
            entries.clear();

            let row_begin = br * b;
            let row_end = (row_begin + b).min(matrix.nrows());

            for row in row_begin..row_end {
                let local_row = row - row_begin;
                let start = matrix.row_ptr()[row] as usize;
                let end = matrix.row_ptr()[row + 1] as usize;
                entries.reserve(end.saturating_sub(start));

                for p in start..end {
                    let col = matrix.col_idx()[p] as usize;
                    let block_col = col / b;
                    let local_col = col % b;
                    let slot = local_row * b + local_col;
                    let block_col =
                        u32::try_from(block_col).map_err(|_| HybitError::SizeOverflow)?;
                    entries.push((block_col, slot, matrix.values()[p]));
                }
            }

            entries.sort_unstable_by_key(|&(block_col, slot, _)| (block_col, slot));

            let mut i = 0usize;
            while i < entries.len() {
                let block_col = entries[i].0;
                let mut local = [0.0f64; 36];
                let mut occupied_mask = 0u64;

                while i < entries.len() && entries[i].0 == block_col {
                    let slot = entries[i].1;
                    local[slot] += entries[i].2;
                    occupied_mask |= 1u64 << slot;
                    i += 1;
                }

                col_idx.push(block_col);
                structural_slots = structural_slots
                    .checked_add(occupied_mask.count_ones() as usize)
                    .ok_or(HybitError::SizeOverflow)?;

                let col_begin = block_col as usize * b;
                let col_end = (col_begin + b).min(matrix.ncols());
                valid_dense_slots = valid_dense_slots
                    .checked_add(
                        row_end
                            .saturating_sub(row_begin)
                            .saturating_mul(col_end.saturating_sub(col_begin)),
                    )
                    .ok_or(HybitError::SizeOverflow)?;

                values.extend_from_slice(&local[..b * b]);
            }

            if col_idx.len() > u32::MAX as usize {
                return Err(HybitError::SizeOverflow);
            }
            row_ptr.push(col_idx.len() as u32);
        }

        Ok(Self {
            nrows: matrix.nrows(),
            ncols: matrix.ncols(),
            block_size,
            row_ptr,
            col_idx,
            values,
            structural_slots,
            valid_dense_slots,
        })
    }

    #[inline]
    pub fn block_size(&self) -> DenseBlockSize {
        self.block_size
    }

    #[inline]
    pub fn block_width(&self) -> usize {
        self.block_size.width()
    }

    #[inline]
    pub fn unique_blocks(&self) -> usize {
        self.col_idx.len()
    }

    #[inline]
    pub fn structural_slots(&self) -> usize {
        self.structural_slots
    }

    #[inline]
    pub fn dense_value_slots(&self) -> usize {
        self.values.len()
    }

    pub fn block_fill(&self) -> f64 {
        if self.valid_dense_slots == 0 {
            0.0
        } else {
            self.structural_slots as f64 / self.valid_dense_slots as f64
        }
    }

    pub fn value_padding_ratio(&self) -> f64 {
        if self.structural_slots == 0 {
            0.0
        } else {
            self.values.len() as f64 / self.structural_slots as f64
        }
    }

    pub fn metadata_bytes(&self) -> usize {
        self.row_ptr
            .len()
            .saturating_mul(std::mem::size_of::<u32>())
            .saturating_add(
                self.col_idx
                    .len()
                    .saturating_mul(std::mem::size_of::<u32>()),
            )
    }

    pub fn storage_bytes(&self) -> usize {
        self.metadata_bytes()
            .saturating_add(self.values.len().saturating_mul(std::mem::size_of::<f64>()))
    }

    pub fn fast_block_fraction(&self) -> f64 {
        if self.col_idx.is_empty() {
            return 0.0;
        }

        let b = self.block_width();
        let full_block_rows = self.nrows / b;
        let full_block_cols = self.ncols / b;
        let mut fast_blocks = 0usize;

        for br in 0..full_block_rows {
            let start = self.row_ptr[br] as usize;
            let end = self.row_ptr[br + 1] as usize;
            fast_blocks += self.col_idx[start..end]
                .iter()
                .take_while(|&&bc| (bc as usize) < full_block_cols)
                .count();
        }

        fast_blocks as f64 / self.col_idx.len() as f64
    }

    fn apply3(&self, x: &[f64], y: &mut [f64]) {
        y.fill(0.0);

        let full_rows = self.nrows / 3;
        let full_cols = self.ncols / 3;
        let col_tail = self.ncols % 3;
        let x_ptr = x.as_ptr();
        let value_ptr = self.values.as_ptr();

        for br in 0..full_rows {
            let mut y0 = 0.0f64;
            let mut y1 = 0.0f64;
            let mut y2 = 0.0f64;

            let start = self.row_ptr[br] as usize;
            let end = self.row_ptr[br + 1] as usize;
            let tail_position =
                if col_tail != 0 && end > start && self.col_idx[end - 1] as usize == full_cols {
                    Some(end - 1)
                } else {
                    None
                };
            let fast_end = tail_position.unwrap_or(end);

            for p in start..fast_end {
                let col = self.col_idx[p] as usize * 3;
                let offset = p * 9;

                // SAFETY: block columns are constructed from validated CSR
                // indices. This loop excludes the only possible partial
                // column block, so all three x values are in range. Every
                // prepared block owns exactly nine values.
                unsafe {
                    let x0 = *x_ptr.add(col);
                    let x1 = *x_ptr.add(col + 1);
                    let x2 = *x_ptr.add(col + 2);

                    y0 += *value_ptr.add(offset) * x0
                        + *value_ptr.add(offset + 1) * x1
                        + *value_ptr.add(offset + 2) * x2;
                    y1 += *value_ptr.add(offset + 3) * x0
                        + *value_ptr.add(offset + 4) * x1
                        + *value_ptr.add(offset + 5) * x2;
                    y2 += *value_ptr.add(offset + 6) * x0
                        + *value_ptr.add(offset + 7) * x1
                        + *value_ptr.add(offset + 8) * x2;
                }
            }

            if let Some(p) = tail_position {
                let col = self.col_idx[p] as usize * 3;
                let offset = p * 9;
                for local_col in 0..col_tail {
                    let xv = x[col + local_col];
                    y0 += self.values[offset + local_col] * xv;
                    y1 += self.values[offset + 3 + local_col] * xv;
                    y2 += self.values[offset + 6 + local_col] * xv;
                }
            }

            let row = br * 3;
            y[row] = y0;
            y[row + 1] = y1;
            y[row + 2] = y2;
        }

        if self.nrows % 3 != 0 {
            self.apply_partial_last_row(x, y, full_rows, 3);
        }
    }

    fn apply6(&self, x: &[f64], y: &mut [f64]) {
        y.fill(0.0);

        let full_rows = self.nrows / 6;
        let full_cols = self.ncols / 6;
        let col_tail = self.ncols % 6;
        let x_ptr = x.as_ptr();
        let value_ptr = self.values.as_ptr();

        for br in 0..full_rows {
            let mut y0 = 0.0f64;
            let mut y1 = 0.0f64;
            let mut y2 = 0.0f64;
            let mut y3 = 0.0f64;
            let mut y4 = 0.0f64;
            let mut y5 = 0.0f64;

            let start = self.row_ptr[br] as usize;
            let end = self.row_ptr[br + 1] as usize;
            let tail_position =
                if col_tail != 0 && end > start && self.col_idx[end - 1] as usize == full_cols {
                    Some(end - 1)
                } else {
                    None
                };
            let fast_end = tail_position.unwrap_or(end);

            for p in start..fast_end {
                let col = self.col_idx[p] as usize * 6;
                let offset = p * 36;

                // SAFETY: same construction invariant as apply3, with six
                // in-range x values and 36 prepared values per block.
                unsafe {
                    let x0 = *x_ptr.add(col);
                    let x1 = *x_ptr.add(col + 1);
                    let x2 = *x_ptr.add(col + 2);
                    let x3 = *x_ptr.add(col + 3);
                    let x4 = *x_ptr.add(col + 4);
                    let x5 = *x_ptr.add(col + 5);

                    y0 += *value_ptr.add(offset) * x0
                        + *value_ptr.add(offset + 1) * x1
                        + *value_ptr.add(offset + 2) * x2
                        + *value_ptr.add(offset + 3) * x3
                        + *value_ptr.add(offset + 4) * x4
                        + *value_ptr.add(offset + 5) * x5;
                    y1 += *value_ptr.add(offset + 6) * x0
                        + *value_ptr.add(offset + 7) * x1
                        + *value_ptr.add(offset + 8) * x2
                        + *value_ptr.add(offset + 9) * x3
                        + *value_ptr.add(offset + 10) * x4
                        + *value_ptr.add(offset + 11) * x5;
                    y2 += *value_ptr.add(offset + 12) * x0
                        + *value_ptr.add(offset + 13) * x1
                        + *value_ptr.add(offset + 14) * x2
                        + *value_ptr.add(offset + 15) * x3
                        + *value_ptr.add(offset + 16) * x4
                        + *value_ptr.add(offset + 17) * x5;
                    y3 += *value_ptr.add(offset + 18) * x0
                        + *value_ptr.add(offset + 19) * x1
                        + *value_ptr.add(offset + 20) * x2
                        + *value_ptr.add(offset + 21) * x3
                        + *value_ptr.add(offset + 22) * x4
                        + *value_ptr.add(offset + 23) * x5;
                    y4 += *value_ptr.add(offset + 24) * x0
                        + *value_ptr.add(offset + 25) * x1
                        + *value_ptr.add(offset + 26) * x2
                        + *value_ptr.add(offset + 27) * x3
                        + *value_ptr.add(offset + 28) * x4
                        + *value_ptr.add(offset + 29) * x5;
                    y5 += *value_ptr.add(offset + 30) * x0
                        + *value_ptr.add(offset + 31) * x1
                        + *value_ptr.add(offset + 32) * x2
                        + *value_ptr.add(offset + 33) * x3
                        + *value_ptr.add(offset + 34) * x4
                        + *value_ptr.add(offset + 35) * x5;
                }
            }

            if let Some(p) = tail_position {
                let col = self.col_idx[p] as usize * 6;
                let offset = p * 36;
                for local_col in 0..col_tail {
                    let xv = x[col + local_col];
                    y0 += self.values[offset + local_col] * xv;
                    y1 += self.values[offset + 6 + local_col] * xv;
                    y2 += self.values[offset + 12 + local_col] * xv;
                    y3 += self.values[offset + 18 + local_col] * xv;
                    y4 += self.values[offset + 24 + local_col] * xv;
                    y5 += self.values[offset + 30 + local_col] * xv;
                }
            }

            let row = br * 6;
            y[row] = y0;
            y[row + 1] = y1;
            y[row + 2] = y2;
            y[row + 3] = y3;
            y[row + 4] = y4;
            y[row + 5] = y5;
        }

        if self.nrows % 6 != 0 {
            self.apply_partial_last_row(x, y, full_rows, 6);
        }
    }

    fn apply_partial_last_row(&self, x: &[f64], y: &mut [f64], br: usize, b: usize) {
        let row_begin = br * b;
        let row_count = self.nrows - row_begin;
        let start = self.row_ptr[br] as usize;
        let end = self.row_ptr[br + 1] as usize;

        for p in start..end {
            let col_begin = self.col_idx[p] as usize * b;
            let col_count = (self.ncols - col_begin).min(b);
            let offset = p * b * b;

            for local_row in 0..row_count {
                let value_begin = offset + local_row * b;
                let mut sum = 0.0f64;
                for local_col in 0..col_count {
                    sum += self.values[value_begin + local_col] * x[col_begin + local_col];
                }
                y[row_begin + local_row] += sum;
            }
        }
    }
}

impl LinearOperator for DenseBlockCsrOperator {
    fn rows(&self) -> usize {
        self.nrows
    }

    fn cols(&self) -> usize {
        self.ncols
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), HybitError> {
        if x.len() != self.ncols {
            return Err(HybitError::DimensionMismatch {
                expected: self.ncols,
                actual: x.len(),
            });
        }
        if y.len() != self.nrows {
            return Err(HybitError::DimensionMismatch {
                expected: self.nrows,
                actual: y.len(),
            });
        }

        match self.block_size {
            DenseBlockSize::B3 => self.apply3(x, y),
            DenseBlockSize::B6 => self.apply6(x, y),
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_to_csr(values: &[Vec<f64>]) -> Csr32Matrix {
        let nrows = values.len();
        let ncols = values.first().map_or(0, Vec::len);
        let mut row_ptr = Vec::with_capacity(nrows + 1);
        let mut col_idx = Vec::new();
        let mut data = Vec::new();
        row_ptr.push(0);

        for row in values {
            assert_eq!(row.len(), ncols);
            for (col, &value) in row.iter().enumerate() {
                if value != 0.0 {
                    col_idx.push(col as u32);
                    data.push(value);
                }
            }
            row_ptr.push(col_idx.len() as u32);
        }

        Csr32Matrix::new(nrows, ncols, row_ptr, col_idx, data).unwrap()
    }

    fn assert_matches(matrix: &Csr32Matrix, block_size: DenseBlockSize) {
        let operator = DenseBlockCsrOperator::from_csr32(matrix, block_size).unwrap();
        let x: Vec<f64> = (0..matrix.ncols())
            .map(|i| (i as f64 + 1.0) * 0.125 - 0.3)
            .collect();

        let reference = matrix.spmv(&x).unwrap();
        let mut actual = vec![0.0; matrix.nrows()];
        operator.apply(&x, &mut actual).unwrap();

        for (&reference, &actual) in reference.iter().zip(&actual) {
            let scale = reference.abs().max(actual.abs()).max(1.0);
            assert!(
                (reference - actual).abs() / scale <= 1.0e-13,
                "reference={reference:e} actual={actual:e}"
            );
        }
    }

    #[test]
    fn b3_matches_csr_on_aligned_blocks() {
        let matrix = dense_to_csr(&[
            vec![4.0, 1.0, 0.5, -1.0, 0.0, 0.0],
            vec![1.0, 5.0, 0.2, 0.0, -0.5, 0.0],
            vec![0.5, 0.2, 6.0, 0.0, 0.0, -0.25],
            vec![-1.0, 0.0, 0.0, 7.0, 0.3, 0.1],
            vec![0.0, -0.5, 0.0, 0.3, 8.0, 0.4],
            vec![0.0, 0.0, -0.25, 0.1, 0.4, 9.0],
        ]);
        assert_matches(&matrix, DenseBlockSize::B3);
    }

    #[test]
    fn b3_matches_csr_with_partial_tail() {
        let matrix = dense_to_csr(&[
            vec![3.0, 0.5, 0.0, 1.0, 0.0],
            vec![0.5, 4.0, 0.25, 0.0, 1.0],
            vec![0.0, 0.25, 5.0, -0.5, 0.0],
            vec![1.0, 0.0, -0.5, 6.0, 0.75],
            vec![0.0, 1.0, 0.0, 0.75, 7.0],
        ]);
        assert_matches(&matrix, DenseBlockSize::B3);
    }

    #[test]
    fn b6_matches_csr_on_aligned_blocks() {
        let mut dense = vec![vec![0.0; 12]; 12];
        for (row, values) in dense.iter_mut().enumerate() {
            values[row] = 10.0 + row as f64;
            if row + 1 < 12 {
                values[row + 1] = -0.5;
            }
            if row >= 1 {
                values[row - 1] = 0.25;
            }
            values[(row + 6) % 12] += 0.125;
        }
        let matrix = dense_to_csr(&dense);
        assert_matches(&matrix, DenseBlockSize::B6);
    }

    #[test]
    fn duplicate_scalar_entries_are_accumulated() {
        let matrix = Csr32Matrix::new(
            3,
            3,
            vec![0, 3, 5, 7],
            vec![0, 0, 1, 0, 1, 1, 2],
            vec![1.0, 2.0, -1.0, 0.5, 3.0, -0.25, 4.0],
        )
        .unwrap();
        assert_matches(&matrix, DenseBlockSize::B3);
    }

    #[test]
    fn reports_compact_block_metadata() {
        let matrix = dense_to_csr(&[
            vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            vec![4.0, 5.0, 6.0, 0.0, 0.0, 0.0],
            vec![7.0, 8.0, 9.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            vec![0.0, 0.0, 0.0, 4.0, 5.0, 6.0],
            vec![0.0, 0.0, 0.0, 7.0, 8.0, 9.0],
        ]);
        let operator = DenseBlockCsrOperator::from_csr32(&matrix, DenseBlockSize::B3).unwrap();

        assert_eq!(operator.unique_blocks(), 2);
        assert_eq!(operator.structural_slots(), 18);
        assert_eq!(operator.dense_value_slots(), 18);
        assert_eq!(operator.block_fill(), 1.0);
        assert_eq!(operator.value_padding_ratio(), 1.0);
        assert_eq!(operator.fast_block_fraction(), 1.0);
        assert!(operator.metadata_bytes() < matrix.metadata_bytes());
    }

    #[test]
    fn apply_checks_dimensions() {
        let matrix = dense_to_csr(&[
            vec![2.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0],
            vec![0.0, 0.0, 4.0],
        ]);
        let operator = DenseBlockCsrOperator::from_csr32(&matrix, DenseBlockSize::B3).unwrap();

        let mut y = vec![0.0; 3];
        assert!(matches!(
            operator.apply(&[1.0, 2.0], &mut y),
            Err(HybitError::DimensionMismatch { .. })
        ));

        let mut short_y = vec![0.0; 2];
        assert!(matches!(
            operator.apply(&[1.0, 2.0, 3.0], &mut short_y),
            Err(HybitError::DimensionMismatch { .. })
        ));
    }
}
