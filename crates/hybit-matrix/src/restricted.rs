use hybit_core::{HybitError, LinearOperator};

use crate::{AbtmMatrix, Csr32Matrix, DofMask, TileKind, TILE_WIDTH};

#[inline(always)]
fn bits_below(bit: usize) -> u64 {
    if bit == 0 {
        0
    } else {
        (1u64 << bit) - 1
    }
}

/// Explicit prepared operator for a fixed column restriction `A * M`.
///
/// The prepared CSR keeps the original global row/column dimensions but stores
/// only entries whose column is active in `M`. Repeated applications therefore
/// consume the original global input vector directly, without a per-apply
/// mask branch or masked-vector refresh.
///
/// Construction from CSR is the conservative default. `from_abtm` is an
/// explicit alternative when an `AbtmMatrix` is already available; no
/// automatic CSR/ABTM preparation routing is performed.
#[derive(Clone, Debug)]
pub struct PreparedColumnRestrictedCsrOperator {
    matrix: Csr32Matrix,
    active_columns: DofMask,
    source_nnz: usize,
}

impl PreparedColumnRestrictedCsrOperator {
    pub fn from_csr32(source: &Csr32Matrix, active_columns: &DofMask) -> Result<Self, HybitError> {
        source.validate()?;
        if active_columns.len() != source.ncols() {
            return Err(HybitError::DimensionMismatch {
                expected: source.ncols(),
                actual: active_columns.len(),
            });
        }

        let mut row_ptr = Vec::with_capacity(source.nrows() + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);

        for row in 0..source.nrows() {
            let start = source.row_ptr()[row] as usize;
            let end = source.row_ptr()[row + 1] as usize;

            for p in start..end {
                let col = source.col_idx()[p] as usize;
                if active_columns.contains(col) {
                    col_idx.push(source.col_idx()[p]);
                    values.push(source.values()[p]);
                }
            }

            if col_idx.len() > u32::MAX as usize {
                return Err(HybitError::SizeOverflow);
            }
            row_ptr.push(col_idx.len() as u32);
        }

        let matrix = Csr32Matrix::new(source.nrows(), source.ncols(), row_ptr, col_idx, values)?;

        Ok(Self {
            matrix,
            active_columns: active_columns.clone(),
            source_nnz: source.nnz(),
        })
    }

    /// Prepare the same fixed-column restriction from an already-prepared ABTM
    /// numerical layout.
    ///
    /// ABTM has already canonicalized duplicate scalar entries and removed
    /// exact numerical zeros, so this constructor preserves ABTM's canonical
    /// operator semantics rather than the original stored-entry multiplicity.
    pub fn from_abtm(source: &AbtmMatrix, active_columns: &DofMask) -> Result<Self, HybitError> {
        let stats = source.stats();
        if active_columns.len() != stats.ncols {
            return Err(HybitError::DimensionMismatch {
                expected: stats.ncols,
                actual: active_columns.len(),
            });
        }

        let mut row_ptr = Vec::with_capacity(stats.nrows + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);

        for row in 0..stats.nrows {
            for tile in source.row_tiles(row)? {
                let word_index = tile.base_col() / TILE_WIDTH;
                let active_word = active_columns.words().get(word_index).copied().unwrap_or(0);
                let mut active = tile.mask & active_word;
                if active == 0 {
                    continue;
                }

                let base = tile.base_col();
                let value_offset = tile.value_offset as usize;

                match tile.kind() {
                    TileKind::Sparse | TileKind::Bitmap => {
                        while active != 0 {
                            let bit = active.trailing_zeros() as usize;
                            let rank = (tile.mask & bits_below(bit)).count_ones() as usize;
                            let col = base.checked_add(bit).ok_or(HybitError::SizeOverflow)?;
                            col_idx.push(u32::try_from(col).map_err(|_| HybitError::SizeOverflow)?);
                            values.push(source.values()[value_offset + rank]);
                            active &= active - 1;
                        }
                    }
                    TileKind::Dense => {
                        while active != 0 {
                            let bit = active.trailing_zeros() as usize;
                            let col = base.checked_add(bit).ok_or(HybitError::SizeOverflow)?;
                            col_idx.push(u32::try_from(col).map_err(|_| HybitError::SizeOverflow)?);
                            values.push(source.values()[value_offset + bit]);
                            active &= active - 1;
                        }
                    }
                }
            }

            if col_idx.len() > u32::MAX as usize {
                return Err(HybitError::SizeOverflow);
            }
            row_ptr.push(col_idx.len() as u32);
        }

        let matrix = Csr32Matrix::new(stats.nrows, stats.ncols, row_ptr, col_idx, values)?;

        Ok(Self {
            matrix,
            active_columns: active_columns.clone(),
            source_nnz: stats.matrix_nnz,
        })
    }

    pub fn matrix(&self) -> &Csr32Matrix {
        &self.matrix
    }

    pub fn active_columns(&self) -> &DofMask {
        &self.active_columns
    }

    pub fn source_nnz(&self) -> usize {
        self.source_nnz
    }

    pub fn retained_nnz(&self) -> usize {
        self.matrix.nnz()
    }

    pub fn retained_fraction(&self) -> f64 {
        if self.source_nnz == 0 {
            0.0
        } else {
            self.retained_nnz() as f64 / self.source_nnz as f64
        }
    }

    pub fn storage_bytes(&self) -> usize {
        self.matrix.storage_bytes().saturating_add(
            self.active_columns
                .words()
                .len()
                .saturating_mul(std::mem::size_of::<u64>()),
        )
    }
}

impl LinearOperator for PreparedColumnRestrictedCsrOperator {
    fn rows(&self) -> usize {
        self.matrix.nrows()
    }

    fn cols(&self) -> usize {
        self.matrix.ncols()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), HybitError> {
        self.matrix.apply(x, y)
    }
}

/// Explicit prepared compact operator for a fixed local restriction `R A R`.
///
/// Local numbering follows ascending global DOF order. Preparation scans only
/// rows contained in `R`, maps retained columns into compact local numbering,
/// and stores an ordinary compact CSR for repeated arithmetic.
///
/// The global-to-local map is preparation scratch and is not retained. Repeated
/// `LinearOperator::apply` therefore operates on local vectors. `gather_input`
/// is provided for callers whose changing input vector remains global.
#[derive(Clone, Debug)]
pub struct PreparedLocalCsrOperator {
    global_size: usize,
    global_nodes: Vec<u32>,
    matrix: Csr32Matrix,
    source_row_nnz: usize,
}

impl PreparedLocalCsrOperator {
    pub fn from_csr32(source: &Csr32Matrix, region: &DofMask) -> Result<Self, HybitError> {
        source.validate()?;
        if source.nrows() != source.ncols() {
            return Err(HybitError::InvalidMatrix(
                "local R A R preparation requires a square matrix",
            ));
        }
        if region.len() != source.nrows() {
            return Err(HybitError::DimensionMismatch {
                expected: source.nrows(),
                actual: region.len(),
            });
        }

        let global_indices = region.indices();
        if global_indices.len() > u32::MAX as usize {
            return Err(HybitError::SizeOverflow);
        }

        let mut global_nodes = Vec::with_capacity(global_indices.len());
        let mut global_to_local = vec![u32::MAX; source.ncols()];

        for (local, &global) in global_indices.iter().enumerate() {
            let global_u32 = u32::try_from(global).map_err(|_| HybitError::SizeOverflow)?;
            let local_u32 = u32::try_from(local).map_err(|_| HybitError::SizeOverflow)?;
            global_nodes.push(global_u32);
            global_to_local[global] = local_u32;
        }

        let mut row_ptr = Vec::with_capacity(global_nodes.len() + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        let mut source_row_nnz = 0usize;
        row_ptr.push(0);

        for &global_row in &global_nodes {
            let global_row = global_row as usize;
            let start = source.row_ptr()[global_row] as usize;
            let end = source.row_ptr()[global_row + 1] as usize;
            source_row_nnz = source_row_nnz.saturating_add(end - start);

            for p in start..end {
                let global_col = source.col_idx()[p] as usize;
                let local_col = global_to_local[global_col];
                if local_col != u32::MAX {
                    col_idx.push(local_col);
                    values.push(source.values()[p]);
                }
            }

            if col_idx.len() > u32::MAX as usize {
                return Err(HybitError::SizeOverflow);
            }
            row_ptr.push(col_idx.len() as u32);
        }

        let matrix = Csr32Matrix::new(
            global_nodes.len(),
            global_nodes.len(),
            row_ptr,
            col_idx,
            values,
        )?;

        Ok(Self {
            global_size: source.nrows(),
            global_nodes,
            matrix,
            source_row_nnz,
        })
    }

    pub fn global_size(&self) -> usize {
        self.global_size
    }

    pub fn global_nodes(&self) -> &[u32] {
        &self.global_nodes
    }

    pub fn matrix(&self) -> &Csr32Matrix {
        &self.matrix
    }

    pub fn local_nodes(&self) -> usize {
        self.global_nodes.len()
    }

    pub fn local_nnz(&self) -> usize {
        self.matrix.nnz()
    }

    pub fn source_row_nnz(&self) -> usize {
        self.source_row_nnz
    }

    pub fn boundary_pruning_ratio(&self) -> f64 {
        if self.source_row_nnz == 0 {
            0.0
        } else {
            1.0 - self.local_nnz() as f64 / self.source_row_nnz as f64
        }
    }

    pub fn storage_bytes(&self) -> usize {
        self.matrix.storage_bytes().saturating_add(
            self.global_nodes
                .len()
                .saturating_mul(std::mem::size_of::<u32>()),
        )
    }

    pub fn gather_input(&self, global: &[f64], local: &mut [f64]) -> Result<(), HybitError> {
        if global.len() != self.global_size {
            return Err(HybitError::DimensionMismatch {
                expected: self.global_size,
                actual: global.len(),
            });
        }
        if local.len() != self.local_nodes() {
            return Err(HybitError::DimensionMismatch {
                expected: self.local_nodes(),
                actual: local.len(),
            });
        }

        for (out, &global_index) in local.iter_mut().zip(&self.global_nodes) {
            *out = global[global_index as usize];
        }
        Ok(())
    }

    pub fn gather_input_vec(&self, global: &[f64]) -> Result<Vec<f64>, HybitError> {
        let mut local = vec![0.0; self.local_nodes()];
        self.gather_input(global, &mut local)?;
        Ok(local)
    }
}

impl LinearOperator for PreparedLocalCsrOperator {
    fn rows(&self) -> usize {
        self.matrix.nrows()
    }

    fn cols(&self) -> usize {
        self.matrix.ncols()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), HybitError> {
        self.matrix.apply(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AbtmConfig;

    fn assert_close(reference: &[f64], actual: &[f64]) {
        assert_eq!(reference.len(), actual.len());
        for (&r, &a) in reference.iter().zip(actual) {
            let scale = r.abs().max(a.abs()).max(1.0);
            assert!(
                (r - a).abs() / scale <= 1.0e-13,
                "reference={r:e} actual={a:e}"
            );
        }
    }

    #[test]
    fn column_restriction_matches_zeroed_input() {
        let matrix = Csr32Matrix::new(
            4,
            4,
            vec![0, 3, 6, 9, 12],
            vec![0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3],
            vec![
                4.0, -1.0, 0.5, -1.0, 5.0, 0.25, 0.25, 6.0, -0.75, 0.5, -0.75, 7.0,
            ],
        )
        .unwrap();
        let mask = DofMask::from_indices(4, &[0, 2]).unwrap();
        let prepared = PreparedColumnRestrictedCsrOperator::from_csr32(&matrix, &mask).unwrap();

        let x = [1.0, 2.0, 3.0, 4.0];
        let x_masked = [1.0, 0.0, 3.0, 0.0];
        let reference = matrix.spmv(&x_masked).unwrap();
        let actual = prepared.matrix().spmv(&x).unwrap();

        assert_close(&reference, &actual);
        assert_eq!(prepared.retained_nnz(), 6);
        assert_eq!(prepared.active_columns().indices(), vec![0, 2]);
    }

    #[test]
    fn column_restriction_from_abtm_matches_csr_path() {
        let matrix = Csr32Matrix::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
        )
        .unwrap();
        let mask = DofMask::from_indices(3, &[0, 2]).unwrap();
        let abtm = AbtmMatrix::from_csr32(&matrix, AbtmConfig::default()).unwrap();

        let direct = PreparedColumnRestrictedCsrOperator::from_csr32(&matrix, &mask).unwrap();
        let metadata = PreparedColumnRestrictedCsrOperator::from_abtm(&abtm, &mask).unwrap();

        assert_eq!(direct.matrix().row_ptr(), metadata.matrix().row_ptr());
        assert_eq!(direct.matrix().col_idx(), metadata.matrix().col_idx());
        assert_eq!(direct.matrix().values(), metadata.matrix().values());
    }

    #[test]
    fn local_restriction_matches_global_region_scan() {
        let matrix = Csr32Matrix::new(
            5,
            5,
            vec![0, 3, 6, 9, 12, 15],
            vec![0, 1, 4, 0, 1, 2, 1, 2, 3, 0, 3, 4, 0, 3, 4],
            vec![
                4.0, -1.0, 0.5, -1.0, 5.0, 0.25, 0.25, 6.0, -0.75, 0.5, 7.0, -1.0, 0.5, -1.0, 8.0,
            ],
        )
        .unwrap();
        let region = DofMask::from_indices(5, &[0, 3, 4]).unwrap();
        let prepared = PreparedLocalCsrOperator::from_csr32(&matrix, &region).unwrap();

        assert_eq!(prepared.global_nodes(), &[0, 3, 4]);
        assert_eq!(prepared.local_nodes(), 3);

        let x_global = [1.0, 2.0, 3.0, 4.0, 5.0];
        let x_local = prepared.gather_input_vec(&x_global).unwrap();
        assert_eq!(x_local, vec![1.0, 4.0, 5.0]);

        let mut reference = vec![0.0; 3];
        for (local_row, &global_row) in prepared.global_nodes().iter().enumerate() {
            let row = global_row as usize;
            let start = matrix.row_ptr()[row] as usize;
            let end = matrix.row_ptr()[row + 1] as usize;
            for p in start..end {
                let col = matrix.col_idx()[p] as usize;
                if region.contains(col) {
                    reference[local_row] += matrix.values()[p] * x_global[col];
                }
            }
        }

        let actual = prepared.matrix().spmv(&x_local).unwrap();
        assert_close(&reference, &actual);
    }

    #[test]
    fn prepared_restrictions_validate_dimensions() {
        let matrix =
            Csr32Matrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0]).unwrap();

        let bad_mask = DofMask::new(2);
        assert!(PreparedColumnRestrictedCsrOperator::from_csr32(&matrix, &bad_mask).is_err());
        assert!(PreparedLocalCsrOperator::from_csr32(&matrix, &bad_mask).is_err());

        let region = DofMask::from_indices(3, &[0, 2]).unwrap();
        let local = PreparedLocalCsrOperator::from_csr32(&matrix, &region).unwrap();
        let mut gathered = vec![0.0; 2];
        assert!(local.gather_input(&[1.0, 2.0], &mut gathered).is_err());
    }
}
