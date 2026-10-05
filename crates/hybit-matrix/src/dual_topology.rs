use hybit_core::HybitError;

use crate::{AbtmTopology, AbtmTopologyRow, Csr32Matrix};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AbtmDualTopologyStats {
    pub nrows: usize,
    pub ncols: usize,
    pub structural_nnz: usize,
    pub row_metadata_bytes: usize,
    pub column_metadata_bytes: usize,
}

impl AbtmDualTopologyStats {
    pub fn total_metadata_bytes(self) -> usize {
        self.row_metadata_bytes
            .saturating_add(self.column_metadata_bytes)
    }

    pub fn metadata_bytes_per_nnz(self) -> f64 {
        if self.structural_nnz == 0 {
            0.0
        } else {
            self.total_metadata_bytes() as f64 / self.structural_nnz as f64
        }
    }
}

/// Row and column structural topology for one sparse matrix.
///
/// `rows` describes the original matrix. `columns` is the topology of the
/// structural transpose, so `column(j)` is represented as row `j` of that
/// topology. Numerical values are not duplicated here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbtmDualTopology {
    rows: AbtmTopology,
    columns: AbtmTopology,
}

impl AbtmDualTopology {
    pub fn from_csr32(csr: &Csr32Matrix) -> Result<Self, HybitError> {
        csr.validate()?;

        let rows = AbtmTopology::from_csr32(csr)?;

        let mut transpose_counts = vec![0u32; csr.ncols()];
        for &col in csr.col_idx() {
            let count = &mut transpose_counts[col as usize];
            *count = count.checked_add(1).ok_or(HybitError::SizeOverflow)?;
        }

        let mut transpose_row_ptr = Vec::with_capacity(csr.ncols() + 1);
        transpose_row_ptr.push(0u32);
        let mut running = 0u32;
        for count in transpose_counts {
            running = running.checked_add(count).ok_or(HybitError::SizeOverflow)?;
            transpose_row_ptr.push(running);
        }

        let mut transpose_col_idx = vec![0u32; csr.nnz()];
        let mut cursors = transpose_row_ptr[..csr.ncols()].to_vec();

        for row in 0..csr.nrows() {
            let start = csr.row_ptr()[row] as usize;
            let end = csr.row_ptr()[row + 1] as usize;
            for &col in &csr.col_idx()[start..end] {
                let cursor = &mut cursors[col as usize];
                let position = *cursor as usize;
                transpose_col_idx[position] =
                    u32::try_from(row).map_err(|_| HybitError::SizeOverflow)?;
                *cursor = cursor.checked_add(1).ok_or(HybitError::SizeOverflow)?;
            }
        }

        let columns = AbtmTopology::from_structural_csr(
            csr.ncols(),
            csr.nrows(),
            &transpose_row_ptr,
            &transpose_col_idx,
        )?;

        let dual = Self { rows, columns };
        dual.validate()?;
        Ok(dual)
    }

    pub fn nrows(&self) -> usize {
        self.rows.nrows()
    }

    pub fn ncols(&self) -> usize {
        self.rows.ncols()
    }

    pub fn row_topology(&self) -> &AbtmTopology {
        &self.rows
    }

    pub fn column_topology(&self) -> &AbtmTopology {
        &self.columns
    }

    pub fn row(&self, row: usize) -> Result<AbtmTopologyRow<'_>, HybitError> {
        self.rows.row(row)
    }

    pub fn column(&self, col: usize) -> Result<AbtmTopologyRow<'_>, HybitError> {
        self.columns.row(col)
    }

    pub fn stats(&self) -> AbtmDualTopologyStats {
        let row = self.rows.stats();
        let column = self.columns.stats();
        AbtmDualTopologyStats {
            nrows: self.nrows(),
            ncols: self.ncols(),
            structural_nnz: row.structural_nnz,
            row_metadata_bytes: row.metadata_bytes,
            column_metadata_bytes: column.metadata_bytes,
        }
    }

    pub fn validate(&self) -> Result<(), HybitError> {
        self.rows.validate()?;
        self.columns.validate()?;

        if self.columns.nrows() != self.rows.ncols() || self.columns.ncols() != self.rows.nrows() {
            return Err(HybitError::InvalidMatrix(
                "ABTM dual topology transpose dimensions are inconsistent",
            ));
        }
        if self.columns.structural_nnz() != self.rows.structural_nnz() {
            return Err(HybitError::InvalidMatrix(
                "ABTM dual topology transpose nnz is inconsistent",
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn support(row: AbtmTopologyRow<'_>) -> Vec<usize> {
        (0..row.popcount())
            .map(|ordinal| row.select(ordinal).unwrap())
            .collect()
    }

    #[test]
    fn dual_topology_columns_match_structural_transpose() {
        let csr = Csr32Matrix::new(
            3,
            4,
            vec![0, 3, 5, 7],
            vec![0, 2, 3, 1, 3, 0, 2],
            vec![1.0; 7],
        )
        .unwrap();

        let dual = AbtmDualTopology::from_csr32(&csr).unwrap();

        assert_eq!(support(dual.row(0).unwrap()), vec![0, 2, 3]);
        assert_eq!(support(dual.row(1).unwrap()), vec![1, 3]);
        assert_eq!(support(dual.row(2).unwrap()), vec![0, 2]);

        assert_eq!(support(dual.column(0).unwrap()), vec![0, 2]);
        assert_eq!(support(dual.column(1).unwrap()), vec![1]);
        assert_eq!(support(dual.column(2).unwrap()), vec![0, 2]);
        assert_eq!(support(dual.column(3).unwrap()), vec![0, 1]);

        dual.validate().unwrap();
    }

    #[test]
    fn dual_topology_stats_account_for_both_orientations() {
        let csr = Csr32Matrix::new(2, 3, vec![0, 2, 4], vec![0, 2, 1, 2], vec![1.0; 4]).unwrap();

        let dual = AbtmDualTopology::from_csr32(&csr).unwrap();
        let stats = dual.stats();

        assert_eq!(stats.structural_nnz, 4);
        assert_eq!(
            stats.total_metadata_bytes(),
            stats.row_metadata_bytes + stats.column_metadata_bytes
        );
        assert!(stats.row_metadata_bytes > 0);
        assert!(stats.column_metadata_bytes > 0);
    }
}
