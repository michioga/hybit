use hybit_core::HybitError;

use crate::{AbtmTopology, Csr32Matrix, DofMask};

/// Work accounting for one or more scalar metadata-first sparse dot products.
///
/// `candidate_products` is the number of structural matrix positions that
/// would participate in an unpruned row dot. `executed_products` is the exact
/// number that survive intersection with the active sparse-vector support.
/// `skipped_products` is their difference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AbtmProductPruningStats {
    pub candidate_products: usize,
    pub executed_products: usize,
    pub skipped_products: usize,
    pub topology_words: usize,
    pub active_words: usize,
}

impl AbtmProductPruningStats {
    pub fn pruning_ratio(self) -> f64 {
        if self.candidate_products == 0 {
            0.0
        } else {
            self.skipped_products as f64 / self.candidate_products as f64
        }
    }

    pub fn empty_word_ratio(self) -> f64 {
        if self.topology_words == 0 {
            0.0
        } else {
            (self.topology_words - self.active_words) as f64 / self.topology_words as f64
        }
    }

    pub fn accumulate(&mut self, other: Self) -> Result<(), HybitError> {
        self.candidate_products = self
            .candidate_products
            .checked_add(other.candidate_products)
            .ok_or(HybitError::SizeOverflow)?;
        self.executed_products = self
            .executed_products
            .checked_add(other.executed_products)
            .ok_or(HybitError::SizeOverflow)?;
        self.skipped_products = self
            .skipped_products
            .checked_add(other.skipped_products)
            .ok_or(HybitError::SizeOverflow)?;
        self.topology_words = self
            .topology_words
            .checked_add(other.topology_words)
            .ok_or(HybitError::SizeOverflow)?;
        self.active_words = self
            .active_words
            .checked_add(other.active_words)
            .ok_or(HybitError::SizeOverflow)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AbtmMetadataFirstStats {
    pub nrows: usize,
    pub ncols: usize,
    pub structural_nnz: usize,
    pub topology_metadata_bytes: usize,
    pub value_row_ptr_bytes: usize,
    pub packed_value_bytes: usize,
}

impl AbtmMetadataFirstStats {
    pub fn total_estimated_bytes(self) -> usize {
        self.topology_metadata_bytes
            .saturating_add(self.value_row_ptr_bytes)
            .saturating_add(self.packed_value_bytes)
    }
}

/// Scalar G2 reference representation for metadata-first numerical execution.
///
/// This is deliberately not a production backend. It owns:
///
/// - the G1 logical [`AbtmTopology`];
/// - one packed numerical value per structural position, in topology order;
/// - a row pointer into that packed value stream.
///
/// Explicit structural zeros are retained. Duplicate CSR columns are summed
/// into one packed numerical value, including the case where the sum is zero.
///
/// The row-level sparse-dot kernel intersects each topology word with an
/// external [`DofMask`] before reading matrix values. It therefore provides a
/// direct reference implementation of "metadata first, numerical work second".
#[derive(Clone, Debug)]
pub struct AbtmMetadataFirstMatrix {
    topology: AbtmTopology,
    row_value_ptr: Vec<u32>,
    values: Vec<f64>,
}

impl AbtmMetadataFirstMatrix {
    pub fn from_csr32(csr: &Csr32Matrix) -> Result<Self, HybitError> {
        csr.validate()?;
        let topology = AbtmTopology::from_csr32(csr)?;

        let mut row_value_ptr = Vec::with_capacity(csr.nrows() + 1);
        let mut values = Vec::with_capacity(topology.structural_nnz());
        row_value_ptr.push(0);

        for row in 0..csr.nrows() {
            let start = csr.row_ptr()[row] as usize;
            let end = csr.row_ptr()[row + 1] as usize;

            let mut entries: Vec<(u32, f64)> = csr.col_idx()[start..end]
                .iter()
                .copied()
                .zip(csr.values()[start..end].iter().copied())
                .collect();
            entries.sort_unstable_by_key(|entry| entry.0);

            let mut canonical = Vec::with_capacity(entries.len());
            let mut index = 0usize;
            while index < entries.len() {
                let col = entries[index].0;
                let mut value = 0.0;
                while index < entries.len() && entries[index].0 == col {
                    value += entries[index].1;
                    index += 1;
                }
                canonical.push((col, value));
            }

            let topology_row = topology.row(row)?;
            if canonical.len() != topology_row.popcount() {
                return Err(HybitError::InvalidMatrix(
                    "ABTM G2 packed values do not match topology cardinality",
                ));
            }

            let mut canonical_index = 0usize;
            for word in topology_row.words() {
                let mut bits = word.mask();
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    let expected_col = word.base_col() + bit;
                    let (actual_col, value) = canonical[canonical_index];
                    if actual_col as usize != expected_col {
                        return Err(HybitError::InvalidMatrix(
                            "ABTM G2 packed value order does not match topology",
                        ));
                    }
                    values.push(value);
                    canonical_index += 1;
                    bits &= bits - 1;
                }
            }

            if canonical_index != canonical.len() {
                return Err(HybitError::InvalidMatrix(
                    "ABTM G2 packed value stream is incomplete",
                ));
            }
            if values.len() > u32::MAX as usize {
                return Err(HybitError::SizeOverflow);
            }
            row_value_ptr.push(values.len() as u32);
        }

        let matrix = Self {
            topology,
            row_value_ptr,
            values,
        };
        matrix.validate()?;
        Ok(matrix)
    }

    pub fn topology(&self) -> &AbtmTopology {
        &self.topology
    }

    pub fn packed_values(&self) -> &[f64] {
        &self.values
    }

    pub fn row_values(&self, row: usize) -> Result<&[f64], HybitError> {
        if row >= self.topology.nrows() {
            return Err(HybitError::InvalidArgument(
                "ABTM G2 row index out of range",
            ));
        }
        let start = self.row_value_ptr[row] as usize;
        let end = self.row_value_ptr[row + 1] as usize;
        Ok(&self.values[start..end])
    }

    pub fn stats(&self) -> AbtmMetadataFirstStats {
        AbtmMetadataFirstStats {
            nrows: self.topology.nrows(),
            ncols: self.topology.ncols(),
            structural_nnz: self.topology.structural_nnz(),
            topology_metadata_bytes: self.topology.stats().metadata_bytes,
            value_row_ptr_bytes: self.row_value_ptr.len() * std::mem::size_of::<u32>(),
            packed_value_bytes: self.values.len() * std::mem::size_of::<f64>(),
        }
    }

    pub fn validate(&self) -> Result<(), HybitError> {
        self.topology.validate()?;
        if self.row_value_ptr.len() != self.topology.nrows().saturating_add(1) {
            return Err(HybitError::InvalidMatrix(
                "ABTM G2 value row pointer length is invalid",
            ));
        }
        if self.row_value_ptr.first().copied() != Some(0) {
            return Err(HybitError::InvalidMatrix(
                "ABTM G2 value row pointer must start at zero",
            ));
        }
        if self.row_value_ptr.last().copied().unwrap_or_default() as usize != self.values.len() {
            return Err(HybitError::InvalidMatrix(
                "ABTM G2 value row pointer terminal offset is invalid",
            ));
        }
        if self.values.len() != self.topology.structural_nnz() {
            return Err(HybitError::InvalidMatrix(
                "ABTM G2 packed value count differs from topology nnz",
            ));
        }
        if self.values.iter().any(|value| !value.is_finite()) {
            return Err(HybitError::InvalidMatrix(
                "ABTM G2 packed values contain NaN or infinity",
            ));
        }

        for row in 0..self.topology.nrows() {
            let start = self.row_value_ptr[row] as usize;
            let end = self.row_value_ptr[row + 1] as usize;
            if start > end || end > self.values.len() {
                return Err(HybitError::InvalidMatrix(
                    "ABTM G2 value row pointer is not monotone",
                ));
            }
            if end - start != self.topology.row(row)?.popcount() {
                return Err(HybitError::InvalidMatrix(
                    "ABTM G2 row value count differs from topology row nnz",
                ));
            }
        }

        Ok(())
    }

    /// Dot one matrix row with a sparse vector represented by `(active, x)`.
    ///
    /// `active` defines the structural support of the vector. Values in `x`
    /// outside that support are never read. Matrix values whose topology bits
    /// do not intersect the support are also never read.
    pub fn sparse_dot_row(
        &self,
        row: usize,
        active: &DofMask,
        x: &[f64],
    ) -> Result<(f64, AbtmProductPruningStats), HybitError> {
        if active.len() != self.topology.ncols() {
            return Err(HybitError::DimensionMismatch {
                expected: self.topology.ncols(),
                actual: active.len(),
            });
        }
        if x.len() != self.topology.ncols() {
            return Err(HybitError::DimensionMismatch {
                expected: self.topology.ncols(),
                actual: x.len(),
            });
        }

        let topology_row = self.topology.row(row)?;
        let row_values = self.row_values(row)?;
        let active_words = active.words();

        let mut stats = AbtmProductPruningStats::default();
        let mut value_word_base = 0usize;
        let mut sum = 0.0;

        for word in topology_row.words() {
            let word_nnz = word.popcount();
            stats.candidate_products = stats
                .candidate_products
                .checked_add(word_nnz)
                .ok_or(HybitError::SizeOverflow)?;
            stats.topology_words = stats
                .topology_words
                .checked_add(1)
                .ok_or(HybitError::SizeOverflow)?;

            let support = active_words
                .get(word.word_index() as usize)
                .copied()
                .unwrap_or(0);
            let mut bits = word.mask() & support;

            if bits != 0 {
                stats.active_words = stats
                    .active_words
                    .checked_add(1)
                    .ok_or(HybitError::SizeOverflow)?;
            }

            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let packed_ordinal = value_word_base + word.rank(bit)?;
                let col = word.base_col() + bit;

                sum += row_values[packed_ordinal] * x[col];
                stats.executed_products = stats
                    .executed_products
                    .checked_add(1)
                    .ok_or(HybitError::SizeOverflow)?;

                bits &= bits - 1;
            }

            value_word_base = value_word_base
                .checked_add(word_nnz)
                .ok_or(HybitError::SizeOverflow)?;
        }

        if value_word_base != row_values.len() {
            return Err(HybitError::InvalidMatrix(
                "ABTM G2 row topology/value traversal mismatch",
            ));
        }

        stats.skipped_products = stats
            .candidate_products
            .checked_sub(stats.executed_products)
            .ok_or(HybitError::InvalidMatrix(
                "ABTM G2 executed products exceed candidate products",
            ))?;

        Ok((sum, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_values_retain_structural_zeros_and_sum_duplicates() {
        let csr = Csr32Matrix::new(
            2,
            70,
            vec![0, 4, 7],
            vec![0, 0, 63, 64, 1, 65, 65],
            vec![0.0, 2.0, -1.0, 0.0, 3.0, 4.0, -4.0],
        )
        .unwrap();

        let prepared = AbtmMetadataFirstMatrix::from_csr32(&csr).unwrap();
        assert_eq!(prepared.topology().structural_nnz(), 5);
        assert_eq!(prepared.row_values(0).unwrap(), &[2.0, -1.0, 0.0]);
        assert_eq!(prepared.row_values(1).unwrap(), &[3.0, 0.0]);
        prepared.validate().unwrap();
    }

    #[test]
    fn sparse_dot_prunes_before_numeric_value_loads() {
        let csr = Csr32Matrix::new(
            1,
            130,
            vec![0, 6],
            vec![0, 2, 63, 64, 127, 129],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let prepared = AbtmMetadataFirstMatrix::from_csr32(&csr).unwrap();
        let active = DofMask::from_indices(130, &[2, 64, 129]).unwrap();

        let mut x = vec![0.0; 130];
        x[2] = 10.0;
        x[64] = 20.0;
        x[129] = 30.0;

        let (dot, stats) = prepared.sparse_dot_row(0, &active, &x).unwrap();
        assert_eq!(dot, 2.0 * 10.0 + 4.0 * 20.0 + 6.0 * 30.0);
        assert_eq!(stats.candidate_products, 6);
        assert_eq!(stats.executed_products, 3);
        assert_eq!(stats.skipped_products, 3);
        assert_eq!(stats.topology_words, 3);
        assert_eq!(stats.active_words, 3);
        assert_eq!(stats.pruning_ratio(), 0.5);
    }

    #[test]
    fn full_support_sparse_dot_matches_csr_row_dot() {
        let csr = Csr32Matrix::new(
            2,
            5,
            vec![0, 3, 6],
            vec![0, 2, 4, 0, 1, 3],
            vec![2.0, -1.0, 0.5, 1.0, 3.0, -2.0],
        )
        .unwrap();
        let prepared = AbtmMetadataFirstMatrix::from_csr32(&csr).unwrap();
        let active = DofMask::from_indices(5, &[0, 1, 2, 3, 4]).unwrap();
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];

        for row in 0..2 {
            let start = csr.row_ptr()[row] as usize;
            let end = csr.row_ptr()[row + 1] as usize;
            let reference = (start..end)
                .map(|p| csr.values()[p] * x[csr.col_idx()[p] as usize])
                .sum::<f64>();
            let (actual, stats) = prepared.sparse_dot_row(row, &active, &x).unwrap();
            assert!((actual - reference).abs() < 1.0e-12);
            assert_eq!(stats.skipped_products, 0);
        }
    }
}
