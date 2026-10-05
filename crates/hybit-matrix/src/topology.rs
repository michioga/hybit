use hybit_core::HybitError;

use crate::Csr32Matrix;

pub const ABTM_TOPOLOGY_WORD_BITS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AbtmTopologyWord {
    word_index: u32,
    mask: u64,
}

impl AbtmTopologyWord {
    #[inline]
    pub fn word_index(self) -> u32 {
        self.word_index
    }
    #[inline]
    pub fn base_col(self) -> usize {
        self.word_index as usize * ABTM_TOPOLOGY_WORD_BITS
    }
    #[inline]
    pub fn mask(self) -> u64 {
        self.mask
    }
    #[inline]
    pub fn popcount(self) -> usize {
        self.mask.count_ones() as usize
    }
    #[inline]
    pub fn contains_offset(self, offset: usize) -> bool {
        offset < ABTM_TOPOLOGY_WORD_BITS && (self.mask & (1u64 << offset)) != 0
    }
    pub fn rank(self, offset: usize) -> Result<usize, HybitError> {
        if offset > ABTM_TOPOLOGY_WORD_BITS {
            return Err(HybitError::InvalidArgument(
                "ABTM topology rank offset out of range",
            ));
        }
        if offset == ABTM_TOPOLOGY_WORD_BITS {
            return Ok(self.popcount());
        }
        if offset == 0 {
            return Ok(0);
        }
        Ok((self.mask & ((1u64 << offset) - 1)).count_ones() as usize)
    }
    pub fn select(self, ordinal: usize) -> Option<usize> {
        if ordinal >= self.popcount() {
            return None;
        }
        let mut bits = self.mask;
        for _ in 0..ordinal {
            bits &= bits - 1;
        }
        Some(bits.trailing_zeros() as usize)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AbtmTopologyRow<'a> {
    ncols: usize,
    word_indices: &'a [u32],
    masks: &'a [u64],
}

impl<'a> AbtmTopologyRow<'a> {
    pub fn nonempty_words(self) -> usize {
        self.masks.len()
    }
    pub fn popcount(self) -> usize {
        self.masks
            .iter()
            .map(|mask| mask.count_ones() as usize)
            .sum()
    }
    pub fn words(self) -> impl Iterator<Item = AbtmTopologyWord> + 'a {
        self.word_indices
            .iter()
            .copied()
            .zip(self.masks.iter().copied())
            .map(|(word_index, mask)| AbtmTopologyWord { word_index, mask })
    }
    pub fn contains(self, col: usize) -> bool {
        if col >= self.ncols {
            return false;
        }
        let word = (col / ABTM_TOPOLOGY_WORD_BITS) as u32;
        let offset = col % ABTM_TOPOLOGY_WORD_BITS;
        match self.word_indices.binary_search(&word) {
            Ok(index) => (self.masks[index] & (1u64 << offset)) != 0,
            Err(_) => false,
        }
    }
    pub fn rank(self, col: usize) -> Result<usize, HybitError> {
        if col > self.ncols {
            return Err(HybitError::InvalidArgument(
                "ABTM topology row rank column out of range",
            ));
        }
        let target_word = col / ABTM_TOPOLOGY_WORD_BITS;
        let target_offset = col % ABTM_TOPOLOGY_WORD_BITS;
        let mut rank = 0usize;
        for (&word_index, &mask) in self.word_indices.iter().zip(self.masks) {
            let word_index = word_index as usize;
            if word_index < target_word {
                rank += mask.count_ones() as usize;
                continue;
            }
            if word_index == target_word && target_offset != 0 {
                rank += (mask & ((1u64 << target_offset) - 1)).count_ones() as usize;
            }
            break;
        }
        Ok(rank)
    }
    pub fn select(self, mut ordinal: usize) -> Option<usize> {
        for word in self.words() {
            let count = word.popcount();
            if ordinal < count {
                let col = word.base_col() + word.select(ordinal)?;
                return (col < self.ncols).then_some(col);
            }
            ordinal -= count;
        }
        None
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AbtmTopologyStats {
    pub nrows: usize,
    pub ncols: usize,
    pub structural_nnz: usize,
    pub nonempty_words: usize,
    pub metadata_bytes: usize,
}

impl AbtmTopologyStats {
    pub fn average_word_nnz(self) -> f64 {
        if self.nonempty_words == 0 {
            0.0
        } else {
            self.structural_nnz as f64 / self.nonempty_words as f64
        }
    }
    pub fn word_fill_ratio(self) -> f64 {
        self.average_word_nnz() / ABTM_TOPOLOGY_WORD_BITS as f64
    }
    pub fn metadata_bytes_per_nnz(self) -> f64 {
        if self.structural_nnz == 0 {
            0.0
        } else {
            self.metadata_bytes as f64 / self.structural_nnz as f64
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbtmTopology {
    nrows: usize,
    ncols: usize,
    row_word_ptr: Vec<u32>,
    word_indices: Vec<u32>,
    masks: Vec<u64>,
    structural_nnz: usize,
}

impl AbtmTopology {
    pub fn from_csr32(csr: &Csr32Matrix) -> Result<Self, HybitError> {
        csr.validate()?;
        Self::from_structural_csr(csr.nrows(), csr.ncols(), csr.row_ptr(), csr.col_idx())
    }

    pub(crate) fn from_structural_csr(
        nrows: usize,
        ncols: usize,
        row_ptr: &[u32],
        col_idx: &[u32],
    ) -> Result<Self, HybitError> {
        if row_ptr.len() != nrows.saturating_add(1) {
            return Err(HybitError::InvalidMatrix(
                "ABTM topology structural row pointer length is invalid",
            ));
        }
        if row_ptr.first().copied() != Some(0) {
            return Err(HybitError::InvalidMatrix(
                "ABTM topology structural row pointer must start at zero",
            ));
        }
        if col_idx.len() > u32::MAX as usize {
            return Err(HybitError::SizeOverflow);
        }
        if row_ptr.last().copied().unwrap_or_default() as usize != col_idx.len() {
            return Err(HybitError::InvalidMatrix(
                "ABTM topology structural row pointer terminal offset is invalid",
            ));
        }
        for pair in row_ptr.windows(2) {
            if pair[0] > pair[1] {
                return Err(HybitError::InvalidMatrix(
                    "ABTM topology structural row pointer is not monotone",
                ));
            }
        }
        if col_idx.iter().any(|&col| col as usize >= ncols) {
            return Err(HybitError::InvalidMatrix(
                "ABTM topology structural column index out of range",
            ));
        }

        let mut row_word_ptr = Vec::with_capacity(nrows + 1);
        let mut word_indices = Vec::new();
        let mut masks = Vec::new();
        let mut structural_nnz = 0usize;
        row_word_ptr.push(0);

        for row in 0..nrows {
            let start = row_ptr[row] as usize;
            let end = row_ptr[row + 1] as usize;
            let mut row_words: Vec<(u32, u64)> = Vec::new();

            for &col in &col_idx[start..end] {
                let col = col as usize;
                let word = col / ABTM_TOPOLOGY_WORD_BITS;
                if word > u32::MAX as usize {
                    return Err(HybitError::SizeOverflow);
                }
                let word = word as u32;
                let bit = col % ABTM_TOPOLOGY_WORD_BITS;

                if let Some((last_word, last_mask)) = row_words.last_mut() {
                    if *last_word == word {
                        *last_mask |= 1u64 << bit;
                        continue;
                    }
                }

                match row_words.binary_search_by_key(&word, |&(index, _)| index) {
                    Ok(index) => row_words[index].1 |= 1u64 << bit,
                    Err(index) => row_words.insert(index, (word, 1u64 << bit)),
                }
            }

            for (word, mask) in row_words {
                structural_nnz = structural_nnz
                    .checked_add(mask.count_ones() as usize)
                    .ok_or(HybitError::SizeOverflow)?;
                word_indices.push(word);
                masks.push(mask);
            }

            if masks.len() > u32::MAX as usize {
                return Err(HybitError::SizeOverflow);
            }
            row_word_ptr.push(masks.len() as u32);
        }

        let topology = Self {
            nrows,
            ncols,
            row_word_ptr,
            word_indices,
            masks,
            structural_nnz,
        };
        topology.validate()?;
        Ok(topology)
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    pub fn structural_nnz(&self) -> usize {
        self.structural_nnz
    }
    pub fn nonempty_words(&self) -> usize {
        self.masks.len()
    }

    pub fn row(&self, row: usize) -> Result<AbtmTopologyRow<'_>, HybitError> {
        if row >= self.nrows {
            return Err(HybitError::InvalidArgument(
                "ABTM topology row index out of range",
            ));
        }
        let start = self.row_word_ptr[row] as usize;
        let end = self.row_word_ptr[row + 1] as usize;
        Ok(AbtmTopologyRow {
            ncols: self.ncols,
            word_indices: &self.word_indices[start..end],
            masks: &self.masks[start..end],
        })
    }

    pub fn stats(&self) -> AbtmTopologyStats {
        AbtmTopologyStats {
            nrows: self.nrows,
            ncols: self.ncols,
            structural_nnz: self.structural_nnz,
            nonempty_words: self.masks.len(),
            metadata_bytes: self
                .row_word_ptr
                .len()
                .saturating_mul(std::mem::size_of::<u32>())
                .saturating_add(
                    self.word_indices
                        .len()
                        .saturating_mul(std::mem::size_of::<u32>()),
                )
                .saturating_add(self.masks.len().saturating_mul(std::mem::size_of::<u64>())),
        }
    }

    pub fn validate(&self) -> Result<(), HybitError> {
        if self.row_word_ptr.len() != self.nrows.saturating_add(1) {
            return Err(HybitError::InvalidMatrix(
                "ABTM topology row pointer length is invalid",
            ));
        }
        if self.row_word_ptr.first().copied() != Some(0) {
            return Err(HybitError::InvalidMatrix(
                "ABTM topology row pointer must start at zero",
            ));
        }
        if self.word_indices.len() != self.masks.len() {
            return Err(HybitError::InvalidMatrix(
                "ABTM topology word/mask lengths differ",
            ));
        }
        if self.row_word_ptr.last().copied().unwrap_or_default() as usize != self.masks.len() {
            return Err(HybitError::InvalidMatrix(
                "ABTM topology row pointer terminal offset is invalid",
            ));
        }
        let mut counted_nnz = 0usize;
        for row in 0..self.nrows {
            let start = self.row_word_ptr[row] as usize;
            let end = self.row_word_ptr[row + 1] as usize;
            if start > end || end > self.masks.len() {
                return Err(HybitError::InvalidMatrix(
                    "ABTM topology row pointer is not monotone",
                ));
            }
            let mut previous_word = None;
            for index in start..end {
                let word = self.word_indices[index] as usize;
                let mask = self.masks[index];
                if mask == 0 {
                    return Err(HybitError::InvalidMatrix(
                        "ABTM topology stores an empty word",
                    ));
                }
                if previous_word.is_some_and(|previous| word <= previous) {
                    return Err(HybitError::InvalidMatrix(
                        "ABTM topology row words are not strictly increasing",
                    ));
                }
                let base = word
                    .checked_mul(ABTM_TOPOLOGY_WORD_BITS)
                    .ok_or(HybitError::SizeOverflow)?;
                if base >= self.ncols {
                    return Err(HybitError::InvalidMatrix(
                        "ABTM topology word starts beyond matrix columns",
                    ));
                }
                let remaining = self.ncols - base;
                if remaining < ABTM_TOPOLOGY_WORD_BITS {
                    let valid_mask = (1u64 << remaining) - 1;
                    if mask & !valid_mask != 0 {
                        return Err(HybitError::InvalidMatrix(
                            "ABTM topology has bits beyond matrix columns",
                        ));
                    }
                }
                counted_nnz = counted_nnz
                    .checked_add(mask.count_ones() as usize)
                    .ok_or(HybitError::SizeOverflow)?;
                previous_word = Some(word);
            }
        }
        if counted_nnz != self.structural_nnz {
            return Err(HybitError::InvalidMatrix(
                "ABTM topology structural nnz is inconsistent",
            ));
        }
        Ok(())
    }

    pub fn intersection(&self, other: &Self) -> Result<Self, HybitError> {
        self.combine(other, |a, b| a & b)
    }
    pub fn union(&self, other: &Self) -> Result<Self, HybitError> {
        self.combine(other, |a, b| a | b)
    }
    pub fn and_not(&self, other: &Self) -> Result<Self, HybitError> {
        self.combine(other, |a, b| a & !b)
    }
    pub fn xor(&self, other: &Self) -> Result<Self, HybitError> {
        self.combine(other, |a, b| a ^ b)
    }

    fn combine<F>(&self, other: &Self, op: F) -> Result<Self, HybitError>
    where
        F: Fn(u64, u64) -> u64,
    {
        if self.nrows != other.nrows || self.ncols != other.ncols {
            return Err(HybitError::InvalidArgument(
                "ABTM topology dimensions differ",
            ));
        }
        let mut row_word_ptr = Vec::with_capacity(self.nrows + 1);
        let mut word_indices = Vec::new();
        let mut masks = Vec::new();
        let mut structural_nnz = 0usize;
        row_word_ptr.push(0);
        for row in 0..self.nrows {
            let a_start = self.row_word_ptr[row] as usize;
            let a_end = self.row_word_ptr[row + 1] as usize;
            let b_start = other.row_word_ptr[row] as usize;
            let b_end = other.row_word_ptr[row + 1] as usize;
            let (mut a, mut b) = (a_start, b_start);
            while a < a_end || b < b_end {
                let (word, a_mask, b_mask) = match (a < a_end, b < b_end) {
                    (true, true) => {
                        let aw = self.word_indices[a];
                        let bw = other.word_indices[b];
                        if aw < bw {
                            let r = (aw, self.masks[a], 0);
                            a += 1;
                            r
                        } else if bw < aw {
                            let r = (bw, 0, other.masks[b]);
                            b += 1;
                            r
                        } else {
                            let r = (aw, self.masks[a], other.masks[b]);
                            a += 1;
                            b += 1;
                            r
                        }
                    }
                    (true, false) => {
                        let r = (self.word_indices[a], self.masks[a], 0);
                        a += 1;
                        r
                    }
                    (false, true) => {
                        let r = (other.word_indices[b], 0, other.masks[b]);
                        b += 1;
                        r
                    }
                    (false, false) => unreachable!(),
                };
                let mask = op(a_mask, b_mask);
                if mask != 0 {
                    structural_nnz = structural_nnz
                        .checked_add(mask.count_ones() as usize)
                        .ok_or(HybitError::SizeOverflow)?;
                    word_indices.push(word);
                    masks.push(mask);
                }
            }
            if masks.len() > u32::MAX as usize {
                return Err(HybitError::SizeOverflow);
            }
            row_word_ptr.push(masks.len() as u32);
        }
        let topology = Self {
            nrows: self.nrows,
            ncols: self.ncols,
            row_word_ptr,
            word_indices,
            masks,
            structural_nnz,
        };
        topology.validate()?;
        Ok(topology)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row_indices(topology: &AbtmTopology, row: usize) -> Vec<usize> {
        let row = topology.row(row).unwrap();
        (0..row.popcount())
            .map(|ordinal| row.select(ordinal).unwrap())
            .collect()
    }

    #[test]
    fn topology_retains_explicit_zero_and_collapses_duplicates() {
        let csr = Csr32Matrix::new(
            2,
            70,
            vec![0, 4, 7],
            vec![0, 0, 63, 64, 1, 65, 65],
            vec![0.0, 2.0, -1.0, 0.0, 3.0, 4.0, -4.0],
        )
        .unwrap();
        let topology = AbtmTopology::from_csr32(&csr).unwrap();
        assert_eq!(topology.structural_nnz(), 5);
        assert_eq!(row_indices(&topology, 0), vec![0, 63, 64]);
        assert_eq!(row_indices(&topology, 1), vec![1, 65]);
    }

    #[test]
    fn word_rank_and_select_are_inverse() {
        let word = AbtmTopologyWord {
            word_index: 3,
            mask: (1u64 << 0) | (1u64 << 2) | (1u64 << 17) | (1u64 << 63),
        };
        assert_eq!(word.rank(18).unwrap(), 3);
        assert_eq!(word.rank(64).unwrap(), 4);
        for ordinal in 0..word.popcount() {
            let bit = word.select(ordinal).unwrap();
            assert_eq!(word.rank(bit).unwrap(), ordinal);
            assert!(word.contains_offset(bit));
        }
    }

    #[test]
    fn row_rank_and_select_are_inverse_across_words() {
        let csr = Csr32Matrix::new(
            1,
            130,
            vec![0, 6],
            vec![0, 2, 63, 64, 127, 129],
            vec![1.0; 6],
        )
        .unwrap();
        let topology = AbtmTopology::from_csr32(&csr).unwrap();
        let row = topology.row(0).unwrap();
        assert_eq!(row.rank(64).unwrap(), 3);
        assert_eq!(row.rank(130).unwrap(), 6);
        for ordinal in 0..row.popcount() {
            let col = row.select(ordinal).unwrap();
            assert_eq!(row.rank(col).unwrap(), ordinal);
            assert!(row.contains(col));
        }
    }

    #[test]
    fn topology_boolean_algebra_matches_expected_support() {
        let a = Csr32Matrix::new(
            2,
            130,
            vec![0, 4, 7],
            vec![0, 2, 64, 129, 1, 65, 127],
            vec![1.0; 7],
        )
        .unwrap();
        let b = Csr32Matrix::new(
            2,
            130,
            vec![0, 4, 7],
            vec![2, 3, 64, 128, 1, 66, 127],
            vec![1.0; 7],
        )
        .unwrap();
        let a = AbtmTopology::from_csr32(&a).unwrap();
        let b = AbtmTopology::from_csr32(&b).unwrap();
        assert_eq!(row_indices(&a.intersection(&b).unwrap(), 0), vec![2, 64]);
        assert_eq!(
            row_indices(&a.union(&b).unwrap(), 0),
            vec![0, 2, 3, 64, 128, 129]
        );
        assert_eq!(row_indices(&a.and_not(&b).unwrap(), 0), vec![0, 129]);
        assert_eq!(row_indices(&a.xor(&b).unwrap(), 0), vec![0, 3, 128, 129]);
        assert_eq!(row_indices(&a.intersection(&b).unwrap(), 1), vec![1, 127]);
        assert_eq!(row_indices(&a.xor(&b).unwrap(), 1), vec![65, 66]);
    }

    #[test]
    fn topology_boolean_algebra_rejects_dimension_mismatch() {
        let a = AbtmTopology::from_csr32(
            &Csr32Matrix::new(1, 2, vec![0, 1], vec![0], vec![1.0]).unwrap(),
        )
        .unwrap();
        let b = AbtmTopology::from_csr32(
            &Csr32Matrix::new(1, 3, vec![0, 1], vec![0], vec![1.0]).unwrap(),
        )
        .unwrap();
        assert!(matches!(a.union(&b), Err(HybitError::InvalidArgument(_))));
    }

    #[test]
    fn topology_stats_use_twelve_bytes_per_nonempty_word_plus_row_ptrs() {
        let csr =
            Csr32Matrix::new(2, 130, vec![0, 3, 5], vec![0, 64, 129, 1, 65], vec![1.0; 5]).unwrap();
        let stats = AbtmTopology::from_csr32(&csr).unwrap().stats();
        assert_eq!(stats.structural_nnz, 5);
        assert_eq!(stats.nonempty_words, 5);
        assert_eq!(stats.metadata_bytes, 3 * 4 + 5 * (4 + 8));
    }
}
