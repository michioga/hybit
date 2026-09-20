use hybit_core::HybitError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DofMask {
    len: usize,
    words: Vec<u64>,
}

impl DofMask {
    pub fn new(len: usize) -> Self {
        Self {
            len,
            words: vec![0; (len + 63) / 64],
        }
    }

    pub fn from_indices(len: usize, indices: &[usize]) -> Result<Self, HybitError> {
        let mut mask = Self::new(len);
        for &index in indices {
            mask.set(index, true)?;
        }
        Ok(mask)
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.count_ones() == 0
    }
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub fn set(&mut self, index: usize, value: bool) -> Result<(), HybitError> {
        if index >= self.len {
            return Err(HybitError::InvalidArgument("DOF mask index out of range"));
        }
        let word = index / 64;
        let bit = index % 64;
        if value {
            self.words[word] |= 1u64 << bit;
        } else {
            self.words[word] &= !(1u64 << bit);
        }
        Ok(())
    }

    #[inline]
    pub fn contains(&self, index: usize) -> bool {
        if index >= self.len {
            return false;
        }
        (self.words[index / 64] & (1u64 << (index % 64))) != 0
    }

    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn union_assign(&mut self, other: &Self) -> Result<(), HybitError> {
        self.ensure_same_len(other)?;
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a |= *b;
        }
        Ok(())
    }

    pub fn intersect_assign(&mut self, other: &Self) -> Result<(), HybitError> {
        self.ensure_same_len(other)?;
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a &= *b;
        }
        Ok(())
    }

    pub fn intersects(&self, other: &Self) -> Result<bool, HybitError> {
        self.ensure_same_len(other)?;
        Ok(self
            .words
            .iter()
            .zip(&other.words)
            .any(|(a, b)| (a & b) != 0))
    }

    pub fn indices(&self) -> Vec<usize> {
        let mut result = Vec::with_capacity(self.count_ones());
        for (word_index, &word) in self.words.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let index = word_index * 64 + bit;
                if index < self.len {
                    result.push(index);
                }
                bits &= bits - 1;
            }
        }
        result
    }

    pub(crate) fn or_word(&mut self, word_index: usize, bits: u64) {
        if word_index < self.words.len() {
            self.words[word_index] |= bits;
            if word_index + 1 == self.words.len() && self.len % 64 != 0 {
                let valid = self.len % 64;
                self.words[word_index] &= (1u64 << valid) - 1;
            }
        }
    }

    fn ensure_same_len(&self, other: &Self) -> Result<(), HybitError> {
        if self.len != other.len {
            return Err(HybitError::DimensionMismatch {
                expected: self.len,
                actual: other.len,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_round_trip() {
        let mask = DofMask::from_indices(130, &[0, 1, 63, 64, 129]).unwrap();
        assert_eq!(mask.count_ones(), 5);
        assert_eq!(mask.indices(), vec![0, 1, 63, 64, 129]);
    }
}
