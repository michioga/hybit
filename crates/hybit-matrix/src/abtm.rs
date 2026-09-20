use std::collections::BTreeMap;

use crate::{Csr32Matrix, DofMask};
use hybit_core::{HybitError, LinearOperator};

pub const TILE_WIDTH: usize = 64;
const KIND_SHIFT: u32 = 30;
const BASE_WORD_MASK: u32 = (1u32 << KIND_SHIFT) - 1;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileKind {
    Sparse = 0,
    Bitmap = 1,
    Dense = 2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TileDesc {
    pub mask: u64,
    pub value_offset: u32,
    meta: u32,
}

impl TileDesc {
    fn new(
        base_col: usize,
        value_offset: usize,
        mask: u64,
        kind: TileKind,
    ) -> Result<Self, HybitError> {
        let word = base_col / TILE_WIDTH;
        if word > BASE_WORD_MASK as usize || value_offset > u32::MAX as usize {
            return Err(HybitError::SizeOverflow);
        }
        Ok(Self {
            mask,
            value_offset: value_offset as u32,
            meta: word as u32 | ((kind as u32) << KIND_SHIFT),
        })
    }

    #[inline(always)]
    pub fn kind(&self) -> TileKind {
        match self.meta >> KIND_SHIFT {
            0 => TileKind::Sparse,
            1 => TileKind::Bitmap,
            2 => TileKind::Dense,
            _ => unreachable!("invalid tile kind"),
        }
    }

    #[inline(always)]
    pub fn base_col(&self) -> usize {
        ((self.meta & BASE_WORD_MASK) as usize) * TILE_WIDTH
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AbtmConfig {
    pub sparse_max_nnz: u8,
    pub dense_min_nnz: u8,
}

impl Default for AbtmConfig {
    fn default() -> Self {
        Self {
            sparse_max_nnz: 8,
            dense_min_nnz: 40,
        }
    }
}

impl AbtmConfig {
    fn validate(self) -> Result<Self, HybitError> {
        if self.sparse_max_nnz == 0
            || self.dense_min_nnz as usize > TILE_WIDTH
            || self.sparse_max_nnz >= self.dense_min_nnz
        {
            return Err(HybitError::InvalidArgument(
                "invalid ABTM density thresholds",
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, Default)]
pub struct AbtmStats {
    pub nrows: usize,
    pub ncols: usize,
    pub matrix_nnz: usize,
    pub tiles: usize,
    pub sparse_tiles: usize,
    pub bitmap_tiles: usize,
    pub dense_tiles: usize,
    pub value_slots: usize,
    pub metadata_bytes: usize,
}

impl AbtmStats {
    pub fn metadata_bytes_per_nnz(&self) -> f64 {
        if self.matrix_nnz == 0 {
            0.0
        } else {
            self.metadata_bytes as f64 / self.matrix_nnz as f64
        }
    }
    pub fn total_estimated_bytes(&self) -> usize {
        self.metadata_bytes + self.value_slots * std::mem::size_of::<f64>()
    }
}

#[derive(Clone, Debug)]
pub struct AbtmMatrix {
    nrows: usize,
    ncols: usize,
    row_tile_ptr: Vec<u32>,
    tiles: Vec<TileDesc>,
    values: Vec<f64>,
    matrix_nnz: usize,
}

impl AbtmMatrix {
    pub fn from_csr32(csr: &Csr32Matrix, config: AbtmConfig) -> Result<Self, HybitError> {
        csr.validate()?;
        let config = config.validate()?;
        let mut row_tile_ptr = Vec::with_capacity(csr.nrows() + 1);
        let mut tiles = Vec::new();
        let mut values = Vec::new();
        let mut matrix_nnz = 0usize;
        row_tile_ptr.push(0);

        for row in 0..csr.nrows() {
            let start = csr.row_ptr()[row] as usize;
            let end = csr.row_ptr()[row + 1] as usize;
            let mut blocks: BTreeMap<usize, BTreeMap<u8, f64>> = BTreeMap::new();
            for p in start..end {
                let col = csr.col_idx()[p] as usize;
                let value = csr.values()[p];
                if value == 0.0 {
                    continue;
                }
                let block = col / TILE_WIDTH;
                let offset = (col % TILE_WIDTH) as u8;
                *blocks.entry(block).or_default().entry(offset).or_default() += value;
            }

            for (block, entries) in blocks {
                let canonical: Vec<(u8, f64)> =
                    entries.into_iter().filter(|(_, v)| *v != 0.0).collect();
                if canonical.is_empty() {
                    continue;
                }
                let nnz = canonical.len();
                matrix_nnz += nnz;
                let base_col = block
                    .checked_mul(TILE_WIDTH)
                    .ok_or(HybitError::SizeOverflow)?;
                let value_offset = values.len();
                let mut mask = 0u64;
                for &(offset, _) in &canonical {
                    mask |= 1u64 << offset;
                }
                let kind = if nnz <= config.sparse_max_nnz as usize {
                    TileKind::Sparse
                } else if nnz >= config.dense_min_nnz as usize {
                    TileKind::Dense
                } else {
                    TileKind::Bitmap
                };
                match kind {
                    TileKind::Sparse | TileKind::Bitmap => {
                        values.extend(canonical.iter().map(|&(_, value)| value));
                    }
                    TileKind::Dense => {
                        let mut dense = [0.0f64; TILE_WIDTH];
                        for &(offset, value) in &canonical {
                            dense[offset as usize] = value;
                        }
                        values.extend_from_slice(&dense);
                    }
                }
                tiles.push(TileDesc::new(base_col, value_offset, mask, kind)?);
            }
            if tiles.len() > u32::MAX as usize {
                return Err(HybitError::SizeOverflow);
            }
            row_tile_ptr.push(tiles.len() as u32);
        }

        Ok(Self {
            nrows: csr.nrows(),
            ncols: csr.ncols(),
            row_tile_ptr,
            tiles,
            values,
            matrix_nnz,
        })
    }

    pub fn expand_mask_one_hop(&self, mask: &DofMask) -> Result<DofMask, HybitError> {
        if self.nrows != self.ncols {
            return Err(HybitError::InvalidMatrix(
                "mask expansion requires a square ABTM matrix",
            ));
        }
        if mask.len() != self.nrows {
            return Err(HybitError::DimensionMismatch {
                expected: self.nrows,
                actual: mask.len(),
            });
        }
        let mut expanded = mask.clone();
        for row in mask.indices() {
            let start = self.row_tile_ptr[row] as usize;
            let end = self.row_tile_ptr[row + 1] as usize;
            for tile in &self.tiles[start..end] {
                expanded.or_word(tile.base_col() / TILE_WIDTH, tile.mask);
            }
        }
        Ok(expanded)
    }

    pub fn stats(&self) -> AbtmStats {
        let mut stats = AbtmStats {
            nrows: self.nrows,
            ncols: self.ncols,
            matrix_nnz: self.matrix_nnz,
            tiles: self.tiles.len(),
            value_slots: self.values.len(),
            metadata_bytes: self.row_tile_ptr.len() * std::mem::size_of::<u32>()
                + self.tiles.len() * std::mem::size_of::<TileDesc>(),
            ..AbtmStats::default()
        };
        for tile in &self.tiles {
            match tile.kind() {
                TileKind::Sparse => stats.sparse_tiles += 1,
                TileKind::Bitmap => stats.bitmap_tiles += 1,
                TileKind::Dense => stats.dense_tiles += 1,
            }
        }
        stats
    }

    #[inline(always)]
    fn apply_compact_tile(&self, tile: &TileDesc, x: &[f64]) -> f64 {
        let mut bits = tile.mask;
        let mut p = tile.value_offset as usize;
        let base = tile.base_col();
        let mut sum = 0.0;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            sum += self.values[p] * x[base + bit];
            p += 1;
            bits &= bits - 1;
        }
        sum
    }
}

impl LinearOperator for AbtmMatrix {
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
        for row in 0..self.nrows {
            let start = self.row_tile_ptr[row] as usize;
            let end = self.row_tile_ptr[row + 1] as usize;
            let mut sum = 0.0;
            for tile in &self.tiles[start..end] {
                match tile.kind() {
                    TileKind::Sparse | TileKind::Bitmap => sum += self.apply_compact_tile(tile, x),
                    TileKind::Dense => {
                        let base = tile.base_col();
                        let vo = tile.value_offset as usize;
                        let limit = (self.ncols - base).min(TILE_WIDTH);
                        for j in 0..limit {
                            sum += self.values[vo + j] * x[base + j];
                        }
                    }
                }
            }
            y[row] = sum;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_descriptor_is_16_bytes() {
        assert_eq!(std::mem::size_of::<TileDesc>(), 16);
    }

    #[test]
    fn bitmap_topology_expands_one_hop() {
        let csr = Csr32Matrix::new(
            4,
            4,
            vec![0, 2, 5, 8, 10],
            vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
            vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 3.0],
        )
        .unwrap();
        let abtm = AbtmMatrix::from_csr32(&csr, AbtmConfig::default()).unwrap();
        let seed = DofMask::from_indices(4, &[1]).unwrap();
        let expanded = abtm.expand_mask_one_hop(&seed).unwrap();
        assert_eq!(expanded.indices(), vec![0, 1, 2]);
    }

    #[test]
    fn abtm_matches_csr() {
        let csr = Csr32Matrix::new(
            4,
            4,
            vec![0, 2, 5, 8, 10],
            vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
            vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 3.0],
        )
        .unwrap();
        let abtm = AbtmMatrix::from_csr32(&csr, AbtmConfig::default()).unwrap();
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut yc = vec![0.0; 4];
        let mut ya = vec![0.0; 4];
        csr.apply(&x, &mut yc).unwrap();
        abtm.apply(&x, &mut ya).unwrap();
        for (a, b) in yc.iter().zip(&ya) {
            assert!((a - b).abs() < 1.0e-12);
        }
    }
}
