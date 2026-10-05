use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use hybit_matrix::{read_matrix_market, Csr32Matrix};

const WORD_BITS: usize = 64;
const WORD_INDEX_MASK: u32 = (1u32 << 26) - 1;
const SPARSE_COUNT_SHIFT: u32 = 26;
const SPARSE_COUNT_MASK: u32 = 0x0f;
const KIND_SHIFT: u32 = 30;

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    pairs_per_row: usize,
    repeats: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut pairs_per_row = 8usize;
        let mut repeats = 5usize;
        let mut it = env::args().skip(1);

        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => {
                    matrix = Some(PathBuf::from(
                        it.next().ok_or("missing value after --matrix")?,
                    ));
                }
                "--pairs-per-row" => {
                    pairs_per_row = it
                        .next()
                        .ok_or("missing value after --pairs-per-row")?
                        .parse()?;
                    if pairs_per_row == 0 {
                        return Err("--pairs-per-row must be >= 1".into());
                    }
                }
                "--repeats" => {
                    repeats = it.next().ok_or("missing value after --repeats")?.parse()?;
                    if repeats == 0 {
                        return Err("--repeats must be >= 1".into());
                    }
                }
                "-h" | "--help" => {
                    println!(
                        "Usage: abtm_typed_compact_dual_g2g --matrix A.mtx [--pairs-per-row N] [--repeats N]"
                    );
                    std::process::exit(0);
                }
                other if !other.starts_with('-') && matrix.is_none() => {
                    matrix = Some(PathBuf::from(other));
                }
                other => return Err(format!("unknown argument '{other}'").into()),
            }
        }

        Ok(Self {
            matrix: matrix.ok_or("missing matrix path; use --matrix FILE.mtx")?,
            pairs_per_row,
            repeats,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompactKind {
    Sparse = 0,
    Bitmap = 1,
    Dense = 2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct CompactTile {
    meta: u32,
    aux: u32,
}

impl CompactTile {
    fn new(
        word_index: usize,
        kind: CompactKind,
        sparse_count: usize,
        aux: usize,
    ) -> Result<Self, Box<dyn Error>> {
        if word_index > WORD_INDEX_MASK as usize || aux > u32::MAX as usize {
            return Err("compact tile index overflow".into());
        }
        if kind == CompactKind::Sparse && !(1..=15).contains(&sparse_count) {
            return Err("compact sparse tile count must fit four bits".into());
        }

        let count = if kind == CompactKind::Sparse {
            sparse_count as u32
        } else {
            0
        };
        Ok(Self {
            meta: word_index as u32 | (count << SPARSE_COUNT_SHIFT) | ((kind as u32) << KIND_SHIFT),
            aux: aux as u32,
        })
    }

    #[inline(always)]
    fn word_index(self) -> u32 {
        self.meta & WORD_INDEX_MASK
    }

    #[inline(always)]
    fn kind(self) -> CompactKind {
        match self.meta >> KIND_SHIFT {
            0 => CompactKind::Sparse,
            1 => CompactKind::Bitmap,
            2 => CompactKind::Dense,
            _ => unreachable!("invalid compact tile kind"),
        }
    }

    #[inline(always)]
    fn sparse_count(self) -> usize {
        ((self.meta >> SPARSE_COUNT_SHIFT) & SPARSE_COUNT_MASK) as usize
    }
}

#[derive(Clone, Copy, Debug)]
struct CompactConfig {
    sparse_max_nnz: usize,
    dense_min_nnz: usize,
}

impl Default for CompactConfig {
    fn default() -> Self {
        Self {
            sparse_max_nnz: 8,
            dense_min_nnz: 40,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CompactStats {
    tiles: usize,
    sparse_tiles: usize,
    bitmap_tiles: usize,
    dense_tiles: usize,
    metadata_bytes: usize,
    value_slots: usize,
}

impl CompactStats {
    fn total_bytes(self) -> usize {
        self.metadata_bytes
            .saturating_add(self.value_slots.saturating_mul(std::mem::size_of::<f64>()))
    }
}

#[derive(Clone, Debug)]
struct TypedCompactMatrix {
    nrows: usize,
    row_tile_ptr: Vec<u32>,
    row_value_ptr: Vec<u32>,
    tiles: Vec<CompactTile>,
    sparse_offsets: Vec<u8>,
    masks: Vec<u64>,
    values: Vec<f64>,
    sparse_tiles: usize,
    bitmap_tiles: usize,
    dense_tiles: usize,
}

impl TypedCompactMatrix {
    fn from_csr32(matrix: &Csr32Matrix, config: CompactConfig) -> Result<Self, Box<dyn Error>> {
        if config.sparse_max_nnz == 0
            || config.sparse_max_nnz > 15
            || config.sparse_max_nnz >= config.dense_min_nnz
            || config.dense_min_nnz > WORD_BITS
        {
            return Err("invalid typed compact thresholds".into());
        }

        let mut row_tile_ptr = Vec::with_capacity(matrix.nrows() + 1);
        let mut row_value_ptr = Vec::with_capacity(matrix.nrows() + 1);
        let mut tiles = Vec::new();
        let mut sparse_offsets = Vec::new();
        let mut masks = Vec::new();
        let mut values = Vec::new();
        let mut sparse_tiles = 0usize;
        let mut bitmap_tiles = 0usize;
        let mut dense_tiles = 0usize;

        row_tile_ptr.push(0);
        row_value_ptr.push(0);

        for row in 0..matrix.nrows() {
            let start = matrix.row_ptr()[row] as usize;
            let end = matrix.row_ptr()[row + 1] as usize;
            let mut blocks: BTreeMap<usize, BTreeMap<u8, f64>> = BTreeMap::new();

            for p in start..end {
                let value = matrix.values()[p];
                if value == 0.0 {
                    continue;
                }
                let col = matrix.col_idx()[p] as usize;
                let word = col / WORD_BITS;
                let offset = (col % WORD_BITS) as u8;
                *blocks.entry(word).or_default().entry(offset).or_default() += value;
            }

            for (word, entries) in blocks {
                let canonical: Vec<(u8, f64)> = entries
                    .into_iter()
                    .filter(|(_, value)| *value != 0.0)
                    .collect();
                if canonical.is_empty() {
                    continue;
                }

                let nnz = canonical.len();
                if nnz <= config.sparse_max_nnz {
                    let aux = sparse_offsets.len();
                    tiles.push(CompactTile::new(word, CompactKind::Sparse, nnz, aux)?);
                    sparse_offsets.extend(canonical.iter().map(|&(offset, _)| offset));
                    values.extend(canonical.iter().map(|&(_, value)| value));
                    sparse_tiles += 1;
                } else {
                    let mut mask = 0u64;
                    for &(offset, _) in &canonical {
                        mask |= 1u64 << offset;
                    }
                    let aux = masks.len();
                    if nnz >= config.dense_min_nnz {
                        tiles.push(CompactTile::new(word, CompactKind::Dense, 0, aux)?);
                        masks.push(mask);
                        let mut dense = [0.0f64; WORD_BITS];
                        for &(offset, value) in &canonical {
                            dense[offset as usize] = value;
                        }
                        values.extend_from_slice(&dense);
                        dense_tiles += 1;
                    } else {
                        tiles.push(CompactTile::new(word, CompactKind::Bitmap, 0, aux)?);
                        masks.push(mask);
                        values.extend(canonical.iter().map(|&(_, value)| value));
                        bitmap_tiles += 1;
                    }
                }
            }

            if tiles.len() > u32::MAX as usize || values.len() > u32::MAX as usize {
                return Err("typed compact stream exceeds u32 range".into());
            }
            row_tile_ptr.push(tiles.len() as u32);
            row_value_ptr.push(values.len() as u32);
        }

        Ok(Self {
            nrows: matrix.nrows(),
            row_tile_ptr,
            row_value_ptr,
            tiles,
            sparse_offsets,
            masks,
            values,
            sparse_tiles,
            bitmap_tiles,
            dense_tiles,
        })
    }

    fn stats(&self) -> CompactStats {
        CompactStats {
            tiles: self.tiles.len(),
            sparse_tiles: self.sparse_tiles,
            bitmap_tiles: self.bitmap_tiles,
            dense_tiles: self.dense_tiles,
            metadata_bytes: std::mem::size_of_val(self.row_tile_ptr.as_slice())
                + std::mem::size_of_val(self.row_value_ptr.as_slice())
                + std::mem::size_of_val(self.tiles.as_slice())
                + std::mem::size_of_val(self.sparse_offsets.as_slice())
                + std::mem::size_of_val(self.masks.as_slice()),
            value_slots: self.values.len(),
        }
    }

    #[inline(always)]
    fn row_cursor(&self, row: usize) -> CompactRowCursor<'_> {
        debug_assert!(row < self.nrows);
        CompactRowCursor {
            matrix: self,
            tile_index: self.row_tile_ptr[row] as usize,
            tile_end: self.row_tile_ptr[row + 1] as usize,
            value_index: self.row_value_ptr[row] as usize,
        }
    }
}

#[derive(Clone, Copy)]
struct CompactTileView<'a> {
    word_index: u32,
    kind: CompactKind,
    sparse_offsets: &'a [u8],
    mask: u64,
    values: &'a [f64],
}

struct CompactRowCursor<'a> {
    matrix: &'a TypedCompactMatrix,
    tile_index: usize,
    tile_end: usize,
    value_index: usize,
}

impl<'a> CompactRowCursor<'a> {
    #[inline(always)]
    fn next_tile(&mut self) -> Option<CompactTileView<'a>> {
        if self.tile_index >= self.tile_end {
            return None;
        }

        let desc = self.matrix.tiles[self.tile_index];
        self.tile_index += 1;

        match desc.kind() {
            CompactKind::Sparse => {
                let count = desc.sparse_count();
                let offset_start = desc.aux as usize;
                let offset_end = offset_start + count;
                let value_end = self.value_index + count;
                let view = CompactTileView {
                    word_index: desc.word_index(),
                    kind: CompactKind::Sparse,
                    sparse_offsets: &self.matrix.sparse_offsets[offset_start..offset_end],
                    mask: 0,
                    values: &self.matrix.values[self.value_index..value_end],
                };
                self.value_index = value_end;
                Some(view)
            }
            CompactKind::Bitmap => {
                let mask = self.matrix.masks[desc.aux as usize];
                let count = mask.count_ones() as usize;
                let value_end = self.value_index + count;
                let view = CompactTileView {
                    word_index: desc.word_index(),
                    kind: CompactKind::Bitmap,
                    sparse_offsets: &[],
                    mask,
                    values: &self.matrix.values[self.value_index..value_end],
                };
                self.value_index = value_end;
                Some(view)
            }
            CompactKind::Dense => {
                let mask = self.matrix.masks[desc.aux as usize];
                let value_end = self.value_index + WORD_BITS;
                let view = CompactTileView {
                    word_index: desc.word_index(),
                    kind: CompactKind::Dense,
                    sparse_offsets: &[],
                    mask,
                    values: &self.matrix.values[self.value_index..value_end],
                };
                self.value_index = value_end;
                Some(view)
            }
        }
    }
}

fn transpose_csr(matrix: &Csr32Matrix) -> Result<Csr32Matrix, Box<dyn Error>> {
    let mut counts = vec![0u32; matrix.ncols()];
    for &col in matrix.col_idx() {
        let count = &mut counts[col as usize];
        *count = count.checked_add(1).ok_or("transpose count overflow")?;
    }

    let mut row_ptr = Vec::with_capacity(matrix.ncols() + 1);
    row_ptr.push(0u32);
    let mut running = 0u32;
    for count in counts {
        running = running
            .checked_add(count)
            .ok_or("transpose row pointer overflow")?;
        row_ptr.push(running);
    }

    let mut col_idx = vec![0u32; matrix.nnz()];
    let mut values = vec![0.0f64; matrix.nnz()];
    let mut cursors = row_ptr[..matrix.ncols()].to_vec();

    for row in 0..matrix.nrows() {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        for p in start..end {
            let col = matrix.col_idx()[p] as usize;
            let dst = cursors[col] as usize;
            col_idx[dst] = u32::try_from(row).map_err(|_| "transpose row index overflow")?;
            values[dst] = matrix.values()[p];
            cursors[col] = cursors[col]
                .checked_add(1)
                .ok_or("transpose cursor overflow")?;
        }
    }

    Ok(Csr32Matrix::new(
        matrix.ncols(),
        matrix.nrows(),
        row_ptr,
        col_idx,
        values,
    )?)
}

fn explicit_numeric_dot(
    row: usize,
    col: usize,
    matrix: &Csr32Matrix,
    transpose: &Csr32Matrix,
) -> f64 {
    let a_start = matrix.row_ptr()[row] as usize;
    let a_end = matrix.row_ptr()[row + 1] as usize;
    let b_start = transpose.row_ptr()[col] as usize;
    let b_end = transpose.row_ptr()[col + 1] as usize;

    let a_idx = &matrix.col_idx()[a_start..a_end];
    let a_val = &matrix.values()[a_start..a_end];
    let b_idx = &transpose.col_idx()[b_start..b_end];
    let b_val = &transpose.values()[b_start..b_end];

    let mut i = 0usize;
    let mut j = 0usize;
    let mut sum = 0.0;

    while i < a_idx.len() && j < b_idx.len() {
        if a_idx[i] < b_idx[j] {
            i += 1;
        } else if b_idx[j] < a_idx[i] {
            j += 1;
        } else {
            sum += a_val[i] * b_val[j];
            i += 1;
            j += 1;
        }
    }

    sum
}

#[inline(always)]
fn compact_rank(mask: u64, bit: usize) -> usize {
    if bit == 0 {
        0
    } else {
        (mask & ((1u64 << bit) - 1)).count_ones() as usize
    }
}

#[inline(always)]
fn bitmap_or_dense_value(tile: CompactTileView<'_>, bit: usize) -> f64 {
    match tile.kind {
        CompactKind::Dense => tile.values[bit],
        CompactKind::Bitmap => tile.values[compact_rank(tile.mask, bit)],
        CompactKind::Sparse => unreachable!("sparse lookup uses offset merge"),
    }
}

#[inline(always)]
fn sparse_sparse_dot(a: CompactTileView<'_>, b: CompactTileView<'_>) -> f64 {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut sum = 0.0;

    while i < a.sparse_offsets.len() && j < b.sparse_offsets.len() {
        if a.sparse_offsets[i] < b.sparse_offsets[j] {
            i += 1;
        } else if b.sparse_offsets[j] < a.sparse_offsets[i] {
            j += 1;
        } else {
            sum += a.values[i] * b.values[j];
            i += 1;
            j += 1;
        }
    }

    sum
}

#[inline(always)]
fn sparse_other_dot(sparse: CompactTileView<'_>, other: CompactTileView<'_>) -> f64 {
    let mut sum = 0.0;

    for (ordinal, &offset) in sparse.sparse_offsets.iter().enumerate() {
        let bit = offset as usize;
        if (other.mask & (1u64 << bit)) != 0 {
            sum += sparse.values[ordinal] * bitmap_or_dense_value(other, bit);
        }
    }

    sum
}

#[inline(always)]
fn bitmap_dense_dot(a: CompactTileView<'_>, b: CompactTileView<'_>) -> f64 {
    let mut active = a.mask & b.mask;
    let mut sum = 0.0;

    while active != 0 {
        let bit = active.trailing_zeros() as usize;
        sum += bitmap_or_dense_value(a, bit) * bitmap_or_dense_value(b, bit);
        active &= active - 1;
    }

    sum
}

#[inline(always)]
fn matched_tile_dot(a: CompactTileView<'_>, b: CompactTileView<'_>) -> f64 {
    match (a.kind, b.kind) {
        (CompactKind::Sparse, CompactKind::Sparse) => sparse_sparse_dot(a, b),
        (CompactKind::Sparse, _) => sparse_other_dot(a, b),
        (_, CompactKind::Sparse) => sparse_other_dot(b, a),
        _ => bitmap_dense_dot(a, b),
    }
}

fn typed_compact_numeric_dot(
    row: usize,
    col: usize,
    rows: &TypedCompactMatrix,
    columns: &TypedCompactMatrix,
) -> f64 {
    let mut a = rows.row_cursor(row);
    let mut b = columns.row_cursor(col);
    let mut aw = a.next_tile();
    let mut bw = b.next_tile();
    let mut sum = 0.0;

    while let (Some(a_tile), Some(b_tile)) = (aw, bw) {
        if a_tile.word_index < b_tile.word_index {
            aw = a.next_tile();
        } else if b_tile.word_index < a_tile.word_index {
            bw = b.next_tile();
        } else {
            sum += matched_tile_dot(a_tile, b_tile);
            aw = a.next_tile();
            bw = b.next_tile();
        }
    }

    sum
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn make_pairs(n: usize, pairs_per_row: usize) -> Vec<(u32, u32)> {
    let mut pairs = Vec::with_capacity(n.saturating_mul(pairs_per_row));

    for row in 0..n {
        for slot in 0..pairs_per_row {
            let col = match slot {
                0 => row,
                1 => (row + 1) % n,
                _ => {
                    let key = (row as u64)
                        ^ (slot as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        ^ 0x38d0_2026_1005_0300;
                    (mix(key) % n as u64) as usize
                }
            };
            pairs.push((row as u32, col as u32));
        }
    }

    pairs
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

fn scaled_error(reference: f64, actual: f64) -> f64 {
    (actual - reference).abs() / reference.abs().max(1.0)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;

    println!(
        "HyBIT {} ABTM G2g typed-compact adaptive dual-numeric sparse dot",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!("pairs per row       : {}", args.pairs_per_row);
    println!("repeats             : {}", args.repeats);

    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    if matrix.nrows() != matrix.ncols() {
        return Err("G2g same-matrix A*A benchmark requires a square matrix".into());
    }
    let transpose = transpose_csr(&matrix)?;

    println!(
        "Matrix Market       : {:?}, {} input entries -> {} CSR nnz",
        mm.symmetry, mm.input_entries, mm.csr_nnz
    );
    println!(
        "dimensions          : {} x {}",
        matrix.nrows(),
        matrix.ncols()
    );
    println!("stored CSR nnz      : {}", matrix.nnz());

    let config = CompactConfig::default();

    let row_start = Instant::now();
    let rows = TypedCompactMatrix::from_csr32(&matrix, config)?;
    let row_prepare_ms = row_start.elapsed().as_secs_f64() * 1.0e3;

    let col_start = Instant::now();
    let columns = TypedCompactMatrix::from_csr32(&transpose, config)?;
    let col_prepare_ms = col_start.elapsed().as_secs_f64() * 1.0e3;

    let row_stats = rows.stats();
    let col_stats = columns.stats();
    let compact_dual_bytes = row_stats
        .total_bytes()
        .saturating_add(col_stats.total_bytes());
    let explicit_dual_bytes = matrix
        .storage_bytes()
        .saturating_add(transpose.storage_bytes());
    let compact_over_explicit_storage =
        compact_dual_bytes as f64 / explicit_dual_bytes.max(1) as f64;

    println!(
        "G2G_PREPARE|tile_desc_bytes={}|sparse_max_nnz={}|dense_min_nnz={}|row_tiles={}|row_sparse={}|row_bitmap={}|row_dense={}|col_tiles={}|col_sparse={}|col_bitmap={}|col_dense={}|row_metadata_bytes={}|col_metadata_bytes={}|row_value_slots={}|col_value_slots={}|compact_dual_bytes={compact_dual_bytes}|explicit_dual_bytes={explicit_dual_bytes}|compact_over_explicit_storage={compact_over_explicit_storage:.9e}|row_prepare_ms={row_prepare_ms:.6}|col_prepare_ms={col_prepare_ms:.6}",
        std::mem::size_of::<CompactTile>(),
        config.sparse_max_nnz,
        config.dense_min_nnz,
        row_stats.tiles,
        row_stats.sparse_tiles,
        row_stats.bitmap_tiles,
        row_stats.dense_tiles,
        col_stats.tiles,
        col_stats.sparse_tiles,
        col_stats.bitmap_tiles,
        col_stats.dense_tiles,
        row_stats.metadata_bytes,
        col_stats.metadata_bytes,
        row_stats.value_slots,
        col_stats.value_slots,
    );

    let pairs = make_pairs(matrix.nrows(), args.pairs_per_row);

    let mut max_scaled_error = 0.0f64;
    for &(row, col) in &pairs {
        let row = row as usize;
        let col = col as usize;
        let reference = explicit_numeric_dot(row, col, &matrix, &transpose);
        let actual = typed_compact_numeric_dot(row, col, &rows, &columns);
        max_scaled_error = max_scaled_error.max(scaled_error(reference, actual));
    }

    if max_scaled_error > 1.0e-10 {
        return Err(
            format!("G2g numerical mismatch: max_scaled_error={max_scaled_error:.3e}").into(),
        );
    }

    let mut explicit_samples = Vec::with_capacity(args.repeats);
    let mut compact_samples = Vec::with_capacity(args.repeats);

    for _ in 0..args.repeats {
        let start = Instant::now();
        let mut checksum = 0.0;
        for &(row, col) in &pairs {
            checksum += explicit_numeric_dot(row as usize, col as usize, &matrix, &transpose);
        }
        explicit_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box(checksum);

        let start = Instant::now();
        let mut checksum = 0.0;
        for &(row, col) in &pairs {
            checksum += typed_compact_numeric_dot(row as usize, col as usize, &rows, &columns);
        }
        compact_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box(checksum);
    }

    let explicit_ms = median(&mut explicit_samples);
    let compact_ms = median(&mut compact_samples);
    let compact_over_explicit = if explicit_ms == 0.0 {
        0.0
    } else {
        compact_ms / explicit_ms
    };

    println!(
        "G2G_NUMERIC|pairs={}|explicit_ms={explicit_ms:.6}|compact_ms={compact_ms:.6}|compact_over_explicit={compact_over_explicit:.9e}|max_scaled_error={max_scaled_error:.9e}",
        pairs.len(),
    );

    Ok(())
}
