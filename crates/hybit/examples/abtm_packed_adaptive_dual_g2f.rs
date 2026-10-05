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
                        "Usage: abtm_packed_adaptive_dual_g2f --matrix A.mtx [--pairs-per-row N] [--repeats N]"
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
enum PackedKind {
    Sparse = 0,
    Bitmap = 1,
    Dense = 2,
}

impl PackedKind {
    fn from_meta(meta: u32) -> Result<Self, Box<dyn Error>> {
        match meta >> KIND_SHIFT {
            0 => Ok(Self::Sparse),
            1 => Ok(Self::Bitmap),
            2 => Ok(Self::Dense),
            _ => Err("invalid packed tile kind".into()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PackedConfig {
    sparse_max_nnz: usize,
    dense_min_nnz: usize,
}

impl Default for PackedConfig {
    fn default() -> Self {
        Self {
            sparse_max_nnz: 8,
            dense_min_nnz: 40,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PackedStats {
    tiles: usize,
    sparse_tiles: usize,
    bitmap_tiles: usize,
    dense_tiles: usize,
    metadata_bytes: usize,
    value_slots: usize,
}

impl PackedStats {
    fn total_bytes(self) -> usize {
        self.metadata_bytes
            .saturating_add(self.value_slots.saturating_mul(std::mem::size_of::<f64>()))
    }
}

#[derive(Clone, Debug)]
struct PackedAdaptiveMatrix {
    nrows: usize,
    row_tile_ptr: Vec<u32>,
    row_payload_ptr: Vec<u32>,
    row_value_ptr: Vec<u32>,
    tile_meta: Vec<u32>,
    payload: Vec<u8>,
    values: Vec<f64>,
    sparse_tiles: usize,
    bitmap_tiles: usize,
    dense_tiles: usize,
}

impl PackedAdaptiveMatrix {
    fn from_csr32(matrix: &Csr32Matrix, config: PackedConfig) -> Result<Self, Box<dyn Error>> {
        if config.sparse_max_nnz == 0
            || config.sparse_max_nnz > 15
            || config.sparse_max_nnz >= config.dense_min_nnz
            || config.dense_min_nnz > WORD_BITS
        {
            return Err("invalid packed adaptive thresholds".into());
        }

        let mut row_tile_ptr = Vec::with_capacity(matrix.nrows() + 1);
        let mut row_payload_ptr = Vec::with_capacity(matrix.nrows() + 1);
        let mut row_value_ptr = Vec::with_capacity(matrix.nrows() + 1);
        let mut tile_meta = Vec::new();
        let mut payload = Vec::new();
        let mut values = Vec::new();
        let mut sparse_tiles = 0usize;
        let mut bitmap_tiles = 0usize;
        let mut dense_tiles = 0usize;

        row_tile_ptr.push(0);
        row_payload_ptr.push(0);
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
                if word > WORD_INDEX_MASK as usize {
                    return Err("packed word index exceeds 26-bit representation".into());
                }

                let canonical: Vec<(u8, f64)> = entries
                    .into_iter()
                    .filter(|(_, value)| *value != 0.0)
                    .collect();
                if canonical.is_empty() {
                    continue;
                }

                let nnz = canonical.len();
                let kind = if nnz <= config.sparse_max_nnz {
                    PackedKind::Sparse
                } else if nnz >= config.dense_min_nnz {
                    PackedKind::Dense
                } else {
                    PackedKind::Bitmap
                };

                let count_bits = if kind == PackedKind::Sparse {
                    u32::try_from(nnz).map_err(|_| "sparse tile count overflow")?
                } else {
                    0
                };
                let meta = (word as u32)
                    | (count_bits << SPARSE_COUNT_SHIFT)
                    | ((kind as u32) << KIND_SHIFT);
                tile_meta.push(meta);

                match kind {
                    PackedKind::Sparse => {
                        sparse_tiles += 1;
                        payload.extend(canonical.iter().map(|&(offset, _)| offset));
                        values.extend(canonical.iter().map(|&(_, value)| value));
                    }
                    PackedKind::Bitmap => {
                        bitmap_tiles += 1;
                        let mut mask = 0u64;
                        for &(offset, _) in &canonical {
                            mask |= 1u64 << offset;
                        }
                        payload.extend_from_slice(&mask.to_le_bytes());
                        values.extend(canonical.iter().map(|&(_, value)| value));
                    }
                    PackedKind::Dense => {
                        dense_tiles += 1;
                        let mut mask = 0u64;
                        let mut dense = [0.0f64; WORD_BITS];
                        for &(offset, value) in &canonical {
                            mask |= 1u64 << offset;
                            dense[offset as usize] = value;
                        }
                        payload.extend_from_slice(&mask.to_le_bytes());
                        values.extend_from_slice(&dense);
                    }
                }
            }

            if tile_meta.len() > u32::MAX as usize
                || payload.len() > u32::MAX as usize
                || values.len() > u32::MAX as usize
            {
                return Err("packed adaptive stream exceeds u32 range".into());
            }
            row_tile_ptr.push(tile_meta.len() as u32);
            row_payload_ptr.push(payload.len() as u32);
            row_value_ptr.push(values.len() as u32);
        }

        Ok(Self {
            nrows: matrix.nrows(),
            row_tile_ptr,
            row_payload_ptr,
            row_value_ptr,
            tile_meta,
            payload,
            values,
            sparse_tiles,
            bitmap_tiles,
            dense_tiles,
        })
    }

    fn stats(&self) -> PackedStats {
        PackedStats {
            tiles: self.tile_meta.len(),
            sparse_tiles: self.sparse_tiles,
            bitmap_tiles: self.bitmap_tiles,
            dense_tiles: self.dense_tiles,
            metadata_bytes: std::mem::size_of_val(self.row_tile_ptr.as_slice())
                + std::mem::size_of_val(self.row_payload_ptr.as_slice())
                + std::mem::size_of_val(self.row_value_ptr.as_slice())
                + std::mem::size_of_val(self.tile_meta.as_slice())
                + std::mem::size_of_val(self.payload.as_slice()),
            value_slots: self.values.len(),
        }
    }

    fn row_iter(&self, row: usize) -> Result<PackedRowIter<'_>, Box<dyn Error>> {
        if row >= self.nrows {
            return Err("packed adaptive row index out of range".into());
        }

        Ok(PackedRowIter {
            matrix: self,
            tile_index: self.row_tile_ptr[row] as usize,
            tile_end: self.row_tile_ptr[row + 1] as usize,
            payload_index: self.row_payload_ptr[row] as usize,
            value_index: self.row_value_ptr[row] as usize,
        })
    }
}

#[derive(Clone, Copy)]
struct PackedTileView<'a> {
    word_index: u32,
    kind: PackedKind,
    sparse_offsets: &'a [u8],
    mask: u64,
    values: &'a [f64],
}

impl PackedTileView<'_> {
    #[inline(always)]
    fn word_index(self) -> u32 {
        self.word_index
    }
}

struct PackedRowIter<'a> {
    matrix: &'a PackedAdaptiveMatrix,
    tile_index: usize,
    tile_end: usize,
    payload_index: usize,
    value_index: usize,
}

impl<'a> Iterator for PackedRowIter<'a> {
    type Item = Result<PackedTileView<'a>, Box<dyn Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.tile_index >= self.tile_end {
            return None;
        }

        let meta = self.matrix.tile_meta[self.tile_index];
        self.tile_index += 1;

        let word_index = meta & WORD_INDEX_MASK;
        let kind = match PackedKind::from_meta(meta) {
            Ok(kind) => kind,
            Err(error) => return Some(Err(error)),
        };

        let item = match kind {
            PackedKind::Sparse => {
                let count = ((meta >> SPARSE_COUNT_SHIFT) & SPARSE_COUNT_MASK) as usize;
                if count == 0 {
                    return Some(Err("packed sparse tile has zero count".into()));
                }
                let payload_end = self.payload_index.saturating_add(count);
                let value_end = self.value_index.saturating_add(count);
                if payload_end > self.matrix.payload.len() || value_end > self.matrix.values.len() {
                    return Some(Err("packed sparse tile stream is truncated".into()));
                }
                let offsets = &self.matrix.payload[self.payload_index..payload_end];
                let values = &self.matrix.values[self.value_index..value_end];
                self.payload_index = payload_end;
                self.value_index = value_end;
                PackedTileView {
                    word_index,
                    kind,
                    sparse_offsets: offsets,
                    mask: 0,
                    values,
                }
            }
            PackedKind::Bitmap => {
                let payload_end = self.payload_index.saturating_add(8);
                if payload_end > self.matrix.payload.len() {
                    return Some(Err("packed bitmap mask stream is truncated".into()));
                }
                let bytes: [u8; 8] =
                    match self.matrix.payload[self.payload_index..payload_end].try_into() {
                        Ok(bytes) => bytes,
                        Err(_) => return Some(Err("invalid bitmap payload width".into())),
                    };
                let mask = u64::from_le_bytes(bytes);
                let count = mask.count_ones() as usize;
                let value_end = self.value_index.saturating_add(count);
                if value_end > self.matrix.values.len() {
                    return Some(Err("packed bitmap value stream is truncated".into()));
                }
                let values = &self.matrix.values[self.value_index..value_end];
                self.payload_index = payload_end;
                self.value_index = value_end;
                PackedTileView {
                    word_index,
                    kind,
                    sparse_offsets: &[],
                    mask,
                    values,
                }
            }
            PackedKind::Dense => {
                let payload_end = self.payload_index.saturating_add(8);
                if payload_end > self.matrix.payload.len() {
                    return Some(Err("packed dense mask stream is truncated".into()));
                }
                let bytes: [u8; 8] =
                    match self.matrix.payload[self.payload_index..payload_end].try_into() {
                        Ok(bytes) => bytes,
                        Err(_) => return Some(Err("invalid dense payload width".into())),
                    };
                let mask = u64::from_le_bytes(bytes);
                let value_end = self.value_index.saturating_add(WORD_BITS);
                if value_end > self.matrix.values.len() {
                    return Some(Err("packed dense value stream is truncated".into()));
                }
                let values = &self.matrix.values[self.value_index..value_end];
                self.payload_index = payload_end;
                self.value_index = value_end;
                PackedTileView {
                    word_index,
                    kind,
                    sparse_offsets: &[],
                    mask,
                    values,
                }
            }
        };

        Some(Ok(item))
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
fn bitmap_or_dense_value(tile: PackedTileView<'_>, bit: usize) -> f64 {
    match tile.kind {
        PackedKind::Dense => tile.values[bit],
        PackedKind::Bitmap => tile.values[compact_rank(tile.mask, bit)],
        PackedKind::Sparse => unreachable!("sparse lookup uses offset merge"),
    }
}

#[inline(always)]
fn sparse_sparse_dot(a: PackedTileView<'_>, b: PackedTileView<'_>) -> f64 {
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
fn sparse_other_dot(sparse: PackedTileView<'_>, other: PackedTileView<'_>) -> f64 {
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
fn bitmap_dense_dot(a: PackedTileView<'_>, b: PackedTileView<'_>) -> f64 {
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
fn matched_tile_dot(a: PackedTileView<'_>, b: PackedTileView<'_>) -> f64 {
    match (a.kind, b.kind) {
        (PackedKind::Sparse, PackedKind::Sparse) => sparse_sparse_dot(a, b),
        (PackedKind::Sparse, _) => sparse_other_dot(a, b),
        (_, PackedKind::Sparse) => sparse_other_dot(b, a),
        _ => bitmap_dense_dot(a, b),
    }
}

fn packed_numeric_dot(
    row: usize,
    col: usize,
    rows: &PackedAdaptiveMatrix,
    columns: &PackedAdaptiveMatrix,
) -> Result<f64, Box<dyn Error>> {
    let mut a = rows.row_iter(row)?.peekable();
    let mut b = columns.row_iter(col)?.peekable();
    let mut sum = 0.0;

    loop {
        let aw = match a.peek() {
            Some(Ok(tile)) => *tile,
            Some(Err(_)) => {
                return a.next().expect("peeked item disappeared").map(|_| 0.0);
            }
            None => break,
        };
        let bw = match b.peek() {
            Some(Ok(tile)) => *tile,
            Some(Err(_)) => {
                return b.next().expect("peeked item disappeared").map(|_| 0.0);
            }
            None => break,
        };

        if aw.word_index() < bw.word_index() {
            a.next();
        } else if bw.word_index() < aw.word_index() {
            b.next();
        } else {
            let a_tile = a.next().expect("peeked item disappeared")?;
            let b_tile = b.next().expect("peeked item disappeared")?;
            sum += matched_tile_dot(a_tile, b_tile);
        }
    }

    Ok(sum)
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
                        ^ 0x38d0_2026_1005_02f0;
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
        "HyBIT {} ABTM G2f packed adaptive dual-numeric sparse dot",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!("pairs per row       : {}", args.pairs_per_row);
    println!("repeats             : {}", args.repeats);

    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    if matrix.nrows() != matrix.ncols() {
        return Err("G2f same-matrix A*A benchmark requires a square matrix".into());
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

    let config = PackedConfig::default();

    let row_start = Instant::now();
    let rows = PackedAdaptiveMatrix::from_csr32(&matrix, config)?;
    let row_prepare_ms = row_start.elapsed().as_secs_f64() * 1.0e3;

    let col_start = Instant::now();
    let columns = PackedAdaptiveMatrix::from_csr32(&transpose, config)?;
    let col_prepare_ms = col_start.elapsed().as_secs_f64() * 1.0e3;

    let row_stats = rows.stats();
    let col_stats = columns.stats();
    let packed_dual_bytes = row_stats
        .total_bytes()
        .saturating_add(col_stats.total_bytes());
    let explicit_dual_bytes = matrix
        .storage_bytes()
        .saturating_add(transpose.storage_bytes());
    let packed_over_explicit_storage = packed_dual_bytes as f64 / explicit_dual_bytes.max(1) as f64;

    println!(
        "G2F_PREPARE|sparse_max_nnz={}|dense_min_nnz={}|row_tiles={}|row_sparse={}|row_bitmap={}|row_dense={}|col_tiles={}|col_sparse={}|col_bitmap={}|col_dense={}|row_metadata_bytes={}|col_metadata_bytes={}|row_value_slots={}|col_value_slots={}|packed_dual_bytes={packed_dual_bytes}|explicit_dual_bytes={explicit_dual_bytes}|packed_over_explicit_storage={packed_over_explicit_storage:.9e}|row_prepare_ms={row_prepare_ms:.6}|col_prepare_ms={col_prepare_ms:.6}",
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
        let actual = packed_numeric_dot(row, col, &rows, &columns)?;
        max_scaled_error = max_scaled_error.max(scaled_error(reference, actual));
    }

    if max_scaled_error > 1.0e-10 {
        return Err(
            format!("G2f numerical mismatch: max_scaled_error={max_scaled_error:.3e}").into(),
        );
    }

    let mut explicit_samples = Vec::with_capacity(args.repeats);
    let mut packed_samples = Vec::with_capacity(args.repeats);

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
            checksum += packed_numeric_dot(row as usize, col as usize, &rows, &columns)?;
        }
        packed_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box(checksum);
    }

    let explicit_ms = median(&mut explicit_samples);
    let packed_ms = median(&mut packed_samples);
    let packed_over_explicit = if explicit_ms == 0.0 {
        0.0
    } else {
        packed_ms / explicit_ms
    };

    println!(
        "G2F_NUMERIC|pairs={}|explicit_ms={explicit_ms:.6}|packed_ms={packed_ms:.6}|packed_over_explicit={packed_over_explicit:.9e}|max_scaled_error={max_scaled_error:.9e}",
        pairs.len(),
    );

    Ok(())
}
