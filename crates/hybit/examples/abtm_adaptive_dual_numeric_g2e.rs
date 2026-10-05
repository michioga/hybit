use std::env;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use hybit_matrix::{read_matrix_market, AbtmConfig, AbtmMatrix, Csr32Matrix, TileDesc, TileKind};

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
                        "Usage: abtm_adaptive_dual_numeric_g2e --matrix A.mtx [--pairs-per-row N] [--repeats N]"
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
fn tile_value(tile: &TileDesc, values: &[f64], bit: usize) -> f64 {
    let offset = tile.value_offset as usize;
    match tile.kind() {
        TileKind::Dense => values[offset + bit],
        TileKind::Sparse | TileKind::Bitmap => values[offset + compact_rank(tile.mask, bit)],
    }
}

#[inline(always)]
fn sparse_sparse_dot(a: &TileDesc, a_values: &[f64], b: &TileDesc, b_values: &[f64]) -> f64 {
    let mut a_bits = a.mask;
    let mut b_bits = b.mask;
    let mut a_ordinal = 0usize;
    let mut b_ordinal = 0usize;
    let mut sum = 0.0;

    while a_bits != 0 && b_bits != 0 {
        let a_bit = a_bits.trailing_zeros() as usize;
        let b_bit = b_bits.trailing_zeros() as usize;

        if a_bit < b_bit {
            a_bits &= a_bits - 1;
            a_ordinal += 1;
        } else if b_bit < a_bit {
            b_bits &= b_bits - 1;
            b_ordinal += 1;
        } else {
            sum += a_values[a.value_offset as usize + a_ordinal]
                * b_values[b.value_offset as usize + b_ordinal];
            a_bits &= a_bits - 1;
            b_bits &= b_bits - 1;
            a_ordinal += 1;
            b_ordinal += 1;
        }
    }

    sum
}

#[inline(always)]
fn sparse_other_dot(
    sparse: &TileDesc,
    sparse_values: &[f64],
    other: &TileDesc,
    other_values: &[f64],
) -> f64 {
    let mut bits = sparse.mask;
    let mut ordinal = 0usize;
    let mut sum = 0.0;

    while bits != 0 {
        let bit = bits.trailing_zeros() as usize;
        if (other.mask & (1u64 << bit)) != 0 {
            sum += sparse_values[sparse.value_offset as usize + ordinal]
                * tile_value(other, other_values, bit);
        }
        ordinal += 1;
        bits &= bits - 1;
    }

    sum
}

#[inline(always)]
fn bitmap_or_dense_dot(a: &TileDesc, a_values: &[f64], b: &TileDesc, b_values: &[f64]) -> f64 {
    let mut active = a.mask & b.mask;
    let mut sum = 0.0;

    while active != 0 {
        let bit = active.trailing_zeros() as usize;
        sum += tile_value(a, a_values, bit) * tile_value(b, b_values, bit);
        active &= active - 1;
    }

    sum
}

#[inline(always)]
fn matched_tile_dot(a: &TileDesc, a_values: &[f64], b: &TileDesc, b_values: &[f64]) -> f64 {
    match (a.kind(), b.kind()) {
        (TileKind::Sparse, TileKind::Sparse) => sparse_sparse_dot(a, a_values, b, b_values),
        (TileKind::Sparse, _) => sparse_other_dot(a, a_values, b, b_values),
        (_, TileKind::Sparse) => sparse_other_dot(b, b_values, a, a_values),
        _ => bitmap_or_dense_dot(a, a_values, b, b_values),
    }
}

fn adaptive_numeric_dot(
    row: usize,
    col: usize,
    rows: &AbtmMatrix,
    columns: &AbtmMatrix,
) -> Result<f64, Box<dyn Error>> {
    let a = rows.row_tiles(row)?;
    let b = columns.row_tiles(col)?;
    let a_values = rows.values();
    let b_values = columns.values();

    let mut i = 0usize;
    let mut j = 0usize;
    let mut sum = 0.0;

    while i < a.len() && j < b.len() {
        let a_base = a[i].base_col();
        let b_base = b[j].base_col();

        if a_base < b_base {
            i += 1;
        } else if b_base < a_base {
            j += 1;
        } else {
            if a[i].mask & b[j].mask != 0 {
                sum += matched_tile_dot(&a[i], a_values, &b[j], b_values);
            }
            i += 1;
            j += 1;
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
                        ^ 0x38d0_2026_1005_02d0;
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
        "HyBIT {} ABTM G2e adaptive dual-numeric sparse dot",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!("pairs per row       : {}", args.pairs_per_row);
    println!("repeats             : {}", args.repeats);

    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    if matrix.nrows() != matrix.ncols() {
        return Err("G2e same-matrix A*A benchmark requires a square matrix".into());
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

    let config = AbtmConfig::default();

    let row_prepare_start = Instant::now();
    let rows = AbtmMatrix::from_csr32(&matrix, config)?;
    let row_prepare_ms = row_prepare_start.elapsed().as_secs_f64() * 1.0e3;

    let col_prepare_start = Instant::now();
    let columns = AbtmMatrix::from_csr32(&transpose, config)?;
    let col_prepare_ms = col_prepare_start.elapsed().as_secs_f64() * 1.0e3;

    let row_stats = rows.stats();
    let col_stats = columns.stats();
    let adaptive_storage = row_stats
        .total_estimated_bytes()
        .saturating_add(col_stats.total_estimated_bytes());
    let explicit_storage = matrix
        .storage_bytes()
        .saturating_add(transpose.storage_bytes());
    let adaptive_over_explicit_storage = adaptive_storage as f64 / explicit_storage.max(1) as f64;

    println!(
        "G2E_PREPARE|sparse_max_nnz={}|dense_min_nnz={}|row_tiles={}|row_sparse={}|row_bitmap={}|row_dense={}|col_tiles={}|col_sparse={}|col_bitmap={}|col_dense={}|row_value_slots={}|col_value_slots={}|adaptive_dual_bytes={adaptive_storage}|explicit_dual_bytes={explicit_storage}|adaptive_over_explicit_storage={adaptive_over_explicit_storage:.9e}|row_prepare_ms={row_prepare_ms:.6}|col_prepare_ms={col_prepare_ms:.6}",
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
        row_stats.value_slots,
        col_stats.value_slots,
    );

    let pairs = make_pairs(matrix.nrows(), args.pairs_per_row);

    let mut max_scaled_error = 0.0f64;
    for &(row, col) in &pairs {
        let row = row as usize;
        let col = col as usize;
        let reference = explicit_numeric_dot(row, col, &matrix, &transpose);
        let actual = adaptive_numeric_dot(row, col, &rows, &columns)?;
        max_scaled_error = max_scaled_error.max(scaled_error(reference, actual));
    }

    if max_scaled_error > 1.0e-8 {
        return Err(
            format!("G2e numerical mismatch: max_scaled_error={max_scaled_error:.3e}").into(),
        );
    }

    let mut explicit_samples = Vec::with_capacity(args.repeats);
    let mut adaptive_samples = Vec::with_capacity(args.repeats);

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
            checksum += adaptive_numeric_dot(row as usize, col as usize, &rows, &columns)?;
        }
        adaptive_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box(checksum);
    }

    let explicit_ms = median(&mut explicit_samples);
    let adaptive_ms = median(&mut adaptive_samples);
    let adaptive_over_explicit = if explicit_ms == 0.0 {
        0.0
    } else {
        adaptive_ms / explicit_ms
    };

    println!(
        "G2E_NUMERIC|pairs={}|explicit_ms={explicit_ms:.6}|adaptive_ms={adaptive_ms:.6}|adaptive_over_explicit={adaptive_over_explicit:.9e}|validation_tolerance=1.000000000e-8|max_scaled_error={max_scaled_error:.9e}",
        pairs.len(),
    );

    Ok(())
}
