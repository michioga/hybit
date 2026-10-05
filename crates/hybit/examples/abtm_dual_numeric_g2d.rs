use std::env;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use hybit_matrix::{
    read_matrix_market, AbtmDualTopology, AbtmMetadataFirstMatrix, AbtmTopologyRow,
};

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
                        "Usage: abtm_dual_numeric_g2d --matrix A.mtx [--pairs-per-row N] [--repeats N]"
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

#[derive(Clone, Debug)]
struct DualNumericLayout {
    row_ptr: Vec<u32>,
    row_indices: Vec<u32>,
    col_ptr: Vec<u32>,
    col_indices: Vec<u32>,
    col_source_value_indices: Vec<u32>,
    col_values: Vec<f64>,
}

impl DualNumericLayout {
    fn from_prepared(
        prepared: &AbtmMetadataFirstMatrix,
        dual: &AbtmDualTopology,
    ) -> Result<Self, Box<dyn Error>> {
        if prepared.topology().nrows() != dual.nrows()
            || prepared.topology().ncols() != dual.ncols()
        {
            return Err("prepared numeric matrix and dual topology dimensions differ".into());
        }

        let mut row_ptr = Vec::with_capacity(dual.nrows() + 1);
        let mut row_indices = Vec::with_capacity(prepared.packed_values().len());
        let mut row_value_base = Vec::with_capacity(dual.nrows());
        row_ptr.push(0u32);

        for row in 0..dual.nrows() {
            row_value_base.push(row_indices.len());
            let topology_row = dual.row(row)?;
            let values = prepared.row_values(row)?;

            let mut ordinal = 0usize;
            for word in topology_row.words() {
                let mut bits = word.mask();
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    let col = word.base_col() + bit;
                    if col > u32::MAX as usize {
                        return Err("row index exceeds u32 range".into());
                    }
                    row_indices.push(col as u32);
                    ordinal += 1;
                    bits &= bits - 1;
                }
            }

            if ordinal != values.len() {
                return Err("row topology/value count mismatch".into());
            }
            if row_indices.len() > u32::MAX as usize {
                return Err("row index stream exceeds u32 range".into());
            }
            row_ptr.push(row_indices.len() as u32);
        }

        if row_indices.len() != prepared.packed_values().len() {
            return Err("row index stream differs from packed-value count".into());
        }

        let mut buckets: Vec<Vec<(u32, u32, f64)>> =
            (0..dual.ncols()).map(|_| Vec::new()).collect();

        for row in 0..dual.nrows() {
            let start = row_ptr[row] as usize;
            let end = row_ptr[row + 1] as usize;
            for local in 0..(end - start) {
                let global = start + local;
                let col = row_indices[global] as usize;
                if global > u32::MAX as usize || row > u32::MAX as usize {
                    return Err("G2d source index exceeds u32 range".into());
                }
                buckets[col].push((row as u32, global as u32, prepared.packed_values()[global]));
            }
        }

        let mut col_ptr = Vec::with_capacity(dual.ncols() + 1);
        let mut col_indices = Vec::with_capacity(row_indices.len());
        let mut col_source_value_indices = Vec::with_capacity(row_indices.len());
        let mut col_values = Vec::with_capacity(row_indices.len());
        col_ptr.push(0u32);

        for (col, bucket) in buckets.iter().enumerate() {
            let topology_col = dual.column(col)?;
            if topology_col.popcount() != bucket.len() {
                return Err(format!(
                    "column {col} topology/value count mismatch: topology={}, values={}",
                    topology_col.popcount(),
                    bucket.len()
                )
                .into());
            }

            for (ordinal, &(row, source, value)) in bucket.iter().enumerate() {
                let topology_row = topology_col
                    .select(ordinal)
                    .ok_or("column topology select unexpectedly failed")?;
                if topology_row != row as usize {
                    return Err(format!(
                        "column {col} topology/value order mismatch at ordinal {ordinal}"
                    )
                    .into());
                }
                col_indices.push(row);
                col_source_value_indices.push(source);
                col_values.push(value);
            }

            if col_indices.len() > u32::MAX as usize {
                return Err("column index stream exceeds u32 range".into());
            }
            col_ptr.push(col_indices.len() as u32);
        }

        if col_indices.len() != row_indices.len()
            || col_source_value_indices.len() != row_indices.len()
            || col_values.len() != row_indices.len()
        {
            return Err("G2d column streams do not preserve structural nnz".into());
        }

        Ok(Self {
            row_ptr,
            row_indices,
            col_ptr,
            col_indices,
            col_source_value_indices,
            col_values,
        })
    }

    fn explicit_total_bytes(&self) -> usize {
        self.row_ptr.len() * std::mem::size_of::<u32>()
            + self.row_indices.len() * std::mem::size_of::<u32>()
            + self.row_indices.len() * std::mem::size_of::<f64>()
            + self.col_ptr.len() * std::mem::size_of::<u32>()
            + self.col_indices.len() * std::mem::size_of::<u32>()
            + self.col_values.len() * std::mem::size_of::<f64>()
    }

    fn mapped_auxiliary_bytes(&self) -> usize {
        self.col_ptr.len() * std::mem::size_of::<u32>()
            + self.col_source_value_indices.len() * std::mem::size_of::<u32>()
    }

    fn duplicated_auxiliary_bytes(&self) -> usize {
        self.col_ptr.len() * std::mem::size_of::<u32>()
            + self.col_values.len() * std::mem::size_of::<f64>()
    }
}

fn explicit_numeric_dot(
    row: usize,
    col: usize,
    prepared: &AbtmMetadataFirstMatrix,
    layout: &DualNumericLayout,
) -> f64 {
    let a_start = layout.row_ptr[row] as usize;
    let a_end = layout.row_ptr[row + 1] as usize;
    let b_start = layout.col_ptr[col] as usize;
    let b_end = layout.col_ptr[col + 1] as usize;

    let a_idx = &layout.row_indices[a_start..a_end];
    let a_val = &prepared.packed_values()[a_start..a_end];
    let b_idx = &layout.col_indices[b_start..b_end];
    let b_val = &layout.col_values[b_start..b_end];

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

#[derive(Clone, Copy)]
struct BitmapValueAccess<'a> {
    row_values: &'a [f64],
    col_source_value_indices: &'a [u32],
    col_values: &'a [f64],
    mapped_column: bool,
}

fn bitmap_numeric_dot(
    a: AbtmTopologyRow<'_>,
    b: AbtmTopologyRow<'_>,
    row_value_start: usize,
    col_value_start: usize,
    access: BitmapValueAccess<'_>,
) -> Result<(f64, usize), Box<dyn Error>> {
    let mut a_words = a.words().peekable();
    let mut b_words = b.words().peekable();
    let mut a_word_value_base = 0usize;
    let mut b_word_value_base = 0usize;
    let mut sum = 0.0;
    let mut products = 0usize;

    while let (Some(aw), Some(bw)) = (a_words.peek().copied(), b_words.peek().copied()) {
        if aw.word_index() < bw.word_index() {
            a_word_value_base += aw.popcount();
            a_words.next();
            continue;
        }
        if bw.word_index() < aw.word_index() {
            b_word_value_base += bw.popcount();
            b_words.next();
            continue;
        }

        let mut active = aw.mask() & bw.mask();
        while active != 0 {
            let bit = active.trailing_zeros() as usize;
            let a_index = row_value_start + a_word_value_base + aw.rank(bit)?;
            let b_index = col_value_start + b_word_value_base + bw.rank(bit)?;

            let b_value = if access.mapped_column {
                let source = access.col_source_value_indices[b_index] as usize;
                access.row_values[source]
            } else {
                access.col_values[b_index]
            };

            sum += access.row_values[a_index] * b_value;
            products += 1;
            active &= active - 1;
        }

        a_word_value_base += aw.popcount();
        b_word_value_base += bw.popcount();
        a_words.next();
        b_words.next();
    }

    Ok((sum, products))
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
        "HyBIT {} ABTM G2d dual-topology numerical sparse dot",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!("pairs per row       : {}", args.pairs_per_row);
    println!("repeats             : {}", args.repeats);

    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    if matrix.nrows() != matrix.ncols() {
        return Err("G2d same-matrix A*A benchmark requires a square matrix".into());
    }

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

    let dual_start = Instant::now();
    let dual = AbtmDualTopology::from_csr32(&matrix)?;
    let dual_prepare_ms = dual_start.elapsed().as_secs_f64() * 1.0e3;

    let numeric_start = Instant::now();
    let prepared = AbtmMetadataFirstMatrix::from_csr32(&matrix)?;
    let numeric_prepare_ms = numeric_start.elapsed().as_secs_f64() * 1.0e3;

    let layout_start = Instant::now();
    let layout = DualNumericLayout::from_prepared(&prepared, &dual)?;
    let layout_prepare_ms = layout_start.elapsed().as_secs_f64() * 1.0e3;

    let topology_bytes = dual.stats().total_metadata_bytes();
    let row_value_bytes = std::mem::size_of_val(prepared.packed_values());
    let mapped_total_bytes = topology_bytes
        .saturating_add(row_value_bytes)
        .saturating_add(layout.mapped_auxiliary_bytes());
    let duplicated_total_bytes = topology_bytes
        .saturating_add(row_value_bytes)
        .saturating_add(layout.duplicated_auxiliary_bytes());
    let explicit_total_bytes = layout.explicit_total_bytes();

    println!(
        "G2D_PREPARE|structural_nnz={}|dual_topology_bytes={topology_bytes}|row_value_bytes={row_value_bytes}|mapped_auxiliary_bytes={}|duplicated_auxiliary_bytes={}|mapped_total_bytes={mapped_total_bytes}|duplicated_total_bytes={duplicated_total_bytes}|explicit_dual_total_bytes={explicit_total_bytes}|mapped_over_explicit_storage={:.9e}|duplicated_over_explicit_storage={:.9e}|dual_prepare_ms={dual_prepare_ms:.6}|numeric_prepare_ms={numeric_prepare_ms:.6}|layout_prepare_ms={layout_prepare_ms:.6}",
        prepared.topology().structural_nnz(),
        layout.mapped_auxiliary_bytes(),
        layout.duplicated_auxiliary_bytes(),
        mapped_total_bytes as f64 / explicit_total_bytes.max(1) as f64,
        duplicated_total_bytes as f64 / explicit_total_bytes.max(1) as f64,
    );

    let pairs = make_pairs(matrix.nrows(), args.pairs_per_row);

    let mut overlap_products = 0usize;
    let mut max_mapped_error = 0.0f64;
    let mut max_duplicated_error = 0.0f64;

    for &(row, col) in &pairs {
        let row = row as usize;
        let col = col as usize;
        let reference = explicit_numeric_dot(row, col, &prepared, &layout);

        let row_start = layout.row_ptr[row] as usize;
        let col_start = layout.col_ptr[col] as usize;

        let (mapped, mapped_products) = bitmap_numeric_dot(
            dual.row(row)?,
            dual.column(col)?,
            row_start,
            col_start,
            BitmapValueAccess {
                row_values: prepared.packed_values(),
                col_source_value_indices: &layout.col_source_value_indices,
                col_values: &layout.col_values,
                mapped_column: true,
            },
        )?;
        let (duplicated, duplicated_products) = bitmap_numeric_dot(
            dual.row(row)?,
            dual.column(col)?,
            row_start,
            col_start,
            BitmapValueAccess {
                row_values: prepared.packed_values(),
                col_source_value_indices: &layout.col_source_value_indices,
                col_values: &layout.col_values,
                mapped_column: false,
            },
        )?;

        if mapped_products != duplicated_products {
            return Err("G2d mapped/duplicated product counts differ".into());
        }

        max_mapped_error = max_mapped_error.max(scaled_error(reference, mapped));
        max_duplicated_error = max_duplicated_error.max(scaled_error(reference, duplicated));
        overlap_products += mapped_products;
    }

    if max_mapped_error > 1.0e-10 || max_duplicated_error > 1.0e-10 {
        return Err(format!(
            "G2d numerical mismatch: mapped={max_mapped_error:.3e}, duplicated={max_duplicated_error:.3e}"
        )
        .into());
    }

    let mut explicit_samples = Vec::with_capacity(args.repeats);
    let mut mapped_samples = Vec::with_capacity(args.repeats);
    let mut duplicated_samples = Vec::with_capacity(args.repeats);

    for _ in 0..args.repeats {
        let start = Instant::now();
        let mut checksum = 0.0;
        for &(row, col) in &pairs {
            checksum += explicit_numeric_dot(row as usize, col as usize, &prepared, &layout);
        }
        explicit_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box(checksum);

        let start = Instant::now();
        let mut checksum = 0.0;
        let mut products = 0usize;
        for &(row, col) in &pairs {
            let row = row as usize;
            let col = col as usize;
            let (value, count) = bitmap_numeric_dot(
                dual.row(row)?,
                dual.column(col)?,
                layout.row_ptr[row] as usize,
                layout.col_ptr[col] as usize,
                BitmapValueAccess {
                    row_values: prepared.packed_values(),
                    col_source_value_indices: &layout.col_source_value_indices,
                    col_values: &layout.col_values,
                    mapped_column: true,
                },
            )?;
            checksum += value;
            products += count;
        }
        mapped_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box((checksum, products));

        let start = Instant::now();
        let mut checksum = 0.0;
        let mut products = 0usize;
        for &(row, col) in &pairs {
            let row = row as usize;
            let col = col as usize;
            let (value, count) = bitmap_numeric_dot(
                dual.row(row)?,
                dual.column(col)?,
                layout.row_ptr[row] as usize,
                layout.col_ptr[col] as usize,
                BitmapValueAccess {
                    row_values: prepared.packed_values(),
                    col_source_value_indices: &layout.col_source_value_indices,
                    col_values: &layout.col_values,
                    mapped_column: false,
                },
            )?;
            checksum += value;
            products += count;
        }
        duplicated_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box((checksum, products));
    }

    let explicit_ms = median(&mut explicit_samples);
    let mapped_ms = median(&mut mapped_samples);
    let duplicated_ms = median(&mut duplicated_samples);

    let mapped_over_explicit = if explicit_ms == 0.0 {
        0.0
    } else {
        mapped_ms / explicit_ms
    };
    let duplicated_over_explicit = if explicit_ms == 0.0 {
        0.0
    } else {
        duplicated_ms / explicit_ms
    };
    let mapped_over_duplicated = if duplicated_ms == 0.0 {
        0.0
    } else {
        mapped_ms / duplicated_ms
    };
    let average_products = if pairs.is_empty() {
        0.0
    } else {
        overlap_products as f64 / pairs.len() as f64
    };

    println!(
        "G2D_NUMERIC|pairs={}|overlap_products={overlap_products}|average_products_per_pair={average_products:.9e}|explicit_ms={explicit_ms:.6}|bitmap_mapped_ms={mapped_ms:.6}|bitmap_duplicated_ms={duplicated_ms:.6}|mapped_over_explicit={mapped_over_explicit:.9e}|duplicated_over_explicit={duplicated_over_explicit:.9e}|mapped_over_duplicated={mapped_over_duplicated:.9e}|max_mapped_scaled_error={max_mapped_error:.9e}|max_duplicated_scaled_error={max_duplicated_error:.9e}",
        pairs.len(),
    );

    Ok(())
}
