use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use hybit_matrix::{
    read_matrix_market, AbtmMetadataFirstMatrix, AbtmProductPruningStats, Csr32Matrix, DofMask,
};

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    repeats: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut repeats = 5usize;
        let mut it = env::args().skip(1);

        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => {
                    matrix = Some(PathBuf::from(
                        it.next().ok_or("missing value after --matrix")?,
                    ));
                }
                "--repeats" => {
                    repeats = it.next().ok_or("missing value after --repeats")?.parse()?;
                    if repeats == 0 {
                        return Err("--repeats must be >= 1".into());
                    }
                }
                "-h" | "--help" => {
                    println!("Usage: abtm_metadata_pruning_g2 --matrix A.mtx [--repeats N]");
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
            repeats,
        })
    }
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

fn make_sparse_vector(
    ncols: usize,
    active_percent: usize,
) -> Result<(DofMask, Vec<f64>), Box<dyn Error>> {
    let mut mask = DofMask::new(ncols);
    let mut x = vec![0.0; ncols];

    for (col, xi) in x.iter_mut().enumerate() {
        let h = mix(col as u64 ^ 0x38d0_2026_1005_0201);
        let active = active_percent == 100 || (h % 100) < active_percent as u64;
        if active {
            mask.set(col, true)?;
            let raw = ((h >> 16) & 0xffff) as f64;
            *xi = 0.5 + raw / 65535.0;
        }
    }

    Ok((mask, x))
}

fn csr_full_dot_all_rows(matrix: &Csr32Matrix, x: &[f64], y: &mut [f64]) {
    for (row, out) in y.iter_mut().enumerate() {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        let mut sum = 0.0;
        for p in start..end {
            sum += matrix.values()[p] * x[matrix.col_idx()[p] as usize];
        }
        *out = sum;
    }
}

fn abtm_pruned_dot_all_rows(
    prepared: &AbtmMetadataFirstMatrix,
    mask: &DofMask,
    x: &[f64],
    y: &mut [f64],
) -> Result<AbtmProductPruningStats, Box<dyn Error>> {
    let mut total = AbtmProductPruningStats::default();
    for (row, out) in y.iter_mut().enumerate() {
        let (sum, stats) = prepared.sparse_dot_row(row, mask, x)?;
        *out = sum;
        total.accumulate(stats)?;
    }
    Ok(total)
}

fn relative_l2(reference: &[f64], actual: &[f64]) -> f64 {
    let mut diff2 = 0.0;
    let mut ref2 = 0.0;
    for (&a, &b) in reference.iter().zip(actual) {
        let d = b - a;
        diff2 += d * d;
        ref2 += a * a;
    }
    if ref2 == 0.0 {
        diff2.sqrt()
    } else {
        (diff2 / ref2).sqrt()
    }
}

fn max_scaled_error(reference: &[f64], actual: &[f64]) -> f64 {
    let mut max_diff = 0.0f64;
    let mut max_ref = 0.0f64;
    for (&a, &b) in reference.iter().zip(actual) {
        max_diff = max_diff.max((b - a).abs());
        max_ref = max_ref.max(a.abs());
    }
    max_diff / max_ref.max(1.0)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;

    println!(
        "HyBIT {} ABTM G2 metadata-first sparse-dot pruning",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!("repeats             : {}", args.repeats);

    let (matrix, mm) = read_matrix_market(&args.matrix)?;
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

    let prepare_start = Instant::now();
    let prepared = AbtmMetadataFirstMatrix::from_csr32(&matrix)?;
    let prepare_ms = prepare_start.elapsed().as_secs_f64() * 1.0e3;
    let prepared_stats = prepared.stats();

    println!(
        "G2_PREPARE|structural_nnz={}|topology_metadata_bytes={}|value_row_ptr_bytes={}|packed_value_bytes={}|total_estimated_bytes={}|prepare_ms={prepare_ms:.6}",
        prepared_stats.structural_nnz,
        prepared_stats.topology_metadata_bytes,
        prepared_stats.value_row_ptr_bytes,
        prepared_stats.packed_value_bytes,
        prepared_stats.total_estimated_bytes(),
    );

    for active_percent in [100usize, 75, 50, 25, 10] {
        let (mask, x) = make_sparse_vector(matrix.ncols(), active_percent)?;
        let active_dofs = mask.count_ones();

        let mut reference = vec![0.0; matrix.nrows()];
        let mut actual = vec![0.0; matrix.nrows()];

        csr_full_dot_all_rows(&matrix, &x, &mut reference);
        let stats = abtm_pruned_dot_all_rows(&prepared, &mask, &x, &mut actual)?;

        let rel_l2 = relative_l2(&reference, &actual);
        let max_scaled = max_scaled_error(&reference, &actual);
        if rel_l2 > 1.0e-10 && max_scaled > 1.0e-10 {
            return Err(format!(
                "G2 numerical mismatch at active_percent={active_percent}: rel_l2={rel_l2:.3e}, max_scaled={max_scaled:.3e}"
            )
            .into());
        }

        let mut csr_samples = Vec::with_capacity(args.repeats);
        let mut abtm_samples = Vec::with_capacity(args.repeats);

        for _ in 0..args.repeats {
            let start = Instant::now();
            csr_full_dot_all_rows(&matrix, &x, &mut reference);
            csr_samples.push(start.elapsed().as_secs_f64() * 1.0e3);

            let start = Instant::now();
            let measured = abtm_pruned_dot_all_rows(&prepared, &mask, &x, &mut actual)?;
            abtm_samples.push(start.elapsed().as_secs_f64() * 1.0e3);

            if measured != stats {
                return Err("G2 pruning stats changed across deterministic repeats".into());
            }
        }

        let csr_ms = median(&mut csr_samples);
        let abtm_ms = median(&mut abtm_samples);
        let abtm_over_csr = if csr_ms == 0.0 { 0.0 } else { abtm_ms / csr_ms };
        let active_fraction = if matrix.ncols() == 0 {
            0.0
        } else {
            active_dofs as f64 / matrix.ncols() as f64
        };

        println!(
            "G2_PRUNE|requested_active_percent={active_percent}|active_dofs={active_dofs}|active_fraction={active_fraction:.9e}|csr_products={}|candidate_products={}|executed_products={}|skipped_products={}|pruning_ratio={:.9e}|topology_words={}|active_words={}|empty_word_ratio={:.9e}|csr_ms={csr_ms:.6}|abtm_ms={abtm_ms:.6}|abtm_over_csr={abtm_over_csr:.9e}|rel_l2={rel_l2:.9e}|max_scaled={max_scaled:.9e}",
            matrix.nnz(),
            stats.candidate_products,
            stats.executed_products,
            stats.skipped_products,
            stats.pruning_ratio(),
            stats.topology_words,
            stats.active_words,
            stats.empty_word_ratio(),
        );
    }

    Ok(())
}
