use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use hybit::{
    read_matrix_market, Csr32Matrix, Ilu0Preconditioner, JacobiPreconditioner, Preconditioner,
};

const APPLY_REPEATS: usize = 3;

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut it = env::args().skip(1);

        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => {
                    matrix = Some(PathBuf::from(
                        it.next().ok_or("missing value after --matrix")?,
                    ));
                }
                "-h" | "--help" => {
                    println!("Usage: general_square_preconditioner_probe --matrix A.mtx");
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
        })
    }
}

fn norm2(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn probe_vector(n: usize, index: usize) -> (Vec<f64>, &'static str) {
    match index {
        0 => (vec![1.0; n], "ones"),
        1 => (
            (0..n)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
            "alternating",
        ),
        _ => {
            let x = (0..n)
                .map(|i| {
                    let bits = splitmix64((i as u64) ^ 0xf7b0_2026_1004_0001);
                    let unit = ((bits >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
                    2.0 * unit - 1.0
                })
                .collect();
            (x, "hashed")
        }
    }
}

fn relative_error(actual: &[f64], exact: &[f64]) -> f64 {
    let diff = actual
        .iter()
        .zip(exact)
        .map(|(&actual_value, &exact_value)| {
            let delta = actual_value - exact_value;
            delta * delta
        })
        .sum::<f64>()
        .sqrt();
    diff / norm2(exact).max(f64::MIN_POSITIVE)
}

fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let rank = p.clamp(0.0, 1.0) * (values.len().saturating_sub(1) as f64);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let t = rank - lo as f64;
        values[lo] * (1.0 - t) + values[hi] * t
    }
}

#[derive(Debug)]
struct MatrixSignals {
    mean_nnz_per_row: f64,
    p50_nnz_per_row: f64,
    p95_nnz_per_row: f64,
    max_nnz_per_row: usize,
    ilu_work_proxy: u128,
    ilu_work_per_nnz: f64,
    diag_l1_p10: f64,
    diag_l1_p50: f64,
    diag_max_p10: f64,
    diag_max_p50: f64,
    diagonal_dominant_fraction: f64,
    opposite_sign_offdiag_fraction: f64,
    row_cancellation_p50: f64,
}

fn matrix_signals(matrix: &Csr32Matrix) -> Result<MatrixSignals, Box<dyn Error>> {
    let n = matrix.nrows();
    let mut row_nnz = Vec::with_capacity(n);
    let mut upper_count = vec![0usize; n];

    let mut diag_l1 = Vec::with_capacity(n);
    let mut diag_max = Vec::with_capacity(n);
    let mut row_cancellation = Vec::with_capacity(n);

    let mut dominant_rows = 0usize;
    let mut opposite_sign_offdiag = 0usize;
    let mut offdiag_count = 0usize;
    let mut max_nnz = 0usize;

    for (row, row_upper_count) in upper_count.iter_mut().enumerate() {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        let count = end - start;
        row_nnz.push(count as f64);
        max_nnz = max_nnz.max(count);

        let mut diag = None;
        let mut sum_abs = 0.0f64;
        let mut sum_abs_off = 0.0f64;
        let mut row_max = 0.0f64;
        let mut signed_sum = 0.0f64;

        for p in start..end {
            let col = matrix.col_idx()[p] as usize;
            let value = matrix.values()[p];
            let abs = value.abs();

            sum_abs += abs;
            row_max = row_max.max(abs);
            signed_sum += value;

            if col == row {
                diag = Some(value);
            } else {
                sum_abs_off += abs;
                offdiag_count += 1;
            }

            if col > row {
                *row_upper_count += 1;
            }
        }

        let diag = diag.ok_or("missing diagonal during F7b signal scan")?;
        let diag_abs = diag.abs();

        diag_l1.push(if sum_abs > 0.0 {
            diag_abs / sum_abs
        } else {
            0.0
        });
        diag_max.push(if row_max > 0.0 {
            diag_abs / row_max
        } else {
            0.0
        });
        row_cancellation.push(if sum_abs > 0.0 {
            signed_sum.abs() / sum_abs
        } else {
            0.0
        });

        if diag_abs >= sum_abs_off {
            dominant_rows += 1;
        }

        for p in start..end {
            let col = matrix.col_idx()[p] as usize;
            if col == row {
                continue;
            }
            let value = matrix.values()[p];
            if value != 0.0 && diag != 0.0 && value.is_sign_positive() != diag.is_sign_positive() {
                opposite_sign_offdiag += 1;
            }
        }
    }

    // This mirrors the dominant inner-loop opportunity count of the current
    // canonical ILU(0) implementation: each lower entry (i,j) scans the upper
    // part of row j and then searches for structurally present targets in row i.
    let mut ilu_work_proxy = 0u128;
    for (row, _) in upper_count.iter().enumerate() {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        for p in start..end {
            let col = matrix.col_idx()[p] as usize;
            if col < row {
                ilu_work_proxy = ilu_work_proxy
                    .checked_add(upper_count[col] as u128)
                    .ok_or("ILU work proxy overflow")?;
            }
        }
    }

    let nnz = matrix.nnz();
    let mut row_nnz_for_p50 = row_nnz.clone();
    let mut row_nnz_for_p95 = row_nnz;
    let mut diag_l1_for_p10 = diag_l1.clone();
    let mut diag_l1_for_p50 = diag_l1;
    let mut diag_max_for_p10 = diag_max.clone();
    let mut diag_max_for_p50 = diag_max;
    let mut cancellation_for_p50 = row_cancellation;

    Ok(MatrixSignals {
        mean_nnz_per_row: nnz as f64 / n.max(1) as f64,
        p50_nnz_per_row: percentile(&mut row_nnz_for_p50, 0.50),
        p95_nnz_per_row: percentile(&mut row_nnz_for_p95, 0.95),
        max_nnz_per_row: max_nnz,
        ilu_work_proxy,
        ilu_work_per_nnz: ilu_work_proxy as f64 / nnz.max(1) as f64,
        diag_l1_p10: percentile(&mut diag_l1_for_p10, 0.10),
        diag_l1_p50: percentile(&mut diag_l1_for_p50, 0.50),
        diag_max_p10: percentile(&mut diag_max_for_p10, 0.10),
        diag_max_p50: percentile(&mut diag_max_for_p50, 0.50),
        diagonal_dominant_fraction: dominant_rows as f64 / n.max(1) as f64,
        opposite_sign_offdiag_fraction: opposite_sign_offdiag as f64 / offdiag_count.max(1) as f64,
        row_cancellation_p50: percentile(&mut cancellation_for_p50, 0.50),
    })
}

fn timed_apply<P: Preconditioner>(
    preconditioner: &P,
    r: &[f64],
) -> Result<(Vec<f64>, f64), Box<dyn Error>> {
    let mut z = vec![0.0; r.len()];

    // One warm-up keeps first-touch effects out of the tiny apply timing.
    preconditioner.apply(r, &mut z)?;

    let start = Instant::now();
    for _ in 0..APPLY_REPEATS {
        preconditioner.apply(r, &mut z)?;
    }
    let average_ms = start.elapsed().as_secs_f64() * 1.0e3 / APPLY_REPEATS as f64;

    Ok((z, average_ms))
}

fn median3(mut values: [f64; 3]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[1]
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;

    println!(
        "HyBIT {} GeneralSquare F7b preconditioner quality/cost probe",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!("apply repeats       : {APPLY_REPEATS}");

    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    if matrix.nrows() != matrix.ncols() {
        return Err("F7b requires a square matrix".into());
    }
    matrix.diagonal()?;

    println!(
        "Matrix Market       : {:?}, {} input entries -> {} CSR nnz",
        mm.symmetry, mm.input_entries, mm.csr_nnz
    );
    println!(
        "dimensions          : {} x {}",
        matrix.nrows(),
        matrix.ncols()
    );
    println!("nnz                 : {}", matrix.nnz());

    let scan_start = Instant::now();
    let signals = matrix_signals(&matrix)?;
    let scan_ms = scan_start.elapsed().as_secs_f64() * 1.0e3;

    println!(
        "F7B_MATRIX|n={}|nnz={}|mean_nnz_per_row={:.9e}|p50_nnz_per_row={:.9e}|p95_nnz_per_row={:.9e}|max_nnz_per_row={}|ilu_work_proxy={}|ilu_work_per_nnz={:.9e}|diag_l1_p10={:.9e}|diag_l1_p50={:.9e}|diag_max_p10={:.9e}|diag_max_p50={:.9e}|diagonal_dominant_fraction={:.9e}|opposite_sign_offdiag_fraction={:.9e}|row_cancellation_p50={:.9e}|scan_ms={scan_ms:.6}",
        matrix.nrows(),
        matrix.nnz(),
        signals.mean_nnz_per_row,
        signals.p50_nnz_per_row,
        signals.p95_nnz_per_row,
        signals.max_nnz_per_row,
        signals.ilu_work_proxy,
        signals.ilu_work_per_nnz,
        signals.diag_l1_p10,
        signals.diag_l1_p50,
        signals.diag_max_p10,
        signals.diag_max_p50,
        signals.diagonal_dominant_fraction,
        signals.opposite_sign_offdiag_fraction,
        signals.row_cancellation_p50,
    );

    let jacobi_start = Instant::now();
    let jacobi = JacobiPreconditioner::from_csr32_general(&matrix)?;
    let jacobi_setup_ms = jacobi_start.elapsed().as_secs_f64() * 1.0e3;

    let ilu_start = Instant::now();
    let ilu = Ilu0Preconditioner::from_csr32_general(&matrix)?;
    let ilu_setup_ms = ilu_start.elapsed().as_secs_f64() * 1.0e3;

    println!(
        "F7B_SETUP|jacobi_ms={jacobi_setup_ms:.6}|ilu_ms={ilu_setup_ms:.6}|ilu_over_jacobi_setup={:.9e}|jacobi_bytes={}|ilu_bytes={}|ilu_adjusted_pivots={}|canonical_nnz={}",
        ilu_setup_ms / jacobi_setup_ms.max(f64::MIN_POSITIVE),
        std::mem::size_of_val(jacobi.inv_diagonal()),
        ilu.factor_bytes(),
        ilu.adjusted_pivots(),
        ilu.canonical_nnz(),
    );

    let mut defect_ratios = [0.0f64; 3];
    let mut apply_ratios = [0.0f64; 3];

    for probe_index in 0..3 {
        let (x, family) = probe_vector(matrix.ncols(), probe_index);
        let b = matrix.spmv(&x)?;

        let (jacobi_z, jacobi_apply_ms) = timed_apply(&jacobi, &b)?;
        let (ilu_z, ilu_apply_ms) = timed_apply(&ilu, &b)?;

        let jacobi_defect = relative_error(&jacobi_z, &x);
        let ilu_defect = relative_error(&ilu_z, &x);
        let defect_ratio = ilu_defect / jacobi_defect.max(f64::MIN_POSITIVE);
        let apply_ratio = ilu_apply_ms / jacobi_apply_ms.max(f64::MIN_POSITIVE);

        defect_ratios[probe_index] = defect_ratio;
        apply_ratios[probe_index] = apply_ratio;

        println!(
            "F7B_PROBE|family={family}|rhs_norm={:.9e}|jacobi_defect={jacobi_defect:.9e}|ilu_defect={ilu_defect:.9e}|ilu_over_jacobi_defect={defect_ratio:.9e}|jacobi_apply_ms={jacobi_apply_ms:.6}|ilu_apply_ms={ilu_apply_ms:.6}|ilu_over_jacobi_apply={apply_ratio:.9e}",
            norm2(&b),
        );
    }

    println!(
        "F7B_SUMMARY|median_ilu_over_jacobi_defect={:.9e}|max_ilu_over_jacobi_defect={:.9e}|median_ilu_over_jacobi_apply={:.9e}|max_ilu_over_jacobi_apply={:.9e}|ilu_work_per_nnz={:.9e}|ilu_setup_ms={ilu_setup_ms:.6}",
        median3(defect_ratios),
        defect_ratios.into_iter().fold(0.0f64, f64::max),
        median3(apply_ratios),
        apply_ratios.into_iter().fold(0.0f64, f64::max),
        signals.ilu_work_per_nnz,
    );

    Ok(())
}
