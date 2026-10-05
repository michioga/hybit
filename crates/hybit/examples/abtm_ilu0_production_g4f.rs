use std::env;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use hybit::{read_matrix_market, Ilu0Preconditioner, Preconditioner};

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
                    println!("Usage: abtm_ilu0_production_g4f --matrix A.mtx [--repeats N]");
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

fn deterministic_rhs(n: usize) -> Vec<f64> {
    (0..n)
        .map(|index| {
            let x = index as f64 + 1.0;
            (x * 0.003_906_25).sin() + (x * 0.001_953_125).cos() * 0.25
        })
        .collect()
}

fn max_scaled_error(reference: &[f64], actual: &[f64]) -> f64 {
    reference
        .iter()
        .zip(actual)
        .map(|(&a, &b)| (a - b).abs() / a.abs().max(1.0))
        .fold(0.0f64, f64::max)
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;

    println!(
        "HyBIT {} ABTM G4f production ILU(0)",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!("repeats             : {}", args.repeats);

    let (matrix, info) = read_matrix_market(&args.matrix)?;
    println!(
        "Matrix Market       : {:?}, {} input entries -> {} CSR nnz",
        info.symmetry, info.input_entries, info.csr_nnz
    );
    println!(
        "dimensions          : {} x {}",
        matrix.nrows(),
        matrix.ncols()
    );
    println!("stored CSR nnz      : {}", matrix.nnz());

    let csr = Ilu0Preconditioner::from_csr32_general(&matrix)?;
    let abtm = Ilu0Preconditioner::from_csr32_general_abtm(&matrix)?;

    if csr.canonical_nnz() != abtm.canonical_nnz()
        || csr.adjusted_pivots() != abtm.adjusted_pivots()
        || csr.factor_bytes() != abtm.factor_bytes()
    {
        return Err("G4f persistent factor metadata differ".into());
    }

    let rhs = deterministic_rhs(matrix.nrows());
    let mut z_csr = vec![0.0; matrix.nrows()];
    let mut z_abtm = vec![0.0; matrix.nrows()];
    csr.apply(&rhs, &mut z_csr)?;
    abtm.apply(&rhs, &mut z_abtm)?;
    let apply_error = max_scaled_error(&z_csr, &z_abtm);

    if apply_error > 1.0e-12 {
        return Err(format!("G4f apply mismatch: {apply_error:.3e}").into());
    }

    println!(
        "G4F_VALIDATE|canonical_nnz={}|adjusted_pivots={}|csr_factor_bytes={}|abtm_factor_bytes={}|apply_max_scaled_error={apply_error:.9e}|mismatched=0",
        csr.canonical_nnz(),
        csr.adjusted_pivots(),
        csr.factor_bytes(),
        abtm.factor_bytes(),
    );

    let mut csr_samples = Vec::with_capacity(args.repeats);
    let mut abtm_samples = Vec::with_capacity(args.repeats);

    for _ in 0..args.repeats {
        let start = Instant::now();
        let factor = Ilu0Preconditioner::from_csr32_general(&matrix)?;
        csr_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box(factor.canonical_nnz());

        let start = Instant::now();
        let factor = Ilu0Preconditioner::from_csr32_general_abtm(&matrix)?;
        abtm_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box(factor.canonical_nnz());
    }

    let csr_ms = median(&mut csr_samples);
    let abtm_ms = median(&mut abtm_samples);
    let ratio = if csr_ms == 0.0 { 0.0 } else { abtm_ms / csr_ms };
    let saved = csr_ms - abtm_ms;

    println!(
        "G4F_TIMING|csr_prepare_ms={csr_ms:.6}|abtm_prepare_ms={abtm_ms:.6}|abtm_over_csr_prepare={ratio:.9e}|saved_prepare_ms={saved:.6}"
    );

    Ok(())
}
