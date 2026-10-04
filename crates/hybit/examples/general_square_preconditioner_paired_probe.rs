use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use hybit::{
    fgmres_with_workspace, read_matrix_market, Csr32Matrix, FgmresOptions, FgmresWorkspace,
    Ilu0Preconditioner, JacobiPreconditioner, SolverOptions,
};

const PROBE_ITERS: [usize; 3] = [4, 8, 16];

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    relative_tolerance: f64,
    restart: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut relative_tolerance = 1.0e-8f64;
        let mut restart = 30usize;

        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => {
                    matrix = Some(PathBuf::from(
                        it.next().ok_or("missing value after --matrix")?,
                    ));
                }
                "--tol" => {
                    relative_tolerance = it.next().ok_or("missing value after --tol")?.parse()?;
                }
                "--restart" => {
                    restart = it.next().ok_or("missing value after --restart")?.parse()?;
                }
                "-h" | "--help" => {
                    println!(
                        "Usage: general_square_preconditioner_paired_probe --matrix A.mtx [--tol 1e-8] [--restart 30]"
                    );
                    std::process::exit(0);
                }
                other if !other.starts_with('-') && matrix.is_none() => {
                    matrix = Some(PathBuf::from(other));
                }
                other => return Err(format!("unknown argument '{other}'").into()),
            }
        }

        let matrix = matrix.ok_or("missing matrix path; use --matrix FILE.mtx")?;
        if restart == 0 {
            return Err("--restart must be > 0".into());
        }
        if !relative_tolerance.is_finite() || relative_tolerance <= 0.0 {
            return Err("--tol must be finite and > 0".into());
        }

        Ok(Self {
            matrix,
            relative_tolerance,
            restart,
        })
    }
}

fn norm2(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn verified_relative_residual(
    matrix: &Csr32Matrix,
    b: &[f64],
    x: &[f64],
) -> Result<f64, Box<dyn Error>> {
    let ax = matrix.spmv(x)?;
    let residual = ax
        .iter()
        .zip(b)
        .map(|(&ax_value, &b_value)| {
            let delta = b_value - ax_value;
            delta * delta
        })
        .sum::<f64>()
        .sqrt();
    let b_norm = norm2(b);
    Ok(if b_norm > 0.0 {
        residual / b_norm
    } else {
        residual
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;

    println!(
        "HyBIT {} GeneralSquare F7c Jacobi-vs-ILU(0) paired Krylov probe",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!("probe iterations    : {:?}", PROBE_ITERS);
    println!("restart             : {}", args.restart);
    println!("relative tolerance  : {:.3e}", args.relative_tolerance);

    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    if matrix.nrows() != matrix.ncols() {
        return Err("F7c requires a square matrix".into());
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

    // Match F7a RHS1 exactly: x_exact = 1 and b = A * 1.
    let exact = vec![1.0; matrix.ncols()];
    let b = matrix.spmv(&exact)?;

    let jacobi_start = Instant::now();
    let jacobi = JacobiPreconditioner::from_csr32_general(&matrix)?;
    let jacobi_setup_ms = jacobi_start.elapsed().as_secs_f64() * 1.0e3;

    let ilu_start = Instant::now();
    let ilu = Ilu0Preconditioner::from_csr32_general(&matrix)?;
    let ilu_setup_ms = ilu_start.elapsed().as_secs_f64() * 1.0e3;

    println!(
        "F7C_SETUP|jacobi_ms={jacobi_setup_ms:.6}|ilu_ms={ilu_setup_ms:.6}|ilu_over_jacobi_setup={:.9e}|ilu_adjusted_pivots={}|ilu_factor_bytes={}",
        ilu_setup_ms / jacobi_setup_ms.max(f64::MIN_POSITIVE),
        ilu.adjusted_pivots(),
        ilu.factor_bytes(),
    );

    for probe_iters in PROBE_ITERS {
        let options = FgmresOptions {
            solver: SolverOptions {
                relative_tolerance: args.relative_tolerance,
                absolute_tolerance: 0.0,
                max_iterations: probe_iters,
            },
            restart: args.restart,
        };

        let mut jacobi_pc = jacobi.clone();
        let mut ilu_pc = ilu.clone();

        let mut jacobi_workspace = FgmresWorkspace::new(matrix.nrows(), args.restart)?;
        let mut ilu_workspace = FgmresWorkspace::new(matrix.nrows(), args.restart)?;

        let mut jacobi_x = vec![0.0; matrix.ncols()];
        let jacobi_start = Instant::now();
        let jacobi_outcome = fgmres_with_workspace(
            &matrix,
            &mut jacobi_pc,
            &b,
            &mut jacobi_x,
            options,
            &mut jacobi_workspace,
        )?;
        let jacobi_ms = jacobi_start.elapsed().as_secs_f64() * 1.0e3;
        let jacobi_ratio = verified_relative_residual(&matrix, &b, &jacobi_x)?;

        let mut ilu_x = vec![0.0; matrix.ncols()];
        let ilu_start = Instant::now();
        let ilu_outcome = fgmres_with_workspace(
            &matrix,
            &mut ilu_pc,
            &b,
            &mut ilu_x,
            options,
            &mut ilu_workspace,
        )?;
        let ilu_ms = ilu_start.elapsed().as_secs_f64() * 1.0e3;
        let ilu_ratio = verified_relative_residual(&matrix, &b, &ilu_x)?;

        println!(
            "F7C_PROBE|probe_iters={probe_iters}|jacobi_status={:?}|jacobi_iterations={}|jacobi_ratio={jacobi_ratio:.9e}|jacobi_ms={jacobi_ms:.6}|ilu_status={:?}|ilu_iterations={}|ilu_ratio={ilu_ratio:.9e}|ilu_ms={ilu_ms:.6}|ilu_over_jacobi_residual={:.9e}|ilu_over_jacobi_probe_ms={:.9e}",
            jacobi_outcome.status,
            jacobi_outcome.iterations,
            ilu_outcome.status,
            ilu_outcome.iterations,
            ilu_ratio / jacobi_ratio.max(f64::MIN_POSITIVE),
            ilu_ms / jacobi_ms.max(f64::MIN_POSITIVE),
        );
    }

    Ok(())
}
