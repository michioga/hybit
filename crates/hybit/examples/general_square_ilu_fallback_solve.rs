use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use hybit::{
    read_matrix_market, GeneralSquareOptions, GeneralSquarePreconditionerPolicy, HybitSolver,
    MatrixProblemClass, SolverOptions,
};

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    restart: usize,
    max_iterations: usize,
    relative_tolerance: f64,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut restart = 30usize;
        let mut max_iterations = 300usize;
        let mut relative_tolerance = 1.0e-8f64;

        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => {
                    matrix = Some(PathBuf::from(
                        it.next().ok_or("missing value after --matrix")?,
                    ));
                }
                "--restart" => {
                    restart = it.next().ok_or("missing value after --restart")?.parse()?;
                }
                "--max-iters" => {
                    max_iterations = it
                        .next()
                        .ok_or("missing value after --max-iters")?
                        .parse()?;
                }
                "--tol" => {
                    relative_tolerance = it.next().ok_or("missing value after --tol")?.parse()?;
                }
                "-h" | "--help" => {
                    print_usage();
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
        if max_iterations == 0 {
            return Err("--max-iters must be > 0".into());
        }
        if !relative_tolerance.is_finite() || relative_tolerance <= 0.0 {
            return Err("--tol must be finite and > 0".into());
        }

        Ok(Self {
            matrix,
            restart,
            max_iterations,
            relative_tolerance,
        })
    }
}

fn print_usage() {
    println!("HyBIT GeneralSquare ILU(0)-fallback bounded solve probe");
    println!();
    println!("Usage:");
    println!(
        "  general_square_ilu_fallback_solve --matrix A.mtx [--restart 30] [--max-iters 300] [--tol 1e-8]"
    );
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn deterministic_exact(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let bits = splitmix64((i as u64) ^ 0xf5b0_2026_1004_0001);
            let unit = ((bits >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
            2.0 * unit - 1.0
        })
        .collect()
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn relative_l2_error(actual: &[f64], expected: &[f64]) -> f64 {
    let diff = actual
        .iter()
        .zip(expected)
        .map(|(&a, &e)| {
            let d = a - e;
            d * d
        })
        .sum::<f64>()
        .sqrt();
    diff / l2_norm(expected).max(f64::MIN_POSITIVE)
}

fn true_relative_residual(
    matrix: &hybit::Csr32Matrix,
    x: &[f64],
    b: &[f64],
) -> Result<f64, Box<dyn Error>> {
    let ax = matrix.spmv(x)?;
    let residual = ax
        .iter()
        .zip(b)
        .map(|(&a, &rhs)| {
            let d = a - rhs;
            d * d
        })
        .sum::<f64>()
        .sqrt();
    let b_norm = l2_norm(b);
    Ok(if b_norm > 0.0 {
        residual / b_norm
    } else {
        residual
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let (matrix, mm) = read_matrix_market(&args.matrix)?;

    if matrix.nrows() != matrix.ncols() {
        return Err("bounded fallback solve requires a square matrix".into());
    }

    println!(
        "HyBIT {} GeneralSquare ILU(0)-fallback bounded solve probe",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
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
    println!("restart             : {}", args.restart);
    println!("max iterations      : {}", args.max_iterations);
    println!("relative tolerance  : {:.3e}", args.relative_tolerance);

    let exact = deterministic_exact(matrix.nrows());
    let b = matrix.spmv(&exact)?;
    let b_norm = l2_norm(&b);
    if !b_norm.is_finite() || b_norm == 0.0 {
        return Err("manufactured RHS is zero or non-finite".into());
    }

    let mut solver = HybitSolver::new();
    solver.set_problem_class(MatrixProblemClass::GeneralSquare);
    solver
        .set_general_square_preconditioner_policy(GeneralSquarePreconditionerPolicy::Ilu0Fallback);
    solver.set_options(SolverOptions {
        relative_tolerance: args.relative_tolerance,
        absolute_tolerance: 0.0,
        max_iterations: args.max_iterations,
    })?;
    solver.set_general_square_options(GeneralSquareOptions {
        restart: args.restart,
        ..GeneralSquareOptions::default()
    })?;

    let analysis_start = Instant::now();
    let analysis = solver.analyze_csr32(&matrix)?;
    let analysis_ms = analysis_start.elapsed().as_secs_f64() * 1.0e3;

    let prepare_start = Instant::now();
    let mut prepared = solver.prepare_csr32(&matrix, &analysis)?;
    let prepare_ms = prepare_start.elapsed().as_secs_f64() * 1.0e3;

    let fallback_used = prepared.general_square_ilu_fallback_used();
    let preconditioner = prepared
        .general_square_preconditioner_kind()
        .ok_or("prepared GeneralSquare preconditioner kind is unavailable")?;

    let mut x = vec![0.0; matrix.nrows()];
    let solve_start = Instant::now();
    let report = prepared.solve(&matrix, &b, &mut x)?;
    let solve_wall_ms = solve_start.elapsed().as_secs_f64() * 1.0e3;

    let true_rel = true_relative_residual(&matrix, &x, &b)?;
    let forward_rel = relative_l2_error(&x, &exact);
    let reduction = report.final_residual / report.initial_residual.max(f64::MIN_POSITIVE);

    if !true_rel.is_finite() || !forward_rel.is_finite() || !reduction.is_finite() {
        return Err("bounded fallback solve produced non-finite verification metrics".into());
    }
    if report.converged() && true_rel > args.relative_tolerance * 100.0 {
        return Err(format!(
            "solver reported convergence but independently verified residual is too large: {true_rel:e}"
        )
        .into());
    }

    println!();
    println!("effective precond.  : {preconditioner:?}");
    println!("fallback used       : {fallback_used}");
    println!("status              : {:?}", report.status);
    println!("iterations          : {}", report.iterations);
    println!("initial residual    : {:.9e}", report.initial_residual);
    println!("final residual      : {:.9e}", report.final_residual);
    println!("reported rel resid. : {:.9e}", report.relative_residual);
    println!("true rel residual   : {:.9e}", true_rel);
    println!("final/initial       : {:.9e}", reduction);
    println!("forward rel error   : {:.9e}", forward_rel);
    println!("analysis            : {:.3} ms", analysis_ms);
    println!("prepare             : {:.3} ms", prepare_ms);
    println!("solve wall          : {:.3} ms", solve_wall_ms);

    println!(
        "FALLBACK_SOLVE|status={:?}|iterations={}|preconditioner={preconditioner:?}|fallback_used={fallback_used}|initial_residual={:.9e}|final_residual={:.9e}|reported_relative_residual={:.9e}|true_relative_residual={:.9e}|final_over_initial={:.9e}|forward_relative_error={:.9e}|analysis_ms={analysis_ms:.6}|prepare_ms={prepare_ms:.6}|solve_wall_ms={solve_wall_ms:.6}|restart={}|max_iterations={}|tolerance={:.9e}|n={}|nnz={}",
        report.status,
        report.iterations,
        report.initial_residual,
        report.final_residual,
        report.relative_residual,
        true_rel,
        reduction,
        forward_rel,
        args.restart,
        args.max_iterations,
        args.relative_tolerance,
        matrix.nrows(),
        matrix.nnz(),
    );

    Ok(())
}
