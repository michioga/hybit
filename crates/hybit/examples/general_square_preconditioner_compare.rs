use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use hybit::{
    read_matrix_market, Csr32Matrix, GeneralSquareOptions, GeneralSquarePreconditionerPolicy,
    HybitPreparedSystem, HybitSolver, MatrixProblemClass, SolveReport, SolveStatus, SolverOptions,
};

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    rhs_count: usize,
    relative_tolerance: f64,
    max_iterations: usize,
    restart: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut rhs_count = 5usize;
        let mut relative_tolerance = 1.0e-8f64;
        let mut max_iterations = 5000usize;
        let mut restart = 30usize;

        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => {
                    matrix = Some(PathBuf::from(
                        it.next().ok_or("missing value after --matrix")?,
                    ));
                }
                "--rhs-count" => {
                    rhs_count = it
                        .next()
                        .ok_or("missing value after --rhs-count")?
                        .parse()?;
                }
                "--tol" => {
                    relative_tolerance = it.next().ok_or("missing value after --tol")?.parse()?;
                }
                "--max-iters" => {
                    max_iterations = it
                        .next()
                        .ok_or("missing value after --max-iters")?
                        .parse()?;
                }
                "--restart" => {
                    restart = it.next().ok_or("missing value after --restart")?.parse()?;
                }
                "-h" | "--help" => {
                    println!(
                        "Usage: general_square_preconditioner_compare --matrix A.mtx [--rhs-count 5] [--tol 1e-8] [--max-iters 5000] [--restart 30]"
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
        if rhs_count == 0 {
            return Err("--rhs-count must be > 0".into());
        }
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
            rhs_count,
            relative_tolerance,
            max_iterations,
            restart,
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

fn exact_solution(n: usize, rhs_index: usize) -> (Vec<f64>, &'static str) {
    match rhs_index {
        0 => (vec![1.0; n], "ones"),
        1 => (
            (0..n)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
            "alternating",
        ),
        _ => {
            let salt = (rhs_index as u64).wrapping_mul(0xd6e8_feb8_6659_fd93);
            let x = (0..n)
                .map(|i| {
                    let bits = splitmix64((i as u64) ^ salt);
                    let unit = ((bits >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
                    2.0 * unit - 1.0
                })
                .collect();
            (x, "hashed")
        }
    }
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

fn prepare(
    matrix: &Csr32Matrix,
    args: &Args,
    policy: GeneralSquarePreconditionerPolicy,
) -> Result<HybitPreparedSystem, Box<dyn Error>> {
    let mut solver = HybitSolver::new();
    solver.set_problem_class(MatrixProblemClass::GeneralSquare);
    solver.set_options(SolverOptions {
        relative_tolerance: args.relative_tolerance,
        absolute_tolerance: 0.0,
        max_iterations: args.max_iterations,
    })?;
    solver.set_general_square_preconditioner_policy(policy);
    solver.set_general_square_options(GeneralSquareOptions {
        restart: args.restart,
        ..GeneralSquareOptions::default()
    })?;

    let analysis = solver.analyze_csr32(matrix)?;
    Ok(solver.prepare_csr32(matrix, &analysis)?)
}

struct TimedSolve {
    report: SolveReport,
    x: Vec<f64>,
    wall_ms: f64,
}

fn solve_prepared(
    prepared: &mut HybitPreparedSystem,
    matrix: &Csr32Matrix,
    b: &[f64],
) -> Result<TimedSolve, Box<dyn Error>> {
    let mut x = vec![0.0; matrix.ncols()];
    let start = Instant::now();
    let report = prepared.solve(matrix, b, &mut x)?;
    let wall_ms = start.elapsed().as_secs_f64() * 1.0e3;
    Ok(TimedSolve { report, x, wall_ms })
}

fn print_setup(name: &str, prepared: &HybitPreparedSystem) {
    println!(
        "F7A_SETUP|preconditioner={name}|analysis_ms={:.6}|prepare_ms={:.6}|setup_ms={:.6}|preconditioner_bytes={}|workspace_bytes={}|adjusted_pivots={}",
        prepared.analysis_seconds() * 1.0e3,
        prepared.prepare_seconds() * 1.0e3,
        (prepared.analysis_seconds() + prepared.prepare_seconds()) * 1.0e3,
        prepared.general_square_preconditioner_bytes(),
        prepared.krylov_workspace_bytes(),
        prepared.general_square_ilu_adjusted_pivots(),
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;

    println!(
        "HyBIT {} GeneralSquare F7a Jacobi-vs-ILU(0) prepared comparison",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!("RHS count           : {}", args.rhs_count);
    println!("restart             : {}", args.restart);
    println!("max iterations      : {}", args.max_iterations);
    println!("relative tolerance  : {:.3e}", args.relative_tolerance);

    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    if matrix.nrows() != matrix.ncols() {
        return Err("F7a requires a square matrix".into());
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

    let mut jacobi = prepare(&matrix, &args, GeneralSquarePreconditionerPolicy::Jacobi)?;
    let mut ilu = prepare(&matrix, &args, GeneralSquarePreconditionerPolicy::Ilu0)?;

    print_setup("jacobi", &jacobi);
    print_setup("ilu0", &ilu);

    let jacobi_setup_ms = (jacobi.analysis_seconds() + jacobi.prepare_seconds()) * 1.0e3;
    let ilu_setup_ms = (ilu.analysis_seconds() + ilu.prepare_seconds()) * 1.0e3;

    let mut jacobi_cumulative_solve_ms = 0.0f64;
    let mut ilu_cumulative_solve_ms = 0.0f64;
    let mut jacobi_all_converged = true;
    let mut ilu_all_converged = true;
    let mut break_even_rhs = None;

    for rhs_index in 0..args.rhs_count {
        let rhs = rhs_index + 1;
        let (exact, family) = exact_solution(matrix.ncols(), rhs_index);
        let b = matrix.spmv(&exact)?;

        let (jacobi_run, ilu_run) = if rhs_index % 2 == 0 {
            let j = solve_prepared(&mut jacobi, &matrix, &b)?;
            let i = solve_prepared(&mut ilu, &matrix, &b)?;
            (j, i)
        } else {
            let i = solve_prepared(&mut ilu, &matrix, &b)?;
            let j = solve_prepared(&mut jacobi, &matrix, &b)?;
            (j, i)
        };

        let jacobi_verified = verified_relative_residual(&matrix, &b, &jacobi_run.x)?;
        let ilu_verified = verified_relative_residual(&matrix, &b, &ilu_run.x)?;
        let jacobi_xerr = relative_error(&jacobi_run.x, &exact);
        let ilu_xerr = relative_error(&ilu_run.x, &exact);

        jacobi_all_converged &= jacobi_run.report.status == SolveStatus::Converged;
        ilu_all_converged &= ilu_run.report.status == SolveStatus::Converged;

        println!(
            "F7A_RESULT|rhs={rhs}|family={family}|preconditioner=jacobi|status={:?}|iterations={}|reported_residual={:.9e}|verified_residual={jacobi_verified:.9e}|x_error={jacobi_xerr:.9e}|solve_ms={:.6}|sequence={}|reused={}",
            jacobi_run.report.status,
            jacobi_run.report.iterations,
            jacobi_run.report.relative_residual,
            jacobi_run.wall_ms,
            jacobi_run.report.solve_sequence,
            jacobi_run.report.preconditioner_reused,
        );
        println!(
            "F7A_RESULT|rhs={rhs}|family={family}|preconditioner=ilu0|status={:?}|iterations={}|reported_residual={:.9e}|verified_residual={ilu_verified:.9e}|x_error={ilu_xerr:.9e}|solve_ms={:.6}|sequence={}|reused={}",
            ilu_run.report.status,
            ilu_run.report.iterations,
            ilu_run.report.relative_residual,
            ilu_run.wall_ms,
            ilu_run.report.solve_sequence,
            ilu_run.report.preconditioner_reused,
        );

        jacobi_cumulative_solve_ms += jacobi_run.wall_ms;
        ilu_cumulative_solve_ms += ilu_run.wall_ms;

        let jacobi_total_ms = jacobi_setup_ms + jacobi_cumulative_solve_ms;
        let ilu_total_ms = ilu_setup_ms + ilu_cumulative_solve_ms;
        let total_ratio = ilu_total_ms / jacobi_total_ms.max(f64::MIN_POSITIVE);
        let solve_ratio =
            ilu_cumulative_solve_ms / jacobi_cumulative_solve_ms.max(f64::MIN_POSITIVE);

        if break_even_rhs.is_none() && ilu_total_ms < jacobi_total_ms {
            break_even_rhs = Some(rhs);
        }

        println!(
            "F7A_CUMULATIVE|rhs_count={rhs}|jacobi_all_converged={jacobi_all_converged}|ilu_all_converged={ilu_all_converged}|jacobi_setup_ms={jacobi_setup_ms:.6}|ilu_setup_ms={ilu_setup_ms:.6}|jacobi_cumulative_solve_ms={jacobi_cumulative_solve_ms:.6}|ilu_cumulative_solve_ms={ilu_cumulative_solve_ms:.6}|jacobi_total_ms={jacobi_total_ms:.6}|ilu_total_ms={ilu_total_ms:.6}|ilu_over_jacobi_solve={solve_ratio:.9e}|ilu_over_jacobi_total={total_ratio:.9e}"
        );

        if jacobi_run.report.status == SolveStatus::Converged
            && jacobi_verified > args.relative_tolerance * 10.0
        {
            return Err(format!(
                "Jacobi reported convergence but verified residual is {jacobi_verified:e}"
            )
            .into());
        }
        if ilu_run.report.status == SolveStatus::Converged
            && ilu_verified > args.relative_tolerance * 10.0
        {
            return Err(format!(
                "ILU0 reported convergence but verified residual is {ilu_verified:e}"
            )
            .into());
        }
    }

    println!(
        "F7A_BREAK_EVEN|rhs={}",
        break_even_rhs
            .map(|rhs| rhs.to_string())
            .unwrap_or_else(|| "none".to_string())
    );

    Ok(())
}
