//! C1-H1: Generic FEM Matrix Market + RHS -> prepared HyBIT FGMRES baseline.
//! Only input/output plumbing; does not alter production routing.
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hybit::{
    GeneralSquareOptions, GeneralSquarePreconditionerPolicy, GeneralSquareRestartPolicy,
    HybitSolver, MatrixProblemClass, SolverOptions, read_matrix_market,
};

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    rhs: PathBuf,
    xref: Option<PathBuf>,
    tol: f64,
    max_iters: usize,
    precond: GeneralSquarePreconditionerPolicy,
    restart: usize,
    max_restart: usize,
    policy: GeneralSquareRestartPolicy,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut rhs = None;
        let mut xref = None;
        let mut tol = 1e-8_f64;
        let mut max_iters = 3000_usize;
        let mut precond = GeneralSquarePreconditionerPolicy::Jacobi;
        let mut restart = 30_usize;
        let mut max_restart = 70_usize;
        let mut policy = GeneralSquareRestartPolicy::Fixed;
        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => matrix = Some(PathBuf::from(required(&mut it, "--matrix")?)),
                "--rhs" => rhs = Some(PathBuf::from(required(&mut it, "--rhs")?)),
                "--xref" => xref = Some(PathBuf::from(required(&mut it, "--xref")?)),
                "--tol" => tol = required(&mut it, "--tol")?.parse()?,
                "--max-iters" => max_iters = required(&mut it, "--max-iters")?.parse()?,
                "--restart" => restart = required(&mut it, "--restart")?.parse()?,
                "--max-restart" => max_restart = required(&mut it, "--max-restart")?.parse()?,
                "--precond" => {
                    precond = match required(&mut it, "--precond")?.as_str() {
                        "jacobi" => GeneralSquarePreconditionerPolicy::Jacobi,
                        "ilu0" => GeneralSquarePreconditionerPolicy::Ilu0,
                        "ilu0-fallback" => GeneralSquarePreconditionerPolicy::Ilu0Fallback,
                        "ilu0-abtm" => GeneralSquarePreconditionerPolicy::Ilu0Abtm,
                        other => return Err(format!("unsupported --precond {other}").into()),
                    };
                }
                "--restart-policy" => {
                    policy = match required(&mut it, "--restart-policy")?.as_str() {
                        "fixed" => GeneralSquareRestartPolicy::Fixed,
                        "budget-aware" => GeneralSquareRestartPolicy::BudgetAware,
                        "escalating" => GeneralSquareRestartPolicy::Escalating,
                        other => return Err(format!("unsupported --restart-policy {other}").into()),
                    };
                }
                "-h" | "--help" => {
                    println!(
                        "Usage: fem_fgmres_bench --matrix FILE.mm --rhs FILE.rhs [--xref known_solution.txt] \\\n  [--tol 1e-8] [--max-iters 3000] [--precond jacobi|ilu0|ilu0-fallback|ilu0-abtm] \\\n  [--restart 30] [--max-restart 70] [--restart-policy fixed|budget-aware|escalating]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other}").into()),
            }
        }
        if !tol.is_finite() || tol <= 0.0 || max_iters == 0 || restart == 0 {
            return Err("invalid positive tolerance / iteration / restart setting".into());
        }
        if policy != GeneralSquareRestartPolicy::Fixed && max_restart < restart {
            return Err("max-restart must be >= restart".into());
        }
        Ok(Self {
            matrix: matrix.ok_or("--matrix required")?,
            rhs: rhs.ok_or("--rhs required; explicit physical or manufactured RHS required")?,
            xref,
            tol,
            max_iters,
            precond,
            restart,
            max_restart,
            policy,
        })
    }
}

fn required(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, Box<dyn Error>> {
    it.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn read_vector(path: &Path, n: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    let input = fs::read_to_string(path)?;
    let mut result = Vec::with_capacity(n);
    for token in input.split_whitespace() {
        // FEM Fortran E-format dumps may use E or D in vector files.
        let value: f64 = token.replace('D', "E").replace('d', "e").parse()?;
        if !value.is_finite() {
            return Err(format!("{} has a non-finite value", path.display()).into());
        }
        result.push(value);
        if result.len() > n {
            return Err(format!("{} has more than {n} values", path.display()).into());
        }
    }
    if result.len() != n {
        return Err(format!(
            "{}: expected {n} values, got {}",
            path.display(),
            result.len()
        )
        .into());
    }
    Ok(result)
}

fn stable_norm(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().fold(0.0_f64, f64::hypot)
}

fn relative_residual(a: &hybit::Csr32Matrix, b: &[f64], x: &[f64]) -> Result<f64, Box<dyn Error>> {
    let ax = a.spmv(x)?;
    let residual = stable_norm(b.iter().zip(ax).map(|(&bi, ai)| bi - ai));
    let bnorm = stable_norm(b.iter().copied());
    Ok(if bnorm == 0.0 {
        residual
    } else {
        residual / bnorm
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let load = Instant::now();
    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    if matrix.nrows() != matrix.ncols() {
        return Err(
            "this example expects a complete square matrix; do not pass unassembled MPI rank dumps"
                .into(),
        );
    }
    let b = read_vector(&args.rhs, matrix.nrows())?;
    let reference = args
        .xref
        .as_ref()
        .map(|path| read_vector(path, matrix.ncols()))
        .transpose()?;
    let load_time = load.elapsed().as_secs_f64();
    let reference_residual = reference
        .as_ref()
        .map(|x| relative_residual(&matrix, &b, x))
        .transpose()?;

    let mut solver = HybitSolver::new();
    solver.set_problem_class(MatrixProblemClass::GeneralSquare);
    solver.set_options(SolverOptions {
        relative_tolerance: args.tol,
        absolute_tolerance: 0.0,
        max_iterations: args.max_iters,
    })?;
    solver.set_general_square_preconditioner_policy(args.precond);
    solver.set_general_square_options(GeneralSquareOptions {
        restart: args.restart,
        restart_policy: args.policy,
        max_restart: args.max_restart,
        escalation_stage_iterations: args.max_iters,
    })?;
    let mut x = vec![0.0; matrix.ncols()];
    let start = Instant::now();
    let report = solver.solve_csr32(&matrix, &b, &mut x)?;
    let wall = start.elapsed().as_secs_f64();
    let verified = relative_residual(&matrix, &b, &x)?;
    if !verified.is_finite() {
        return Err("non-finite verified residual".into());
    }
    println!("HyBIT 0.10.0 FEM FGMRES benchmark (experimental)");
    println!("matrix             : {}", args.matrix.display());
    println!("RHS                : {}", args.rhs.display());
    println!(
        "dimensions         : {} x {}",
        matrix.nrows(),
        matrix.ncols()
    );
    println!("input/csr nnz      : {} / {}", mm.input_entries, mm.csr_nnz);
    println!("matrix class       : GeneralSquare (FGMRES; no SPD assertion)");
    println!("solver             : {:?}", report.solver);
    println!("preconditioner     : {:?}", report.preconditioner);
    println!(
        "restart policy     : {:?}, initial {}, max {}",
        args.policy, args.restart, args.max_restart
    );
    println!("status             : {:?}", report.status);
    println!("iterations         : {}", report.iterations);
    println!("reported residual  : {:.6e}", report.relative_residual);
    println!("verified residual  : {:.6e}", verified);
    if let Some(rr) = reference_residual {
        println!("reference residual : {:.6e}", rr);
    }
    if let Some(ref xref) = reference {
        let err = stable_norm(x.iter().zip(xref).map(|(&a, &r)| a - r));
        let denom = stable_norm(xref.iter().copied());
        println!(
            "relative x error   : {:.6e}",
            if denom == 0.0 { err } else { err / denom }
        );
    }
    println!("load time          : {:.3} ms", load_time * 1e3);
    println!(
        "analysis           : {:.3} ms",
        report.analysis_seconds * 1e3
    );
    println!(
        "prepare            : {:.3} ms",
        report.prepare_seconds * 1e3
    );
    println!("solver time        : {:.3} ms", report.solve_seconds * 1e3);
    println!("total wall         : {:.3} ms", wall * 1e3);
    // Status is semantic; unlike process errors, MaxIterations is returned as a report.
    // Supervising scripts MUST check status AND verified residual.
    Ok(())
}

#[cfg(test)]
mod fem_fgmres_bench_tests {
    use super::stable_norm;
    #[test]
    fn stable_norm_handles_large_and_small_values() {
        assert!((stable_norm([3.0, 4.0]) - 5.0).abs() < 1e-14);
        assert!(stable_norm([1e200, 1e-200]).is_finite());
    }
}
