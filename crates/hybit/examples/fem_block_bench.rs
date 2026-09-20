use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use hybit::{
    analyze_csr32, pcg, read_matrix_market, BlockJacobiPreconditioner, Csr32Matrix, SolverOptions,
};

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    relative_tolerance: f64,
    max_iterations: usize,
    block_size: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut relative_tolerance: f64 = 1.0e-8;
        let mut max_iterations = 3000usize;
        let mut block_size = 3usize;
        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => matrix = Some(PathBuf::from(next_value(&mut it, "--matrix")?)),
                "--tol" => relative_tolerance = next_value(&mut it, "--tol")?.parse()?,
                "--max-iters" => max_iterations = next_value(&mut it, "--max-iters")?.parse()?,
                "--block-size" => block_size = next_value(&mut it, "--block-size")?.parse()?,
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                other if !other.starts_with('-') && matrix.is_none() => {
                    matrix = Some(PathBuf::from(other))
                }
                other => return Err(format!("unknown argument '{other}'").into()),
            }
        }
        let matrix = matrix.ok_or("missing matrix path; use --matrix FILE.mtx")?;
        if !relative_tolerance.is_finite() || relative_tolerance <= 0.0 {
            return Err("--tol must be finite and > 0".into());
        }
        if max_iterations == 0 {
            return Err("--max-iters must be > 0".into());
        }
        if block_size == 0 {
            return Err("--block-size must be > 0".into());
        }
        Ok(Self {
            matrix,
            relative_tolerance,
            max_iterations,
            block_size,
        })
    }
}

fn next_value<I: Iterator<Item = String>>(
    it: &mut I,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    it.next()
        .ok_or_else(|| format!("missing value after {flag}").into())
}

fn print_usage() {
    println!("HyBIT block-Jacobi FEM benchmark");
    println!(
        "Usage: fem_block_bench --matrix K.mtx [--tol 1e-8] [--max-iters 3000] [--block-size 3]"
    );
}

fn norm2(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn verified_relative_residual(
    a: &Csr32Matrix,
    b: &[f64],
    x: &[f64],
) -> Result<f64, Box<dyn Error>> {
    let ax = a.spmv(x)?;
    let sum = b
        .iter()
        .zip(ax.iter())
        .map(|(&bi, &ai)| {
            let r = bi - ai;
            r * r
        })
        .sum::<f64>();
    let denom = norm2(b);
    Ok(if denom == 0.0 {
        sum.sqrt()
    } else {
        sum.sqrt() / denom
    })
}

fn relative_error_to_ones(x: &[f64]) -> f64 {
    let diff = x
        .iter()
        .map(|&xi| {
            let d = xi - 1.0;
            d * d
        })
        .sum::<f64>();
    diff.sqrt() / (x.len() as f64).sqrt().max(f64::MIN_POSITIVE)
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    println!(
        "HyBIT {} block-Jacobi FEM benchmark",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix             : {}", args.matrix.display());

    let load_start = Instant::now();
    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let profile = analyze_csr32(&matrix)?;
    println!(
        "Matrix Market      : {:?}, {} input entries -> {} CSR nnz",
        mm.symmetry, mm.input_entries, mm.csr_nnz
    );
    println!("dimensions         : {} x {}", profile.nrows, profile.ncols);
    println!("nnz                : {}", profile.nnz);
    println!(
        "CSR storage        : {:.3} MiB",
        mib(matrix.storage_bytes())
    );
    println!("load time          : {:.3} ms", load_seconds * 1.0e3);
    println!("block size         : {}", args.block_size);

    if !profile.square || !profile.full_diagonal || !profile.positive_diagonal {
        return Err(
            "block-Jacobi PCG requires a square matrix with a complete positive diagonal".into(),
        );
    }

    let ones = vec![1.0; matrix.ncols()];
    let b = matrix.spmv(&ones)?;
    let b_norm = norm2(&b).max(f64::MIN_POSITIVE);
    println!("RHS                : generated as b=A*1 (known exact solution)");

    let setup_start = Instant::now();
    let precond = BlockJacobiPreconditioner::from_csr32(&matrix, args.block_size)?;
    let setup_seconds = setup_start.elapsed().as_secs_f64();
    println!("block count        : {}", precond.block_count());
    println!(
        "factor storage     : {:.3} MiB",
        mib(precond.factor_bytes())
    );
    println!("setup              : {:.3} ms", setup_seconds * 1.0e3);

    let options = SolverOptions {
        relative_tolerance: args.relative_tolerance,
        absolute_tolerance: 0.0,
        max_iterations: args.max_iterations,
    };
    let mut x = vec![0.0; matrix.ncols()];
    let solve_start = Instant::now();
    let outcome = pcg(&matrix, &precond, &b, &mut x, options)?;
    let solve_seconds = solve_start.elapsed().as_secs_f64();
    let verified = verified_relative_residual(&matrix, &b, &x)?;

    println!();
    println!("Block-Jacobi-PCG");
    println!("status             : {:?}", outcome.status);
    println!("iterations         : {}", outcome.iterations);
    println!(
        "reported residual  : {:.6e}",
        outcome.final_residual / b_norm
    );
    println!("verified residual  : {:.6e}", verified);
    println!("relative x error   : {:.6e}", relative_error_to_ones(&x));
    println!("solve time         : {:.3} ms", solve_seconds * 1.0e3);
    println!(
        "total setup+solve  : {:.3} ms",
        (setup_seconds + solve_seconds) * 1.0e3
    );

    if !verified.is_finite() {
        return Err("non-finite independently verified residual".into());
    }
    Ok(())
}
