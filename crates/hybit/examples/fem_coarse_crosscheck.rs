use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use hybit::{
    pcg, read_matrix_market, JacobiPreconditioner, SolverOptions, TwoLevelAggregation,
    TwoLevelBasis, TwoLevelBlockJacobiPreconditioner, TwoLevelCoarseApplyPolicy,
    TwoLevelTransferApplyPolicy, TwoLevelTransferOptions, TwoLevelTransferStoragePolicy,
    TwoLevelTransferValueStoragePolicy,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    DirectCoarse,
    ProbeThenCoarse,
}

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    dofs_per_node: usize,
    target_coarse_dimension: usize,
    relative_tolerance: f64,
    max_iterations: usize,
    probe_iterations: usize,
    mode: Mode,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut dofs_per_node = 3usize;
        let mut target_coarse_dimension = 1536usize;
        let mut relative_tolerance = 1.0e-8;
        let mut max_iterations = 3000usize;
        let mut probe_iterations = 12usize;
        let mut mode = Mode::DirectCoarse;

        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => matrix = Some(PathBuf::from(next_value(&mut it, "--matrix")?)),
                "--coarse-dofs" => dofs_per_node = next_value(&mut it, "--coarse-dofs")?.parse()?,
                "--coarse-target" => {
                    target_coarse_dimension = next_value(&mut it, "--coarse-target")?.parse()?
                }
                "--tol" => relative_tolerance = next_value(&mut it, "--tol")?.parse()?,
                "--max-iters" => max_iterations = next_value(&mut it, "--max-iters")?.parse()?,
                "--probe-iters" => {
                    probe_iterations = next_value(&mut it, "--probe-iters")?.parse()?
                }
                "--mode" => {
                    mode = match next_value(&mut it, "--mode")?.as_str() {
                        "direct" | "coarse-only" => Mode::DirectCoarse,
                        "probe" | "probe-coarse" => Mode::ProbeThenCoarse,
                        other => {
                            return Err(
                                format!("unknown mode '{other}'; use direct|probe-coarse").into()
                            )
                        }
                    }
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
        if dofs_per_node == 0 {
            return Err("--coarse-dofs must be > 0".into());
        }
        if target_coarse_dimension < dofs_per_node {
            return Err("--coarse-target must be >= --coarse-dofs".into());
        }
        if max_iterations == 0 {
            return Err("--max-iters must be > 0".into());
        }
        if probe_iterations >= max_iterations {
            return Err("--probe-iters must be smaller than --max-iters".into());
        }

        Ok(Self {
            matrix,
            dofs_per_node,
            target_coarse_dimension,
            relative_tolerance,
            max_iterations,
            probe_iterations,
            mode,
        })
    }
}

fn next_value<I>(it: &mut I, flag: &str) -> Result<String, Box<dyn Error>>
where
    I: Iterator<Item = String>,
{
    it.next()
        .ok_or_else(|| format!("missing value after {flag}").into())
}

fn print_usage() {
    println!(
        "Usage: cargo run --release --example fem_coarse_crosscheck -- --matrix FILE.mtx [options]"
    );
    println!("  --coarse-dofs N      block DOFs per node (default: 3)");
    println!("  --coarse-target N    target coarse dimension (default: 1536)");
    println!("  --tol X              relative tolerance (default: 1e-8)");
    println!("  --max-iters N        total PCG iteration budget (default: 3000)");
    println!("  --probe-iters N      Jacobi probe iterations in probe-coarse mode (default: 12)");
    println!("  --mode direct|probe-coarse (default: direct)");
}

fn norm2(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn relative_error_to_ones(x: &[f64]) -> f64 {
    let err = x
        .iter()
        .map(|&v| {
            let d = v - 1.0;
            d * d
        })
        .sum::<f64>()
        .sqrt();
    err / (x.len() as f64).sqrt()
}

fn verified_relative_residual(
    matrix: &hybit::Csr32Matrix,
    b: &[f64],
    x: &[f64],
) -> Result<f64, Box<dyn Error>> {
    let ax = matrix.spmv(x)?;
    let r = b
        .iter()
        .zip(ax)
        .map(|(&bi, axi)| {
            let d = bi - axi;
            d * d
        })
        .sum::<f64>()
        .sqrt();
    Ok(r / norm2(b).max(f64::MIN_POSITIVE))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    println!(
        "HyBIT {} algebraic-coarse cross-check",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix             : {}", args.matrix.display());
    println!("mode               : {:?}", args.mode);

    let load_start = Instant::now();
    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1.0e3;
    println!(
        "Matrix Market      : {:?}, {} input entries -> {} CSR nnz",
        mm.symmetry, mm.input_entries, mm.csr_nnz
    );
    println!(
        "coalesced / zeros  : {} / {}",
        mm.duplicate_entries_combined, mm.zero_entries_removed
    );
    println!(
        "dimensions         : {} x {}",
        matrix.nrows(),
        matrix.ncols()
    );
    println!("load time          : {:.3} ms", load_ms);

    if matrix.nrows() != matrix.ncols() {
        return Err("matrix must be square".into());
    }
    if matrix.nrows() % args.dofs_per_node != 0 {
        return Err("matrix dimension is not divisible by --coarse-dofs".into());
    }

    let exact = vec![1.0; matrix.ncols()];
    let b = matrix.spmv(&exact)?;
    println!("RHS                : generated as b=A*1 (known exact solution)");

    let node_count = matrix.nrows() / args.dofs_per_node;
    let max_aggregates = (args.target_coarse_dimension / args.dofs_per_node).max(1);
    let aggregate_nodes = node_count.div_ceil(max_aggregates).max(1);
    println!("coarse DOFs/node   : {}", args.dofs_per_node);
    println!("coarse target dim  : {}", args.target_coarse_dimension);
    println!("aggregate target   : {} nodes", aggregate_nodes);
    println!("coarse aggregation : Graph");
    println!("coarse basis       : JacobiSmoothed");
    println!("coarse transfer    : Parallel / Wide / F32");
    println!("coarse apply req.  : Auto");

    let setup_start = Instant::now();
    let coarse =
        TwoLevelBlockJacobiPreconditioner::from_csr32_with_aggregation_basis_and_transfer_options(
            &matrix,
            args.dofs_per_node,
            aggregate_nodes,
            TwoLevelAggregation::Graph,
            TwoLevelBasis::JacobiSmoothed,
            TwoLevelCoarseApplyPolicy::Auto,
            TwoLevelTransferOptions {
                apply_policy: TwoLevelTransferApplyPolicy::Parallel,
                storage_policy: TwoLevelTransferStoragePolicy::Wide,
                value_storage_policy: TwoLevelTransferValueStoragePolicy::F32,
            },
        )?;
    let coarse_setup_ms = setup_start.elapsed().as_secs_f64() * 1.0e3;
    println!("coarse dimension   : {}", coarse.coarse_dimension());
    println!("coarse apply eff.  : {:?}", coarse.coarse_apply_policy());
    println!("coarse aggregate   : {} nodes", coarse.aggregate_nodes());
    println!(
        "coarse memory      : {:.3} MiB",
        coarse.factor_bytes() as f64 / (1024.0 * 1024.0)
    );
    println!("coarse setup       : {:.3} ms", coarse_setup_ms);
    println!("transfer nnz       : {}", coarse.transfer_nnz());
    println!(
        "transfer index     : {:.3} MiB",
        coarse.transfer_index_bytes() as f64 / (1024.0 * 1024.0)
    );
    println!(
        "transfer values    : {:.3} MiB",
        coarse.transfer_value_bytes() as f64 / (1024.0 * 1024.0)
    );

    let options = SolverOptions {
        relative_tolerance: args.relative_tolerance,
        absolute_tolerance: 0.0,
        max_iterations: args.max_iterations,
    };
    let mut x = vec![0.0; matrix.ncols()];

    let solve_start = Instant::now();
    let (status, total_iterations, reported_final_residual, probe_ms) = match args.mode {
        Mode::DirectCoarse => {
            let out = pcg(&matrix, &coarse, &b, &mut x, options)?;
            (out.status, out.iterations, out.final_residual, 0.0)
        }
        Mode::ProbeThenCoarse => {
            let jacobi = JacobiPreconditioner::from_csr32(&matrix)?;
            let mut probe_options = options;
            probe_options.max_iterations = args.probe_iterations;
            let probe_start = Instant::now();
            let probe = pcg(&matrix, &jacobi, &b, &mut x, probe_options)?;
            let probe_ms = probe_start.elapsed().as_secs_f64() * 1.0e3;
            if probe.status == hybit::SolveStatus::Converged {
                (
                    probe.status,
                    probe.iterations,
                    probe.final_residual,
                    probe_ms,
                )
            } else {
                let mut coarse_options = options;
                coarse_options.max_iterations =
                    args.max_iterations.saturating_sub(probe.iterations);
                let coarse_out = pcg(&matrix, &coarse, &b, &mut x, coarse_options)?;
                (
                    coarse_out.status,
                    probe.iterations + coarse_out.iterations,
                    coarse_out.final_residual,
                    probe_ms,
                )
            }
        }
    };
    let solve_ms = solve_start.elapsed().as_secs_f64() * 1.0e3;
    let bnorm = norm2(&b).max(f64::MIN_POSITIVE);
    let verified = verified_relative_residual(&matrix, &b, &x)?;

    println!("status             : {:?}", status);
    println!("iterations         : {}", total_iterations);
    println!(
        "reported residual  : {:.6e}",
        reported_final_residual / bnorm
    );
    println!("verified residual  : {:.6e}", verified);
    if args.mode == Mode::ProbeThenCoarse {
        println!(
            "probe              : {} iters, {:.3} ms",
            args.probe_iterations, probe_ms
        );
    }
    println!("solver time        : {:.3} ms", solve_ms);
    println!(
        "total wall         : {:.3} ms",
        load_ms + coarse_setup_ms + solve_ms
    );
    println!("relative x error   : {:.6e}", relative_error_to_ones(&x));

    Ok(())
}
