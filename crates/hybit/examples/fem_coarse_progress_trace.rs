use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hybit::{
    read_matrix_market, SolverOptions, TwoLevelAggregation, TwoLevelBasis,
    TwoLevelBlockJacobiPreconditioner, TwoLevelCoarseApplyPolicy, TwoLevelTransferApplyPolicy,
    TwoLevelTransferOptions, TwoLevelTransferStoragePolicy, TwoLevelTransferValueStoragePolicy,
};
use hybit_krylov::{pcg_continue_with_workspace, pcg_start_with_workspace, PcgWorkspace};

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    rhs: Option<PathBuf>,
    dofs_per_node: usize,
    target_coarse_dimension: usize,
    relative_tolerance: f64,
    max_iterations: usize,
    checkpoints: Vec<usize>,
    reference_window: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut rhs = None;
        let mut dofs_per_node = 3usize;
        let mut target_coarse_dimension = 1536usize;
        let mut relative_tolerance: f64 = 1.0e-8;
        let mut max_iterations = 3000usize;
        let mut checkpoints = vec![12, 24, 48, 96, 192, 384, 768, 1536, 3000];
        let mut reference_window = 12usize;

        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => matrix = Some(PathBuf::from(next_value(&mut it, "--matrix")?)),
                "--rhs" => rhs = Some(PathBuf::from(next_value(&mut it, "--rhs")?)),
                "--coarse-dofs" => dofs_per_node = next_value(&mut it, "--coarse-dofs")?.parse()?,
                "--coarse-target" => {
                    target_coarse_dimension = next_value(&mut it, "--coarse-target")?.parse()?
                }
                "--tol" => relative_tolerance = next_value(&mut it, "--tol")?.parse()?,
                "--max-iters" => max_iterations = next_value(&mut it, "--max-iters")?.parse()?,
                "--checkpoints" => {
                    checkpoints = parse_checkpoints(&next_value(&mut it, "--checkpoints")?)?
                }
                "--reference-window" => {
                    reference_window = next_value(&mut it, "--reference-window")?.parse()?
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
        if !relative_tolerance.is_finite() || relative_tolerance <= 0.0 {
            return Err("--tol must be finite and > 0".into());
        }
        if max_iterations == 0 {
            return Err("--max-iters must be > 0".into());
        }
        if reference_window == 0 {
            return Err("--reference-window must be > 0".into());
        }

        checkpoints.retain(|&value| value <= max_iterations);
        if checkpoints.last().copied() != Some(max_iterations) {
            checkpoints.push(max_iterations);
        }
        checkpoints.sort_unstable();
        checkpoints.dedup();

        Ok(Self {
            matrix,
            rhs,
            dofs_per_node,
            target_coarse_dimension,
            relative_tolerance,
            max_iterations,
            checkpoints,
            reference_window,
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

fn parse_checkpoints(value: &str) -> Result<Vec<usize>, Box<dyn Error>> {
    let mut checkpoints = Vec::new();
    for token in value.split(',') {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            continue;
        }
        let checkpoint: usize = trimmed.parse()?;
        if checkpoint == 0 {
            return Err("--checkpoints values must be > 0".into());
        }
        checkpoints.push(checkpoint);
    }
    if checkpoints.is_empty() {
        return Err("--checkpoints must contain at least one positive integer".into());
    }
    Ok(checkpoints)
}

fn print_usage() {
    println!("HyBIT algebraic-coarse progress trace");
    println!("Usage:");
    println!("  cargo run --release -p hybit --example fem_coarse_progress_trace -- --matrix FILE.mtx [options]");
    println!("Options:");
    println!("  --rhs FILE             whitespace-separated RHS; default b=A*1");
    println!("  --coarse-dofs N        block DOFs per node (default 3)");
    println!("  --coarse-target N      target coarse dimension (default 1536)");
    println!("  --tol X                relative tolerance (default 1e-8)");
    println!("  --max-iters N          total PCG iteration budget (default 3000)");
    println!("  --checkpoints LIST     cumulative trace points, comma separated");
    println!("                         default 12,24,48,96,192,384,768,1536,3000");
    println!("  --reference-window N   normalize segment decay to N iterations (default 12)");
}

fn load_rhs(path: &Path, n: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut values = Vec::with_capacity(n);
    for (line_index, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim_start_matches('\u{feff}').trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('%') {
            continue;
        }
        for (token_index, token) in line.split_whitespace().enumerate() {
            let value = token.parse::<f64>().map_err(|error| {
                format!(
                    "{}: invalid RHS float at line {}, token {}: {:?} ({error})",
                    path.display(),
                    line_index + 1,
                    token_index + 1,
                    token
                )
            })?;
            if !value.is_finite() {
                return Err(format!(
                    "{}: non-finite RHS value at line {}, token {}",
                    path.display(),
                    line_index + 1,
                    token_index + 1
                )
                .into());
            }
            values.push(value);
        }
    }
    if values.len() != n {
        return Err(format!(
            "{}: RHS length mismatch: expected {n}, got {}",
            path.display(),
            values.len()
        )
        .into());
    }
    Ok(values)
}

fn norm2(x: &[f64]) -> f64 {
    x.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn verified_relative_residual(
    matrix: &hybit::Csr32Matrix,
    b: &[f64],
    x: &[f64],
) -> Result<f64, Box<dyn Error>> {
    let ax = matrix.spmv(x)?;
    let residual = b
        .iter()
        .zip(ax)
        .map(|(&bi, axi)| {
            let delta = bi - axi;
            delta * delta
        })
        .sum::<f64>()
        .sqrt();
    Ok(residual / norm2(b).max(f64::MIN_POSITIVE))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    println!(
        "HyBIT {} algebraic-coarse progress trace",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix             : {}", args.matrix.display());

    let load_start = Instant::now();
    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1.0e3;
    println!(
        "Matrix Market      : {:?}, {} input entries -> {} CSR nnz",
        mm.symmetry, mm.input_entries, mm.csr_nnz
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

    let b = if let Some(path) = args.rhs.as_deref() {
        println!("RHS                : {}", path.display());
        load_rhs(path, matrix.nrows())?
    } else {
        println!("RHS                : generated as b=A*1");
        matrix.spmv(&vec![1.0; matrix.ncols()])?
    };
    let b_norm = norm2(&b).max(f64::MIN_POSITIVE);

    let node_count = matrix.nrows() / args.dofs_per_node;
    let max_aggregates = (args.target_coarse_dimension / args.dofs_per_node).max(1);
    let aggregate_nodes = node_count.div_ceil(max_aggregates).max(1);
    println!("coarse DOFs/node   : {}", args.dofs_per_node);
    println!("coarse target dim  : {}", args.target_coarse_dimension);
    println!("coarse aggregation : Graph");
    println!("coarse basis       : JacobiSmoothed");
    println!("coarse transfer    : Parallel / Wide / F32");
    println!("coarse apply req.  : Auto");
    println!("trace checkpoints  : {:?}", args.checkpoints);
    println!("reference window   : {} iterations", args.reference_window);

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
    let setup_ms = setup_start.elapsed().as_secs_f64() * 1.0e3;
    println!("coarse dimension   : {}", coarse.coarse_dimension());
    println!("coarse apply eff.  : {:?}", coarse.coarse_apply_policy());
    println!(
        "coarse memory      : {:.3} MiB",
        coarse.factor_bytes() as f64 / (1024.0 * 1024.0)
    );
    println!("coarse setup       : {:.3} ms", setup_ms);

    let options = SolverOptions {
        relative_tolerance: args.relative_tolerance,
        absolute_tolerance: 0.0,
        max_iterations: args.max_iterations,
    };
    let mut x = vec![0.0; matrix.ncols()];
    let mut workspace = PcgWorkspace::new(matrix.nrows());
    let mut session =
        pcg_start_with_workspace(&matrix, &coarse, &b, &mut x, options, &mut workspace)?;

    println!();
    println!("TRACE columns: checkpoint segment total segment_ratio equiv_ref_ratio per_iter_rho rel_res status");
    let solve_start = Instant::now();
    let mut total_iterations = 0usize;
    let mut final_status = hybit::SolveStatus::MaxIterations;
    let mut final_residual = session.final_residual();

    for &checkpoint in &args.checkpoints {
        if total_iterations >= args.max_iterations || session.is_finished() {
            break;
        }
        let segment_budget = checkpoint.saturating_sub(total_iterations);
        if segment_budget == 0 {
            continue;
        }
        let segment = pcg_continue_with_workspace(
            &matrix,
            &coarse,
            &b,
            &mut x,
            segment_budget,
            &mut session,
            &mut workspace,
        )?;
        total_iterations += segment.iterations;
        final_status = segment.status;
        final_residual = segment.final_residual;

        let segment_ratio = if segment.initial_residual > 0.0 {
            segment.final_residual / segment.initial_residual
        } else {
            0.0
        };
        let per_iter_rho = if segment.iterations > 0 && segment_ratio > 0.0 {
            segment_ratio.powf(1.0 / segment.iterations as f64)
        } else if segment_ratio == 0.0 {
            0.0
        } else {
            1.0
        };
        let equiv_reference_ratio = if segment.iterations > 0 {
            per_iter_rho.powf(args.reference_window as f64)
        } else {
            1.0
        };
        let relative_residual = segment.final_residual / b_norm;

        println!(
            "TRACE checkpoint={} segment={} total={} segment_ratio={:.9e} equiv_ref_ratio={:.9e} per_iter_rho={:.9e} rel_res={:.9e} status={:?}",
            checkpoint,
            segment.iterations,
            total_iterations,
            segment_ratio,
            equiv_reference_ratio,
            per_iter_rho,
            relative_residual,
            segment.status
        );

        if segment.status != hybit::SolveStatus::MaxIterations {
            break;
        }
    }

    let solve_ms = solve_start.elapsed().as_secs_f64() * 1.0e3;
    let verified = verified_relative_residual(&matrix, &b, &x)?;
    println!();
    println!("status             : {:?}", final_status);
    println!("iterations         : {}", total_iterations);
    println!("reported residual  : {:.6e}", final_residual / b_norm);
    println!("verified residual  : {:.6e}", verified);
    println!("solver time        : {:.3} ms", solve_ms);
    println!(
        "total wall         : {:.3} ms",
        load_ms + setup_ms + solve_ms
    );

    Ok(())
}
