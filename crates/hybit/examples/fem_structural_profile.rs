use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

use hybit::{
    analyze_csr32, pcg_with_workspace, read_matrix_market,
    recommend_rigid_body_aggregate_nodes, Csr32Matrix, LinearOperator, PcgWorkspace,
    RigidBodyAggregation, RigidBodyTwoLevelBlockJacobiPreconditioner, SolverOptions,
};

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    coordinates: PathBuf,
    rhs: Option<PathBuf>,
    relative_tolerance: f64,
    max_iterations: usize,
    target_coarse_dimension: usize,
    aggregation: RigidBodyAggregation,
    kernel_repeats: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut coordinates = None;
        let mut rhs = None;
        let mut relative_tolerance: f64 = 1.0e-8;
        let mut max_iterations = 3000usize;
        let mut target_coarse_dimension = 1536usize;
        let mut aggregation = RigidBodyAggregation::Graph;
        let mut kernel_repeats = 20usize;
        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => matrix = Some(PathBuf::from(next_value(&mut it, "--matrix")?)),
                "--coords" => coordinates = Some(PathBuf::from(next_value(&mut it, "--coords")?)),
                "--rhs" => rhs = Some(PathBuf::from(next_value(&mut it, "--rhs")?)),
                "--tol" => relative_tolerance = next_value(&mut it, "--tol")?.parse()?,
                "--max-iters" => max_iterations = next_value(&mut it, "--max-iters")?.parse()?,
                "--target-coarse-dim" => target_coarse_dimension = next_value(&mut it, "--target-coarse-dim")?.parse()?,
                "--kernel-repeats" => kernel_repeats = next_value(&mut it, "--kernel-repeats")?.parse()?,
                "--aggregation" => {
                    aggregation = match next_value(&mut it, "--aggregation")?.to_ascii_lowercase().as_str() {
                        "contiguous" => RigidBodyAggregation::Contiguous,
                        "graph" => RigidBodyAggregation::Graph,
                        other => return Err(format!("unknown aggregation '{other}'; use contiguous or graph").into()),
                    }
                }
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                other if !other.starts_with('-') && matrix.is_none() => matrix = Some(PathBuf::from(other)),
                other => return Err(format!("unknown argument '{other}'").into()),
            }
        }
        let matrix = matrix.ok_or("missing matrix path; use --matrix FILE.mtx")?;
        let coordinates = coordinates.unwrap_or_else(|| matrix.with_extension("coords"));
        if !relative_tolerance.is_finite() || relative_tolerance <= 0.0 {
            return Err("--tol must be finite and > 0".into());
        }
        if max_iterations == 0 { return Err("--max-iters must be > 0".into()); }
        if target_coarse_dimension < 6 { return Err("--target-coarse-dim must be >= 6".into()); }
        if kernel_repeats == 0 { return Err("--kernel-repeats must be > 0".into()); }
        Ok(Self {
            matrix,
            coordinates,
            rhs,
            relative_tolerance,
            max_iterations,
            target_coarse_dimension,
            aggregation,
            kernel_repeats,
        })
    }
}

fn next_value<I: Iterator<Item = String>>(it: &mut I, flag: &str) -> Result<String, Box<dyn Error>> {
    it.next().ok_or_else(|| format!("missing value after {flag}").into())
}

fn print_usage() {
    println!("HyBIT structural kernel profile");
    println!("Usage: fem_structural_profile --matrix K.mtx [--coords K.coords] [--rhs b.txt] [--tol 1e-8] [--max-iters 3000] [--target-coarse-dim 1536] [--aggregation graph|contiguous] [--kernel-repeats 20]");
}

fn read_coordinates(path: &Path) -> Result<Vec<[f64; 3]>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut expected = None::<usize>;
    let mut coordinates = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        let text = line.trim();
        if text.is_empty() || text.starts_with('#') { continue; }
        if expected.is_none() {
            expected = Some(text.parse::<usize>().map_err(|e| {
                format!("{}:{}: invalid coordinate count: {e}", path.display(), line_no + 1)
            })?);
            coordinates.reserve(expected.unwrap());
            continue;
        }
        let fields: Vec<&str> = text.split_whitespace().collect();
        if fields.len() != 3 {
            return Err(format!("{}:{}: expected three coordinates", path.display(), line_no + 1).into());
        }
        let xyz = [fields[0].parse::<f64>()?, fields[1].parse::<f64>()?, fields[2].parse::<f64>()?];
        if xyz.iter().any(|v| !v.is_finite()) {
            return Err(format!("{}:{}: non-finite coordinate", path.display(), line_no + 1).into());
        }
        coordinates.push(xyz);
    }
    let expected = expected.ok_or_else(|| format!("{}: missing coordinate count", path.display()))?;
    if coordinates.len() != expected {
        return Err(format!("{}: coordinate count mismatch: header says {}, read {}", path.display(), expected, coordinates.len()).into());
    }
    Ok(coordinates)
}

fn load_rhs(path: &Path, n: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut values = Vec::with_capacity(n);
    for (line_index, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim_start_matches('\u{feff}').trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('%') { continue; }
        for (token_index, token) in line.split_whitespace().enumerate() {
            let value = token.parse::<f64>().map_err(|e| {
                format!("{}: invalid RHS float at line {}, token {}: {:?} ({e})", path.display(), line_index + 1, token_index + 1, token)
            })?;
            if !value.is_finite() {
                return Err(format!("{}: non-finite RHS value at line {}, token {}", path.display(), line_index + 1, token_index + 1).into());
            }
            values.push(value);
        }
    }
    if values.len() != n {
        return Err(format!("{}: RHS length mismatch: expected {n}, got {}", path.display(), values.len()).into());
    }
    Ok(values)
}

fn norm2(x: &[f64]) -> f64 { x.iter().map(|v| v * v).sum::<f64>().sqrt() }
fn mib(bytes: usize) -> f64 { bytes as f64 / (1024.0 * 1024.0) }
fn ms_per(total_seconds: f64, repeats: usize) -> f64 { total_seconds * 1.0e3 / repeats as f64 }

fn verified_relative_residual(a: &Csr32Matrix, b: &[f64], x: &[f64]) -> Result<f64, Box<dyn Error>> {
    let ax = a.spmv(x)?;
    let rr = b.iter().zip(&ax).map(|(&bi, &ai)| { let r = bi - ai; r * r }).sum::<f64>().sqrt();
    let bn = norm2(b);
    Ok(if bn == 0.0 { rr } else { rr / bn })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    println!("HyBIT {} structural kernel profile", env!("CARGO_PKG_VERSION"));
    println!("matrix             : {}", args.matrix.display());
    println!("coordinates        : {}", args.coordinates.display());

    let load_start = Instant::now();
    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    let matrix_load = load_start.elapsed().as_secs_f64();
    let coord_start = Instant::now();
    let coordinates = read_coordinates(&args.coordinates)?;
    let coord_load = coord_start.elapsed().as_secs_f64();
    let profile = analyze_csr32(&matrix)?;

    println!("Matrix Market      : {:?}, {} input entries -> {} CSR nnz", mm.symmetry, mm.input_entries, mm.csr_nnz);
    println!("dimensions         : {} x {}", profile.nrows, profile.ncols);
    println!("nnz                : {}", profile.nnz);
    println!("CSR storage        : {:.3} MiB", mib(matrix.storage_bytes()));
    println!("matrix load        : {:.3} ms", matrix_load * 1.0e3);
    println!("coordinate nodes   : {}", coordinates.len());
    println!("coordinate load    : {:.3} ms", coord_load * 1.0e3);
    println!("aggregation        : {:?}", args.aggregation);
    println!("target coarse dim  : {}", args.target_coarse_dimension);
    println!("kernel repeats     : {}", args.kernel_repeats);

    if !profile.square || !profile.full_diagonal || !profile.positive_diagonal {
        return Err("structural PCG profile requires a square matrix with a complete positive diagonal".into());
    }
    if matrix.nrows() != coordinates.len() * 3 {
        return Err(format!("matrix/coordinate mismatch: {} matrix rows != {} coordinate nodes * 3", matrix.nrows(), coordinates.len()).into());
    }

    let b = if let Some(path) = args.rhs.as_deref() {
        println!("RHS                : {}", path.display());
        load_rhs(path, matrix.nrows())?
    } else {
        println!("RHS                : generated as b=A*1 (known exact solution)");
        matrix.spmv(&vec![1.0; matrix.ncols()])?
    };

    let aggregate_nodes = recommend_rigid_body_aggregate_nodes(coordinates.len(), args.target_coarse_dimension)?;
    let setup_start = Instant::now();
    let preconditioner = match args.aggregation {
        RigidBodyAggregation::Graph => RigidBodyTwoLevelBlockJacobiPreconditioner::from_csr32_graph(&matrix, &coordinates, aggregate_nodes)?,
        RigidBodyAggregation::Contiguous => RigidBodyTwoLevelBlockJacobiPreconditioner::from_csr32(&matrix, &coordinates, aggregate_nodes)?,
        RigidBodyAggregation::Auto => unreachable!(),
    };
    let setup_seconds = setup_start.elapsed().as_secs_f64();

    println!("aggregate target   : {} nodes", aggregate_nodes);
    println!("aggregate count    : {}", preconditioner.aggregate_count());
    println!("aggregate min/max  : {} / {} nodes", preconditioner.min_aggregate_nodes(), preconditioner.max_aggregate_nodes());
    println!("coarse dimension   : {}", preconditioner.coarse_dimension());
    println!("base factor        : {:.3} MiB", mib(preconditioner.base_factor_bytes()));
    println!("coarse factor      : {:.3} MiB", mib(preconditioner.coarse_factor_bytes()));
    println!("total prec storage : {:.3} MiB", mib(preconditioner.factor_bytes()));
    println!("setup              : {:.3} ms", setup_seconds * 1.0e3);

    let mut y = vec![0.0; matrix.nrows()];
    matrix.apply(&b, &mut y)?;
    let spmv_start = Instant::now();
    for _ in 0..args.kernel_repeats {
        matrix.apply(&b, &mut y)?;
    }
    let spmv_seconds = spmv_start.elapsed().as_secs_f64();

    let mut z = vec![0.0; matrix.nrows()];
    let component_profile = preconditioner.profile_apply_components(&b, &mut z, args.kernel_repeats)?;
    let base_seconds = component_profile.base.as_secs_f64();
    let restriction_seconds = component_profile.restriction.as_secs_f64();
    let coarse_seconds = component_profile.coarse_solve.as_secs_f64();
    let prolongation_seconds = component_profile.prolongation.as_secs_f64();
    let precond_seconds = component_profile.total().as_secs_f64();

    println!();
    println!("Kernel profile ({} repeats, normal solve path remains uninstrumented)", args.kernel_repeats);
    println!("CSR SpMV / call     : {:.3} ms", ms_per(spmv_seconds, args.kernel_repeats));
    println!("3x3 block Jacobi    : {:.3} ms/apply", ms_per(base_seconds, args.kernel_repeats));
    println!("restriction Z^T r   : {:.3} ms/apply", ms_per(restriction_seconds, args.kernel_repeats));
    println!("coarse solve        : {:.3} ms/apply", ms_per(coarse_seconds, args.kernel_repeats));
    println!("prolongation Z e    : {:.3} ms/apply", ms_per(prolongation_seconds, args.kernel_repeats));
    println!("precond components  : {:.3} ms/apply", ms_per(precond_seconds, args.kernel_repeats));

    let options = SolverOptions {
        relative_tolerance: args.relative_tolerance,
        absolute_tolerance: 0.0,
        max_iterations: args.max_iterations,
    };
    let mut x = vec![0.0; matrix.ncols()];
    let mut workspace = PcgWorkspace::new(matrix.nrows());
    let solve_start = Instant::now();
    let outcome = pcg_with_workspace(&matrix, &preconditioner, &b, &mut x, options, &mut workspace)?;
    let solve_seconds = solve_start.elapsed().as_secs_f64();
    let verified = verified_relative_residual(&matrix, &b, &x)?;

    let iterations = outcome.iterations.max(1);
    let observed_ms_per_iter = solve_seconds * 1.0e3 / iterations as f64;
    let spmv_ms = ms_per(spmv_seconds, args.kernel_repeats);
    let precond_ms = ms_per(precond_seconds, args.kernel_repeats);
    let estimated_kernel_total_ms = (outcome.iterations + 1) as f64 * spmv_ms
        + outcome.iterations as f64 * precond_ms;
    let other_ms = (solve_seconds * 1.0e3 - estimated_kernel_total_ms).max(0.0);

    println!();
    println!("PCG solve");
    println!("status             : {:?}", outcome.status);
    println!("iterations         : {}", outcome.iterations);
    println!("reported residual  : {:.6e}", outcome.final_residual / norm2(&b).max(f64::MIN_POSITIVE));
    println!("verified residual  : {:.6e}", verified);
    println!("solve time         : {:.3} ms", solve_seconds * 1.0e3);
    println!("observed / iter    : {:.3} ms", observed_ms_per_iter);
    println!("estimated SpMV     : {:.3} ms total", (outcome.iterations + 1) as f64 * spmv_ms);
    println!("estimated precond  : {:.3} ms total", outcome.iterations as f64 * precond_ms);
    println!("estimated other    : {:.3} ms total", other_ms);
    println!("note               : kernel totals are microbenchmark estimates, not nested timers inside PCG");

    if !verified.is_finite() { return Err("non-finite independently verified residual".into()); }
    Ok(())
}
