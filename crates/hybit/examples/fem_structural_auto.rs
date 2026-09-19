use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

use hybit::{
    analyze_csr32, read_matrix_market, Csr32Matrix, HybitSolver, ParallelCsr32Operator, RigidBodyAggregation, SolverOptions, StructuralOptions, StructuralPreconditionerPolicy, StructuralSpmvPolicy,
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
    spmv_policy: StructuralSpmvPolicy,
    preconditioner_policy: StructuralPreconditionerPolicy,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut coordinates = None;
        let mut rhs = None;
        let mut relative_tolerance: f64 = 1.0e-8;
        let mut max_iterations = 3000usize;
        let mut target_coarse_dimension = 1536usize;
        let mut aggregation = RigidBodyAggregation::Auto;
        let mut spmv_policy = StructuralSpmvPolicy::Auto;
        let mut preconditioner_policy = StructuralPreconditionerPolicy::Auto;
        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => matrix = Some(PathBuf::from(next_value(&mut it, "--matrix")?)),
                "--coords" => coordinates = Some(PathBuf::from(next_value(&mut it, "--coords")?)),
                "--rhs" => rhs = Some(PathBuf::from(next_value(&mut it, "--rhs")?)),
                "--tol" => relative_tolerance = next_value(&mut it, "--tol")?.parse()?,
                "--max-iters" => max_iterations = next_value(&mut it, "--max-iters")?.parse()?,
                "--target-coarse-dim" => target_coarse_dimension = next_value(&mut it, "--target-coarse-dim")?.parse()?,
                "--aggregation" => {
                    aggregation = match next_value(&mut it, "--aggregation")?.to_ascii_lowercase().as_str() {
                        "auto" => RigidBodyAggregation::Auto,
                        "contiguous" => RigidBodyAggregation::Contiguous,
                        "graph" => RigidBodyAggregation::Graph,
                        other => return Err(format!("unknown aggregation '{other}'; use auto, contiguous, or graph").into()),
                    }
                }
                "--spmv" => {
                    spmv_policy = match next_value(&mut it, "--spmv")?.to_ascii_lowercase().as_str() {
                        "auto" => StructuralSpmvPolicy::Auto,
                        "serial" => StructuralSpmvPolicy::Serial,
                        "parallel" => StructuralSpmvPolicy::Parallel,
                        other => return Err(format!("unknown SpMV policy '{other}'; use auto, serial, or parallel").into()),
                    }
                }
                "--precond" => {
                    preconditioner_policy = match next_value(&mut it, "--precond")?.to_ascii_lowercase().as_str() {
                        "auto" => StructuralPreconditionerPolicy::Auto,
                        "serial" => StructuralPreconditionerPolicy::Serial,
                        "parallel" => StructuralPreconditionerPolicy::Parallel,
                        other => return Err(format!("unknown preconditioner policy '{other}'; use auto, serial, or parallel").into()),
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
        Ok(Self { matrix, coordinates, rhs, relative_tolerance, max_iterations, target_coarse_dimension, aggregation, spmv_policy, preconditioner_policy })
    }
}

fn next_value<I: Iterator<Item = String>>(it: &mut I, flag: &str) -> Result<String, Box<dyn Error>> {
    it.next().ok_or_else(|| format!("missing value after {flag}").into())
}

fn print_usage() {
    println!("HyBIT structural-auto FEM benchmark");
    println!("Usage: fem_structural_auto --matrix K.mtx [--coords K.coords] [--rhs b.txt] [--tol 1e-8] [--max-iters 3000] [--target-coarse-dim 1536] [--aggregation auto|contiguous|graph] [--spmv auto|serial|parallel] [--precond auto|serial|parallel]");
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
        let x: f64 = fields[0].parse()?;
        let y: f64 = fields[1].parse()?;
        let z: f64 = fields[2].parse()?;
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return Err(format!("{}:{}: non-finite coordinate", path.display(), line_no + 1).into());
        }
        coordinates.push([x, y, z]);
    }

    let expected = expected.ok_or_else(|| format!("{}: missing coordinate count", path.display()))?;
    if coordinates.len() != expected {
        return Err(format!(
            "{}: coordinate count mismatch: header says {}, read {}",
            path.display(), expected, coordinates.len()
        ).into());
    }
    Ok(coordinates)
}

fn load_rhs(path: &Path, n: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut values = Vec::with_capacity(n);

    for (line_index, raw_line) in text.lines().enumerate() {
        // Be tolerant of an UTF-8 BOM and optional comment lines, while never
        // silently ignoring malformed numeric data.
        let line = raw_line.trim_start_matches('\u{feff}').trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('%') {
            continue;
        }
        for (token_index, token) in line.split_whitespace().enumerate() {
            let value = token.parse::<f64>().map_err(|e| {
                format!(
                    "{}: invalid RHS float at line {}, token {}: {:?} ({e})",
                    path.display(),
                    line_index + 1,
                    token_index + 1,
                    token
                )
            })?;
            if !value.is_finite() {
                return Err(format!(
                    "{}: non-finite RHS value at line {}, token {}: {:?}",
                    path.display(),
                    line_index + 1,
                    token_index + 1,
                    token
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
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn verified_relative_residual(a: &Csr32Matrix, b: &[f64], x: &[f64]) -> Result<f64, Box<dyn Error>> {
    let ax = a.spmv(x)?;
    let sum = b.iter().zip(ax.iter()).map(|(&bi, &ai)| {
        let r = bi - ai;
        r * r
    }).sum::<f64>();
    let denom = norm2(b);
    Ok(if denom == 0.0 { sum.sqrt() } else { sum.sqrt() / denom })
}

fn relative_error_to_ones(x: &[f64]) -> f64 {
    let diff = x.iter().map(|&xi| {
        let d = xi - 1.0;
        d * d
    }).sum::<f64>();
    diff.sqrt() / (x.len() as f64).sqrt().max(f64::MIN_POSITIVE)
}

fn mib(bytes: usize) -> f64 { bytes as f64 / (1024.0 * 1024.0) }

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    println!("HyBIT {} structural-auto FEM benchmark", env!("CARGO_PKG_VERSION"));
    println!("matrix             : {}", args.matrix.display());
    println!("coordinates        : {}", args.coordinates.display());

    let load_start = Instant::now();
    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    let matrix_load_seconds = load_start.elapsed().as_secs_f64();
    let coord_start = Instant::now();
    let coordinates = read_coordinates(&args.coordinates)?;
    let coord_load_seconds = coord_start.elapsed().as_secs_f64();
    let profile = analyze_csr32(&matrix)?;

    println!("Matrix Market      : {:?}, {} input entries -> {} CSR nnz", mm.symmetry, mm.input_entries, mm.csr_nnz);
    println!("dimensions         : {} x {}", profile.nrows, profile.ncols);
    println!("nnz                : {}", profile.nnz);
    println!("CSR storage        : {:.3} MiB", mib(matrix.storage_bytes()));
    println!("matrix load        : {:.3} ms", matrix_load_seconds * 1.0e3);
    println!("coordinate nodes   : {}", coordinates.len());
    println!("coordinate load    : {:.3} ms", coord_load_seconds * 1.0e3);
    

    if !profile.square || !profile.full_diagonal || !profile.positive_diagonal {
        return Err("rigid-body two-level PCG requires a square matrix with a complete positive diagonal".into());
    }
    if matrix.nrows() != coordinates.len() * 3 {
        return Err(format!(
            "matrix/coordinate mismatch: {} matrix rows != {} coordinate nodes * 3",
            matrix.nrows(), coordinates.len()
        ).into());
    }

    let generated_rhs = args.rhs.is_none();
    let b = if let Some(path) = args.rhs.as_deref() {
        println!("RHS                : {}", path.display());
        load_rhs(path, matrix.nrows())?
    } else {
        println!("RHS                : generated as b=A*1 (known exact solution)");
        let ones = vec![1.0; matrix.ncols()];
        matrix.spmv(&ones)?
    };
    let mut solver = HybitSolver::new();
    solver.set_options(SolverOptions {
        relative_tolerance: args.relative_tolerance,
        absolute_tolerance: 0.0,
        max_iterations: args.max_iterations,
    })?;
    solver.set_structural_options(StructuralOptions {
        target_coarse_dimension: args.target_coarse_dimension,
        aggregation: args.aggregation,
        spmv_policy: args.spmv_policy,
        preconditioner_policy: args.preconditioner_policy,
    })?;

    let analysis = solver.analyze_csr32(&matrix)?;
    let mut prepared = solver.prepare_structural_csr32(&matrix, &analysis, &coordinates)?;
    println!("policy             : StructuralAuto/RigidBodyTwoLevel");
    println!("aggregation        : {:?}", prepared.aggregation());
    println!("SpMV policy        : {:?}", prepared.spmv_policy());
    println!("precond policy     : {:?}", prepared.structural_preconditioner_policy());
    if prepared.parallel_spmv_enabled() || prepared.parallel_preconditioner_enabled() {
        println!("Rayon threads      : {}", ParallelCsr32Operator::new(&matrix).rayon_threads());
    }
    if prepared.parallel_preconditioner_enabled() {
        println!("parallel index     : {:.3} MiB", mib(prepared.parallel_preconditioner_index_bytes()));
    }
    println!("target coarse dim  : {}", args.target_coarse_dimension);
    println!("aggregate nodes    : {} (auto-selected)", prepared.aggregate_nodes());
    println!("fine block size    : 3");
    println!("aggregate count    : {}", prepared.aggregate_count());
    println!("aggregate min/max  : {} / {} nodes", prepared.min_aggregate_nodes(), prepared.max_aggregate_nodes());
    println!("modes/aggregate    : 6");
    println!("coarse dimension   : {}", prepared.coarse_dimension());
    println!("base factor        : {:.3} MiB", mib(prepared.base_factor_bytes()));
    println!("coarse factor      : {:.3} MiB", mib(prepared.coarse_factor_bytes()));
    println!("geometry storage   : {:.3} MiB", mib(prepared.geometry_bytes()));
    println!("total prec storage : {:.3} MiB", mib(prepared.preconditioner_bytes()));
    println!("analysis           : {:.3} ms", prepared.analysis_seconds() * 1.0e3);
    println!("prepare            : {:.3} ms", prepared.prepare_seconds() * 1.0e3);

    let mut x = vec![0.0; matrix.ncols()];
    let report = prepared.solve(&matrix, &b, &mut x)?;
    let verified = verified_relative_residual(&matrix, &b, &x)?;

    println!();
    println!("Structural Auto API: 3x3 Block-Jacobi + six rigid-body coarse modes");
    println!("status             : {:?}", report.status);
    println!("preconditioner     : {:?}", report.preconditioner);
    println!("iterations         : {}", report.iterations);
    println!("reported residual  : {:.6e}", report.relative_residual);
    println!("verified residual  : {:.6e}", verified);
    if generated_rhs {
        println!("relative x error   : {:.6e}", relative_error_to_ones(&x));
    }
    println!("solve time         : {:.3} ms", report.solve_seconds * 1.0e3);
    println!("total setup+solve  : {:.3} ms", (report.setup_seconds + report.solve_seconds) * 1.0e3);

    if !verified.is_finite() {
        return Err("non-finite independently verified residual".into());
    }
    Ok(())
}
