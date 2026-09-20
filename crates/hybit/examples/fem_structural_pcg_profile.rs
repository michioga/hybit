use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

use hybit::{
    analyze_csr32, pcg_with_workspace, read_matrix_market, recommend_rigid_body_aggregate_nodes,
    Csr32Matrix, HybitError, KrylovOutcome, LinearOperator, ParallelCsr32Operator,
    ParallelRigidBodyTwoLevelPreconditioner, PcgWorkspace, Preconditioner, RigidBodyAggregation,
    RigidBodyTwoLevelBlockJacobiPreconditioner, SolveStatus, SolverOptions,
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
        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => matrix = Some(PathBuf::from(next_value(&mut it, "--matrix")?)),
                "--coords" => coordinates = Some(PathBuf::from(next_value(&mut it, "--coords")?)),
                "--rhs" => rhs = Some(PathBuf::from(next_value(&mut it, "--rhs")?)),
                "--tol" => relative_tolerance = next_value(&mut it, "--tol")?.parse()?,
                "--max-iters" => max_iterations = next_value(&mut it, "--max-iters")?.parse()?,
                "--target-coarse-dim" => {
                    target_coarse_dimension = next_value(&mut it, "--target-coarse-dim")?.parse()?
                }
                "--aggregation" => {
                    aggregation = match next_value(&mut it, "--aggregation")?
                        .to_ascii_lowercase()
                        .as_str()
                    {
                        "contiguous" => RigidBodyAggregation::Contiguous,
                        "graph" => RigidBodyAggregation::Graph,
                        other => {
                            return Err(format!(
                                "unknown aggregation '{other}'; use contiguous or graph"
                            )
                            .into())
                        }
                    }
                }
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
        let coordinates = coordinates.unwrap_or_else(|| matrix.with_extension("coords"));
        if !relative_tolerance.is_finite() || relative_tolerance <= 0.0 {
            return Err("--tol must be finite and > 0".into());
        }
        if max_iterations == 0 {
            return Err("--max-iters must be > 0".into());
        }
        if target_coarse_dimension < 6 {
            return Err("--target-coarse-dim must be >= 6".into());
        }
        Ok(Self {
            matrix,
            coordinates,
            rhs,
            relative_tolerance,
            max_iterations,
            target_coarse_dimension,
            aggregation,
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
    println!("HyBIT PCG vector-kernel profile");
    println!("Usage: fem_structural_pcg_profile --matrix K.mtx [--coords K.coords] [--rhs b.txt] [--tol 1e-8] [--max-iters 3000] [--target-coarse-dim 1536] [--aggregation graph|contiguous]");
    println!("The profiled path uses Parallel CSR + Parallel rigid-body preconditioner and times actual PCG stages.");
    println!("Set RAYON_NUM_THREADS before launch to control the shared Rayon pool.");
}

fn read_coordinates(path: &Path) -> Result<Vec<[f64; 3]>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut expected = None::<usize>;
    let mut coordinates = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        let text = line.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        if expected.is_none() {
            expected = Some(text.parse::<usize>().map_err(|e| {
                format!(
                    "{}:{}: invalid coordinate count: {e}",
                    path.display(),
                    line_no + 1
                )
            })?);
            coordinates.reserve(expected.unwrap());
            continue;
        }
        let fields: Vec<&str> = text.split_whitespace().collect();
        if fields.len() != 3 {
            return Err(format!(
                "{}:{}: expected three coordinates",
                path.display(),
                line_no + 1
            )
            .into());
        }
        let xyz = [
            fields[0].parse::<f64>()?,
            fields[1].parse::<f64>()?,
            fields[2].parse::<f64>()?,
        ];
        if xyz.iter().any(|v| !v.is_finite()) {
            return Err(
                format!("{}:{}: non-finite coordinate", path.display(), line_no + 1).into(),
            );
        }
        coordinates.push(xyz);
    }
    let expected =
        expected.ok_or_else(|| format!("{}: missing coordinate count", path.display()))?;
    if coordinates.len() != expected {
        return Err(format!(
            "{}: coordinate count mismatch: header says {}, read {}",
            path.display(),
            expected,
            coordinates.len()
        )
        .into());
    }
    Ok(coordinates)
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

fn dot_serial(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm2(x: &[f64]) -> f64 {
    dot_serial(x, x).sqrt()
}
fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn verified_relative_residual(
    a: &Csr32Matrix,
    b: &[f64],
    x: &[f64],
) -> Result<f64, Box<dyn Error>> {
    let ax = a.spmv(x)?;
    let rr = b
        .iter()
        .zip(&ax)
        .map(|(&bi, &ai)| {
            let r = bi - ai;
            r * r
        })
        .sum::<f64>()
        .sqrt();
    let bn = norm2(b);
    Ok(if bn == 0.0 { rr } else { rr / bn })
}

#[derive(Clone, Copy, Debug, Default)]
struct PcgStageProfile {
    operator_seconds: f64,
    preconditioner_seconds: f64,
    dot_seconds: f64,
    norm_seconds: f64,
    residual_init_seconds: f64,
    update_x_r_seconds: f64,
    update_p_seconds: f64,
    copy_p_seconds: f64,
    operator_calls: usize,
    preconditioner_calls: usize,
    dot_calls: usize,
    norm_calls: usize,
    update_x_r_calls: usize,
    update_p_calls: usize,
}

impl PcgStageProfile {
    fn accounted_seconds(&self) -> f64 {
        self.operator_seconds
            + self.preconditioner_seconds
            + self.dot_seconds
            + self.norm_seconds
            + self.residual_init_seconds
            + self.update_x_r_seconds
            + self.update_p_seconds
            + self.copy_p_seconds
    }
}

fn timed_dot(a: &[f64], b: &[f64], profile: &mut PcgStageProfile) -> f64 {
    let start = Instant::now();
    let value = dot_serial(a, b);
    profile.dot_seconds += start.elapsed().as_secs_f64();
    profile.dot_calls += 1;
    value
}

fn timed_norm(x: &[f64], profile: &mut PcgStageProfile) -> f64 {
    let start = Instant::now();
    let value = norm2(x);
    profile.norm_seconds += start.elapsed().as_secs_f64();
    profile.norm_calls += 1;
    value
}

fn profiled_pcg(
    a: &dyn LinearOperator,
    m: &dyn Preconditioner,
    b: &[f64],
    x: &mut [f64],
    options: SolverOptions,
) -> Result<(KrylovOutcome, PcgStageProfile, f64), HybitError> {
    options.validate()?;
    if a.rows() != a.cols() {
        return Err(HybitError::InvalidMatrix("PCG requires a square operator"));
    }
    let n = a.rows();
    if b.len() != n {
        return Err(HybitError::DimensionMismatch {
            expected: n,
            actual: b.len(),
        });
    }
    if x.len() != n {
        return Err(HybitError::DimensionMismatch {
            expected: n,
            actual: x.len(),
        });
    }
    if m.len() != n {
        return Err(HybitError::DimensionMismatch {
            expected: n,
            actual: m.len(),
        });
    }

    let mut profile = PcgStageProfile::default();
    let mut ax = vec![0.0; n];
    let mut r = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut ap = vec![0.0; n];
    // Match the production prepared path: workspace allocation is setup cost,
    // not part of the Krylov solve timer.
    let wall_start = Instant::now();

    let start = Instant::now();
    a.apply(x, &mut ax)?;
    profile.operator_seconds += start.elapsed().as_secs_f64();
    profile.operator_calls += 1;

    let start = Instant::now();
    for i in 0..n {
        r[i] = b[i] - ax[i];
    }
    profile.residual_init_seconds += start.elapsed().as_secs_f64();

    let initial_residual = timed_norm(&r, &mut profile);
    let b_norm = timed_norm(b, &mut profile);
    let target = options
        .absolute_tolerance
        .max(options.relative_tolerance * b_norm.max(f64::MIN_POSITIVE));
    if initial_residual <= target {
        let wall = wall_start.elapsed().as_secs_f64();
        return Ok((
            KrylovOutcome {
                status: SolveStatus::Converged,
                iterations: 0,
                initial_residual,
                final_residual: initial_residual,
            },
            profile,
            wall,
        ));
    }

    let start = Instant::now();
    m.apply(&r, &mut z)?;
    profile.preconditioner_seconds += start.elapsed().as_secs_f64();
    profile.preconditioner_calls += 1;

    let start = Instant::now();
    p.copy_from_slice(&z);
    profile.copy_p_seconds += start.elapsed().as_secs_f64();

    let mut rz_old = timed_dot(&r, &z, &mut profile);
    if !rz_old.is_finite() || rz_old <= 0.0 {
        return Err(HybitError::NumericalBreakdown(
            "non-positive r^T M^-1 r; PCG assumptions may be violated",
        ));
    }

    let mut final_residual = initial_residual;
    for iter in 1..=options.max_iterations {
        let start = Instant::now();
        a.apply(&p, &mut ap)?;
        profile.operator_seconds += start.elapsed().as_secs_f64();
        profile.operator_calls += 1;

        let denom = timed_dot(&p, &ap, &mut profile);
        if !denom.is_finite() || denom <= 0.0 {
            let wall = wall_start.elapsed().as_secs_f64();
            return Ok((
                KrylovOutcome {
                    status: SolveStatus::Breakdown,
                    iterations: iter - 1,
                    initial_residual,
                    final_residual,
                },
                profile,
                wall,
            ));
        }

        let alpha = rz_old / denom;
        let start = Instant::now();
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        profile.update_x_r_seconds += start.elapsed().as_secs_f64();
        profile.update_x_r_calls += 1;

        final_residual = timed_norm(&r, &mut profile);
        if final_residual <= target {
            let wall = wall_start.elapsed().as_secs_f64();
            return Ok((
                KrylovOutcome {
                    status: SolveStatus::Converged,
                    iterations: iter,
                    initial_residual,
                    final_residual,
                },
                profile,
                wall,
            ));
        }

        let start = Instant::now();
        m.apply(&r, &mut z)?;
        profile.preconditioner_seconds += start.elapsed().as_secs_f64();
        profile.preconditioner_calls += 1;

        let rz_new = timed_dot(&r, &z, &mut profile);
        if !rz_new.is_finite() || rz_new <= 0.0 {
            let wall = wall_start.elapsed().as_secs_f64();
            return Ok((
                KrylovOutcome {
                    status: SolveStatus::Breakdown,
                    iterations: iter,
                    initial_residual,
                    final_residual,
                },
                profile,
                wall,
            ));
        }

        let beta = rz_new / rz_old;
        let start = Instant::now();
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        profile.update_p_seconds += start.elapsed().as_secs_f64();
        profile.update_p_calls += 1;
        rz_old = rz_new;
    }

    let wall = wall_start.elapsed().as_secs_f64();
    Ok((
        KrylovOutcome {
            status: SolveStatus::MaxIterations,
            iterations: options.max_iterations,
            initial_residual,
            final_residual,
        },
        profile,
        wall,
    ))
}

fn print_stage(label: &str, seconds: f64, calls: usize, wall: f64) {
    let total_ms = seconds * 1.0e3;
    let per_call_ms = if calls == 0 {
        0.0
    } else {
        total_ms / calls as f64
    };
    let share = if wall == 0.0 {
        0.0
    } else {
        100.0 * seconds / wall
    };
    println!(
        "{label:<24}: {:9.3} ms total  {:7.3} ms/call  {:5.1}%  ({} calls)",
        total_ms, per_call_ms, share, calls
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    println!(
        "HyBIT {} PCG vector-kernel profile",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix             : {}", args.matrix.display());
    println!("coordinates        : {}", args.coordinates.display());

    let load_start = Instant::now();
    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    let matrix_load = load_start.elapsed().as_secs_f64();
    let coord_start = Instant::now();
    let coordinates = read_coordinates(&args.coordinates)?;
    let coord_load = coord_start.elapsed().as_secs_f64();
    let matrix_profile = analyze_csr32(&matrix)?;

    println!(
        "Matrix Market      : {:?}, {} input entries -> {} CSR nnz",
        mm.symmetry, mm.input_entries, mm.csr_nnz
    );
    println!(
        "dimensions         : {} x {}",
        matrix_profile.nrows, matrix_profile.ncols
    );
    println!("nnz                : {}", matrix_profile.nnz);
    println!(
        "CSR storage        : {:.3} MiB",
        mib(matrix.storage_bytes())
    );
    println!("matrix load        : {:.3} ms", matrix_load * 1.0e3);
    println!("coordinate nodes   : {}", coordinates.len());
    println!("coordinate load    : {:.3} ms", coord_load * 1.0e3);
    println!("aggregation        : {:?}", args.aggregation);
    println!("target coarse dim  : {}", args.target_coarse_dimension);

    if !matrix_profile.square || !matrix_profile.full_diagonal || !matrix_profile.positive_diagonal
    {
        return Err(
            "structural PCG benchmark requires a square matrix with a complete positive diagonal"
                .into(),
        );
    }
    if matrix.nrows() != coordinates.len() * 3 {
        return Err(format!(
            "matrix/coordinate mismatch: {} matrix rows != {} coordinate nodes * 3",
            matrix.nrows(),
            coordinates.len()
        )
        .into());
    }

    let b = if let Some(path) = args.rhs.as_deref() {
        println!("RHS                : {}", path.display());
        load_rhs(path, matrix.nrows())?
    } else {
        println!("RHS                : generated as b=A*1 (known exact solution)");
        matrix.spmv(&vec![1.0; matrix.ncols()])?
    };

    let aggregate_nodes =
        recommend_rigid_body_aggregate_nodes(coordinates.len(), args.target_coarse_dimension)?;
    let setup_start = Instant::now();
    let preconditioner = match args.aggregation {
        RigidBodyAggregation::Graph => {
            RigidBodyTwoLevelBlockJacobiPreconditioner::from_csr32_graph(
                &matrix,
                &coordinates,
                aggregate_nodes,
            )?
        }
        RigidBodyAggregation::Contiguous => RigidBodyTwoLevelBlockJacobiPreconditioner::from_csr32(
            &matrix,
            &coordinates,
            aggregate_nodes,
        )?,
        RigidBodyAggregation::Auto => unreachable!(),
    };
    let parallel_preconditioner = ParallelRigidBodyTwoLevelPreconditioner::new(&preconditioner)?;
    let parallel_operator = ParallelCsr32Operator::new(&matrix);
    let setup_seconds = setup_start.elapsed().as_secs_f64();

    println!("aggregate target   : {} nodes", aggregate_nodes);
    println!("aggregate count    : {}", preconditioner.aggregate_count());
    println!("coarse dimension   : {}", preconditioner.coarse_dimension());
    println!(
        "prec storage       : {:.3} MiB",
        mib(preconditioner.factor_bytes())
    );
    println!(
        "parallel index     : {:.3} MiB",
        mib(parallel_preconditioner.index_storage_bytes())
    );
    println!("setup              : {:.3} ms", setup_seconds * 1.0e3);
    println!(
        "Rayon threads      : {}",
        parallel_preconditioner.rayon_threads()
    );

    let options = SolverOptions {
        relative_tolerance: args.relative_tolerance,
        absolute_tolerance: 0.0,
        max_iterations: args.max_iterations,
    };

    let mut x_profiled = vec![0.0; matrix.ncols()];
    let (profiled_out, stages, profiled_wall) = profiled_pcg(
        &parallel_operator,
        &parallel_preconditioner,
        &b,
        &mut x_profiled,
        options,
    )?;
    let profiled_verified = verified_relative_residual(&matrix, &b, &x_profiled)?;

    let mut x_control = vec![0.0; matrix.ncols()];
    let mut workspace = PcgWorkspace::new(matrix.nrows());
    let control_start = Instant::now();
    let control_out = pcg_with_workspace(
        &parallel_operator,
        &parallel_preconditioner,
        &b,
        &mut x_control,
        options,
        &mut workspace,
    )?;
    let control_wall = control_start.elapsed().as_secs_f64();
    let control_verified = verified_relative_residual(&matrix, &b, &x_control)?;

    println!();
    println!("Profiled PCG");
    println!("status             : {:?}", profiled_out.status);
    println!("iterations         : {}", profiled_out.iterations);
    println!(
        "reported residual  : {:.6e}",
        profiled_out.final_residual / norm2(&b).max(f64::MIN_POSITIVE)
    );
    println!("verified residual  : {:.6e}", profiled_verified);
    println!("profiled wall      : {:.3} ms", profiled_wall * 1.0e3);
    println!(
        "observed / iter    : {:.3} ms",
        if profiled_out.iterations == 0 {
            0.0
        } else {
            profiled_wall * 1.0e3 / profiled_out.iterations as f64
        }
    );
    println!();
    println!("Actual nested PCG stage timings");
    print_stage(
        "operator A*x",
        stages.operator_seconds,
        stages.operator_calls,
        profiled_wall,
    );
    print_stage(
        "preconditioner",
        stages.preconditioner_seconds,
        stages.preconditioner_calls,
        profiled_wall,
    );
    print_stage(
        "dot reductions",
        stages.dot_seconds,
        stages.dot_calls,
        profiled_wall,
    );
    print_stage(
        "norm reductions",
        stages.norm_seconds,
        stages.norm_calls,
        profiled_wall,
    );
    print_stage(
        "x/r fused update",
        stages.update_x_r_seconds,
        stages.update_x_r_calls,
        profiled_wall,
    );
    print_stage(
        "p update",
        stages.update_p_seconds,
        stages.update_p_calls,
        profiled_wall,
    );
    print_stage(
        "initial residual",
        stages.residual_init_seconds,
        1,
        profiled_wall,
    );
    print_stage("initial p copy", stages.copy_p_seconds, 1, profiled_wall);
    let accounted = stages.accounted_seconds();
    let unaccounted = (profiled_wall - accounted).max(0.0);
    println!(
        "accounted total        : {:9.3} ms  {:5.1}%",
        accounted * 1.0e3,
        100.0 * accounted / profiled_wall.max(f64::MIN_POSITIVE)
    );
    println!(
        "timer/control overhead : {:9.3} ms  {:5.1}%",
        unaccounted * 1.0e3,
        100.0 * unaccounted / profiled_wall.max(f64::MIN_POSITIVE)
    );

    println!();
    println!("Uninstrumented control");
    println!("status             : {:?}", control_out.status);
    println!("iterations         : {}", control_out.iterations);
    println!(
        "reported residual  : {:.6e}",
        control_out.final_residual / norm2(&b).max(f64::MIN_POSITIVE)
    );
    println!("verified residual  : {:.6e}", control_verified);
    println!("solve time         : {:.3} ms", control_wall * 1.0e3);
    println!(
        "profile/control    : {:.3}x",
        profiled_wall / control_wall.max(f64::MIN_POSITIVE)
    );
    println!("note               : profile uses the same serial PCG vector kernels as production; only nested timers are added");

    Ok(())
}
