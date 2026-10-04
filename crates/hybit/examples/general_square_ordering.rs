use std::collections::VecDeque;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hybit::{
    read_matrix_market, Csr32Matrix, GeneralSquareOptions, GeneralSquarePreconditionerPolicy,
    HybitSolver, MatrixProblemClass, SolveReport, SolveStatus, SolverOptions,
};

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    rhs: Option<PathBuf>,
    relative_tolerance: f64,
    max_iterations: usize,
    restart: usize,
    preflight_only: bool,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut rhs = None;
        let mut relative_tolerance: f64 = 1.0e-8;
        let mut max_iterations = 1000usize;
        let mut restart = 30usize;
        let mut preflight_only = false;

        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => matrix = Some(PathBuf::from(next_value(&mut it, "--matrix")?)),
                "--rhs" => rhs = Some(PathBuf::from(next_value(&mut it, "--rhs")?)),
                "--tol" => {
                    relative_tolerance = next_value(&mut it, "--tol")?.parse()?;
                }
                "--max-iters" => {
                    max_iterations = next_value(&mut it, "--max-iters")?.parse()?;
                }
                "--restart" => {
                    restart = next_value(&mut it, "--restart")?.parse()?;
                }
                "--preflight-only" => {
                    preflight_only = true;
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
        if !relative_tolerance.is_finite() || relative_tolerance <= 0.0 {
            return Err("--tol must be finite and > 0".into());
        }
        if max_iterations == 0 {
            return Err("--max-iters must be > 0".into());
        }
        if restart == 0 {
            return Err("--restart must be > 0".into());
        }

        Ok(Self {
            matrix,
            rhs,
            relative_tolerance,
            max_iterations,
            restart,
            preflight_only,
        })
    }
}

#[derive(Debug)]
struct RawCase {
    report: SolveReport,
    x: Vec<f64>,
    analysis_seconds: f64,
    prepare_seconds: f64,
    solve_wall_seconds: f64,
    preconditioner_bytes: usize,
    krylov_workspace_bytes: usize,
    adjusted_pivots: usize,
}

#[derive(Debug)]
struct CaseMetrics {
    report: SolveReport,
    x: Vec<f64>,
    analysis_seconds: f64,
    prepare_seconds: f64,
    solve_wall_seconds: f64,
    preconditioner_bytes: usize,
    krylov_workspace_bytes: usize,
    adjusted_pivots: usize,
    verified_relative_residual: f64,
    relative_x_error: Option<f64>,
}

struct BenchmarkContext<'a> {
    args: &'a Args,
    original_matrix: &'a Csr32Matrix,
    original_b: &'a [f64],
    generated_rhs: bool,
}

fn next_value<I: Iterator<Item = String>>(
    it: &mut I,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    it.next()
        .ok_or_else(|| format!("missing value after {flag}").into())
}

fn print_usage() {
    println!("HyBIT GeneralSquare ILU(0) ordering benchmark");
    println!();
    println!("Usage:");
    println!(
        "  cargo run --release -p hybit --example general_square_ordering -- --matrix A.mtx [options]"
    );
    println!();
    println!("Options:");
    println!("  --rhs FILE        whitespace-separated RHS vector; default: b=A*1");
    println!("  --tol VALUE       relative tolerance (default 1e-8)");
    println!("  --max-iters N     maximum FGMRES iterations (default 1000)");
    println!("  --restart N       fixed FGMRES restart dimension (default 30)");
    println!("  --preflight-only  validate corpus eligibility without solving");
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

fn verified_relative_residual(
    a: &Csr32Matrix,
    b: &[f64],
    x: &[f64],
) -> Result<f64, Box<dyn Error>> {
    let ax = a.spmv(x)?;
    let residual_norm = b
        .iter()
        .zip(ax.iter())
        .map(|(&bi, &axi)| {
            let r = bi - axi;
            r * r
        })
        .sum::<f64>()
        .sqrt();
    let rhs_norm = norm2(b);
    Ok(if rhs_norm == 0.0 {
        residual_norm
    } else {
        residual_norm / rhs_norm
    })
}

fn relative_error_to_ones(x: &[f64]) -> f64 {
    x.iter()
        .map(|&xi| {
            let d = xi - 1.0;
            d * d
        })
        .sum::<f64>()
        .sqrt()
        / (x.len() as f64).sqrt().max(f64::MIN_POSITIVE)
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

#[derive(Clone, Copy, Debug)]
struct DiagonalPreflight {
    missing: usize,
    zero: usize,
    first_missing: Option<usize>,
    first_zero: Option<usize>,
}

fn diagonal_preflight(a: &Csr32Matrix) -> DiagonalPreflight {
    let mut stats = DiagonalPreflight {
        missing: 0,
        zero: 0,
        first_missing: None,
        first_zero: None,
    };

    for row in 0..a.nrows() {
        let start = a.row_ptr()[row] as usize;
        let end = a.row_ptr()[row + 1] as usize;
        let mut found = false;
        let mut diagonal = 0.0;

        for p in start..end {
            if a.col_idx()[p] as usize == row {
                found = true;
                diagonal += a.values()[p];
            }
        }

        if !found {
            stats.missing += 1;
            stats.first_missing.get_or_insert(row);
        } else if diagonal == 0.0 {
            stats.zero += 1;
            stats.first_zero.get_or_insert(row);
        }
    }

    stats
}

fn structural_bandwidth(a: &Csr32Matrix) -> usize {
    let mut bandwidth = 0usize;
    for row in 0..a.nrows() {
        let start = a.row_ptr()[row] as usize;
        let end = a.row_ptr()[row + 1] as usize;
        for &col in &a.col_idx()[start..end] {
            bandwidth = bandwidth.max(row.abs_diff(col as usize));
        }
    }
    bandwidth
}

fn build_symmetrized_graph(a: &Csr32Matrix) -> Vec<Vec<u32>> {
    let n = a.nrows();
    let mut graph = vec![Vec::<u32>::new(); n];

    for row in 0..n {
        let start = a.row_ptr()[row] as usize;
        let end = a.row_ptr()[row + 1] as usize;
        for &col_u32 in &a.col_idx()[start..end] {
            let col = col_u32 as usize;
            if row == col {
                continue;
            }
            graph[row].push(col_u32);
            graph[col].push(row as u32);
        }
    }

    for neighbors in &mut graph {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    graph
}

fn reverse_cuthill_mckee(graph: &mut [Vec<u32>]) -> Vec<usize> {
    let n = graph.len();
    let degree: Vec<usize> = graph.iter().map(Vec::len).collect();

    for neighbors in graph.iter_mut() {
        neighbors.sort_unstable_by_key(|&node| (degree[node as usize], node));
    }

    let mut visited = vec![false; n];
    let mut permutation = Vec::with_capacity(n);
    let mut queue = VecDeque::new();

    while permutation.len() < n {
        let start = (0..n)
            .filter(|&node| !visited[node])
            .min_by_key(|&node| (degree[node], node))
            .expect("at least one unvisited node must remain");

        let component_begin = permutation.len();
        visited[start] = true;
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            permutation.push(node);
            for &neighbor_u32 in &graph[node] {
                let neighbor = neighbor_u32 as usize;
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        permutation[component_begin..].reverse();
    }

    permutation
}

fn symmetric_permute(a: &Csr32Matrix, new_to_old: &[usize]) -> Result<Csr32Matrix, Box<dyn Error>> {
    let n = a.nrows();
    if a.ncols() != n || new_to_old.len() != n {
        return Err(
            "symmetric permutation requires a square matrix and n-entry permutation".into(),
        );
    }

    let mut old_to_new = vec![usize::MAX; n];
    for (new, &old) in new_to_old.iter().enumerate() {
        if old >= n || old_to_new[old] != usize::MAX {
            return Err("invalid RCM permutation".into());
        }
        old_to_new[old] = new;
    }

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(a.nnz());
    let mut values = Vec::with_capacity(a.nnz());
    let mut row_entries = Vec::<(u32, f64)>::new();
    row_ptr.push(0u32);

    for &old_row in new_to_old {
        row_entries.clear();
        let start = a.row_ptr()[old_row] as usize;
        let end = a.row_ptr()[old_row + 1] as usize;

        for p in start..end {
            let old_col = a.col_idx()[p] as usize;
            let new_col = old_to_new[old_col];
            row_entries.push((u32::try_from(new_col)?, a.values()[p]));
        }
        row_entries.sort_unstable_by_key(|&(col, _)| col);

        for &(col, value) in &row_entries {
            col_idx.push(col);
            values.push(value);
        }
        row_ptr.push(u32::try_from(col_idx.len())?);
    }

    Ok(Csr32Matrix::new(n, n, row_ptr, col_idx, values)?)
}

fn permute_vector(old: &[f64], new_to_old: &[usize]) -> Vec<f64> {
    new_to_old.iter().map(|&old_index| old[old_index]).collect()
}

fn unpermute_vector(new: &[f64], new_to_old: &[usize]) -> Vec<f64> {
    let mut old = vec![0.0; new.len()];
    for (new_index, &old_index) in new_to_old.iter().enumerate() {
        old[old_index] = new[new_index];
    }
    old
}

fn preconditioner_name(policy: GeneralSquarePreconditionerPolicy) -> &'static str {
    match policy {
        GeneralSquarePreconditionerPolicy::Jacobi => "Jacobi",
        GeneralSquarePreconditionerPolicy::Ilu0 => "ILU0",
    }
}

fn run_case(
    matrix: &Csr32Matrix,
    b: &[f64],
    policy: GeneralSquarePreconditionerPolicy,
    args: &Args,
) -> Result<RawCase, Box<dyn Error>> {
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
    let analysis_seconds = analysis.analysis_seconds();
    let mut prepared = solver.prepare_csr32(matrix, &analysis)?;
    let prepare_seconds = prepared.prepare_seconds();
    let preconditioner_bytes = prepared.general_square_preconditioner_bytes();
    let adjusted_pivots = prepared.general_square_ilu_adjusted_pivots();
    let krylov_workspace_bytes = prepared.krylov_workspace_bytes();

    let mut x = vec![0.0; matrix.ncols()];
    let solve_start = Instant::now();
    let report = prepared.solve(matrix, b, &mut x)?;
    let solve_wall_seconds = solve_start.elapsed().as_secs_f64();

    Ok(RawCase {
        report,
        x,
        analysis_seconds,
        prepare_seconds,
        solve_wall_seconds,
        preconditioner_bytes,
        krylov_workspace_bytes,
        adjusted_pivots,
    })
}

fn benchmark_case(
    context: &BenchmarkContext<'_>,
    ordering: &str,
    matrix: &Csr32Matrix,
    b: &[f64],
    policy: GeneralSquarePreconditionerPolicy,
    new_to_old: Option<&[usize]>,
) -> Option<CaseMetrics> {
    println!();
    println!("== {ordering} / {} ==", preconditioner_name(policy));

    let raw = match run_case(matrix, b, policy, context.args) {
        Ok(metrics) => metrics,
        Err(error) => {
            println!("case error          : {error}");
            println!(
                "RESULT|ordering={ordering}|preconditioner={}|status=ERROR|error={}",
                preconditioner_name(policy),
                error.to_string().replace('|', "/")
            );
            return None;
        }
    };

    let x_original = if let Some(permutation) = new_to_old {
        unpermute_vector(&raw.x, permutation)
    } else {
        raw.x
    };

    let verified_relative_residual = match verified_relative_residual(
        context.original_matrix,
        context.original_b,
        &x_original,
    ) {
        Ok(value) => value,
        Err(error) => {
            println!("verification error  : {error}");
            return None;
        }
    };

    let relative_x_error = context
        .generated_rhs
        .then(|| relative_error_to_ones(&x_original));

    println!("status              : {:?}", raw.report.status);
    println!("iterations          : {}", raw.report.iterations);
    println!("reported residual   : {:.6e}", raw.report.relative_residual);
    println!("verified residual   : {:.6e}", verified_relative_residual);
    println!(
        "analysis            : {:.3} ms",
        raw.analysis_seconds * 1.0e3
    );
    println!(
        "prepare             : {:.3} ms",
        raw.prepare_seconds * 1.0e3
    );
    println!(
        "solve report        : {:.3} ms",
        raw.report.solve_seconds * 1.0e3
    );
    println!(
        "solve wall          : {:.3} ms",
        raw.solve_wall_seconds * 1.0e3
    );
    println!(
        "preconditioner      : {:.3} MiB",
        mib(raw.preconditioner_bytes)
    );
    println!(
        "Krylov workspace    : {:.3} MiB",
        mib(raw.krylov_workspace_bytes)
    );
    println!("adjusted pivots     : {}", raw.adjusted_pivots);
    if let Some(error) = relative_x_error {
        println!("relative x error    : {:.6e}", error);
    }

    println!(
        "RESULT|ordering={ordering}|preconditioner={}|status={:?}|iterations={}|reported_residual={:.6e}|verified_residual={:.6e}|analysis_ms={:.6}|prepare_ms={:.6}|solve_ms={:.6}|solve_wall_ms={:.6}|preconditioner_bytes={}|workspace_bytes={}|adjusted_pivots={}",
        preconditioner_name(policy),
        raw.report.status,
        raw.report.iterations,
        raw.report.relative_residual,
        verified_relative_residual,
        raw.analysis_seconds * 1.0e3,
        raw.prepare_seconds * 1.0e3,
        raw.report.solve_seconds * 1.0e3,
        raw.solve_wall_seconds * 1.0e3,
        raw.preconditioner_bytes,
        raw.krylov_workspace_bytes,
        raw.adjusted_pivots
    );

    if raw.report.status == SolveStatus::Converged
        && verified_relative_residual > context.args.relative_tolerance * 10.0
    {
        eprintln!(
            "WARNING: reported convergence but independently verified residual {:.6e} exceeds 10x tolerance",
            verified_relative_residual
        );
    }

    Some(CaseMetrics {
        report: raw.report,
        x: x_original,
        analysis_seconds: raw.analysis_seconds,
        prepare_seconds: raw.prepare_seconds,
        solve_wall_seconds: raw.solve_wall_seconds,
        preconditioner_bytes: raw.preconditioner_bytes,
        krylov_workspace_bytes: raw.krylov_workspace_bytes,
        adjusted_pivots: raw.adjusted_pivots,
        verified_relative_residual,
        relative_x_error,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;

    println!(
        "HyBIT {} GeneralSquare ordering benchmark",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!("tolerance           : {:.3e}", args.relative_tolerance);
    println!("max iterations      : {}", args.max_iterations);
    println!("FGMRES restart      : {}", args.restart);

    let load_start = Instant::now();
    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    let load_seconds = load_start.elapsed().as_secs_f64();

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
    println!(
        "CSR storage         : {:.3} MiB",
        mib(matrix.storage_bytes())
    );
    println!("load time           : {:.3} ms", load_seconds * 1.0e3);

    if matrix.nrows() != matrix.ncols() {
        println!("preflight           : unsupported (matrix is not square)");
        println!(
            "PREFLIGHT|supported=false|reason=not_square|nrows={}|ncols={}|missing_diagonal=0|zero_diagonal=0",
            matrix.nrows(),
            matrix.ncols()
        );
        return Ok(());
    }

    let diagonal = diagonal_preflight(&matrix);
    println!(
        "diagonal preflight  : missing={} zero={}",
        diagonal.missing, diagonal.zero
    );
    if let Some(row) = diagonal.first_missing {
        println!("first missing diag  : row {row}");
    }
    if let Some(row) = diagonal.first_zero {
        println!("first zero diag     : row {row}");
    }

    if diagonal.missing != 0 || diagonal.zero != 0 {
        let reason = match (diagonal.missing != 0, diagonal.zero != 0) {
            (true, true) => "missing_and_zero_diagonal",
            (true, false) => "missing_diagonal",
            (false, true) => "zero_diagonal",
            (false, false) => unreachable!(),
        };
        println!(
            "PREFLIGHT|supported=false|reason={reason}|nrows={}|ncols={}|missing_diagonal={}|zero_diagonal={}",
            matrix.nrows(),
            matrix.ncols(),
            diagonal.missing,
            diagonal.zero
        );
        return Ok(());
    }

    println!(
        "PREFLIGHT|supported=true|reason=ok|nrows={}|ncols={}|missing_diagonal=0|zero_diagonal=0",
        matrix.nrows(),
        matrix.ncols()
    );
    if args.preflight_only {
        println!("preflight-only      : eligible for Jacobi/ILU0 ordering cross-check");
        return Ok(());
    }

    let generated_rhs = args.rhs.is_none();
    let b = if let Some(path) = args.rhs.as_deref() {
        println!("RHS                 : {}", path.display());
        load_rhs(path, matrix.nrows())?
    } else {
        println!("RHS                 : generated as b=A*1");
        matrix.spmv(&vec![1.0; matrix.ncols()])?
    };

    let natural_bandwidth = structural_bandwidth(&matrix);

    println!();
    println!("== RCM construction on pattern(A + A^T) ==");
    let graph_start = Instant::now();
    let mut graph = build_symmetrized_graph(&matrix);
    let graph_seconds = graph_start.elapsed().as_secs_f64();
    let undirected_edges = graph.iter().map(Vec::len).sum::<usize>() / 2;

    let rcm_start = Instant::now();
    let new_to_old = reverse_cuthill_mckee(&mut graph);
    let rcm_seconds = rcm_start.elapsed().as_secs_f64();

    let permutation_start = Instant::now();
    let rcm_matrix = symmetric_permute(&matrix, &new_to_old)?;
    let rcm_b = permute_vector(&b, &new_to_old);
    let permutation_seconds = permutation_start.elapsed().as_secs_f64();
    let rcm_bandwidth = structural_bandwidth(&rcm_matrix);

    println!("undirected edges    : {undirected_edges}");
    println!("natural bandwidth   : {natural_bandwidth}");
    println!("RCM bandwidth       : {rcm_bandwidth}");
    if natural_bandwidth > 0 {
        println!(
            "bandwidth ratio     : {:.6} (RCM / natural)",
            rcm_bandwidth as f64 / natural_bandwidth as f64
        );
    }
    println!("graph build         : {:.3} ms", graph_seconds * 1.0e3);
    println!("RCM ordering        : {:.3} ms", rcm_seconds * 1.0e3);
    println!(
        "matrix permutation  : {:.3} ms",
        permutation_seconds * 1.0e3
    );
    let ordering_seconds = graph_seconds + rcm_seconds + permutation_seconds;
    println!("ordering total      : {:.3} ms", ordering_seconds * 1.0e3);
    println!(
        "ORDERING|natural_bandwidth={natural_bandwidth}|rcm_bandwidth={rcm_bandwidth}|bandwidth_ratio={:.9}|ordering_ms={:.6}",
        if natural_bandwidth == 0 {
            0.0
        } else {
            rcm_bandwidth as f64 / natural_bandwidth as f64
        },
        ordering_seconds * 1.0e3
    );

    let context = BenchmarkContext {
        args: &args,
        original_matrix: &matrix,
        original_b: &b,
        generated_rhs,
    };

    let natural_jacobi = benchmark_case(
        &context,
        "natural",
        &matrix,
        &b,
        GeneralSquarePreconditionerPolicy::Jacobi,
        None,
    );

    let rcm_jacobi = benchmark_case(
        &context,
        "rcm",
        &rcm_matrix,
        &rcm_b,
        GeneralSquarePreconditionerPolicy::Jacobi,
        Some(&new_to_old),
    );

    let natural_ilu = benchmark_case(
        &context,
        "natural",
        &matrix,
        &b,
        GeneralSquarePreconditionerPolicy::Ilu0,
        None,
    );

    let rcm_ilu = benchmark_case(
        &context,
        "rcm",
        &rcm_matrix,
        &rcm_b,
        GeneralSquarePreconditionerPolicy::Ilu0,
        Some(&new_to_old),
    );

    println!();
    println!("== Comparison ==");

    if let (Some(natural), Some(rcm)) = (&natural_jacobi, &rcm_jacobi) {
        println!(
            "Jacobi iteration ratio       : {:.6} (RCM / natural)",
            rcm.report.iterations as f64 / natural.report.iterations.max(1) as f64
        );
        println!(
            "Jacobi solve-wall ratio      : {:.6} (RCM / natural)",
            rcm.solve_wall_seconds / natural.solve_wall_seconds.max(f64::MIN_POSITIVE)
        );
        println!(
            "Jacobi verified residuals    : {:.6e} / {:.6e} (natural / RCM)",
            natural.verified_relative_residual, rcm.verified_relative_residual
        );
    } else {
        println!("Jacobi comparison            : incomplete");
    }

    if let (Some(natural), Some(rcm)) = (&natural_ilu, &rcm_ilu) {
        println!(
            "ILU0 iteration ratio         : {:.6} (RCM / natural)",
            rcm.report.iterations as f64 / natural.report.iterations.max(1) as f64
        );
        println!(
            "ILU0 prepare ratio           : {:.6} (RCM / natural)",
            rcm.prepare_seconds / natural.prepare_seconds.max(f64::MIN_POSITIVE)
        );
        println!(
            "ILU0 solve-wall ratio        : {:.6} (RCM / natural)",
            rcm.solve_wall_seconds / natural.solve_wall_seconds.max(f64::MIN_POSITIVE)
        );
        println!(
            "ILU0 adjusted pivots         : {} / {} (natural / RCM)",
            natural.adjusted_pivots, rcm.adjusted_pivots
        );
        println!(
            "ILU0 verified residuals      : {:.6e} / {:.6e} (natural / RCM)",
            natural.verified_relative_residual, rcm.verified_relative_residual
        );
    } else {
        println!("ILU0 comparison              : incomplete");
    }

    // Read these fields in the final checkpoint so clippy keeps the complete
    // measured payload honest even when only selected values are compared above.
    let _telemetry_checksum = [
        natural_jacobi.as_ref(),
        rcm_jacobi.as_ref(),
        natural_ilu.as_ref(),
        rcm_ilu.as_ref(),
    ]
    .into_iter()
    .flatten()
    .fold(0usize, |acc, case| {
        acc ^ case.x.len()
            ^ case.preconditioner_bytes
            ^ case.krylov_workspace_bytes
            ^ case.adjusted_pivots
            ^ case.analysis_seconds.to_bits() as usize
            ^ case.relative_x_error.unwrap_or(0.0).to_bits() as usize
    });

    Ok(())
}
