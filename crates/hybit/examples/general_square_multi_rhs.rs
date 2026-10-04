use std::collections::VecDeque;
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use hybit::{
    read_matrix_market, Csr32Matrix, GeneralSquareOptions, GeneralSquarePreconditionerPolicy,
    HybitPreparedSystem, HybitSolver, MatrixProblemClass, SolveReport, SolverOptions,
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
        let mut relative_tolerance: f64 = 1.0e-8;
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
        if rhs_count == 0 {
            return Err("--rhs-count must be > 0".into());
        }
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
            rhs_count,
            relative_tolerance,
            max_iterations,
            restart,
        })
    }
}

fn print_usage() {
    println!("HyBIT GeneralSquare prepared multi-RHS ILU(0) benchmark");
    println!();
    println!("Usage:");
    println!(
        "  cargo run --release -p hybit --example general_square_multi_rhs -- --matrix A.mtx [options]"
    );
    println!();
    println!("Options:");
    println!("  --rhs-count N     deterministic RHS count (default 5)");
    println!("  --tol VALUE       relative tolerance (default 1e-8)");
    println!("  --max-iters N     maximum FGMRES iterations per RHS (default 5000)");
    println!("  --restart N       fixed FGMRES restart dimension (default 30)");
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
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

fn relative_error(x: &[f64], exact: &[f64]) -> f64 {
    let diff = x
        .iter()
        .zip(exact.iter())
        .map(|(&xi, &ei)| {
            let d = xi - ei;
            d * d
        })
        .sum::<f64>()
        .sqrt();
    diff / norm2(exact).max(f64::MIN_POSITIVE)
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

fn prepare_ilu0(matrix: &Csr32Matrix, args: &Args) -> Result<HybitPreparedSystem, Box<dyn Error>> {
    let mut solver = HybitSolver::new();
    solver.set_problem_class(MatrixProblemClass::GeneralSquare);
    solver.set_options(SolverOptions {
        relative_tolerance: args.relative_tolerance,
        absolute_tolerance: 0.0,
        max_iterations: args.max_iterations,
    })?;
    solver.set_general_square_preconditioner_policy(GeneralSquarePreconditionerPolicy::Ilu0);
    solver.set_general_square_options(GeneralSquareOptions {
        restart: args.restart,
        ..GeneralSquareOptions::default()
    })?;

    let analysis = solver.analyze_csr32(matrix)?;
    Ok(solver.prepare_csr32(matrix, &analysis)?)
}

struct RawSolve {
    report: SolveReport,
    x_original: Vec<f64>,
    solve_wall_seconds: f64,
    solution_unpermute_seconds: f64,
}

struct SolveMetrics {
    report: SolveReport,
    verified_relative_residual: f64,
    relative_x_error: f64,
    solve_wall_seconds: f64,
    solution_unpermute_seconds: f64,
}

fn solve_prepared_raw(
    prepared: &mut HybitPreparedSystem,
    solve_matrix: &Csr32Matrix,
    solve_b: &[f64],
    new_to_old: Option<&[usize]>,
) -> Result<RawSolve, Box<dyn Error>> {
    let mut x = vec![0.0; solve_matrix.ncols()];
    let start = Instant::now();
    let report = prepared.solve(solve_matrix, solve_b, &mut x)?;
    let solve_wall_seconds = start.elapsed().as_secs_f64();

    let transform_start = Instant::now();
    let x_original = if let Some(permutation) = new_to_old {
        unpermute_vector(&x, permutation)
    } else {
        x
    };
    let solution_unpermute_seconds = if new_to_old.is_some() {
        transform_start.elapsed().as_secs_f64()
    } else {
        0.0
    };

    Ok(RawSolve {
        report,
        x_original,
        solve_wall_seconds,
        solution_unpermute_seconds,
    })
}

fn finish_metrics(
    raw: RawSolve,
    original_matrix: &Csr32Matrix,
    original_b: &[f64],
    exact: &[f64],
) -> Result<SolveMetrics, Box<dyn Error>> {
    Ok(SolveMetrics {
        verified_relative_residual: verified_relative_residual(
            original_matrix,
            original_b,
            &raw.x_original,
        )?,
        relative_x_error: relative_error(&raw.x_original, exact),
        report: raw.report,
        solve_wall_seconds: raw.solve_wall_seconds,
        solution_unpermute_seconds: raw.solution_unpermute_seconds,
    })
}

fn print_result(
    ordering: &str,
    rhs: usize,
    family: &str,
    rhs_permute_seconds: f64,
    m: &SolveMetrics,
) {
    let transform_ms = (rhs_permute_seconds + m.solution_unpermute_seconds) * 1.0e3;
    println!(
        "{ordering:7} rhs {rhs:2} ({family:11}) : {:?}, {:5} it, residual {:.6e}, xerr {:.6e}, wall {:.3} ms, transform {:.3} ms, sequence {}, reused={}",
        m.report.status,
        m.report.iterations,
        m.verified_relative_residual,
        m.relative_x_error,
        m.solve_wall_seconds * 1.0e3,
        transform_ms,
        m.report.solve_sequence,
        m.report.preconditioner_reused
    );
    println!(
        "REUSE_RESULT|ordering={ordering}|rhs={rhs}|family={family}|sequence={}|reused={}|status={:?}|iterations={}|reported_residual={:.6e}|verified_residual={:.6e}|relative_x_error={:.6e}|solve_ms={:.6}|solve_wall_ms={:.6}|rhs_permute_ms={:.6}|solution_unpermute_ms={:.6}|transform_ms={:.6}",
        m.report.solve_sequence,
        m.report.preconditioner_reused,
        m.report.status,
        m.report.iterations,
        m.report.relative_residual,
        m.verified_relative_residual,
        m.relative_x_error,
        m.report.solve_seconds * 1.0e3,
        m.solve_wall_seconds * 1.0e3,
        rhs_permute_seconds * 1.0e3,
        m.solution_unpermute_seconds * 1.0e3,
        transform_ms
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;

    println!(
        "HyBIT {} GeneralSquare prepared multi-RHS ILU(0) benchmark",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!("RHS count           : {}", args.rhs_count);
    println!("tolerance           : {:.3e}", args.relative_tolerance);
    println!("max iterations      : {}", args.max_iterations);
    println!("FGMRES restart      : {}", args.restart);

    let load_start = Instant::now();
    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    let load_seconds = load_start.elapsed().as_secs_f64();

    if matrix.nrows() != matrix.ncols() {
        return Err("multi-RHS benchmark requires a square matrix".into());
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
    println!(
        "CSR storage         : {:.3} MiB",
        mib(matrix.storage_bytes())
    );
    println!("load time           : {:.3} ms", load_seconds * 1.0e3);

    println!();
    println!("== RCM construction on pattern(A + A^T) ==");
    let natural_bandwidth = structural_bandwidth(&matrix);

    let graph_start = Instant::now();
    let mut graph = build_symmetrized_graph(&matrix);
    let graph_seconds = graph_start.elapsed().as_secs_f64();

    let rcm_start = Instant::now();
    let new_to_old = reverse_cuthill_mckee(&mut graph);
    let rcm_seconds = rcm_start.elapsed().as_secs_f64();

    let permutation_start = Instant::now();
    let rcm_matrix = symmetric_permute(&matrix, &new_to_old)?;
    let permutation_seconds = permutation_start.elapsed().as_secs_f64();

    let rcm_bandwidth = structural_bandwidth(&rcm_matrix);
    let ordering_seconds = graph_seconds + rcm_seconds + permutation_seconds;

    println!("natural bandwidth   : {natural_bandwidth}");
    println!("RCM bandwidth       : {rcm_bandwidth}");
    println!("graph build         : {:.3} ms", graph_seconds * 1.0e3);
    println!("RCM ordering        : {:.3} ms", rcm_seconds * 1.0e3);
    println!(
        "matrix permutation  : {:.3} ms",
        permutation_seconds * 1.0e3
    );
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

    println!();
    println!("== prepare Natural ILU(0) ==");
    let mut natural = prepare_ilu0(&matrix, &args)?;
    println!(
        "analysis            : {:.3} ms",
        natural.analysis_seconds() * 1.0e3
    );
    println!(
        "prepare             : {:.3} ms",
        natural.prepare_seconds() * 1.0e3
    );
    println!(
        "preconditioner      : {:.3} MiB",
        mib(natural.general_square_preconditioner_bytes())
    );
    println!(
        "Krylov workspace    : {:.3} MiB",
        mib(natural.krylov_workspace_bytes())
    );
    println!(
        "adjusted pivots     : {}",
        natural.general_square_ilu_adjusted_pivots()
    );
    println!(
        "SETUP|ordering=natural|ordering_ms=0.000000|analysis_ms={:.6}|prepare_ms={:.6}|preconditioner_bytes={}|workspace_bytes={}|adjusted_pivots={}",
        natural.analysis_seconds() * 1.0e3,
        natural.prepare_seconds() * 1.0e3,
        natural.general_square_preconditioner_bytes(),
        natural.krylov_workspace_bytes(),
        natural.general_square_ilu_adjusted_pivots()
    );

    println!();
    println!("== prepare RCM ILU(0) ==");
    let mut rcm = prepare_ilu0(&rcm_matrix, &args)?;
    println!(
        "analysis            : {:.3} ms",
        rcm.analysis_seconds() * 1.0e3
    );
    println!(
        "prepare             : {:.3} ms",
        rcm.prepare_seconds() * 1.0e3
    );
    println!(
        "preconditioner      : {:.3} MiB",
        mib(rcm.general_square_preconditioner_bytes())
    );
    println!(
        "Krylov workspace    : {:.3} MiB",
        mib(rcm.krylov_workspace_bytes())
    );
    println!(
        "adjusted pivots     : {}",
        rcm.general_square_ilu_adjusted_pivots()
    );
    println!(
        "SETUP|ordering=rcm|ordering_ms={:.6}|analysis_ms={:.6}|prepare_ms={:.6}|preconditioner_bytes={}|workspace_bytes={}|adjusted_pivots={}",
        ordering_seconds * 1.0e3,
        rcm.analysis_seconds() * 1.0e3,
        rcm.prepare_seconds() * 1.0e3,
        rcm.general_square_preconditioner_bytes(),
        rcm.krylov_workspace_bytes(),
        rcm.general_square_ilu_adjusted_pivots()
    );

    let natural_setup_ms = (natural.analysis_seconds() + natural.prepare_seconds()) * 1.0e3;
    let rcm_setup_ms =
        ordering_seconds * 1.0e3 + (rcm.analysis_seconds() + rcm.prepare_seconds()) * 1.0e3;

    let mut natural_solve_ms = 0.0f64;
    let mut rcm_solve_ms = 0.0f64;
    let mut rcm_transform_ms = 0.0f64;
    let mut first_break_even_solve_only = None;
    let mut first_break_even_end_to_end = None;

    println!();
    println!("== prepared solve-many ==");

    for rhs_index in 0..args.rhs_count {
        let rhs_number = rhs_index + 1;
        let (exact, family) = exact_solution(matrix.ncols(), rhs_index);
        let original_b = matrix.spmv(&exact)?;

        let (natural_raw, rcm_raw, rhs_permute_seconds) = if rhs_index % 2 == 0 {
            let n = solve_prepared_raw(&mut natural, &matrix, &original_b, None)?;

            let rhs_transform_start = Instant::now();
            let rcm_b = permute_vector(&original_b, &new_to_old);
            let rhs_permute_seconds = rhs_transform_start.elapsed().as_secs_f64();

            let r = solve_prepared_raw(&mut rcm, &rcm_matrix, &rcm_b, Some(&new_to_old))?;
            (n, r, rhs_permute_seconds)
        } else {
            let rhs_transform_start = Instant::now();
            let rcm_b = permute_vector(&original_b, &new_to_old);
            let rhs_permute_seconds = rhs_transform_start.elapsed().as_secs_f64();

            let r = solve_prepared_raw(&mut rcm, &rcm_matrix, &rcm_b, Some(&new_to_old))?;
            let n = solve_prepared_raw(&mut natural, &matrix, &original_b, None)?;
            (n, r, rhs_permute_seconds)
        };

        // Verify both solutions only after both timed solves are complete so
        // the original-matrix residual check cannot warm one timed solver path.
        let natural_metrics = finish_metrics(natural_raw, &matrix, &original_b, &exact)?;
        let rcm_metrics = finish_metrics(rcm_raw, &matrix, &original_b, &exact)?;

        if natural_metrics.report.solve_sequence != rhs_number
            || rcm_metrics.report.solve_sequence != rhs_number
        {
            return Err(format!(
                "prepared solve sequence mismatch at RHS {rhs_number}: natural={} rcm={}",
                natural_metrics.report.solve_sequence, rcm_metrics.report.solve_sequence
            )
            .into());
        }

        let expected_reuse = rhs_number > 1;
        if natural_metrics.report.preconditioner_reused != expected_reuse
            || rcm_metrics.report.preconditioner_reused != expected_reuse
        {
            return Err(format!(
                "prepared reuse flag mismatch at RHS {rhs_number}: natural={} rcm={} expected={expected_reuse}",
                natural_metrics.report.preconditioner_reused,
                rcm_metrics.report.preconditioner_reused
            )
            .into());
        }

        print_result("natural", rhs_number, family, 0.0, &natural_metrics);
        print_result("rcm", rhs_number, family, rhs_permute_seconds, &rcm_metrics);

        natural_solve_ms += natural_metrics.solve_wall_seconds * 1.0e3;
        rcm_solve_ms += rcm_metrics.solve_wall_seconds * 1.0e3;
        rcm_transform_ms += (rhs_permute_seconds + rcm_metrics.solution_unpermute_seconds) * 1.0e3;

        let natural_total_solve_only_ms = natural_setup_ms + natural_solve_ms;
        let rcm_total_solve_only_ms = rcm_setup_ms + rcm_solve_ms;
        let solve_only_ratio =
            rcm_total_solve_only_ms / natural_total_solve_only_ms.max(f64::MIN_POSITIVE);

        let natural_total_end_to_end_ms = natural_total_solve_only_ms;
        let rcm_total_end_to_end_ms = rcm_total_solve_only_ms + rcm_transform_ms;
        let end_to_end_ratio =
            rcm_total_end_to_end_ms / natural_total_end_to_end_ms.max(f64::MIN_POSITIVE);

        if first_break_even_solve_only.is_none()
            && rcm_total_solve_only_ms < natural_total_solve_only_ms
        {
            first_break_even_solve_only = Some(rhs_number);
        }
        if first_break_even_end_to_end.is_none()
            && rcm_total_end_to_end_ms < natural_total_end_to_end_ms
        {
            first_break_even_end_to_end = Some(rhs_number);
        }

        println!(
            "solve-only through RHS {rhs_number}: Natural {:.3} ms, RCM {:.3} ms, ratio {:.6}",
            natural_total_solve_only_ms, rcm_total_solve_only_ms, solve_only_ratio
        );
        println!(
            "AMORTIZED_SOLVE_ONLY|rhs_count={rhs_number}|natural_setup_ms={natural_setup_ms:.6}|rcm_setup_ms={rcm_setup_ms:.6}|natural_cumulative_solve_ms={natural_solve_ms:.6}|rcm_cumulative_solve_ms={rcm_solve_ms:.6}|natural_total_ms={natural_total_solve_only_ms:.6}|rcm_total_ms={rcm_total_solve_only_ms:.6}|rcm_over_natural={solve_only_ratio:.9}"
        );
        println!(
            "end-to-end through RHS {rhs_number}: Natural {:.3} ms, RCM {:.3} ms, transform {:.3} ms, ratio {:.6}",
            natural_total_end_to_end_ms,
            rcm_total_end_to_end_ms,
            rcm_transform_ms,
            end_to_end_ratio
        );
        println!(
            "AMORTIZED_END_TO_END|rhs_count={rhs_number}|natural_total_ms={natural_total_end_to_end_ms:.6}|rcm_total_ms={rcm_total_end_to_end_ms:.6}|rcm_cumulative_transform_ms={rcm_transform_ms:.6}|rcm_over_natural={end_to_end_ratio:.9}"
        );
    }

    println!();
    println!("== final reuse summary ==");
    println!("Natural solve count : {}", natural.solve_count());
    println!("RCM solve count     : {}", rcm.solve_count());
    match first_break_even_solve_only {
        Some(rhs) => println!("RCM solve-only break-even : RHS {rhs}"),
        None => println!("RCM solve-only break-even : not reached"),
    }
    match first_break_even_end_to_end {
        Some(rhs) => println!("RCM end-to-end break-even : RHS {rhs}"),
        None => println!("RCM end-to-end break-even : not reached"),
    }
    println!(
        "BREAK_EVEN_SOLVE_ONLY|rhs={}",
        first_break_even_solve_only
            .map(|v| v.to_string())
            .unwrap_or_else(|| "none".to_string())
    );
    println!(
        "BREAK_EVEN_END_TO_END|rhs={}",
        first_break_even_end_to_end
            .map(|v| v.to_string())
            .unwrap_or_else(|| "none".to_string())
    );

    Ok(())
}
