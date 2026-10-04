use std::collections::VecDeque;
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use hybit::{
    fgmres_with_workspace_and_restart_controller, read_matrix_market, Csr32Matrix, FgmresOptions,
    FgmresRestartProgress, FgmresWorkspace, HybitError, Ilu0Preconditioner, KrylovOutcome,
    SolverOptions,
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
    println!("HyBIT GeneralSquare F6 early-progress ordering signal probe");
    println!();
    println!(
        "Usage: general_square_ordering_progress --matrix A.mtx [--rhs-count 5] [--tol 1e-8] [--max-iters 5000] [--restart 30]"
    );
}

fn norm2(x: &[f64]) -> f64 {
    x.iter().map(|value| value * value).sum::<f64>().sqrt()
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

fn relative_error(actual: &[f64], expected: &[f64]) -> f64 {
    let diff = actual
        .iter()
        .zip(expected)
        .map(|(&actual_value, &expected_value)| {
            let delta = actual_value - expected_value;
            delta * delta
        })
        .sum::<f64>()
        .sqrt();
    diff / norm2(expected).max(f64::MIN_POSITIVE)
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

fn structural_bandwidth(matrix: &Csr32Matrix) -> usize {
    let mut bandwidth = 0usize;
    for row in 0..matrix.nrows() {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        for &col in &matrix.col_idx()[start..end] {
            bandwidth = bandwidth.max(row.abs_diff(col as usize));
        }
    }
    bandwidth
}

fn build_symmetrized_graph(matrix: &Csr32Matrix) -> Vec<Vec<u32>> {
    let n = matrix.nrows();
    let mut graph = vec![Vec::<u32>::new(); n];

    for row in 0..n {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        for &col_u32 in &matrix.col_idx()[start..end] {
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

fn symmetric_permute(
    matrix: &Csr32Matrix,
    new_to_old: &[usize],
) -> Result<Csr32Matrix, Box<dyn Error>> {
    let n = matrix.nrows();
    if matrix.ncols() != n || new_to_old.len() != n {
        return Err("symmetric permutation requires a square matrix".into());
    }

    let mut old_to_new = vec![usize::MAX; n];
    for (new, &old) in new_to_old.iter().enumerate() {
        if old >= n || old_to_new[old] != usize::MAX {
            return Err("invalid RCM permutation".into());
        }
        old_to_new[old] = new;
    }

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(matrix.nnz());
    let mut values = Vec::with_capacity(matrix.nnz());
    let mut entries = Vec::<(u32, f64)>::new();
    row_ptr.push(0u32);

    for &old_row in new_to_old {
        entries.clear();
        let start = matrix.row_ptr()[old_row] as usize;
        let end = matrix.row_ptr()[old_row + 1] as usize;

        for p in start..end {
            let old_col = matrix.col_idx()[p] as usize;
            entries.push((u32::try_from(old_to_new[old_col])?, matrix.values()[p]));
        }
        entries.sort_unstable_by_key(|&(col, _)| col);

        for &(col, value) in &entries {
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

struct SolveRun {
    outcome: KrylovOutcome,
    x: Vec<f64>,
    progress: Vec<FgmresRestartProgress>,
    solve_ms: f64,
}

fn solve_with_progress(
    matrix: &Csr32Matrix,
    preconditioner: &mut Ilu0Preconditioner,
    b: &[f64],
    args: &Args,
) -> Result<SolveRun, HybitError> {
    let mut x = vec![0.0; matrix.ncols()];
    let mut workspace = FgmresWorkspace::new(matrix.nrows(), args.restart)?;
    let options = FgmresOptions {
        solver: SolverOptions {
            relative_tolerance: args.relative_tolerance,
            absolute_tolerance: 0.0,
            max_iterations: args.max_iterations,
        },
        restart: args.restart,
    };
    let mut progress = Vec::new();

    let start = Instant::now();
    let outcome = fgmres_with_workspace_and_restart_controller(
        matrix,
        preconditioner,
        b,
        &mut x,
        options,
        &mut workspace,
        |boundary| {
            progress.push(boundary);
            Ok(args.restart)
        },
    )?;
    let solve_ms = start.elapsed().as_secs_f64() * 1.0e3;

    Ok(SolveRun {
        outcome,
        x,
        progress,
        solve_ms,
    })
}

fn log_decay_per_iteration(initial: f64, final_residual: f64, iterations: usize) -> Option<f64> {
    if iterations == 0
        || !initial.is_finite()
        || !final_residual.is_finite()
        || initial <= 0.0
        || final_residual <= 0.0
        || final_residual >= initial
    {
        return None;
    }
    Some((initial / final_residual).ln() / iterations as f64)
}

fn projected_total_iterations(
    initial: f64,
    residual: f64,
    target: f64,
    completed_iterations: usize,
) -> Option<f64> {
    let rate = log_decay_per_iteration(initial, residual, completed_iterations)?;
    if residual <= target {
        return Some(completed_iterations as f64);
    }
    if rate <= 0.0 || !rate.is_finite() {
        return None;
    }
    let remaining = (residual / target).ln() / rate;
    Some(completed_iterations as f64 + remaining.max(0.0))
}

fn option_number(value: Option<f64>) -> String {
    match value {
        Some(value) if value.is_finite() => format!("{value:.9e}"),
        _ => "none".to_string(),
    }
}

fn emit_progress(ordering: &str, rhs: usize, family: &str, progress: &[FgmresRestartProgress]) {
    for (cycle_index, boundary) in progress.iter().enumerate() {
        let ratio =
            boundary.final_residual / boundary.cycle_initial_residual.max(f64::MIN_POSITIVE);
        let rate = log_decay_per_iteration(
            boundary.cycle_initial_residual,
            boundary.final_residual,
            boundary.cycle_iterations,
        );
        println!(
            "F6_PROGRESS|ordering={ordering}|rhs={rhs}|family={family}|cycle={}|restart={}|cycle_iterations={}|total_iterations={}|cycle_initial_residual={:.9e}|final_residual={:.9e}|cycle_ratio={ratio:.9e}|log_decay_per_iter={}|remaining_iterations={}",
            cycle_index + 1,
            boundary.restart,
            boundary.cycle_iterations,
            boundary.total_iterations,
            boundary.cycle_initial_residual,
            boundary.final_residual,
            option_number(rate),
            boundary.remaining_iterations,
        );
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;

    println!(
        "HyBIT {} GeneralSquare F6 early-progress ordering signal probe",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!("RHS count           : {}", args.rhs_count);
    println!("restart             : {}", args.restart);
    println!("max iterations      : {}", args.max_iterations);
    println!("relative tolerance  : {:.3e}", args.relative_tolerance);

    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    if matrix.nrows() != matrix.ncols() {
        return Err("F6 ordering-progress probe requires a square matrix".into());
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

    let natural_bandwidth = structural_bandwidth(&matrix);

    let graph_start = Instant::now();
    let mut graph = build_symmetrized_graph(&matrix);
    let graph_ms = graph_start.elapsed().as_secs_f64() * 1.0e3;

    let rcm_start = Instant::now();
    let new_to_old = reverse_cuthill_mckee(&mut graph);
    let rcm_ms = rcm_start.elapsed().as_secs_f64() * 1.0e3;

    let permutation_start = Instant::now();
    let rcm_matrix = symmetric_permute(&matrix, &new_to_old)?;
    let permutation_ms = permutation_start.elapsed().as_secs_f64() * 1.0e3;
    let ordering_ms = graph_ms + rcm_ms + permutation_ms;
    let rcm_bandwidth = structural_bandwidth(&rcm_matrix);
    let bandwidth_ratio = if natural_bandwidth == 0 {
        0.0
    } else {
        rcm_bandwidth as f64 / natural_bandwidth as f64
    };

    println!(
        "F6_ORDERING|natural_bandwidth={natural_bandwidth}|rcm_bandwidth={rcm_bandwidth}|bandwidth_ratio={bandwidth_ratio:.9e}|graph_ms={graph_ms:.6}|rcm_ms={rcm_ms:.6}|permutation_ms={permutation_ms:.6}|ordering_ms={ordering_ms:.6}"
    );

    let natural_prepare_start = Instant::now();
    let mut natural_ilu = Ilu0Preconditioner::from_csr32_general(&matrix)?;
    let natural_prepare_ms = natural_prepare_start.elapsed().as_secs_f64() * 1.0e3;

    let rcm_prepare_start = Instant::now();
    let mut rcm_ilu = Ilu0Preconditioner::from_csr32_general(&rcm_matrix)?;
    let rcm_prepare_ms = rcm_prepare_start.elapsed().as_secs_f64() * 1.0e3;

    println!(
        "F6_SETUP|ordering=natural|prepare_ms={natural_prepare_ms:.6}|factor_bytes={}|adjusted_pivots={}",
        natural_ilu.factor_bytes(),
        natural_ilu.adjusted_pivots()
    );
    println!(
        "F6_SETUP|ordering=rcm|prepare_ms={rcm_prepare_ms:.6}|factor_bytes={}|adjusted_pivots={}",
        rcm_ilu.factor_bytes(),
        rcm_ilu.adjusted_pivots()
    );

    for rhs_index in 0..args.rhs_count {
        let rhs = rhs_index + 1;
        let (exact, family) = exact_solution(matrix.ncols(), rhs_index);
        let b = matrix.spmv(&exact)?;
        let b_norm = norm2(&b);
        if !b_norm.is_finite() || b_norm == 0.0 {
            return Err(format!("RHS {rhs} has zero or non-finite norm").into());
        }
        let target = args.relative_tolerance * b_norm.max(f64::MIN_POSITIVE);

        let b_permute_start = Instant::now();
        let rcm_b = permute_vector(&b, &new_to_old);
        let rhs_permute_ms = b_permute_start.elapsed().as_secs_f64() * 1.0e3;

        let (natural, mut rcm) = if rhs_index % 2 == 0 {
            let natural = solve_with_progress(&matrix, &mut natural_ilu, &b, &args)?;
            let rcm = solve_with_progress(&rcm_matrix, &mut rcm_ilu, &rcm_b, &args)?;
            (natural, rcm)
        } else {
            let rcm = solve_with_progress(&rcm_matrix, &mut rcm_ilu, &rcm_b, &args)?;
            let natural = solve_with_progress(&matrix, &mut natural_ilu, &b, &args)?;
            (natural, rcm)
        };

        emit_progress("natural", rhs, family, &natural.progress);
        emit_progress("rcm", rhs, family, &rcm.progress);

        let natural_verified = verified_relative_residual(&matrix, &b, &natural.x)?;
        let natural_x_error = relative_error(&natural.x, &exact);

        let unpermute_start = Instant::now();
        let rcm_x_original = unpermute_vector(&rcm.x, &new_to_old);
        let solution_unpermute_ms = unpermute_start.elapsed().as_secs_f64() * 1.0e3;
        rcm.x = rcm_x_original;

        let rcm_verified = verified_relative_residual(&matrix, &b, &rcm.x)?;
        let rcm_x_error = relative_error(&rcm.x, &exact);

        let natural_first_cycle_converged = natural.outcome.iterations <= args.restart
            && natural.outcome.status == hybit::SolveStatus::Converged;

        let (probe_iterations, probe_residual) = if let Some(first) = natural.progress.first() {
            (first.total_iterations, first.final_residual)
        } else {
            (natural.outcome.iterations, natural.outcome.final_residual)
        };

        let first_ratio = probe_residual / natural.outcome.initial_residual.max(f64::MIN_POSITIVE);
        let first_rate = log_decay_per_iteration(
            natural.outcome.initial_residual,
            probe_residual,
            probe_iterations,
        );
        let projected_total = projected_total_iterations(
            natural.outcome.initial_residual,
            probe_residual,
            target,
            probe_iterations,
        );

        let cycle2_rate_ratio = if natural.progress.len() >= 2 {
            let p1 = natural.progress[0];
            let p2 = natural.progress[1];
            let r1 = log_decay_per_iteration(
                p1.cycle_initial_residual,
                p1.final_residual,
                p1.cycle_iterations,
            );
            let r2 = log_decay_per_iteration(
                p2.cycle_initial_residual,
                p2.final_residual,
                p2.cycle_iterations,
            );
            match (r1, r2) {
                (Some(a), Some(b)) if a > 0.0 => Some(b / a),
                _ => None,
            }
        } else {
            None
        };

        let iteration_ratio =
            rcm.outcome.iterations as f64 / (natural.outcome.iterations.max(1) as f64);
        let solve_ratio = rcm.solve_ms / natural.solve_ms.max(f64::MIN_POSITIVE);
        let rcm_first_rhs_cost_ms =
            ordering_ms + rcm_prepare_ms + rhs_permute_ms + rcm.solve_ms + solution_unpermute_ms;
        let natural_first_rhs_cost_ms = natural_prepare_ms + natural.solve_ms;
        let first_rhs_ratio =
            rcm_first_rhs_cost_ms / natural_first_rhs_cost_ms.max(f64::MIN_POSITIVE);

        println!(
            "F6_RESULT|rhs={rhs}|family={family}|natural_status={:?}|natural_iterations={}|natural_verified_residual={natural_verified:.9e}|natural_x_error={natural_x_error:.9e}|natural_solve_ms={:.6}|rcm_status={:?}|rcm_iterations={}|rcm_verified_residual={rcm_verified:.9e}|rcm_x_error={rcm_x_error:.9e}|rcm_solve_ms={:.6}|iteration_ratio={iteration_ratio:.9e}|solve_ratio={solve_ratio:.9e}|rcm_first_rhs_cost_ms={rcm_first_rhs_cost_ms:.6}|natural_first_rhs_cost_ms={natural_first_rhs_cost_ms:.6}|first_rhs_ratio={first_rhs_ratio:.9e}",
            natural.outcome.status,
            natural.outcome.iterations,
            natural.solve_ms,
            rcm.outcome.status,
            rcm.outcome.iterations,
            rcm.solve_ms,
        );

        println!(
            "F6_SIGNAL|rhs={rhs}|family={family}|converged_first_cycle={natural_first_cycle_converged}|probe_iterations={probe_iterations}|first_ratio={first_ratio:.9e}|first_log_decay_per_iter={}|projected_total_iterations={}|cycle2_rate_ratio={}|bandwidth_ratio={bandwidth_ratio:.9e}|natural_actual_iterations={}|rcm_actual_iterations={}|iteration_ratio={iteration_ratio:.9e}|solve_ratio={solve_ratio:.9e}|first_rhs_ratio={first_rhs_ratio:.9e}",
            option_number(first_rate),
            option_number(projected_total),
            option_number(cycle2_rate_ratio),
            natural.outcome.iterations,
            rcm.outcome.iterations,
        );

        if natural.outcome.status == hybit::SolveStatus::Converged
            && natural_verified > args.relative_tolerance * 10.0
        {
            return Err(format!(
                "Natural reported convergence but verified residual is {natural_verified:e}"
            )
            .into());
        }
        if rcm.outcome.status == hybit::SolveStatus::Converged
            && rcm_verified > args.relative_tolerance * 10.0
        {
            return Err(format!(
                "RCM reported convergence but verified residual is {rcm_verified:e}"
            )
            .into());
        }
    }

    Ok(())
}
