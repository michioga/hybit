//! C1-H15-C1: read-only research replay of FGMRES restart-boundary telemetry.
//!
//! This example never changes HyBIT's prepared API or automatic routing.
//! `BudgetAwareMirror` intentionally copies the current internal controller;
//! its parity with `fem_fgmres_bench --restart-policy budget-aware` must be
//! independently checked on the same matrix, RHS and budgets.

use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hybit::{
    Csr32Matrix, FgmresOptions, FgmresRestartProgress, FgmresWorkspace, HybitError,
    Ilu0Preconditioner, SolveStatus, SolverOptions, fgmres_with_workspace_and_restart_controller,
    read_matrix_market,
};

const WINDOW_CYCLES: usize = 4;
const BUDGET_PRESSURE: f64 = 1.10;
const EMERGENCY_PRESSURE: f64 = 1.40;
const DECISION_HITS: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy {
    Fixed,
    BudgetAwareMirror,
}

impl Policy {
    fn name(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::BudgetAwareMirror => "budget-aware-mirror",
        }
    }
}

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    rhs: PathBuf,
    policy: Policy,
    tol: f64,
    max_iters: usize,
    restart: usize,
    max_restart: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut rhs = None;
        let mut policy = Policy::BudgetAwareMirror;
        let mut tol = 1.0e-10_f64;
        let mut max_iters = 360_usize;
        let mut restart = 30_usize;
        let mut max_restart = 70_usize;
        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            let value = match arg.as_str() {
                "--matrix" | "--rhs" | "--policy" | "--tol" | "--max-iters" | "--restart"
                | "--max-restart" => it.next().ok_or(format!("{arg} requires a value"))?,
                "-h" | "--help" => {
                    println!("Usage: c1h15_restart_trace --matrix FILE.mm --rhs FILE.rhs");
                    println!("  [--policy fixed|budget-aware-mirror] [--tol 1e-10]");
                    println!("  [--max-iters 360] [--restart 30] [--max-restart 70]");
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument: {arg}").into()),
            };
            match arg.as_str() {
                "--matrix" => matrix = Some(PathBuf::from(value)),
                "--rhs" => rhs = Some(PathBuf::from(value)),
                "--policy" => {
                    policy = match value.as_str() {
                        "fixed" => Policy::Fixed,
                        "budget-aware-mirror" => Policy::BudgetAwareMirror,
                        _ => return Err(format!("unknown policy: {value}").into()),
                    };
                }
                "--tol" => tol = value.parse()?,
                "--max-iters" => max_iters = value.parse()?,
                "--restart" => restart = value.parse()?,
                "--max-restart" => max_restart = value.parse()?,
                _ => unreachable!(),
            }
        }
        if !tol.is_finite() || tol <= 0.0 || max_iters == 0 || restart == 0 || max_restart < restart
        {
            return Err("invalid tolerance or restart/iteration budget".into());
        }
        Ok(Self {
            matrix: matrix.ok_or("--matrix is required")?,
            rhs: rhs.ok_or("--rhs is required")?,
            policy,
            tol,
            max_iters,
            restart,
            max_restart,
        })
    }
}

fn read_rhs(path: &Path, n: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    let source = fs::read_to_string(path)?;
    let mut rhs = Vec::with_capacity(n);
    for word in source.split_whitespace() {
        let value: f64 = word.replace('D', "E").replace('d', "e").parse()?;
        if !value.is_finite() || rhs.len() >= n {
            return Err("invalid or oversized RHS".into());
        }
        rhs.push(value);
    }
    if rhs.len() != n {
        return Err(format!("RHS has {} values, expected {n}", rhs.len()).into());
    }
    Ok(rhs)
}

fn norm(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().fold(0.0_f64, f64::hypot)
}

fn independent_relative(a: &Csr32Matrix, b: &[f64], x: &[f64]) -> Result<f64, HybitError> {
    let ax = a.spmv(x)?;
    let residual = norm(b.iter().zip(ax).map(|(&bi, ai)| bi - ai));
    let bnorm = norm(b.iter().copied());
    Ok(if bnorm == 0.0 {
        residual
    } else {
        residual / bnorm
    })
}

/// Snapshot of current `hybit-auto` BudgetAwareFgmresRestartController.
/// This intentionally remains in an example, not the production API.
struct BudgetAwareMirror {
    target: f64,
    max_restart: usize,
    window_start_residual: Option<f64>,
    window_iterations: usize,
    window_cycles: usize,
    previous_rate: Option<f64>,
    pressure_hits: usize,
}

impl BudgetAwareMirror {
    fn new(target: f64, max_restart: usize) -> Self {
        Self {
            target,
            max_restart,
            window_start_residual: None,
            window_iterations: 0,
            window_cycles: 0,
            previous_rate: None,
            pressure_hits: 0,
        }
    }

    fn next_restart(&mut self, p: FgmresRestartProgress) -> usize {
        if p.restart >= self.max_restart {
            return p.restart;
        }
        let window_start = self
            .window_start_residual
            .get_or_insert(p.cycle_initial_residual);
        self.window_iterations = self.window_iterations.saturating_add(p.cycle_iterations);
        self.window_cycles = self.window_cycles.saturating_add(1);
        if self.window_cycles < WINDOW_CYCLES {
            return p.restart;
        }

        let ratio = p.final_residual / (*window_start).max(f64::MIN_POSITIVE);
        let rate = if ratio.is_finite() && ratio > 0.0 && ratio < 1.0 && self.window_iterations > 0
        {
            -ratio.ln() / self.window_iterations as f64
        } else {
            0.0
        };
        let needed = if p.final_residual <= self.target {
            0.0
        } else if rate.is_finite() && rate > 0.0 {
            (p.final_residual / self.target).ln() / rate
        } else {
            f64::INFINITY
        };
        let pressure = if p.remaining_iterations == 0 {
            f64::INFINITY
        } else {
            needed / p.remaining_iterations as f64
        };
        let rate_ratio = self
            .previous_rate
            .filter(|previous| *previous > 0.0)
            .map(|previous| rate / previous);
        let trend_allows_escalation = rate_ratio.is_some_and(|ratio| ratio <= 1.0);
        let emergency = pressure >= EMERGENCY_PRESSURE;
        let ordinary = pressure >= BUDGET_PRESSURE && trend_allows_escalation;

        if ordinary || emergency {
            self.pressure_hits = self.pressure_hits.saturating_add(1);
        } else {
            self.pressure_hits = 0;
        }
        let mut next_restart = p.restart;
        if self.pressure_hits >= DECISION_HITS {
            next_restart = p.restart.saturating_mul(2).min(self.max_restart);
            self.pressure_hits = 0;
            self.previous_rate = None;
        } else {
            self.previous_rate = Some(rate);
        }
        self.window_start_residual = Some(p.final_residual);
        self.window_iterations = 0;
        self.window_cycles = 0;
        next_restart
    }
}

fn trace_progress(
    cycle: usize,
    progress: FgmresRestartProgress,
    next_restart: usize,
    rhs_norm: f64,
) {
    let relative = if rhs_norm == 0.0 {
        progress.final_residual
    } else {
        progress.final_residual / rhs_norm
    };
    let decay_rate = if progress.cycle_iterations > 0
        && progress.cycle_initial_residual.is_finite()
        && progress.cycle_initial_residual > 0.0
        && progress.final_residual.is_finite()
        && progress.final_residual > 0.0
    {
        (progress.cycle_initial_residual / progress.final_residual).ln()
            / progress.cycle_iterations as f64
    } else {
        f64::NAN
    };
    println!(
        "C1H15C1_BOUNDARY|cycle={cycle}|iterations={}|cycle_iterations={}|restart={}|next_restart={next_restart}|remaining={}|true_residual_norm={:.12e}|true_relative={relative:.12e}|cycle_log_decay_per_iter={decay_rate:.12e}",
        progress.total_iterations,
        progress.cycle_iterations,
        progress.restart,
        progress.remaining_iterations,
        progress.final_residual,
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let load_start = Instant::now();
    let (matrix, info) = read_matrix_market(&args.matrix)?;
    if matrix.nrows() == 0 || matrix.nrows() != matrix.ncols() {
        return Err("complete nonempty square matrix required".into());
    }
    let b = read_rhs(&args.rhs, matrix.nrows())?;
    let rhs_norm = norm(b.iter().copied());
    if !rhs_norm.is_finite() {
        return Err("non-finite RHS norm".into());
    }
    println!(
        "C1H15C1_INPUT|rows={}|input_nnz={}|csr_nnz={}|rhs_norm={rhs_norm:.12e}|load_ms={:.6}",
        matrix.nrows(),
        info.input_entries,
        info.csr_nnz,
        load_start.elapsed().as_secs_f64() * 1e3,
    );

    let setup_start = Instant::now();
    let mut factor = Ilu0Preconditioner::from_csr32_general_abtm(&matrix)?;
    let factor_bytes = factor.factor_bytes();
    let setup_ms = setup_start.elapsed().as_secs_f64() * 1e3;
    let restart_capacity = match args.policy {
        Policy::Fixed => args.restart,
        Policy::BudgetAwareMirror => args.max_restart,
    };
    let mut workspace = FgmresWorkspace::new(matrix.nrows(), restart_capacity)?;
    let workspace_bytes = workspace.bytes();
    let target = args.tol * rhs_norm.max(f64::MIN_POSITIVE);
    let mut mirror = BudgetAwareMirror::new(target, args.max_restart);
    let mut x = vec![0.0; matrix.ncols()];
    let mut boundary_count = 0usize;
    let mut growth_count = 0usize;
    let solve_start = Instant::now();
    let outcome = fgmres_with_workspace_and_restart_controller(
        &matrix,
        &mut factor,
        &b,
        &mut x,
        FgmresOptions {
            solver: SolverOptions {
                relative_tolerance: args.tol,
                absolute_tolerance: 0.0,
                max_iterations: args.max_iters,
            },
            restart: args.restart,
        },
        &mut workspace,
        |progress| {
            boundary_count += 1;
            let next_restart = match args.policy {
                Policy::Fixed => progress.restart,
                Policy::BudgetAwareMirror => mirror.next_restart(progress),
            };
            if next_restart > progress.restart {
                growth_count += 1;
            }
            trace_progress(boundary_count, progress, next_restart, rhs_norm);
            Ok(next_restart)
        },
    )?;
    let solver_ms = solve_start.elapsed().as_secs_f64() * 1e3;
    let verified_relative = independent_relative(&matrix, &b, &x)?;
    if !verified_relative.is_finite() {
        return Err("non-finite independently verified relative residual".into());
    }
    if outcome.status == SolveStatus::Converged && verified_relative > args.tol * (1.0 + 1.0e-8) {
        return Err("reported convergence rejected by true-residual gate".into());
    }
    println!(
        "C1H15C1_RESULT|policy={}|status={:?}|iterations={}|verified_relative={verified_relative:.12e}|reported_final_norm={:.12e}|boundaries={boundary_count}|growths={growth_count}|setup_ms={setup_ms:.6}|solver_ms={solver_ms:.6}|factor_bytes={factor_bytes}|workspace_bytes={workspace_bytes}|restart={}|max_restart={}|max_iters={}|tol={:.12e}",
        args.policy.name(),
        outcome.status,
        outcome.iterations,
        outcome.final_residual,
        args.restart,
        args.max_restart,
        args.max_iters,
        args.tol,
    );
    // MaxIterations is an evidence result, not an error exit. Supervisors
    // must inspect this status and the independent verified residual.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hybit::Preconditioner;

    struct Identity(usize);

    impl Preconditioner for Identity {
        fn len(&self) -> usize {
            self.0
        }

        fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
            z.copy_from_slice(r);
            Ok(())
        }
    }

    #[test]
    fn budget_mirror_escalates_only_after_two_pressure_windows() {
        let mut controller = BudgetAwareMirror::new(1.0e-12, 70);
        for cycle in 1..=8 {
            let progress = FgmresRestartProgress {
                restart: 30,
                cycle_iterations: 30,
                total_iterations: cycle * 30,
                initial_residual: 1.0,
                cycle_initial_residual: 1.0,
                final_residual: 0.99,
                remaining_iterations: 360 - cycle * 30,
            };
            let next = controller.next_restart(progress);
            assert_eq!(next, if cycle == 8 { 60 } else { 30 });
        }
    }

    #[test]
    fn fixed_trace_reports_true_restart_boundaries() {
        let a = Csr32Matrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
        let mut precond = Identity(2);
        let mut x = [0.0; 2];
        let mut workspace = FgmresWorkspace::new(2, 1).unwrap();
        let mut boundaries = Vec::new();
        let outcome = fgmres_with_workspace_and_restart_controller(
            &a,
            &mut precond,
            &[1.0, 1.0],
            &mut x,
            FgmresOptions {
                solver: SolverOptions {
                    relative_tolerance: 1.0e-14,
                    absolute_tolerance: 0.0,
                    max_iterations: 5,
                },
                restart: 1,
            },
            &mut workspace,
            |progress| {
                boundaries.push(progress);
                Ok(1)
            },
        )
        .unwrap();
        assert_eq!(outcome.iterations, 5);
        assert_eq!(boundaries.len(), 4);
        assert!(boundaries.iter().all(|p| p.restart == 1));
        assert!(boundaries.iter().all(|p| p.cycle_iterations == 1));
        assert!(
            boundaries
                .windows(2)
                .all(|pair| pair[1].final_residual < pair[0].final_residual)
        );
    }

    #[test]
    fn independent_residual_is_zero_for_exact_solution() {
        let a = Csr32Matrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 3.0]).unwrap();
        assert_eq!(
            independent_relative(&a, &[2.0, 6.0], &[1.0, 2.0]).unwrap(),
            0.0
        );
    }
}
