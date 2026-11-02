//! C1-H10: measured multi-RHS cost of plain PCG, fresh Auto and prepared Auto.
//! Synthetic SPD corpus only; no real-FEM speedup or generalized Auto claim.
use std::error::Error;
use std::time::Instant;

use hybit::{
    BackendPolicy, Csr32Matrix, HybitSolver, JacobiPreconditioner, SolveStatus, SolverOptions, pcg,
};

const N: usize = 160;
const LOCAL: usize = 96;
const TOL: f64 = 1.0e-10;
const MAX_ITERS: usize = 30;

type AnyResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone)]
struct Case {
    rhs: Vec<f64>,
    reference: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Arm {
    Plain,
    Fresh,
    Prepared,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Fresh => "fresh",
            Self::Prepared => "prepared",
        }
    }
}

#[derive(Clone)]
struct Observation {
    status: SolveStatus,
    iterations: usize,
    residual: f64,
    error: f64,
    x: Vec<f64>,
    promoted: bool,
    reused: bool,
}

#[derive(Default)]
struct Counters {
    wall_ms: f64,
    analysis_ms: f64,
    prepare_ms: f64,
    diag_ms: f64,
    factor_ms: f64,
    probe_ms: f64,
    solve_ms: f64,
    iterations: usize,
    converged: usize,
    promotions: usize,
    reuse_events: usize,
    peak_factor_bytes: usize,
    max_true_residual: f64,
    max_forward_error: f64,
}

fn norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |acc, &x| acc.hypot(x))
}

fn true_relative(matrix: &Csr32Matrix, rhs: &[f64], x: &[f64]) -> AnyResult<f64> {
    let ax = matrix.spmv(x)?;
    let diff = rhs.iter().zip(ax).map(|(&b, a)| b - a).collect::<Vec<_>>();
    Ok(norm(&diff) / norm(rhs).max(f64::MIN_POSITIVE))
}

fn forward_error(x: &[f64], reference: &[f64]) -> f64 {
    let diff = x
        .iter()
        .zip(reference)
        .map(|(&a, &b)| a - b)
        .collect::<Vec<_>>();
    norm(&diff) / norm(reference).max(f64::MIN_POSITIVE)
}

// One Dirichlet 96-DOF chain, diagonal exterior and a positive weak bridge.
// A is SPD: the original Dirichlet chain is SPD, the exterior diagonal is
// positive, and the weak bridge adds a PSD edge Laplacian.
fn matrix() -> AnyResult<Csr32Matrix> {
    let mut rows = vec![Vec::<(usize, f64)>::new(); N];
    let mut diag = [2.0_f64; N];
    for i in 0..LOCAL - 1 {
        rows[i].push((i + 1, -1.0));
        rows[i + 1].push((i, -1.0));
    }
    diag[LOCAL - 1] += 1.0e-4;
    diag[LOCAL] += 1.0e-4;
    rows[LOCAL - 1].push((LOCAL, -1.0e-4));
    rows[LOCAL].push((LOCAL - 1, -1.0e-4));
    let mut ptr = vec![0_u32];
    let mut col = Vec::new();
    let mut vals = Vec::new();
    for (i, row) in rows.iter_mut().enumerate() {
        row.push((i, diag[i]));
        row.sort_unstable_by_key(|&(j, _)| j);
        for &(j, value) in row.iter() {
            col.push(u32::try_from(j)?);
            vals.push(value);
        }
        ptr.push(u32::try_from(col.len())?);
    }
    Csr32Matrix::new(N, N, ptr, col, vals).map_err(Into::into)
}

fn reference(k: usize) -> Vec<f64> {
    if k == 1 {
        return reference(0).iter().map(|&v| -2.0 * v).collect();
    }
    let f = k as f64;
    (0..N)
        .map(|i| {
            if k % 8 == 7 {
                // Easy exterior-only load; the cached factor must not be used.
                return if i == 140 { 1.0 } else { 0.0 };
            }
            if i >= LOCAL {
                0.7 + 0.2 * (0.13 * i as f64 + 0.07 * f).sin()
            } else {
                let t = std::f64::consts::PI * (i + 1) as f64 / (LOCAL + 1) as f64;
                match k % 8 {
                    0 => t.sin(),        // exact C1-H9 canonical hard RHS
                    1 => -2.0 * t.sin(), // cached-factor positive control
                    2 => t.sin() + 0.12 * (2.0 * t).sin(),
                    3 => t.sin() - 0.09 * (3.0 * t).sin(),
                    4 => 0.8 * t.sin() + 0.14 * (4.0 * t).sin(),
                    5 => 0.6 * t.sin() + 0.07 * (5.0 * t).sin(),
                    _ => 1.25 * t.sin() - 0.03 * (6.0 * t).sin(),
                }
            }
        })
        .collect()
}

fn cases(a: &Csr32Matrix, count: usize) -> AnyResult<Vec<Case>> {
    (0..count)
        .map(|k| {
            let reference = reference(k);
            let rhs = a.spmv(&reference)?;
            Ok(Case { rhs, reference })
        })
        .collect()
}

fn solver() -> AnyResult<HybitSolver> {
    let mut solver = HybitSolver::new();
    solver.set_options(SolverOptions {
        relative_tolerance: TOL,
        absolute_tolerance: 0.0,
        max_iterations: MAX_ITERS,
    })?;
    solver.set_backend_policy(BackendPolicy::Csr32);
    let mut hybrid = solver.hybrid_options();
    hybrid.abtm_weighted_routing_enabled = true;
    hybrid.probe_iterations = 12;
    hybrid.max_local_region_size = 128;
    hybrid.max_local_regions = 1;
    hybrid.max_escalations = 1;
    hybrid.overlap_layers = 0;
    hybrid.max_local_factor_bytes = 1024 * 1024;
    solver.set_hybrid_options(hybrid)?;
    Ok(solver)
}

fn run_arm(a: &Csr32Matrix, cases: &[Case], arm: Arm) -> AnyResult<(Counters, Vec<Observation>)> {
    let mut counts = Counters::default();
    let mut results = Vec::with_capacity(cases.len());
    let start = Instant::now();
    let opts = SolverOptions {
        relative_tolerance: TOL,
        absolute_tolerance: 0.0,
        max_iterations: MAX_ITERS,
    };
    let jacobi = if arm == Arm::Plain {
        Some(JacobiPreconditioner::from_csr32(a)?)
    } else {
        None
    };
    // Prepared setup is timed once per RHS batch. Fresh Auto performs
    // analyze+prepare independently on each RHS inside solve_csr32.
    let mut prepared = if arm == Arm::Prepared {
        Some(solver()?.prepare(a)?)
    } else {
        None
    };
    for case in cases {
        let mut x = vec![0.0; N];
        let mut promoted = false;
        let mut reused = false;
        let (status, iterations) = if let Some(jacobi) = jacobi.as_ref() {
            let out = pcg(a, jacobi, &case.rhs, &mut x, opts)?;
            (out.status, out.iterations)
        } else {
            let report = if let Some(prepared) = prepared.as_mut() {
                prepared.solve(a, &case.rhs, &mut x)?
            } else {
                solver()?.solve_csr32(a, &case.rhs, &mut x)?
            };
            promoted = report.escalations > 0;
            reused = report.preconditioner_reused;
            counts.analysis_ms += 1000.0 * report.analysis_seconds;
            counts.prepare_ms += 1000.0 * report.prepare_seconds;
            counts.diag_ms += 1000.0 * report.diagnostics_seconds;
            counts.factor_ms += 1000.0 * report.local_factor_seconds;
            counts.probe_ms += 1000.0 * report.probe_seconds;
            counts.solve_ms += 1000.0 * report.solve_seconds;
            counts.peak_factor_bytes = counts.peak_factor_bytes.max(report.local_factor_bytes);
            (report.status, report.iterations)
        };
        // Independent CSR check for every mode, not inferred from Krylov's
        // recursive residual or from the output of another mode.
        let residual = true_relative(a, &case.rhs, &x)?;
        let error = forward_error(&x, &case.reference);
        if !residual.is_finite() || !error.is_finite() {
            return Err("nonfinite true residual or forward error".into());
        }
        if status == SolveStatus::Converged && (residual > TOL * 1.0001 || error > 1.0e-6) {
            return Err(format!(
                "false convergence: mode={} residual={residual:e} error={error:e}",
                arm.name()
            )
            .into());
        }
        counts.iterations += iterations;
        counts.converged += usize::from(status == SolveStatus::Converged);
        counts.promotions += usize::from(promoted);
        counts.reuse_events += usize::from(reused);
        counts.max_true_residual = counts.max_true_residual.max(residual);
        counts.max_forward_error = counts.max_forward_error.max(error);
        results.push(Observation {
            status,
            iterations,
            residual,
            error,
            x,
            promoted,
            reused,
        });
    }
    counts.wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok((counts, results))
}

fn certify_comparison(
    rhs_count: usize,
    plain: &[Observation],
    fresh: &[Observation],
    prepared: &[Observation],
) -> AnyResult<()> {
    if plain.len() != rhs_count || fresh.len() != rhs_count || prepared.len() != rhs_count {
        return Err("RHS count mismatch".into());
    }
    for (i, (f, p)) in fresh.iter().zip(prepared).enumerate() {
        if f.status != p.status || f.iterations != p.iterations || f.promoted != p.promoted {
            return Err(format!("fresh/prepared routing mismatch on RHS {i}").into());
        }
        let diff =
            f.x.iter()
                .zip(&p.x)
                .map(|(&a, &b)| a - b)
                .collect::<Vec<_>>();
        if norm(&diff) > 1.0e-10 * norm(&f.x).max(1.0) {
            return Err(format!("fresh/prepared solution mismatch on RHS {i}").into());
        }
        if (f.residual - p.residual).abs() > 1.0e-10 || (f.error - p.error).abs() > 1.0e-10 {
            return Err(format!("fresh/prepared error metric mismatch on RHS {i}").into());
        }
        if !f.promoted && p.reused {
            return Err(format!("unexpected local factor use for unpromoted RHS {i}").into());
        }
    }
    // Canonical control from C1-H9: Jacobi fails within 30, Auto rescues.
    if plain[0].status != SolveStatus::MaxIterations
        || fresh[0].status != SolveStatus::Converged
        || !fresh[0].promoted
        || prepared[0].status != SolveStatus::Converged
    {
        return Err("C1-H9 canonical positive control regressed".into());
    }
    if rhs_count >= 2 && (!prepared[1].reused || !prepared[1].promoted) {
        return Err("C1-H9 scaled RHS factor reuse regressed".into());
    }
    Ok(())
}

fn parse_args() -> AnyResult<(Vec<usize>, usize, usize)> {
    let mut counts = vec![1, 2, 4, 8, 16, 32];
    let mut repeats = 11usize;
    let mut warmups = 2usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args.next().ok_or("expected argument value")?;
        match arg.as_str() {
            "--counts" => {
                counts = value
                    .split(',')
                    .map(str::parse::<usize>)
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--repeats" => repeats = value.parse()?,
            "--warmups" => warmups = value.parse()?,
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    if counts.is_empty()
        || counts.iter().any(|&c| c == 0 || c > 64)
        || counts.windows(2).any(|w| w[0] >= w[1])
        || !(3..=101).contains(&repeats)
        || warmups > 10
    {
        return Err("invalid counts, repeats, or warmups".into());
    }
    Ok((counts, repeats, warmups))
}

fn main() -> AnyResult<()> {
    let (counts, repeats, warmups) = parse_args()?;
    let a = matrix()?;
    if a.nrows() != N || a.nnz() != 352 {
        return Err("unexpected benchmark matrix".into());
    }
    println!(
        "mode,rhs_count,repeat,warmup,wall_ms,analysis_ms,prepare_ms,diagnostics_ms,factor_ms,probe_ms,solve_ms,iterations,converged,promotions,reuse_events,peak_factor_bytes,max_true_residual,max_forward_error"
    );
    for count in counts {
        let load_cases = cases(&a, count)?;
        for repeat in 0..(repeats + warmups) {
            let mut measured: [Option<(Counters, Vec<Observation>)>; 3] = [None, None, None];
            // Rotate execution order to reduce systematic per-arm thermal bias.
            for offset in 0..3 {
                let index = (repeat + offset) % 3;
                let arm = [Arm::Plain, Arm::Fresh, Arm::Prepared][index];
                measured[index] = Some(run_arm(&a, &load_cases, arm)?);
            }
            let plain = measured[0].as_ref().ok_or("missing plain")?;
            let fresh = measured[1].as_ref().ok_or("missing fresh")?;
            let prepared = measured[2].as_ref().ok_or("missing prepared")?;
            certify_comparison(count, &plain.1, &fresh.1, &prepared.1)?;
            for (index, arm) in [Arm::Plain, Arm::Fresh, Arm::Prepared].iter().enumerate() {
                let (v, _) = measured[index].as_ref().ok_or("missing arm")?;
                println!(
                    "{},{count},{repeat},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{},{},{},{:.12e},{:.12e}",
                    arm.name(),
                    usize::from(repeat < warmups),
                    v.wall_ms,
                    v.analysis_ms,
                    v.prepare_ms,
                    v.diag_ms,
                    v.factor_ms,
                    v.probe_ms,
                    v.solve_ms,
                    v.iterations,
                    v.converged,
                    v.promotions,
                    v.reuse_events,
                    v.peak_factor_bytes,
                    v.max_true_residual,
                    v.max_forward_error
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c1h10_spd_matrix_has_expected_symmetric_weak_cut() {
        let a = matrix().unwrap();
        assert_eq!(a.nrows(), N);
        assert_eq!(a.nnz(), 352);
        for i in 0..N {
            for p in a.row_ptr()[i] as usize..a.row_ptr()[i + 1] as usize {
                let j = a.col_idx()[p] as usize;
                let start = a.row_ptr()[j] as usize;
                let end = a.row_ptr()[j + 1] as usize;
                let q = a.col_idx()[start..end].binary_search(&(i as u32)).unwrap();
                assert_eq!(a.values()[p].to_bits(), a.values()[start + q].to_bits());
            }
        }
    }

    #[test]
    fn c1h10_second_rhs_scales_first_and_easy_control_is_exterior() {
        let a = matrix().unwrap();
        let inputs = cases(&a, 8).unwrap();
        for (&x, &y) in inputs[0].rhs.iter().zip(&inputs[1].rhs) {
            assert!((y + 2.0 * x).abs() <= 5.0e-14);
        }
        assert_eq!(inputs[7].reference.iter().filter(|&&v| v != 0.0).count(), 1);
        assert_eq!(inputs[7].reference[140], 1.0);
    }
}
