use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use hybit::{
    Csr32Matrix, JacobiPreconditioner, PcgWorkspace, Preconditioner, SolveStatus, SolverOptions,
    TwoLevelAggregation, TwoLevelBasis, TwoLevelBlockJacobiPreconditioner,
    TwoLevelCoarseApplyPolicy, TwoLevelTransferApplyPolicy, TwoLevelTransferOptions,
    TwoLevelTransferStoragePolicy, TwoLevelTransferValueStoragePolicy, pcg, read_matrix_market,
};
use hybit_krylov::{pcg_continue_with_workspace, pcg_start_with_workspace};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    JacobiOnly,
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
    aggregation: TwoLevelAggregation,
    basis: TwoLevelBasis,
    transfer_value_storage: TwoLevelTransferValueStoragePolicy,
    report_c1h12a4: bool,
    certified_restart: bool,
    cert_periodic_check: Option<usize>,
    cert_gap_ratio: f64,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut dofs_per_node = 3usize;
        let mut target_coarse_dimension = 1536usize;
        let mut relative_tolerance = 1.0e-8_f64;
        let mut max_iterations = 3000usize;
        let mut probe_iterations = 12usize;
        let mut mode = Mode::DirectCoarse;
        let mut aggregation = TwoLevelAggregation::Graph;
        let mut basis = TwoLevelBasis::JacobiSmoothed;
        let mut transfer_value_storage = TwoLevelTransferValueStoragePolicy::F32;
        let mut report_c1h12a4 = false;
        let mut certified_restart = false;
        let mut cert_periodic_check = None;
        let mut cert_gap_ratio = 2.0_f64;

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
                        "jacobi" | "jacobi-only" => Mode::JacobiOnly,
                        "direct" | "coarse-only" => Mode::DirectCoarse,
                        "probe" | "probe-coarse" => Mode::ProbeThenCoarse,
                        other => {
                            return Err(format!(
                                "unknown mode '{other}'; use jacobi|direct|probe-coarse"
                            )
                            .into());
                        }
                    }
                }
                "--coarse-aggregation" => {
                    aggregation = match next_value(&mut it, "--coarse-aggregation")?.as_str() {
                        "contiguous" => TwoLevelAggregation::Contiguous,
                        "graph" => TwoLevelAggregation::Graph,
                        "strong-graph" => TwoLevelAggregation::StrongGraph,
                        other => return Err(format!("unknown aggregation '{other}'").into()),
                    }
                }
                "--coarse-basis" => {
                    basis = match next_value(&mut it, "--coarse-basis")?.as_str() {
                        "piecewise" => TwoLevelBasis::PiecewiseConstant,
                        "smoothed" => TwoLevelBasis::JacobiSmoothed,
                        other => return Err(format!("unknown basis '{other}'").into()),
                    }
                }
                "--coarse-values" => {
                    transfer_value_storage = match next_value(&mut it, "--coarse-values")?.as_str()
                    {
                        "f64" => TwoLevelTransferValueStoragePolicy::F64,
                        "f32" => TwoLevelTransferValueStoragePolicy::F32,
                        other => return Err(format!("unknown transfer values '{other}'").into()),
                    }
                }
                "--report-c1h12a4" => report_c1h12a4 = true,
                "--certified-restart" => certified_restart = true,
                "--cert-periodic-check" => {
                    cert_periodic_check =
                        Some(next_value(&mut it, "--cert-periodic-check")?.parse()?);
                }
                "--cert-gap-ratio" => {
                    cert_gap_ratio = next_value(&mut it, "--cert-gap-ratio")?.parse()?;
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
        if !relative_tolerance.is_finite() || relative_tolerance <= 0.0 {
            return Err("--tol must be finite and positive".into());
        }
        if certified_restart && mode == Mode::ProbeThenCoarse {
            return Err("--certified-restart currently supports only jacobi/direct modes".into());
        }
        if cert_periodic_check.is_some_and(|n| n == 0 || n > max_iterations) {
            return Err("--cert-periodic-check must be in 1..=max-iters".into());
        }
        if cert_periodic_check.is_some() && !certified_restart {
            return Err("--cert-periodic-check requires --certified-restart".into());
        }
        if !cert_gap_ratio.is_finite() || cert_gap_ratio <= 1.0 {
            return Err("--cert-gap-ratio must be finite and > 1".into());
        }

        Ok(Self {
            matrix,
            dofs_per_node,
            target_coarse_dimension,
            relative_tolerance,
            max_iterations,
            probe_iterations,
            mode,
            aggregation,
            basis,
            transfer_value_storage,
            report_c1h12a4,
            certified_restart,
            cert_periodic_check,
            cert_gap_ratio,
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
    println!("  --mode jacobi|direct|probe-coarse (default: direct)");
    println!("  --coarse-aggregation contiguous|graph|strong-graph (default: graph)");
    println!("  --coarse-basis piecewise|smoothed (default: smoothed)");
    println!("  --coarse-values f64|f32 (default: f32)");
    println!("  --report-c1h12a4    emit one machine-readable diagnostic row");
    println!("  --certified-restart opt-in true-residual validation and PCG restart");
    println!("  --cert-periodic-check N  A6 periodic true-residual sampling every N iterations");
    println!(
        "  --cert-gap-ratio X      A6 restart when true/recursive residual gap exceeds X (default: 2)"
    );
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

#[derive(Debug, Clone, Copy)]
struct CertifiedPcgOutcome {
    status: SolveStatus,
    iterations: usize,
    reported_final_residual: f64,
    true_checks: usize,
    restarts: usize,
    rejected_recursive_convergences: usize,
    final_true_relative: f64,
}

/// Experimental wrapper: never treat a recursively converged PCG segment as
/// solved until the original CSR certifies b - A*x.  If the certificate fails,
/// restart the unchanged PCG/preconditioner at the current x and spend only
/// the remaining global iteration budget. Production PCG remains untouched.
fn pcg_certified_restart(
    matrix: &Csr32Matrix,
    preconditioner: &dyn Preconditioner,
    b: &[f64],
    x: &mut [f64],
    options: SolverOptions,
) -> Result<CertifiedPcgOutcome, Box<dyn Error>> {
    let mut iterations = 0;
    let mut true_checks = 0;
    let mut restarts = 0;
    let mut rejected_recursive_convergences = 0;
    let mut reported_final_residual = norm2(b);
    let threshold = options.relative_tolerance;

    loop {
        let remaining = options.max_iterations - iterations;
        if remaining == 0 {
            let final_true_relative = verified_relative_residual(matrix, b, x)?;
            return Ok(CertifiedPcgOutcome {
                status: SolveStatus::MaxIterations,
                iterations,
                reported_final_residual,
                true_checks,
                restarts,
                rejected_recursive_convergences,
                final_true_relative,
            });
        }

        let mut segment_options = options;
        segment_options.max_iterations = remaining;
        let out = pcg(matrix, preconditioner, b, x, segment_options)?;
        iterations += out.iterations;
        reported_final_residual = out.final_residual;
        let final_true_relative = verified_relative_residual(matrix, b, x)?;
        true_checks += 1;
        if !final_true_relative.is_finite() {
            return Err("nonfinite certified true residual".into());
        }
        if out.status == SolveStatus::Converged && final_true_relative <= threshold {
            return Ok(CertifiedPcgOutcome {
                status: SolveStatus::Converged,
                iterations,
                reported_final_residual,
                true_checks,
                restarts,
                rejected_recursive_convergences,
                final_true_relative,
            });
        }
        if out.status == SolveStatus::Converged {
            rejected_recursive_convergences += 1;
            if iterations < options.max_iterations && out.iterations > 0 {
                restarts += 1;
                continue;
            }
            // Either the budget is exhausted or a zero-step recurrence
            // repeatedly claims convergence contrary to original CSR.
            return Ok(CertifiedPcgOutcome {
                status: if iterations == options.max_iterations {
                    SolveStatus::MaxIterations
                } else {
                    SolveStatus::Breakdown
                },
                iterations,
                reported_final_residual,
                true_checks,
                restarts,
                rejected_recursive_convergences,
                final_true_relative,
            });
        }
        return Ok(CertifiedPcgOutcome {
            status: out.status,
            iterations,
            reported_final_residual,
            true_checks,
            restarts,
            rejected_recursive_convergences,
            final_true_relative,
        });
    }
}

/// A6 experimental periodic certificate. Advance one *resumable* PCG
/// recurrence between checks; rebuild it only when an independently measured
/// residual gap is excessive or a recursive convergence claim is false.
/// The preconditioner and original CSR never change during the solve.
#[derive(Debug, Clone, Copy)]
struct PeriodicCertifiedOutcome {
    base: CertifiedPcgOutcome,
    periodic_checks: usize,
    proactive_restarts: usize,
    peak_gap_ratio: f64,
    check_ms: f64,
}

fn pcg_certified_periodic(
    matrix: &Csr32Matrix,
    preconditioner: &dyn Preconditioner,
    b: &[f64],
    x: &mut [f64],
    options: SolverOptions,
    check_every: usize,
    trigger_ratio: f64,
) -> Result<PeriodicCertifiedOutcome, Box<dyn Error>> {
    if check_every == 0 || !trigger_ratio.is_finite() || trigger_ratio <= 1.0 {
        return Err("invalid periodic certification policy".into());
    }
    let bnorm = norm2(b).max(f64::MIN_POSITIVE);
    let mut workspace = PcgWorkspace::new(matrix.nrows());
    let mut session =
        pcg_start_with_workspace(matrix, preconditioner, b, x, options, &mut workspace)?;
    let mut iterations = 0usize;
    let mut true_checks = 0usize;
    let mut periodic_checks = 0usize;
    let mut restarts = 0usize;
    let mut proactive_restarts = 0usize;
    let mut rejected_recursive_convergences = 0usize;
    let mut peak_gap_ratio = 0.0_f64;
    let mut check_ms = 0.0_f64;

    loop {
        let remaining = options.max_iterations - iterations;
        let step = remaining.min(check_every);
        let out = pcg_continue_with_workspace(
            matrix,
            preconditioner,
            b,
            x,
            step,
            &mut session,
            &mut workspace,
        )?;
        iterations += out.iterations;
        let reported_final_residual = out.final_residual;

        let verify_start = Instant::now();
        let true_relative = verified_relative_residual(matrix, b, x)?;
        check_ms += verify_start.elapsed().as_secs_f64() * 1.0e3;
        true_checks += 1;
        periodic_checks += 1;
        if !true_relative.is_finite() {
            return Err("nonfinite A6 true residual".into());
        }

        let recursive_relative = out.final_residual / bnorm;
        let raw_gap = true_relative / recursive_relative.max(f64::MIN_POSITIVE);
        let gap_ratio = if raw_gap.is_finite() {
            raw_gap
        } else {
            f64::MAX
        };
        peak_gap_ratio = peak_gap_ratio.max(gap_ratio);

        let certified =
            out.status == SolveStatus::Converged && true_relative <= options.relative_tolerance;
        let terminate =
            certified || out.status == SolveStatus::Breakdown || remaining == out.iterations;
        if terminate {
            let status = if certified {
                SolveStatus::Converged
            } else if out.status == SolveStatus::Breakdown {
                SolveStatus::Breakdown
            } else {
                SolveStatus::MaxIterations
            };
            return Ok(PeriodicCertifiedOutcome {
                base: CertifiedPcgOutcome {
                    status,
                    iterations,
                    reported_final_residual,
                    true_checks,
                    restarts,
                    rejected_recursive_convergences,
                    final_true_relative: true_relative,
                },
                periodic_checks,
                proactive_restarts,
                peak_gap_ratio,
                check_ms,
            });
        }

        let false_recursive = out.status == SolveStatus::Converged;
        let excessive_gap =
            gap_ratio >= trigger_ratio && true_relative > options.relative_tolerance;
        if false_recursive || excessive_gap {
            // Starting PCG anew recomputes r=b-A*x and resets p, z and r^T M^-1 r.
            // Merely overwriting the recurrence residual would break conjugacy.
            if out.iterations == 0 {
                return Ok(PeriodicCertifiedOutcome {
                    base: CertifiedPcgOutcome {
                        status: SolveStatus::Breakdown,
                        iterations,
                        reported_final_residual,
                        true_checks,
                        restarts,
                        rejected_recursive_convergences,
                        final_true_relative: true_relative,
                    },
                    periodic_checks,
                    proactive_restarts,
                    peak_gap_ratio,
                    check_ms,
                });
            }
            if false_recursive {
                rejected_recursive_convergences += 1;
            } else {
                proactive_restarts += 1;
            }
            restarts += 1;
            session =
                pcg_start_with_workspace(matrix, preconditioner, b, x, options, &mut workspace)?;
        }
    }
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
    println!("coarse aggregation : {:?}", args.aggregation);
    println!("coarse basis       : {:?}", args.basis);
    println!(
        "coarse transfer    : Parallel / Wide / {:?}",
        args.transfer_value_storage
    );
    println!("coarse apply req.  : Auto");

    let setup_start = Instant::now();
    let coarse = if args.mode == Mode::JacobiOnly {
        None
    } else {
        Some(
        TwoLevelBlockJacobiPreconditioner::from_csr32_with_aggregation_basis_and_transfer_options(
            &matrix,
            args.dofs_per_node,
            aggregate_nodes,
            args.aggregation,
            args.basis,
            TwoLevelCoarseApplyPolicy::Auto,
            TwoLevelTransferOptions {
                apply_policy: TwoLevelTransferApplyPolicy::Parallel,
                storage_policy: TwoLevelTransferStoragePolicy::Wide,
                value_storage_policy: args.transfer_value_storage,
            },
        )?)
    };
    let coarse_setup_ms = setup_start.elapsed().as_secs_f64() * 1.0e3;
    if let Some(coarse) = &coarse {
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
    }

    let jacobi_setup_start = Instant::now();
    let jacobi_only = if args.mode == Mode::JacobiOnly {
        Some(JacobiPreconditioner::from_csr32(&matrix)?)
    } else {
        None
    };
    let jacobi_setup_ms = jacobi_setup_start.elapsed().as_secs_f64() * 1.0e3;

    let options = SolverOptions {
        relative_tolerance: args.relative_tolerance,
        absolute_tolerance: 0.0,
        max_iterations: args.max_iterations,
    };
    let mut x = vec![0.0; matrix.ncols()];

    let solve_start = Instant::now();
    let mut certified = None;
    let mut periodic = None;
    let (status, total_iterations, reported_final_residual, probe_ms) = match args.mode {
        Mode::JacobiOnly => {
            let jacobi = jacobi_only
                .as_ref()
                .ok_or("missing Jacobi preconditioner")?;
            if args.certified_restart {
                if let Some(every) = args.cert_periodic_check {
                    let out = pcg_certified_periodic(
                        &matrix,
                        jacobi,
                        &b,
                        &mut x,
                        options,
                        every,
                        args.cert_gap_ratio,
                    )?;
                    periodic = Some(out);
                    (
                        out.base.status,
                        out.base.iterations,
                        out.base.reported_final_residual,
                        0.0,
                    )
                } else {
                    let out = pcg_certified_restart(&matrix, jacobi, &b, &mut x, options)?;
                    certified = Some(out);
                    (out.status, out.iterations, out.reported_final_residual, 0.0)
                }
            } else {
                let out = pcg(&matrix, jacobi, &b, &mut x, options)?;
                (out.status, out.iterations, out.final_residual, 0.0)
            }
        }
        Mode::DirectCoarse => {
            let coarse = coarse.as_ref().ok_or("missing coarse preconditioner")?;
            if args.certified_restart {
                if let Some(every) = args.cert_periodic_check {
                    let out = pcg_certified_periodic(
                        &matrix,
                        coarse,
                        &b,
                        &mut x,
                        options,
                        every,
                        args.cert_gap_ratio,
                    )?;
                    periodic = Some(out);
                    (
                        out.base.status,
                        out.base.iterations,
                        out.base.reported_final_residual,
                        0.0,
                    )
                } else {
                    let out = pcg_certified_restart(&matrix, coarse, &b, &mut x, options)?;
                    certified = Some(out);
                    (out.status, out.iterations, out.reported_final_residual, 0.0)
                }
            } else {
                let out = pcg(&matrix, coarse, &b, &mut x, options)?;
                (out.status, out.iterations, out.final_residual, 0.0)
            }
        }
        Mode::ProbeThenCoarse => {
            let coarse = coarse.as_ref().ok_or("missing coarse preconditioner")?;
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
                let coarse_out = pcg(&matrix, coarse, &b, &mut x, coarse_options)?;
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
    if let Some(certified) = certified {
        println!(
            "C1H12A5|policy=CertifiedRestart|true_checks={}|restarts={}|rejected_recursive_convergences={}|final_true_relative={:.12e}|global_iterations={total_iterations}|status={:?}",
            certified.true_checks,
            certified.restarts,
            certified.rejected_recursive_convergences,
            certified.final_true_relative,
            certified.status,
        );
    }
    if let Some(periodic) = periodic {
        println!(
            "C1H12A6|policy=PeriodicCertifiedRestart|check_every={}|trigger_ratio={:.6}|true_checks={}|periodic_checks={}|restarts={}|proactive_restarts={}|rejected_recursive_convergences={}|peak_gap_ratio={:.12e}|check_ms={:.6}|final_true_relative={:.12e}|global_iterations={total_iterations}|status={:?}",
            args.cert_periodic_check.unwrap_or(0),
            args.cert_gap_ratio,
            periodic.base.true_checks,
            periodic.periodic_checks,
            periodic.base.restarts,
            periodic.proactive_restarts,
            periodic.base.rejected_recursive_convergences,
            periodic.peak_gap_ratio,
            periodic.check_ms,
            periodic.base.final_true_relative,
            periodic.base.status,
        );
    }

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
        load_ms + coarse_setup_ms + jacobi_setup_ms + solve_ms
    );
    let x_error = relative_error_to_ones(&x);
    println!("relative x error   : {:.6e}", x_error);
    if args.report_c1h12a4 {
        println!(
            "C1H12A4|mode={:?}|aggregation={:?}|basis={:?}|transfer_values={:?}|rows={}|nnz={}|coarse_dimension={}|coarse_factor_bytes={}|status={:?}|iterations={total_iterations}|tol={:.12e}|reported_relative={:.12e}|verified_true_relative={verified:.12e}|relative_x_error={x_error:.12e}|load_ms={load_ms:.6}|setup_ms={:.6}|solve_ms={solve_ms:.6}|total_wall_ms={:.6}",
            args.mode,
            args.aggregation,
            args.basis,
            args.transfer_value_storage,
            matrix.nrows(),
            matrix.nnz(),
            coarse.as_ref().map_or(0, |c| c.coarse_dimension()),
            coarse.as_ref().map_or(0, |c| c.factor_bytes()),
            status,
            args.relative_tolerance,
            reported_final_residual / bnorm,
            coarse_setup_ms + jacobi_setup_ms,
            load_ms + coarse_setup_ms + jacobi_setup_ms + solve_ms,
        );
    }

    Ok(())
}

#[cfg(test)]
mod c1h12a5_tests {
    use super::*;

    fn three_by_three_spd() -> Csr32Matrix {
        Csr32Matrix::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![2.1, -0.2, -0.2, 1.8, -0.3, -0.3, 2.6],
        )
        .unwrap()
    }

    #[test]
    fn certified_pcg_converges_on_small_spd() {
        let matrix = three_by_three_spd();
        let jacobi = JacobiPreconditioner::from_csr32(&matrix).unwrap();
        let b = matrix.spmv(&[1.0, 1.0, 1.0]).unwrap();
        let mut x = vec![0.0; 3];
        let out = pcg_certified_restart(
            &matrix,
            &jacobi,
            &b,
            &mut x,
            SolverOptions {
                relative_tolerance: 1e-10,
                absolute_tolerance: 0.0,
                max_iterations: 30,
            },
        )
        .unwrap();
        assert_eq!(out.status, SolveStatus::Converged);
        assert!(out.final_true_relative <= 1e-10);
        assert_eq!(out.true_checks, 1);
        assert_eq!(out.rejected_recursive_convergences, 0);
        assert_eq!(out.restarts, 0);
    }

    #[test]
    fn certified_pcg_never_claims_unverified_sub_machine_tolerance() {
        let matrix = three_by_three_spd();
        let jacobi = JacobiPreconditioner::from_csr32(&matrix).unwrap();
        let b = vec![0.7, -0.8, 1.3];
        let mut x = vec![0.0; 3];
        let out = pcg_certified_restart(
            &matrix,
            &jacobi,
            &b,
            &mut x,
            SolverOptions {
                relative_tolerance: 1e-22,
                absolute_tolerance: 0.0,
                max_iterations: 30,
            },
        )
        .unwrap();
        assert!(out.iterations <= 30);
        assert!(out.true_checks >= 1);
        assert!(out.final_true_relative.is_finite());
        if out.final_true_relative > 1e-22 {
            assert_ne!(out.status, SolveStatus::Converged);
        }
    }

    #[test]
    fn periodic_certificate_converges_on_small_spd() {
        let matrix = three_by_three_spd();
        let jacobi = JacobiPreconditioner::from_csr32(&matrix).unwrap();
        let b = matrix.spmv(&[1.0, 1.0, 1.0]).unwrap();
        let mut x = vec![0.0; 3];
        let out = pcg_certified_periodic(
            &matrix,
            &jacobi,
            &b,
            &mut x,
            SolverOptions {
                relative_tolerance: 1e-10,
                absolute_tolerance: 0.0,
                max_iterations: 30,
            },
            1,
            2.0,
        )
        .unwrap();
        assert_eq!(out.base.status, SolveStatus::Converged);
        assert!(out.base.final_true_relative <= 1e-10);
        assert!(out.base.iterations <= 30);
        assert!(out.periodic_checks > 0);
        assert_eq!(out.base.true_checks, out.periodic_checks);
        assert_eq!(out.base.restarts, 0);
        assert!(out.check_ms.is_finite());
    }

    #[test]
    fn periodic_certificate_rejects_sub_machine_false_convergence() {
        let matrix = three_by_three_spd();
        let jacobi = JacobiPreconditioner::from_csr32(&matrix).unwrap();
        let b = vec![0.7, -0.8, 1.3];
        let mut x = vec![0.0; 3];
        let out = pcg_certified_periodic(
            &matrix,
            &jacobi,
            &b,
            &mut x,
            SolverOptions {
                relative_tolerance: 1e-22,
                absolute_tolerance: 0.0,
                max_iterations: 30,
            },
            2,
            2.0,
        )
        .unwrap();
        assert!(out.base.iterations <= 30);
        assert!(out.base.true_checks >= 1);
        assert!(out.base.final_true_relative.is_finite());
        if out.base.final_true_relative > 1e-22 {
            assert_ne!(out.base.status, SolveStatus::Converged);
        }
    }

    #[test]
    fn periodic_certificate_rejects_bad_policy() {
        let matrix = three_by_three_spd();
        let jacobi = JacobiPreconditioner::from_csr32(&matrix).unwrap();
        let b = vec![1.0; 3];
        let options = SolverOptions {
            relative_tolerance: 1e-10,
            absolute_tolerance: 0.0,
            max_iterations: 30,
        };
        assert!(
            pcg_certified_periodic(&matrix, &jacobi, &b, &mut [0.0; 3], options, 0, 2.0).is_err()
        );
        assert!(
            pcg_certified_periodic(&matrix, &jacobi, &b, &mut [0.0; 3], options, 2, f64::NAN)
                .is_err()
        );
        assert!(
            pcg_certified_periodic(&matrix, &jacobi, &b, &mut [0.0; 3], options, 2, 1.0).is_err()
        );
    }
}
