//! G8-C12a: guarded rank-constant coarse MPI PCG experiment, NEVER a proof of global SPD.
//!
//! A manufactured right-hand side tests internal consistency but does not
//! certify physical correctness or positive-definiteness. Unverified matrices
//! require an explicit permission token and are not correctness-gated.
use hybit::distributed::mpi_backend::MpiRuntime;
use hybit::distributed::mpi_input::prepare_owned_rows;
use hybit::distributed::mpi_mtx::{distribute_matrix_market_with_policy, MtxPcgPolicy};
use hybit::distributed::mpi_pcg::DistributedAdaptiveBlockScore;
use hybit::distributed::mpi_pcg::{
    DistributedAdaptivePcgPolicy, DistributedAdaptiveSolveOptions, DistributedPcg,
    DistributedPcgOptions, DistributedPcgPreconditioner, DistributedPcgStatus,
};
use mpi::collective::SystemOperation;
use mpi::traits::*;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn max_f64(value: f64) -> f64 {
    let world = mpi::topology::SimpleCommunicator::world();
    let mut result = 0.0;
    world.all_reduce_into(&value, &mut result, SystemOperation::max());
    result
}
/// Write one rank-specific, run-specific CSV without concurrent writers.
fn write_adaptive_trace(
    csv: &Path,
    rank: i32,
    first_owned: u64,
    mode: DistributedAdaptiveBlockScore,
    trace: &[hybit::distributed::mpi_pcg::DistributedAdaptiveBlockDecision],
) -> std::io::Result<()> {
    let suffix = format!("rank{rank}.blocks.csv");
    let trace_path = csv.with_extension(suffix);
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(trace_path)?;
    writeln!(out, "rank,mode,block_index,global_start,size,dense_factor_bytes,residual_energy,internal_l1,boundary_l1,score,selected")?;
    let name = match mode {
        DistributedAdaptiveBlockScore::ResidualEnergy => "residual",
        DistributedAdaptiveBlockScore::BoundaryCoupling => "boundary",
        DistributedAdaptiveBlockScore::SwapOne { .. } => "swap",
        DistributedAdaptiveBlockScore::ReplayMask { .. } => "replay",
        DistributedAdaptiveBlockScore::ReplayMaskNoCoarse { .. } => "replaync",
        DistributedAdaptiveBlockScore::ReplayPooled { .. } => "replayg",
    };
    for d in trace {
        let start = first_owned + d.first_owned as u64;
        writeln!(
            out,
            "{rank},{name},{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{}",
            d.block_index,
            start,
            d.size,
            d.factor_bytes,
            d.residual_energy,
            d.internal_l1,
            d.boundary_l1,
            d.score,
            u8::from(d.selected)
        )?;
    }
    Ok(())
}

/// D6-B: exact manufactured solutions, not measured physical load patterns.
fn rhs_truth(global: u64, kind: &str) -> f64 {
    let t = global as f64 + 1.0;
    match kind {
        "smooth" => truth(global),
        "mixed" => truth(global) + 0.18 * (0.031 * t).sin() + 0.035 * (0.17 * t).cos(),
        "localized" => {
            let u = (t - 4500.0) / 175.0;
            let v = (t - 2200.0) / 70.0;
            0.05 * truth(global) + (-u * u).exp() + 0.25 * (-v * v).exp()
        }
        _ => unreachable!("validated RHS kind"),
    }
}

fn write_residual_trace(
    csv: &Path,
    history: &[hybit::distributed::mpi_pcg::DistributedPcgResidualSample],
) -> std::io::Result<()> {
    let path = csv.with_extension("residual.csv");
    let mut out = OpenOptions::new().write(true).create_new(true).open(path)?;
    writeln!(
        out,
        "iteration,phase,recursive_relative_residual,true_relative_residual"
    )?;
    for entry in history {
        let true_value = entry
            .true_relative_residual
            .map(|v| format!("{v:.17e}"))
            .unwrap_or_default();
        writeln!(
            out,
            "{},{},{:.17e},{}",
            entry.iteration, entry.phase, entry.recursive_relative_residual, true_value
        )?;
    }
    Ok(())
}

fn truth(global: u64) -> f64 {
    let t = global as f64 + 1.0;
    (0.0017 * t).sin() + 0.21 * (0.0061 * t).cos()
}
fn allowed_label(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 9
        && args.len() != 10
        && args.len() != 12
        && args.len() != 13
        && args.len() != 14
        && args.len() != 11
        && args.len() != 15
    {
        return Err("usage: mpi_fem_experiment <ranks> <csv> <mtx> <label> <certified|experimental> <jacobi|twolevel|block:32|hybrid:32|ic0:64> <max_iters> <max_nnz> <--certified-smoke|--permit-unverified-spd> [relative_tolerance] [adaptive_probe_iterations adaptive_poor_progress_ratio [factor_budget_bytes_global [residual|boundary|swap:R:FROM:TO]]]".into());
    }
    let expected: i32 = args[0].parse()?;
    let csv = Path::new(&args[1]);
    let input = Path::new(&args[2]);
    let label = &args[3];
    if !allowed_label(label) {
        return Err("label may contain only ASCII letters, digits, '_' or '-'".into());
    }
    let policy = match (args[4].as_str(), args[8].as_str()) {
        ("certified", "--certified-smoke") => MtxPcgPolicy::StrictCertified,
        ("experimental", "--permit-unverified-spd") => MtxPcgPolicy::ExperimentalUnverified,
        _ => return Err("explicit permission token does not match the input policy".into()),
    };
    let selection = &args[5];
    let mut adaptive_block_size = None;
    let choice = if selection == "jacobi" {
        DistributedPcgPreconditioner::Jacobi
    } else if selection == "twolevel" {
        DistributedPcgPreconditioner::RankConstantTwoLevelJacobi
    } else if let Some(size) = selection.strip_prefix("block:") {
        let block_size: usize = size.parse()?;
        if !(2..=512).contains(&block_size) {
            return Err("local Cholesky block size must be 2..=512".into());
        }
        DistributedPcgPreconditioner::LocalBlockCholesky { block_size }
    } else if let Some(size) = selection.strip_prefix("hybrid:") {
        let block_size: usize = size.parse()?;
        if !(2..=512).contains(&block_size) {
            return Err("hybrid local Cholesky block size must be 2..=512".into());
        }
        DistributedPcgPreconditioner::RankConstantTwoLevelBlockCholesky { block_size }
    } else if let Some(size) = selection.strip_prefix("adaptive:") {
        let block_size: usize = size.parse()?;
        if !(2..=512).contains(&block_size) {
            return Err("adaptive Cholesky block size must be 2..=512".into());
        }
        adaptive_block_size = Some(block_size);
        DistributedPcgPreconditioner::RankConstantTwoLevelJacobi
    } else if let Some(size) = selection.strip_prefix("ic0:") {
        let block_size: usize = size.parse()?;
        if !(2..=512).contains(&block_size) {
            return Err("IC(0) block size must be 2..=512".into());
        }
        DistributedPcgPreconditioner::LocalBlockIc0 { block_size }
    } else {
        return Err(
            "preconditioner must be jacobi, twolevel, block:N, hybrid:N, adaptive:N, or ic0:N"
                .into(),
        );
    };
    let max_iterations: usize = args[6].parse()?;
    let max_nnz: u64 = args[7].parse()?;
    if !(1..=100_000).contains(&max_iterations) || !(1..=20_000_000).contains(&max_nnz) {
        return Err("invalid iteration or expanded-NNZ cap".into());
    }
    // 9-argument legacy CLI retains exactly the G8-C12d-1 tolerance.
    // The optional 10th numeric argument is deliberately restricted to
    // validation experiments; it does not alter any production solver default.
    let requested_tolerance: f64 = if args.len() >= 10 {
        // G8-C12d-5: the optional 14th argument is only a block score mode.
        // Keep the explicit tolerance in args[9] for every extended CLI form.
        args[9].parse()?
    } else {
        1.0e-8
    };
    if !requested_tolerance.is_finite() || !(1.0e-14..=1.0e-5).contains(&requested_tolerance) {
        return Err("relative_tolerance must be finite and within 1e-14..=1e-5".into());
    }
    let adaptive_policy = if let Some(block_size) = adaptive_block_size {
        let probe_iterations: usize = if args.len() >= 12 {
            args[10].parse()?
        } else {
            64
        };
        let poor_progress_ratio: f64 = if args.len() >= 12 {
            args[11].parse()?
        } else {
            0.5
        };
        if probe_iterations == 0
            || probe_iterations >= max_iterations
            || !poor_progress_ratio.is_finite()
            || poor_progress_ratio <= 0.0
            || poor_progress_ratio >= 1.0
        {
            return Err("invalid adaptive probe_iterations or poor_progress_ratio".into());
        }
        Some(DistributedAdaptivePcgPolicy {
            probe_iterations,
            poor_progress_ratio,
            block_size,
            factor_budget_bytes_global: if args.len() >= 13 {
                args[12].parse()?
            } else {
                // Preserve G8-C12d-3 behavior if no budget argument is given.
                usize::MAX
            },
            selection_mode: if args.len() >= 14 {
                match args[13].as_str() {
                    "residual" => DistributedAdaptiveBlockScore::ResidualEnergy,
                    "boundary" => DistributedAdaptiveBlockScore::BoundaryCoupling,
                    replay_pooled if replay_pooled.starts_with("replayg:") => {
                        let data = replay_pooled.strip_prefix("replayg:").unwrap_or_default();
                        let parts: Vec<&str> = data.split(',').collect();
                        if parts.len() != 8 {
                            return Err("replayg requires exactly eight rank masks".into());
                        }
                        let mut masks = [0u32; 8];
                        for (dst, part) in masks.iter_mut().zip(parts.iter()) {
                            if part.len() != 5 || !part.bytes().all(|b| b.is_ascii_hexdigit()) {
                                return Err("replayg masks must be exactly five hexadecimal digits".into());
                            }
                            *dst = u32::from_str_radix(part, 16)?;
                        }
                        DistributedAdaptiveBlockScore::ReplayPooled { masks }
                    },
                    replay_no_coarse if replay_no_coarse.starts_with("replaync:") => {
                        let data = replay_no_coarse.strip_prefix("replaync:").unwrap_or_default();
                        let parts: Vec<&str> = data.split(',').collect();
                        if parts.len() != 8 {
                            return Err("replaync requires exactly eight rank masks".into());
                        }
                        let mut masks = [0u32; 8];
                        for (dst, part) in masks.iter_mut().zip(parts.iter()) {
                            if part.len() != 5 || !part.bytes().all(|b| b.is_ascii_hexdigit()) {
                                return Err("replaync masks must be exactly five hexadecimal digits".into());
                            }
                            *dst = u32::from_str_radix(part, 16)?;
                        }
                        DistributedAdaptiveBlockScore::ReplayMaskNoCoarse { masks }
                    },
                    replay if replay.starts_with("replay:") => {
                        let data = replay.strip_prefix("replay:").unwrap_or_default();
                        let parts: Vec<&str> = data.split(',').collect();
                        if parts.len() != 8 {
                            return Err("replay requires exactly eight rank masks".into());
                        }
                        let mut masks = [0u32; 8];
                        for (dst, part) in masks.iter_mut().zip(parts.iter()) {
                            if part.len() != 5 || !part.bytes().all(|b| b.is_ascii_hexdigit()) {
                                return Err("replay masks must be exactly five hexadecimal digits".into());
                            }
                            *dst = u32::from_str_radix(part, 16)?;
                        }
                        DistributedAdaptiveBlockScore::ReplayMask { masks }
                    },
                    intervention if intervention.starts_with("swap:") => {
                        let mut fields = intervention.split(':');
                        let _prefix = fields.next();
                        let target_rank: usize =
                            fields.next().ok_or("missing swap rank")?.parse()?;
                        let remove: usize =
                            fields.next().ok_or("missing swap remove block")?.parse()?;
                        let add: usize = fields.next().ok_or("missing swap add block")?.parse()?;
                        if fields.next().is_some() {
                            return Err("expected swap:rank:remove:add".into());
                        }
                        DistributedAdaptiveBlockScore::SwapOne {
                            target_rank,
                            remove,
                            add,
                        }
                    }
                    _ => {
                        return Err(
                            "selection mode must be residual, boundary, swap:rank:remove:add, replay:<eight-hex-masks> or replaync:<eight-hex-masks> or replayg:<eight-hex-masks>"
                                .into(),
                        )
                    }
                }
            } else {
                DistributedAdaptiveBlockScore::ResidualEnergy
            },
        })
    } else {
        if args.len() >= 12 {
            return Err("adaptive probe options require adaptive:N".into());
        }
        None
    };
    // Existing 9/10/12/13/14-argument invocations preserve the original RHS.
    // 11 arguments: a fixed method plus RHS kind. 15: adaptive plus RHS kind.
    let rhs_kind: &str = match args.len() {
        11 => args[10].as_str(),
        15 => args[14].as_str(),
        _ => "smooth",
    };
    if !matches!(rhs_kind, "smooth" | "mixed" | "localized") {
        return Err("RHS kind must be smooth, mixed or localized".into());
    }
    if args.len() == 15 && adaptive_block_size.is_none() {
        return Err("15-argument RHS form requires adaptive:N".into());
    }
    let trace_stride = if matches!(args.len(), 11 | 15) {
        64usize
    } else {
        0usize
    };
    let mpi = MpiRuntime::initialize()?;
    if mpi.size() != expected || !matches!(expected, 1 | 2 | 4 | 8) {
        return Err("MPI rank mismatch or unsupported rank count (1,2,4,8 only)".into());
    }
    if mpi.rank() == 0 {
        let needs_header = !csv.exists() || std::fs::metadata(csv)?.len() == 0;
        if needs_header {
            let mut out = OpenOptions::new().append(true).create(true).open(csv)?;
            writeln!(out, "case,policy,ranks,n,nnz,preconditioner,iterations,status,interpretation,true_relative_residual,max_abs_error,spmv_calls,allreduce_calls,max_setup_ms,max_solve_ms,max_spmv_ms,max_allreduce_ms,max_preconditioner_apply_ms,coarse_allreduce_calls,max_coarse_allreduce_ms,factor_bytes_global,adaptive_escalation_iteration,adaptive_probe_ratio,adaptive_factor_setup_max_ms,final_factor_bytes_global")?;
        }
    }
    if mpi.rank() == 0 && trace_stride > 0 {
        println!("G8-C12d-6B RHS={rhs_kind} trace_stride={trace_stride} label={label}");
    }
    mpi.barrier();
    // Root-only file read followed by rank-local owned-CSR scatter.
    // All ranks enter the same collectives, including rejection paths.
    let (partition, owned, stats) = distribute_matrix_market_with_policy(&mpi, input, policy)?;
    if stats.global_nnz > max_nnz {
        return Err(format!(
            "expanded input nnz {} exceeds safety limit {max_nnz}",
            stats.global_nnz
        )
        .into());
    }
    let (plan, local, _halo) = prepare_owned_rows(&mpi, &partition, &owned)?;
    let mut rhs = vec![0.0; local.owned_len()];
    for (row, entry) in rhs.iter_mut().enumerate() {
        let start = owned.row_ptr[row] as usize;
        let end = owned.row_ptr[row + 1] as usize;
        for p in start..end {
            *entry += owned.values[p] * rhs_truth(owned.global_col_idx[p], rhs_kind);
        }
    }
    mpi.barrier();
    let prep_start = Instant::now();
    let mut solver = DistributedPcg::prepare_with_preconditioner(&mpi, &plan, &local, choice)?;
    let max_setup_ms = max_f64(prep_start.elapsed().as_secs_f64() * 1e3);
    let factor_bytes_global = mpi.all_reduce_sum_u64(solver.preconditioner_factor_bytes() as u64);
    // D6-B: only the new 11/15-argument experiment enables sampling.
    // Legacy test calls keep identical numerical iteration and communication.
    solver.set_residual_trace_stride(trace_stride);
    let options = DistributedPcgOptions {
        relative_tolerance: requested_tolerance,
        absolute_tolerance: 1e-12,
        max_iterations,
    };
    let mut x = vec![0.0; local.owned_len()];
    mpi.barrier();
    let report = if let Some(policy) = adaptive_policy {
        solver.solve_adaptive(
            &mpi,
            &plan,
            &local,
            &rhs,
            &mut x,
            DistributedAdaptiveSolveOptions {
                pcg: options,
                policy,
            },
        )?
    } else {
        solver.solve(&mpi, &local, &rhs, &mut x, options)?
    };
    let final_factor_bytes_global =
        mpi.all_reduce_sum_u64(solver.preconditioner_factor_bytes() as u64);
    let adaptive_factor_setup_max_ms = max_f64(report.adaptive_factor_setup_ns as f64 * 1e-6);
    // Ranks write separate files. Reduce local I/O errors *collectively*
    // before any subsequent reductions to avoid MPI asymmetric exit/hang.
    let trace_write = match adaptive_policy {
        Some(config) if report.adaptive_escalation_iteration > 0 => write_adaptive_trace(
            csv,
            mpi.rank(),
            owned.owned.start,
            config.selection_mode,
            solver.adaptive_block_trace(),
        ),
        _ => Ok(()),
    };
    let trace_io_failures = mpi.all_reduce_sum_u64(u64::from(trace_write.is_err()));
    if trace_io_failures > 0 {
        return Err("at least one MPI rank failed to export its block trace".into());
    }
    trace_write?;
    // The rank-zero file operation is followed by a collective status check.
    // The trace contains globally reduced norms; no per-rank duplicate needed.
    let history_io = if trace_stride > 0 && mpi.rank() == 0 {
        write_residual_trace(csv, solver.residual_trace())
    } else {
        Ok(())
    };
    let history_failures = mpi.all_reduce_sum_u64(u64::from(history_io.is_err()));
    if history_failures != 0 {
        return Err("at least one MPI rank failed to write the residual trace".into());
    }
    history_io?;
    let max_solve_ms = max_f64(report.elapsed_ns as f64 * 1e-6);
    let max_spmv_ms = max_f64(report.spmv_elapsed_ns as f64 * 1e-6);
    let max_allreduce_ms = max_f64(report.allreduce_elapsed_ns as f64 * 1e-6);
    let max_preconditioner_apply_ms = max_f64(report.preconditioner_apply_ns as f64 * 1e-6);
    let max_coarse_allreduce_ms = max_f64(report.coarse_allreduce_elapsed_ns as f64 * 1e-6);
    let local_error = x
        .iter()
        .enumerate()
        .map(|(i, &v)| (v - rhs_truth(owned.owned.start + i as u64, rhs_kind)).abs())
        .fold(0.0_f64, f64::max);
    let max_abs_error = max_f64(local_error);
    let residual_ok = report.status == DistributedPcgStatus::Converged
        && report.true_relative_residual.is_finite()
        && report.true_relative_residual <= 1.1e-8;
    let accurate = residual_ok && max_abs_error.is_finite() && max_abs_error <= 2e-5;
    let interpretation = if accurate {
        "CONVERGED_ACCURATE"
    } else if residual_ok {
        "CONVERGED_RESIDUAL_ONLY"
    } else {
        "NOT_CONVERGED_OR_BREAKDOWN"
    };
    if mpi.rank() == 0 {
        let policy_label = if policy == MtxPcgPolicy::StrictCertified {
            "certified"
        } else {
            "experimental_unverified"
        };
        let mut out = OpenOptions::new().append(true).open(csv)?;
        writeln!(out, "{label},{policy_label},{expected},{},{},{selection},{},{:?},{interpretation},{:.12e},{max_abs_error:.12e},{},{},{max_setup_ms:.6},{max_solve_ms:.6},{max_spmv_ms:.6},{max_allreduce_ms:.6},{max_preconditioner_apply_ms:.6},{},{max_coarse_allreduce_ms:.6},{factor_bytes_global},{},{:.12e},{adaptive_factor_setup_max_ms:.6},{final_factor_bytes_global}",
            partition.global_dofs(), stats.global_nnz, report.iterations, report.status,
            report.true_relative_residual, report.spmv_calls, report.allreduce_calls, report.coarse_allreduce_calls,
            report.adaptive_escalation_iteration, report.adaptive_probe_ratio)?;
        println!("G8-C12a {interpretation} {label}/{selection}: ranks={expected} n={} nnz={} iter={} status={:?} true_rel={:.4e} err={max_abs_error:.3e} solve={max_solve_ms:.3}ms",
            partition.global_dofs(), stats.global_nnz, report.iterations, report.status, report.true_relative_residual);
        if adaptive_policy.is_some() {
            println!(
                "G8-C12d-3 ADAPTIVE {label}/{selection}: probe_iter={} ratio={:.6e} promoted_at={} direct_setup_max_ms={:.4} final_factor_bytes_global={final_factor_bytes_global}",
                adaptive_policy.map_or(0, |v| v.probe_iterations),
                report.adaptive_probe_ratio,
                report.adaptive_escalation_iteration,
                adaptive_factor_setup_max_ms,
            );
        }
        if policy == MtxPcgPolicy::ExperimentalUnverified {
            println!("G8-C12a WARNING: manufactured-solution convergence does not prove matrix SPD or physical validity");
        }
    }
    mpi.barrier();
    if policy == MtxPcgPolicy::StrictCertified && !accurate {
        return Err("certified smoke test failed numerical correctness gate".into());
    }
    Ok(())
}
