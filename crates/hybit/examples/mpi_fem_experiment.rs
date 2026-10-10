//! G8-C10: guarded MPI PCG experiment, NEVER a proof of global SPD.
//!
//! A manufactured right-hand side tests internal consistency but does not
//! certify physical correctness or positive-definiteness. Unverified matrices
//! require an explicit permission token and are not correctness-gated.
use hybit::distributed::mpi_backend::MpiRuntime;
use hybit::distributed::mpi_input::prepare_owned_rows;
use hybit::distributed::mpi_mtx::{distribute_matrix_market_with_policy, MtxPcgPolicy};
use hybit::distributed::mpi_pcg::{
    DistributedPcg, DistributedPcgOptions, DistributedPcgPreconditioner, DistributedPcgStatus,
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
    if args.len() != 9 {
        return Err("usage: mpi_fem_experiment <ranks> <csv> <mtx> <label> <certified|experimental> <jacobi|ic0:64|ic0:128> <max_iters> <max_nnz> <--certified-smoke|--permit-unverified-spd>".into());
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
    let choice = if selection == "jacobi" {
        DistributedPcgPreconditioner::Jacobi
    } else if let Some(size) = selection.strip_prefix("ic0:") {
        let block_size: usize = size.parse()?;
        if !(2..=512).contains(&block_size) {
            return Err("IC(0) block size must be 2..=512".into());
        }
        DistributedPcgPreconditioner::LocalBlockIc0 { block_size }
    } else {
        return Err("preconditioner must be jacobi or ic0:<blocksize>".into());
    };
    let max_iterations: usize = args[6].parse()?;
    let max_nnz: u64 = args[7].parse()?;
    if !(1..=100_000).contains(&max_iterations) || !(1..=20_000_000).contains(&max_nnz) {
        return Err("invalid iteration or expanded-NNZ cap".into());
    }
    let mpi = MpiRuntime::initialize()?;
    if mpi.size() != expected || !matches!(expected, 1 | 2 | 4 | 8) {
        return Err("MPI rank mismatch or unsupported rank count (1,2,4,8 only)".into());
    }
    if mpi.rank() == 0 {
        let needs_header = !csv.exists() || std::fs::metadata(csv)?.len() == 0;
        if needs_header {
            let mut out = OpenOptions::new().append(true).create(true).open(csv)?;
            writeln!(out, "case,policy,ranks,n,nnz,preconditioner,iterations,status,interpretation,true_relative_residual,max_abs_error,spmv_calls,allreduce_calls,max_setup_ms,max_solve_ms,max_spmv_ms,max_allreduce_ms,max_preconditioner_apply_ms,factor_bytes_global")?;
        }
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
            *entry += owned.values[p] * truth(owned.global_col_idx[p]);
        }
    }
    mpi.barrier();
    let prep_start = Instant::now();
    let mut solver = DistributedPcg::prepare_with_preconditioner(&mpi, &plan, &local, choice)?;
    let max_setup_ms = max_f64(prep_start.elapsed().as_secs_f64() * 1e3);
    let factor_bytes_global = mpi.all_reduce_sum_u64(solver.preconditioner_factor_bytes() as u64);
    let options = DistributedPcgOptions {
        relative_tolerance: 1e-8,
        absolute_tolerance: 1e-12,
        max_iterations,
    };
    let mut x = vec![0.0; local.owned_len()];
    mpi.barrier();
    let report = solver.solve(&mpi, &local, &rhs, &mut x, options)?;
    let max_solve_ms = max_f64(report.elapsed_ns as f64 * 1e-6);
    let max_spmv_ms = max_f64(report.spmv_elapsed_ns as f64 * 1e-6);
    let max_allreduce_ms = max_f64(report.allreduce_elapsed_ns as f64 * 1e-6);
    let max_preconditioner_apply_ms = max_f64(report.preconditioner_apply_ns as f64 * 1e-6);
    let local_error = x
        .iter()
        .enumerate()
        .map(|(i, &v)| (v - truth(owned.owned.start + i as u64)).abs())
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
        writeln!(out, "{label},{policy_label},{expected},{},{},{selection},{},{:?},{interpretation},{:.12e},{max_abs_error:.12e},{},{},{max_setup_ms:.6},{max_solve_ms:.6},{max_spmv_ms:.6},{max_allreduce_ms:.6},{max_preconditioner_apply_ms:.6},{factor_bytes_global}",
            partition.global_dofs(), stats.global_nnz, report.iterations, report.status,
            report.true_relative_residual, report.spmv_calls, report.allreduce_calls)?;
        println!("G8-C10 {interpretation} {label}/{selection}: ranks={expected} n={} nnz={} iter={} status={:?} true_rel={:.4e} err={max_abs_error:.3e} solve={max_solve_ms:.3}ms",
            partition.global_dofs(), stats.global_nnz, report.iterations, report.status, report.true_relative_residual);
        if policy == MtxPcgPolicy::ExperimentalUnverified {
            println!("G8-C10 WARNING: manufactured-solution convergence does not prove matrix SPD or physical validity");
        }
    }
    mpi.barrier();
    if policy == MtxPcgPolicy::StrictCertified && !accurate {
        return Err("certified smoke test failed numerical correctness gate".into());
    }
    Ok(())
}
