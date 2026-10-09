//! G8-C3: native MPI PCG from a rank-zero Matrix Market file.
//!
//! Input is SPD-certified at root by positive strict row diagonal dominance
//! and numeric symmetry. Each rank receives ONLY its owned CSR rows. The RHS
//! is manufactured row-locally, with no replicated solution/RHS arrays.

use hybit::distributed::mpi_backend::MpiRuntime;
use hybit::distributed::mpi_input::prepare_owned_rows;
use hybit::distributed::mpi_mtx::distribute_matrix_market;
use hybit::distributed::mpi_pcg::{DistributedPcg, DistributedPcgOptions, DistributedPcgStatus};
use mpi::collective::SystemOperation;
use mpi::traits::*;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn truth(global: u64, trial: usize) -> f64 {
    let seed = (global + 1) as f64;
    (seed * (trial as f64 + 1.0) * 0.071).sin() + 0.2 * (seed * 0.031).cos()
}

fn max_u64(local: u64) -> u64 {
    let world = mpi::topology::SimpleCommunicator::world();
    let mut global = 0u64;
    world.all_reduce_into(&local, &mut global, SystemOperation::max());
    global
}

fn max_f64(local: f64) -> f64 {
    let world = mpi::topology::SimpleCommunicator::world();
    let mut global = 0.0f64;
    world.all_reduce_into(&local, &mut global, SystemOperation::max());
    global
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: mpi_mtx_pcg <ranks> <csv> <matrix.mtx> <label>".into());
    }
    let expected: i32 = args[0].parse()?;
    let csv_path = &args[1];
    let matrix_path = Path::new(&args[2]);
    let label = &args[3];
    let mpi = MpiRuntime::initialize()?;
    if mpi.size() != expected || !matches!(mpi.size(), 1 | 2 | 4 | 8) {
        return Err("G8-C3 MPI ranks must match requested 1/2/4/8".into());
    }
    if mpi.rank() == 0 {
        let write_header = !Path::new(csv_path).exists() || std::fs::metadata(csv_path)?.len() == 0;
        if write_header {
            let mut f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(csv_path)?;
            writeln!(f, "case,ranks,n,nnz,rhs,iterations,status,true_relative_residual,max_abs_error,spmv_calls,allreduce_calls,global_ghosts,global_operator_bytes,root_read_ms,max_distribution_ms,max_halo_prepare_ms,max_pcg_prepare_ms,max_solve_ms")?;
        }
    }
    mpi.barrier();
    let (partition, owned, ingest) = distribute_matrix_market(&mpi, matrix_path)?;
    let halo_start = Instant::now();
    let (plan, local, local_stats) = prepare_owned_rows(&mpi, &partition, &owned)?;
    let halo_ms = halo_start.elapsed().as_secs_f64() * 1e3;
    let pcg_start = Instant::now();
    let mut solver = DistributedPcg::prepare(&mpi, &plan, &local)?;
    let pcg_prepare_ms = pcg_start.elapsed().as_secs_f64() * 1e3;
    let root_read_ms = mpi.all_reduce_sum_f64(ingest.root_read_ns as f64 * 1e-6);
    let max_distribution_ms = max_f64(ingest.distribution_ns as f64 * 1e-6);
    let max_halo_ms = max_f64(halo_ms);
    let max_pcg_prepare_ms = max_f64(pcg_prepare_ms);
    let global_ghosts = mpi.all_reduce_sum_u64(local_stats.ghosts as u64);
    let global_bytes = mpi.all_reduce_sum_u64(ingest.owned_bytes as u64);
    let _max_owned_bytes = max_u64(ingest.owned_bytes as u64);

    for trial in 0..2usize {
        let mut b = vec![0.0; plan.owned_len()];
        for (row, rhs) in b.iter_mut().enumerate() {
            let first = owned.row_ptr[row] as usize;
            let last = owned.row_ptr[row + 1] as usize;
            for position in first..last {
                *rhs += owned.values[position] * truth(owned.global_col_idx[position], trial);
            }
        }
        let mut x = vec![0.0; plan.owned_len()];
        let opts = DistributedPcgOptions {
            relative_tolerance: 1e-9,
            absolute_tolerance: 1e-13,
            max_iterations: 2500,
        };
        mpi.barrier();
        let result = solver.solve(&mpi, &local, &b, &mut x, opts)?;
        let max_solve_ms = max_f64(result.elapsed_ns as f64 * 1e-6);
        let mut local_error = 0.0_f64;
        for (index, &value) in x.iter().enumerate() {
            local_error =
                local_error.max((value - truth(owned.owned.start + index as u64, trial)).abs());
        }
        let global_error = max_f64(local_error);
        let bad = result.status != DistributedPcgStatus::Converged
            || !result.true_relative_residual.is_finite()
            || result.true_relative_residual > 1.10e-9
            || !global_error.is_finite()
            || global_error > 2e-7;
        if mpi.all_reduce_sum_u64(u64::from(bad)) != 0 {
            return Err(format!(
                "G8-C3 FAILED {label} rhs={trial}: status={:?} rel={:.5e} error={global_error:.5e}",
                result.status, result.true_relative_residual
            )
            .into());
        }
        mpi.barrier();
        if mpi.rank() == 0 {
            let mut f = OpenOptions::new().append(true).open(csv_path)?;
            writeln!(f, "{label},{expected},{},{},{trial},{},{:?},{:.12e},{:.12e},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
                partition.global_dofs(), ingest.global_nnz, result.iterations,
                result.status, result.true_relative_residual, global_error,
                result.spmv_calls, result.allreduce_calls, global_ghosts, global_bytes,
                root_read_ms, max_distribution_ms, max_halo_ms, max_pcg_prepare_ms, max_solve_ms)?;
            println!("PASS C3 {label} rhs={trial}: ranks={expected} n={} nnz={} iter={} true_rel={:.3e} max_abs_error={global_error:.3e} ghosts={global_ghosts} max_solve_ms={max_solve_ms:.4}",
                partition.global_dofs(), ingest.global_nnz, result.iterations, result.true_relative_residual);
        }
        mpi.barrier();
    }
    if mpi.rank() == 0 {
        println!("=== HYBIT 0.9 G8-C3 MPI {expected}-RANK MATRIX MARKET / PCG PASS ===");
    }
    mpi.barrier();
    Ok(())
}
