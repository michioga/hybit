//! G8-C8: apples-to-apples rank-local Jacobi / dense block / IC(0) comparison.
//!
//! No change to the collective PCG algorithm or input-policy semantics.
//! Setup + median solve estimates are NOT timed multiple-RHS workloads.
//! Factor storage is only the allocated factor arrays (not process RSS).
//! ExperimentalUnverified is not a mathematical SPD certificate.

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

fn max_f64(local: f64) -> f64 {
    let mut global = 0.0;
    mpi::topology::SimpleCommunicator::world().all_reduce_into(
        &local,
        &mut global,
        SystemOperation::max(),
    );
    global
}

fn max_u64(local: u64) -> u64 {
    let mut global = 0;
    mpi::topology::SimpleCommunicator::world().all_reduce_into(
        &local,
        &mut global,
        SystemOperation::max(),
    );
    global
}

fn median(items: &mut [f64]) -> f64 {
    items.sort_by(f64::total_cmp);
    let midpoint = items.len() / 2;
    if items.len() % 2 == 0 {
        0.5 * (items[midpoint - 1] + items[midpoint])
    } else {
        items[midpoint]
    }
}

fn truth(global: u64) -> f64 {
    let x = global as f64 + 1.0;
    (x * 0.017).sin() + 0.21 * (x * 0.0061).cos()
}

#[derive(Clone, Copy)]
struct Score {
    name: &'static str,
    requested: usize,
    setup_ms: f64,
    solve_ms: f64,
}
impl Score {
    fn estimate(self, rhs_count: f64) -> f64 {
        self.setup_ms + rhs_count * self.solve_ms
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 8 {
        return Err("usage: mpi_ic0_sweep <ranks> <csv> <matrix.mtx> <case> <certified|experimental> <warmups> <repeats> <max_iters>".into());
    }
    let expected: i32 = args[0].parse()?;
    let csv = Path::new(&args[1]);
    let matrix = Path::new(&args[2]);
    let label = &args[3];
    let (policy, policy_name) = match args[4].as_str() {
        "certified" => (MtxPcgPolicy::StrictCertified, "certified"),
        "experimental" => (
            MtxPcgPolicy::ExperimentalUnverified,
            "experimental_unverified",
        ),
        _ => return Err("policy must be certified or experimental".into()),
    };
    let warmups: usize = args[5].parse()?;
    let repeats: usize = args[6].parse()?;
    let max_iters: usize = args[7].parse()?;
    if warmups > 50 || !(1..=50).contains(&repeats) || !(1..=100_000).contains(&max_iters) {
        return Err("unsupported warmup/repeats/max_iters".into());
    }
    let mpi = MpiRuntime::initialize()?;
    if mpi.size() != expected || !matches!(expected, 1 | 2 | 4 | 8) {
        return Err("MPI rank mismatch: expected 1/2/4/8".into());
    }
    if mpi.rank() == 0 && (!csv.exists() || std::fs::metadata(csv)?.len() == 0) {
        let mut file = OpenOptions::new().create(true).append(true).open(csv)?;
        writeln!(file, "case,policy,ranks,n,nnz,preconditioner,requested_block_rows,max_effective_block_rows,global_blocks,factor_sum_bytes,factor_max_rank_bytes,factor_sum_values,warmups,repeats,setup_max_ms,iterations,status,true_relative_residual,max_abs_error,median_max_solve_ms,median_max_spmv_ms,median_max_allreduce_ms,median_max_precond_apply_ms,estimated_one_rhs_ms,estimated_ten_rhs_ms,spmv_calls,allreduce_calls")?;
    }
    mpi.barrier();
    let (partition, owned, ingest) = distribute_matrix_market_with_policy(&mpi, matrix, policy)?;
    let (plan, local, _) = prepare_owned_rows(&mpi, &partition, &owned)?;
    let mut rhs = vec![0.0; plan.owned_len()];
    for (row, value) in rhs.iter_mut().enumerate() {
        let start = owned.row_ptr[row] as usize;
        let end = owned.row_ptr[row + 1] as usize;
        for p in start..end {
            *value += owned.values[p] * truth(owned.global_col_idx[p]);
        }
    }
    let options = DistributedPcgOptions {
        relative_tolerance: 1.0e-11,
        absolute_tolerance: 1.0e-13,
        max_iterations: max_iters,
    };
    const SIZES: [usize; 5] = [16, 32, 64, 128, 256];
    let mut candidates = vec![("jacobi", 0usize, DistributedPcgPreconditioner::Jacobi)];
    for block_size in SIZES {
        candidates.push((
            "dense_block_cholesky",
            block_size,
            DistributedPcgPreconditioner::LocalBlockCholesky { block_size },
        ));
        candidates.push((
            "sparse_block_ic0",
            block_size,
            DistributedPcgPreconditioner::LocalBlockIc0 { block_size },
        ));
    }
    let mut scores = Vec::<Score>::new();
    for (name, block_size, preconditioner) in candidates {
        mpi.barrier();
        let prepare_start = Instant::now();
        let mut solver =
            DistributedPcg::prepare_with_preconditioner(&mpi, &plan, &local, preconditioner)?;
        let setup_ms = max_f64(prepare_start.elapsed().as_secs_f64() * 1000.0);
        let local_bytes = solver.preconditioner_factor_bytes() as u64;
        let local_values = solver.preconditioner_factor_values() as u64;
        let sum_bytes = mpi.all_reduce_sum_u64(local_bytes);
        let max_bytes = max_u64(local_bytes);
        let sum_values = mpi.all_reduce_sum_u64(local_values);
        let effective = if block_size == 0 {
            1
        } else {
            block_size.min(local.owned_len())
        };
        let max_effective = max_u64(effective as u64);
        let local_blocks = if block_size == 0 {
            local.owned_len()
        } else {
            local.owned_len().div_ceil(block_size)
        };
        let global_blocks = mpi.all_reduce_sum_u64(local_blocks as u64);
        let mut times = Vec::with_capacity(repeats);
        let mut apply_times = Vec::with_capacity(repeats);
        let mut reduce_times = Vec::with_capacity(repeats);
        let mut spmv_times = Vec::with_capacity(repeats);
        let mut iterations = 0usize;
        let mut relative = f64::INFINITY;
        let mut abs_error = f64::INFINITY;
        let mut status = DistributedPcgStatus::MaxIterations;
        let mut spmv_calls = 0usize;
        let mut allreduce_calls = 0usize;
        for trial in 0..(warmups + repeats) {
            let mut x = vec![0.0; plan.owned_len()];
            mpi.barrier();
            let report = solver.solve(&mpi, &local, &rhs, &mut x, options)?;
            let elapsed = max_f64(report.elapsed_ns as f64 * 1.0e-6);
            let precond_time = max_f64(report.preconditioner_apply_ns as f64 * 1.0e-6);
            let reduce_time = max_f64(report.allreduce_elapsed_ns as f64 * 1.0e-6);
            let spmv_time = max_f64(report.spmv_elapsed_ns as f64 * 1.0e-6);
            let err_local = x
                .iter()
                .enumerate()
                .map(|(i, &xi)| (xi - truth(owned.owned.start + i as u64)).abs())
                .fold(0.0, f64::max);
            let err_global = max_f64(err_local);
            let bad = report.status != DistributedPcgStatus::Converged
                || !report.true_relative_residual.is_finite()
                || report.true_relative_residual > 1.1e-11
                || !err_global.is_finite()
                || err_global > 2.0e-7;
            if mpi.all_reduce_sum_u64(u64::from(bad)) != 0 {
                return Err(format!(
                    "G8-C8 failed {label}/{name}[{block_size}] rank={expected} run={trial} status={:?} iters={} rel={:.3e} error={err_global:.3e}",
                    report.status, report.iterations, report.true_relative_residual
                ).into());
            }
            if trial >= warmups {
                times.push(elapsed);
                apply_times.push(precond_time);
                reduce_times.push(reduce_time);
                spmv_times.push(spmv_time);
                iterations = report.iterations;
                relative = report.true_relative_residual;
                abs_error = err_global;
                status = report.status;
                spmv_calls = report.spmv_calls;
                allreduce_calls = report.allreduce_calls;
            }
        }
        let solve_ms = median(&mut times);
        let apply_ms = median(&mut apply_times);
        let reduce_ms = median(&mut reduce_times);
        let spmv_ms = median(&mut spmv_times);
        let score = Score {
            name,
            requested: block_size,
            setup_ms,
            solve_ms,
        };
        scores.push(score);
        if mpi.rank() == 0 {
            let mut f = OpenOptions::new().append(true).open(csv)?;
            writeln!(f, "{label},{policy_name},{expected},{},{},{name},{block_size},{max_effective},{global_blocks},{sum_bytes},{max_bytes},{sum_values},{warmups},{repeats},{setup_ms:.6},{iterations},{status:?},{relative:.12e},{abs_error:.12e},{solve_ms:.6},{spmv_ms:.6},{reduce_ms:.6},{apply_ms:.6},{:.6},{:.6},{spmv_calls},{allreduce_calls}",
                partition.global_dofs(), ingest.global_nnz, score.estimate(1.0), score.estimate(10.0))?;
            println!("PASS G8-C8 {label}/{name}[{block_size}] ranks={expected}: iter={iterations}, rel={relative:.3e}, bytes={sum_bytes}, setup={setup_ms:.4}ms, solve={solve_ms:.4}ms, apply={apply_ms:.4}ms, 1rhs={:.4}ms, 10rhs={:.4}ms", score.estimate(1.0), score.estimate(10.0));
        }
        mpi.barrier();
    }
    if mpi.rank() == 0 {
        let best1 = scores
            .iter()
            .min_by(|a, b| a.estimate(1.0).total_cmp(&b.estimate(1.0)))
            .ok_or("no scores")?;
        let best10 = scores
            .iter()
            .min_by(|a, b| a.estimate(10.0).total_cmp(&b.estimate(10.0)))
            .ok_or("no scores")?;
        println!(
            "G8-C8 SELECT {label} ranks={expected}: one={}({}) {:.4}ms ten={}({}) {:.4}ms",
            best1.name,
            best1.requested,
            best1.estimate(1.0),
            best10.name,
            best10.requested,
            best10.estimate(10.0)
        );
        if policy == MtxPcgPolicy::ExperimentalUnverified {
            println!(
                "G8-C8 WARNING: exploratory Matrix Market acceptance does not establish global SPD"
            );
        }
        println!("=== HYBIT 0.9 G8-C8 MPI {expected}-RANK IC0 SWEEP PASS ===");
    }
    mpi.barrier();
    Ok(())
}
