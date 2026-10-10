//! G8-C6: reproducible paired Jacobi / local-block-Cholesky MPI PCG probe.
//! The experimental policy is NOT a mathematical global SPD certificate.
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

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn truth(global: u64) -> f64 {
    let t = global as f64 + 1.0;
    (t * 0.017).sin() + 0.21 * (t * 0.0061).cos()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 9 {
        return Err("usage: mpi_block_pcg <ranks> <csv> <matrix.mtx> <label> <certified|experimental> <block_size> <warmups> <repeats> <max_iters>".into());
    }
    let expected: i32 = args[0].parse()?;
    let csv = Path::new(&args[1]);
    let matrix = Path::new(&args[2]);
    let label = &args[3];
    let (policy, policy_label) = match args[4].as_str() {
        "certified" => (MtxPcgPolicy::StrictCertified, "certified"),
        "experimental" => (
            MtxPcgPolicy::ExperimentalUnverified,
            "experimental_unverified",
        ),
        _ => return Err("policy must be certified or experimental".into()),
    };
    let block_size: usize = args[5].parse()?;
    let warmups: usize = args[6].parse()?;
    let repeats: usize = args[7].parse()?;
    let max_iters: usize = args[8].parse()?;
    if !(2..=512).contains(&block_size)
        || repeats == 0
        || repeats > 50
        || warmups > 50
        || max_iters == 0
        || max_iters > 100_000
    {
        return Err("invalid block_size / warmups / repeats / max_iters".into());
    }
    let mpi = MpiRuntime::initialize()?;
    if expected != mpi.size() || !matches!(expected, 1 | 2 | 4 | 8) {
        return Err("MPI rank mismatch; expected 1/2/4/8".into());
    }
    if mpi.rank() == 0 {
        let needs_header = !csv.exists() || std::fs::metadata(csv)?.len() == 0;
        if needs_header {
            let mut f = OpenOptions::new().create(true).append(true).open(csv)?;
            writeln!(f, "case,policy,ranks,n,nnz,preconditioner,block_size,warmups,repeats,iterations,status,true_relative_residual,max_abs_error,spmv_calls,allreduce_calls,preconditioner_prepare_max_ms,median_max_solve_ms,median_max_spmv_ms,median_max_allreduce_ms,median_max_precond_apply_ms")?;
        }
    }
    mpi.barrier();
    let (partition, owned, ingest) = distribute_matrix_market_with_policy(&mpi, matrix, policy)?;
    let (plan, local, _stats) = prepare_owned_rows(&mpi, &partition, &owned)?;
    let mut b = vec![0.0; plan.owned_len()];
    for (row, rhs) in b.iter_mut().enumerate() {
        let begin = owned.row_ptr[row] as usize;
        let end = owned.row_ptr[row + 1] as usize;
        for p in begin..end {
            *rhs += owned.values[p] * truth(owned.global_col_idx[p]);
        }
    }
    let opts = DistributedPcgOptions {
        relative_tolerance: 1.0e-11,
        absolute_tolerance: 1.0e-13,
        max_iterations: max_iters,
    };
    let mut baseline_iters = None;
    for (name, choice) in [
        ("jacobi", DistributedPcgPreconditioner::Jacobi),
        (
            "local_block_cholesky",
            DistributedPcgPreconditioner::LocalBlockCholesky { block_size },
        ),
    ] {
        mpi.barrier();
        let started = Instant::now();
        let mut solver = DistributedPcg::prepare_with_preconditioner(&mpi, &plan, &local, choice)?;
        let setup_ms = max_f64(started.elapsed().as_secs_f64() * 1e3);
        let mut solve_times = Vec::with_capacity(repeats);
        let mut spmv_times = Vec::with_capacity(repeats);
        let mut reduction_times = Vec::with_capacity(repeats);
        let mut precond_times = Vec::with_capacity(repeats);
        let mut last_iters = 0usize;
        let mut last_rel = f64::INFINITY;
        let mut last_error = f64::INFINITY;
        let mut last_spmvs = 0usize;
        let mut last_reductions = 0usize;
        let mut last_status = DistributedPcgStatus::MaxIterations;
        for trial in 0..(warmups + repeats) {
            let mut x = vec![0.0; plan.owned_len()];
            mpi.barrier();
            let report = solver.solve(&mpi, &local, &b, &mut x, opts)?;
            let solve_ms = max_f64(report.elapsed_ns as f64 * 1.0e-6);
            let spmv_ms = max_f64(report.spmv_elapsed_ns as f64 * 1.0e-6);
            let reduction_ms = max_f64(report.allreduce_elapsed_ns as f64 * 1.0e-6);
            let precond_ms = max_f64(report.preconditioner_apply_ns as f64 * 1.0e-6);
            let local_err = x
                .iter()
                .enumerate()
                .map(|(i, &value)| (value - truth(owned.owned.start + i as u64)).abs())
                .fold(0.0_f64, f64::max);
            let global_err = max_f64(local_err);
            // Treat every repetition as a numerical correctness test, not
            // just the last repetition, to detect unstable convergence.
            let bad = report.status != DistributedPcgStatus::Converged
                || !report.true_relative_residual.is_finite()
                || report.true_relative_residual > 1.1e-11
                || !global_err.is_finite()
                || global_err > 2.0e-7;
            if mpi.all_reduce_sum_u64(u64::from(bad)) != 0 {
                return Err(format!(
                    "G8-C6 numerical gate failed: {label}/{name} run={trial} iterations={} relative={:.3e} error={global_err:.3e}",
                    report.iterations, report.true_relative_residual,
                ).into());
            }
            if trial >= warmups {
                solve_times.push(solve_ms);
                spmv_times.push(spmv_ms);
                reduction_times.push(reduction_ms);
                precond_times.push(precond_ms);
                last_iters = report.iterations;
                last_rel = report.true_relative_residual;
                last_error = global_err;
                last_spmvs = report.spmv_calls;
                last_reductions = report.allreduce_calls;
                last_status = report.status;
            }
        }
        let solve_median = median(&mut solve_times);
        let spmv_median = median(&mut spmv_times);
        let reduction_median = median(&mut reduction_times);
        let precond_median = median(&mut precond_times);
        mpi.barrier();
        if mpi.rank() == 0 {
            let mut f = OpenOptions::new().append(true).open(csv)?;
            writeln!(f, "{label},{policy_label},{expected},{},{},{name},{block_size},{warmups},{repeats},{last_iters},{last_status:?},{last_rel:.12e},{last_error:.12e},{last_spmvs},{last_reductions},{setup_ms:.6},{solve_median:.6},{spmv_median:.6},{reduction_median:.6},{precond_median:.6}", partition.global_dofs(), ingest.global_nnz)?;
            println!("PASS G8-C6 {label}/{name}: ranks={expected} iters={last_iters} rel={last_rel:.3e} err={last_error:.3e} solve_med={solve_median:.4}ms reduce_med={reduction_median:.4}ms setup={setup_ms:.4}ms");
        }
        if name == "jacobi" {
            baseline_iters = Some(last_iters);
        } else if label == "nonstrict_spd_512" && last_iters >= baseline_iters.unwrap_or(0) {
            return Err(
                "G8-C6 nonstrict fixture block preconditioner failed to reduce iterations".into(),
            );
        }
        mpi.barrier();
    }
    if mpi.rank() == 0 {
        if policy == MtxPcgPolicy::ExperimentalUnverified {
            println!("G8-C6 WARNING: exploratory input policy does not prove global SPD");
        }
        println!("=== HYBIT 0.9 G8-C6 MPI {expected}-RANK BLOCK PCG PASS ===");
    }
    mpi.barrier();
    Ok(())
}
