//! G8-C7: MPI PCG sweep of SPD-compatible rank-local dense block sizes.
//!
//! Setup + median solve is an ESTIMATE for a batch of RHS vectors, not an
//! end-to-end timed assembly/solve workflow. CSV metrics are independent
//! medians of rank-max measurements and are not additive.
//! ExperimentalUnverified never certifies a global matrix as SPD.

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

fn rank_max_f64(local: f64) -> f64 {
    let mut global = 0.0;
    mpi::topology::SimpleCommunicator::world().all_reduce_into(
        &local,
        &mut global,
        SystemOperation::max(),
    );
    global
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    let mid = samples.len() / 2;
    if samples.len() % 2 == 0 {
        (samples[mid - 1] + samples[mid]) * 0.5
    } else {
        samples[mid]
    }
}

fn truth(global: u64) -> f64 {
    let t = global as f64 + 1.0;
    (t * 0.017).sin() + 0.21 * (t * 0.0061).cos()
}

#[derive(Clone, Copy)]
struct Score {
    label: &'static str,
    block_size: usize,
    setup_ms: f64,
    solve_ms: f64,
    iterations: usize,
}

impl Score {
    fn estimated_ms(self, nrhs: f64) -> f64 {
        self.setup_ms + nrhs * self.solve_ms
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 8 {
        return Err("usage: mpi_block_sweep <ranks> <csv> <matrix.mtx> <label> <certified|experimental> <warmups> <repeats> <max_iters>".into());
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
    let warmups: usize = args[5].parse()?;
    let repeats: usize = args[6].parse()?;
    let max_iters: usize = args[7].parse()?;
    if warmups > 30 || !(1..=30).contains(&repeats) || !(1..=100_000).contains(&max_iters) {
        return Err("warmup/repeats/max_iters out of supported range".into());
    }
    let mpi = MpiRuntime::initialize()?;
    if mpi.size() != expected || !matches!(expected, 1 | 2 | 4 | 8) {
        return Err("MPI rank mismatch: expected 1,2,4,8".into());
    }
    if mpi.rank() == 0 && (!csv.exists() || std::fs::metadata(csv)?.len() == 0) {
        let mut f = OpenOptions::new().create(true).append(true).open(csv)?;
        writeln!(f, "case,policy,ranks,n,nnz,preconditioner,block_size,warmups,repeats,iterations,status,true_relative_residual,max_abs_error,spmv_calls,allreduce_calls,setup_max_ms,median_max_solve_ms,median_max_spmv_ms,median_max_allreduce_ms,median_max_precond_apply_ms,median_max_halo_wait_ms,estimated_one_rhs_ms,estimated_ten_rhs_ms")?;
    }
    mpi.barrier();
    let (partition, owned, ingest) = distribute_matrix_market_with_policy(&mpi, matrix, policy)?;
    let (plan, local, _) = prepare_owned_rows(&mpi, &partition, &owned)?;
    let mut rhs = vec![0.0; plan.owned_len()];
    for (row, value) in rhs.iter_mut().enumerate() {
        let begin = owned.row_ptr[row] as usize;
        let end = owned.row_ptr[row + 1] as usize;
        for i in begin..end {
            *value += owned.values[i] * truth(owned.global_col_idx[i]);
        }
    }
    let opts = DistributedPcgOptions {
        relative_tolerance: 1.0e-11,
        absolute_tolerance: 1.0e-13,
        max_iterations: max_iters,
    };
    let candidates: [(&str, usize); 7] = [
        ("jacobi", 0),
        ("local_block_cholesky", 8),
        ("local_block_cholesky", 16),
        ("local_block_cholesky", 32),
        ("local_block_cholesky", 64),
        ("local_block_cholesky", 128),
        ("local_block_cholesky", 256),
    ];
    let mut scores = Vec::<Score>::with_capacity(candidates.len());
    for &(name, block_size) in &candidates {
        let choice = if block_size == 0 {
            DistributedPcgPreconditioner::Jacobi
        } else {
            DistributedPcgPreconditioner::LocalBlockCholesky { block_size }
        };
        mpi.barrier();
        let started = Instant::now();
        let mut solver = DistributedPcg::prepare_with_preconditioner(&mpi, &plan, &local, choice)?;
        let setup_ms = rank_max_f64(started.elapsed().as_secs_f64() * 1.0e3);
        let mut solve_samples = Vec::with_capacity(repeats);
        let mut spmv_samples = Vec::with_capacity(repeats);
        let mut reduce_samples = Vec::with_capacity(repeats);
        let mut apply_samples = Vec::with_capacity(repeats);
        let mut wait_samples = Vec::with_capacity(repeats);
        let mut iterations = 0usize;
        let mut relative = f64::INFINITY;
        let mut abs_error = f64::INFINITY;
        let mut status = DistributedPcgStatus::MaxIterations;
        let mut spmv_calls = 0usize;
        let mut allreduce_calls = 0usize;
        for run in 0..(warmups + repeats) {
            let mut x = vec![0.0; plan.owned_len()];
            mpi.barrier();
            let report = solver.solve(&mpi, &local, &rhs, &mut x, opts)?;
            let elapsed_ms = rank_max_f64(report.elapsed_ns as f64 * 1.0e-6);
            let spmv_ms = rank_max_f64(report.spmv_elapsed_ns as f64 * 1.0e-6);
            let reduce_ms = rank_max_f64(report.allreduce_elapsed_ns as f64 * 1.0e-6);
            let apply_ms = rank_max_f64(report.preconditioner_apply_ns as f64 * 1.0e-6);
            let wait_ms = rank_max_f64(report.halo_wait_ns as f64 * 1.0e-6);
            let local_error = x
                .iter()
                .enumerate()
                .map(|(i, &xi)| (xi - truth(owned.owned.start + i as u64)).abs())
                .fold(0.0_f64, f64::max);
            let global_error = rank_max_f64(local_error);
            let bad = report.status != DistributedPcgStatus::Converged
                || !report.true_relative_residual.is_finite()
                || report.true_relative_residual > 1.1e-11
                || !global_error.is_finite()
                || global_error > 2.0e-7;
            // Every run is a numerical gate; all ranks reject in the same epoch.
            if mpi.all_reduce_sum_u64(u64::from(bad)) != 0 {
                return Err(format!(
                    "G8-C7 solve failed: {label}/{name}/block={block_size}/run={run}, iters={}, rel={:.3e}, error={global_error:.3e}",
                    report.iterations,
                    report.true_relative_residual,
                ).into());
            }
            if run >= warmups {
                solve_samples.push(elapsed_ms);
                spmv_samples.push(spmv_ms);
                reduce_samples.push(reduce_ms);
                apply_samples.push(apply_ms);
                wait_samples.push(wait_ms);
                iterations = report.iterations;
                relative = report.true_relative_residual;
                abs_error = global_error;
                status = report.status;
                spmv_calls = report.spmv_calls;
                allreduce_calls = report.allreduce_calls;
            }
        }
        let solve_ms = median(&mut solve_samples);
        let spmv_ms = median(&mut spmv_samples);
        let reduce_ms = median(&mut reduce_samples);
        let apply_ms = median(&mut apply_samples);
        let wait_ms = median(&mut wait_samples);
        let score = Score {
            label: if block_size == 0 { "jacobi" } else { "block" },
            block_size,
            setup_ms,
            solve_ms,
            iterations,
        };
        scores.push(score);
        if mpi.rank() == 0 {
            let mut f = OpenOptions::new().append(true).open(csv)?;
            writeln!(f, "{label},{policy_label},{expected},{},{},{name},{block_size},{warmups},{repeats},{iterations},{status:?},{relative:.12e},{abs_error:.12e},{spmv_calls},{allreduce_calls},{setup_ms:.6},{solve_ms:.6},{spmv_ms:.6},{reduce_ms:.6},{apply_ms:.6},{wait_ms:.6},{:.6},{:.6}", partition.global_dofs(), ingest.global_nnz, score.estimated_ms(1.0), score.estimated_ms(10.0))?;
            println!("PASS G8-C7 {label}/{name}[{block_size}] ranks={expected}: iters={iterations}, rel={relative:.3e}, setup={setup_ms:.4}ms, solve={solve_ms:.4}ms, apply={apply_ms:.4}ms, est_1rhs={:.4}ms, est_10rhs={:.4}ms", score.estimated_ms(1.0), score.estimated_ms(10.0));
        }
        mpi.barrier();
    }
    if mpi.rank() == 0 {
        // Rank selection uses the same cross-rank maximum metric for all options.
        let best_one = scores
            .iter()
            .min_by(|a, b| a.estimated_ms(1.0).total_cmp(&b.estimated_ms(1.0)))
            .ok_or("no benchmark candidates")?;
        let best_ten = scores
            .iter()
            .min_by(|a, b| a.estimated_ms(10.0).total_cmp(&b.estimated_ms(10.0)))
            .ok_or("no benchmark candidates")?;
        println!(
            "G8-C7 selection {label} ranks={expected}: 1RHS={}({}) {:.4}ms, 10RHS={}({}) {:.4}ms",
            best_one.label,
            best_one.block_size,
            best_one.estimated_ms(1.0),
            best_ten.label,
            best_ten.block_size,
            best_ten.estimated_ms(10.0)
        );
        let best_iters = scores.iter().map(|s| s.iterations).min().unwrap_or(0);
        println!("G8-C7 minimum iterations = {best_iters} (not necessarily fastest)");
        if policy == MtxPcgPolicy::ExperimentalUnverified {
            println!("G8-C7 WARNING: experimental policy does NOT certify global SPD");
        }
        println!("=== HYBIT 0.9 G8-C7 MPI {expected}-RANK BLOCK SWEEP PASS ===");
    }
    mpi.barrier();
    Ok(())
}
