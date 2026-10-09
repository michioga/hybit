//! G8-C4: diagnostic, repeatable rank-local MPI-PCG benchmark of Matrix Market inputs.
//! ExperimentalUnverified DOES NOT establish SPD. A non-converged solve is data,
//! not a successful solver validation. Never use an unverified result silently.
use hybit::distributed::mpi_backend::MpiRuntime;
use hybit::distributed::mpi_input::prepare_owned_rows;
use hybit::distributed::mpi_mtx::{distribute_matrix_market_with_policy, MtxPcgPolicy};
use hybit::distributed::mpi_pcg::{DistributedPcg, DistributedPcgOptions, DistributedPcgStatus};
use mpi::collective::SystemOperation;
use mpi::traits::*;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn max_f64(local: f64) -> f64 {
    let world = mpi::topology::SimpleCommunicator::world();
    let mut result = 0.0;
    world.all_reduce_into(&local, &mut result, SystemOperation::max());
    result
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
    let u = global as f64 + 1.0;
    (u * 0.017).sin() + 0.21 * (u * 0.0061).cos()
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 8 {
        return Err("usage: mpi_mtx_profile <ranks> <csv> <matrix.mtx> <label> <certified|experimental> <warmups> <repeats> <max_iters>".into());
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
    if repeats == 0 || repeats > 100 || warmups > 100 || max_iters == 0 || max_iters > 100_000 {
        return Err("invalid benchmark warmup/repeats/max_iters".into());
    }
    let mpi = MpiRuntime::initialize()?;
    if expected != mpi.size() || !matches!(expected, 1 | 2 | 4 | 8) {
        return Err("MPI rank mismatch; expected 1/2/4/8".into());
    }
    if mpi.rank() == 0 {
        let needs_header = !csv.exists() || std::fs::metadata(csv)?.len() == 0;
        if needs_header {
            let mut f = OpenOptions::new().create(true).append(true).open(csv)?;
            writeln!(f, "case,policy,ranks,n,nnz,warmups,repeats,iterations,status,true_relative_residual,max_abs_error,global_ghosts,root_read_ms,max_distribute_ms,max_halo_ms,max_pcg_prepare_ms,median_max_solve_ms,min_max_solve_ms,max_max_solve_ms,median_max_over_mean,spmv_calls,allreduce_calls")?;
        }
    }
    mpi.barrier();
    let (partition, owned, ingest) = distribute_matrix_market_with_policy(&mpi, matrix, policy)?;
    let start_halo = Instant::now();
    let (plan, local, stats) = prepare_owned_rows(&mpi, &partition, &owned)?;
    let halo_ms = max_f64(start_halo.elapsed().as_secs_f64() * 1e3);
    let start_pcg = Instant::now();
    let mut pcg = DistributedPcg::prepare(&mpi, &plan, &local)?;
    let prepare_ms = max_f64(start_pcg.elapsed().as_secs_f64() * 1e3);
    let read_ms = mpi.all_reduce_sum_f64(ingest.root_read_ns as f64 * 1e-6);
    let scatter_ms = max_f64(ingest.distribution_ns as f64 * 1e-6);
    let ghosts = mpi.all_reduce_sum_u64(stats.ghosts as u64);

    // A manufactured RHS uses only locally held rows and global column IDs.
    let mut b = vec![0.0; plan.owned_len()];
    for (row, rhs) in b.iter_mut().enumerate() {
        let first = owned.row_ptr[row] as usize;
        let last = owned.row_ptr[row + 1] as usize;
        for pos in first..last {
            *rhs += owned.values[pos] * truth(owned.global_col_idx[pos]);
        }
    }
    let options = DistributedPcgOptions {
        relative_tolerance: 1e-9,
        absolute_tolerance: 1e-13,
        max_iterations: max_iters,
    };
    let mut times = Vec::with_capacity(repeats);
    let mut imbalance = Vec::with_capacity(repeats);
    let mut iterations = 0;
    let mut residual = f64::INFINITY;
    let mut absolute_error = f64::INFINITY;
    let mut status = DistributedPcgStatus::MaxIterations;
    let mut spmv_calls = 0;
    let mut reductions = 0;
    for run in 0..(warmups + repeats) {
        let mut x = vec![0.0; plan.owned_len()];
        mpi.barrier();
        let report = pcg.solve(&mpi, &local, &b, &mut x, options)?;
        let local_ms = report.elapsed_ns as f64 * 1e-6;
        let max_ms = max_f64(local_ms);
        let mean_ms = mpi.all_reduce_sum_f64(local_ms) / f64::from(expected);
        let local_error = x
            .iter()
            .enumerate()
            .map(|(i, &v)| (v - truth(owned.owned.start + i as u64)).abs())
            .fold(0.0_f64, f64::max);
        let error = max_f64(local_error);
        if run >= warmups {
            times.push(max_ms);
            imbalance.push(if mean_ms > 0.0 { max_ms / mean_ms } else { 1.0 });
            iterations = report.iterations;
            residual = report.true_relative_residual;
            absolute_error = error;
            status = report.status;
            spmv_calls = report.spmv_calls;
            reductions = report.allreduce_calls;
        }
    }
    let min_ms = times.iter().copied().fold(f64::INFINITY, f64::min);
    let max_ms = times.iter().copied().fold(0.0_f64, f64::max);
    let median_ms = median(&mut times);
    let median_imbalance = median(&mut imbalance);
    // All ranks see the same global status/residual. Do not mislabel failures.
    let is_converged = status == DistributedPcgStatus::Converged
        && residual.is_finite()
        && residual <= 1.1e-9
        && absolute_error.is_finite()
        && absolute_error <= 2e-7;
    mpi.barrier();
    if mpi.rank() == 0 {
        let mut f = OpenOptions::new().append(true).open(csv)?;
        writeln!(f, "{label},{policy_name},{expected},{},{},{warmups},{repeats},{iterations},{status:?},{residual:.12e},{absolute_error:.12e},{ghosts},{read_ms:.6},{scatter_ms:.6},{halo_ms:.6},{prepare_ms:.6},{median_ms:.6},{min_ms:.6},{max_ms:.6},{median_imbalance:.6},{spmv_calls},{reductions}", partition.global_dofs(), ingest.global_nnz)?;
        let verdict = if is_converged {
            "CONVERGED"
        } else {
            "NOT_CONVERGED"
        };
        println!("G8-C4 {verdict} {label} policy={policy_name} ranks={expected} n={} iters={iterations} true_rel={residual:.3e} error={absolute_error:.3e} median_max_ms={median_ms:.4} imbalance={median_imbalance:.3}", partition.global_dofs());
        if policy == MtxPcgPolicy::ExperimentalUnverified {
            println!(
                "G8-C4 WARNING: experimental policy does not certify SPD, even if PCG converged"
            );
        }
    }
    mpi.barrier();
    // Certified benchmark is a correctness gate; experimental is diagnostic.
    if policy == MtxPcgPolicy::StrictCertified && !is_converged {
        return Err("certified SPD fixture PCG failed to converge accurately".into());
    }
    Ok(())
}
