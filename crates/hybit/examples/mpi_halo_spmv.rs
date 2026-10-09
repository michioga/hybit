//! G8-B2 native MPI halo/CSR cross-check. Run with mpiexec -n 1/2/4.
//!
//! Every rank constructs the same deterministic *reference* CSR for this test.
//! Only owned x values are passed to the MPI operator; ghosts arrive from peers.
//! Global matrix replication is a test harness, not a production input strategy.

use hybit::distributed::mpi_backend::MpiRuntime;
use hybit::distributed::{
    build_contiguous_halo_plans, prepare_rank_local_csr, ContiguousPartition,
};
use hybit::Csr32Matrix;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::Write;

fn matrix(n: usize, kind: &str) -> Csr32Matrix {
    let mut row_ptr = vec![0u32];
    let mut cols = Vec::new();
    let mut values = Vec::new();
    for row in 0..n {
        let mut entries = Vec::new();
        match kind {
            "chain" => {
                if row > 0 {
                    entries.push((row - 1, -1.0));
                }
                entries.push((row, 3.5));
                if row + 1 < n {
                    entries.push((row + 1, -1.0));
                }
            }
            "banded" => {
                for distance in (1..=3).rev() {
                    if row >= distance {
                        entries.push((row - distance, -0.25));
                    }
                }
                entries.push((row, 8.0));
                for distance in 1..=3 {
                    if row + distance < n {
                        entries.push((row + distance, -0.25));
                    }
                }
            }
            "directed" => {
                entries.push((row, 3.0));
                if row + 1 < n {
                    entries.push((row + 1, -0.5));
                }
            }
            "disconnected" => {
                if row % 4 != 0 {
                    entries.push((row - 1, -1.0));
                }
                entries.push((row, 3.0));
                if row % 4 != 3 {
                    entries.push((row + 1, -1.0));
                }
            }
            _ => unreachable!(),
        }
        for (col, value) in entries {
            cols.push(col as u32);
            values.push(value);
        }
        row_ptr.push(cols.len() as u32);
    }
    Csr32Matrix::new(n, n, row_ptr, cols, values).expect("valid reference CSR")
}

fn run_case(
    mpi: &MpiRuntime,
    case: &'static str,
    n: usize,
    csv_path: &str,
) -> Result<(), Box<dyn Error>> {
    let rank = mpi.rank() as usize;
    let size = mpi.size() as usize;
    let a = matrix(n, case);
    let partition = ContiguousPartition::balanced(n as u64, size as u32)?;
    let plans = build_contiguous_halo_plans(&a, &partition)?;
    let plan = &plans[rank];
    let local = prepare_rank_local_csr(&a, plan)?;
    let owned = plan.owned_range();
    let start = owned.start as usize;
    let end = owned.end as usize;
    let mut extended = vec![0.0; plan.extended_len()];
    let mut actual = vec![0.0; plan.owned_len()];
    let mut local_max_error = 0.0_f64;
    let mut sent = 0u64;
    let mut received = 0u64;
    let mut exchange_ns = 0u64;

    for trial in 0..3usize {
        let global_x: Vec<f64> = (0..n)
            .map(|i| {
                let seed = ((i + 1) * (trial + 3)) as f64;
                seed.sin() + 0.125 * seed.cos()
            })
            .collect();
        let expected = a.spmv(&global_x)?;
        let traffic = mpi.spmv_local(
            plan,
            &local,
            &global_x[start..end],
            &mut extended,
            &mut actual,
        )?;
        for (got, reference) in actual.iter().zip(&expected[start..end]) {
            local_max_error = local_max_error.max((got - reference).abs());
        }
        for (slot, &global) in plan.ghost_globals().iter().enumerate() {
            local_max_error = local_max_error
                .max((extended[plan.owned_len() + slot] - global_x[global as usize]).abs());
        }
        sent += traffic.sent_values as u64;
        received += traffic.received_values as u64;
        exchange_ns += u64::try_from(traffic.elapsed_ns)?;
    }

    // Exercise structural corner cases, without requiring symmetric matrices.
    let directed_oneway = plan
        .peers()
        .iter()
        .filter(|peer| peer.send_globals().is_empty() != peer.recv_globals().is_empty())
        .count() as u64;
    let any_oneway = mpi.all_reduce_sum_u64(directed_oneway);
    let peers = mpi.all_reduce_sum_u64(plan.peers().len() as u64);
    let ghost_count = mpi.all_reduce_sum_u64(plan.ghost_len() as u64);
    let remote_nnz = mpi.all_reduce_sum_u64(
        local
            .col_idx()
            .iter()
            .filter(|&&j| j as usize >= plan.owned_len())
            .count() as u64,
    );
    let global_error_sum = mpi.all_reduce_sum_f64(local_max_error);
    let global_sent = mpi.all_reduce_sum_u64(sent);
    let global_recv = mpi.all_reduce_sum_u64(received);
    let mean_exchange_ms = mpi.all_reduce_sum_f64(exchange_ns as f64 / 3e6) / size as f64;

    if global_sent != global_recv {
        return Err(format!("{case}: global sent/received values differ").into());
    }
    if global_error_sum > 1.0e-12 {
        return Err(format!("{case}: serial vs MPI CSR mismatch {global_error_sum:e}").into());
    }
    if case == "directed" && size > 1 && any_oneway == 0 {
        return Err("directed test did not exercise one-way peer traffic".into());
    }
    if case == "disconnected" && (ghost_count != 0 || peers != 0) {
        return Err("disconnected test unexpectedly required a halo".into());
    }
    if case == "banded" && size > 1 && remote_nnz <= ghost_count {
        return Err("banded test did not exercise duplicated remote references".into());
    }
    mpi.barrier();
    if mpi.rank() == 0 {
        let mut csv = OpenOptions::new().append(true).open(csv_path)?;
        writeln!(
            csv,
            "{case},{size},{n},{ghost_count},{remote_nnz},{any_oneway},{global_sent},{global_recv},{global_error_sum:.12e},{mean_exchange_ms:.9}",
        )?;
        println!(
            "PASS B2 {case}: ranks={size} n={n} ghosts={ghost_count} remote_nnz={remote_nnz} one_way={any_oneway} sent={global_sent} recv={global_recv} error_sum={global_error_sum:.3e} avg_exchange_ms={mean_exchange_ms:.6}",
        );
    }
    mpi.barrier();
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let expected_ranks: i32 = std::env::args()
        .nth(1)
        .ok_or("usage: mpi_halo_spmv <expected_ranks> <csv_file>")?
        .parse()?;
    let csv_path = std::env::args()
        .nth(2)
        .ok_or("usage: mpi_halo_spmv <expected_ranks> <csv_file>")?;
    let mpi = MpiRuntime::initialize()?;
    if mpi.size() != expected_ranks || !matches!(mpi.size(), 1 | 2 | 4) {
        return Err("MPI count must match the expected 1, 2, or 4 ranks".into());
    }
    if mpi.rank() == 0 {
        let mut file = File::create(&csv_path)?;
        writeln!(file, "case,ranks,n,ghosts,remote_nnz,oneway_peer_links,sent_values,received_values,error_sum,avg_exchange_ms")?;
    }
    mpi.barrier();
    for (kind, n) in [
        ("chain", 19),
        ("banded", 24),
        ("directed", 20),
        ("disconnected", 16),
    ] {
        run_case(&mpi, kind, n, &csv_path)?;
    }
    if mpi.rank() == 0 {
        println!(
            "=== HYBIT 0.9 G8-B2 NATIVE MPI {}-RANK HALO/SPMV PASS ===",
            mpi.size()
        );
    }
    mpi.barrier();
    Ok(())
}
