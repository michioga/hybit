//! G8-B3 native MPI overlap vs G8-B2 blocking reference, 1/2/4 processes.
//! Global matrix replication is ONLY a deterministic test harness.

use hybit::distributed::mpi_backend::MpiRuntime;
use hybit::distributed::mpi_overlap::OverlapSpmv;
use hybit::distributed::{
    build_contiguous_halo_plans, prepare_rank_local_csr, ContiguousPartition,
};
use hybit::Csr32Matrix;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::time::Instant;

fn matrix(n: usize, kind: &str) -> Csr32Matrix {
    let mut ptr = vec![0u32];
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..n {
        let mut entries = Vec::new();
        match kind {
            "chain" => {
                if i > 0 {
                    entries.push((i - 1, -1.0));
                }
                entries.push((i, 3.5));
                if i + 1 < n {
                    entries.push((i + 1, -1.0));
                }
            }
            "banded" => {
                for d in (1..=3).rev() {
                    if i >= d {
                        entries.push((i - d, -0.25));
                    }
                }
                entries.push((i, 8.0));
                for d in 1..=3 {
                    if i + d < n {
                        entries.push((i + d, -0.25));
                    }
                }
            }
            "directed" => {
                entries.push((i, 3.0));
                if i + 1 < n {
                    entries.push((i + 1, -0.5));
                }
            }
            "disconnected" => {
                if i % 4 != 0 {
                    entries.push((i - 1, -1.0));
                }
                entries.push((i, 3.0));
                if i % 4 != 3 {
                    entries.push((i + 1, -1.0));
                }
            }
            _ => unreachable!(),
        }
        for (col, val) in entries {
            cols.push(col as u32);
            vals.push(val);
        }
        ptr.push(cols.len() as u32);
    }
    Csr32Matrix::new(n, n, ptr, cols, vals).expect("valid reference CSR")
}

fn run_case(
    mpi: &MpiRuntime,
    kind: &'static str,
    n: usize,
    csv: &str,
) -> Result<(), Box<dyn Error>> {
    let size = mpi.size() as usize;
    let rank = mpi.rank() as usize;
    let a = matrix(n, kind);
    let ownership = ContiguousPartition::balanced(n as u64, size as u32)?;
    let plans = build_contiguous_halo_plans(&a, &ownership)?;
    let plan = &plans[rank];
    let local = prepare_rank_local_csr(&a, plan)?;
    let start = plan.owned_range().start as usize;
    let end = plan.owned_range().end as usize;
    let prepared_start = Instant::now();
    let mut overlap = OverlapSpmv::prepare(mpi, plan, &local)?;
    let prepare_ms = prepared_start.elapsed().as_secs_f64() * 1.0e3;
    let mut blocking_extended = vec![0.0; plan.extended_len()];
    let mut blocking_y = vec![0.0; plan.owned_len()];
    let mut async_y = vec![0.0; plan.owned_len()];
    let mut max_error = 0.0f64;
    let mut block_ns = 0u128;
    let mut async_ns = 0u128;
    let mut interior_ns = 0u128;
    let mut wait_ns = 0u128;
    let mut post_ns = 0u128;
    let mut sent = 0u64;
    let mut recv = 0u64;
    const TRIALS: usize = 5;
    for trial in 0..TRIALS {
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let seed = ((i + 1) * (trial + 3)) as f64;
                seed.sin() + 0.125 * seed.cos()
            })
            .collect();
        let reference = a.spmv(&x)?;
        let t_block = Instant::now();
        let blocked = mpi.spmv_local(
            plan,
            &local,
            &x[start..end],
            &mut blocking_extended,
            &mut blocking_y,
        )?;
        block_ns += t_block.elapsed().as_nanos();
        let timed = overlap.spmv_overlap(mpi, &local, &x[start..end], &mut async_y)?;
        async_ns += timed.elapsed_ns;
        interior_ns += timed.interior_ns;
        post_ns += timed.post_ns;
        wait_ns += timed.wait_ns;
        sent += timed.send_values as u64;
        recv += timed.recv_values as u64;
        if blocked.sent_values != timed.send_values || blocked.received_values != timed.recv_values
        {
            return Err("MPI blocking and overlap transfer volumes differ".into());
        }
        for ((&b, &v), &r) in blocking_y.iter().zip(&async_y).zip(&reference[start..end]) {
            if !b.is_finite() || !v.is_finite() || !r.is_finite() {
                max_error = f64::INFINITY;
            } else {
                max_error = max_error
                    .max((b - v).abs())
                    .max((v - r).abs())
                    .max((b - r).abs());
            }
        }
        for (slot, &global) in plan.ghost_globals().iter().enumerate() {
            let idx = plan.owned_len() + slot;
            let expected = x[global as usize];
            let async_ghost = overlap.extended()[idx];
            let blocked_ghost = blocking_extended[idx];
            if !async_ghost.is_finite() || !blocked_ghost.is_finite() {
                max_error = f64::INFINITY;
            } else {
                max_error = max_error
                    .max((async_ghost - expected).abs())
                    .max((blocked_ghost - expected).abs());
            }
        }
    }
    let ghost = mpi.all_reduce_sum_u64(plan.ghost_len() as u64);
    let interior = mpi.all_reduce_sum_u64(overlap.interior_len() as u64);
    let boundary = mpi.all_reduce_sum_u64(overlap.boundary_len() as u64);
    let global_send = mpi.all_reduce_sum_u64(sent);
    let global_recv = mpi.all_reduce_sum_u64(recv);
    let error_sum = mpi.all_reduce_sum_f64(max_error);
    let avg_block_ms = mpi.all_reduce_sum_f64(block_ns as f64 * 1e-6 / TRIALS as f64) / size as f64;
    let avg_async_ms = mpi.all_reduce_sum_f64(async_ns as f64 * 1e-6 / TRIALS as f64) / size as f64;
    let avg_wait_ms = mpi.all_reduce_sum_f64(wait_ns as f64 * 1e-6 / TRIALS as f64) / size as f64;
    let avg_interior_ms =
        mpi.all_reduce_sum_f64(interior_ns as f64 * 1e-6 / TRIALS as f64) / size as f64;
    let avg_post_ms = mpi.all_reduce_sum_f64(post_ns as f64 * 1e-6 / TRIALS as f64) / size as f64;
    let avg_prepare_ms = mpi.all_reduce_sum_f64(prepare_ms) / size as f64;
    if global_send != global_recv || interior + boundary != n as u64 || error_sum > 1e-12 {
        return Err(format!("B3 failed {kind}: sent={global_send} recv={global_recv} interior={interior} boundary={boundary} error={error_sum:e}").into());
    }
    if kind == "disconnected" && ghost != 0 {
        return Err("disconnected case unexpectedly has ghosts".into());
    }
    if kind != "disconnected" && size > 1 && boundary == 0 {
        return Err("communicating case lacks boundary rows".into());
    }
    mpi.barrier();
    if mpi.rank() == 0 {
        let mut f = OpenOptions::new().append(true).open(csv)?;
        writeln!(f,"{kind},{size},{n},{interior},{boundary},{ghost},{global_send},{global_recv},{error_sum:.12e},{avg_prepare_ms:.6},{avg_block_ms:.6},{avg_async_ms:.6},{avg_post_ms:.6},{avg_interior_ms:.6},{avg_wait_ms:.6}")?;
        println!("PASS B3 {kind} ranks={size} n={n} interior={interior} boundary={boundary} ghost={ghost} err={error_sum:.3e} blocking={avg_block_ms:.6}ms overlap={avg_async_ms:.6}ms wait={avg_wait_ms:.6}ms");
    }
    mpi.barrier();
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let expected: i32 = args
        .next()
        .ok_or("usage: mpi_overlap_spmv <ranks> <csv>")?
        .parse()?;
    let path = args.next().ok_or("usage: mpi_overlap_spmv <ranks> <csv>")?;
    let mpi = MpiRuntime::initialize()?;
    if !matches!(mpi.size(), 1 | 2 | 4) || expected != mpi.size() {
        return Err("MPI rank count must equal requested 1/2/4".into());
    }
    if mpi.rank() == 0 {
        let mut f = File::create(&path)?;
        writeln!(f,"case,ranks,n,interior_rows,boundary_rows,ghosts,sent_values,recv_values,error_sum,prepare_ms,blocking_spmv_ms,overlap_spmv_ms,post_ms,interior_spmv_ms,wait_ms")?;
    }
    mpi.barrier();
    for (kind, n) in [
        ("chain", 131),
        ("banded", 128),
        ("directed", 120),
        ("disconnected", 128),
    ] {
        run_case(&mpi, kind, n, &path)?;
    }
    mpi.barrier();
    if mpi.rank() == 0 {
        println!(
            "=== HYBIT 0.9 G8-B3 MPI {}-RANK OVERLAP/SPMV PASS ===",
            mpi.size()
        );
    }
    Ok(())
}
