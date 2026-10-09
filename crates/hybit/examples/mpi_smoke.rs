//! G8-B1 multi-process native MPI smoke test.
//! Launch e.g. `mpiexec -n 4 <compiled mpi_smoke executable> 4`.

use hybit::distributed::mpi_backend::MpiRuntime;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let expected: i32 = std::env::args()
        .nth(1)
        .ok_or("usage: mpi_smoke <expected_process_count>")?
        .parse()?;
    if expected < 1 {
        return Err("expected_process_count must be positive".into());
    }
    let mpi = MpiRuntime::initialize()?;
    let rank = mpi.rank();
    let size = mpi.size();
    if size != expected {
        return Err(format!(
            "MPI process count mismatch: launcher started {size}, expected {expected}"
        )
        .into());
    }
    let expected_rank_sum = (i64::from(size) * (i64::from(size) - 1)) / 2;
    let count = mpi.all_reduce_sum_u64(1);
    let rank_sum = mpi.all_reduce_sum_f64(f64::from(rank));
    if count != size as u64 || rank_sum != expected_rank_sum as f64 {
        return Err("MPI scalar allreduce failed".into());
    }
    let local = [f64::from(rank) + 1.0, 2.0];
    let mut global = [0.0f64; 2];
    mpi.all_reduce_sum_slice_f64(&local, &mut global)?;
    if global
        != [
            expected_rank_sum as f64 + f64::from(size),
            2.0 * f64::from(size),
        ]
    {
        return Err("MPI vector allreduce failed".into());
    }
    let previous = (rank + size - 1) % size;
    let from_previous = mpi.ring_exchange_i32(rank * 17 + 3);
    if from_previous != previous * 17 + 3 {
        return Err("MPI ring send/receive failed".into());
    }
    mpi.barrier();
    println!("G8-B1 rank={rank}/{size} sum={rank_sum} ring_from={previous} local_transport=PASS");
    mpi.barrier();
    if rank == 0 {
        println!("=== HYBIT 0.9 G8-B1 MPI {size}-RANK SMOKE PASS ===");
    }
    Ok(())
}
