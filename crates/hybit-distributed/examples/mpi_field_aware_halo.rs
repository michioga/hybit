//! H1-C2 native MS-MPI regression: block-aligned contiguous ownership
//! versus the existing owned-CSR HaloPlan transport, with CSR SpMV truth.
//!
//! A replicated global CSR is used ONLY as a tiny deterministic test fixture.
//! Production input remains rank-owned and uses mpi_input::prepare_owned_rows.

use hybit_distributed::field_aligned_contiguous_partition;
use hybit_distributed::mpi_backend::MpiRuntime;
use hybit_distributed::mpi_input::{OwnedCsrRows, prepare_owned_rows};
use hybit_distributed::{
    ContiguousPartition, PreparedFieldAwareHalo, build_contiguous_halo_plans, partition_telemetry,
};
use hybit_matrix::{Csr32Matrix, DofBlock, DofEntityKind, FieldAwareDofLayout};
use std::error::Error;
use std::time::Instant;

fn fixture() -> (FieldAwareDofLayout, Csr32Matrix) {
    let mut blocks = Vec::new();
    for i in 0..64u64 {
        let components = [1u32, 3, 6, 5, 2, 4, 3, 1][i as usize % 8];
        blocks.push(DofBlock {
            entity_kind: if i % 8 == 7 {
                DofEntityKind::Multiplier
            } else {
                DofEntityKind::Node
            },
            entity_id: i / 4,
            field_id: (i % 4) as u32,
            components,
        });
    }
    let layout = FieldAwareDofLayout::new(blocks).expect("valid mixed-DOF fixture");
    let n = layout.total_dofs() as usize;
    let mut row_ptr = vec![0u32];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for row in 0..n {
        for (col, value) in [
            (row, 4.5),
            ((row + 1) % n, -0.5),
            ((row + 11) % n, if row % 7 == 0 { 0.0 } else { 0.25 }),
        ] {
            col_idx.push(col as u32);
            values.push(value);
        }
        row_ptr.push(values.len() as u32);
    }
    let matrix = Csr32Matrix::new(n, n, row_ptr, col_idx, values).unwrap();
    (layout, matrix)
}

fn owned_rows(matrix: &Csr32Matrix, owned: std::ops::Range<u64>) -> OwnedCsrRows {
    let first = owned.start as usize;
    let last = owned.end as usize;
    let nnz_first = matrix.row_ptr()[first] as usize;
    let nnz_last = matrix.row_ptr()[last] as usize;
    let row_ptr = matrix.row_ptr()[first..=last]
        .iter()
        .map(|&p| p - matrix.row_ptr()[first])
        .collect();
    OwnedCsrRows {
        global_dofs: matrix.nrows() as u64,
        owned,
        row_ptr,
        global_col_idx: matrix.col_idx()[nnz_first..nnz_last]
            .iter()
            .map(|&col| u64::from(col))
            .collect(),
        values: matrix.values()[nnz_first..nnz_last].to_vec(),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let expected: i32 = std::env::args()
        .nth(1)
        .ok_or("usage: mpi_field_aware_halo <ranks>")?
        .parse()?;
    let mpi = MpiRuntime::initialize()?;
    if mpi.size() != expected || !matches!(mpi.size(), 1 | 2 | 4 | 8) {
        return Err("expected exactly 1/2/4/8 MS-MPI ranks".into());
    }
    let (layout, matrix) = fixture();
    let ranks = mpi.size() as u32;
    let rank = mpi.rank() as usize;
    let plan_start = Instant::now();
    let prepared = PreparedFieldAwareHalo::prepare(&matrix, &layout, ranks)?;
    let plan_ms = plan_start.elapsed().as_secs_f64() * 1.0e3;
    let partition = field_aligned_contiguous_partition(&layout, ranks)?;
    if prepared.partition() != &partition {
        return Err("inconsistent field partition".into());
    }
    let native_owned = owned_rows(&matrix, partition.owned_range(rank as u32)?);
    let owned_prepare_start = Instant::now();
    let (halo, local, stats) = prepare_owned_rows(&mpi, &partition, &native_owned)?;
    let owned_prepare_ms = owned_prepare_start.elapsed().as_secs_f64() * 1.0e3;
    let reference_halo = &prepared.plans()[rank];
    if &halo != reference_halo
        || local.row_ptr() != prepared.local_operators()[rank].row_ptr()
        || local.col_idx() != prepared.local_operators()[rank].col_idx()
        || local.values() != prepared.local_operators()[rank].values()
    {
        return Err("native owned-CSR and global-reference halo/CSR plans differ".into());
    }

    let mut extended = vec![0.0; halo.extended_len()];
    let mut output = vec![0.0; halo.owned_len()];
    let mut local_max_error = 0.0f64;
    let mut time_ns = 0u128;
    let mut sent = 0usize;
    let mut received = 0usize;
    const TRIALS: usize = 5;
    for trial in 0..TRIALS {
        let x: Vec<f64> = (0..matrix.nrows())
            .map(|i| ((i + 1 + trial) as f64 * 0.019).sin())
            .collect();
        let truth = matrix.spmv(&x)?;
        let start = Instant::now();
        let traffic = mpi.spmv_local(
            &halo,
            &local,
            &x[native_owned.owned.start as usize..native_owned.owned.end as usize],
            &mut extended,
            &mut output,
        )?;
        time_ns += start.elapsed().as_nanos();
        sent += traffic.sent_values;
        received += traffic.received_values;
        for (&actual, &correct) in output
            .iter()
            .zip(&truth[native_owned.owned.start as usize..native_owned.owned.end as usize])
        {
            local_max_error = local_max_error.max((actual - correct).abs());
        }
        for (slot, &global) in halo.ghost_globals().iter().enumerate() {
            local_max_error =
                local_max_error.max((extended[halo.owned_len() + slot] - x[global as usize]).abs());
        }
    }
    let total_err = mpi.all_reduce_sum_f64(local_max_error);
    let global_send = mpi.all_reduce_sum_u64(sent as u64);
    let global_receive = mpi.all_reduce_sum_u64(received as u64);
    let global_ghosts = mpi.all_reduce_sum_u64(stats.ghosts as u64);
    let mean_rank_spmv_ms =
        mpi.all_reduce_sum_f64(time_ns as f64 * 1e-6 / TRIALS as f64) / f64::from(ranks);
    let mean_rank_refplan_ms = mpi.all_reduce_sum_f64(plan_ms) / f64::from(ranks);
    let mean_rank_owned_setup_ms = mpi.all_reduce_sum_f64(owned_prepare_ms) / f64::from(ranks);
    let sum_owned = mpi.all_reduce_sum_u64(stats.owned_rows as u64);
    let prepared_stats = prepared.telemetry();
    let baseline = ContiguousPartition::balanced(layout.total_dofs(), ranks)?;
    let base_halo = build_contiguous_halo_plans(&matrix, &baseline)?;
    let baseline_stats = partition_telemetry(&matrix, &baseline, &base_halo)?;
    let pass = total_err <= 1e-12
        && global_send == global_receive
        && global_ghosts as usize == prepared_stats.communication_volume
        && sum_owned == layout.total_dofs();
    if mpi.all_reduce_sum_u64(u64::from(!pass)) != 0 {
        return Err(format!("H1-C2 FAIL ranks={ranks} error={total_err:e}").into());
    }
    mpi.barrier();
    if rank == 0 {
        println!(
            "PASS H1-C2 ranks={ranks} n={} nnz={} ghosts={} baseline_ghosts={} peers={} baseline_peers={} owned_imbalance={:.3} baseline_owned_imbalance={:.3} nnz_imbalance={:.3} baseline_nnz_imbalance={:.3} send_values={} recv_values={} abs_err={total_err:.3e} mean_rank_spmv_ms={mean_rank_spmv_ms:.6} mean_rank_reference_plan_ms={mean_rank_refplan_ms:.6} mean_rank_native_owned_setup_ms={mean_rank_owned_setup_ms:.6}",
            layout.total_dofs(),
            matrix.nnz(),
            prepared_stats.communication_volume,
            baseline_stats.communication_volume,
            prepared_stats.directional_peer_relations,
            baseline_stats.directional_peer_relations,
            prepared_stats.owned_dof_imbalance,
            baseline_stats.owned_dof_imbalance,
            prepared_stats.local_nnz_imbalance,
            baseline_stats.local_nnz_imbalance,
            global_send,
            global_receive,
        );
    }
    mpi.barrier();
    Ok(())
}
