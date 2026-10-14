//! H1-C3 MS-MPI owned-CSR SpMV regression; the global CSR is a small test
//! fixture ONLY. Actual SpMV calls the unchanged native MS-MPI owned-Halo path.
//! Three policies are compared in reference telemetry: DOF-balanced,
//! H1-C2 field-aligned, and H1-C3 graph-refined contiguous partitions.

use hybit_distributed::mpi_backend::MpiRuntime;
use hybit_distributed::mpi_input::{prepare_owned_rows, OwnedCsrRows};
use hybit_distributed::{
    build_contiguous_halo_plans, field_aligned_contiguous_partition, partition_telemetry,
    ContiguousPartition, PreparedFieldGraphHalo,
};
use hybit_matrix::{Csr32Matrix, DofBlock, DofEntityKind, FieldAwareDofLayout};
use std::error::Error;
use std::time::Instant;

fn fixture() -> (FieldAwareDofLayout, Csr32Matrix) {
    let mut blocks = Vec::new();
    for i in 0..64u64 {
        let components = [1, 3, 6, 5, 2, 4, 3, 1][i as usize % 8];
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
    let layout = FieldAwareDofLayout::new(blocks).unwrap();
    let n = layout.total_dofs() as usize;
    let starts: Vec<usize> = (0..layout.blocks().len())
        .map(|i| layout.block_range(i).unwrap().start as usize)
        .collect();
    let bounds = [0usize, 15, 32, 48, 64];
    let mut row_ptr = vec![0u32];
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for row in 0..n {
        let block = layout.locate(row as u64).unwrap().block_index;
        let cluster = bounds
            .windows(2)
            .position(|range| block >= range[0] && block < range[1])
            .unwrap();
        let first = starts[bounds[cluster]];
        let last = if bounds[cluster + 1] == 64 {
            n
        } else {
            starts[bounds[cluster + 1]]
        };
        // Multiple rows depend on the same remote reference after partition.
        // Stored numeric zeros retain symbolic/Halo dependencies.
        for (col, value) in [
            (row, 5.0),
            (first, -0.3),
            (first + 1, -0.2),
            (if row + 1 < last { row + 1 } else { first }, -0.4),
            (
                if row > first { row - 1 } else { last - 1 },
                if row % 9 == 0 { 0.0 } else { -0.15 },
            ),
        ] {
            cols.push(col as u32);
            vals.push(value);
        }
        row_ptr.push(vals.len() as u32);
    }
    (layout, Csr32Matrix::new(n, n, row_ptr, cols, vals).unwrap())
}

fn owned_rows(matrix: &Csr32Matrix, owned: std::ops::Range<u64>) -> OwnedCsrRows {
    let first = owned.start as usize;
    let last = owned.end as usize;
    let nnz_first = matrix.row_ptr()[first] as usize;
    let nnz_last = matrix.row_ptr()[last] as usize;
    OwnedCsrRows {
        global_dofs: matrix.nrows() as u64,
        owned,
        row_ptr: matrix.row_ptr()[first..=last]
            .iter()
            .map(|&p| p - matrix.row_ptr()[first])
            .collect(),
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
        .ok_or("usage: mpi_field_partition_compare <ranks>")?
        .parse()?;
    let mpi = MpiRuntime::initialize()?;
    if mpi.size() != expected || !matches!(expected, 1 | 2 | 4 | 8) {
        return Err("expected native MS-MPI ranks 1/2/4/8".into());
    }
    let ranks = expected as u32;
    let rank = mpi.rank() as usize;
    let (layout, matrix) = fixture();
    let start = Instant::now();
    let optimized = PreparedFieldGraphHalo::prepare(&matrix, &layout, ranks, 5)?;
    let graph_prepare_ms = start.elapsed().as_secs_f64() * 1e3;
    let equal = ContiguousPartition::balanced(layout.total_dofs(), ranks)?;
    let equal_halos = build_contiguous_halo_plans(&matrix, &equal)?;
    let equal_stats = partition_telemetry(&matrix, &equal, &equal_halos)?;
    let field = field_aligned_contiguous_partition(&layout, ranks)?;
    let field_halos = build_contiguous_halo_plans(&matrix, &field)?;
    let field_stats = partition_telemetry(&matrix, &field, &field_halos)?;
    let selected = optimized.comparison();
    if field_stats != selected.baseline {
        return Err("graph partition field baseline mismatch".into());
    }
    let partition = optimized.partition();
    let rows = owned_rows(&matrix, partition.owned_range(rank as u32)?);
    let prepare_started = Instant::now();
    let (halo, local, stats) = prepare_owned_rows(&mpi, partition, &rows)?;
    let native_setup_ms = prepare_started.elapsed().as_secs_f64() * 1e3;
    let reference_plan = &optimized.plans()[rank];
    if &halo != reference_plan
        || local.row_ptr() != optimized.local_operators()[rank].row_ptr()
        || local.col_idx() != optimized.local_operators()[rank].col_idx()
        || local.values() != optimized.local_operators()[rank].values()
    {
        return Err("MS-MPI owned HaloPlan differs from global CSR reference".into());
    }
    let mut extended = vec![0.0; halo.extended_len()];
    let mut y_owned = vec![0.0; halo.owned_len()];
    let mut error = 0.0f64;
    let mut time_ns = 0u128;
    let mut sent = 0u64;
    let mut received = 0u64;
    const TRIALS: usize = 5;
    for trial in 0..TRIALS {
        let x: Vec<f64> = (0..matrix.nrows())
            .map(|i| ((i + trial + 1) as f64 * 0.027).sin())
            .collect();
        let reference = matrix.spmv(&x)?;
        let started = Instant::now();
        let traffic = mpi.spmv_local(
            &halo,
            &local,
            &x[rows.owned.start as usize..rows.owned.end as usize],
            &mut extended,
            &mut y_owned,
        )?;
        time_ns += started.elapsed().as_nanos();
        sent += traffic.sent_values as u64;
        received += traffic.received_values as u64;
        for (&actual, &correct) in y_owned
            .iter()
            .zip(&reference[rows.owned.start as usize..rows.owned.end as usize])
        {
            error = error.max((actual - correct).abs());
        }
        for (slot, &global) in halo.ghost_globals().iter().enumerate() {
            error = error.max((extended[halo.owned_len() + slot] - x[global as usize]).abs());
        }
    }
    let total_error = mpi.all_reduce_sum_f64(error);
    let total_sent = mpi.all_reduce_sum_u64(sent);
    let total_recv = mpi.all_reduce_sum_u64(received);
    let total_ghost = mpi.all_reduce_sum_u64(stats.ghosts as u64);
    let total_owned = mpi.all_reduce_sum_u64(stats.owned_rows as u64);
    let mean_spmv_ms = mpi.all_reduce_sum_f64(time_ns as f64 * 1e-6 / TRIALS as f64) / ranks as f64;
    let mean_graph_prepare_ms = mpi.all_reduce_sum_f64(graph_prepare_ms) / ranks as f64;
    let mean_native_setup_ms = mpi.all_reduce_sum_f64(native_setup_ms) / ranks as f64;
    let pass = total_error <= 1e-12
        && total_sent == total_recv
        && total_recv == total_ghost * TRIALS as u64
        && total_ghost as usize == selected.selected.communication_volume
        && total_owned == layout.total_dofs()
        && selected.selected.communication_volume <= selected.baseline.communication_volume
        && selected.selected.cut_nnz <= selected.baseline.cut_nnz
        && selected.selected.directional_peer_relations
            <= selected.baseline.directional_peer_relations;
    if mpi.all_reduce_sum_u64(u64::from(!pass)) != 0 {
        return Err(format!("H1-C3 MPI numerical/traffic mismatch: err={total_error:.3e}").into());
    }
    mpi.barrier();
    if rank == 0 {
        println!(
            "PASS H1-C3 ranks={ranks} n={} nnz={} accepted={} ghosts={} field_ghosts={} equal_ghosts={} peers={} field_peers={} equal_peers={} cut={} field_cut={} equal_cut={} owned_imbalance={:.3} field_owned_imbalance={:.3} nnz_imbalance={:.3} field_nnz_imbalance={:.3} abs_err={total_error:.3e} sent={total_sent} recv={total_recv} mean_rank_spmv_ms={mean_spmv_ms:.6} mean_rank_graph_prepare_ms={mean_graph_prepare_ms:.6} mean_rank_native_setup_ms={mean_native_setup_ms:.6}",
            layout.total_dofs(), matrix.nnz(), selected.candidate_accepted,
            selected.selected.communication_volume, field_stats.communication_volume,
            equal_stats.communication_volume,
            selected.selected.directional_peer_relations, field_stats.directional_peer_relations,
            equal_stats.directional_peer_relations,
            selected.selected.cut_nnz, field_stats.cut_nnz, equal_stats.cut_nnz,
            selected.selected.owned_dof_imbalance, field_stats.owned_dof_imbalance,
            selected.selected.local_nnz_imbalance, field_stats.local_nnz_imbalance,
        );
    }
    mpi.barrier();
    Ok(())
}
