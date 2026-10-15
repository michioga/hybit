//! H1-C3E native MS-MPI: only root owns full CSR and executes the selector.
//! Non-root ranks construct ONLY their rank-owned CSR rows after Bcast.
//! This synthetic smoke is not a real distributed mesh reader or speed test.

use hybit_distributed::mpi_backend::MpiRuntime;
use hybit_distributed::mpi_input::{prepare_owned_rows, OwnedCsrRows};
use hybit_distributed::{
    broadcast_root_cost_selection_mpi, capture_partition_calibration,
    cost_aware_contiguous_partition_scoped, select_and_broadcast_cost_partition_mpi,
    ContiguousPartition, CostAwarePartitionOptions, DistributedTopologyError,
    PartitionCostCalibration, ScopedCostPartitionResult,
};
use hybit_matrix::{Csr32Matrix, DofBlock, DofEntityKind, FieldAwareDofLayout};
use std::error::Error;

fn layout(changed_metadata: bool) -> Result<FieldAwareDofLayout, Box<dyn Error>> {
    let blocks = (0..64u64)
        .map(|i| DofBlock {
            entity_kind: DofEntityKind::Node,
            entity_id: i,
            field_id: u32::from(changed_metadata && i == 0),
            components: [3, 1, 2, 4][i as usize % 4],
        })
        .collect();
    Ok(FieldAwareDofLayout::new(blocks)?)
}

fn entries(row: usize, n: usize) -> [(u32, f64); 3] {
    [
        (row as u32, 4.0),
        (((row + 1) % n) as u32, -0.5),
        (((row + n - 1) % n) as u32, -0.5),
    ]
}

fn root_matrix(n: usize) -> Result<Csr32Matrix, Box<dyn Error>> {
    let mut row_ptr = vec![0u32];
    let mut cols = Vec::with_capacity(3 * n);
    let mut values = Vec::with_capacity(3 * n);
    for row in 0..n {
        for (col, val) in entries(row, n) {
            cols.push(col);
            values.push(val);
        }
        row_ptr.push(cols.len() as u32);
    }
    Ok(Csr32Matrix::new(n, n, row_ptr, cols, values)?)
}

fn rank_owned_rows(n: usize, begin: usize, end: usize) -> OwnedCsrRows {
    let mut row_ptr = vec![0u32];
    let mut cols = Vec::new();
    let mut values = Vec::new();
    for row in begin..end {
        for (col, value) in entries(row, n) {
            cols.push(u64::from(col));
            values.push(value);
        }
        row_ptr.push(cols.len() as u32);
    }
    OwnedCsrRows {
        global_dofs: n as u64,
        owned: begin as u64..end as u64,
        row_ptr,
        global_col_idx: cols,
        values,
    }
}

fn corrupt_boundary(
    selection: &mut ScopedCostPartitionResult,
) -> Result<(), DistributedTopologyError> {
    let mut cuts = selection.partition.offsets().to_vec();
    // First physical field block has 3 DOFs: cut 1 is valid contiguous
    // ownership but INVALID against the physical field boundary.
    cuts[1] = 1;
    selection.partition = ContiguousPartition::from_offsets(cuts)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.len() > 3 {
        return Err("usage: mpi_root_cost_partition_smoke <ranks> [--mismatch-env|--mismatch-options|--mismatch-layout|--root-error|--invalid-offset]".into());
    }
    let expected: i32 = args[1].parse()?;
    let mode = args.get(2).map_or("", String::as_str);
    if !matches!(
        mode,
        "" | "--mismatch-env"
            | "--mismatch-options"
            | "--mismatch-layout"
            | "--root-error"
            | "--invalid-offset"
    ) {
        return Err("unknown fault injection mode".into());
    }
    let negative = !mode.is_empty();
    let mpi = MpiRuntime::initialize()?;
    if mpi.size() != expected || !matches!(expected, 1 | 2 | 4 | 8) {
        return Err("expected native ranks 1,2,4,8".into());
    }
    if negative && expected < 2 {
        return Err("negative tests require at least two ranks".into());
    }
    let rank = mpi.rank();
    let local_layout = layout(mode == "--mismatch-layout" && rank == 1)?;
    let n = local_layout.total_dofs() as usize;
    let mut options = CostAwarePartitionOptions {
        expected_spmv_calls: 1000,
        search_radius_blocks: 3,
        max_owned_dof_imbalance: 1.20,
        max_local_nnz_imbalance: 1.20,
        ..CostAwarePartitionOptions::default()
    };
    if mode == "--mismatch-options" && rank == 1 {
        options.expected_spmv_calls += 1;
    }
    let environment = if mode == "--mismatch-env" && rank == 1 {
        "H1C3E-injected-difference"
    } else {
        "H1C3E-fixture-msmpi-v1"
    };

    // Large global matrix is owned solely by rank 0. All ranks do retain
    // global field metadata, as required by the current field alignment API.
    let global = if rank == 0 {
        Some(root_matrix(n)?)
    } else {
        None
    };
    let timings = PartitionCostCalibration {
        baseline_spmv_ns: 1000,
        graph_spmv_ns: 900,
        graph_prepare_budget_ns: 100_000,
    };
    let calibration = if rank == 0 {
        Some(capture_partition_calibration(
            global.as_ref().expect("root has CSR"),
            &local_layout,
            expected as u32,
            options,
            environment,
            timings,
        )?)
    } else {
        None
    };

    let distributed = if mode == "--root-error" || mode == "--invalid-offset" {
        // Exercise lower-level coordinated error / malformed offset paths.
        // The actual selection remains root-only; workers pass None.
        let root_selection: Option<Result<ScopedCostPartitionResult, DistributedTopologyError>> =
            if rank == 0 {
                let mut result = cost_aware_contiguous_partition_scoped(
                    global.as_ref().expect("root CSR"),
                    &local_layout,
                    expected as u32,
                    options,
                    environment,
                    calibration.as_ref(),
                );
                if mode == "--root-error" {
                    result = Err(DistributedTopologyError::InvalidPartition(
                        "injected root selector error",
                    ));
                } else if let Ok(ref mut selection) = result {
                    corrupt_boundary(selection)?;
                }
                Some(result)
            } else {
                None
            };
        broadcast_root_cost_selection_mpi(
            root_selection.as_ref(),
            &local_layout,
            options,
            environment,
        )
    } else {
        select_and_broadcast_cost_partition_mpi(
            global.as_ref(),
            &local_layout,
            options,
            environment,
            calibration.as_ref(),
        )
    };

    if negative {
        if distributed.is_ok() {
            return Err("injected divergence unexpectedly accepted".into());
        }
        mpi.barrier();
        if rank == 0 {
            println!("PASS H1-C3E ranks={expected} injection={mode} rejected_collectively=true");
        }
        return Ok(());
    }
    let selected = distributed?;
    let own = selected.partition.owned_range(rank as u32)?;
    let rows = rank_owned_rows(n, own.start as usize, own.end as usize);
    let (halo, operator, stats) = prepare_owned_rows(&mpi, &selected.partition, &rows)?;
    let input: Vec<f64> = (0..n).map(|i| ((i + 1) as f64 * 0.017).sin()).collect();
    let mut extended = vec![0.0; halo.extended_len()];
    let mut output = vec![0.0; halo.owned_len()];
    let traffic = mpi.spmv_local(
        &halo,
        &operator,
        &input[own.start as usize..own.end as usize],
        &mut extended,
        &mut output,
    )?;
    let mut err = 0.0f64;
    for (idx, &actual) in output.iter().enumerate() {
        let row = own.start as usize + idx;
        let expected_val =
            4.0 * input[row] - 0.5 * input[(row + 1) % n] - 0.5 * input[(row + n - 1) % n];
        err = err.max((actual - expected_val).abs());
    }
    let total_err = mpi.all_reduce_sum_f64(err);
    let sent = mpi.all_reduce_sum_u64(traffic.sent_values as u64);
    let recv = mpi.all_reduce_sum_u64(traffic.received_values as u64);
    let ghosts = mpi.all_reduce_sum_u64(stats.ghosts as u64);
    let owned = mpi.all_reduce_sum_u64(stats.owned_rows as u64);
    let workers_with_full_csr = mpi.all_reduce_sum_u64(u64::from(rank != 0 && global.is_some()));
    if total_err > 1e-12
        || sent != recv
        || recv != ghosts
        || owned != n as u64
        || workers_with_full_csr != 0
    {
        return Err("H1-C3E MPI numerical/traffic/global-CSR-ownership check failed".into());
    }
    mpi.barrier();
    if rank == 0 {
        println!(
            "PASS H1-C3E ranks={expected} policy={:?} decision={:?} guard={:?} ghosts={ghosts} sent={sent} recv={recv} owned={owned} workers_global_csr=0 abs_err={total_err:.3e}",
            selected.selected_kind, selected.decision, selected.calibration_guard,
        );
    }
    Ok(())
}
