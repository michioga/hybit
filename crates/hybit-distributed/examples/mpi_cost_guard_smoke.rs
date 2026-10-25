//! H1-C3D: native MPI identity + full-offset agreement + numerical Halo smoke.
//! The global CSR fixture is deliberately replicated for testing only.

use hybit_distributed::mpi_backend::MpiRuntime;
use hybit_distributed::mpi_input::{OwnedCsrRows, prepare_owned_rows};
use hybit_distributed::{
    CalibrationGuardDecision, CostAwarePartitionOptions, PartitionCostCalibration,
    build_contiguous_halo_plans, capture_partition_calibration,
    cost_aware_contiguous_partition_scoped, verify_cost_partition_mpi_consensus,
};
use hybit_matrix::{Csr32Matrix, DofBlock, DofEntityKind, FieldAwareDofLayout};
use std::error::Error;

fn fixture() -> Result<(FieldAwareDofLayout, Csr32Matrix), Box<dyn Error>> {
    let blocks = (0..64u64)
        .map(|i| DofBlock {
            entity_kind: DofEntityKind::Node,
            entity_id: i,
            field_id: 0,
            components: 2,
        })
        .collect();
    let layout = FieldAwareDofLayout::new(blocks)?;
    let n = layout.total_dofs() as usize;
    let mut row_ptr = vec![0u32];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for row in 0..n {
        for (col, value) in [(row, 4.0), ((row + 1) % n, -0.5), ((row + n - 1) % n, -0.5)] {
            col_idx.push(col as u32);
            values.push(value);
        }
        row_ptr.push(col_idx.len() as u32);
    }
    Ok((layout, Csr32Matrix::new(n, n, row_ptr, col_idx, values)?))
}

fn local_rows(matrix: &Csr32Matrix, begin: usize, end: usize) -> OwnedCsrRows {
    let offset = matrix.row_ptr()[begin];
    let first = offset as usize;
    let last = matrix.row_ptr()[end] as usize;
    OwnedCsrRows {
        global_dofs: matrix.nrows() as u64,
        owned: begin as u64..end as u64,
        row_ptr: matrix.row_ptr()[begin..=end]
            .iter()
            .map(|&p| p - offset)
            .collect(),
        global_col_idx: matrix.col_idx()[first..last]
            .iter()
            .map(|&c| u64::from(c))
            .collect(),
        values: matrix.values()[first..last].to_vec(),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 && args.len() != 3 {
        return Err("usage: mpi_cost_guard_smoke <ranks> [--mismatch-env]".into());
    }
    let expected_ranks: i32 = args[1].parse()?;
    let negative = args
        .get(2)
        .is_some_and(|arg| arg.as_str() == "--mismatch-env");
    if args.len() == 3 && !negative {
        return Err("only --mismatch-env is supported".into());
    }
    let mpi = MpiRuntime::initialize()?;
    if mpi.size() != expected_ranks || !matches!(expected_ranks, 1 | 2 | 4 | 8) {
        return Err("MPI world size must be 1, 2, 4 or 8 as requested".into());
    }
    if negative && expected_ranks == 1 {
        return Err("mismatch mode requires at least two ranks".into());
    }
    let (layout, matrix) = fixture()?;
    let opts = CostAwarePartitionOptions {
        expected_spmv_calls: 1000,
        search_radius_blocks: 3,
        max_owned_dof_imbalance: 1.20,
        max_local_nnz_imbalance: 1.20,
        ..CostAwarePartitionOptions::default()
    };
    // In production the timing evidence must originate from previously
    // measured matching baseline and radius. These numbers are only a smoke
    // fixture; this example makes no performance claim.
    let calibration = capture_partition_calibration(
        &matrix,
        &layout,
        expected_ranks as u32,
        opts,
        "H1C3D-fixture-msmpi-v1",
        PartitionCostCalibration {
            baseline_spmv_ns: 1_000,
            graph_spmv_ns: 900,
            graph_prepare_budget_ns: 100_000,
        },
    )?;
    let environment_tag = if negative && mpi.rank() == 1 {
        "H1C3D-injected-difference"
    } else {
        "H1C3D-fixture-msmpi-v1"
    };
    let selection = cost_aware_contiguous_partition_scoped(
        &matrix,
        &layout,
        expected_ranks as u32,
        opts,
        environment_tag,
        Some(&calibration),
    )?;
    let consensus = verify_cost_partition_mpi_consensus(&selection, opts, Some(&calibration));
    if negative {
        if consensus.is_ok() {
            return Err("expected an MPI cross-rank scope mismatch".into());
        }
        mpi.barrier();
        if mpi.rank() == 0 {
            println!(
                "PASS H1-C3D injected_environment_mismatch ranks={expected_ranks} rejected_collectively=true"
            );
        }
        return Ok(());
    }
    consensus?;
    if selection.guard != CalibrationGuardDecision::Matched {
        return Err("matching calibration was not recognized".into());
    }

    let plans = build_contiguous_halo_plans(&matrix, &selection.partition)?;
    let rank = mpi.rank() as usize;
    let plan = &plans[rank];
    let owned = plan.owned_range();
    let rows = local_rows(&matrix, owned.start as usize, owned.end as usize);
    let (halo, local, _) = prepare_owned_rows(&mpi, &selection.partition, &rows)?;
    if &halo != plan {
        return Err("MPI Halo differs from exact CSR reference".into());
    }
    let input: Vec<f64> = (0..matrix.nrows())
        .map(|i| ((i + 1) as f64 * 0.017).sin())
        .collect();
    let reference = matrix.spmv(&input)?;
    let mut extended = vec![0.0; halo.extended_len()];
    let mut result = vec![0.0; halo.owned_len()];
    let traffic = mpi.spmv_local(
        &halo,
        &local,
        &input[owned.start as usize..owned.end as usize],
        &mut extended,
        &mut result,
    )?;
    let local_error = result
        .iter()
        .zip(&reference[owned.start as usize..owned.end as usize])
        .fold(0.0_f64, |err, (&actual, &correct)| {
            err.max((actual - correct).abs())
        });
    let total_error = mpi.all_reduce_sum_f64(local_error);
    let sent = mpi.all_reduce_sum_u64(traffic.sent_values as u64);
    let received = mpi.all_reduce_sum_u64(traffic.received_values as u64);
    let ghosts = mpi.all_reduce_sum_u64(halo.ghost_len() as u64);
    if total_error > 1.0e-12 || sent != received || received != ghosts {
        return Err("H1-C3D numerical or Halo traffic mismatch".into());
    }
    mpi.barrier();
    if mpi.rank() == 0 {
        println!(
            "PASS H1-C3D ranks={expected_ranks} guard={:?} decision={:?} selected={:?} ghosts={ghosts} sent={sent} received={received} abs_error={total_error:.3e}",
            selection.guard, selection.report.decision, selection.report.selected_kind,
        );
    }
    Ok(())
}
