//! H1-C3C non-MPI policy probe on bounded Matrix Market input.
//! No actual timed SpMV is performed here; the calibration is supplied by
//! the caller from earlier independent and representative MPI observations.
//! MTX carries no physical field metadata: one DOF = one synthetic block.

use hybit_distributed::{
    cost_aware_contiguous_partition, CostAwarePartitionOptions, PartitionCostCalibration,
};
use hybit_matrix::{read_matrix_market, DofBlock, DofEntityKind, FieldAwareDofLayout};
use std::error::Error;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 8 && args.len() != 9 {
        return Err("usage: cost_field_partition_probe <matrix.mtx> <ranks> <expected_spmv_calls> <baseline_spmv_ns> <graph_spmv_ns> <graph_prepare_budget_ns> <search_radius_blocks> <max_owned_imbalance> [max_nnz_imbalance]".into());
    }
    let file = Path::new(&args[0]);
    if std::fs::metadata(file)?.len() > 110 * 1024 * 1024 {
        return Err("matrix file over 110 MiB; this is a bounded global CSR experiment".into());
    }
    let ranks: u32 = args[1].parse()?;
    let expected_spmv_calls: u64 = args[2].parse()?;
    let baseline_spmv_ns: u64 = args[3].parse()?;
    let graph_spmv_ns: u64 = args[4].parse()?;
    let graph_prepare_budget_ns: u64 = args[5].parse()?;
    let search_radius_blocks: usize = args[6].parse()?;
    let max_owned_dof_imbalance: f64 = args[7].parse()?;
    let max_local_nnz_imbalance: f64 = if args.len() == 9 {
        args[8].parse()?
    } else {
        1.15
    };
    let (matrix, info) = read_matrix_market(file)?;
    if matrix.nrows() > 200_000 || matrix.nnz() > 8_000_000 {
        return Err("CSR after expansion exceeds 200k DOFs or 8M NNZ".into());
    }
    let blocks = (0..matrix.nrows())
        .map(|i| DofBlock {
            entity_kind: DofEntityKind::Node,
            entity_id: i as u64,
            field_id: 0,
            components: 1,
        })
        .collect();
    let layout = FieldAwareDofLayout::new(blocks)?;
    let (partition, report) = cost_aware_contiguous_partition(
        &matrix,
        &layout,
        ranks,
        CostAwarePartitionOptions {
            expected_spmv_calls,
            calibration: Some(PartitionCostCalibration {
                baseline_spmv_ns,
                graph_spmv_ns,
                graph_prepare_budget_ns,
            }),
            search_radius_blocks,
            max_owned_dof_imbalance,
            max_local_nnz_imbalance,
            minimum_net_savings_ns: 0,
        },
    )?;
    println!(
        "H1-C3C matrix={} ranks={} n={} input_entries={} csr_nnz={} symmetry={:?}",
        file.display(),
        ranks,
        matrix.nrows(),
        info.input_entries,
        matrix.nnz(),
        info.symmetry
    );
    println!(
        "selected={:?} baseline={:?} reason={:?} attempted_graph={} modeled_net_savings_ns={:?} actual_local_graph_prepare_ns={:?}",
        report.selected_kind, report.baseline_kind, report.decision,
        report.graph_candidate.is_some(), report.modeled_net_savings_ns,
        report.local_graph_prepare_ns,
    );
    println!(
        "ghosts={} -> {} cut_nnz={} -> {} peers={} -> {} owned_imbalance={:.4} -> {:.4} nnz_imbalance={:.4} -> {:.4}",
        report.baseline.communication_volume,
        report.selected.communication_volume,
        report.baseline.cut_nnz,
        report.selected.cut_nnz,
        report.baseline.directional_peer_relations,
        report.selected.directional_peer_relations,
        report.baseline.owned_dof_imbalance,
        report.selected.owned_dof_imbalance,
        report.baseline.local_nnz_imbalance,
        report.selected.local_nnz_imbalance,
    );
    println!("partition offsets={:?}", partition.offsets());
    Ok(())
}
