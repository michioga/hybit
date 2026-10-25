//! H1-C3K5-C: cache-aware, medium/large, original-coefficient MPI SpMV diagnostic.
//!
//! Bounded global CSR replication on all ranks remains experimental scaffolding.
//! The native numeric Halo exchange is MPI_Alltoallv on rank-local CSR rows.
//! No production solver routing, PCG, field-aware ownership or ABI change.
//! Only Equal / ABTM Region Growth / ABTM Multi-source are compared.
//! The measured SpMV uses the ORIGINAL numerical matrix coefficients.
//! Synthetic SPD surrogates and PCG iterations are not part of this probe.

use hybit_distributed::mpi_assignment::{
    AssignmentHaloPlan, AssignmentLocalCsr, AssignmentOwnedCsrRows, AssignmentSpmvWorkspace,
    prepare_assignment_owned_rows, spmv_assignment,
};
use hybit_distributed::mpi_backend::MpiRuntime;
use hybit_distributed::{
    ContiguousPartition, PartitionAssignment, PartitionTelemetry,
    abtm_balanced_multisource_partition, abtm_region_grow_partition,
    partition_telemetry_assignment,
};
use hybit_matrix::{Csr32Matrix, read_matrix_market};
use mpi::collective::SystemOperation;
use mpi::traits::*;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

type BenchResult<T> = Result<T, Box<dyn Error>>;
const MAX_INPUT_BYTES: u64 = 384 * 1024 * 1024;
const MAX_DOFS: usize = 1_200_000;
const MAX_NNZ: usize = 16_000_000;

#[derive(Clone, Copy)]
enum Policy {
    Equal,
    Region,
    Multisource,
}
impl Policy {
    fn label(self) -> &'static str {
        match self {
            Self::Equal => "equal",
            Self::Region => "abtm_region_growth",
            Self::Multisource => "abtm_multisource",
        }
    }
}

fn max_f64(value: f64) -> f64 {
    let world = mpi::topology::SimpleCommunicator::world();
    let mut result = 0.0;
    world.all_reduce_into(&value, &mut result, SystemOperation::max());
    result
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let i = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[i - 1] + sorted[i])
    } else {
        sorted[i]
    }
}

fn propose(matrix: &Csr32Matrix, ranks: u32, policy: Policy) -> BenchResult<PartitionAssignment> {
    Ok(match policy {
        Policy::Equal => PartitionAssignment::from_contiguous(&ContiguousPartition::balanced(
            matrix.nrows() as u64,
            ranks,
        )?)?,
        Policy::Region => abtm_region_grow_partition(matrix, ranks)?.0,
        Policy::Multisource => abtm_balanced_multisource_partition(matrix, ranks)?.0,
    })
}

struct Case {
    policy: Policy,
    telemetry: PartitionTelemetry,
    halo: AssignmentHaloPlan,
    local: AssignmentLocalCsr,
    scratch: AssignmentSpmvWorkspace,
    output: Vec<f64>,
    sent_total: u64,
    partition_ms: f64,
    extraction_ms: f64,
    setup_ms: f64,
    samples: Vec<f64>,
    halo_samples: Vec<f64>,
    compute_samples: Vec<f64>,
    worst_abs: f64,
    worst_rel: f64,
    resident_bytes_sum: u64,
}

fn prepare_case(
    mpi: &MpiRuntime,
    matrix: &Csr32Matrix,
    ranks: u32,
    policy: Policy,
) -> BenchResult<Case> {
    mpi.barrier();
    let begin = Instant::now();
    let root_proposal = if mpi.rank() == 0 {
        Some(propose(matrix, ranks, policy))
    } else {
        None
    };
    if mpi.all_reduce_sum_u64(u64::from(
        root_proposal.as_ref().is_some_and(Result::is_err),
    )) != 0
    {
        if let Some(Err(e)) = &root_proposal {
            eprintln!("K3 root partition {} failed: {e}", policy.label());
        }
        return Err("root partition generation failed".into());
    }
    let mut owner_map = if let Some(Ok(assignment)) = &root_proposal {
        assignment.owners().to_vec()
    } else {
        vec![0u32; matrix.nrows()]
    };
    mpi::topology::SimpleCommunicator::world()
        .process_at_rank(0)
        .broadcast_into(&mut owner_map[..]);
    let assignment = PartitionAssignment::from_owners(ranks, owner_map)?;
    let partition_ms = max_f64(begin.elapsed().as_secs_f64() * 1000.0);
    let telemetry = partition_telemetry_assignment(matrix, &assignment)?;
    mpi.barrier();
    let extraction_started = Instant::now();
    let rank = mpi.rank() as u32;
    let rows = AssignmentOwnedCsrRows::from_global(matrix, &assignment, rank)?;
    let extraction_ms = max_f64(extraction_started.elapsed().as_secs_f64() * 1000.0);
    mpi.barrier();
    let begin = Instant::now();
    let (halo, local, stats) = prepare_assignment_owned_rows(mpi, &assignment, &rows)?;
    let setup_ms = max_f64(begin.elapsed().as_secs_f64() * 1000.0);
    let sum_ghosts = mpi.all_reduce_sum_u64(stats.ghosts as u64);
    let sum_peers = mpi.all_reduce_sum_u64(stats.peer_ranks as u64);
    let sum_sent = mpi.all_reduce_sum_u64(stats.sent_f64 as u64);
    let sum_recv = mpi.all_reduce_sum_u64(stats.received_f64 as u64);
    let resident_bytes_sum = mpi.all_reduce_sum_u64(stats.rank_operator_halo_bytes as u64);
    let mismatch = sum_ghosts != telemetry.communication_volume as u64
        || sum_peers != telemetry.directional_peer_relations as u64
        || sum_sent != sum_recv
        || sum_recv != sum_ghosts
        || stats.owned_rows != halo.owned_len()
        || stats.local_nnz != local.nnz();
    if mpi.all_reduce_sum_u64(u64::from(mismatch)) != 0 {
        return Err(format!(
            "{} Halo topology differs from reference telemetry",
            policy.label()
        )
        .into());
    }
    let scratch = halo.workspace();
    let output = vec![0.0; halo.owned_len()];
    Ok(Case {
        policy,
        telemetry,
        halo,
        local,
        scratch,
        output,
        sent_total: sum_sent,
        partition_ms,
        extraction_ms,
        setup_ms,
        samples: Vec::new(),
        halo_samples: Vec::new(),
        compute_samples: Vec::new(),
        worst_abs: 0.0,
        worst_rel: 0.0,
        resident_bytes_sum,
    })
}

fn run_one(
    mpi: &MpiRuntime,
    case: &mut Case,
    x: &[f64],
    reference: &[f64],
    measured: bool,
    validate: bool,
) -> BenchResult<()> {
    let x_owned = case
        .halo
        .owned_globals
        .iter()
        .map(|&global| x[global as usize])
        .collect::<Vec<_>>();
    mpi.barrier();
    let begun = Instant::now();
    let timing = spmv_assignment(
        mpi,
        &case.halo,
        &case.local,
        &x_owned,
        &mut case.scratch,
        &mut case.output,
    )?;
    let wall = max_f64(begun.elapsed().as_secs_f64() * 1000.0);
    let halo = max_f64(timing.halo_ns as f64 * 1e-6);
    let compute = max_f64(timing.numeric_ns as f64 * 1e-6);
    if validate {
        let mut abs_err = 0.0_f64;
        let mut finite = true;
        for (&actual, &global) in case.output.iter().zip(&case.halo.owned_globals) {
            let expected = reference[global as usize];
            finite &= actual.is_finite() && expected.is_finite();
            abs_err = abs_err.max((actual - expected).abs());
        }
        for (i, &global) in case.halo.ghost_globals.iter().enumerate() {
            let actual = case.scratch.extended[case.halo.owned_len() + i];
            finite &= actual.is_finite();
            abs_err = abs_err.max((actual - x[global as usize]).abs());
        }
        let abs_err = max_f64(abs_err);
        let reference_scale = max_f64(reference.iter().fold(1.0_f64, |a, &v| a.max(v.abs())));
        let rel_err = abs_err / reference_scale;
        if mpi.all_reduce_sum_u64(u64::from(
            !finite || !rel_err.is_finite() || rel_err > 1e-11,
        )) != 0
        {
            return Err(format!(
                "{} native SpMV failed: abs={abs_err:e} rel={rel_err:e}",
                case.policy.label()
            )
            .into());
        }
        case.worst_abs = case.worst_abs.max(abs_err);
        case.worst_rel = case.worst_rel.max(rel_err);
    }
    if measured {
        case.samples.push(wall);
        case.halo_samples.push(halo);
        case.compute_samples.push(compute);
    }
    Ok(())
}

fn main() -> BenchResult<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 7 {
        return Err("usage: mpi_k5c_large_spmv <ranks> <matrix.mtx> <warmups> <repeats> <projected_spmv_calls> <rotate0..rotate2> <results.csv>".into());
    }
    let expected: i32 = args[0].parse()?;
    let path = Path::new(&args[1]);
    let warmups: usize = args[2].parse()?;
    let repeats: usize = args[3].parse()?;
    let projected: usize = args[4].parse()?;
    let shift = parse_rotation(&args[5])?;
    let csv = Path::new(&args[6]);
    if !matches!(expected, 1 | 2 | 4 | 8)
        || !(1..=20).contains(&warmups)
        || !(3..=256).contains(&repeats)
        || !(1..=1_000_000).contains(&projected)
    {
        return Err("invalid rank/radius/warmups/repeats/projection bounds".into());
    }
    let mpi = MpiRuntime::initialize()?;
    if mpi.size() != expected {
        return Err("MPI rank count differs from request".into());
    }
    if mpi.all_reduce_sum_u64(u64::from(mpi.rank() == 0 && csv.exists())) != 0 {
        return Err("output CSV exists; refuse overwrite".into());
    }
    mpi.barrier();
    let read_begin = Instant::now();
    let loaded: BenchResult<_> = (|| {
        if std::fs::metadata(path)?.len() > MAX_INPUT_BYTES {
            return Err("input exceeds 384 MiB per-rank Matrix Market bound".into());
        }
        Ok(read_matrix_market(path)?)
    })();
    if mpi.all_reduce_sum_u64(u64::from(loaded.is_err())) != 0 {
        if let Err(err) = &loaded {
            eprintln!("K5-C matrix read: {err}");
        }
        return Err("Matrix Market load failed on at least one rank".into());
    }
    let (matrix, info) = loaded?;
    let n = matrix.nrows();
    if mpi.all_reduce_sum_u64(u64::from(
        n == 0
            || n > MAX_DOFS
            || n != matrix.ncols()
            || matrix.nnz() > MAX_NNZ
            || n < expected as usize,
    )) != 0
    {
        return Err("input outside K5-C replicated-CSR dimensions or nnz bounds".into());
    }
    let read_ms = max_f64(read_begin.elapsed().as_secs_f64() * 1000.0);
    let global_csr_bytes = (std::mem::size_of_val(matrix.row_ptr())
        + std::mem::size_of_val(matrix.col_idx())
        + std::mem::size_of_val(matrix.values())) as u64;

    // Precompute two references once. No full global CSR multiply occurs between
    // timed SpMV iterations, which avoids artificial last-level cache eviction.
    // The two vectors alternate, rather than repeatedly multiplying one vector.
    let x0: Vec<f64> = (0..n).map(|i| ((i as f64 + 1.0) * 0.017).sin()).collect();
    let x1: Vec<f64> = (0..n).map(|i| ((i as f64 + 1.0) * 0.0081).cos()).collect();
    let y0 = matrix.spmv(&x0)?;
    let y1 = matrix.spmv(&x1)?;
    let xs = [&x0, &x1];
    let refs = [&y0, &y1];
    let kinds = [Policy::Equal, Policy::Region, Policy::Multisource];
    let mut records = Vec::new();
    for slot in 0..kinds.len() {
        let policy = kinds[(slot + shift) % kinds.len()];
        // Prepare and drop one policy at a time; never retain all local CSRs.
        let mut case = prepare_case(&mpi, &matrix, expected as u32, policy)?;
        for trial in 0..(warmups + repeats) {
            let parity = trial % 2;
            let measured = trial >= warmups;
            let validate = trial <= 1 || trial == warmups + repeats - 1;
            run_one(
                &mpi,
                &mut case,
                xs[parity],
                refs[parity],
                measured,
                validate,
            )?;
        }
        let sum_spmv_ms: f64 = case.samples.iter().sum();
        let sum_halo_ms: f64 = case.halo_samples.iter().sum();
        let sum_numeric_ms: f64 = case.compute_samples.iter().sum();
        let median_spmv_ms = median(&case.samples);
        let median_halo_ms = median(&case.halo_samples);
        let median_numeric_ms = median(&case.compute_samples);
        let prep = case.partition_ms + case.extraction_ms + case.setup_ms;
        let projected_solve_ms = median_spmv_ms * projected as f64;
        let estimated_total_ms = prep + projected_solve_ms;
        let s = &case.telemetry;
        if mpi.rank() == 0 {
            println!(
                "K5C NATIVE PASS method={} ranks={} n={} nnz={} order_slot={} ghosts={} cut={} partition_ms={:.3} extraction_ms={:.3} halo_setup_ms={:.3} median_spmv_ms={:.6} median_halo_ms={:.6} numeric_ms={:.6} measured_spmv_sum_ms={:.3} projected_total_ms={:.3} rel_err={:.3e}",
                policy.label(),
                expected,
                n,
                matrix.nnz(),
                slot + 1,
                s.communication_volume,
                s.cut_nnz,
                case.partition_ms,
                case.extraction_ms,
                case.setup_ms,
                median_spmv_ms,
                median_halo_ms,
                median_numeric_ms,
                sum_spmv_ms,
                estimated_total_ms,
                case.worst_rel
            );
        }
        records.push(format!("{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6},{},{},{},{},{:.8},{:.8},{},{:.12e}",
            path.display(), expected, n, info.input_entries, matrix.nnz(),
            global_csr_bytes, policy.label(), slot+1, warmups, repeats, projected,
            read_ms, case.partition_ms, case.extraction_ms, case.setup_ms,
            median_spmv_ms, median_halo_ms, median_numeric_ms, sum_spmv_ms,
            sum_halo_ms, case.samples.len(), sum_numeric_ms, estimated_total_ms,
            s.communication_volume, s.cut_nnz, s.directional_peer_relations,
            case.resident_bytes_sum, s.owned_dof_imbalance, s.local_nnz_imbalance,
            case.sent_total, case.worst_rel));
        // Case dropped here; only global CSR and reference vectors persist.
    }
    mpi.barrier();
    if mpi.rank() == 0 {
        let mut out = OpenOptions::new().create_new(true).write(true).open(csv)?;
        writeln!(
            out,
            "matrix,ranks,n,input_nnz,csr_nnz,global_csr_bytes_per_rank,method,order_slot,warmups,repeats,projected_spmv_calls,read_max_ms,partition_max_ms,extraction_max_ms,halo_setup_max_ms,median_max_spmv_ms,median_max_halo_ms,median_max_numeric_ms,measured_spmv_sum_ms,measured_halo_sum_ms,measured_calls,measured_numeric_sum_ms,projected_prepared_total_ms,ghosts,cut_nnz,peer_relations,operator_halo_sum_bytes,owned_imbalance,nnz_imbalance,sent_f64,worst_rel_error"
        )?;
        for record in records {
            writeln!(out, "{record}")?;
        }
        out.flush()?;
        println!(
            "=== K5-C SINGLE-NODE NATIVE MPI ORIGINAL-COEFFICIENT SpMV PASS: {} ===",
            csv.display()
        );
        println!(
            "NOTE: projected_prepared_total_ms is an extrapolation, not a measured solve-many time; global CSR is replicated on every rank."
        );
    }
    Ok(())
}

fn parse_rotation(s: &str) -> BenchResult<usize> {
    match s {
        "rotate0" => Ok(0),
        "rotate1" => Ok(1),
        "rotate2" => Ok(2),
        _ => Err("rotation must be rotate0..rotate2".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rotation_bounds() {
        assert_eq!(parse_rotation("rotate0").unwrap(), 0);
        assert_eq!(parse_rotation("rotate2").unwrap(), 2);
        assert!(parse_rotation("rotate3").is_err());
    }
    #[test]
    fn sample_median() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0]), 1.5);
    }
}
