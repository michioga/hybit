//! H1-C3K3 native MS-MPI SpMV experiment: contiguous vs ABTM ownership.
//!
//! Bounded global CSR replication on all ranks is intentional test scaffolding.
//! The native numeric Halo exchange is MPI_Alltoallv on rank-local CSR rows.
//! No production solver routing, PCG, field-aware ownership or ABI change.

use hybit_distributed::mpi_assignment::{
    prepare_assignment_owned_rows, spmv_assignment, AssignmentHaloPlan, AssignmentLocalCsr,
    AssignmentOwnedCsrRows, AssignmentSpmvWorkspace,
};
use hybit_distributed::mpi_backend::MpiRuntime;
use hybit_distributed::{
    abtm_balanced_multisource_partition, abtm_multilevel_partition, abtm_region_grow_partition,
    field_graph_contiguous_partition, partition_telemetry_assignment, AbtmMultilevelOptions,
    ContiguousPartition, PartitionAssignment, PartitionTelemetry,
};
use hybit_matrix::{read_matrix_market, Csr32Matrix, DofBlock, DofEntityKind, FieldAwareDofLayout};
use mpi::collective::SystemOperation;
use mpi::traits::*;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

type BenchResult<T> = Result<T, Box<dyn Error>>;
const MAX_INPUT_BYTES: u64 = 110 * 1024 * 1024;
const MAX_DOFS: usize = 200_000;
const MAX_NNZ: usize = 8_000_000;

#[derive(Clone, Copy)]
enum Policy {
    Equal,
    Graph,
    Region,
    Multisource,
    Multilevel,
}
impl Policy {
    fn label(self) -> &'static str {
        match self {
            Self::Equal => "equal",
            Self::Graph => "graph_contiguous",
            Self::Region => "abtm_region_growth",
            Self::Multisource => "abtm_multisource",
            Self::Multilevel => "abtm_multilevel",
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
    if sorted.len() % 2 == 0 {
        0.5 * (sorted[i - 1] + sorted[i])
    } else {
        sorted[i]
    }
}

fn scalar_layout(n: usize) -> BenchResult<FieldAwareDofLayout> {
    Ok(FieldAwareDofLayout::new(
        (0..n)
            .map(|i| DofBlock {
                entity_kind: DofEntityKind::Node,
                entity_id: i as u64,
                field_id: 0,
                components: 1,
            })
            .collect(),
    )?)
}

fn propose(
    matrix: &Csr32Matrix,
    ranks: u32,
    radius: usize,
    policy: Policy,
) -> BenchResult<PartitionAssignment> {
    Ok(match policy {
        Policy::Equal => PartitionAssignment::from_contiguous(&ContiguousPartition::balanced(
            matrix.nrows() as u64,
            ranks,
        )?)?,
        Policy::Graph => {
            let (partition, _) = field_graph_contiguous_partition(
                matrix,
                &scalar_layout(matrix.nrows())?,
                ranks,
                radius,
            )?;
            PartitionAssignment::from_contiguous(&partition)?
        }
        Policy::Region => abtm_region_grow_partition(matrix, ranks)?.0,
        Policy::Multisource => abtm_balanced_multisource_partition(matrix, ranks)?.0,
        Policy::Multilevel => {
            abtm_multilevel_partition(matrix, ranks, AbtmMultilevelOptions::default())?.0
        }
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
    received_total: u64,
    partition_ms: f64,
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
    radius: usize,
    policy: Policy,
) -> BenchResult<Case> {
    mpi.barrier();
    let begin = Instant::now();
    let root_proposal = if mpi.rank() == 0 {
        Some(propose(matrix, ranks, radius, policy))
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
    let rank = mpi.rank() as u32;
    let rows = AssignmentOwnedCsrRows::from_global(matrix, &assignment, rank)?;
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
        received_total: sum_recv,
        partition_ms,
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
    if measured {
        case.samples.push(wall);
        case.halo_samples.push(halo);
        case.compute_samples.push(compute);
        case.worst_abs = case.worst_abs.max(abs_err);
        case.worst_rel = case.worst_rel.max(rel_err);
    }
    Ok(())
}

fn main() -> BenchResult<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 6 {
        return Err("usage: mpi_k3_assignment_spmv <ranks> <matrix.mtx> <radius> <warmups> <repeats> <results.csv>".into());
    }
    let expected: i32 = args[0].parse()?;
    let path = Path::new(&args[1]);
    let radius: usize = args[2].parse()?;
    let warmups: usize = args[3].parse()?;
    let repeats: usize = args[4].parse()?;
    let csv = Path::new(&args[5]);
    if !matches!(expected, 1 | 2 | 4 | 8)
        || !(1..=256).contains(&radius)
        || warmups > 20
        || !(1..=100).contains(&repeats)
    {
        return Err("invalid rank/radius/warmups/repeats".into());
    }
    let mpi = MpiRuntime::initialize()?;
    if mpi.size() != expected {
        return Err("MPI ranks disagree with requested count".into());
    }
    // This smoke intentionally loads a copy on every rank; use explicit caps.
    let loaded: BenchResult<_> = (|| {
        let md = std::fs::metadata(path)?;
        if md.len() > MAX_INPUT_BYTES {
            return Err("input exceeds K3 replicated-CSR benchmark size cap".into());
        }
        Ok(read_matrix_market(path)?)
    })();
    if mpi.all_reduce_sum_u64(u64::from(loaded.is_err())) != 0 {
        if let Err(err) = &loaded {
            eprintln!("K3 Matrix Market read failed: {err}");
        }
        return Err("matrix unavailable on one or more MPI ranks".into());
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
        return Err("matrix outside K3 square/n/nnz/rank safety caps".into());
    }
    let kinds = [
        Policy::Equal,
        Policy::Graph,
        Policy::Region,
        Policy::Multisource,
        Policy::Multilevel,
    ];
    let mut cases = Vec::with_capacity(kinds.len());
    for policy in kinds {
        cases.push(prepare_case(
            &mpi,
            &matrix,
            expected as u32,
            radius,
            policy,
        )?);
    }
    for trial in 0..(warmups + repeats) {
        let x = (0..n)
            .map(|i| {
                let k = (i as f64 + 1.0) * (trial as f64 + 1.0);
                (k * 0.017).sin() + 0.125 * (k * 0.0031).cos()
            })
            .collect::<Vec<_>>();
        let reference = matrix.spmv(&x)?;
        for shift in 0..cases.len() {
            let index = (shift + trial) % cases.len();
            run_one(&mpi, &mut cases[index], &x, &reference, trial >= warmups)?;
        }
    }
    mpi.barrier();
    if mpi.rank() == 0 {
        let mut out = OpenOptions::new().create_new(true).write(true).open(csv)?;
        writeln!(out, "matrix,ranks,n,input_nnz,expanded_nnz,method,radius,warmups,repeats,partition_max_ms,halo_setup_max_ms,median_max_spmv_ms,median_max_exchange_ms,median_max_numeric_ms,ghosts,cut_nnz,peers,owned_imbalance,nnz_imbalance,operator_halo_sum_bytes,abs_error,rel_error,sent_f64,received_f64,scope")?;
        for case in &cases {
            let s = &case.telemetry;
            let median_total = median(&case.samples);
            let median_halo = median(&case.halo_samples);
            let median_numeric = median(&case.compute_samples);
            writeln!(out, "{},{},{},{},{},{},{},{},{},{:.6},{:.6},{median_total:.6},{median_halo:.6},{median_numeric:.6},{},{},{},{:.8},{:.8},{},{:.12e},{:.12e},{},{},native_noncontiguous_spmv_experiment",
                path.display(), expected, n,
                info.input_entries, matrix.nnz(), case.policy.label(), radius, warmups, repeats,
                case.partition_ms, case.setup_ms, s.communication_volume, s.cut_nnz,
                s.directional_peer_relations, s.owned_dof_imbalance, s.local_nnz_imbalance,
                case.resident_bytes_sum, case.worst_abs, case.worst_rel,
                case.sent_total, case.received_total)?;
            println!("K3 NATIVE PASS method={} ranks={expected} n={n} nnz={} cut={} ghosts={} peers={} partition_ms={:.3} setup_ms={:.3} median_max_spmv_ms={median_total:.6} median_max_exchange_ms={median_halo:.6} rel_err={:.3e} sent_f64={} recv_f64={}",
                case.policy.label(), matrix.nnz(), s.cut_nnz, s.communication_volume,
                s.directional_peer_relations, case.partition_ms, case.setup_ms, case.worst_rel,
                case.sent_total, case.received_total);
        }
        out.flush()?;
        println!(
            "=== H1-C3K3 NATIVE MPI ASSIGNMENT SpMV PASS: {} ===",
            csv.display()
        );
    }
    Ok(())
}
