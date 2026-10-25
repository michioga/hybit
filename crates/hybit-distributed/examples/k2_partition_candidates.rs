//! H1-C3K2/B: compare partition candidates, NOT MPI solver performance.
//! All candidates consume the same CSR matrix and rank count.
//! ABTM noncontiguous assignments have NOT been routed through MPI-PCG yet.

use hybit_distributed::{
    AbtmMultilevelOptions, ContiguousPartition, PartitionAssignment, PartitionTelemetry,
    abtm_balanced_multisource_partition, abtm_multilevel_partition, abtm_region_grow_partition,
    field_graph_contiguous_partition, partition_telemetry_assignment,
};
use hybit_matrix::{Csr32Matrix, DofBlock, DofEntityKind, FieldAwareDofLayout, read_matrix_market};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

type BenchResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy)]
enum Candidate {
    Equal,
    ScalarGraph,
    AbtmRegion,
    AbtmMultisource,
    AbtmMultilevel,
}

impl Candidate {
    fn label(self) -> &'static str {
        match self {
            Self::Equal => "equal",
            Self::ScalarGraph => "scalar_graph_contiguous",
            Self::AbtmRegion => "abtm_region_growth",
            Self::AbtmMultisource => "abtm_multisource",
            Self::AbtmMultilevel => "abtm_multilevel",
        }
    }

    fn owner_class(self) -> &'static str {
        match self {
            Self::Equal | Self::ScalarGraph => "contiguous",
            _ => "noncontiguous",
        }
    }

    fn imbalance_tolerance_per_mille(self) -> u32 {
        match self {
            Self::AbtmMultilevel => 30,
            Self::ScalarGraph => 150,
            _ => 0,
        }
    }
}

struct CandidateRun {
    assignment: PartitionAssignment,
    candidate_accepted: Option<bool>,
    levels_built: Option<usize>,
    abtm_metadata_bytes: Option<usize>,
}

fn scalar_layout(n: usize) -> BenchResult<FieldAwareDofLayout> {
    let blocks = (0..n)
        .map(|i| DofBlock {
            entity_kind: DofEntityKind::Node,
            entity_id: i as u64,
            field_id: 0,
            components: 1,
        })
        .collect();
    Ok(FieldAwareDofLayout::new(blocks)?)
}

fn prepare_candidate(
    matrix: &Csr32Matrix,
    ranks: u32,
    candidate: Candidate,
    radius: usize,
) -> BenchResult<CandidateRun> {
    let mut accepted = None;
    let mut levels = None;
    let mut metadata_bytes = None;
    let assignment = match candidate {
        Candidate::Equal => {
            let partition = ContiguousPartition::balanced(matrix.nrows() as u64, ranks)?;
            PartitionAssignment::from_contiguous(&partition)?
        }
        Candidate::ScalarGraph => {
            // Matrix Market supplies no genuine physical field metadata.
            let layout = scalar_layout(matrix.nrows())?;
            let (partition, comparison) =
                field_graph_contiguous_partition(matrix, &layout, ranks, radius)?;
            accepted = Some(comparison.candidate_accepted);
            PartitionAssignment::from_contiguous(&partition)?
        }
        Candidate::AbtmRegion => {
            let (owners, stats) = abtm_region_grow_partition(matrix, ranks)?;
            metadata_bytes = Some(stats.dual_topology_metadata_bytes);
            owners
        }
        Candidate::AbtmMultisource => {
            let (owners, stats) = abtm_balanced_multisource_partition(matrix, ranks)?;
            metadata_bytes = Some(stats.dual_topology_metadata_bytes);
            owners
        }
        Candidate::AbtmMultilevel => {
            let (owners, stats) =
                abtm_multilevel_partition(matrix, ranks, AbtmMultilevelOptions::default())?;
            levels = Some(stats.levels_built);
            owners
        }
    };
    Ok(CandidateRun {
        assignment,
        candidate_accepted: accepted,
        levels_built: levels,
        abtm_metadata_bytes: metadata_bytes,
    })
}

fn verify_telemetry(
    matrix: &Csr32Matrix,
    ranks: u32,
    assignment: &PartitionAssignment,
    stats: &PartitionTelemetry,
) -> BenchResult<()> {
    if assignment.global_dofs() != matrix.nrows() as u64
        || assignment.rank_count() != ranks
        || stats.global_nnz != matrix.nnz()
        || stats.global_dofs != matrix.nrows() as u64
        || stats.min_owned_dofs == 0
        || stats.min_local_nnz > stats.max_local_nnz
    {
        return Err("partition/telemetry contract mismatch".into());
    }
    let total_owned = (0..ranks)
        .map(|rank| assignment.owned_count(rank))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .sum::<usize>();
    if total_owned != matrix.nrows() {
        return Err("partition does not own each row exactly once".into());
    }
    Ok(())
}

fn write_row(
    out: &mut BufWriter<std::fs::File>,
    row_metadata: (u32, usize, Candidate),
    prep_ms: f64,
    telemetry_ms: f64,
    changed: bool,
    run: &CandidateRun,
    stats: &PartitionTelemetry,
) -> BenchResult<()> {
    let (ranks, trial, candidate) = row_metadata;
    let accepted = match run.candidate_accepted {
        Some(true) => "true",
        Some(false) => "false",
        None => "na",
    };
    let levels = run.levels_built.map(|x| x.to_string()).unwrap_or_default();
    let bytes = run
        .abtm_metadata_bytes
        .map(|x| x.to_string())
        .unwrap_or_default();
    writeln!(
        out,
        "{},{},{},{},{},{:.6},{:.6},{},{},{},{},{},{},{},{},{},{:.8},{:.8},{},{},{},{},{},partition_only",
        candidate.label(),
        candidate.owner_class(),
        ranks,
        trial,
        stats.global_dofs,
        prep_ms,
        telemetry_ms,
        stats.global_nnz,
        stats.cut_nnz,
        stats.communication_volume,
        stats.directional_peer_relations,
        stats.max_neighbors,
        stats.min_owned_dofs,
        stats.max_owned_dofs,
        stats.min_local_nnz,
        stats.max_local_nnz,
        stats.owned_dof_imbalance,
        stats.local_nnz_imbalance,
        candidate.imbalance_tolerance_per_mille(),
        changed,
        accepted,
        levels,
        bytes,
    )?;
    Ok(())
}

fn run() -> BenchResult<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        return Err(
            "usage: partition_candidate_compare MATRIX.mtx OUTPUT.csv REPEATS RANKS_CSV RADIUS\nexample: partition_candidate_compare thermal1.mtx results.csv 3 2,4,8 16"
                .into(),
        );
    }
    let matrix_path = Path::new(&args[1]);
    let output_path = Path::new(&args[2]);
    if !matrix_path.is_file() || output_path.exists() {
        return Err("matrix must exist and CSV output must not already exist".into());
    }
    let repeats: usize = args[3].parse()?;
    let radius: usize = args[5].parse()?;
    if !(1..=10).contains(&repeats) || !(1..=4096).contains(&radius) {
        return Err("repeats must be 1..=10 and radius must be 1..=4096".into());
    }
    let ranks = args[4]
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    if ranks.is_empty() || ranks.iter().any(|&r| !matches!(r, 1 | 2 | 4 | 8)) {
        return Err("ranks must be comma-separated values from 1,2,4,8".into());
    }
    if ranks.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("rank counts must be strictly increasing without duplicates".into());
    }
    let read_started = Instant::now();
    let (matrix, info) = read_matrix_market(matrix_path)?;
    let read_ms = read_started.elapsed().as_secs_f64() * 1000.0;
    if matrix.nrows() != matrix.ncols() || ranks.iter().any(|&rank| rank as usize > matrix.nrows())
    {
        return Err("matrix must be square with enough DOFs for each rank count".into());
    }
    println!(
        "K2 ABTM partition candidates: n={} nnz={} input_nnz={} read_ms={:.3}",
        matrix.nrows(),
        matrix.nnz(),
        info.input_entries,
        read_ms
    );
    println!("WARNING: noncontiguous ABTM variants are NOT measured in MPI-PCG");
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output_path)?;
    let mut out = BufWriter::new(file);
    writeln!(
        out,
        "method,ownership_kind,ranks,trial,n,partition_ms,telemetry_ms,nnz,cut_nnz,communication_volume,directional_peer_relations,max_neighbors,min_owned_dofs,max_owned_dofs,min_local_nnz,max_local_nnz,owned_dof_imbalance,nnz_imbalance,configured_imbalance_per_mille,changed_vs_equal,graph_candidate_accepted,levels_built,abtm_metadata_bytes,scope"
    )?;
    let kinds = [
        Candidate::Equal,
        Candidate::ScalarGraph,
        Candidate::AbtmRegion,
        Candidate::AbtmMultisource,
        Candidate::AbtmMultilevel,
    ];
    let mut prior_owners: BTreeMap<(u32, &'static str), Vec<u32>> = BTreeMap::new();
    for ranks in ranks {
        let baseline = PartitionAssignment::from_contiguous(&ContiguousPartition::balanced(
            matrix.nrows() as u64,
            ranks,
        )?)?;
        for trial in 1..=repeats {
            // Rotate evaluation order to reduce systematic frequency/thermal bias.
            for shift in 0..kinds.len() {
                let kind = kinds[(shift + trial - 1) % kinds.len()];
                let started = Instant::now();
                let result = prepare_candidate(&matrix, ranks, kind, radius)?;
                let partition_ms = started.elapsed().as_secs_f64() * 1000.0;
                let telemetry_started = Instant::now();
                let telemetry = partition_telemetry_assignment(&matrix, &result.assignment)?;
                let telemetry_ms = telemetry_started.elapsed().as_secs_f64() * 1000.0;
                verify_telemetry(&matrix, ranks, &result.assignment, &telemetry)?;
                let key = (ranks, kind.label());
                if let Some(previous) = prior_owners.get(&key) {
                    if previous != result.assignment.owners() {
                        return Err(
                            format!("nondeterministic partition for {}", kind.label()).into()
                        );
                    }
                } else {
                    prior_owners.insert(key, result.assignment.owners().to_vec());
                }
                let changed = result.assignment.owners() != baseline.owners();
                write_row(
                    &mut out,
                    (ranks, trial, kind),
                    partition_ms,
                    telemetry_ms,
                    changed,
                    &result,
                    &telemetry,
                )?;
                println!(
                    "K2 QUALITY ranks={} trial={} method={} partition_ms={:.3} cut={} ghosts={} peers={} nnz_imbalance={:.4} changed={} accepted={:?}",
                    ranks,
                    trial,
                    kind.label(),
                    partition_ms,
                    telemetry.cut_nnz,
                    telemetry.communication_volume,
                    telemetry.directional_peer_relations,
                    telemetry.local_nnz_imbalance,
                    changed,
                    result.candidate_accepted,
                );
            }
        }
    }
    out.flush()?;
    println!(
        "=== K2 ABTM PARTITION QUALITY PASS: {} ===",
        output_path.display()
    );
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("K2 ABTM PARTITION QUALITY FAIL: {err}");
        std::process::exit(1);
    }
}
