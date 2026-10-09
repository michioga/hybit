use hybit_distributed::{
    abtm_balanced_multisource_partition, abtm_multilevel_partition, abtm_region_grow_partition,
    partition_telemetry_assignment, AbtmMultilevelOptions, ContiguousPartition,
    PartitionAssignment, PartitionTelemetry,
};
use hybit_matrix::read_matrix_market;
use std::env;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::time::Instant;

fn print_metrics(label: &str, metrics: &PartitionTelemetry) {
    println!("{label}");
    println!("  cut nnz              : {}", metrics.cut_nnz);
    println!("  communication volume : {}", metrics.communication_volume);
    println!(
        "  peer relations       : {}",
        metrics.directional_peer_relations
    );
    println!("  max neighbors        : {}", metrics.max_neighbors);
    println!(
        "  owned DOF min/max    : {} / {}",
        metrics.min_owned_dofs, metrics.max_owned_dofs
    );
    println!(
        "  owned DOF imbalance  : {:.6}",
        metrics.owned_dof_imbalance
    );
    println!(
        "  local nnz min/max    : {} / {}",
        metrics.min_local_nnz, metrics.max_local_nnz
    );
    println!(
        "  local nnz imbalance  : {:.6}",
        metrics.local_nnz_imbalance
    );
}

fn read_owner_labels(path: &str, ranks: u32) -> Result<PartitionAssignment, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut owners = Vec::new();

    for token in text.split_whitespace() {
        if token.starts_with('#') || token.starts_with('%') {
            continue;
        }
        owners.push(token.parse::<u32>()?);
    }

    Ok(PartitionAssignment::from_owners(ranks, owners)?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if !(args.len() == 3 || args.len() == 4) {
        eprintln!("usage: partition_quality_probe <matrix.mtx> <ranks> [owner_labels.txt]");
        eprintln!(
            "owner_labels.txt is zero-based: one rank id per global DOF (gpmetis .part.N compatible)"
        );
        std::process::exit(2);
    }

    let ranks: u32 = args[2].parse()?;
    let (matrix, info) = read_matrix_market(&args[1])?;

    let contiguous = ContiguousPartition::balanced(matrix.nrows() as u64, ranks)?;
    let contiguous_assignment = PartitionAssignment::from_contiguous(&contiguous)?;

    let t0 = Instant::now();
    let contiguous_metrics = partition_telemetry_assignment(&matrix, &contiguous_assignment)?;
    let contiguous_ms = t0.elapsed().as_secs_f64() * 1.0e3;

    let t1 = Instant::now();
    let (abtm_assignment, abtm_stats) = abtm_region_grow_partition(&matrix, ranks)?;
    let abtm_partition_ms = t1.elapsed().as_secs_f64() * 1.0e3;

    let t2 = Instant::now();
    let abtm_metrics = partition_telemetry_assignment(&matrix, &abtm_assignment)?;
    let abtm_metrics_ms = t2.elapsed().as_secs_f64() * 1.0e3;

    println!("HyBIT G8 partition quality probe");
    println!("matrix                  : {}", args[1]);
    println!("shape                   : {} x {}", info.nrows, info.ncols);
    println!("nnz                     : {}", matrix.nnz());
    println!("ranks                   : {}", ranks);
    println!();
    println!("contiguous metric ms    : {:.6}", contiguous_ms);
    print_metrics("contiguous baseline", &contiguous_metrics);
    println!();
    println!("ABTM partition ms       : {:.6}", abtm_partition_ms);
    println!("ABTM metric ms          : {:.6}", abtm_metrics_ms);
    println!(
        "ABTM topology bytes     : {}",
        abtm_stats.dual_topology_metadata_bytes
    );
    println!("ABTM seeds              : {}", abtm_stats.seeds_started);
    println!(
        "ABTM disconnected rest. : {}",
        abtm_stats.disconnected_restarts
    );
    println!(
        "ABTM frontier expanded  : {}",
        abtm_stats.frontier_nodes_expanded
    );
    println!(
        "ABTM topology words     : {}",
        abtm_stats.topology_words_visited
    );
    println!(
        "ABTM candidate bits     : {}",
        abtm_stats.candidate_neighbor_bits
    );
    print_metrics("ABTM region-growth prototype", &abtm_metrics);

    println!();
    println!(
        "ABTM cut delta          : {}",
        abtm_metrics.cut_nnz as i128 - contiguous_metrics.cut_nnz as i128
    );
    println!(
        "ABTM comm-volume delta  : {}",
        abtm_metrics.communication_volume as i128 - contiguous_metrics.communication_volume as i128
    );

    let t3 = Instant::now();
    let (multisource_assignment, multisource_stats) =
        abtm_balanced_multisource_partition(&matrix, ranks)?;
    let multisource_partition_ms = t3.elapsed().as_secs_f64() * 1.0e3;

    let t4 = Instant::now();
    let multisource_metrics = partition_telemetry_assignment(&matrix, &multisource_assignment)?;
    let multisource_metrics_ms = t4.elapsed().as_secs_f64() * 1.0e3;

    println!();
    println!("ABTM multisource ms     : {:.6}", multisource_partition_ms);
    println!("multisource metric ms   : {:.6}", multisource_metrics_ms);
    println!(
        "multisource topology B  : {}",
        multisource_stats.dual_topology_metadata_bytes
    );
    println!(
        "multisource seeds       : {}",
        multisource_stats.seeds_started
    );
    println!(
        "seed-distance BFS runs  : {}",
        multisource_stats.seed_distance_bfs_runs
    );
    println!(
        "multisource restarts    : {}",
        multisource_stats.disconnected_restarts
    );
    println!(
        "frontier nodes claimed  : {}",
        multisource_stats.frontier_nodes_claimed
    );
    println!(
        "multisource topo words  : {}",
        multisource_stats.topology_words_visited
    );
    println!(
        "multisource cand bits   : {}",
        multisource_stats.candidate_neighbor_bits
    );
    print_metrics("ABTM balanced multi-source", &multisource_metrics);
    println!(
        "multisource cut delta   : {}",
        multisource_metrics.cut_nnz as i128 - contiguous_metrics.cut_nnz as i128
    );
    println!(
        "multisource comm delta  : {}",
        multisource_metrics.communication_volume as i128
            - contiguous_metrics.communication_volume as i128
    );
    println!(
        "multisource vs A3 cut   : {}",
        multisource_metrics.cut_nnz as i128 - abtm_metrics.cut_nnz as i128
    );
    println!(
        "multisource vs A3 comm  : {}",
        multisource_metrics.communication_volume as i128
            - abtm_metrics.communication_volume as i128
    );

    let mut multilevel_options = AbtmMultilevelOptions::default();
    if let Ok(levels) = std::env::var("HYBIT_A6_MAX_LEVELS") {
        multilevel_options.max_levels = levels.parse()?;
    }
    if let Ok(vertices) = std::env::var("HYBIT_A6_COARSE_PER_RANK") {
        multilevel_options.coarse_vertices_per_rank = vertices.parse()?;
    }
    if let Ok(passes) = std::env::var("HYBIT_A6_REFINEMENT_PASSES") {
        multilevel_options.refinement_passes = passes.parse()?;
    }
    println!(
        "A6 options              : max_levels={}, coarse_per_rank={}, refinement_passes={}, imbalance_per_mille={}",
        multilevel_options.max_levels,
        multilevel_options.coarse_vertices_per_rank,
        multilevel_options.refinement_passes,
        multilevel_options.imbalance_per_mille
    );
    let t5 = Instant::now();
    let (multilevel_assignment, multilevel_stats) =
        abtm_multilevel_partition(&matrix, ranks, multilevel_options)?;
    let multilevel_partition_ms = t5.elapsed().as_secs_f64() * 1.0e3;

    let t6 = Instant::now();
    let multilevel_metrics = partition_telemetry_assignment(&matrix, &multilevel_assignment)?;
    let multilevel_metrics_ms = t6.elapsed().as_secs_f64() * 1.0e3;

    println!();
    // F14 diagnostic only: export complete owner labels for byte-exact
    // comparisons between reference, merged and production-default paths.
    if let Ok(path) = env::var("HYBIT_A6_F14_OWNER_EXPORT") {
        let mut writer = std::io::BufWriter::new(fs::File::create(&path)?);
        for &owner in multilevel_assignment.owners() {
            writeln!(writer, "{owner}")?;
        }
        writer.flush()?;
        println!("F14 owner dump          : {path}");
    }
    // F12: stable, cross-run owner-label hash, distinct from cut/halo telemetry.
    let mut owner_hash = 0xcbf29ce484222325u64;
    for &owner in multilevel_assignment.owners() {
        for byte in owner.to_le_bytes() {
            owner_hash ^= u64::from(byte);
            owner_hash = owner_hash.wrapping_mul(0x100000001b3);
        }
    }
    println!("multilevel owner FNV64   : {:016x}", owner_hash);
    println!("ABTM multilevel ms      : {:.6}", multilevel_partition_ms);
    println!("multilevel metric ms    : {:.6}", multilevel_metrics_ms);
    println!(
        "multilevel levels       : {}",
        multilevel_stats.levels_built
    );
    println!(
        "coarsest vertices       : {}",
        multilevel_stats.coarsest_vertices
    );
    println!(
        "matched pairs           : {}",
        multilevel_stats.matched_pairs
    );
    println!(
        "singleton aggregates    : {}",
        multilevel_stats.singleton_aggregates
    );
    println!(
        "coarse restarts         : {}",
        multilevel_stats.coarse_restarts
    );
    println!(
        "refinement moves        : {}",
        multilevel_stats.refinement_moves
    );
    print_metrics("ABTM G8-A6 multilevel", &multilevel_metrics);
    println!(
        "multilevel vs A4 cut    : {}",
        multilevel_metrics.cut_nnz as i128 - multisource_metrics.cut_nnz as i128
    );
    println!(
        "multilevel vs A4 comm   : {}",
        multilevel_metrics.communication_volume as i128
            - multisource_metrics.communication_volume as i128
    );
    println!(
        "multilevel vs contig cut: {}",
        multilevel_metrics.cut_nnz as i128 - contiguous_metrics.cut_nnz as i128
    );
    println!(
        "multilevel vs contig com: {}",
        multilevel_metrics.communication_volume as i128
            - contiguous_metrics.communication_volume as i128
    );
    if args.len() == 4 {
        let external = read_owner_labels(&args[3], ranks)?;
        if external.global_dofs() != matrix.nrows() as u64 {
            return Err(format!(
                "external labels contain {} DOFs; matrix has {}",
                external.global_dofs(),
                matrix.nrows()
            )
            .into());
        }

        let t7 = Instant::now();
        let external_metrics = partition_telemetry_assignment(&matrix, &external)?;
        let external_ms = t7.elapsed().as_secs_f64() * 1.0e3;

        println!();
        println!("external labels         : {}", args[3]);
        println!("external metric ms      : {:.6}", external_ms);
        print_metrics("external partition", &external_metrics);
    }

    Ok(())
}
