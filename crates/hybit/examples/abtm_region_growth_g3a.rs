use std::env;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use hybit_matrix::{
    grow_undirected_region, read_matrix_market, AbtmDualTopology, Csr32Matrix, DofMask,
};

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    regions: usize,
    hops: usize,
    repeats: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut regions = 64usize;
        let mut hops = 2usize;
        let mut repeats = 5usize;
        let mut it = env::args().skip(1);

        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => {
                    matrix = Some(PathBuf::from(
                        it.next().ok_or("missing value after --matrix")?,
                    ));
                }
                "--regions" => {
                    regions = it.next().ok_or("missing value after --regions")?.parse()?;
                    if regions == 0 {
                        return Err("--regions must be >= 1".into());
                    }
                }
                "--hops" => {
                    hops = it.next().ok_or("missing value after --hops")?.parse()?;
                }
                "--repeats" => {
                    repeats = it.next().ok_or("missing value after --repeats")?.parse()?;
                    if repeats == 0 {
                        return Err("--repeats must be >= 1".into());
                    }
                }
                "-h" | "--help" => {
                    println!(
                        "Usage: abtm_region_growth_g3a --matrix A.mtx [--regions N] [--hops N] [--repeats N]"
                    );
                    std::process::exit(0);
                }
                other if !other.starts_with('-') && matrix.is_none() => {
                    matrix = Some(PathBuf::from(other));
                }
                other => return Err(format!("unknown argument '{other}'").into()),
            }
        }

        Ok(Self {
            matrix: matrix.ok_or("missing matrix path; use --matrix FILE.mtx")?,
            regions,
            hops,
            repeats,
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CsrGrowthStats {
    hops_completed: usize,
    frontier_nodes_visited: usize,
    neighbor_entries_visited: usize,
}

fn transpose_csr(matrix: &Csr32Matrix) -> Result<Csr32Matrix, Box<dyn Error>> {
    let mut counts = vec![0u32; matrix.ncols()];
    for &col in matrix.col_idx() {
        counts[col as usize] = counts[col as usize]
            .checked_add(1)
            .ok_or("transpose count overflow")?;
    }

    let mut row_ptr = Vec::with_capacity(matrix.ncols() + 1);
    row_ptr.push(0u32);
    let mut running = 0u32;
    for count in counts {
        running = running
            .checked_add(count)
            .ok_or("transpose row pointer overflow")?;
        row_ptr.push(running);
    }

    let mut col_idx = vec![0u32; matrix.nnz()];
    let mut values = vec![0.0f64; matrix.nnz()];
    let mut cursors = row_ptr[..matrix.ncols()].to_vec();

    for row in 0..matrix.nrows() {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        for p in start..end {
            let col = matrix.col_idx()[p] as usize;
            let dst = cursors[col] as usize;
            col_idx[dst] = u32::try_from(row).map_err(|_| "transpose row index overflow")?;
            values[dst] = matrix.values()[p];
            cursors[col] = cursors[col]
                .checked_add(1)
                .ok_or("transpose cursor overflow")?;
        }
    }

    Ok(Csr32Matrix::new(
        matrix.ncols(),
        matrix.nrows(),
        row_ptr,
        col_idx,
        values,
    )?)
}

fn csr_grow_undirected_region(
    matrix: &Csr32Matrix,
    transpose: &Csr32Matrix,
    seed: usize,
    hops: usize,
) -> Result<(DofMask, CsrGrowthStats), Box<dyn Error>> {
    let n = matrix.nrows();
    let mut region = DofMask::from_indices(n, &[seed])?;
    let mut frontier = DofMask::from_indices(n, &[seed])?;
    let mut stats = CsrGrowthStats::default();

    for _ in 0..hops {
        if frontier.is_empty() {
            break;
        }

        let frontier_indices = frontier.indices();
        stats.frontier_nodes_visited += frontier_indices.len();
        let mut next = DofMask::new(n);

        for node in frontier_indices {
            let start = matrix.row_ptr()[node] as usize;
            let end = matrix.row_ptr()[node + 1] as usize;
            stats.neighbor_entries_visited += end - start;
            for &neighbor in &matrix.col_idx()[start..end] {
                let neighbor = neighbor as usize;
                if !region.contains(neighbor) {
                    next.set(neighbor, true)?;
                }
            }

            let start = transpose.row_ptr()[node] as usize;
            let end = transpose.row_ptr()[node + 1] as usize;
            stats.neighbor_entries_visited += end - start;
            for &neighbor in &transpose.col_idx()[start..end] {
                let neighbor = neighbor as usize;
                if !region.contains(neighbor) {
                    next.set(neighbor, true)?;
                }
            }
        }

        region.union_assign(&next)?;
        frontier = next;
        stats.hops_completed += 1;
    }

    Ok((region, stats))
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn deterministic_seeds(n: usize, requested: usize) -> Vec<usize> {
    let target = requested.min(n);
    let mut chosen = DofMask::new(n);
    let mut seeds = Vec::with_capacity(target);
    let mut ordinal = 0u64;

    while seeds.len() < target {
        let mut candidate = (mix(ordinal ^ 0x38d0_2026_1005_0310) % n as u64) as usize;
        while chosen.contains(candidate) {
            candidate += 1;
            if candidate == n {
                candidate = 0;
            }
        }
        chosen
            .set(candidate, true)
            .expect("deterministic seed index must be valid");
        seeds.push(candidate);
        ordinal = ordinal.wrapping_add(1);
    }

    seeds.sort_unstable();
    seeds
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;

    println!(
        "HyBIT {} ABTM G3a undirected region growth",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!("regions             : {}", args.regions);
    println!("hops                : {}", args.hops);
    println!("repeats             : {}", args.repeats);

    let (matrix, info) = read_matrix_market(&args.matrix)?;
    if matrix.nrows() != matrix.ncols() {
        return Err("G3a region growth requires a square matrix".into());
    }

    println!(
        "Matrix Market       : {:?}, {} input entries -> {} CSR nnz",
        info.symmetry, info.input_entries, info.csr_nnz
    );
    println!(
        "dimensions          : {} x {}",
        matrix.nrows(),
        matrix.ncols()
    );
    println!("stored CSR nnz      : {}", matrix.nnz());

    let transpose_start = Instant::now();
    let transpose = transpose_csr(&matrix)?;
    let transpose_prepare_ms = transpose_start.elapsed().as_secs_f64() * 1.0e3;

    let dual_start = Instant::now();
    let dual = AbtmDualTopology::from_csr32(&matrix)?;
    let dual_prepare_ms = dual_start.elapsed().as_secs_f64() * 1.0e3;

    let explicit_metadata_bytes = matrix
        .metadata_bytes()
        .saturating_add(transpose.metadata_bytes());

    println!(
        "G3A_PREPARE|structural_nnz={}|explicit_dual_metadata_bytes={explicit_metadata_bytes}|transpose_prepare_ms={transpose_prepare_ms:.6}|dual_prepare_ms={dual_prepare_ms:.6}",
        matrix.nnz(),
    );

    let seeds = deterministic_seeds(matrix.nrows(), args.regions);

    let mut total_region_nodes = 0usize;
    let mut total_frontier_visits = 0usize;
    let mut total_topology_words = 0usize;
    let mut total_candidate_neighbor_bits = 0usize;
    let mut total_csr_neighbor_entries = 0usize;

    for &seed in &seeds {
        let seed_mask = DofMask::from_indices(matrix.nrows(), &[seed])?;
        let abtm = grow_undirected_region(&dual, &seed_mask, args.hops)?;
        let (csr, csr_stats) = csr_grow_undirected_region(&matrix, &transpose, seed, args.hops)?;

        if abtm.region() != &csr {
            return Err(format!(
                "G3a region mismatch for seed {seed}: ABTM nodes={}, CSR nodes={}",
                abtm.region().count_ones(),
                csr.count_ones()
            )
            .into());
        }

        let stats = abtm.stats();
        total_region_nodes = total_region_nodes.saturating_add(stats.region_nodes);
        total_frontier_visits = total_frontier_visits.saturating_add(stats.frontier_nodes_visited);
        total_topology_words = total_topology_words.saturating_add(stats.topology_words_visited());
        total_candidate_neighbor_bits =
            total_candidate_neighbor_bits.saturating_add(stats.candidate_neighbor_bits);
        total_csr_neighbor_entries =
            total_csr_neighbor_entries.saturating_add(csr_stats.neighbor_entries_visited);
    }

    let mut csr_samples = Vec::with_capacity(args.repeats);
    let mut abtm_samples = Vec::with_capacity(args.repeats);

    for _ in 0..args.repeats {
        let start = Instant::now();
        let mut checksum = 0usize;
        for &seed in &seeds {
            let (region, _) = csr_grow_undirected_region(&matrix, &transpose, seed, args.hops)?;
            checksum = checksum.wrapping_add(region.count_ones());
        }
        csr_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box(checksum);

        let start = Instant::now();
        let mut checksum = 0usize;
        for &seed in &seeds {
            let seed_mask = DofMask::from_indices(matrix.nrows(), &[seed])?;
            let growth = grow_undirected_region(&dual, &seed_mask, args.hops)?;
            checksum = checksum.wrapping_add(growth.region().count_ones());
        }
        abtm_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box(checksum);
    }

    let csr_ms = median(&mut csr_samples);
    let abtm_ms = median(&mut abtm_samples);
    let abtm_over_csr = if csr_ms == 0.0 { 0.0 } else { abtm_ms / csr_ms };
    let average_region_nodes = total_region_nodes as f64 / seeds.len() as f64;
    let word_over_entries = if total_csr_neighbor_entries == 0 {
        0.0
    } else {
        total_topology_words as f64 / total_csr_neighbor_entries as f64
    };

    println!(
        "G3A_REGION|regions={}|hops={}|total_region_nodes={total_region_nodes}|average_region_nodes={average_region_nodes:.9e}|frontier_nodes_visited={total_frontier_visits}|topology_words_visited={total_topology_words}|candidate_neighbor_bits={total_candidate_neighbor_bits}|csr_neighbor_entries_visited={total_csr_neighbor_entries}|word_over_csr_entries={word_over_entries:.9e}|csr_ms={csr_ms:.6}|abtm_ms={abtm_ms:.6}|abtm_over_csr={abtm_over_csr:.9e}|mismatched_regions=0",
        seeds.len(),
        args.hops,
    );

    Ok(())
}
