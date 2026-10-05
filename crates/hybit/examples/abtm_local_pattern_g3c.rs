use std::env;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use hybit_matrix::{
    extract_local_submatrix_pattern, grow_undirected_region, read_matrix_market, AbtmDualTopology,
    Csr32Matrix, DofMask,
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
                        "Usage: abtm_local_pattern_g3c --matrix A.mtx [--regions N] [--hops N] [--repeats N]"
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct ReferencePattern {
    global_nodes: Vec<u32>,
    row_ptr: Vec<u32>,
    col_idx: Vec<u32>,
    csr_entries_scanned: usize,
    mapping_scratch_bytes: usize,
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
) -> Result<DofMask, Box<dyn Error>> {
    let n = matrix.nrows();
    let mut region = DofMask::from_indices(n, &[seed])?;
    let mut frontier = DofMask::from_indices(n, &[seed])?;

    for _ in 0..hops {
        if frontier.is_empty() {
            break;
        }

        let frontier_indices = frontier.indices();
        let mut next = DofMask::new(n);

        for node in frontier_indices {
            let start = matrix.row_ptr()[node] as usize;
            let end = matrix.row_ptr()[node + 1] as usize;
            for &neighbor in &matrix.col_idx()[start..end] {
                let neighbor = neighbor as usize;
                if !region.contains(neighbor) {
                    next.set(neighbor, true)?;
                }
            }

            let start = transpose.row_ptr()[node] as usize;
            let end = transpose.row_ptr()[node + 1] as usize;
            for &neighbor in &transpose.col_idx()[start..end] {
                let neighbor = neighbor as usize;
                if !region.contains(neighbor) {
                    next.set(neighbor, true)?;
                }
            }
        }

        region.union_assign(&next)?;
        frontier = next;
    }

    Ok(region)
}

fn reference_pattern(
    matrix: &Csr32Matrix,
    region: &DofMask,
) -> Result<ReferencePattern, Box<dyn Error>> {
    let global_indices = region.indices();
    if global_indices.len() > u32::MAX as usize {
        return Err("reference local node count exceeds u32".into());
    }

    let mut global_nodes = Vec::with_capacity(global_indices.len());
    let mut global_to_local = vec![u32::MAX; matrix.nrows()];

    for (local, &global) in global_indices.iter().enumerate() {
        global_nodes.push(u32::try_from(global)?);
        global_to_local[global] = u32::try_from(local)?;
    }

    let mut row_ptr = Vec::with_capacity(global_nodes.len() + 1);
    let mut col_idx = Vec::new();
    let mut csr_entries_scanned = 0usize;
    row_ptr.push(0);

    for &global_row in &global_nodes {
        let row = global_row as usize;
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        csr_entries_scanned = csr_entries_scanned.saturating_add(end - start);

        for &global_col in &matrix.col_idx()[start..end] {
            let local_col = global_to_local[global_col as usize];
            if local_col != u32::MAX {
                col_idx.push(local_col);
            }
        }

        if col_idx.len() > u32::MAX as usize {
            return Err("reference local nnz exceeds u32".into());
        }
        row_ptr.push(col_idx.len() as u32);
    }

    Ok(ReferencePattern {
        global_nodes,
        row_ptr,
        col_idx,
        csr_entries_scanned,
        mapping_scratch_bytes: global_to_local
            .len()
            .saturating_mul(std::mem::size_of::<u32>()),
    })
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
        let mut candidate = (mix(ordinal ^ 0x38d0_2026_1005_0330) % n as u64) as usize;
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
        "HyBIT {} ABTM G3c structural local-submatrix extraction",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!("regions             : {}", args.regions);
    println!("hops                : {}", args.hops);
    println!("repeats             : {}", args.repeats);

    let (matrix, info) = read_matrix_market(&args.matrix)?;
    if matrix.nrows() != matrix.ncols() {
        return Err("G3c requires a square matrix".into());
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

    let transpose = transpose_csr(&matrix)?;

    let dual_start = Instant::now();
    let dual = AbtmDualTopology::from_csr32(&matrix)?;
    let dual_prepare_ms = dual_start.elapsed().as_secs_f64() * 1.0e3;

    let seeds = deterministic_seeds(matrix.nrows(), args.regions);
    let mut regions = Vec::with_capacity(seeds.len());

    for &seed in &seeds {
        let seed_mask = DofMask::from_indices(matrix.nrows(), &[seed])?;
        let abtm = grow_undirected_region(&dual, &seed_mask, args.hops)?;
        let csr = csr_grow_undirected_region(&matrix, &transpose, seed, args.hops)?;
        if abtm.region() != &csr {
            return Err(format!("G3c region mismatch for seed {seed}").into());
        }
        regions.push(abtm.region().clone());
    }

    let mut total_local_nodes = 0usize;
    let mut total_local_nnz = 0usize;
    let mut total_words = 0usize;
    let mut total_candidates = 0usize;
    let mut total_kept = 0usize;
    let mut total_csr_entries = 0usize;
    let mut total_pattern_bytes = 0usize;
    let mut peak_mapping_scratch_bytes = 0usize;

    for (region_index, region) in regions.iter().enumerate() {
        let reference = reference_pattern(&matrix, region)?;
        let abtm = extract_local_submatrix_pattern(&dual, region)?;

        if abtm.global_nodes() != reference.global_nodes.as_slice()
            || abtm.row_ptr() != reference.row_ptr.as_slice()
            || abtm.col_idx() != reference.col_idx.as_slice()
        {
            return Err(format!("G3c local pattern mismatch for region {region_index}").into());
        }

        let stats = abtm.stats();
        total_local_nodes = total_local_nodes.saturating_add(stats.local_nodes);
        total_local_nnz = total_local_nnz.saturating_add(stats.local_nnz);
        total_words = total_words.saturating_add(stats.topology_words_visited);
        total_candidates = total_candidates.saturating_add(stats.candidate_neighbor_bits);
        total_kept = total_kept.saturating_add(stats.kept_neighbor_bits);
        total_csr_entries = total_csr_entries.saturating_add(reference.csr_entries_scanned);
        total_pattern_bytes = total_pattern_bytes.saturating_add(stats.pattern_bytes);
        peak_mapping_scratch_bytes = peak_mapping_scratch_bytes
            .max(stats.mapping_scratch_bytes)
            .max(reference.mapping_scratch_bytes);
    }

    let average_local_nodes = total_local_nodes as f64 / regions.len() as f64;
    let average_local_nnz = total_local_nnz as f64 / regions.len() as f64;
    let retained_fraction = if total_candidates == 0 {
        0.0
    } else {
        total_kept as f64 / total_candidates as f64
    };
    let pruning_ratio = 1.0 - retained_fraction;
    let word_over_csr_entries = if total_csr_entries == 0 {
        0.0
    } else {
        total_words as f64 / total_csr_entries as f64
    };

    println!(
        "G3C_PATTERN|regions={}|hops={}|total_local_nodes={total_local_nodes}|average_local_nodes={average_local_nodes:.9e}|total_local_nnz={total_local_nnz}|average_local_nnz={average_local_nnz:.9e}|topology_words_visited={total_words}|candidate_neighbor_bits={total_candidates}|kept_neighbor_bits={total_kept}|csr_entries_scanned={total_csr_entries}|retained_fraction={retained_fraction:.9e}|pruning_ratio={pruning_ratio:.9e}|word_over_csr_entries={word_over_csr_entries:.9e}|total_pattern_bytes={total_pattern_bytes}|peak_mapping_scratch_bytes={peak_mapping_scratch_bytes}|mismatched_regions=0|mismatched_patterns=0",
        regions.len(),
        args.hops,
    );

    println!(
        "G3C_PREPARE|dual_prepare_ms={dual_prepare_ms:.6}|regions={}|hops={}",
        regions.len(),
        args.hops,
    );

    let mut csr_samples = Vec::with_capacity(args.repeats);
    let mut abtm_samples = Vec::with_capacity(args.repeats);

    for _ in 0..args.repeats {
        let start = Instant::now();
        let mut checksum = 0usize;
        for region in &regions {
            let pattern = reference_pattern(&matrix, region)?;
            checksum = checksum.wrapping_add(pattern.col_idx.len());
        }
        csr_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box(checksum);

        let start = Instant::now();
        let mut checksum = 0usize;
        for region in &regions {
            let pattern = extract_local_submatrix_pattern(&dual, region)?;
            checksum = checksum.wrapping_add(pattern.nnz());
        }
        abtm_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box(checksum);
    }

    let csr_ms = median(&mut csr_samples);
    let abtm_ms = median(&mut abtm_samples);
    let abtm_over_csr = if csr_ms == 0.0 { 0.0 } else { abtm_ms / csr_ms };

    println!(
        "G3C_TIMING|regions={}|hops={}|csr_extract_ms={csr_ms:.6}|abtm_extract_ms={abtm_ms:.6}|abtm_over_csr_extract={abtm_over_csr:.9e}",
        regions.len(),
        args.hops,
    );

    Ok(())
}
