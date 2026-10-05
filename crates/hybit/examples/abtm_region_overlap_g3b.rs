use std::env;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use hybit_matrix::{
    grow_undirected_region, read_matrix_market, region_multiplicity, AbtmDualTopology, Csr32Matrix,
    DofMask,
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
                        "Usage: abtm_region_overlap_g3b --matrix A.mtx [--regions N] [--hops N] [--repeats N]"
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

fn reference_counts(node_count: usize, regions: &[DofMask]) -> Result<Vec<u32>, Box<dyn Error>> {
    let mut counts = vec![0u32; node_count];
    for region in regions {
        if region.len() != node_count {
            return Err("reference region length mismatch".into());
        }
        for node in region.indices() {
            counts[node] = counts[node]
                .checked_add(1)
                .ok_or("reference multiplicity overflow")?;
        }
    }
    Ok(counts)
}

fn pairwise_overlap_stats(regions: &[DofMask]) -> Result<(usize, u64, usize), Box<dyn Error>> {
    let mut overlapping_pairs = 0usize;
    let mut pair_overlap_memberships = 0u64;
    let mut max_pair_intersection = 0usize;

    for i in 0..regions.len() {
        for j in (i + 1)..regions.len() {
            if regions[i].len() != regions[j].len() {
                return Err("pairwise region length mismatch".into());
            }
            let intersection = regions[i]
                .words()
                .iter()
                .zip(regions[j].words())
                .map(|(a, b)| (a & b).count_ones() as usize)
                .sum::<usize>();
            if intersection != 0 {
                overlapping_pairs += 1;
                pair_overlap_memberships = pair_overlap_memberships
                    .checked_add(intersection as u64)
                    .ok_or("pairwise overlap overflow")?;
                max_pair_intersection = max_pair_intersection.max(intersection);
            }
        }
    }

    Ok((
        overlapping_pairs,
        pair_overlap_memberships,
        max_pair_intersection,
    ))
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
        let mut candidate = (mix(ordinal ^ 0x38d0_2026_1005_0320) % n as u64) as usize;
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
        "HyBIT {} ABTM G3b overlap and multiplicity",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!("regions             : {}", args.regions);
    println!("hops                : {}", args.hops);
    println!("repeats             : {}", args.repeats);

    let (matrix, info) = read_matrix_market(&args.matrix)?;
    if matrix.nrows() != matrix.ncols() {
        return Err("G3b requires a square matrix".into());
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

    let seeds = deterministic_seeds(matrix.nrows(), args.regions);
    let seed_masks = seeds
        .iter()
        .map(|&seed| DofMask::from_indices(matrix.nrows(), &[seed]))
        .collect::<Result<Vec<_>, _>>()?;

    let mut csr_regions = Vec::with_capacity(seeds.len());
    let mut abtm_regions = Vec::with_capacity(seeds.len());

    for (index, &seed) in seeds.iter().enumerate() {
        let csr = csr_grow_undirected_region(&matrix, &transpose, seed, args.hops)?;
        let abtm = grow_undirected_region(&dual, &seed_masks[index], args.hops)?;
        if abtm.region() != &csr {
            return Err(format!(
                "G3b region mismatch for seed {seed}: ABTM nodes={}, CSR nodes={}",
                abtm.region().count_ones(),
                csr.count_ones()
            )
            .into());
        }
        csr_regions.push(csr);
        abtm_regions.push(abtm.region().clone());
    }

    let multiplicity = region_multiplicity(matrix.nrows(), &abtm_regions)?;
    let reference = reference_counts(matrix.nrows(), &csr_regions)?;
    if multiplicity.counts() != reference.as_slice() {
        return Err("G3b multiplicity mismatch against reference counts".into());
    }

    let (overlapping_pairs, pair_overlap_reference, max_pair_intersection) =
        pairwise_overlap_stats(&abtm_regions)?;
    let stats = multiplicity.stats();
    if pair_overlap_reference != stats.pair_overlap_memberships {
        return Err(format!(
            "G3b pair-overlap identity mismatch: pairwise={pair_overlap_reference}, multiplicity={}",
            stats.pair_overlap_memberships
        )
        .into());
    }

    let possible_pairs = seeds.len().saturating_mul(seeds.len().saturating_sub(1)) / 2;
    let pair_overlap_fraction = if possible_pairs == 0 {
        0.0
    } else {
        overlapping_pairs as f64 / possible_pairs as f64
    };
    let region_mask_bytes = abtm_regions
        .len()
        .saturating_mul(matrix.nrows().div_ceil(64))
        .saturating_mul(std::mem::size_of::<u64>());

    println!(
        "G3B_PREPARE|regions={}|hops={}|transpose_prepare_ms={transpose_prepare_ms:.6}|dual_prepare_ms={dual_prepare_ms:.6}|region_mask_bytes={region_mask_bytes}",
        seeds.len(),
        args.hops,
    );

    println!(
        "G3B_OVERLAP|regions={}|hops={}|total_memberships={}|covered_nodes={}|overlap_nodes={}|extra_memberships={}|average_multiplicity={:.9e}|overlap_fraction={:.9e}|max_multiplicity={}|overlapping_pairs={overlapping_pairs}|possible_pairs={possible_pairs}|pair_overlap_fraction={pair_overlap_fraction:.9e}|pair_overlap_memberships={}|max_pair_intersection={max_pair_intersection}|mismatched_regions=0|mismatched_multiplicity=0",
        stats.region_count,
        args.hops,
        stats.total_memberships,
        stats.covered_nodes,
        stats.overlap_nodes,
        stats.extra_memberships(),
        stats.average_multiplicity_on_covered(),
        stats.overlap_fraction_on_covered(),
        stats.max_multiplicity,
        stats.pair_overlap_memberships,
    );

    let mut csr_pipeline_samples = Vec::with_capacity(args.repeats);
    let mut abtm_pipeline_samples = Vec::with_capacity(args.repeats);
    let mut multiplicity_samples = Vec::with_capacity(args.repeats);

    for _ in 0..args.repeats {
        let start = Instant::now();
        let mut regions = Vec::with_capacity(seeds.len());
        for &seed in &seeds {
            regions.push(csr_grow_undirected_region(
                &matrix, &transpose, seed, args.hops,
            )?);
        }
        let m = region_multiplicity(matrix.nrows(), &regions)?;
        csr_pipeline_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box(m.stats().total_memberships);

        let start = Instant::now();
        let mut regions = Vec::with_capacity(seeds.len());
        for seed in &seed_masks {
            regions.push(
                grow_undirected_region(&dual, seed, args.hops)?
                    .region()
                    .clone(),
            );
        }
        let m = region_multiplicity(matrix.nrows(), &regions)?;
        abtm_pipeline_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box(m.stats().total_memberships);

        let start = Instant::now();
        let m = region_multiplicity(matrix.nrows(), &abtm_regions)?;
        multiplicity_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box(m.stats().pair_overlap_memberships);
    }

    let csr_pipeline_ms = median(&mut csr_pipeline_samples);
    let abtm_pipeline_ms = median(&mut abtm_pipeline_samples);
    let multiplicity_ms = median(&mut multiplicity_samples);
    let abtm_over_csr_pipeline = if csr_pipeline_ms == 0.0 {
        0.0
    } else {
        abtm_pipeline_ms / csr_pipeline_ms
    };

    println!(
        "G3B_TIMING|regions={}|hops={}|csr_pipeline_ms={csr_pipeline_ms:.6}|abtm_pipeline_ms={abtm_pipeline_ms:.6}|abtm_over_csr_pipeline={abtm_over_csr_pipeline:.9e}|multiplicity_ms={multiplicity_ms:.6}",
        seeds.len(),
        args.hops,
    );

    Ok(())
}
