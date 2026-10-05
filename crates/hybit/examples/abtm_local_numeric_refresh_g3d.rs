use std::env;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use hybit_matrix::{
    grow_undirected_region, prepare_local_numeric_plan, read_matrix_market, AbtmDualTopology,
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
                        "Usage: abtm_local_numeric_refresh_g3d --matrix A.mtx [--regions N] [--hops N] [--repeats N]"
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

#[derive(Clone, Debug)]
struct ReferenceLocalNumeric {
    global_nodes: Vec<u32>,
    row_ptr: Vec<u32>,
    col_idx: Vec<u32>,
    values: Vec<f64>,
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

fn direct_numeric_extract(
    matrix: &Csr32Matrix,
    values: &[f64],
    region: &DofMask,
) -> Result<ReferenceLocalNumeric, Box<dyn Error>> {
    if values.len() != matrix.values().len() {
        return Err("direct numeric values length mismatch".into());
    }

    let global_indices = region.indices();
    let mut global_nodes = Vec::with_capacity(global_indices.len());
    let mut global_to_local = vec![u32::MAX; matrix.ncols()];

    for (local, &global) in global_indices.iter().enumerate() {
        global_nodes.push(u32::try_from(global)?);
        global_to_local[global] = u32::try_from(local)?;
    }

    let mut row_ptr = Vec::with_capacity(global_nodes.len() + 1);
    let mut col_idx = Vec::new();
    let mut local_values = Vec::new();
    row_ptr.push(0);

    for &global_row in &global_nodes {
        let row = global_row as usize;
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;

        let mut row_entries: std::collections::BTreeMap<u32, f64> =
            std::collections::BTreeMap::new();

        for (offset, &value) in values[start..end].iter().enumerate() {
            let p = start + offset;
            let global_col = matrix.col_idx()[p] as usize;
            let local_col = global_to_local[global_col];
            if local_col != u32::MAX {
                *row_entries.entry(local_col).or_default() += value;
            }
        }

        for (local_col, value) in row_entries {
            col_idx.push(local_col);
            local_values.push(value);
        }

        if col_idx.len() > u32::MAX as usize {
            return Err("direct local nnz exceeds u32".into());
        }
        row_ptr.push(col_idx.len() as u32);
    }

    Ok(ReferenceLocalNumeric {
        global_nodes,
        row_ptr,
        col_idx,
        values: local_values,
    })
}

fn refreshed_values(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let phase = ((index as u64).wrapping_mul(0x9e37_79b9) & 0xffff) as f64;
            value * (1.0 + phase * 1.0e-10) + ((index % 7) as f64 - 3.0) * 1.0e-13
        })
        .collect()
}

fn max_scaled_error(reference: &[f64], actual: &[f64]) -> f64 {
    reference
        .iter()
        .zip(actual)
        .map(|(&a, &b)| (a - b).abs() / a.abs().max(1.0))
        .fold(0.0f64, f64::max)
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
        let mut candidate = (mix(ordinal ^ 0x38d0_2026_1005_0340) % n as u64) as usize;
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
        "HyBIT {} ABTM G3d prepared local numeric refresh",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!("regions             : {}", args.regions);
    println!("hops                : {}", args.hops);
    println!("repeats             : {}", args.repeats);

    let (matrix, info) = read_matrix_market(&args.matrix)?;
    if matrix.nrows() != matrix.ncols() {
        return Err("G3d requires a square matrix".into());
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
            return Err(format!("G3d region mismatch for seed {seed}").into());
        }
        regions.push(abtm.region().clone());
    }

    let plan_start = Instant::now();
    let plans = regions
        .iter()
        .map(|region| prepare_local_numeric_plan(&dual, &matrix, region))
        .collect::<Result<Vec<_>, _>>()?;
    let plan_prepare_ms = plan_start.elapsed().as_secs_f64() * 1.0e3;

    let refresh = refreshed_values(matrix.values());

    let mut total_local_nodes = 0usize;
    let mut total_local_nnz = 0usize;
    let mut total_source_terms = 0usize;
    let mut duplicate_source_terms = 0usize;
    let mut direct_source_plans = 0usize;
    let mut total_plan_bytes = 0usize;
    let mut max_error_original = 0.0f64;
    let mut max_error_refresh = 0.0f64;

    for (index, (plan, region)) in plans.iter().zip(&regions).enumerate() {
        let direct_original = direct_numeric_extract(&matrix, matrix.values(), region)?;
        let direct_refresh = direct_numeric_extract(&matrix, &refresh, region)?;

        if plan.pattern().global_nodes() != direct_original.global_nodes.as_slice()
            || plan.pattern().row_ptr() != direct_original.row_ptr.as_slice()
            || plan.pattern().col_idx() != direct_original.col_idx.as_slice()
        {
            return Err(format!("G3d pattern mismatch for region {index}").into());
        }

        let gathered_original = plan.gather_values(matrix.values())?;
        let gathered_refresh = plan.gather_values(&refresh)?;

        if gathered_original.len() != direct_original.values.len()
            || gathered_refresh.len() != direct_refresh.values.len()
        {
            return Err(format!("G3d value length mismatch for region {index}").into());
        }

        max_error_original = max_error_original.max(max_scaled_error(
            &direct_original.values,
            &gathered_original,
        ));
        max_error_refresh =
            max_error_refresh.max(max_scaled_error(&direct_refresh.values, &gathered_refresh));

        let stats = plan.stats();
        total_local_nodes = total_local_nodes.saturating_add(stats.local_nodes);
        total_local_nnz = total_local_nnz.saturating_add(stats.local_nnz);
        total_source_terms = total_source_terms.saturating_add(stats.source_terms);
        duplicate_source_terms =
            duplicate_source_terms.saturating_add(stats.duplicate_source_terms);
        direct_source_plans += usize::from(stats.direct_source_map);
        total_plan_bytes = total_plan_bytes.saturating_add(stats.plan_bytes);
    }

    const VALIDATION_TOLERANCE: f64 = 1.0e-10;
    let max_scaled_error = max_error_original.max(max_error_refresh);
    if max_scaled_error > VALIDATION_TOLERANCE {
        return Err(format!(
            "G3d numerical refresh mismatch: max_scaled_error={max_scaled_error:.3e}"
        )
        .into());
    }

    println!(
        "G3D_PLAN|regions={}|hops={}|dual_prepare_ms={dual_prepare_ms:.6}|plan_prepare_ms={plan_prepare_ms:.6}|total_local_nodes={total_local_nodes}|total_local_nnz={total_local_nnz}|source_terms={total_source_terms}|duplicate_source_terms={duplicate_source_terms}|direct_source_plans={direct_source_plans}|total_plan_bytes={total_plan_bytes}",
        regions.len(),
        args.hops,
    );

    println!(
        "G3D_VALIDATE|validation_tolerance={VALIDATION_TOLERANCE:.9e}|max_scaled_error_original={max_error_original:.9e}|max_scaled_error_refresh={max_error_refresh:.9e}|mismatched_regions=0|mismatched_patterns=0|mismatched_values=0"
    );

    let mut direct_samples = Vec::with_capacity(args.repeats);
    let mut refresh_samples = Vec::with_capacity(args.repeats);

    for repeat in 0..args.repeats {
        let values = if repeat % 2 == 0 {
            matrix.values()
        } else {
            refresh.as_slice()
        };

        let start = Instant::now();
        let mut checksum = 0.0f64;
        for region in &regions {
            let local = direct_numeric_extract(&matrix, values, region)?;
            checksum += local.values.iter().sum::<f64>();
        }
        direct_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box(checksum);

        let start = Instant::now();
        let mut checksum = 0.0f64;
        for plan in &plans {
            let local = plan.gather_values(values)?;
            checksum += local.iter().sum::<f64>();
        }
        refresh_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box(checksum);
    }

    let direct_ms = median(&mut direct_samples);
    let refresh_ms = median(&mut refresh_samples);
    let refresh_over_direct = if direct_ms == 0.0 {
        0.0
    } else {
        refresh_ms / direct_ms
    };

    let saved_per_refresh_ms = (direct_ms - refresh_ms).max(0.0);
    let break_even_refreshes = if saved_per_refresh_ms == 0.0 {
        f64::INFINITY
    } else {
        plan_prepare_ms / saved_per_refresh_ms
    };

    println!(
        "G3D_TIMING|regions={}|hops={}|direct_numeric_extract_ms={direct_ms:.6}|prepared_refresh_ms={refresh_ms:.6}|refresh_over_direct={refresh_over_direct:.9e}|saved_per_refresh_ms={saved_per_refresh_ms:.6}|break_even_refreshes={break_even_refreshes:.9e}",
        regions.len(),
        args.hops,
    );

    Ok(())
}
