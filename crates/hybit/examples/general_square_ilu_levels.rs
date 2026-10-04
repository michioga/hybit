use std::collections::VecDeque;
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use hybit::{read_matrix_market, Csr32Matrix, Ilu0Preconditioner};

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut it = env::args().skip(1);

        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => {
                    matrix = Some(PathBuf::from(
                        it.next().ok_or("missing value after --matrix")?,
                    ));
                }
                "-h" | "--help" => {
                    print_usage();
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
        })
    }
}

fn print_usage() {
    println!("HyBIT GeneralSquare ILU(0) triangular dependency / level profile");
    println!();
    println!("Usage:");
    println!(
        "  cargo run --release -p hybit --example general_square_ilu_levels -- --matrix A.mtx"
    );
}

fn structural_bandwidth(a: &Csr32Matrix) -> usize {
    let mut bandwidth = 0usize;
    for row in 0..a.nrows() {
        let start = a.row_ptr()[row] as usize;
        let end = a.row_ptr()[row + 1] as usize;
        for &col in &a.col_idx()[start..end] {
            bandwidth = bandwidth.max(row.abs_diff(col as usize));
        }
    }
    bandwidth
}

fn build_symmetrized_graph(a: &Csr32Matrix) -> Vec<Vec<u32>> {
    let n = a.nrows();
    let mut graph = vec![Vec::<u32>::new(); n];

    for row in 0..n {
        let start = a.row_ptr()[row] as usize;
        let end = a.row_ptr()[row + 1] as usize;
        for &col_u32 in &a.col_idx()[start..end] {
            let col = col_u32 as usize;
            if row == col {
                continue;
            }
            graph[row].push(col_u32);
            graph[col].push(row as u32);
        }
    }

    for neighbors in &mut graph {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    graph
}

fn reverse_cuthill_mckee(graph: &mut [Vec<u32>]) -> Vec<usize> {
    let n = graph.len();
    let degree: Vec<usize> = graph.iter().map(Vec::len).collect();

    for neighbors in graph.iter_mut() {
        neighbors.sort_unstable_by_key(|&node| (degree[node as usize], node));
    }

    let mut visited = vec![false; n];
    let mut permutation = Vec::with_capacity(n);
    let mut queue = VecDeque::new();

    while permutation.len() < n {
        let start = (0..n)
            .filter(|&node| !visited[node])
            .min_by_key(|&node| (degree[node], node))
            .expect("at least one unvisited node must remain");

        let component_begin = permutation.len();
        visited[start] = true;
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            permutation.push(node);
            for &neighbor_u32 in &graph[node] {
                let neighbor = neighbor_u32 as usize;
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        permutation[component_begin..].reverse();
    }

    permutation
}

fn symmetric_permute(a: &Csr32Matrix, new_to_old: &[usize]) -> Result<Csr32Matrix, Box<dyn Error>> {
    let n = a.nrows();
    if a.ncols() != n || new_to_old.len() != n {
        return Err(
            "symmetric permutation requires a square matrix and n-entry permutation".into(),
        );
    }

    let mut old_to_new = vec![usize::MAX; n];
    for (new, &old) in new_to_old.iter().enumerate() {
        if old >= n || old_to_new[old] != usize::MAX {
            return Err("invalid RCM permutation".into());
        }
        old_to_new[old] = new;
    }

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(a.nnz());
    let mut values = Vec::with_capacity(a.nnz());
    let mut row_entries = Vec::<(u32, f64)>::new();
    row_ptr.push(0u32);

    for &old_row in new_to_old {
        row_entries.clear();
        let start = a.row_ptr()[old_row] as usize;
        let end = a.row_ptr()[old_row + 1] as usize;

        for p in start..end {
            let old_col = a.col_idx()[p] as usize;
            row_entries.push((u32::try_from(old_to_new[old_col])?, a.values()[p]));
        }
        row_entries.sort_unstable_by_key(|&(col, _)| col);

        for &(col, value) in &row_entries {
            col_idx.push(col);
            values.push(value);
        }
        row_ptr.push(u32::try_from(col_idx.len())?);
    }

    Ok(Csr32Matrix::new(n, n, row_ptr, col_idx, values)?)
}

#[derive(Debug)]
struct CanonicalPattern {
    row_ptr: Vec<usize>,
    col_idx: Vec<u32>,
}

fn canonical_pattern(a: &Csr32Matrix) -> Result<CanonicalPattern, Box<dyn Error>> {
    if a.nrows() != a.ncols() {
        return Err("ILU(0) pattern profile requires a square matrix".into());
    }

    let n = a.nrows();
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(a.nnz());
    let mut entries = Vec::<(u32, f64)>::new();
    row_ptr.push(0);

    for row in 0..n {
        entries.clear();
        let start = a.row_ptr()[row] as usize;
        let end = a.row_ptr()[row + 1] as usize;

        entries.extend(
            a.col_idx()[start..end]
                .iter()
                .copied()
                .zip(a.values()[start..end].iter().copied()),
        );
        entries.sort_unstable_by_key(|&(col, _)| col);

        let mut found_diagonal = false;
        let mut p = 0usize;
        while p < entries.len() {
            let col = entries[p].0;
            let mut value = entries[p].1;
            p += 1;

            while p < entries.len() && entries[p].0 == col {
                value += entries[p].1;
                p += 1;
            }

            if !value.is_finite() {
                return Err("canonical duplicate accumulation became non-finite".into());
            }

            if col as usize == row {
                found_diagonal = true;
            } else if value == 0.0 {
                continue;
            }

            col_idx.push(col);
        }

        if !found_diagonal {
            return Err(format!("missing diagonal at row {row}").into());
        }

        row_ptr.push(col_idx.len());
    }

    Ok(CanonicalPattern { row_ptr, col_idx })
}

#[derive(Debug)]
struct WidthStats {
    median: usize,
    p95: usize,
    max: usize,
    mean: f64,
}

fn percentile_sorted(values: &[usize], q: f64) -> usize {
    if values.is_empty() {
        return 0;
    }
    let pos = ((values.len() - 1) as f64 * q).round() as usize;
    values[pos.min(values.len() - 1)]
}

fn width_stats(widths: &[usize]) -> WidthStats {
    let mut sorted = widths.to_vec();
    sorted.sort_unstable();
    WidthStats {
        median: percentile_sorted(&sorted, 0.50),
        p95: percentile_sorted(&sorted, 0.95),
        max: sorted.last().copied().unwrap_or(0),
        mean: if widths.is_empty() {
            0.0
        } else {
            widths.iter().sum::<usize>() as f64 / widths.len() as f64
        },
    }
}

#[derive(Debug)]
struct DistanceStats {
    mean: f64,
    median: usize,
    p95: usize,
    max: usize,
}

fn distance_stats(mut distances: Vec<usize>) -> DistanceStats {
    if distances.is_empty() {
        return DistanceStats {
            mean: 0.0,
            median: 0,
            p95: 0,
            max: 0,
        };
    }

    let mean = distances.iter().sum::<usize>() as f64 / distances.len() as f64;
    distances.sort_unstable();

    DistanceStats {
        mean,
        median: percentile_sorted(&distances, 0.50),
        p95: percentile_sorted(&distances, 0.95),
        max: *distances.last().expect("non-empty"),
    }
}

#[derive(Debug)]
struct DirectionProfile {
    levels: usize,
    width_stats: WidthStats,
    dependencies: usize,
    avg_dependencies_per_row: f64,
    max_dependencies_per_row: usize,
    level_work_mean: f64,
    level_work_max: usize,
    level_work_imbalance: f64,
    distance_stats: DistanceStats,
}

#[derive(Debug)]
struct TriangularProfile {
    forward: DirectionProfile,
    backward: DirectionProfile,
}

fn build_direction_profile(
    row_levels: &[usize],
    row_dependencies: &[usize],
    distances: Vec<usize>,
) -> DirectionProfile {
    let levels = row_levels.iter().copied().max().unwrap_or(0);
    let mut widths = vec![0usize; levels];
    let mut work = vec![0usize; levels];

    for row in 0..row_levels.len() {
        let level = row_levels[row];
        if level == 0 {
            continue;
        }
        widths[level - 1] += 1;
        // Count one diagonal operation plus the structural dependencies.
        work[level - 1] += row_dependencies[row] + 1;
    }

    let dependencies = row_dependencies.iter().sum::<usize>();
    let max_dependencies_per_row = row_dependencies.iter().copied().max().unwrap_or(0);
    let level_work_max = work.iter().copied().max().unwrap_or(0);
    let level_work_mean = if work.is_empty() {
        0.0
    } else {
        work.iter().sum::<usize>() as f64 / work.len() as f64
    };
    let level_work_imbalance = if level_work_mean == 0.0 {
        0.0
    } else {
        level_work_max as f64 / level_work_mean
    };

    DirectionProfile {
        levels,
        width_stats: width_stats(&widths),
        dependencies,
        avg_dependencies_per_row: if row_levels.is_empty() {
            0.0
        } else {
            dependencies as f64 / row_levels.len() as f64
        },
        max_dependencies_per_row,
        level_work_mean,
        level_work_max,
        level_work_imbalance,
        distance_stats: distance_stats(distances),
    }
}

fn profile_triangular(pattern: &CanonicalPattern, n: usize) -> TriangularProfile {
    let mut forward_level = vec![1usize; n];
    let mut backward_level = vec![1usize; n];
    let mut lower_degree = vec![0usize; n];
    let mut upper_degree = vec![0usize; n];
    let mut lower_distances = Vec::<usize>::new();
    let mut upper_distances = Vec::<usize>::new();

    for row in 0..n {
        let start = pattern.row_ptr[row];
        let end = pattern.row_ptr[row + 1];
        let mut level = 1usize;

        for &col_u32 in &pattern.col_idx[start..end] {
            let col = col_u32 as usize;
            if col < row {
                lower_degree[row] += 1;
                lower_distances.push(row - col);
                level = level.max(forward_level[col] + 1);
            }
        }

        forward_level[row] = level;
    }

    for row in (0..n).rev() {
        let start = pattern.row_ptr[row];
        let end = pattern.row_ptr[row + 1];
        let mut level = 1usize;

        for &col_u32 in &pattern.col_idx[start..end] {
            let col = col_u32 as usize;
            if col > row {
                upper_degree[row] += 1;
                upper_distances.push(col - row);
                level = level.max(backward_level[col] + 1);
            }
        }

        backward_level[row] = level;
    }

    TriangularProfile {
        forward: build_direction_profile(&forward_level, &lower_degree, lower_distances),
        backward: build_direction_profile(&backward_level, &upper_degree, upper_distances),
    }
}

fn print_direction(ordering: &str, direction: &str, p: &DirectionProfile, n: usize) {
    let average_parallelism = if p.levels == 0 {
        0.0
    } else {
        n as f64 / p.levels as f64
    };

    println!(
        "{ordering:7} {direction:8}: levels {:6}, avg width {:9.3}, median {:6}, p95 {:6}, max {:6}, deps {}, avg deps/row {:.3}",
        p.levels,
        p.width_stats.mean,
        p.width_stats.median,
        p.width_stats.p95,
        p.width_stats.max,
        p.dependencies,
        p.avg_dependencies_per_row
    );
    println!(
        "LEVEL_PROFILE|ordering={ordering}|direction={direction}|levels={}|avg_parallelism={average_parallelism:.6}|width_mean={:.6}|width_median={}|width_p95={}|width_max={}|dependencies={}|avg_dependencies_per_row={:.6}|max_dependencies_per_row={}|level_work_mean={:.6}|level_work_max={}|level_work_imbalance={:.6}|distance_mean={:.6}|distance_median={}|distance_p95={}|distance_max={}",
        p.levels,
        p.width_stats.mean,
        p.width_stats.median,
        p.width_stats.p95,
        p.width_stats.max,
        p.dependencies,
        p.avg_dependencies_per_row,
        p.max_dependencies_per_row,
        p.level_work_mean,
        p.level_work_max,
        p.level_work_imbalance,
        p.distance_stats.mean,
        p.distance_stats.median,
        p.distance_stats.p95,
        p.distance_stats.max
    );
}

fn print_profile(ordering: &str, p: &TriangularProfile, n: usize) {
    print_direction(ordering, "forward", &p.forward, n);
    print_direction(ordering, "backward", &p.backward, n);

    println!(
        "TRIANGULAR_SUMMARY|ordering={ordering}|forward_levels={}|backward_levels={}|forward_avg_parallelism={:.6}|backward_avg_parallelism={:.6}|forward_width_max={}|backward_width_max={}|forward_distance_p95={}|backward_distance_p95={}|forward_distance_max={}|backward_distance_max={}",
        p.forward.levels,
        p.backward.levels,
        if p.forward.levels == 0 {
            0.0
        } else {
            n as f64 / p.forward.levels as f64
        },
        if p.backward.levels == 0 {
            0.0
        } else {
            n as f64 / p.backward.levels as f64
        },
        p.forward.width_stats.max,
        p.backward.width_stats.max,
        p.forward.distance_stats.p95,
        p.backward.distance_stats.p95,
        p.forward.distance_stats.max,
        p.backward.distance_stats.max
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;

    println!(
        "HyBIT {} GeneralSquare ILU(0) triangular dependency / level profile",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());

    let load_start = Instant::now();
    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1.0e3;

    if matrix.nrows() != matrix.ncols() {
        return Err("profile requires a square matrix".into());
    }

    println!(
        "Matrix Market       : {:?}, {} input entries -> {} CSR nnz",
        mm.symmetry, mm.input_entries, mm.csr_nnz
    );
    println!(
        "dimensions          : {} x {}",
        matrix.nrows(),
        matrix.ncols()
    );
    println!("CSR nnz             : {}", matrix.nnz());
    println!("load                : {:.3} ms", load_ms);

    let natural_bandwidth = structural_bandwidth(&matrix);

    println!();
    println!("== RCM construction ==");
    let ordering_start = Instant::now();
    let mut graph = build_symmetrized_graph(&matrix);
    let new_to_old = reverse_cuthill_mckee(&mut graph);
    let rcm_matrix = symmetric_permute(&matrix, &new_to_old)?;
    let ordering_ms = ordering_start.elapsed().as_secs_f64() * 1.0e3;
    let rcm_bandwidth = structural_bandwidth(&rcm_matrix);

    println!("natural bandwidth   : {natural_bandwidth}");
    println!("RCM bandwidth       : {rcm_bandwidth}");
    println!("ordering total      : {:.3} ms", ordering_ms);

    println!();
    println!("== canonical ILU(0) pattern cross-check ==");
    let natural_pattern_start = Instant::now();
    let natural_pattern = canonical_pattern(&matrix)?;
    let natural_pattern_ms = natural_pattern_start.elapsed().as_secs_f64() * 1.0e3;

    let rcm_pattern_start = Instant::now();
    let rcm_pattern = canonical_pattern(&rcm_matrix)?;
    let rcm_pattern_ms = rcm_pattern_start.elapsed().as_secs_f64() * 1.0e3;

    let natural_ilu = Ilu0Preconditioner::from_csr32_general(&matrix)?;
    let rcm_ilu = Ilu0Preconditioner::from_csr32_general(&rcm_matrix)?;

    println!(
        "natural canonical nnz : {} (ILU reports {})",
        natural_pattern.col_idx.len(),
        natural_ilu.canonical_nnz()
    );
    println!(
        "RCM canonical nnz     : {} (ILU reports {})",
        rcm_pattern.col_idx.len(),
        rcm_ilu.canonical_nnz()
    );

    if natural_pattern.col_idx.len() != natural_ilu.canonical_nnz() {
        return Err("Natural benchmark canonical pattern disagrees with ILU(0)".into());
    }
    if rcm_pattern.col_idx.len() != rcm_ilu.canonical_nnz() {
        return Err("RCM benchmark canonical pattern disagrees with ILU(0)".into());
    }
    if natural_pattern.col_idx.len() != rcm_pattern.col_idx.len() {
        return Err("Natural and RCM canonical nnz differ unexpectedly".into());
    }

    println!(
        "PATTERN|ordering=natural|canonical_nnz={}|build_ms={natural_pattern_ms:.6}|adjusted_pivots={}",
        natural_pattern.col_idx.len(),
        natural_ilu.adjusted_pivots()
    );
    println!(
        "PATTERN|ordering=rcm|canonical_nnz={}|build_ms={rcm_pattern_ms:.6}|adjusted_pivots={}",
        rcm_pattern.col_idx.len(),
        rcm_ilu.adjusted_pivots()
    );

    println!();
    println!("== triangular dependency levels ==");
    let natural_profile_start = Instant::now();
    let natural_profile = profile_triangular(&natural_pattern, matrix.nrows());
    let natural_profile_ms = natural_profile_start.elapsed().as_secs_f64() * 1.0e3;

    let rcm_profile_start = Instant::now();
    let rcm_profile = profile_triangular(&rcm_pattern, rcm_matrix.nrows());
    let rcm_profile_ms = rcm_profile_start.elapsed().as_secs_f64() * 1.0e3;

    print_profile("natural", &natural_profile, matrix.nrows());
    print_profile("rcm", &rcm_profile, matrix.nrows());

    println!(
        "LEVEL_COMPARE|forward_levels_ratio={:.9}|backward_levels_ratio={:.9}|forward_parallelism_ratio={:.9}|backward_parallelism_ratio={:.9}|natural_profile_ms={natural_profile_ms:.6}|rcm_profile_ms={rcm_profile_ms:.6}|ordering_ms={ordering_ms:.6}|natural_bandwidth={natural_bandwidth}|rcm_bandwidth={rcm_bandwidth}",
        rcm_profile.forward.levels as f64
            / natural_profile.forward.levels.max(1) as f64,
        rcm_profile.backward.levels as f64
            / natural_profile.backward.levels.max(1) as f64,
        (matrix.nrows() as f64 / rcm_profile.forward.levels.max(1) as f64)
            / (matrix.nrows() as f64 / natural_profile.forward.levels.max(1) as f64),
        (matrix.nrows() as f64 / rcm_profile.backward.levels.max(1) as f64)
            / (matrix.nrows() as f64 / natural_profile.backward.levels.max(1) as f64),
    );

    Ok(())
}
