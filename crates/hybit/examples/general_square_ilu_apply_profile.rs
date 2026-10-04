use std::collections::VecDeque;
use std::env;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use hybit::{read_matrix_market, Csr32Matrix, Ilu0Preconditioner, LinearOperator, Preconditioner};

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    samples: usize,
    batch: usize,
    warmup: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut samples = 9usize;
        let mut batch = 50usize;
        let mut warmup = 5usize;

        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => {
                    matrix = Some(PathBuf::from(
                        it.next().ok_or("missing value after --matrix")?,
                    ));
                }
                "--samples" => {
                    samples = it.next().ok_or("missing value after --samples")?.parse()?;
                }
                "--batch" => {
                    batch = it.next().ok_or("missing value after --batch")?.parse()?;
                }
                "--warmup" => {
                    warmup = it.next().ok_or("missing value after --warmup")?.parse()?;
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

        let matrix = matrix.ok_or("missing matrix path; use --matrix FILE.mtx")?;
        if samples < 3 {
            return Err("--samples must be >= 3".into());
        }
        if batch == 0 {
            return Err("--batch must be > 0".into());
        }

        Ok(Self {
            matrix,
            samples,
            batch,
            warmup,
        })
    }
}

fn print_usage() {
    println!("HyBIT GeneralSquare ILU(0) serial triangular-apply profile");
    println!();
    println!("Usage:");
    println!(
        "  cargo run --release -p hybit --example general_square_ilu_apply_profile -- --matrix A.mtx [options]"
    );
    println!();
    println!("Options:");
    println!("  --samples N    timed sample count (default 9)");
    println!("  --batch N      applies per timed sample (default 50)");
    println!("  --warmup N     untimed applies per ordering/kernel (default 5)");
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn deterministic_vector(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let bits = splitmix64((i as u64) ^ 0x38d0_2027_d6e8_feb8);
            let unit = ((bits >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
            2.0 * unit - 1.0
        })
        .collect()
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
            let new_col = old_to_new[old_col];
            row_entries.push((u32::try_from(new_col)?, a.values()[p]));
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

fn permute_vector(old: &[f64], new_to_old: &[usize]) -> Vec<f64> {
    new_to_old.iter().map(|&old_index| old[old_index]).collect()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

#[derive(Debug)]
struct KernelStats {
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

fn stats(mut samples_ms: Vec<f64>) -> KernelStats {
    let min_ms = samples_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let max_ms = samples_ms.iter().copied().fold(0.0, f64::max);
    let median_ms = median(&mut samples_ms);
    KernelStats {
        median_ms,
        min_ms,
        max_ms,
    }
}

fn warm_preconditioner(
    ilu: &Ilu0Preconditioner,
    r: &[f64],
    z: &mut [f64],
    repeats: usize,
) -> Result<(), Box<dyn Error>> {
    for _ in 0..repeats {
        ilu.apply(black_box(r), black_box(z))?;
        black_box(z[z.len() / 2]);
    }
    Ok(())
}

fn time_preconditioner_batch(
    ilu: &Ilu0Preconditioner,
    r: &[f64],
    z: &mut [f64],
    batch: usize,
) -> Result<f64, Box<dyn Error>> {
    let start = Instant::now();
    for _ in 0..batch {
        ilu.apply(black_box(r), black_box(z))?;
    }
    let seconds = start.elapsed().as_secs_f64();
    black_box(z[z.len() / 2]);
    Ok(seconds * 1.0e3 / batch as f64)
}

fn warm_spmv(
    a: &Csr32Matrix,
    x: &[f64],
    y: &mut [f64],
    repeats: usize,
) -> Result<(), Box<dyn Error>> {
    for _ in 0..repeats {
        a.apply(black_box(x), black_box(y))?;
        black_box(y[y.len() / 2]);
    }
    Ok(())
}

fn time_spmv_batch(
    a: &Csr32Matrix,
    x: &[f64],
    y: &mut [f64],
    batch: usize,
) -> Result<f64, Box<dyn Error>> {
    let start = Instant::now();
    for _ in 0..batch {
        a.apply(black_box(x), black_box(y))?;
    }
    let seconds = start.elapsed().as_secs_f64();
    black_box(y[y.len() / 2]);
    Ok(seconds * 1.0e3 / batch as f64)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;

    println!(
        "HyBIT {} GeneralSquare ILU(0) serial triangular-apply profile",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!("samples             : {}", args.samples);
    println!("batch / sample      : {}", args.batch);
    println!("warmup              : {}", args.warmup);

    let load_start = Instant::now();
    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1.0e3;

    if matrix.nrows() != matrix.ncols() {
        return Err("profile requires a square matrix".into());
    }
    matrix.diagonal()?;

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
    println!(
        "CSR storage         : {:.3} MiB",
        mib(matrix.storage_bytes())
    );
    println!("load                : {:.3} ms", load_ms);

    println!();
    println!("== RCM construction ==");
    let natural_bandwidth = structural_bandwidth(&matrix);

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
    println!("== ILU(0) preparation ==");
    let natural_prepare_start = Instant::now();
    let natural_ilu = Ilu0Preconditioner::from_csr32_general(&matrix)?;
    let natural_prepare_ms = natural_prepare_start.elapsed().as_secs_f64() * 1.0e3;

    let rcm_prepare_start = Instant::now();
    let rcm_ilu = Ilu0Preconditioner::from_csr32_general(&rcm_matrix)?;
    let rcm_prepare_ms = rcm_prepare_start.elapsed().as_secs_f64() * 1.0e3;

    println!("Natural canonical nnz : {}", natural_ilu.canonical_nnz());
    println!("RCM canonical nnz     : {}", rcm_ilu.canonical_nnz());
    println!(
        "Natural factor bytes  : {:.3} MiB",
        mib(natural_ilu.factor_bytes())
    );
    println!(
        "RCM factor bytes      : {:.3} MiB",
        mib(rcm_ilu.factor_bytes())
    );
    println!("Natural prepare       : {:.3} ms", natural_prepare_ms);
    println!("RCM prepare           : {:.3} ms", rcm_prepare_ms);
    println!(
        "adjusted pivots N/RCM : {} / {}",
        natural_ilu.adjusted_pivots(),
        rcm_ilu.adjusted_pivots()
    );

    if natural_ilu.canonical_nnz() != rcm_ilu.canonical_nnz() {
        return Err("Natural and RCM canonical ILU nnz differ unexpectedly".into());
    }

    let r_natural = deterministic_vector(matrix.nrows());
    let r_rcm = permute_vector(&r_natural, &new_to_old);

    let mut z_natural = vec![0.0; matrix.nrows()];
    let mut z_rcm = vec![0.0; matrix.nrows()];
    let mut y_natural = vec![0.0; matrix.nrows()];
    let mut y_rcm = vec![0.0; matrix.nrows()];

    warm_preconditioner(&natural_ilu, &r_natural, &mut z_natural, args.warmup)?;
    warm_preconditioner(&rcm_ilu, &r_rcm, &mut z_rcm, args.warmup)?;
    warm_spmv(&matrix, &r_natural, &mut y_natural, args.warmup)?;
    warm_spmv(&rcm_matrix, &r_rcm, &mut y_rcm, args.warmup)?;

    let mut natural_ilu_samples = Vec::with_capacity(args.samples);
    let mut rcm_ilu_samples = Vec::with_capacity(args.samples);
    let mut natural_spmv_samples = Vec::with_capacity(args.samples);
    let mut rcm_spmv_samples = Vec::with_capacity(args.samples);

    println!();
    println!("== timed samples ==");
    for sample in 0..args.samples {
        let sample_number = sample + 1;

        let (natural_ilu_ms, rcm_ilu_ms) = if sample % 2 == 0 {
            let n =
                time_preconditioner_batch(&natural_ilu, &r_natural, &mut z_natural, args.batch)?;
            let r = time_preconditioner_batch(&rcm_ilu, &r_rcm, &mut z_rcm, args.batch)?;
            (n, r)
        } else {
            let r = time_preconditioner_batch(&rcm_ilu, &r_rcm, &mut z_rcm, args.batch)?;
            let n =
                time_preconditioner_batch(&natural_ilu, &r_natural, &mut z_natural, args.batch)?;
            (n, r)
        };

        let (natural_spmv_ms, rcm_spmv_ms) = if sample % 2 == 0 {
            let n = time_spmv_batch(&matrix, &r_natural, &mut y_natural, args.batch)?;
            let r = time_spmv_batch(&rcm_matrix, &r_rcm, &mut y_rcm, args.batch)?;
            (n, r)
        } else {
            let r = time_spmv_batch(&rcm_matrix, &r_rcm, &mut y_rcm, args.batch)?;
            let n = time_spmv_batch(&matrix, &r_natural, &mut y_natural, args.batch)?;
            (n, r)
        };

        natural_ilu_samples.push(natural_ilu_ms);
        rcm_ilu_samples.push(rcm_ilu_ms);
        natural_spmv_samples.push(natural_spmv_ms);
        rcm_spmv_samples.push(rcm_spmv_ms);

        println!(
            "SAMPLE|sample={sample_number}|natural_ilu_ms={natural_ilu_ms:.6}|rcm_ilu_ms={rcm_ilu_ms:.6}|natural_spmv_ms={natural_spmv_ms:.6}|rcm_spmv_ms={rcm_spmv_ms:.6}"
        );
    }

    let natural_ilu_stats = stats(natural_ilu_samples);
    let rcm_ilu_stats = stats(rcm_ilu_samples);
    let natural_spmv_stats = stats(natural_spmv_samples);
    let rcm_spmv_stats = stats(rcm_spmv_samples);

    let canonical_nnz = natural_ilu.canonical_nnz();
    let natural_ns_per_nnz = natural_ilu_stats.median_ms * 1.0e6 / canonical_nnz.max(1) as f64;
    let rcm_ns_per_nnz = rcm_ilu_stats.median_ms * 1.0e6 / canonical_nnz.max(1) as f64;

    println!();
    println!("== medians ==");
    println!(
        "Natural ILU apply    : {:.6} ms/call (min {:.6}, max {:.6})",
        natural_ilu_stats.median_ms, natural_ilu_stats.min_ms, natural_ilu_stats.max_ms
    );
    println!(
        "RCM ILU apply        : {:.6} ms/call (min {:.6}, max {:.6})",
        rcm_ilu_stats.median_ms, rcm_ilu_stats.min_ms, rcm_ilu_stats.max_ms
    );
    println!(
        "Natural CSR SpMV     : {:.6} ms/call",
        natural_spmv_stats.median_ms
    );
    println!(
        "RCM CSR SpMV         : {:.6} ms/call",
        rcm_spmv_stats.median_ms
    );
    println!(
        "RCM/Natural ILU      : {:.6}",
        rcm_ilu_stats.median_ms / natural_ilu_stats.median_ms.max(f64::MIN_POSITIVE)
    );
    println!(
        "RCM/Natural SpMV     : {:.6}",
        rcm_spmv_stats.median_ms / natural_spmv_stats.median_ms.max(f64::MIN_POSITIVE)
    );
    println!(
        "ILU/SpMV Natural     : {:.6}",
        natural_ilu_stats.median_ms / natural_spmv_stats.median_ms.max(f64::MIN_POSITIVE)
    );
    println!(
        "ILU/SpMV RCM         : {:.6}",
        rcm_ilu_stats.median_ms / rcm_spmv_stats.median_ms.max(f64::MIN_POSITIVE)
    );

    println!(
        "ILU_APPLY|ordering=natural|samples={}|batch={}|median_ms={:.6}|min_ms={:.6}|max_ms={:.6}|canonical_nnz={}|ns_per_nnz={:.6}|prepare_ms={:.6}|factor_bytes={}|adjusted_pivots={}",
        args.samples,
        args.batch,
        natural_ilu_stats.median_ms,
        natural_ilu_stats.min_ms,
        natural_ilu_stats.max_ms,
        canonical_nnz,
        natural_ns_per_nnz,
        natural_prepare_ms,
        natural_ilu.factor_bytes(),
        natural_ilu.adjusted_pivots()
    );
    println!(
        "ILU_APPLY|ordering=rcm|samples={}|batch={}|median_ms={:.6}|min_ms={:.6}|max_ms={:.6}|canonical_nnz={}|ns_per_nnz={:.6}|prepare_ms={:.6}|factor_bytes={}|adjusted_pivots={}",
        args.samples,
        args.batch,
        rcm_ilu_stats.median_ms,
        rcm_ilu_stats.min_ms,
        rcm_ilu_stats.max_ms,
        canonical_nnz,
        rcm_ns_per_nnz,
        rcm_prepare_ms,
        rcm_ilu.factor_bytes(),
        rcm_ilu.adjusted_pivots()
    );
    println!(
        "SPMV_PROFILE|ordering=natural|median_ms={:.6}|min_ms={:.6}|max_ms={:.6}|nnz={}",
        natural_spmv_stats.median_ms,
        natural_spmv_stats.min_ms,
        natural_spmv_stats.max_ms,
        matrix.nnz()
    );
    println!(
        "SPMV_PROFILE|ordering=rcm|median_ms={:.6}|min_ms={:.6}|max_ms={:.6}|nnz={}",
        rcm_spmv_stats.median_ms,
        rcm_spmv_stats.min_ms,
        rcm_spmv_stats.max_ms,
        rcm_matrix.nnz()
    );
    println!(
        "KERNEL_COMPARE|rcm_over_natural_ilu={:.9}|rcm_over_natural_spmv={:.9}|natural_ilu_over_spmv={:.9}|rcm_ilu_over_spmv={:.9}|ordering_ms={:.6}|natural_bandwidth={}|rcm_bandwidth={}",
        rcm_ilu_stats.median_ms / natural_ilu_stats.median_ms.max(f64::MIN_POSITIVE),
        rcm_spmv_stats.median_ms / natural_spmv_stats.median_ms.max(f64::MIN_POSITIVE),
        natural_ilu_stats.median_ms / natural_spmv_stats.median_ms.max(f64::MIN_POSITIVE),
        rcm_ilu_stats.median_ms / rcm_spmv_stats.median_ms.max(f64::MIN_POSITIVE),
        ordering_ms,
        natural_bandwidth,
        rcm_bandwidth
    );

    Ok(())
}
