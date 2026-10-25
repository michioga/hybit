//! H1-C3K2 research gate: exact A/B of the G8-A6 stamped common-neighbor
//! matching vs sparse 64-bit ABTM row masks. Diagnostic only; no solver routing.
//! Usage: cargo run --release --locked -p hybit-distributed \
//!   --example abtm_affinity_ab -- D:\Work\thermal1.mtx [repeats]
use hybit_matrix::{AbtmDualTopology, Csr32Matrix, read_matrix_market};
use std::cmp::Reverse;
use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

type Words = Vec<Vec<(usize, u64)>>;

fn adjacency(matrix: &Csr32Matrix) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
    if matrix.nrows() != matrix.ncols() {
        return Err("matrix must be square".into());
    }
    let dual = AbtmDualTopology::from_csr32(matrix)?;
    let mut result = Vec::with_capacity(matrix.nrows());
    for node in 0..matrix.nrows() {
        let mut neighbors = Vec::new();
        for row in [dual.row(node)?, dual.column(node)?] {
            for word in row.words() {
                let mut mask = word.mask();
                while mask != 0 {
                    let col = word.base_col() + mask.trailing_zeros() as usize;
                    mask &= mask - 1;
                    if col != node && col < matrix.nrows() {
                        neighbors.push(col);
                    }
                }
            }
        }
        neighbors.sort_unstable();
        neighbors.dedup();
        result.push(neighbors);
    }
    Ok(result)
}

fn pack(adjacency: &[Vec<usize>]) -> Words {
    adjacency
        .iter()
        .map(|neighbors| {
            let mut row: Vec<(usize, u64)> = Vec::new();
            for &neighbor in neighbors {
                let index = neighbor / 64;
                let mask = 1u64 << (neighbor % 64);
                if let Some((last, bits)) = row.last_mut()
                    && *last == index
                {
                    *bits |= mask;
                    continue;
                }
                row.push((index, mask));
            }
            row
        })
        .collect()
}

#[inline]
fn bit_intersection(a: &[(usize, u64)], b: &[(usize, u64)]) -> usize {
    let (mut i, mut j, mut count) = (0, 0, 0usize);
    while i < a.len() && j < b.len() {
        if a[i].0 == b[j].0 {
            count += (a[i].1 & b[j].1).count_ones() as usize;
            i += 1;
            j += 1;
        } else if a[i].0 < b[j].0 {
            i += 1;
        } else {
            j += 1;
        }
    }
    count
}

fn check_all_pair_scores(adj: &[Vec<usize>], words: &Words) -> Result<usize, Box<dyn Error>> {
    let mut stamps = vec![usize::MAX; adj.len()];
    let mut checks = 0usize;
    for (node, neighbors) in adj.iter().enumerate() {
        for &shared in neighbors {
            stamps[shared] = node;
        }
        for &neighbor in neighbors {
            let stamp = adj[neighbor]
                .iter()
                .filter(|&&shared| stamps[shared] == node)
                .count();
            let packed = bit_intersection(&words[node], &words[neighbor]);
            if stamp != packed {
                return Err(format!("affinity mismatch node={node} neighbor={neighbor}: stamp={stamp} bitmap={packed}").into());
            }
            checks += 1;
        }
    }
    Ok(checks)
}

#[derive(Clone, Copy)]
enum Method {
    Stamp,
    Bitmap,
}

// Intentionally mirror the G8-A6 coarsen_once_cached node order, score, and
// tie-breaks. No coarse graph construction, so this isolates *pair selection*.
fn one_pass(adj: &[Vec<usize>], words: &Words, method: Method) -> Vec<usize> {
    let n = adj.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&node| (Reverse(adj[node].len()), node));
    let mut coarse_of = vec![usize::MAX; n];
    let mut stamps = if matches!(method, Method::Stamp) {
        vec![usize::MAX; n]
    } else {
        Vec::new()
    };
    let mut coarse = 0usize;
    for node in order {
        if coarse_of[node] != usize::MAX {
            continue;
        }
        if matches!(method, Method::Stamp) {
            for &neighbor in &adj[node] {
                stamps[neighbor] = node;
            }
        }
        let mut best_neighbor: Option<(usize, usize, usize)> = None;
        for &neighbor in &adj[node] {
            if coarse_of[neighbor] != usize::MAX {
                continue;
            }
            let affinity = match method {
                Method::Stamp => adj[neighbor]
                    .iter()
                    .filter(|&&shared| stamps[shared] == node)
                    .count(),
                Method::Bitmap => bit_intersection(&words[node], &words[neighbor]),
            };
            let candidate = (
                affinity,
                usize::MAX - adj[neighbor].len(),
                usize::MAX - neighbor,
            );
            let replace = match best_neighbor {
                None => true,
                Some(current) => candidate > current,
            };
            if replace {
                best_neighbor = Some(candidate);
            }
        }
        coarse_of[node] = coarse;
        if let Some((_, _, reverse_neighbor)) = best_neighbor {
            coarse_of[usize::MAX - reverse_neighbor] = coarse;
        }
        coarse += 1;
    }
    coarse_of
}

fn median(mut d: Vec<Duration>) -> f64 {
    d.sort();
    let i = d.len() / 2;
    if d.len().is_multiple_of(2) {
        (d[i - 1].as_secs_f64() + d[i].as_secs_f64()) * 500.0
    } else {
        d[i].as_secs_f64() * 1000.0
    }
}

fn timed_pass(adj: &[Vec<usize>], words: &Words, method: Method) -> Duration {
    let start = Instant::now();
    black_box(one_pass(adj, words, method));
    start.elapsed()
}

fn run_paired_bench(adj: &[Vec<usize>], words: &Words, repeats: usize) -> (f64, f64) {
    let mut stamp_times = Vec::with_capacity(repeats);
    let mut bitmap_times = Vec::with_capacity(repeats);
    for iteration in 0..repeats {
        if iteration % 2 == 0 {
            stamp_times.push(timed_pass(adj, words, Method::Stamp));
            bitmap_times.push(timed_pass(adj, words, Method::Bitmap));
        } else {
            bitmap_times.push(timed_pass(adj, words, Method::Bitmap));
            stamp_times.push(timed_pass(adj, words, Method::Stamp));
        }
    }
    (median(stamp_times), median(bitmap_times))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.is_empty() || args.len() > 2 {
        return Err("usage: abtm_affinity_ab <matrix.mtx> [repeats: 1..=20]".into());
    }
    let repeats: usize = if let Some(value) = args.get(1) {
        value.parse()?
    } else {
        5
    };
    if !(1..=20).contains(&repeats) {
        return Err("repeats must be 1..=20".into());
    }
    let read_started = Instant::now();
    let (matrix, info) = read_matrix_market(&args[0])?;
    let read_ms = read_started.elapsed().as_secs_f64() * 1.0e3;
    if matrix.nrows() < 8 || matrix.nrows() > 250_000 || matrix.nnz() > 12_000_000 {
        return Err("outside diagnostic matrix size limits".into());
    }
    let adjacency_started = Instant::now();
    let adj = adjacency(&matrix)?;
    let adjacency_ms = adjacency_started.elapsed().as_secs_f64() * 1.0e3;
    let pack_started = Instant::now();
    let words = pack(&adj);
    let bitmap_prep_ms = pack_started.elapsed().as_secs_f64() * 1.0e3;
    let checked = check_all_pair_scores(&adj, &words)?;
    let baseline = one_pass(&adj, &words, Method::Stamp);
    let alternative = one_pass(&adj, &words, Method::Bitmap);
    if baseline != alternative {
        return Err("FAIL: coarse assignments differ".into());
    }
    let (stamp_ms, bitmap_ms) = run_paired_bench(&adj, &words, repeats);
    let coarse_vertices = baseline.iter().copied().max().unwrap_or(0) + 1;
    let word_count: usize = words.iter().map(Vec::len).sum();
    let raw_adj_entries: usize = adj.iter().map(Vec::len).sum();
    println!("=== H1-C3K2 ABTM AFFINITY A/B ===");
    println!(
        "matrix={} dofs={} csr_nnz={} input_entries={}",
        args[0],
        matrix.nrows(),
        matrix.nnz(),
        info.input_entries
    );
    println!(
        "source_read_ms={read_ms:.4} dual_adjacency_ms={adjacency_ms:.4} bitmap_prep_ms={bitmap_prep_ms:.4}"
    );
    println!(
        "adjacency_edges_directed={raw_adj_entries} bitmap_nonempty_words={word_count} bitmap_payload_est_bytes={}",
        word_count * std::mem::size_of::<(usize, u64)>()
    );
    println!(
        "pairs_checked={checked} coarse_vertices={coarse_vertices} exact_pair_scores=PASS exact_matching=PASS"
    );
    println!(
        "stamped_pair_median_ms={stamp_ms:.4} bitmap_pair_median_ms={bitmap_ms:.4} pair_speed_ratio={:.4}",
        stamp_ms / bitmap_ms
    );
    println!(
        "bitmap_build_plus_pair_ms={:.4} stamp_pair_ms={stamp_ms:.4}",
        bitmap_prep_ms + bitmap_ms
    );
    println!(
        "NOTE: diagnostic only, not a full coarsening/MPI/PCG timing and not production routing"
    );
    Ok(())
}
