//! C1-H12-A3: CSR versus ABTM for symbolic piecewise-constant Galerkin.
//!
//! For a deterministic, one-hot, contiguous aggregation P, this probe
//! computes the Boolean *candidate* support of P^T A P. Numerical cancellations
//! may remove candidates. It does not construct a multilevel AMG solver.

use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use hybit::{AbtmTopology, Csr32Matrix, read_matrix_market};

struct Args {
    matrix: PathBuf,
    aggregate_nodes: usize,
    repeats: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut aggregate_nodes = 16_usize;
        let mut repeats = 5_usize;
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--matrix" => matrix = Some(PathBuf::from(args.next().ok_or("missing --matrix")?)),
                "--aggregate-nodes" => {
                    aggregate_nodes = args.next().ok_or("missing --aggregate-nodes")?.parse()?
                }
                "--repeats" => repeats = args.next().ok_or("missing --repeats")?.parse()?,
                "-h" | "--help" => {
                    println!(
                        "Usage: c1h12_a3_galerkin_symbolic_probe --matrix A.mtx [--aggregate-nodes 16] [--repeats 5]"
                    );
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument: {arg}").into()),
            }
        }
        if aggregate_nodes == 0 {
            return Err("--aggregate-nodes must be positive".into());
        }
        if !(1..=1000).contains(&repeats) {
            return Err("--repeats must be in 1..=1000".into());
        }
        Ok(Self {
            matrix: matrix.ok_or("missing --matrix")?,
            aggregate_nodes,
            repeats,
        })
    }
}

/// The exact dofs_per_node=1 contiguous aggregation map used by the existing
/// HyBIT piecewise-constant coarse basis, without changing production code.
fn aggregation_map(n: usize, aggregate_nodes: usize) -> Result<(Vec<u32>, usize), Box<dyn Error>> {
    if n == 0 || aggregate_nodes == 0 {
        return Err("aggregation requires a nonempty matrix and positive size".into());
    }
    let coarse_n = n.div_ceil(aggregate_nodes);
    let mut map = Vec::with_capacity(n);
    for fine in 0..n {
        map.push(u32::try_from(fine / aggregate_nodes)?);
    }
    Ok((map, coarse_n))
}

/// Shared Boolean pattern of A P, with one-hot P. Every column in each fine
/// row is mapped to its aggregate; sorting and dedup produce canonical rows.
fn fine_to_coarse_pattern(
    matrix: &Csr32Matrix,
    aggregate_of: &[u32],
) -> Result<Vec<Vec<u32>>, Box<dyn Error>> {
    if matrix.nrows() != matrix.ncols() || matrix.nrows() != aggregate_of.len() {
        return Err("probe requires a square matrix with full aggregate mapping".into());
    }
    let mut graph = Vec::with_capacity(matrix.nrows());
    for row in 0..matrix.nrows() {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        let mut neighbors = Vec::with_capacity(end - start);
        for &fine_col in &matrix.col_idx()[start..end] {
            neighbors.push(aggregate_of[fine_col as usize]);
        }
        neighbors.sort_unstable();
        neighbors.dedup();
        graph.push(neighbors);
    }
    Ok(graph)
}

fn fine_pattern_as_csr(
    pattern: &[Vec<u32>],
    coarse_n: usize,
) -> Result<Csr32Matrix, Box<dyn Error>> {
    let mut row_ptr = Vec::with_capacity(pattern.len() + 1);
    let mut col_idx = Vec::new();
    row_ptr.push(0_u32);
    for row in pattern {
        col_idx.extend(row);
        row_ptr.push(u32::try_from(col_idx.len())?);
    }
    let values = vec![1.0; col_idx.len()];
    Ok(Csr32Matrix::new(
        pattern.len(),
        coarse_n,
        row_ptr,
        col_idx,
        values,
    )?)
}

/// Reference: each coarse row is the sorted union of A P rows whose fine
/// indices belong to the aggregate. This is support(P^T A P) over Booleans.
fn csr_galerkin_pattern(
    pattern: &[Vec<u32>],
    aggregate_nodes: usize,
    coarse_n: usize,
) -> Vec<Vec<u32>> {
    let mut result = Vec::with_capacity(coarse_n);
    for coarse_row in 0..coarse_n {
        let mut support = Vec::new();
        let start = coarse_row * aggregate_nodes;
        let end = start.saturating_add(aggregate_nodes).min(pattern.len());
        for row in &pattern[start..end] {
            support.extend_from_slice(row);
        }
        support.sort_unstable();
        support.dedup();
        result.push(support);
    }
    result
}

/// Stronger CSR baseline: O(number of input neighbors) stamped deduplication
/// followed by a sort only on distinct coarse columns. Avoids concatenation
/// and duplicate-heavy sort cost in the reference above.
fn csr_galerkin_stamped(
    pattern: &[Vec<u32>],
    aggregate_nodes: usize,
    coarse_n: usize,
) -> Vec<Vec<u32>> {
    let mut stamps = vec![usize::MAX; coarse_n];
    let mut result = Vec::with_capacity(coarse_n);
    for coarse_row in 0..coarse_n {
        let mut touched = Vec::new();
        let start = coarse_row * aggregate_nodes;
        let end = start.saturating_add(aggregate_nodes).min(pattern.len());
        for row in &pattern[start..end] {
            for &coarse_col in row {
                if stamps[coarse_col as usize] != coarse_row {
                    stamps[coarse_col as usize] = coarse_row;
                    touched.push(coarse_col);
                }
            }
        }
        touched.sort_unstable();
        result.push(touched);
    }
    result
}

/// ABTM: OR each fine A P row's 64-bit words into the owning coarse row.
/// Stamps avoid clearing a dense bitmap for each coarse row.
fn abtm_galerkin_pattern(
    topology: &AbtmTopology,
    aggregate_nodes: usize,
    coarse_n: usize,
) -> Result<Vec<Vec<u32>>, Box<dyn Error>> {
    let mut stamp = vec![usize::MAX; topology.ncols().div_ceil(64)];
    let mut masks = vec![0_u64; stamp.len()];
    let mut touched = Vec::<usize>::new();
    let mut result = Vec::with_capacity(coarse_n);
    for coarse_row in 0..coarse_n {
        touched.clear();
        let start = coarse_row * aggregate_nodes;
        let end = start.saturating_add(aggregate_nodes).min(topology.nrows());
        for fine_row in start..end {
            for word in topology.row(fine_row)?.words() {
                let index = word.word_index() as usize;
                if stamp[index] != coarse_row {
                    stamp[index] = coarse_row;
                    masks[index] = 0;
                    touched.push(index);
                }
                masks[index] |= word.mask();
            }
        }
        touched.sort_unstable();
        let mut support = Vec::new();
        for &index in &touched {
            let mut mask = masks[index];
            while mask != 0 {
                support.push(u32::try_from(index * 64 + mask.trailing_zeros() as usize)?);
                mask &= mask - 1;
            }
        }
        result.push(support);
    }
    Ok(result)
}

/// Independent numerical accumulation of Z^T A Z (one-hot Z). This checks
/// every symbolic candidate is represented and counts exact cancellations.
/// This reference is *not* included in the timed symbolic kernels.
fn numerical_galerkin_reference(
    matrix: &Csr32Matrix,
    aggregate_of: &[u32],
    coarse_n: usize,
) -> Result<Vec<BTreeMap<u32, f64>>, Box<dyn Error>> {
    let mut result = vec![BTreeMap::<u32, f64>::new(); coarse_n];
    for row in 0..matrix.nrows() {
        let cr = aggregate_of[row] as usize;
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        for p in start..end {
            let cc = aggregate_of[matrix.col_idx()[p] as usize];
            let value = matrix.values()[p];
            if !value.is_finite() {
                return Err("non-finite matrix value".into());
            }
            *result[cr].entry(cc).or_default() += value;
        }
    }
    if result
        .iter()
        .flat_map(BTreeMap::values)
        .any(|v| !v.is_finite())
    {
        return Err("non-finite accumulated coarse value".into());
    }
    Ok(result)
}

fn profile<T>(repeats: usize, mut operation: impl FnMut() -> T) -> (T, f64) {
    let _ = black_box(operation());
    let mut times = Vec::with_capacity(repeats);
    let mut last = None;
    for _ in 0..repeats {
        let start = Instant::now();
        let output = black_box(operation());
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        last = Some(output);
    }
    times.sort_by(f64::total_cmp);
    (last.expect("positive repeat count"), times[times.len() / 2])
}

fn hash_pattern(rows: &[Vec<u32>]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for row in rows {
        hash ^= row.len() as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        for &col in row {
            hash ^= col as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn run(matrix: &Csr32Matrix, aggregate_nodes: usize, repeats: usize) -> Result<(), Box<dyn Error>> {
    if matrix.nrows() != matrix.ncols() {
        return Err("Galerkin probe requires a square matrix".into());
    }
    let (aggregate_of, coarse_n) = aggregation_map(matrix.nrows(), aggregate_nodes)?;
    let start = Instant::now();
    let fine_pattern = fine_to_coarse_pattern(matrix, &aggregate_of)?;
    let graph_build_ms = start.elapsed().as_secs_f64() * 1000.0;
    let fine_pattern_edges: usize = fine_pattern.iter().map(Vec::len).sum();

    let start = Instant::now();
    let structural_csr = fine_pattern_as_csr(&fine_pattern, coarse_n)?;
    let graph_csr_prepare_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let topology = AbtmTopology::from_csr32(&structural_csr)?;
    topology.validate()?;
    let abtm_conversion_ms = start.elapsed().as_secs_f64() * 1000.0;

    let (csr, csr_sort_ms) = profile(repeats, || {
        csr_galerkin_pattern(&fine_pattern, aggregate_nodes, coarse_n)
    });
    let (csr_stamped, csr_stamp_ms) = profile(repeats, || {
        csr_galerkin_stamped(&fine_pattern, aggregate_nodes, coarse_n)
    });
    let (abtm, abtm_symbolic_ms) = profile(repeats, || {
        abtm_galerkin_pattern(&topology, aggregate_nodes, coarse_n)
    });
    let abtm = abtm?;
    if csr != csr_stamped || csr != abtm {
        return Err("CSR-sort/CSR-stamp/ABTM Galerkin patterns mismatch".into());
    }

    let start = Instant::now();
    let numeric = numerical_galerkin_reference(matrix, &aggregate_of, coarse_n)?;
    let numeric_reference_ms = start.elapsed().as_secs_f64() * 1000.0;
    for (support, values) in csr.iter().zip(&numeric) {
        if !support.iter().copied().eq(values.keys().copied()) {
            return Err("symbolic pattern excludes a numerically assembled entry".into());
        }
    }
    let symbolic_nnz: usize = csr.iter().map(Vec::len).sum();
    let numeric_nnz: usize = numeric
        .iter()
        .map(|row| row.values().filter(|&&v| v != 0.0).count())
        .sum();
    let csr_adjacency_bytes = fine_pattern.capacity() * std::mem::size_of::<Vec<u32>>()
        + fine_pattern
            .iter()
            .map(|row| row.capacity() * std::mem::size_of::<u32>())
            .sum::<usize>();
    let temporary_csr_logical_bytes = (matrix.nrows() + 1) * std::mem::size_of::<u32>()
        + fine_pattern_edges * (std::mem::size_of::<u32>() + std::mem::size_of::<f64>());

    println!(
        "C1H12A3|rows={}|nnz={}|aggregate_nodes={aggregate_nodes}|coarse_n={coarse_n}|fine_pattern_edges={fine_pattern_edges}|symbolic_nnz={symbolic_nnz}|numeric_nnz={numeric_nnz}|exact_zero_candidates={}|csr_adjacency_bytes={csr_adjacency_bytes}|temporary_csr_logical_bytes={temporary_csr_logical_bytes}|abtm_metadata_bytes={}|abtm_nonempty_words={}|graph_build_ms={graph_build_ms:.6}|graph_csr_prepare_ms={graph_csr_prepare_ms:.6}|abtm_conversion_ms={abtm_conversion_ms:.6}|numeric_reference_ms={numeric_reference_ms:.6}|csr_sort_ms={csr_sort_ms:.6}|csr_stamp_ms={csr_stamp_ms:.6}|abtm_symbolic_ms={abtm_symbolic_ms:.6}|repeats={repeats}|symbolic_hash={:016x}|exact_equal=true",
        matrix.nrows(),
        matrix.nnz(),
        symbolic_nnz - numeric_nnz,
        topology.stats().metadata_bytes,
        topology.stats().nonempty_words,
        hash_pattern(&csr)
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let (matrix, _) = read_matrix_market(&args.matrix)?;
    run(&matrix, args.aggregate_nodes, args.repeats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_galerkin_pattern_matches_across_word_boundaries() {
        let n = 260_usize;
        let mut row_ptr = vec![0];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for row in 0..n {
            for col in 0..n {
                if row == col || row.abs_diff(col) == 1 {
                    col_idx.push(col as u32);
                    values.push(if row == col { 4.0 } else { -1.0 });
                }
            }
            row_ptr.push(col_idx.len() as u32);
        }
        let matrix = Csr32Matrix::new(n, n, row_ptr, col_idx, values).unwrap();
        for aggregate_nodes in [1, 2, 4, 16] {
            let (map, coarse_n) = aggregation_map(n, aggregate_nodes).unwrap();
            let pattern = fine_to_coarse_pattern(&matrix, &map).unwrap();
            let topo = AbtmTopology::from_csr32(&fine_pattern_as_csr(&pattern, coarse_n).unwrap())
                .unwrap();
            let csr = csr_galerkin_pattern(&pattern, aggregate_nodes, coarse_n);
            assert_eq!(
                csr,
                csr_galerkin_stamped(&pattern, aggregate_nodes, coarse_n)
            );
            assert_eq!(
                csr,
                abtm_galerkin_pattern(&topo, aggregate_nodes, coarse_n).unwrap()
            );
            let numeric = numerical_galerkin_reference(&matrix, &map, coarse_n).unwrap();
            for (cols, values) in csr.iter().zip(numeric.iter()) {
                assert_eq!(*cols, values.keys().copied().collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn exact_numeric_cancellation_preserves_boolean_candidate() {
        let matrix = Csr32Matrix::new(
            4,
            4,
            vec![0, 2, 4, 7, 8],
            vec![0, 2, 1, 2, 0, 1, 2, 3],
            vec![5.0, 1.0, 5.0, -1.0, 1.0, -1.0, 5.0, 5.0],
        )
        .unwrap();
        let (map, coarse_n) = aggregation_map(4, 2).unwrap();
        let graph = fine_to_coarse_pattern(&matrix, &map).unwrap();
        let topo =
            AbtmTopology::from_csr32(&fine_pattern_as_csr(&graph, coarse_n).unwrap()).unwrap();
        let csr = csr_galerkin_pattern(&graph, 2, coarse_n);
        assert_eq!(csr, csr_galerkin_stamped(&graph, 2, coarse_n));
        assert_eq!(csr, abtm_galerkin_pattern(&topo, 2, coarse_n).unwrap());
        assert_eq!(csr, vec![vec![0, 1], vec![0, 1]]);
        let coarse = numerical_galerkin_reference(&matrix, &map, coarse_n).unwrap();
        assert_eq!(coarse[0][&1], 0.0);
        assert_eq!(coarse[1][&0], 0.0);
    }

    #[test]
    fn invalid_aggregate_size_and_nonsquare_matrix_are_rejected() {
        assert!(aggregation_map(4, 0).is_err());
        let nonsquare = Csr32Matrix::new(2, 3, vec![0, 1, 2], vec![0, 2], vec![1.0; 2]).unwrap();
        let (map, _) = aggregation_map(2, 1).unwrap();
        assert!(fine_to_coarse_pattern(&nonsquare, &map).is_err());
    }
}
