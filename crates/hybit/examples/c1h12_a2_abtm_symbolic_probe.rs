//! C1-H12-A2: opt-in CSR adjacency vs ABTM symbolic two-hop union probe.
//!
//! The generated two-hop support is a proxy for AMG graph/transfer symbolic
//! work, NOT an implementation of Galerkin `P^T A P` or an AMG solver.
//! The same floating-point strength-of-connection scan feeds both arms.

use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use hybit::{AbtmTopology, Csr32Matrix, read_matrix_market};

struct Args {
    matrix: PathBuf,
    threshold: f64,
    repeats: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut threshold = 0.25_f64;
        let mut repeats = 5;
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--matrix" => {
                    matrix = Some(PathBuf::from(args.next().ok_or("missing --matrix value")?));
                }
                "--threshold" => {
                    threshold = args.next().ok_or("missing --threshold value")?.parse()?;
                }
                "--repeats" => {
                    repeats = args.next().ok_or("missing --repeats value")?.parse()?;
                }
                "--help" | "-h" => {
                    println!(
                        "Usage: c1h12_a2_abtm_symbolic_probe --matrix A.mtx [--threshold 0.25] [--repeats 5]"
                    );
                    std::process::exit(0);
                }
                _ => return Err(format!("unexpected argument: {arg}").into()),
            }
        }
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err("--threshold must be finite and in [0,1]".into());
        }
        if repeats == 0 || repeats > 1000 {
            return Err("--repeats must be in 1..=1000".into());
        }
        Ok(Self {
            matrix: matrix.ok_or("missing --matrix")?,
            threshold,
            repeats,
        })
    }
}

/// Deterministic scalar near-neighbor strength graph. The numeric threshold
/// is deliberately shared by both arms; ABTM only handles Boolean topology.
/// Positive diagonal alone is not a proof that the matrix is SPD.
fn strong_graph(matrix: &Csr32Matrix, threshold: f64) -> Result<Vec<Vec<u32>>, Box<dyn Error>> {
    if matrix.nrows() != matrix.ncols() || matrix.nrows() == 0 {
        return Err("probe requires a nonempty square matrix".into());
    }
    let n = matrix.nrows();
    let diag = matrix.diagonal()?;
    if diag.iter().any(|&value| value <= 0.0 || !value.is_finite()) {
        return Err("probe requires a finite positive diagonal".into());
    }
    let sqrt_diag: Vec<f64> = diag.iter().map(|value| value.sqrt()).collect();
    let mut result = Vec::with_capacity(n);
    for row in 0..n {
        let mut strengths = BTreeMap::<u32, f64>::new();
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        for p in start..end {
            let col = matrix.col_idx()[p];
            if col as usize == row {
                continue;
            }
            // Do not use signed negative-only M-matrix strength: the corpus
            // also contains rotated diffusion with signed mixed couplings.
            let denominator = sqrt_diag[row] * sqrt_diag[col as usize];
            let strength = matrix.values()[p].abs() / denominator;
            if !strength.is_finite() {
                return Err("non-finite normalized connection strength".into());
            }
            *strengths.entry(col).or_default() += strength;
        }
        let strongest = strengths.values().copied().fold(0.0_f64, f64::max);
        let mut neighbors = Vec::new();
        if strongest > 0.0 {
            for (col, value) in strengths {
                if value > 0.0 && value >= strongest * threshold {
                    neighbors.push(col);
                }
            }
        }
        result.push(neighbors);
    }
    Ok(result)
}

fn graph_as_csr(graph: &[Vec<u32>]) -> Result<Csr32Matrix, Box<dyn Error>> {
    let mut row_ptr = Vec::with_capacity(graph.len() + 1);
    let mut columns = Vec::new();
    row_ptr.push(0_u32);
    for row in graph {
        columns.extend_from_slice(row);
        row_ptr.push(u32::try_from(columns.len())?);
    }
    let ones = vec![1.0; columns.len()];
    Ok(Csr32Matrix::new(
        graph.len(),
        graph.len(),
        row_ptr,
        columns,
        ones,
    )?)
}

fn graph_from_abtm(topology: &AbtmTopology) -> Result<Vec<Vec<u32>>, Box<dyn Error>> {
    let mut output = Vec::with_capacity(topology.nrows());
    for row in 0..topology.nrows() {
        let mut indices = Vec::new();
        for word in topology.row(row)?.words() {
            let mut bits = word.mask();
            while bits != 0 {
                indices.push(u32::try_from(
                    word.base_col() + bits.trailing_zeros() as usize,
                )?);
                bits &= bits - 1;
            }
        }
        output.push(indices);
    }
    Ok(output)
}

/// Reference: explicit adjacency expansion followed by sort/dedup.
fn csr_two_hop(graph: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let mut result = Vec::with_capacity(graph.len());
    for neighbors in graph {
        let mut support = Vec::new();
        for &neighbor in neighbors {
            support.extend_from_slice(&graph[neighbor as usize]);
        }
        support.sort_unstable();
        support.dedup();
        result.push(support);
    }
    result
}

/// ABTM: OR 64-bit topology words, then enumerate the exact same support.
fn abtm_two_hop(topology: &AbtmTopology) -> Result<Vec<Vec<u32>>, Box<dyn Error>> {
    let word_count = topology.ncols().div_ceil(64);
    let mut stamps = vec![usize::MAX; word_count];
    let mut masks = vec![0_u64; word_count];
    let mut touched = Vec::<usize>::new();
    let mut result = Vec::with_capacity(topology.nrows());
    for row in 0..topology.nrows() {
        touched.clear();
        for word in topology.row(row)?.words() {
            let mut bits = word.mask();
            while bits != 0 {
                let neighbor = word.base_col() + bits.trailing_zeros() as usize;
                for next in topology.row(neighbor)?.words() {
                    let index = next.word_index() as usize;
                    if stamps[index] != row {
                        stamps[index] = row;
                        masks[index] = 0;
                        touched.push(index);
                    }
                    masks[index] |= next.mask();
                }
                bits &= bits - 1;
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

fn profile<T>(repeats: usize, mut operation: impl FnMut() -> T) -> (T, f64) {
    let mut times = Vec::with_capacity(repeats);
    let mut last = None;
    let _ = black_box(operation()); // warm up outside the measured sample
    for _ in 0..repeats {
        let start = Instant::now();
        let result = black_box(operation());
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        last = Some(result);
    }
    times.sort_by(f64::total_cmp);
    (last.expect("nonzero repeats"), times[times.len() / 2])
}

fn hash_support(rows: &[Vec<u32>]) -> u64 {
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

fn run(matrix: &Csr32Matrix, threshold: f64, repeats: usize) -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let graph = strong_graph(matrix, threshold)?;
    let numeric_strength_ms = start.elapsed().as_secs_f64() * 1000.0;
    let edges: usize = graph.iter().map(Vec::len).sum();
    let start = Instant::now();
    let reference_csr = graph_as_csr(&graph)?;
    let graph_csr_prepare_ms = start.elapsed().as_secs_f64() * 1000.0;

    let start = Instant::now();
    let topology = AbtmTopology::from_csr32(&reference_csr)?;
    topology.validate()?;
    let abtm_conversion_ms = start.elapsed().as_secs_f64() * 1000.0;
    if graph_from_abtm(&topology)? != graph {
        return Err("strong-connection graph CSR/ABTM mismatch".into());
    }

    let (csr, csr_ms) = profile(repeats, || csr_two_hop(&graph));
    let (abtm_result, abtm_ms) = profile(repeats, || abtm_two_hop(&topology));
    let abtm_result = abtm_result?;
    if csr != abtm_result {
        return Err("two-hop symbolic support CSR/ABTM mismatch".into());
    }
    let output_nnz: usize = csr.iter().map(Vec::len).sum();
    let csr_graph_bytes = graph
        .iter()
        .map(|row| row.capacity() * std::mem::size_of::<u32>())
        .sum::<usize>()
        + graph.capacity() * std::mem::size_of::<Vec<u32>>();
    println!(
        "C1H12A2|rows={}|nnz={}|threshold={threshold:.6}|strong_edges={edges}|symbolic_nnz={output_nnz}|csr_graph_bytes={csr_graph_bytes}|abtm_metadata_bytes={}|abtm_nonempty_words={}|strength_ms={numeric_strength_ms:.6}|graph_csr_prepare_ms={graph_csr_prepare_ms:.6}|abtm_conversion_ms={abtm_conversion_ms:.6}|csr_symbolic_ms={csr_ms:.6}|abtm_symbolic_ms={abtm_ms:.6}|repeats={repeats}|symbolic_hash={:016x}|exact_equal=true",
        matrix.nrows(),
        matrix.nnz(),
        topology.stats().metadata_bytes,
        topology.stats().nonempty_words,
        hash_support(&csr)
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let (matrix, _) = read_matrix_market(&args.matrix)?;
    run(&matrix, args.threshold, args.repeats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_and_abtm_symbolic_are_identical_across_word_boundaries() {
        let mut row_ptr = vec![0_u32];
        let mut columns = Vec::new();
        let mut values = Vec::new();
        let n = 131_usize;
        for row in 0..n {
            for col in 0..n {
                if row == col || row.abs_diff(col) == 1 {
                    columns.push(col as u32);
                    values.push(if row == col { 4.0 } else { -1.0 });
                }
            }
            row_ptr.push(columns.len() as u32);
        }
        let matrix = Csr32Matrix::new(n, n, row_ptr, columns, values).unwrap();
        let graph = strong_graph(&matrix, 0.25).unwrap();
        let abtm = AbtmTopology::from_csr32(&graph_as_csr(&graph).unwrap()).unwrap();
        assert_eq!(graph, graph_from_abtm(&abtm).unwrap());
        assert_eq!(csr_two_hop(&graph), abtm_two_hop(&abtm).unwrap());
    }

    #[test]
    fn disconnected_and_empty_rows_remain_exact() {
        let matrix = Csr32Matrix::new(
            4,
            4,
            vec![0, 2, 4, 5, 6],
            vec![0, 1, 0, 1, 2, 3],
            vec![2.0, -1.0, -1.0, 2.0, 1.0, 3.0],
        )
        .unwrap();
        let graph = strong_graph(&matrix, 0.25).unwrap();
        assert!(graph[2].is_empty());
        let abtm = AbtmTopology::from_csr32(&graph_as_csr(&graph).unwrap()).unwrap();
        assert_eq!(csr_two_hop(&graph), abtm_two_hop(&abtm).unwrap());
    }
}
