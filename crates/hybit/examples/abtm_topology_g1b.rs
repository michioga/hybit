use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use hybit::{read_matrix_market, AbtmTopology, Csr32Matrix};

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
                    println!("Usage: abtm_topology_g1b --matrix A.mtx");
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

fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn percentile(sorted: &[usize], q: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let index = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[index]
}

fn filtered_csr(matrix: &Csr32Matrix, lane: u32) -> Result<Csr32Matrix, Box<dyn Error>> {
    let mut row_ptr = Vec::with_capacity(matrix.nrows() + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0u32);

    for row in 0..matrix.nrows() {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;

        for &col in &matrix.col_idx()[start..end] {
            let key = ((row as u64) << 32) ^ col as u64 ^ 0x38d0_2026_1005_0001;
            let h = mix(key);
            let two_bits = ((h >> lane) & 0b11) as u8;

            // Each topology keeps roughly 75% of structural positions. Lanes
            // 0 and 2 are disjoint hash-bit pairs, so their overlap exercises
            // partial word intersections without depending on numeric values.
            if two_bits != 0 {
                col_idx.push(col);
                values.push(1.0);
            }
        }

        if col_idx.len() > u32::MAX as usize {
            return Err("filtered CSR exceeds u32 offset range".into());
        }
        row_ptr.push(col_idx.len() as u32);
    }

    Ok(Csr32Matrix::new(
        matrix.nrows(),
        matrix.ncols(),
        row_ptr,
        col_idx,
        values,
    )?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;

    println!(
        "HyBIT {} ABTM G1b topology occupancy/rank-select/merge profile",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());

    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    println!(
        "Matrix Market       : {:?}, {} input entries -> {} CSR nnz",
        mm.symmetry, mm.input_entries, mm.csr_nnz
    );
    println!(
        "dimensions          : {} x {}",
        matrix.nrows(),
        matrix.ncols()
    );
    println!("stored CSR nnz      : {}", matrix.nnz());

    let topology_start = Instant::now();
    let topology = AbtmTopology::from_csr32(&matrix)?;
    let topology_build_ms = topology_start.elapsed().as_secs_f64() * 1.0e3;
    topology.validate()?;
    let stats = topology.stats();

    let csr_metadata_bytes = matrix.metadata_bytes();
    let topology_metadata_bytes = stats.metadata_bytes;
    let topology_over_csr_metadata = if csr_metadata_bytes == 0 {
        0.0
    } else {
        topology_metadata_bytes as f64 / csr_metadata_bytes as f64
    };

    println!(
        "G1B_METADATA|nrows={}|ncols={}|structural_nnz={}|nonempty_words={}|csr_metadata_bytes={csr_metadata_bytes}|topology_metadata_bytes={topology_metadata_bytes}|topology_over_csr_metadata={topology_over_csr_metadata:.9e}|metadata_bytes_per_nnz={:.9e}|build_ms={topology_build_ms:.6}",
        stats.nrows,
        stats.ncols,
        stats.structural_nnz,
        stats.nonempty_words,
        stats.metadata_bytes_per_nnz(),
    );

    let mut occupancy = Vec::with_capacity(stats.nonempty_words);
    for row in 0..topology.nrows() {
        occupancy.extend(topology.row(row)?.words().map(|word| word.popcount()));
    }
    occupancy.sort_unstable();

    let words_le3 = occupancy.iter().filter(|&&count| count <= 3).count();
    let words_le8 = occupancy.iter().filter(|&&count| count <= 8).count();
    let words_ge40 = occupancy.iter().filter(|&&count| count >= 40).count();
    let denom = occupancy.len().max(1) as f64;

    println!(
        "G1B_OCCUPANCY|words={}|min={}|p10={}|p50={}|p90={}|p95={}|max={}|average={:.9e}|fraction_le3={:.9e}|fraction_le8={:.9e}|fraction_ge40={:.9e}",
        occupancy.len(),
        occupancy.first().copied().unwrap_or(0),
        percentile(&occupancy, 0.10),
        percentile(&occupancy, 0.50),
        percentile(&occupancy, 0.90),
        percentile(&occupancy, 0.95),
        occupancy.last().copied().unwrap_or(0),
        stats.average_word_nnz(),
        words_le3 as f64 / denom,
        words_le8 as f64 / denom,
        words_ge40 as f64 / denom,
    );

    // Measure the chunk-local rank/select primitive that a packed value stream
    // actually needs. This intentionally excludes AbtmTopologyRow::rank/select,
    // whose row-wide scan is a convenience API rather than the G1 packed-value
    // addressing primitive.
    let word_rank_start = Instant::now();
    let mut checked = 0usize;
    let mut fingerprint = 0u64;

    for row in 0..topology.nrows() {
        for word in topology.row(row)?.words() {
            for ordinal in 0..word.popcount() {
                let bit = word
                    .select(ordinal)
                    .ok_or("word select unexpectedly failed")?;
                if word.rank(bit)? != ordinal {
                    return Err("word rank/select invariant failed".into());
                }
                fingerprint ^= mix(((row as u64) << 32)
                    ^ word.word_index() as u64
                    ^ ((bit as u64) << 48)
                    ^ ordinal as u64);
                checked += 1;
            }
        }
    }
    let word_rank_select_ms = word_rank_start.elapsed().as_secs_f64() * 1.0e3;
    let ns_per_structural_nnz = if checked == 0 {
        0.0
    } else {
        word_rank_select_ms * 1.0e6 / checked as f64
    };

    println!(
        "G1B_WORD_RANK_SELECT|checked={checked}|fingerprint=0x{fingerprint:016x}|elapsed_ms={word_rank_select_ms:.6}|ns_per_structural_nnz={ns_per_structural_nnz:.9e}"
    );

    // Build two deterministic, partially overlapping structural views. This
    // exercises the real merge path (missing words + partial masks) rather
    // than only the A op A fast/equal-word case used by G1a invariants.
    let split_start = Instant::now();
    let a_csr = filtered_csr(&matrix, 0)?;
    let b_csr = filtered_csr(&matrix, 2)?;
    let split_csr_ms = split_start.elapsed().as_secs_f64() * 1.0e3;

    let start = Instant::now();
    let a = AbtmTopology::from_csr32(&a_csr)?;
    let a_build_ms = start.elapsed().as_secs_f64() * 1.0e3;

    let start = Instant::now();
    let b = AbtmTopology::from_csr32(&b_csr)?;
    let b_build_ms = start.elapsed().as_secs_f64() * 1.0e3;

    let start = Instant::now();
    let intersection = a.intersection(&b)?;
    let intersection_ms = start.elapsed().as_secs_f64() * 1.0e3;

    let start = Instant::now();
    let union = a.union(&b)?;
    let union_ms = start.elapsed().as_secs_f64() * 1.0e3;

    let start = Instant::now();
    let a_not_b = a.and_not(&b)?;
    let and_not_ms = start.elapsed().as_secs_f64() * 1.0e3;

    let start = Instant::now();
    let xor = a.xor(&b)?;
    let xor_ms = start.elapsed().as_secs_f64() * 1.0e3;

    // Boolean identities over independently prepared topologies.
    if intersection.union(&xor)? != union {
        return Err("(A AND B) OR (A XOR B) != A OR B".into());
    }
    if a_not_b.intersection(&b)?.structural_nnz() != 0 {
        return Err("(A AND-NOT B) AND B is not empty".into());
    }
    if intersection.structural_nnz() + xor.structural_nnz() != union.structural_nnz() {
        return Err("intersection/xor/union cardinality invariant failed".into());
    }

    let source_nnz = topology.structural_nnz().max(1) as f64;
    println!(
        "G1B_MERGE|source_nnz={}|a_nnz={}|b_nnz={}|intersection_nnz={}|union_nnz={}|a_not_b_nnz={}|xor_nnz={}|intersection_over_source={:.9e}|union_over_source={:.9e}|split_csr_ms={split_csr_ms:.6}|a_build_ms={a_build_ms:.6}|b_build_ms={b_build_ms:.6}|intersection_ms={intersection_ms:.6}|union_ms={union_ms:.6}|and_not_ms={and_not_ms:.6}|xor_ms={xor_ms:.6}",
        topology.structural_nnz(),
        a.structural_nnz(),
        b.structural_nnz(),
        intersection.structural_nnz(),
        union.structural_nnz(),
        a_not_b.structural_nnz(),
        xor.structural_nnz(),
        intersection.structural_nnz() as f64 / source_nnz,
        union.structural_nnz() as f64 / source_nnz,
    );

    Ok(())
}
