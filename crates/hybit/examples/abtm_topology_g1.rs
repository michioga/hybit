use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use hybit::{read_matrix_market, AbtmTopology};

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
                    ))
                }
                "-h" | "--help" => {
                    println!("Usage: abtm_topology_g1 --matrix A.mtx");
                    std::process::exit(0);
                }
                other if !other.starts_with('-') && matrix.is_none() => {
                    matrix = Some(PathBuf::from(other))
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

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    println!(
        "HyBIT {} ABTM G1 scalar topology algebra profile",
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

    let start = Instant::now();
    let topology = AbtmTopology::from_csr32(&matrix)?;
    let build_ms = start.elapsed().as_secs_f64() * 1.0e3;
    topology.validate()?;
    let stats = topology.stats();
    println!(
        "G1_TOPOLOGY|nrows={}|ncols={}|stored_csr_nnz={}|structural_nnz={}|nonempty_words={}|average_word_nnz={:.9e}|word_fill_ratio={:.9e}|metadata_bytes={}|metadata_bytes_per_nnz={:.9e}|build_ms={build_ms:.6}",
        stats.nrows, stats.ncols, matrix.nnz(), stats.structural_nnz, stats.nonempty_words,
        stats.average_word_nnz(), stats.word_fill_ratio(), stats.metadata_bytes,
        stats.metadata_bytes_per_nnz()
    );

    let start = Instant::now();
    let mut fingerprint = 0u64;
    let mut selected = 0usize;
    for row_index in 0..topology.nrows() {
        let row = topology.row(row_index)?;
        let mut row_ordinal = 0usize;
        for word in row.words() {
            for word_ordinal in 0..word.popcount() {
                let bit = word
                    .select(word_ordinal)
                    .ok_or("G1 word select unexpectedly failed")?;
                if word.rank(bit)? != word_ordinal {
                    return Err("G1 word rank/select invariant failed".into());
                }
                let col = word.base_col() + bit;
                if row.rank(col)? != row_ordinal {
                    return Err("G1 row rank invariant failed".into());
                }
                if row.select(row_ordinal) != Some(col) {
                    return Err("G1 row select invariant failed".into());
                }
                fingerprint ^= mix(((row_index as u64) << 32)
                    ^ col as u64
                    ^ (row_ordinal as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
                row_ordinal += 1;
                selected += 1;
            }
        }
        if row_ordinal != row.popcount() {
            return Err("G1 row popcount invariant failed".into());
        }
    }
    if selected != topology.structural_nnz() {
        return Err("G1 topology popcount invariant failed".into());
    }
    let rank_select_ms = start.elapsed().as_secs_f64() * 1.0e3;
    println!("G1_RANK_SELECT|selected={selected}|fingerprint=0x{fingerprint:016x}|elapsed_ms={rank_select_ms:.6}");

    let start = Instant::now();
    let intersection = topology.intersection(&topology)?;
    let intersection_ms = start.elapsed().as_secs_f64() * 1.0e3;
    let start = Instant::now();
    let union = topology.union(&topology)?;
    let union_ms = start.elapsed().as_secs_f64() * 1.0e3;
    let start = Instant::now();
    let and_not = topology.and_not(&topology)?;
    let and_not_ms = start.elapsed().as_secs_f64() * 1.0e3;
    let start = Instant::now();
    let xor = topology.xor(&topology)?;
    let xor_ms = start.elapsed().as_secs_f64() * 1.0e3;

    if intersection != topology || union != topology {
        return Err("G1 idempotent Boolean algebra invariant failed".into());
    }
    if and_not.structural_nnz() != 0 || xor.structural_nnz() != 0 {
        return Err("G1 self-difference invariant failed".into());
    }
    println!(
        "G1_ALGEBRA|intersection_nnz={}|union_nnz={}|and_not_nnz={}|xor_nnz={}|intersection_ms={intersection_ms:.6}|union_ms={union_ms:.6}|and_not_ms={and_not_ms:.6}|xor_ms={xor_ms:.6}",
        intersection.structural_nnz(), union.structural_nnz(), and_not.structural_nnz(), xor.structural_nnz()
    );
    Ok(())
}
