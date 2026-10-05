use std::env;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use hybit_matrix::{read_matrix_market, AbtmDualTopology, AbtmTopology, AbtmTopologyRow};

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    pairs_per_row: usize,
    repeats: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut pairs_per_row = 8usize;
        let mut repeats = 5usize;
        let mut it = env::args().skip(1);

        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => {
                    matrix = Some(PathBuf::from(
                        it.next().ok_or("missing value after --matrix")?,
                    ));
                }
                "--pairs-per-row" => {
                    pairs_per_row = it
                        .next()
                        .ok_or("missing value after --pairs-per-row")?
                        .parse()?;
                    if pairs_per_row == 0 {
                        return Err("--pairs-per-row must be >= 1".into());
                    }
                }
                "--repeats" => {
                    repeats = it.next().ok_or("missing value after --repeats")?.parse()?;
                    if repeats == 0 {
                        return Err("--repeats must be >= 1".into());
                    }
                }
                "-h" | "--help" => {
                    println!(
                        "Usage: abtm_dual_topology_g2c --matrix A.mtx [--pairs-per-row N] [--repeats N]"
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
            pairs_per_row,
            repeats,
        })
    }
}

#[derive(Clone, Debug)]
struct ExplicitSupport {
    row_ptr: Vec<u32>,
    indices: Vec<u32>,
}

impl ExplicitSupport {
    fn from_topology(topology: &AbtmTopology) -> Result<Self, Box<dyn Error>> {
        let mut row_ptr = Vec::with_capacity(topology.nrows() + 1);
        let mut indices = Vec::with_capacity(topology.structural_nnz());
        row_ptr.push(0);

        for row in 0..topology.nrows() {
            let support = topology.row(row)?;
            for ordinal in 0..support.popcount() {
                let index = support
                    .select(ordinal)
                    .ok_or("explicit support select unexpectedly failed")?;
                if index > u32::MAX as usize {
                    return Err("support index exceeds u32 range".into());
                }
                indices.push(index as u32);
            }
            if indices.len() > u32::MAX as usize {
                return Err("support entry count exceeds u32 range".into());
            }
            row_ptr.push(indices.len() as u32);
        }

        Ok(Self { row_ptr, indices })
    }

    fn row(&self, row: usize) -> &[u32] {
        let start = self.row_ptr[row] as usize;
        let end = self.row_ptr[row + 1] as usize;
        &self.indices[start..end]
    }

    fn metadata_bytes(&self) -> usize {
        self.row_ptr.len() * std::mem::size_of::<u32>()
            + self.indices.len() * std::mem::size_of::<u32>()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct IntersectionStats {
    overlap_products: usize,
    metadata_comparisons: usize,
    mask_ands: usize,
}

fn explicit_intersection(a: &[u32], b: &[u32]) -> IntersectionStats {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut stats = IntersectionStats::default();

    while i < a.len() && j < b.len() {
        stats.metadata_comparisons += 1;
        if a[i] < b[j] {
            i += 1;
        } else if b[j] < a[i] {
            j += 1;
        } else {
            stats.overlap_products += 1;
            i += 1;
            j += 1;
        }
    }

    stats
}

fn bitmap_intersection(a: AbtmTopologyRow<'_>, b: AbtmTopologyRow<'_>) -> IntersectionStats {
    let mut a_words = a.words().peekable();
    let mut b_words = b.words().peekable();
    let mut stats = IntersectionStats::default();

    while let (Some(aw), Some(bw)) = (a_words.peek().copied(), b_words.peek().copied()) {
        stats.metadata_comparisons += 1;
        if aw.word_index() < bw.word_index() {
            a_words.next();
        } else if bw.word_index() < aw.word_index() {
            b_words.next();
        } else {
            stats.mask_ands += 1;
            stats.overlap_products += (aw.mask() & bw.mask()).count_ones() as usize;
            a_words.next();
            b_words.next();
        }
    }

    stats
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn make_pairs(n: usize, pairs_per_row: usize) -> Vec<(u32, u32)> {
    let mut pairs = Vec::with_capacity(n.saturating_mul(pairs_per_row));

    for row in 0..n {
        for slot in 0..pairs_per_row {
            let col = match slot {
                0 => row,
                1 => (row + 1) % n,
                _ => {
                    let key = (row as u64)
                        ^ (slot as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        ^ 0x38d0_2026_1005_02c0;
                    (mix(key) % n as u64) as usize
                }
            };
            pairs.push((row as u32, col as u32));
        }
    }

    pairs
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;

    println!(
        "HyBIT {} ABTM G2c dual-topology sparse-dot support intersection",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!("pairs per row       : {}", args.pairs_per_row);
    println!("repeats             : {}", args.repeats);

    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    if matrix.nrows() != matrix.ncols() {
        return Err("G2c same-matrix A*A benchmark requires a square matrix".into());
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
    println!("stored CSR nnz      : {}", matrix.nnz());

    let dual_start = Instant::now();
    let dual = AbtmDualTopology::from_csr32(&matrix)?;
    let dual_prepare_ms = dual_start.elapsed().as_secs_f64() * 1.0e3;
    let dual_stats = dual.stats();

    let explicit_start = Instant::now();
    let explicit_rows = ExplicitSupport::from_topology(dual.row_topology())?;
    let explicit_cols = ExplicitSupport::from_topology(dual.column_topology())?;
    let explicit_prepare_ms = explicit_start.elapsed().as_secs_f64() * 1.0e3;

    let explicit_metadata_bytes = explicit_rows
        .metadata_bytes()
        .saturating_add(explicit_cols.metadata_bytes());
    let dual_over_explicit_metadata = if explicit_metadata_bytes == 0 {
        0.0
    } else {
        dual_stats.total_metadata_bytes() as f64 / explicit_metadata_bytes as f64
    };

    println!(
        "G2C_PREPARE|structural_nnz={}|row_topology_bytes={}|column_topology_bytes={}|dual_topology_bytes={}|explicit_row_column_bytes={explicit_metadata_bytes}|dual_over_explicit_metadata={dual_over_explicit_metadata:.9e}|dual_prepare_ms={dual_prepare_ms:.6}|explicit_prepare_ms={explicit_prepare_ms:.6}",
        dual_stats.structural_nnz,
        dual_stats.row_metadata_bytes,
        dual_stats.column_metadata_bytes,
        dual_stats.total_metadata_bytes(),
    );

    let pairs = make_pairs(matrix.nrows(), args.pairs_per_row);

    let mut explicit_total = IntersectionStats::default();
    let mut bitmap_total = IntersectionStats::default();
    let mut empty_pairs = 0usize;

    for &(row, col) in &pairs {
        let row = row as usize;
        let col = col as usize;

        let explicit = explicit_intersection(explicit_rows.row(row), explicit_cols.row(col));
        let bitmap = bitmap_intersection(dual.row(row)?, dual.column(col)?);

        if explicit.overlap_products != bitmap.overlap_products {
            return Err(format!(
                "G2c overlap mismatch at row={row}, col={col}: explicit={}, bitmap={}",
                explicit.overlap_products, bitmap.overlap_products
            )
            .into());
        }

        explicit_total.overlap_products += explicit.overlap_products;
        explicit_total.metadata_comparisons += explicit.metadata_comparisons;

        bitmap_total.overlap_products += bitmap.overlap_products;
        bitmap_total.metadata_comparisons += bitmap.metadata_comparisons;
        bitmap_total.mask_ands += bitmap.mask_ands;

        if bitmap.overlap_products == 0 {
            empty_pairs += 1;
        }
    }

    if explicit_total.overlap_products != bitmap_total.overlap_products {
        return Err("G2c aggregate overlap mismatch".into());
    }

    let mut explicit_samples = Vec::with_capacity(args.repeats);
    let mut bitmap_samples = Vec::with_capacity(args.repeats);

    for _ in 0..args.repeats {
        let start = Instant::now();
        let mut overlap = 0usize;
        let mut comparisons = 0usize;
        for &(row, col) in &pairs {
            let result = explicit_intersection(
                explicit_rows.row(row as usize),
                explicit_cols.row(col as usize),
            );
            overlap += result.overlap_products;
            comparisons += result.metadata_comparisons;
        }
        explicit_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box((overlap, comparisons));

        let start = Instant::now();
        let mut overlap = 0usize;
        let mut comparisons = 0usize;
        let mut ands = 0usize;
        for &(row, col) in &pairs {
            let result = bitmap_intersection(dual.row(row as usize)?, dual.column(col as usize)?);
            overlap += result.overlap_products;
            comparisons += result.metadata_comparisons;
            ands += result.mask_ands;
        }
        bitmap_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box((overlap, comparisons, ands));
    }

    let explicit_ms = median(&mut explicit_samples);
    let bitmap_ms = median(&mut bitmap_samples);
    let bitmap_over_explicit = if explicit_ms == 0.0 {
        0.0
    } else {
        bitmap_ms / explicit_ms
    };
    let comparison_reduction = if explicit_total.metadata_comparisons == 0 {
        0.0
    } else {
        1.0 - bitmap_total.metadata_comparisons as f64 / explicit_total.metadata_comparisons as f64
    };
    let empty_pair_ratio = if pairs.is_empty() {
        0.0
    } else {
        empty_pairs as f64 / pairs.len() as f64
    };
    let average_overlap = if pairs.is_empty() {
        0.0
    } else {
        bitmap_total.overlap_products as f64 / pairs.len() as f64
    };

    println!(
        "G2C_INTERSECTION|pairs={}|empty_pairs={empty_pairs}|empty_pair_ratio={empty_pair_ratio:.9e}|overlap_products={}|average_overlap_per_pair={average_overlap:.9e}|explicit_index_comparisons={}|bitmap_word_comparisons={}|bitmap_mask_ands={}|comparison_reduction={comparison_reduction:.9e}|explicit_ms={explicit_ms:.6}|bitmap_ms={bitmap_ms:.6}|bitmap_over_explicit={bitmap_over_explicit:.9e}",
        pairs.len(),
        bitmap_total.overlap_products,
        explicit_total.metadata_comparisons,
        bitmap_total.metadata_comparisons,
        bitmap_total.mask_ands,
    );

    Ok(())
}
