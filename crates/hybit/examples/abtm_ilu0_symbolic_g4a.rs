use std::env;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use hybit_matrix::{read_matrix_market, AbtmTopology, Csr32Matrix};

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    repeats: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut repeats = 5usize;
        let mut it = env::args().skip(1);

        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => {
                    matrix = Some(PathBuf::from(
                        it.next().ok_or("missing value after --matrix")?,
                    ));
                }
                "--repeats" => {
                    repeats = it.next().ok_or("missing value after --repeats")?.parse()?;
                    if repeats == 0 {
                        return Err("--repeats must be >= 1".into());
                    }
                }
                "-h" | "--help" => {
                    println!("Usage: abtm_ilu0_symbolic_g4a --matrix A.mtx [--repeats N]");
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
            repeats,
        })
    }
}

#[derive(Clone, Debug)]
struct PreparedWordRows {
    nrows: usize,
    row_ptr: Vec<u32>,
    word_indices: Vec<u32>,
    masks: Vec<u64>,
}

impl PreparedWordRows {
    fn from_topology(topology: &AbtmTopology) -> Result<Self, Box<dyn Error>> {
        let mut row_ptr = Vec::with_capacity(topology.nrows() + 1);
        let mut word_indices = Vec::with_capacity(topology.nonempty_words());
        let mut masks = Vec::with_capacity(topology.nonempty_words());
        row_ptr.push(0);

        for row in 0..topology.nrows() {
            for word in topology.row(row)?.words() {
                word_indices.push(word.word_index());
                masks.push(word.mask());
            }
            row_ptr.push(u32::try_from(word_indices.len())?);
        }

        Ok(Self {
            nrows: topology.nrows(),
            row_ptr,
            word_indices,
            masks,
        })
    }

    fn row(&self, row: usize) -> (&[u32], &[u64]) {
        debug_assert!(row < self.nrows);
        let start = self.row_ptr[row] as usize;
        let end = self.row_ptr[row + 1] as usize;
        (&self.word_indices[start..end], &self.masks[start..end])
    }

    fn metadata_bytes(&self) -> usize {
        self.row_ptr.len() * std::mem::size_of::<u32>()
            + self.word_indices.len() * std::mem::size_of::<u32>()
            + self.masks.len() * std::mem::size_of::<u64>()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CsrSymbolicStats {
    lower_pivots: usize,
    upper_candidates: usize,
    successful_updates: usize,
    checksum: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct AbtmSymbolicStats {
    lower_pivots: usize,
    word_seek_operations: usize,
    word_merge_steps: usize,
    intersection_words: usize,
    successful_updates: usize,
    checksum: u64,
}

fn canonicalize_like_ilu0(matrix: &Csr32Matrix) -> Result<Csr32Matrix, Box<dyn Error>> {
    if matrix.nrows() != matrix.ncols() {
        return Err("ILU(0) symbolic analysis requires a square matrix".into());
    }

    let n = matrix.nrows();
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(matrix.nnz());
    let mut values = Vec::with_capacity(matrix.nnz());
    let mut entries = Vec::<(u32, f64)>::new();
    row_ptr.push(0);

    for row in 0..n {
        entries.clear();
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        entries.extend(
            matrix.col_idx()[start..end]
                .iter()
                .copied()
                .zip(matrix.values()[start..end].iter().copied()),
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
                return Err("ILU(0) duplicate accumulation became non-finite".into());
            }

            if col as usize == row {
                found_diagonal = true;
            } else if value == 0.0 {
                continue;
            }

            col_idx.push(col);
            values.push(value);
        }

        if !found_diagonal {
            return Err(
                format!("ILU(0) canonical pattern is missing diagonal at row {row}").into(),
            );
        }

        row_ptr.push(u32::try_from(col_idx.len())?);
    }

    Ok(Csr32Matrix::new(n, n, row_ptr, col_idx, values)?)
}

fn diagonal_positions(matrix: &Csr32Matrix) -> Result<Vec<usize>, Box<dyn Error>> {
    let mut diag = Vec::with_capacity(matrix.nrows());

    for row in 0..matrix.nrows() {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        let offset = matrix.col_idx()[start..end]
            .binary_search(&(row as u32))
            .map_err(|_| format!("canonical ILU(0) pattern is missing diagonal at row {row}"))?;
        diag.push(start + offset);
    }

    Ok(diag)
}

#[inline]
fn update_checksum(checksum: &mut u64, row: usize, pivot: usize, target: usize) {
    let mut x = (row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (pivot as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ (target as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    *checksum = checksum.rotate_left(7) ^ x;
}

fn lower_bound(words: &[u32], target: u32) -> usize {
    match words.binary_search(&target) {
        Ok(index) | Err(index) => index,
    }
}

fn mask_strictly_above(mask: u64, word_index: u32, pivot: usize) -> u64 {
    let pivot_word = (pivot / 64) as u32;
    if word_index < pivot_word {
        return 0;
    }
    if word_index > pivot_word {
        return mask;
    }

    let bit = pivot % 64;
    if bit == 63 {
        0
    } else {
        mask & (!0u64 << (bit + 1))
    }
}

fn enumerate_abtm_intersection(
    prepared: &PreparedWordRows,
    row: usize,
    pivot: usize,
    out: &mut Vec<u32>,
    stats: &mut AbtmSymbolicStats,
) {
    out.clear();

    let (a_words, a_masks) = prepared.row(row);
    let (b_words, b_masks) = prepared.row(pivot);
    let pivot_word = (pivot / 64) as u32;

    stats.word_seek_operations = stats.word_seek_operations.saturating_add(2);
    let mut a = lower_bound(a_words, pivot_word);
    let mut b = lower_bound(b_words, pivot_word);

    while a < a_words.len() && b < b_words.len() {
        stats.word_merge_steps = stats.word_merge_steps.saturating_add(1);

        let aw = a_words[a];
        let bw = b_words[b];

        if aw < bw {
            a += 1;
            continue;
        }
        if bw < aw {
            b += 1;
            continue;
        }

        let mut bits = a_masks[a] & b_masks[b];
        bits = mask_strictly_above(bits, aw, pivot);

        if bits != 0 {
            stats.intersection_words = stats.intersection_words.saturating_add(1);
        }

        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            let target = aw as usize * 64 + bit;
            out.push(target as u32);
            stats.successful_updates = stats.successful_updates.saturating_add(1);
            update_checksum(&mut stats.checksum, row, pivot, target);
            bits &= bits - 1;
        }

        a += 1;
        b += 1;
    }
}

fn csr_symbolic(matrix: &Csr32Matrix, diag: &[usize]) -> CsrSymbolicStats {
    let mut stats = CsrSymbolicStats::default();

    for (row, &row_diag) in diag.iter().enumerate().take(matrix.nrows()) {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;

        for p in start..row_diag {
            stats.lower_pivots = stats.lower_pivots.saturating_add(1);
            let pivot = matrix.col_idx()[p] as usize;
            let pivot_diag = diag[pivot];
            let pivot_end = matrix.row_ptr()[pivot + 1] as usize;

            for q in (pivot_diag + 1)..pivot_end {
                stats.upper_candidates = stats.upper_candidates.saturating_add(1);
                let target = matrix.col_idx()[q];
                if matrix.col_idx()[(p + 1)..end]
                    .binary_search(&target)
                    .is_ok()
                {
                    stats.successful_updates = stats.successful_updates.saturating_add(1);
                    update_checksum(&mut stats.checksum, row, pivot, target as usize);
                }
            }
        }
    }

    stats
}

fn abtm_symbolic(
    matrix: &Csr32Matrix,
    diag: &[usize],
    prepared: &PreparedWordRows,
) -> AbtmSymbolicStats {
    let mut stats = AbtmSymbolicStats::default();
    let mut scratch = Vec::<u32>::new();

    for (row, &row_diag) in diag.iter().enumerate().take(matrix.nrows()) {
        let start = matrix.row_ptr()[row] as usize;

        for p in start..row_diag {
            stats.lower_pivots = stats.lower_pivots.saturating_add(1);
            let pivot = matrix.col_idx()[p] as usize;
            enumerate_abtm_intersection(prepared, row, pivot, &mut scratch, &mut stats);
        }
    }

    stats
}

fn validate_exact(
    matrix: &Csr32Matrix,
    diag: &[usize],
    prepared: &PreparedWordRows,
) -> Result<usize, Box<dyn Error>> {
    let mut expected = Vec::<u32>::new();
    let mut actual = Vec::<u32>::new();
    let mut dummy_stats = AbtmSymbolicStats::default();
    let mut checked_pivots = 0usize;

    for (row, &row_diag) in diag.iter().enumerate().take(matrix.nrows()) {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;

        for p in start..row_diag {
            checked_pivots = checked_pivots.saturating_add(1);
            let pivot = matrix.col_idx()[p] as usize;
            let pivot_diag = diag[pivot];
            let pivot_end = matrix.row_ptr()[pivot + 1] as usize;

            expected.clear();
            for q in (pivot_diag + 1)..pivot_end {
                let target = matrix.col_idx()[q];
                if matrix.col_idx()[(p + 1)..end]
                    .binary_search(&target)
                    .is_ok()
                {
                    expected.push(target);
                }
            }

            enumerate_abtm_intersection(prepared, row, pivot, &mut actual, &mut dummy_stats);

            if actual != expected {
                return Err(format!(
                    "G4a symbolic mismatch at row={row}, pivot={pivot}: CSR={} targets, ABTM={} targets",
                    expected.len(),
                    actual.len()
                )
                .into());
            }
        }
    }

    Ok(checked_pivots)
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;

    println!(
        "HyBIT {} ABTM G4a ILU(0) symbolic intersection",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!("repeats             : {}", args.repeats);

    let (matrix, info) = read_matrix_market(&args.matrix)?;
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

    let start = Instant::now();
    let canonical = canonicalize_like_ilu0(&matrix)?;
    let canonicalize_ms = start.elapsed().as_secs_f64() * 1.0e3;

    let diag = diagonal_positions(&canonical)?;

    let start = Instant::now();
    let topology = AbtmTopology::from_csr32(&canonical)?;
    let topology_prepare_ms = start.elapsed().as_secs_f64() * 1.0e3;

    let start = Instant::now();
    let prepared = PreparedWordRows::from_topology(&topology)?;
    let word_prepare_ms = start.elapsed().as_secs_f64() * 1.0e3;

    let topo_stats = topology.stats();

    println!(
        "G4A_PREPARE|canonical_nnz={}|canonicalize_ms={canonicalize_ms:.6}|topology_prepare_ms={topology_prepare_ms:.6}|word_prepare_ms={word_prepare_ms:.6}|topology_words={}|topology_metadata_bytes={}|prepared_word_bytes={}",
        canonical.nnz(),
        topo_stats.nonempty_words,
        topo_stats.metadata_bytes,
        prepared.metadata_bytes(),
    );

    let checked_pivots = validate_exact(&canonical, &diag, &prepared)?;

    let csr_once = csr_symbolic(&canonical, &diag);
    let abtm_once = abtm_symbolic(&canonical, &diag, &prepared);

    if csr_once.lower_pivots != abtm_once.lower_pivots
        || csr_once.successful_updates != abtm_once.successful_updates
        || csr_once.checksum != abtm_once.checksum
        || checked_pivots != csr_once.lower_pivots
    {
        return Err("G4a aggregate symbolic validation failed".into());
    }

    let pruning_ratio = if csr_once.upper_candidates == 0 {
        0.0
    } else {
        1.0 - csr_once.successful_updates as f64 / csr_once.upper_candidates as f64
    };
    let word_steps_over_candidates = if csr_once.upper_candidates == 0 {
        0.0
    } else {
        abtm_once.word_merge_steps as f64 / csr_once.upper_candidates as f64
    };

    println!(
        "G4A_SYMBOLIC|lower_pivots={}|csr_upper_candidates={}|successful_updates={}|pruning_ratio={pruning_ratio:.9e}|abtm_word_seek_operations={}|abtm_word_merge_steps={}|abtm_intersection_words={}|word_steps_over_candidates={word_steps_over_candidates:.9e}|mismatched_pivots=0",
        csr_once.lower_pivots,
        csr_once.upper_candidates,
        csr_once.successful_updates,
        abtm_once.word_seek_operations,
        abtm_once.word_merge_steps,
        abtm_once.intersection_words,
    );

    let mut csr_samples = Vec::with_capacity(args.repeats);
    let mut abtm_samples = Vec::with_capacity(args.repeats);

    for _ in 0..args.repeats {
        let start = Instant::now();
        let stats = csr_symbolic(&canonical, &diag);
        csr_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box(stats.checksum);
        black_box(stats.successful_updates);

        let start = Instant::now();
        let stats = abtm_symbolic(&canonical, &diag, &prepared);
        abtm_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box(stats.checksum);
        black_box(stats.successful_updates);
    }

    let csr_ms = median(&mut csr_samples);
    let abtm_ms = median(&mut abtm_samples);
    let abtm_over_csr = if csr_ms == 0.0 { 0.0 } else { abtm_ms / csr_ms };

    println!(
        "G4A_TIMING|csr_symbolic_ms={csr_ms:.6}|abtm_symbolic_ms={abtm_ms:.6}|abtm_over_csr_symbolic={abtm_over_csr:.9e}"
    );

    Ok(())
}
