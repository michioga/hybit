use std::env;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use hybit_matrix::{read_matrix_market, AbtmTopology, Csr32Matrix};

const ILU0_RELATIVE_PIVOT_FLOOR: f64 = 1.0e-12;

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    repeats: usize,
    thresholds: Vec<usize>,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut repeats = 5usize;
        let mut thresholds = vec![1usize, 2, 4, 6, 8, 12, 16, 24, 32];
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
                "--thresholds" => {
                    let raw = it.next().ok_or("missing value after --thresholds")?;
                    thresholds = raw
                        .split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::parse::<usize>)
                        .collect::<Result<Vec<_>, _>>()?;
                    if thresholds.is_empty() || thresholds.iter().any(|&v| !(1..=64).contains(&v)) {
                        return Err(
                            "--thresholds must contain comma-separated values in 1..=64".into()
                        );
                    }
                    thresholds.sort_unstable();
                    thresholds.dedup();
                }
                "-h" | "--help" => {
                    println!(
                        "Usage: abtm_ilu0_adaptive_rank_g4d --matrix A.mtx [--repeats N] [--thresholds 1,2,4,6,8,12,16,24,32]"
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
            repeats,
            thresholds,
        })
    }
}

#[derive(Clone, Debug)]
struct PreparedWordRows {
    nrows: usize,
    row_ptr: Vec<u32>,
    word_indices: Vec<u32>,
    masks: Vec<u64>,
    nnz_prefix: Vec<u32>,
}

impl PreparedWordRows {
    fn from_topology(topology: &AbtmTopology) -> Result<Self, Box<dyn Error>> {
        let mut row_ptr = Vec::with_capacity(topology.nrows() + 1);
        let mut word_indices = Vec::with_capacity(topology.nonempty_words());
        let mut masks = Vec::with_capacity(topology.nonempty_words());
        let mut nnz_prefix = Vec::with_capacity(topology.nonempty_words());
        row_ptr.push(0);

        for row in 0..topology.nrows() {
            let mut prefix = 0usize;
            for word in topology.row(row)?.words() {
                word_indices.push(word.word_index());
                masks.push(word.mask());
                nnz_prefix.push(u32::try_from(prefix)?);
                prefix = prefix
                    .checked_add(word.popcount())
                    .ok_or("row nnz prefix overflow")?;
            }
            row_ptr.push(u32::try_from(word_indices.len())?);
        }

        Ok(Self {
            nrows: topology.nrows(),
            row_ptr,
            word_indices,
            masks,
            nnz_prefix,
        })
    }

    fn row(&self, row: usize) -> (&[u32], &[u64], &[u32]) {
        debug_assert!(row < self.nrows);
        let start = self.row_ptr[row] as usize;
        let end = self.row_ptr[row + 1] as usize;
        (
            &self.word_indices[start..end],
            &self.masks[start..end],
            &self.nnz_prefix[start..end],
        )
    }

    fn metadata_bytes(&self) -> usize {
        self.row_ptr.len() * std::mem::size_of::<u32>()
            + self.word_indices.len() * std::mem::size_of::<u32>()
            + self.masks.len() * std::mem::size_of::<u64>()
            + self.nnz_prefix.len() * std::mem::size_of::<u32>()
    }
}

#[derive(Clone, Debug)]
struct AdaptiveRankLut {
    threshold: usize,
    lut_index: Vec<u32>,
    ranks: Vec<[u8; 64]>,
}

impl AdaptiveRankLut {
    fn from_prepared(
        prepared: &PreparedWordRows,
        threshold: usize,
    ) -> Result<Self, Box<dyn Error>> {
        if !(1..=64).contains(&threshold) {
            return Err("adaptive rank threshold must be in 1..=64".into());
        }

        let mut lut_index = Vec::with_capacity(prepared.masks.len());
        let mut ranks = Vec::new();

        for &mask in &prepared.masks {
            if mask.count_ones() as usize >= threshold {
                let index = u32::try_from(ranks.len())?;
                let mut table = [u8::MAX; 64];
                let mut ordinal = 0u8;
                let mut bits = mask;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    table[bit] = ordinal;
                    ordinal = ordinal.checked_add(1).ok_or("rank LUT ordinal overflow")?;
                    bits &= bits - 1;
                }
                ranks.push(table);
                lut_index.push(index);
            } else {
                lut_index.push(u32::MAX);
            }
        }

        Ok(Self {
            threshold,
            lut_index,
            ranks,
        })
    }

    #[inline]
    fn table(&self, word_slot: usize) -> Option<&[u8; 64]> {
        let index = self.lut_index[word_slot];
        if index == u32::MAX {
            None
        } else {
            Some(&self.ranks[index as usize])
        }
    }

    fn threshold(&self) -> usize {
        self.threshold
    }

    fn selected_words(&self) -> usize {
        self.ranks.len()
    }

    fn bytes(&self) -> usize {
        self.lut_index.len() * std::mem::size_of::<u32>()
            + self.ranks.len() * std::mem::size_of::<[u8; 64]>()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct NumericStats {
    lower_pivots: usize,
    candidate_updates: usize,
    executed_updates: usize,
    word_merge_steps: usize,
    lut_rank_uses: usize,
    popcount_rank_uses: usize,
    adjusted_pivots: usize,
}

#[derive(Clone, Debug)]
struct FactorResult {
    lu: Vec<f64>,
    stats: NumericStats,
}

fn canonicalize_like_ilu0(matrix: &Csr32Matrix) -> Result<Csr32Matrix, Box<dyn Error>> {
    if matrix.nrows() != matrix.ncols() {
        return Err("ILU(0) factorization requires a square matrix".into());
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

fn row_scales(matrix: &Csr32Matrix) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut scales = Vec::with_capacity(matrix.nrows());

    for row in 0..matrix.nrows() {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        let scale = matrix.values()[start..end]
            .iter()
            .fold(0.0f64, |acc, &value| acc.max(value.abs()));

        if !scale.is_finite() || scale == 0.0 {
            return Err(format!("ILU(0) row {row} scale is zero or non-finite").into());
        }
        scales.push(scale);
    }

    Ok(scales)
}

#[inline]
fn rank_in_word(mask: u64, bit: usize) -> usize {
    if bit == 0 {
        0
    } else {
        (mask & ((1u64 << bit) - 1)).count_ones() as usize
    }
}

#[inline]
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

fn lower_bound(words: &[u32], target: u32) -> usize {
    match words.binary_search(&target) {
        Ok(index) | Err(index) => index,
    }
}

fn factor_csr(
    matrix: &Csr32Matrix,
    diag: &[usize],
    scales: &[f64],
) -> Result<FactorResult, Box<dyn Error>> {
    let mut lu = matrix.values().to_vec();
    let mut stats = NumericStats::default();

    for (row, (&diag_pos, &scale)) in diag.iter().zip(scales).enumerate() {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;

        for p in start..diag_pos {
            stats.lower_pivots = stats.lower_pivots.saturating_add(1);
            let lower_col = matrix.col_idx()[p] as usize;
            let lower_diag = diag[lower_col];
            let pivot = lu[lower_diag];

            if !pivot.is_finite() || pivot == 0.0 {
                return Err("CSR ILU(0) previous pivot became zero or non-finite".into());
            }

            let multiplier = lu[p] / pivot;
            if !multiplier.is_finite() {
                return Err("CSR ILU(0) multiplier became non-finite".into());
            }
            lu[p] = multiplier;

            let lower_end = matrix.row_ptr()[lower_col + 1] as usize;
            for q in (lower_diag + 1)..lower_end {
                stats.candidate_updates = stats.candidate_updates.saturating_add(1);
                let target_col = matrix.col_idx()[q];

                if let Ok(offset) = matrix.col_idx()[(p + 1)..end].binary_search(&target_col) {
                    let target = p + 1 + offset;
                    lu[target] -= multiplier * lu[q];
                    if !lu[target].is_finite() {
                        return Err("CSR ILU(0) numeric update became non-finite".into());
                    }
                    stats.executed_updates = stats.executed_updates.saturating_add(1);
                }
            }
        }

        let raw_pivot = lu[diag_pos];
        if !raw_pivot.is_finite() {
            return Err("CSR ILU(0) pivot became non-finite".into());
        }

        let floor = (ILU0_RELATIVE_PIVOT_FLOOR * scale).max(f64::MIN_POSITIVE);
        if raw_pivot.abs() <= floor {
            lu[diag_pos] = if raw_pivot.is_sign_negative() {
                -floor
            } else {
                floor
            };
            stats.adjusted_pivots = stats.adjusted_pivots.saturating_add(1);
        }
    }

    Ok(FactorResult { lu, stats })
}

fn apply_intersection_popcount(
    matrix: &Csr32Matrix,
    prepared: &PreparedWordRows,
    rows: (usize, usize),
    multiplier: f64,
    lu: &mut [f64],
    stats: &mut NumericStats,
) -> Result<(), Box<dyn Error>> {
    let (row, pivot_row) = rows;
    let (row_words, row_masks, row_prefix) = prepared.row(row);
    let (pivot_words, pivot_masks, pivot_prefix) = prepared.row(pivot_row);

    let pivot_word = (pivot_row / 64) as u32;
    let mut a = lower_bound(row_words, pivot_word);
    let mut b = lower_bound(pivot_words, pivot_word);

    while a < row_words.len() && b < pivot_words.len() {
        stats.word_merge_steps = stats.word_merge_steps.saturating_add(1);

        let aw = row_words[a];
        let bw = pivot_words[b];

        if aw < bw {
            a += 1;
            continue;
        }
        if bw < aw {
            b += 1;
            continue;
        }

        let mut bits = row_masks[a] & pivot_masks[b];
        bits = mask_strictly_above(bits, aw, pivot_row);

        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            let row_ordinal = row_prefix[a] as usize + rank_in_word(row_masks[a], bit);
            let pivot_ordinal = pivot_prefix[b] as usize + rank_in_word(pivot_masks[b], bit);

            let target = matrix.row_ptr()[row] as usize + row_ordinal;
            let source = matrix.row_ptr()[pivot_row] as usize + pivot_ordinal;

            lu[target] -= multiplier * lu[source];
            if !lu[target].is_finite() {
                return Err("ABTM popcount ILU(0) numeric update became non-finite".into());
            }

            stats.executed_updates = stats.executed_updates.saturating_add(1);
            stats.popcount_rank_uses = stats.popcount_rank_uses.saturating_add(2);
            bits &= bits - 1;
        }

        a += 1;
        b += 1;
    }

    Ok(())
}

struct AdaptiveContext<'a> {
    matrix: &'a Csr32Matrix,
    prepared: &'a PreparedWordRows,
    lut: &'a AdaptiveRankLut,
}

fn apply_intersection_adaptive(
    ctx: &AdaptiveContext<'_>,
    rows: (usize, usize),
    multiplier: f64,
    lu: &mut [f64],
    stats: &mut NumericStats,
) -> Result<(), Box<dyn Error>> {
    let (row, pivot_row) = rows;
    let row_base = ctx.prepared.row_ptr[row] as usize;
    let row_end = ctx.prepared.row_ptr[row + 1] as usize;
    let pivot_base = ctx.prepared.row_ptr[pivot_row] as usize;
    let pivot_end = ctx.prepared.row_ptr[pivot_row + 1] as usize;

    let row_words = &ctx.prepared.word_indices[row_base..row_end];
    let row_masks = &ctx.prepared.masks[row_base..row_end];
    let row_prefix = &ctx.prepared.nnz_prefix[row_base..row_end];

    let pivot_words = &ctx.prepared.word_indices[pivot_base..pivot_end];
    let pivot_masks = &ctx.prepared.masks[pivot_base..pivot_end];
    let pivot_prefix = &ctx.prepared.nnz_prefix[pivot_base..pivot_end];

    let pivot_word = (pivot_row / 64) as u32;
    let mut a = lower_bound(row_words, pivot_word);
    let mut b = lower_bound(pivot_words, pivot_word);

    while a < row_words.len() && b < pivot_words.len() {
        stats.word_merge_steps = stats.word_merge_steps.saturating_add(1);

        let aw = row_words[a];
        let bw = pivot_words[b];

        if aw < bw {
            a += 1;
            continue;
        }
        if bw < aw {
            b += 1;
            continue;
        }

        let mut bits = row_masks[a] & pivot_masks[b];
        bits = mask_strictly_above(bits, aw, pivot_row);

        let row_slot = row_base + a;
        let pivot_slot = pivot_base + b;
        let row_table = ctx.lut.table(row_slot);
        let pivot_table = ctx.lut.table(pivot_slot);

        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;

            let row_rank = if let Some(table) = row_table {
                stats.lut_rank_uses = stats.lut_rank_uses.saturating_add(1);
                let rank = table[bit];
                debug_assert_ne!(rank, u8::MAX);
                rank as usize
            } else {
                stats.popcount_rank_uses = stats.popcount_rank_uses.saturating_add(1);
                rank_in_word(row_masks[a], bit)
            };

            let pivot_rank = if let Some(table) = pivot_table {
                stats.lut_rank_uses = stats.lut_rank_uses.saturating_add(1);
                let rank = table[bit];
                debug_assert_ne!(rank, u8::MAX);
                rank as usize
            } else {
                stats.popcount_rank_uses = stats.popcount_rank_uses.saturating_add(1);
                rank_in_word(pivot_masks[b], bit)
            };

            let target = ctx.matrix.row_ptr()[row] as usize + row_prefix[a] as usize + row_rank;
            let source =
                ctx.matrix.row_ptr()[pivot_row] as usize + pivot_prefix[b] as usize + pivot_rank;

            lu[target] -= multiplier * lu[source];
            if !lu[target].is_finite() {
                return Err("ABTM adaptive ILU(0) numeric update became non-finite".into());
            }

            stats.executed_updates = stats.executed_updates.saturating_add(1);
            bits &= bits - 1;
        }

        a += 1;
        b += 1;
    }

    Ok(())
}

fn factor_abtm_popcount(
    matrix: &Csr32Matrix,
    diag: &[usize],
    scales: &[f64],
    prepared: &PreparedWordRows,
) -> Result<FactorResult, Box<dyn Error>> {
    let mut lu = matrix.values().to_vec();
    let mut stats = NumericStats::default();

    for (row, (&diag_pos, &scale)) in diag.iter().zip(scales).enumerate() {
        let start = matrix.row_ptr()[row] as usize;

        for p in start..diag_pos {
            stats.lower_pivots = stats.lower_pivots.saturating_add(1);
            let pivot_row = matrix.col_idx()[p] as usize;
            let pivot = lu[diag[pivot_row]];

            if !pivot.is_finite() || pivot == 0.0 {
                return Err("ABTM popcount previous pivot became zero or non-finite".into());
            }

            let multiplier = lu[p] / pivot;
            if !multiplier.is_finite() {
                return Err("ABTM popcount multiplier became non-finite".into());
            }
            lu[p] = multiplier;

            apply_intersection_popcount(
                matrix,
                prepared,
                (row, pivot_row),
                multiplier,
                &mut lu,
                &mut stats,
            )?;
        }

        let raw_pivot = lu[diag_pos];
        if !raw_pivot.is_finite() {
            return Err("ABTM popcount pivot became non-finite".into());
        }

        let floor = (ILU0_RELATIVE_PIVOT_FLOOR * scale).max(f64::MIN_POSITIVE);
        if raw_pivot.abs() <= floor {
            lu[diag_pos] = if raw_pivot.is_sign_negative() {
                -floor
            } else {
                floor
            };
            stats.adjusted_pivots = stats.adjusted_pivots.saturating_add(1);
        }
    }

    Ok(FactorResult { lu, stats })
}

fn factor_abtm_adaptive(
    matrix: &Csr32Matrix,
    diag: &[usize],
    scales: &[f64],
    prepared: &PreparedWordRows,
    lut: &AdaptiveRankLut,
) -> Result<FactorResult, Box<dyn Error>> {
    let mut lu = matrix.values().to_vec();
    let mut stats = NumericStats::default();
    let ctx = AdaptiveContext {
        matrix,
        prepared,
        lut,
    };

    for (row, (&diag_pos, &scale)) in diag.iter().zip(scales).enumerate() {
        let start = matrix.row_ptr()[row] as usize;

        for p in start..diag_pos {
            stats.lower_pivots = stats.lower_pivots.saturating_add(1);
            let pivot_row = matrix.col_idx()[p] as usize;
            let pivot = lu[diag[pivot_row]];

            if !pivot.is_finite() || pivot == 0.0 {
                return Err("ABTM adaptive previous pivot became zero or non-finite".into());
            }

            let multiplier = lu[p] / pivot;
            if !multiplier.is_finite() {
                return Err("ABTM adaptive multiplier became non-finite".into());
            }
            lu[p] = multiplier;

            apply_intersection_adaptive(&ctx, (row, pivot_row), multiplier, &mut lu, &mut stats)?;
        }

        let raw_pivot = lu[diag_pos];
        if !raw_pivot.is_finite() {
            return Err("ABTM adaptive pivot became non-finite".into());
        }

        let floor = (ILU0_RELATIVE_PIVOT_FLOOR * scale).max(f64::MIN_POSITIVE);
        if raw_pivot.abs() <= floor {
            lu[diag_pos] = if raw_pivot.is_sign_negative() {
                -floor
            } else {
                floor
            };
            stats.adjusted_pivots = stats.adjusted_pivots.saturating_add(1);
        }
    }

    Ok(FactorResult { lu, stats })
}

fn apply_factor(
    matrix: &Csr32Matrix,
    diag: &[usize],
    lu: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>, Box<dyn Error>> {
    let n = matrix.nrows();
    if rhs.len() != n || lu.len() != matrix.nnz() {
        return Err("ILU(0) apply dimension mismatch".into());
    }

    let mut z = rhs.to_vec();

    for (row, &diag_pos) in diag.iter().enumerate() {
        let start = matrix.row_ptr()[row] as usize;
        let mut sum = z[row];
        for p in start..diag_pos {
            sum -= lu[p] * z[matrix.col_idx()[p] as usize];
        }
        z[row] = sum;
    }

    for (row, &diag_pos) in diag.iter().enumerate().rev() {
        let end = matrix.row_ptr()[row + 1] as usize;
        let mut sum = z[row];
        for p in (diag_pos + 1)..end {
            sum -= lu[p] * z[matrix.col_idx()[p] as usize];
        }

        let pivot = lu[diag_pos];
        if !pivot.is_finite() || pivot == 0.0 {
            return Err("ILU(0) apply encountered invalid pivot".into());
        }

        z[row] = sum / pivot;
        if !z[row].is_finite() {
            return Err("ILU(0) apply became non-finite".into());
        }
    }

    Ok(z)
}

fn deterministic_rhs(n: usize) -> Vec<f64> {
    (0..n)
        .map(|index| {
            let x = index as f64 + 1.0;
            (x * 0.003_906_25).sin() + (x * 0.001_953_125).cos() * 0.25
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

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;

    println!(
        "HyBIT {} ABTM G4d adaptive rank-LUT sweep",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!("repeats             : {}", args.repeats);
    println!("thresholds          : {:?}", args.thresholds);

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
    let scales = row_scales(&canonical)?;

    let start = Instant::now();
    let topology = AbtmTopology::from_csr32(&canonical)?;
    let topology_prepare_ms = start.elapsed().as_secs_f64() * 1.0e3;

    let start = Instant::now();
    let prepared = PreparedWordRows::from_topology(&topology)?;
    let word_prepare_ms = start.elapsed().as_secs_f64() * 1.0e3;

    let topo_stats = topology.stats();
    let average_word_nnz = topo_stats.average_word_nnz();

    println!(
        "G4D_PREPARE|canonical_nnz={}|canonicalize_ms={canonicalize_ms:.6}|topology_prepare_ms={topology_prepare_ms:.6}|word_prepare_ms={word_prepare_ms:.6}|topology_words={}|average_word_nnz={average_word_nnz:.9e}|topology_metadata_bytes={}|prepared_word_bytes={}",
        canonical.nnz(),
        topo_stats.nonempty_words,
        topo_stats.metadata_bytes,
        prepared.metadata_bytes(),
    );

    let csr_once = factor_csr(&canonical, &diag, &scales)?;
    let popcount_once = factor_abtm_popcount(&canonical, &diag, &scales, &prepared)?;

    let popcount_factor_error = max_scaled_error(&csr_once.lu, &popcount_once.lu);
    let rhs = deterministic_rhs(canonical.nrows());
    let csr_apply = apply_factor(&canonical, &diag, &csr_once.lu, &rhs)?;
    let popcount_apply = apply_factor(&canonical, &diag, &popcount_once.lu, &rhs)?;
    let popcount_apply_error = max_scaled_error(&csr_apply, &popcount_apply);

    const VALIDATION_TOLERANCE: f64 = 1.0e-12;
    if popcount_factor_error > VALIDATION_TOLERANCE || popcount_apply_error > VALIDATION_TOLERANCE {
        return Err("G4d popcount baseline mismatch".into());
    }

    let mut csr_samples = Vec::with_capacity(args.repeats);
    let mut popcount_samples = Vec::with_capacity(args.repeats);

    for _ in 0..args.repeats {
        let start = Instant::now();
        let factor = factor_csr(&canonical, &diag, &scales)?;
        csr_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box(factor.lu.last().copied());

        let start = Instant::now();
        let factor = factor_abtm_popcount(&canonical, &diag, &scales, &prepared)?;
        popcount_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        black_box(factor.lu.last().copied());
    }

    let csr_ms = median(&mut csr_samples);
    let popcount_ms = median(&mut popcount_samples);
    let popcount_over_csr = if csr_ms == 0.0 {
        0.0
    } else {
        popcount_ms / csr_ms
    };

    println!(
        "G4D_BASE|csr_numeric_ms={csr_ms:.6}|popcount_numeric_ms={popcount_ms:.6}|popcount_over_csr={popcount_over_csr:.9e}|executed_updates={}|popcount_rank_uses={}",
        popcount_once.stats.executed_updates,
        popcount_once.stats.popcount_rank_uses,
    );

    for &threshold in &args.thresholds {
        let start = Instant::now();
        let lut = AdaptiveRankLut::from_prepared(&prepared, threshold)?;
        let lut_prepare_ms = start.elapsed().as_secs_f64() * 1.0e3;

        let factor = factor_abtm_adaptive(&canonical, &diag, &scales, &prepared, &lut)?;

        if factor.stats.lower_pivots != csr_once.stats.lower_pivots
            || factor.stats.executed_updates != csr_once.stats.executed_updates
            || factor.stats.adjusted_pivots != csr_once.stats.adjusted_pivots
        {
            return Err(format!(
                "G4d aggregate numeric statistics differ at threshold {threshold}"
            )
            .into());
        }

        let bit_mismatches = csr_once
            .lu
            .iter()
            .zip(&factor.lu)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let factor_error = max_scaled_error(&csr_once.lu, &factor.lu);
        let adaptive_apply = apply_factor(&canonical, &diag, &factor.lu, &rhs)?;
        let apply_error = max_scaled_error(&csr_apply, &adaptive_apply);

        if factor_error > VALIDATION_TOLERANCE || apply_error > VALIDATION_TOLERANCE {
            return Err(format!(
                "G4d numerical mismatch at threshold {threshold}: factor={factor_error:.3e}, apply={apply_error:.3e}"
            )
            .into());
        }

        let mut samples = Vec::with_capacity(args.repeats);
        for _ in 0..args.repeats {
            let start = Instant::now();
            let timed = factor_abtm_adaptive(&canonical, &diag, &scales, &prepared, &lut)?;
            samples.push(start.elapsed().as_secs_f64() * 1.0e3);
            black_box(timed.lu.last().copied());
        }
        let adaptive_ms = median(&mut samples);

        let over_csr = if csr_ms == 0.0 {
            0.0
        } else {
            adaptive_ms / csr_ms
        };
        let over_popcount = if popcount_ms == 0.0 {
            0.0
        } else {
            adaptive_ms / popcount_ms
        };

        let selected_fraction = if topo_stats.nonempty_words == 0 {
            0.0
        } else {
            lut.selected_words() as f64 / topo_stats.nonempty_words as f64
        };

        let rank_total = factor
            .stats
            .lut_rank_uses
            .saturating_add(factor.stats.popcount_rank_uses);
        let lut_rank_fraction = if rank_total == 0 {
            0.0
        } else {
            factor.stats.lut_rank_uses as f64 / rank_total as f64
        };

        let saved_vs_popcount_ms = (popcount_ms - adaptive_ms).max(0.0);
        let break_even_factorizations = if saved_vs_popcount_ms == 0.0 {
            f64::INFINITY
        } else {
            lut_prepare_ms / saved_vs_popcount_ms
        };

        let total_prepared_bytes = prepared.metadata_bytes().saturating_add(lut.bytes());

        println!(
            "G4D_THRESHOLD|threshold={}|selected_words={}|selected_word_fraction={selected_fraction:.9e}|lut_prepare_ms={lut_prepare_ms:.6}|adaptive_lut_bytes={}|total_prepared_bytes={total_prepared_bytes}|lut_rank_uses={}|popcount_rank_uses={}|lut_rank_fraction={lut_rank_fraction:.9e}|factor_bit_mismatches={bit_mismatches}|factor_max_scaled_error={factor_error:.9e}|apply_max_scaled_error={apply_error:.9e}|numeric_ms={adaptive_ms:.6}|over_csr={over_csr:.9e}|over_popcount={over_popcount:.9e}|saved_vs_popcount_ms={saved_vs_popcount_ms:.6}|break_even_factorizations={break_even_factorizations:.9e}|mismatched=0",
            lut.threshold(),
            lut.selected_words(),
            lut.bytes(),
            factor.stats.lut_rank_uses,
            factor.stats.popcount_rank_uses,
        );
    }

    Ok(())
}
