//! G8-C10: read-only, bounded full-reorthogonalized Lanczos spectrum probe.
//!
//! A negative Ritz value is evidence of a negative Rayleigh direction; a
//! nonnegative minimum Ritz value NEVER certifies global SPD.  Every result
//! is a finite-step diagnostic, not a complete inertia calculation.
use hybit_matrix::{read_matrix_market, Csr32Matrix};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
struct Header {
    nrows: usize,
    ncols: usize,
    entries: usize,
    expansion: usize,
    supported: bool,
}

fn sniff(path: &Path) -> Result<Header, Box<dyn Error>> {
    let mut lines = BufReader::new(File::open(path)?).lines();
    let header = lines.next().ok_or("empty file")??;
    let words: Vec<_> = header.split_whitespace().collect();
    if words.len() != 5
        || !words[0].eq_ignore_ascii_case("%%MatrixMarket")
        || !words[1].eq_ignore_ascii_case("matrix")
    {
        return Err("unsupported Matrix Market banner".into());
    }
    let supported = words[2].eq_ignore_ascii_case("coordinate")
        && (words[3].eq_ignore_ascii_case("real") || words[3].eq_ignore_ascii_case("integer"))
        && (words[4].eq_ignore_ascii_case("general") || words[4].eq_ignore_ascii_case("symmetric"));
    let expansion = if words[4].eq_ignore_ascii_case("symmetric") {
        2
    } else {
        1
    };
    if !supported {
        return Ok(Header {
            nrows: 0,
            ncols: 0,
            entries: 0,
            expansion,
            supported,
        });
    }
    for line in lines {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }
        let dims: Vec<_> = trimmed.split_whitespace().collect();
        if dims.len() != 3 {
            return Err("invalid size record".into());
        }
        return Ok(Header {
            nrows: dims[0].parse()?,
            ncols: dims[1].parse()?,
            entries: dims[2].parse()?,
            expansion,
            supported,
        });
    }
    Err("missing Matrix Market size record".into())
}

fn exact_symmetric(a: &Csr32Matrix) -> bool {
    if a.nrows() != a.ncols() {
        return false;
    }
    for row in 0..a.nrows() {
        let start = a.row_ptr()[row] as usize;
        let end = a.row_ptr()[row + 1] as usize;
        for p in start..end {
            let col = a.col_idx()[p] as usize;
            let ts = a.row_ptr()[col] as usize;
            let te = a.row_ptr()[col + 1] as usize;
            let Ok(k) = a.col_idx()[ts..te].binary_search(&(row as u32)) else {
                return false;
            };
            if a.values()[p] != a.values()[ts + k] {
                return false;
            }
        }
    }
    true
}

fn spmv(a: &Csr32Matrix, x: &[f64], output: &mut [f64]) {
    for (i, yi) in output.iter_mut().enumerate() {
        let start = a.row_ptr()[i] as usize;
        let end = a.row_ptr()[i + 1] as usize;
        let mut value = 0.0;
        for p in start..end {
            value += a.values()[p] * x[a.col_idx()[p] as usize];
        }
        *yi = value;
    }
}
fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(&a, &b)| a * b).sum()
}
fn norm(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}

fn deterministic_seed(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        v.push(((z >> 11) as f64) * (2.0 / (1u64 << 53) as f64) - 1.0);
    }
    v
}

#[derive(Clone, Copy, Debug)]
struct Spectrum {
    steps: usize,
    min_ritz: f64,
    max_ritz: f64,
    terminal_beta: f64,
}

fn symmetric_eigen_extrema(diagonal: &[f64], off_diagonal: &[f64]) -> (f64, f64) {
    let m = diagonal.len();
    let mut matrix = vec![0.0; m * m];
    for (i, &value) in diagonal.iter().enumerate() {
        matrix[i * m + i] = value;
    }
    for (i, &value) in off_diagonal.iter().enumerate() {
        matrix[i * m + i + 1] = value;
        matrix[(i + 1) * m + i] = value;
    }
    let scale = matrix.iter().fold(1.0f64, |a, &b| a.max(b.abs()));
    for _sweep in 0..30 {
        let mut changed = false;
        for p in 0..m {
            for q in p + 1..m {
                let off = matrix[p * m + q];
                if off.abs() <= 1e-14 * scale {
                    continue;
                }
                changed = true;
                let tau = (matrix[q * m + q] - matrix[p * m + p]) / (2.0 * off);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let pp = matrix[p * m + p];
                let qq = matrix[q * m + q];
                matrix[p * m + p] = pp - t * off;
                matrix[q * m + q] = qq + t * off;
                matrix[p * m + q] = 0.0;
                matrix[q * m + p] = 0.0;
                for k in 0..m {
                    if k == p || k == q {
                        continue;
                    }
                    let kp = matrix[k * m + p];
                    let kq = matrix[k * m + q];
                    matrix[k * m + p] = c * kp - s * kq;
                    matrix[p * m + k] = matrix[k * m + p];
                    matrix[k * m + q] = s * kp + c * kq;
                    matrix[q * m + k] = matrix[k * m + q];
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut smallest = f64::INFINITY;
    let mut largest = f64::NEG_INFINITY;
    for i in 0..m {
        smallest = smallest.min(matrix[i * m + i]);
        largest = largest.max(matrix[i * m + i]);
    }
    (smallest, largest)
}

/// Two-pass full reorthogonalization bounds loss of orthogonality.  The
/// eigenvalues of the projected tridiagonal are not spectral certificates.
fn lanczos(a: &Csr32Matrix, max_steps: usize, seed: u64) -> Result<Spectrum, &'static str> {
    let n = a.nrows();
    if n == 0 || n != a.ncols() || max_steps == 0 {
        return Err("invalid dimensions/steps");
    }
    let mut q = deterministic_seed(n, seed);
    let initial = norm(&q);
    if initial == 0.0 || !initial.is_finite() {
        return Err("invalid starting vector");
    }
    for qi in &mut q {
        *qi /= initial;
    }
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(max_steps.min(n));
    let mut diagonal: Vec<f64> = Vec::new();
    let mut off_diagonal: Vec<f64> = Vec::new();
    let mut previous_beta = 0.0;
    let mut terminal_beta = 0.0;
    for _ in 0..max_steps.min(n) {
        let mut w = vec![0.0; n];
        spmv(a, &q, &mut w);
        let alpha = dot(&q, &w);
        if !alpha.is_finite() {
            return Err("nonfinite Lanczos alpha");
        }
        if let Some(previous) = basis.last() {
            for (wi, &pi) in w.iter_mut().zip(previous) {
                *wi -= previous_beta * pi;
            }
        }
        for (wi, &qi) in w.iter_mut().zip(&q) {
            *wi -= alpha * qi;
        }
        // Orthogonalize against all previously accepted basis vectors plus q.
        for _pass in 0..2 {
            for vector in basis.iter().chain(std::iter::once(&q)) {
                let projection = dot(vector, &w);
                for (wi, &vi) in w.iter_mut().zip(vector) {
                    *wi -= projection * vi;
                }
            }
        }
        let beta = norm(&w);
        if !beta.is_finite() {
            return Err("nonfinite Lanczos beta");
        }
        diagonal.push(alpha);
        terminal_beta = beta;
        if beta <= 1e-13 * (1.0 + alpha.abs()) || diagonal.len() == max_steps.min(n) {
            break;
        }
        off_diagonal.push(beta);
        basis.push(q);
        q = w.into_iter().map(|v| v / beta).collect();
        previous_beta = beta;
    }
    let (min_ritz, max_ritz) = symmetric_eigen_extrema(&diagonal, &off_diagonal);
    Ok(Spectrum {
        steps: diagonal.len(),
        min_ritz,
        max_ritz,
        terminal_beta,
    })
}

fn csv(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 4 {
        return Err("usage: mtx_fem_spectral <output.csv> <max_expanded_nnz> <steps:2..64> <matrix.mtx> ...".into());
    }
    let destination = Path::new(&args[0]);
    let max_nnz: usize = args[1].parse()?;
    let steps: usize = args[2].parse()?;
    if !(2..=64).contains(&steps) || !(1..=100_000_000).contains(&max_nnz) {
        return Err("max_nnz outside 1..=100000000 or steps outside 2..=64".into());
    }
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)?;
    writeln!(output, "file,status,reason,n,nnz,lanczos_steps,min_ritz,max_ritz,terminal_beta,negative_threshold,elapsed_ms")?;
    for arg in args.iter().skip(3) {
        let path = Path::new(arg);
        let timer = Instant::now();
        let mut status = String::from("SKIP");
        let reason: String;
        let mut n = 0usize;
        let mut nnz = 0usize;
        let mut iterations = 0usize;
        let mut min_ritz = f64::NAN;
        let mut max_ritz = f64::NAN;
        let mut beta = f64::NAN;
        let mut threshold = f64::NAN;
        match sniff(path) {
            Err(error) => {
                reason = error.to_string();
            }
            Ok(header) if !header.supported => {
                status = "SKIP_UNSUPPORTED".into();
                reason = "expected coordinate real/integer general/symmetric".into();
            }
            Ok(header) if header.nrows == 0 || header.nrows != header.ncols => {
                status = "SKIP_NON_SQUARE".into();
                reason = "spectral symmetric probe requires square input".into();
            }
            Ok(header)
                if header.nrows > 5_000_000
                    || header.entries.saturating_mul(header.expansion) > max_nnz =>
            {
                status = "SKIP_LIMIT".into();
                reason = "dimension or expanded-nnz safety cap exceeded".into();
            }
            Ok(_) => {
                match read_matrix_market(path) {
                    Err(error) => {
                        status = "SKIP_PARSE_FAILURE".into();
                        reason = error.to_string();
                    }
                    Ok((a, _)) => {
                        n = a.nrows();
                        nnz = a.nnz();
                        if !exact_symmetric(&a) {
                            status = "SKIP_NOT_EXACT_SYMMETRIC".into();
                            reason = "no valid symmetric Lanczos projection".into();
                        } else {
                            match lanczos(&a, steps, 0x38d0_2027_u64) {
                                Err(msg) => {
                                    status = "PROBE_FAILED".into();
                                    reason = msg.into();
                                }
                                Ok(spectrum) => {
                                    iterations = spectrum.steps;
                                    min_ritz = spectrum.min_ritz;
                                    max_ritz = spectrum.max_ritz;
                                    beta = spectrum.terminal_beta;
                                    threshold = 1e-10 * min_ritz.abs().max(max_ritz.abs()).max(1.0);
                                    if min_ritz < -threshold {
                                        status = "NEGATIVE_RITZ_EVIDENCE".into();
                                        reason = "negative Ritz value observed; verify with independent factorization".into();
                                    } else {
                                        status = "SPD_UNVERIFIED_BY_LANCZOS".into();
                                        reason = "finite-step nonnegative Ritz minimum does not prove SPD".into();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let elapsed = timer.elapsed().as_secs_f64() * 1000.0;
        writeln!(output, "{},{},{},{n},{nnz},{iterations},{min_ritz:.12e},{max_ritz:.12e},{beta:.12e},{threshold:.12e},{elapsed:.4}",
            csv(arg), csv(&status), csv(&reason))?;
        output.flush()?;
        println!("G8-C10 SPECTRAL {} {status}: n={n} nnz={nnz} steps={iterations} min_ritz={min_ritz:.6e} max_ritz={max_ritz:.6e}", path.display());
    }
    println!("=== G8-C10 SPECTRAL DIAGNOSTIC COMPLETE: NO SPD CERTIFICATION ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn csr(n: usize, row_ptr: Vec<u32>, cols: Vec<u32>, values: Vec<f64>) -> Csr32Matrix {
        Csr32Matrix::new(n, n, row_ptr, cols, values).unwrap()
    }
    #[test]
    fn diagonal_spd_projection_stays_positive() {
        let a = csr(2, vec![0, 1, 2], vec![0, 1], vec![2.0, 4.0]);
        let result = lanczos(&a, 8, 17).unwrap();
        assert!(result.min_ritz > 1.99);
        assert!(result.max_ritz < 4.01);
    }
    #[test]
    fn indefinite_symmetric_matrix_has_negative_ritz() {
        let a = csr(2, vec![0, 2, 4], vec![0, 1, 0, 1], vec![1.0, 2.0, 2.0, 1.0]);
        let result = lanczos(&a, 8, 17).unwrap();
        assert!(result.min_ritz < -0.99);
        assert!(result.max_ritz > 2.99);
    }
    #[test]
    fn exact_symmetry_detects_missing_transpose() {
        let a = csr(2, vec![0, 2, 3], vec![0, 1, 1], vec![2.0, -1.0, 2.0]);
        assert!(!exact_symmetric(&a));
    }
    #[test]
    fn smallest_nonnegative_ritz_does_not_prove_spd() {
        let a = csr(
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, -1.0, -1.0, 1.0],
        );
        let result = lanczos(&a, 8, 17).unwrap();
        assert!(result.min_ritz.abs() < 1e-8);
    }
}
