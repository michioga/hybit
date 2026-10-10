//! G8-C9: serial, read-only Matrix Market diagnostic gate for candidate FEM matrices.
//!
//! NEVER treats positive diagonals, numerical symmetry or PCG convergence as
//! a general SPD proof. Certificates below rely on EXACT symmetry of the
//! assembled floating-point CSR and strictly/irreducibly diagonally dominant
//! sufficient conditions. No MPI, Cholesky, or Krylov solve occurs here.
use hybit_matrix::{read_matrix_market, Csr32Matrix};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;

#[derive(Clone, Debug)]
struct Header {
    field: String,
    symmetry: String,
    format: String,
    nrows: usize,
    ncols: usize,
    entries: usize,
}

fn read_header(path: &Path) -> Result<Header, Box<dyn Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut first = String::new();
    if reader.read_line(&mut first)? == 0 {
        return Err("empty Matrix Market file".into());
    }
    let tokens: Vec<_> = first.split_whitespace().collect();
    if tokens.len() != 5
        || !tokens[0].eq_ignore_ascii_case("%%MatrixMarket")
        || !tokens[1].eq_ignore_ascii_case("matrix")
    {
        return Err("invalid Matrix Market header".into());
    }
    let field = tokens[3].to_ascii_lowercase();
    let symmetry = tokens[4].to_ascii_lowercase();
    let format = tokens[2].to_ascii_lowercase();
    loop {
        let mut size = String::new();
        if reader.read_line(&mut size)? == 0 {
            return Err("missing Matrix Market size record".into());
        }
        let line = size.trim();
        if line.is_empty() || line.starts_with('%') {
            continue;
        }
        if format != "coordinate" {
            return Ok(Header {
                field,
                symmetry,
                format,
                nrows: 0,
                ncols: 0,
                entries: 0,
            });
        }
        let items: Vec<_> = line.split_whitespace().collect();
        if items.len() != 3 {
            return Err("invalid coordinate size record".into());
        }
        return Ok(Header {
            field,
            symmetry,
            format,
            nrows: items[0].parse()?,
            ncols: items[1].parse()?,
            entries: items[2].parse()?,
        });
    }
}

#[derive(Clone, Debug)]
struct Analysis {
    classification: &'static str,
    nnz: usize,
    strict_rows: usize,
    weak_rows: usize,
    positive_diagonal_rows: usize,
    missing_or_nonpositive_diagonal_rows: usize,
    missing_transpose: usize,
    asymmetric_exact: usize,
    asymmetric_tolerance: usize,
    connected_components: usize,
    max_row_nnz: usize,
    max_band_offset: usize,
    min_margin: f64,
    factor_candidate: &'static str,
}

#[derive(Clone, Debug)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}
impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }
    fn root(&mut self, x: usize) -> usize {
        let p = self.parent[x];
        if p != x {
            self.parent[x] = self.root(p);
        }
        self.parent[x]
    }
    fn unite(&mut self, x: usize, y: usize) {
        let mut x = self.root(x);
        let mut y = self.root(y);
        if x == y {
            return;
        }
        if self.rank[x] < self.rank[y] {
            std::mem::swap(&mut x, &mut y);
        }
        self.parent[y] = x;
        if self.rank[x] == self.rank[y] {
            self.rank[x] += 1;
        }
    }
}

fn analyze(a: &Csr32Matrix) -> Analysis {
    let n = a.nrows();
    let square = n == a.ncols();
    let mut uf = UnionFind::new(n);
    let mut strict = vec![false; n];
    let mut strict_rows = 0;
    let mut weak_rows = 0;
    let mut positive_diagonal_rows = 0;
    let mut max_row_nnz = 0;
    let mut max_band_offset = 0;
    let mut min_margin = f64::INFINITY;
    let mut missing_transpose = 0;
    let mut asymmetric_exact = 0;
    let mut asymmetric_tolerance = 0;

    for (row, strict_flag) in strict.iter_mut().enumerate() {
        let start = a.row_ptr()[row] as usize;
        let end = a.row_ptr()[row + 1] as usize;
        max_row_nnz = max_row_nnz.max(end - start);
        let mut diag = 0.0;
        let mut off = 0.0;
        for p in start..end {
            let col = a.col_idx()[p] as usize;
            let val = a.values()[p];
            max_band_offset = max_band_offset.max(row.abs_diff(col));
            if row == col {
                diag += val;
            } else {
                off += val.abs();
                if square {
                    uf.unite(row, col);
                    let ts = a.row_ptr()[col] as usize;
                    let te = a.row_ptr()[col + 1] as usize;
                    let twin = a.col_idx()[ts..te].binary_search(&(row as u32));
                    match twin {
                        Ok(q) => {
                            let transposed = a.values()[ts + q];
                            if val != transposed {
                                asymmetric_exact += 1;
                            }
                            if (val - transposed).abs()
                                > 1.0e-12 * (1.0 + val.abs().max(transposed.abs()))
                            {
                                asymmetric_tolerance += 1;
                            }
                        }
                        Err(_) => {
                            missing_transpose += 1;
                        }
                    }
                }
            }
        }
        if diag > 0.0 && diag.is_finite() {
            positive_diagonal_rows += 1;
        }
        let margin = diag - off;
        min_margin = min_margin.min(margin);
        if margin >= 0.0 {
            weak_rows += 1;
        }
        if margin > 0.0 {
            strict_rows += 1;
            *strict_flag = true;
        }
    }
    let mut root_has_strict = vec![false; n];
    let mut root_seen = vec![false; n];
    for (i, &s) in strict.iter().enumerate() {
        let root = uf.root(i);
        root_seen[root] = true;
        root_has_strict[root] |= s;
    }
    let connected_components = root_seen.iter().filter(|&&seen| seen).count();
    let each_component_strict = (0..n).filter(|&i| root_seen[i]).all(|i| root_has_strict[i]);
    let exactly_symmetric = square && missing_transpose == 0 && asymmetric_exact == 0;
    let approximately_symmetric = square && missing_transpose == 0 && asymmetric_tolerance == 0;
    let positive_diagonal = positive_diagonal_rows == n;
    // These are sufficient conditions on the assembled CSR values, not
    // universal SPD tests for FEM stiffness matrices.
    let classification = if !square {
        "NON_SQUARE_NOT_PCG"
    } else if !approximately_symmetric {
        "NON_SYMMETRIC_NOT_PCG"
    } else if !positive_diagonal {
        "NONPOSITIVE_DIAGONAL_NOT_PCG"
    } else if !exactly_symmetric {
        "APPROX_SYMMETRIC_SPD_UNVERIFIED"
    } else if strict_rows == n {
        "EXACT_SYMMETRIC_STRICT_DD_SPD"
    } else if weak_rows == n && each_component_strict {
        "EXACT_SYMMETRIC_IRREDUCIBLE_WEAK_DD_SPD"
    } else {
        "EXACT_SYMMETRIC_SPD_UNVERIFIED"
    };
    let factor_candidate = if classification.ends_with("_SPD") {
        "CERTIFIED_FOR_PCG_REVIEW" // selection is still a separate explicit step
    } else {
        "DIAGNOSTIC_ONLY"
    };
    Analysis {
        classification,
        nnz: a.nnz(),
        strict_rows,
        weak_rows,
        positive_diagonal_rows,
        missing_or_nonpositive_diagonal_rows: n - positive_diagonal_rows,
        missing_transpose,
        asymmetric_exact,
        asymmetric_tolerance,
        connected_components,
        max_row_nnz,
        max_band_offset,
        min_margin,
        factor_candidate,
    }
}

fn csv_field(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err(
            "usage: mtx_fem_audit <output.csv> <max_expanded_nnz> <matrix1.mtx> [matrix2.mtx ...]"
                .into(),
        );
    }
    let csv = Path::new(&args[0]);
    let limit: usize = args[1].parse()?;
    if !(1..=100_000_000).contains(&limit) {
        return Err("max_expanded_nnz must be in 1..=100000000".into());
    }
    let mut out = OpenOptions::new().create_new(true).write(true).open(csv)?;
    writeln!(out, "file,status,reason,field,symmetry,nrows,ncols,declared_entries,csr_nnz,strict_dd_rows,weak_dd_rows,positive_diag_rows,nonpositive_or_missing_diag_rows,missing_transpose_entries,asymmetric_exact_entries,asymmetric_tol_entries,components,max_row_nnz,max_band_offset,min_dd_margin,csr_storage_bytes,parse_and_audit_ms,pcg_gate")?;
    for arg in args.iter().skip(2) {
        let path = Path::new(arg);
        let start = Instant::now();
        let mut status = "SKIP".to_string();
        let reason: String;
        let mut field = String::new();
        let mut symmetry = String::new();
        let mut nrows = 0usize;
        let mut ncols = 0usize;
        let mut entries = 0usize;
        let mut nnz = 0usize;
        let mut strict_rows = 0usize;
        let mut weak_rows = 0usize;
        let mut positive = 0usize;
        let mut nonpositive = 0usize;
        let mut missing = 0usize;
        let mut asym_exact = 0usize;
        let mut asym_tol = 0usize;
        let mut components = 0usize;
        let mut max_row = 0usize;
        let mut max_band = 0usize;
        let mut margin = 0.0;
        let mut csr_bytes = 0usize;
        let mut gate = "DIAGNOSTIC_ONLY";
        match read_header(path) {
            Err(e) => {
                reason = e.to_string();
            }
            Ok(header) => {
                field = header.field;
                symmetry = header.symmetry;
                nrows = header.nrows;
                ncols = header.ncols;
                entries = header.entries;
                let expansion = if symmetry == "symmetric" {
                    2usize
                } else {
                    1usize
                };
                if header.format != "coordinate"
                    || !(field == "real" || field == "integer")
                    || !(symmetry == "general" || symmetry == "symmetric")
                {
                    status = "SKIP_UNSUPPORTED".to_string();
                    reason =
                        "only coordinate real/integer general/symmetric is supported".to_string();
                } else if nrows == 0 || ncols == 0 {
                    status = "SKIP_INVALID".to_string();
                    reason = "empty matrix".to_string();
                } else if entries.saturating_mul(expansion) > limit
                    || nrows > 5_000_000
                    || ncols > 5_000_000
                    || nrows > u32::MAX as usize
                    || ncols > u32::MAX as usize
                {
                    status = "SKIP_RESOURCE_LIMIT".to_string();
                    reason =
                        "matrix exceeds expanded-NNZ, 5M dimension, or index limit".to_string();
                } else {
                    match read_matrix_market(path) {
                        Err(e) => {
                            status = "SKIP_PARSE_FAILURE".to_string();
                            reason = e.to_string();
                        }
                        Ok((matrix, info)) => {
                            let diag = analyze(&matrix);
                            status = diag.classification.to_string();
                            reason = format!(
                                "combined_duplicates={} removed_zeros={}",
                                info.duplicate_entries_combined, info.zero_entries_removed
                            );
                            nnz = diag.nnz;
                            strict_rows = diag.strict_rows;
                            weak_rows = diag.weak_rows;
                            positive = diag.positive_diagonal_rows;
                            nonpositive = diag.missing_or_nonpositive_diagonal_rows;
                            missing = diag.missing_transpose;
                            asym_exact = diag.asymmetric_exact;
                            asym_tol = diag.asymmetric_tolerance;
                            components = diag.connected_components;
                            max_row = diag.max_row_nnz;
                            max_band = diag.max_band_offset;
                            margin = diag.min_margin;
                            csr_bytes = matrix.storage_bytes();
                            gate = diag.factor_candidate;
                        }
                    }
                }
            }
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        writeln!(out,
            "{},{},{},{},{},{nrows},{ncols},{entries},{nnz},{strict_rows},{weak_rows},{positive},{nonpositive},{missing},{asym_exact},{asym_tol},{components},{max_row},{max_band},{margin:.12e},{csr_bytes},{elapsed_ms:.4},{}",
            csv_field(arg), csv_field(&status), csv_field(&reason), csv_field(&field), csv_field(&symmetry), csv_field(gate)
        )?;
        out.flush()?;
        println!("G8-C9 AUDIT {} {status}: n={nrows} nnz={nnz} strict_dd={strict_rows} asymmetric={asym_tol} components={components} ({elapsed_ms:.1} ms)", path.display());
    }
    println!("=== HYBIT 0.9 G8-C9 FEM MATRIX AUDIT COMPLETE: NO SOLVES PERFORMED ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn csr(n: usize, row_ptr: Vec<u32>, col_idx: Vec<u32>, values: Vec<f64>) -> Csr32Matrix {
        Csr32Matrix::new(n, n, row_ptr, col_idx, values).unwrap()
    }
    #[test]
    fn strict_diagonal_dominance_certifies_spd() {
        let a = csr(
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![4.0, -1.0, -1.0, 4.0],
        );
        assert_eq!(analyze(&a).classification, "EXACT_SYMMETRIC_STRICT_DD_SPD");
    }
    #[test]
    fn irreducibly_weak_dd_certifies_spd() {
        let a = csr(
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0],
        );
        // First row is strict; the last two rows are weakly dominant.
        assert_eq!(
            analyze(&a).classification,
            "EXACT_SYMMETRIC_IRREDUCIBLE_WEAK_DD_SPD"
        );
    }
    #[test]
    fn positive_diagonal_is_not_spd_certificate() {
        let a = csr(2, vec![0, 2, 4], vec![0, 1, 0, 1], vec![1.0, 2.0, 2.0, 1.0]);
        assert_eq!(analyze(&a).classification, "EXACT_SYMMETRIC_SPD_UNVERIFIED");
    }
    #[test]
    fn asymmetry_and_missing_transpose_reject_pcg() {
        let a = csr(2, vec![0, 2, 3], vec![0, 1, 1], vec![4.0, -1.0, 4.0]);
        assert_eq!(analyze(&a).classification, "NON_SYMMETRIC_NOT_PCG");
    }
    #[test]
    fn two_disconnected_weak_blocks_without_strict_rows_unverified() {
        let a = csr(
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, -1.0, -1.0, 1.0],
        );
        assert_eq!(analyze(&a).classification, "EXACT_SYMMETRIC_SPD_UNVERIFIED");
    }
}
