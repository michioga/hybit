//! C1-G5F research example: run the library feature observer, not a selector.
//! CLI is development-only; all diagnostics are calculated by hybit-auto.

use std::error::Error;
use std::path::PathBuf;

use hybit::{HybitSolver, SelectionModel, read_matrix_market};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut matrix: Option<PathBuf> = None;
    let mut reuse: Option<usize> = None;
    while let Some(flag) = args.next() {
        let value = args.next().ok_or("missing flag value")?;
        match flag.as_str() {
            "--matrix" => matrix = Some(PathBuf::from(value)),
            "--reuse" => reuse = Some(value.parse()?),
            _ => return Err(format!("unknown argument: {flag}").into()),
        }
    }
    let matrix_path = matrix.ok_or("missing --matrix")?;
    let reuse = reuse.ok_or("missing --reuse")?;
    let (a, _) = read_matrix_market(&matrix_path)?;
    let exact = vec![1.0_f64; a.ncols()];
    let rhs = a.spmv(&exact)?;
    let model = SelectionModel::embedded_standard();
    let observed = HybitSolver::new().shadow_audit_csr32_with_model(&a, &rhs, reuse, &model)?;
    let features = observed.coupling;
    println!(
        "C1G5F|path={}|rows={}|nnz={}|offdiag={}|positive_fraction={:.12e}|nondominant_rows={:?}|max_offdiag_ratio={:?}|strong_fraction={:?}|invalid_ratio={}|rhs_nnz_fraction={:.12e}|diagonal_spread={:?}|shadow_reason={:?}|research_rank={:?}|production_recommended={:?}|audit_ms={:.6}",
        matrix_path.display(),
        a.nrows(),
        a.nnz(),
        features.offdiagonal_entries,
        features.positive_offdiagonal_fraction,
        features.diagonal_non_dominant_rows,
        features.max_offdiagonal_to_diagonal,
        features.strong_normalized_offdiagonal_fraction,
        features.nonfinite_ratio_detected,
        observed.model.base.features.rhs_nonzero_fraction,
        observed.model.base.features.diagonal_log10_spread,
        observed.model.base.reason,
        observed.model.diagnostic_candidate,
        observed.model.base.recommended,
        observed.audit_ms,
    );
    Ok(())
}
