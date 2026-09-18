use hybit_core::HybitError;
use crate::Csr32Matrix;

#[derive(Clone, Debug)]
pub struct MatrixProfile {
    pub nrows: usize,
    pub ncols: usize,
    pub nnz: usize,
    pub square: bool,
    pub full_diagonal: bool,
    pub positive_diagonal: bool,
    pub avg_nnz_per_row: f64,
    pub max_nnz_per_row: usize,
    pub csr_metadata_bytes: usize,
}

pub fn analyze_csr32(matrix: &Csr32Matrix) -> Result<MatrixProfile, HybitError> {
    matrix.validate()?;
    let mut full_diagonal = matrix.nrows() == matrix.ncols();
    let mut positive_diagonal = full_diagonal;
    let mut max_nnz_per_row = 0usize;

    if full_diagonal {
        for row in 0..matrix.nrows() {
            let start = matrix.row_ptr()[row] as usize;
            let end = matrix.row_ptr()[row + 1] as usize;
            max_nnz_per_row = max_nnz_per_row.max(end - start);
            let mut diag = 0.0;
            let mut found = false;
            for p in start..end {
                if matrix.col_idx()[p] as usize == row {
                    diag += matrix.values()[p];
                    found = true;
                }
            }
            full_diagonal &= found;
            positive_diagonal &= found && diag > 0.0 && diag.is_finite();
        }
    } else {
        for row in 0..matrix.nrows() {
            let start = matrix.row_ptr()[row] as usize;
            let end = matrix.row_ptr()[row + 1] as usize;
            max_nnz_per_row = max_nnz_per_row.max(end - start);
        }
    }

    Ok(MatrixProfile {
        nrows: matrix.nrows(),
        ncols: matrix.ncols(),
        nnz: matrix.nnz(),
        square: matrix.nrows() == matrix.ncols(),
        full_diagonal,
        positive_diagonal,
        avg_nnz_per_row: if matrix.nrows() == 0 { 0.0 } else { matrix.nnz() as f64 / matrix.nrows() as f64 },
        max_nnz_per_row,
        csr_metadata_bytes: matrix.metadata_bytes(),
    })
}
