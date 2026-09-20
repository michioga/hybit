use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use hybit_core::HybitError;

use crate::Csr32Matrix;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixMarketSymmetry {
    General,
    Symmetric,
}

#[derive(Clone, Debug)]
pub struct MatrixMarketInfo {
    pub nrows: usize,
    pub ncols: usize,
    /// Number of entries stated in the Matrix Market size line.
    pub input_entries: usize,
    /// Number of entries after symmetric expansion, duplicate summation and
    /// removal of exact zeros.
    pub csr_nnz: usize,
    pub symmetry: MatrixMarketSymmetry,
    pub duplicate_entries_combined: usize,
    pub zero_entries_removed: usize,
}

#[derive(Debug)]
pub enum MatrixMarketError {
    Io(std::io::Error),
    Format(String),
    Matrix(HybitError),
}

impl Display for MatrixMarketError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "Matrix Market I/O error: {e}"),
            Self::Format(msg) => write!(f, "invalid Matrix Market file: {msg}"),
            Self::Matrix(e) => write!(f, "invalid matrix after Matrix Market import: {e}"),
        }
    }
}

impl Error for MatrixMarketError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Matrix(e) => Some(e),
            Self::Format(_) => None,
        }
    }
}

impl From<std::io::Error> for MatrixMarketError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<HybitError> for MatrixMarketError {
    fn from(value: HybitError) -> Self {
        Self::Matrix(value)
    }
}

pub fn write_matrix_market_general<P: AsRef<Path>>(
    path: P,
    matrix: &Csr32Matrix,
) -> Result<(), MatrixMarketError> {
    matrix.validate()?;
    let file = File::create(path)?;
    let mut out = BufWriter::new(file);
    writeln!(out, "%%MatrixMarket matrix coordinate real general")?;
    writeln!(out, "% Written by HyBIT")?;
    writeln!(
        out,
        "{} {} {}",
        matrix.nrows(),
        matrix.ncols(),
        matrix.nnz()
    )?;
    for row in 0..matrix.nrows() {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        for p in start..end {
            writeln!(
                out,
                "{} {} {:.17e}",
                row + 1,
                matrix.col_idx()[p] as usize + 1,
                matrix.values()[p]
            )?;
        }
    }
    out.flush()?;
    Ok(())
}

pub fn read_matrix_market<P: AsRef<Path>>(
    path: P,
) -> Result<(Csr32Matrix, MatrixMarketInfo), MatrixMarketError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    read_matrix_market_from_reader(reader)
}

pub fn read_matrix_market_from_reader<R: BufRead>(
    reader: R,
) -> Result<(Csr32Matrix, MatrixMarketInfo), MatrixMarketError> {
    let mut lines = reader.lines();
    let header = lines
        .next()
        .ok_or_else(|| MatrixMarketError::Format("missing header".into()))??;
    let header_tokens: Vec<_> = header.split_whitespace().collect();
    if header_tokens.len() != 5
        || !header_tokens[0].eq_ignore_ascii_case("%%MatrixMarket")
        || !header_tokens[1].eq_ignore_ascii_case("matrix")
        || !header_tokens[2].eq_ignore_ascii_case("coordinate")
    {
        return Err(MatrixMarketError::Format(
            "expected '%%MatrixMarket matrix coordinate <field> <symmetry>'".into(),
        ));
    }
    if !(header_tokens[3].eq_ignore_ascii_case("real")
        || header_tokens[3].eq_ignore_ascii_case("integer"))
    {
        return Err(MatrixMarketError::Format(
            "only real and integer coordinate matrices are supported".into(),
        ));
    }
    let symmetry = if header_tokens[4].eq_ignore_ascii_case("general") {
        MatrixMarketSymmetry::General
    } else if header_tokens[4].eq_ignore_ascii_case("symmetric") {
        MatrixMarketSymmetry::Symmetric
    } else {
        return Err(MatrixMarketError::Format(
            "only general and symmetric matrices are supported".into(),
        ));
    };

    let size_line = loop {
        let line = lines
            .next()
            .ok_or_else(|| MatrixMarketError::Format("missing matrix size line".into()))??;
        let t = line.trim();
        if !t.is_empty() && !t.starts_with('%') {
            break t.to_owned();
        }
    };
    let size: Vec<_> = size_line.split_whitespace().collect();
    if size.len() != 3 {
        return Err(MatrixMarketError::Format(
            "size line must contain nrows ncols nnz".into(),
        ));
    }
    let nrows = parse_usize(size[0], "nrows")?;
    let ncols = parse_usize(size[1], "ncols")?;
    let input_entries = parse_usize(size[2], "nnz")?;
    if nrows > u32::MAX as usize || ncols > u32::MAX as usize {
        return Err(MatrixMarketError::Matrix(HybitError::SizeOverflow));
    }
    if symmetry == MatrixMarketSymmetry::Symmetric && nrows != ncols {
        return Err(MatrixMarketError::Format(
            "symmetric Matrix Market matrix must be square".into(),
        ));
    }

    let reserve = if symmetry == MatrixMarketSymmetry::Symmetric {
        input_entries.saturating_mul(2)
    } else {
        input_entries
    };
    let mut entries: Vec<(u32, u32, f64)> = Vec::with_capacity(reserve);
    let mut read_entries = 0usize;

    for line in lines {
        let line = line?;
        let t = line.trim();
        if t.is_empty() || t.starts_with('%') {
            continue;
        }
        if read_entries >= input_entries {
            return Err(MatrixMarketError::Format(
                "more data entries than declared in size line".into(),
            ));
        }
        let fields: Vec<_> = t.split_whitespace().collect();
        if fields.len() != 3 {
            return Err(MatrixMarketError::Format(
                "coordinate entry must contain row column value".into(),
            ));
        }
        let row1 = parse_usize(fields[0], "row index")?;
        let col1 = parse_usize(fields[1], "column index")?;
        if row1 == 0 || row1 > nrows || col1 == 0 || col1 > ncols {
            return Err(MatrixMarketError::Format(
                "Matrix Market indices are 1-based and must lie inside matrix dimensions".into(),
            ));
        }
        let value: f64 = fields[2].parse().map_err(|_| {
            MatrixMarketError::Format(format!("invalid numeric value '{}'", fields[2]))
        })?;
        if !value.is_finite() {
            return Err(MatrixMarketError::Format("NaN/Inf matrix value".into()));
        }
        let row = (row1 - 1) as u32;
        let col = (col1 - 1) as u32;
        entries.push((row, col, value));
        if symmetry == MatrixMarketSymmetry::Symmetric && row != col {
            entries.push((col, row, value));
        }
        read_entries += 1;
    }
    if read_entries != input_entries {
        return Err(MatrixMarketError::Format(format!(
            "size line declares {input_entries} entries but {read_entries} were read"
        )));
    }

    entries.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    let mut duplicate_entries_combined = 0usize;
    let mut compact: Vec<(u32, u32, f64)> = Vec::with_capacity(entries.len());
    for (row, col, value) in entries {
        if let Some(last) = compact.last_mut() {
            if last.0 == row && last.1 == col {
                last.2 += value;
                if !last.2.is_finite() {
                    return Err(MatrixMarketError::Format(
                        "duplicate summation produced NaN/Inf".into(),
                    ));
                }
                duplicate_entries_combined += 1;
                continue;
            }
        }
        compact.push((row, col, value));
    }
    let before_zero_removal = compact.len();
    compact.retain(|entry| entry.2 != 0.0);
    let zero_entries_removed = before_zero_removal - compact.len();
    if compact.len() > u32::MAX as usize {
        return Err(MatrixMarketError::Matrix(HybitError::SizeOverflow));
    }

    let mut row_ptr = vec![0u32; nrows + 1];
    for &(row, _, _) in &compact {
        row_ptr[row as usize + 1] = row_ptr[row as usize + 1]
            .checked_add(1)
            .ok_or(MatrixMarketError::Matrix(HybitError::SizeOverflow))?;
    }
    for i in 0..nrows {
        row_ptr[i + 1] = row_ptr[i + 1]
            .checked_add(row_ptr[i])
            .ok_or(MatrixMarketError::Matrix(HybitError::SizeOverflow))?;
    }
    let mut col_idx = Vec::with_capacity(compact.len());
    let mut values = Vec::with_capacity(compact.len());
    for (_, col, value) in compact {
        col_idx.push(col);
        values.push(value);
    }
    let matrix = Csr32Matrix::new(nrows, ncols, row_ptr, col_idx, values)?;
    let info = MatrixMarketInfo {
        nrows,
        ncols,
        input_entries,
        csr_nnz: matrix.nnz(),
        symmetry,
        duplicate_entries_combined,
        zero_entries_removed,
    };
    Ok((matrix, info))
}

fn parse_usize(text: &str, what: &str) -> Result<usize, MatrixMarketError> {
    text.parse::<usize>()
        .map_err(|_| MatrixMarketError::Format(format!("invalid {what} '{text}'")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn reads_symmetric_coordinate_matrix() {
        let text = b"%%MatrixMarket matrix coordinate real symmetric\n% demo\n3 3 5\n1 1 2\n2 1 -1\n2 2 2\n3 2 -1\n3 3 2\n";
        let (a, info) = read_matrix_market_from_reader(Cursor::new(text)).unwrap();
        assert_eq!(info.symmetry, MatrixMarketSymmetry::Symmetric);
        assert_eq!(a.nnz(), 7);
        let y = a.spmv(&[1.0, 1.0, 1.0]).unwrap();
        assert_eq!(y, vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn writer_round_trips_general_matrix() {
        let a = Csr32Matrix::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![2.0, -1.0, -1.0, 3.0],
        )
        .unwrap();
        let mut path = std::env::temp_dir();
        path.push(format!("hybit-mm-{}-{}.mtx", std::process::id(), 1));
        write_matrix_market_general(&path, &a).unwrap();
        let (b, info) = read_matrix_market(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(info.symmetry, MatrixMarketSymmetry::General);
        assert_eq!(a.row_ptr(), b.row_ptr());
        assert_eq!(a.col_idx(), b.col_idx());
        assert_eq!(a.values(), b.values());
    }

    #[test]
    fn combines_duplicate_entries() {
        let text =
            b"%%MatrixMarket matrix coordinate real general\n2 2 4\n1 1 1\n1 1 2\n2 2 4\n1 2 0\n";
        let (a, info) = read_matrix_market_from_reader(Cursor::new(text)).unwrap();
        assert_eq!(a.nnz(), 2);
        assert_eq!(info.duplicate_entries_combined, 1);
        assert_eq!(info.zero_entries_removed, 1);
        assert_eq!(a.values(), &[3.0, 4.0]);
    }
}
