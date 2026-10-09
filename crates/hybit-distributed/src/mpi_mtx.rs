//! G8-C3: rank-zero Matrix Market ingestion and MPI-owned CSR distribution.
//!
//! The root temporarily holds the complete input. Other ranks never do.
//! After redistribution, the existing G8-C2 `prepare_owned_rows` builds halos
//! without full matrix replication. Matrix Market support is intentionally
//! restricted to coordinate real/integer general/symmetric matrices.
//!
//! The default PCG eligibility is conservative: symmetric, positive diagonal,
//! strictly row diagonally dominant. G8-C4 adds an explicitly experimental
//! acceptance policy that is NOT a mathematical SPD certificate.

use crate::mpi_backend::MpiRuntime;
use crate::mpi_input::OwnedCsrRows;
use crate::ContiguousPartition;
use mpi::traits::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// G8-C4: parser acceptance is separate from mathematical SPD certification.
/// StrictCertified is the safe default; ExperimentalUnverified permits
/// symmetric positive-diagonal candidates, but DOES NOT prove they are SPD.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MtxPcgPolicy {
    StrictCertified,
    ExperimentalUnverified,
}

#[derive(Debug)]
struct RootCsr {
    n: u64,
    row_ptr: Vec<u32>,
    cols: Vec<u64>,
    vals: Vec<f64>,
}

fn next_record<I: Iterator<Item = io::Result<String>>>(
    lines: &mut I,
) -> io::Result<Option<String>> {
    for line in lines {
        let line = line?;
        let line = line.trim();
        if !line.is_empty() && !line.starts_with('%') {
            return Ok(Some(line.to_string()));
        }
    }
    Ok(None)
}

fn read_root<R: BufRead>(reader: R, policy: MtxPcgPolicy) -> io::Result<RootCsr> {
    let mut lines = reader.lines();
    let header = lines
        .next()
        .ok_or_else(|| invalid("empty Matrix Market input"))??;
    let fields: Vec<&str> = header.split_whitespace().collect();
    if fields.len() != 5
        || !fields[0].eq_ignore_ascii_case("%%MatrixMarket")
        || !fields[1].eq_ignore_ascii_case("matrix")
        || !fields[2].eq_ignore_ascii_case("coordinate")
        || !(fields[3].eq_ignore_ascii_case("real") || fields[3].eq_ignore_ascii_case("integer"))
        || !(fields[4].eq_ignore_ascii_case("symmetric")
            || fields[4].eq_ignore_ascii_case("general"))
    {
        return Err(invalid(
            "supported Matrix Market: matrix coordinate real/integer symmetric/general",
        ));
    }
    let symmetric = fields[4].eq_ignore_ascii_case("symmetric");
    let dims =
        next_record(&mut lines)?.ok_or_else(|| invalid("missing Matrix Market dimensions"))?;
    let d: Vec<&str> = dims.split_whitespace().collect();
    if d.len() != 3 {
        return Err(invalid("Matrix Market dimension record needs 3 integers"));
    }
    let parse_usize = |s: &str| {
        s.parse::<usize>()
            .map_err(|_| invalid("invalid matrix size or index"))
    };
    let n = parse_usize(d[0])?;
    let cols = parse_usize(d[1])?;
    let entries = parse_usize(d[2])?;
    // Both local CSR and MPI payload counts currently use 32-bit indices.
    if n == 0 || n != cols || n > i32::MAX as usize || entries > i32::MAX as usize / 2 {
        return Err(invalid(
            "matrix must be square and fit local/MPI index limits",
        ));
    }
    let mut rows = vec![BTreeMap::<usize, f64>::new(); n];
    for k in 0..entries {
        let rec =
            next_record(&mut lines)?.ok_or_else(|| invalid(format!("missing coordinate {k}")))?;
        let f: Vec<&str> = rec.split_whitespace().collect();
        if f.len() != 3 {
            return Err(invalid("coordinate records require row column value"));
        }
        let r = parse_usize(f[0])?;
        let c = parse_usize(f[1])?;
        let value: f64 = f[2]
            .parse()
            .map_err(|_| invalid("invalid Matrix Market coefficient"))?;
        if r == 0 || c == 0 || r > n || c > n || !value.is_finite() {
            return Err(invalid(
                "Matrix Market row/column/value outside valid domain",
            ));
        }
        *rows[r - 1].entry(c - 1).or_insert(0.0) += value;
        if symmetric && r != c {
            *rows[c - 1].entry(r - 1).or_insert(0.0) += value;
        }
    }
    if next_record(&mut lines)?.is_some() {
        return Err(invalid("extra coordinate records beyond declared nnz"));
    }
    // A positive symmetric strictly diagonally dominant real matrix is SPD.
    for (i, row) in rows.iter().enumerate() {
        let diagonal = *row.get(&i).unwrap_or(&0.0);
        let mut off_sum = 0.0;
        for (&j, &value) in row {
            if !value.is_finite() {
                return Err(invalid("nonfinite coefficient after duplicate coalescing"));
            }
            if i != j {
                off_sum += value.abs();
                let transpose = rows[j].get(&i).copied().unwrap_or(0.0);
                if !transpose.is_finite()
                    || (value - transpose).abs()
                        > 1.0e-12 * (1.0 + value.abs().max(transpose.abs()))
                {
                    return Err(invalid("matrix is not numerically symmetric: refuse PCG"));
                }
            }
        }
        if !diagonal.is_finite() || diagonal <= 0.0 || !off_sum.is_finite() {
            return Err(invalid(
                "matrix has invalid/nonpositive diagonal or off-diagonal sum",
            ));
        }
        if policy == MtxPcgPolicy::StrictCertified && diagonal <= off_sum {
            return Err(invalid(
                "matrix lacks positive strict diagonal dominance: SPD not certified",
            ));
        }
    }
    let total_nnz: usize = rows.iter().map(BTreeMap::len).sum();
    if total_nnz > i32::MAX as usize {
        return Err(invalid(
            "expanded/coalesced CSR exceeds MPI i32 payload count",
        ));
    }
    let mut ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(total_nnz);
    let mut values = Vec::with_capacity(total_nnz);
    ptr.push(0u32);
    for row in rows {
        for (col, value) in row {
            if value != 0.0 {
                col_idx.push(col as u64);
                values.push(value);
            }
        }
        ptr.push(u32::try_from(values.len()).map_err(|_| invalid("CSR pointer overflow"))?);
    }
    Ok(RootCsr {
        n: n as u64,
        row_ptr: ptr,
        cols: col_idx,
        vals: values,
    })
}

impl RootCsr {
    fn owned(&self, partition: &ContiguousPartition, rank: u32) -> io::Result<OwnedCsrRows> {
        let owned = partition
            .owned_range(rank)
            .map_err(|e| invalid(e.to_string()))?;
        let start = owned.start as usize;
        let end = owned.end as usize;
        let first = self.row_ptr[start] as usize;
        let last = self.row_ptr[end] as usize;
        Ok(OwnedCsrRows {
            global_dofs: self.n,
            owned,
            row_ptr: self.row_ptr[start..=end]
                .iter()
                .map(|&p| p - self.row_ptr[start])
                .collect(),
            global_col_idx: self.cols[first..last].to_vec(),
            values: self.vals[first..last].to_vec(),
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MtxDistributionStats {
    pub global_nnz: u64,
    /// Nonzero only on rank 0.
    pub root_read_ns: u128,
    /// Rank-local duration, including final input validation collective.
    pub distribution_ns: u128,
    /// Rank-local owned CSR payload estimate (excludes the temporary root copy).
    pub owned_bytes: usize,
}

/// Rank 0 parses and SPD-certifies the file, then sends one owned CSR portion
/// to each rank. Non-root ranks never allocate the full matrix. This is root
/// distribution, not parallel filesystem I/O; the root read is a setup cost.
///
/// Every rank must pass the same path, call in the same epoch, and use MPI
/// from the initializing thread. Root read failures are propagated to all ranks
/// before any point-to-point traffic starts.
pub fn distribute_matrix_market(
    mpi: &MpiRuntime,
    path: &Path,
) -> io::Result<(ContiguousPartition, OwnedCsrRows, MtxDistributionStats)> {
    distribute_matrix_market_with_policy(mpi, path, MtxPcgPolicy::StrictCertified)
}

/// G8-C4 experimental input path. Non-strict candidates are NOT SPD-certified.
/// Use only with a caller-authorized exploratory PCG run, and inspect status,
/// curvature and true residual before accepting a solution.
pub fn distribute_matrix_market_with_policy(
    mpi: &MpiRuntime,
    path: &Path,
    policy: MtxPcgPolicy,
) -> io::Result<(ContiguousPartition, OwnedCsrRows, MtxDistributionStats)> {
    let size = mpi.size();
    let rank = mpi.rank();
    if size < 1 || rank < 0 {
        return Err(invalid("invalid MPI communicator"));
    }
    let read_start = Instant::now();
    let loaded = if rank == 0 {
        Some(File::open(path).and_then(|file| read_root(BufReader::new(file), policy)))
    } else {
        None
    };
    let root_read_ns = if rank == 0 {
        read_start.elapsed().as_nanos()
    } else {
        0
    };
    if mpi.all_reduce_sum_u64(u64::from(loaded.as_ref().is_some_and(Result::is_err))) > 0 {
        if let Some(Err(err)) = &loaded {
            eprintln!("G8-C3 Matrix Market rejected: {err}");
        }
        return Err(invalid("Matrix Market input rejected on root rank"));
    }
    let root = loaded.transpose()?;
    let global_n = mpi.all_reduce_sum_u64(root.as_ref().map_or(0, |m| m.n));
    let global_nnz = mpi.all_reduce_sum_u64(root.as_ref().map_or(0, |m| m.vals.len() as u64));
    let partition = ContiguousPartition::balanced(global_n, size as u32)
        .map_err(|err| invalid(err.to_string()))?;
    mpi.barrier();
    let start = Instant::now();
    let world = mpi::topology::SimpleCommunicator::world();
    let owned = if rank == 0 {
        let root = root.ok_or_else(|| invalid("missing root CSR"))?;
        let mut own = None;
        for target_rank in 0..size {
            let chunk = root.owned(&partition, target_rank as u32)?;
            if target_rank == 0 {
                own = Some(chunk);
            } else {
                let target = world.process_at_rank(target_rank);
                let len = [chunk.values.len() as u64];
                target.send(&len[..]);
                target.send(&chunk.row_ptr[..]);
                target.send(&chunk.global_col_idx[..]);
                target.send(&chunk.values[..]);
            }
        }
        own.ok_or_else(|| invalid("root owned rows missing"))?
    } else {
        let owned = partition
            .owned_range(rank as u32)
            .map_err(|e| invalid(e.to_string()))?;
        let source = world.process_at_rank(0);
        let mut header = [0u64; 1];
        source.receive_into(&mut header[..]);
        let len = usize::try_from(header[0]).map_err(|_| invalid("rank-local nnz overflow"))?;
        if len > i32::MAX as usize {
            return Err(invalid("MPI rank-local nnz exceeds Count"));
        }
        let mut ptr = vec![0u32; (owned.end - owned.start) as usize + 1];
        let mut cols = vec![0u64; len];
        let mut vals = vec![0.0f64; len];
        source.receive_into(&mut ptr[..]);
        source.receive_into(&mut cols[..]);
        source.receive_into(&mut vals[..]);
        OwnedCsrRows {
            global_dofs: global_n,
            owned,
            row_ptr: ptr,
            global_col_idx: cols,
            values: vals,
        }
    };
    let local_bad = owned.validate().is_err();
    if mpi.all_reduce_sum_u64(u64::from(local_bad)) != 0 {
        return Err(invalid(
            "distributed Matrix Market local CSR failed validation",
        ));
    }
    let bytes = owned.row_ptr.len() * 4 + owned.global_col_idx.len() * 8 + owned.values.len() * 8;
    let stats = MtxDistributionStats {
        global_nnz,
        root_read_ns,
        distribution_ns: start.elapsed().as_nanos(),
        owned_bytes: bytes,
    };
    Ok((partition, owned, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn symmetric_and_duplicate_entries_coalesce() {
        let input = "%%MatrixMarket matrix coordinate real symmetric\n% c\n3 3 6\n1 1 4\n1 2 -1\n2 2 5\n2 3 -1\n3 3 4\n1 2 -0.25\n";
        let a = read_root(Cursor::new(input), MtxPcgPolicy::StrictCertified).unwrap();
        assert_eq!(a.n, 3);
        assert_eq!(&a.row_ptr[..], &[0, 2, 5, 7]);
        assert_eq!(a.vals[1], -1.25);
        assert_eq!(a.vals[2], -1.25);
    }

    #[test]
    fn non_strict_spd_can_be_experimentally_loaded_but_is_not_certified() {
        // Dirichlet 1-D Laplacian is SPD, but the central row is only weakly DD.
        let mtx = "%%MatrixMarket matrix coordinate real symmetric\n3 3 5\n1 1 2\n2 2 2\n3 3 2\n1 2 -1\n2 3 -1\n";
        assert!(read_root(Cursor::new(mtx), MtxPcgPolicy::StrictCertified).is_err());
        assert!(read_root(Cursor::new(mtx), MtxPcgPolicy::ExperimentalUnverified).is_ok());
    }

    #[test]
    fn experimental_policy_is_not_an_spd_certificate() {
        // Positive diagonals and symmetry still admit an INDEFINITE matrix.
        let indef = "%%MatrixMarket matrix coordinate real symmetric\n2 2 3\n1 1 1\n2 2 1\n1 2 2\n";
        assert!(read_root(Cursor::new(indef), MtxPcgPolicy::StrictCertified).is_err());
        assert!(read_root(Cursor::new(indef), MtxPcgPolicy::ExperimentalUnverified).is_ok());
        let negative = "%%MatrixMarket matrix coordinate real symmetric\n2 2 2\n1 1 -1\n2 2 3\n";
        assert!(read_root(Cursor::new(negative), MtxPcgPolicy::ExperimentalUnverified).is_err());
    }

    #[test]
    fn reject_nonsymmetric_general_or_uncertified_spd() {
        let input = "%%MatrixMarket matrix coordinate real general\n2 2 3\n1 1 4\n1 2 -1\n2 2 4\n";
        assert!(read_root(Cursor::new(input), MtxPcgPolicy::StrictCertified).is_err());
        let indefinite =
            "%%MatrixMarket matrix coordinate real symmetric\n2 2 3\n1 1 1\n2 2 1\n1 2 -2\n";
        assert!(read_root(Cursor::new(indefinite), MtxPcgPolicy::StrictCertified).is_err());
    }
}
