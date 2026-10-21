//! H1-C3K6: root-only CSR to noncontiguous AssignmentOwnedCsrRows distribution.
//!
//! Only root supplies the global CSR and owner assignment. Other ranks receive
//! a replicated O(global DOF) owner map and their own CSR rows. After this call,
//! the caller can release root's full matrix; numerical SpMV uses only local
//! rows, packed ghosts, and MPI_Alltoallv through mpi_assignment.
//!
//! This is **not** parallel file I/O, nor an owner-map-free fully distributed
//! partitioner. Large MPI count limits and pre-transfer errors are checked
//! collectively before point-to-point sends start.

use crate::mpi_assignment::AssignmentOwnedCsrRows;
use crate::mpi_backend::MpiRuntime;
use crate::PartitionAssignment;
use hybit_matrix::Csr32Matrix;
use mpi::traits::*;
use std::io;
use std::time::Instant;

fn invalid(msg: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

fn prepare_chunks(
    matrix: &Csr32Matrix,
    assignment: &PartitionAssignment,
) -> io::Result<Vec<AssignmentOwnedCsrRows>> {
    let n = matrix.nrows();
    let size = assignment.rank_count();
    if n == 0
        || n != matrix.ncols()
        || n as u64 != assignment.global_dofs()
        || size == 0
        || n > i32::MAX as usize
        || matrix.nnz() > i32::MAX as usize
        || assignment.owners().len() != n
    {
        return Err(invalid("invalid root CSR or partition dimensions"));
    }
    let mut chunks = Vec::with_capacity(size as usize);
    for target in 0..size {
        let part = AssignmentOwnedCsrRows::from_global(matrix, assignment, target)?;
        if part.owned_globals.len() > i32::MAX as usize
            || part.row_ptr.len() > i32::MAX as usize
            || part.values.len() > i32::MAX as usize
            || part.global_col_idx.len() > i32::MAX as usize
        {
            return Err(invalid("rank-local MPI payload exceeds i32 Count"));
        }
        part.validate(assignment, target)?;
        chunks.push(part);
    }
    Ok(chunks)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AssignmentScatterStats {
    pub global_nnz: u64,
    pub local_nnz: usize,
    pub local_owned_bytes: usize,
    /// Local wall time for the owner-map broadcast and row distribution.
    pub scatter_ns: u128,
    /// Root's CSR exists until the caller drops its input after distribution.
    pub root_full_csr_before_drop: bool,
}

/// Distribute root's validated global CSR according to a noncontiguous owner map.
/// All ranks must enter collectively and root must be rank 0.
///
/// Root provides `Some(matrix)` and `Some(assignment)`; all other ranks MUST
/// provide `None` for both. The returned map is replicated on each rank.
/// All ranks check errors before the root sends point-to-point row payloads.
/// This operation does not change the production contiguous MPI path.
///
/// Current root setup builds all owned chunks prior to sending, trading root
/// peak memory for simpler collective preflight and failure isolation.
pub fn scatter_assignment_from_root(
    mpi: &MpiRuntime,
    root_matrix: Option<&Csr32Matrix>,
    root_assignment: Option<&PartitionAssignment>,
) -> io::Result<(
    PartitionAssignment,
    AssignmentOwnedCsrRows,
    AssignmentScatterStats,
)> {
    let size = mpi.size();
    let rank = mpi.rank();
    if size < 1 || rank < 0 {
        return Err(invalid("invalid MPI communicator"));
    }
    let world = mpi::topology::SimpleCommunicator::world();
    let begin = Instant::now();
    let wrong_arguments = if rank == 0 {
        root_matrix.is_none() || root_assignment.is_none()
    } else {
        root_matrix.is_some() || root_assignment.is_some()
    };
    if mpi.all_reduce_sum_u64(u64::from(wrong_arguments)) != 0 {
        return Err(invalid(
            "root-only CSR and ownership argument contract violated",
        ));
    }
    let prepared = if rank == 0 {
        Some((|| -> io::Result<_> {
            let matrix = root_matrix.ok_or_else(|| invalid("missing root matrix"))?;
            let assign = root_assignment.ok_or_else(|| invalid("missing root assignment"))?;
            if assign.rank_count() != size as u32 {
                return Err(invalid("assignment MPI rank count mismatch"));
            }
            let chunks = prepare_chunks(matrix, assign)?;
            Ok((assign.clone(), chunks, matrix.nnz() as u64))
        })())
    } else {
        None
    };
    if mpi.all_reduce_sum_u64(u64::from(prepared.as_ref().is_some_and(Result::is_err))) != 0 {
        if let Some(Err(e)) = &prepared {
            eprintln!("K6 root-owned CSR preparation failed: {e}");
        }
        return Err(invalid("root-owned CSR preflight failed"));
    }
    let mut prepared = prepared.transpose()?;
    let mut header = [0u64; 2];
    if let Some((assignment, _, nnz)) = &prepared {
        header = [assignment.global_dofs(), *nnz];
    }
    world.process_at_rank(0).broadcast_into(&mut header[..]);
    let n = usize::try_from(header[0]).map_err(|_| invalid("global DOF usize overflow"))?;
    if n == 0 || n > i32::MAX as usize || header[1] > i32::MAX as u64 {
        return Err(invalid(
            "global CSR input exceeds bounded root-scatter limits",
        ));
    }
    let mut owners = vec![0u32; n];
    if let Some((assignment, _, _)) = &prepared {
        owners.copy_from_slice(assignment.owners());
    }
    world.process_at_rank(0).broadcast_into(&mut owners[..]);
    let built_assignment = PartitionAssignment::from_owners(size as u32, owners)
        .map_err(|_| invalid("invalid broadcast root owner map"));
    if mpi.all_reduce_sum_u64(u64::from(built_assignment.is_err())) != 0 {
        return Err(invalid("invalid broadcast owner map on at least one rank"));
    }
    let assignment = built_assignment?;
    // Each receiving rank knows its row count before receiving the header.
    let owned_len = assignment
        .owners()
        .iter()
        .filter(|&&owner| owner == rank as u32)
        .count();
    let rows = if rank == 0 {
        let (_, chunks, _) = prepared
            .take()
            .ok_or_else(|| invalid("missing root payloads"))?;
        for target_rank in 1..size {
            let chunk = &chunks[target_rank as usize];
            let dest = world.process_at_rank(target_rank);
            let counts = [chunk.owned_globals.len() as u64, chunk.values.len() as u64];
            dest.send(&counts[..]);
            dest.send(&chunk.owned_globals[..]);
            dest.send(&chunk.row_ptr[..]);
            dest.send(&chunk.global_col_idx[..]);
            dest.send(&chunk.values[..]);
        }
        chunks
            .into_iter()
            .next()
            .ok_or_else(|| invalid("root has no owned rows"))?
    } else {
        let source = world.process_at_rank(0);
        let mut counts = [0u64; 2];
        source.receive_into(&mut counts[..]);
        // Every root chunk was validated to fit i32 before sending.
        if counts[0] != owned_len as u64 || counts[1] > i32::MAX as u64 {
            return Err(invalid("received root row counts do not match owner map"));
        }
        let nnz = counts[1] as usize;
        let mut owned_globals = vec![0u64; owned_len];
        let mut row_ptr = vec![0u32; owned_len + 1];
        let mut global_col_idx = vec![0u64; nnz];
        let mut values = vec![0.0_f64; nnz];
        source.receive_into(&mut owned_globals[..]);
        source.receive_into(&mut row_ptr[..]);
        source.receive_into(&mut global_col_idx[..]);
        source.receive_into(&mut values[..]);
        AssignmentOwnedCsrRows {
            global_dofs: header[0],
            owned_globals,
            row_ptr,
            global_col_idx,
            values,
        }
    };
    let invalid_chunk = rows.validate(&assignment, rank as u32).is_err();
    if mpi.all_reduce_sum_u64(u64::from(invalid_chunk)) != 0 {
        return Err(invalid("distributed noncontiguous row CSR is invalid"));
    }
    let stats = AssignmentScatterStats {
        global_nnz: header[1],
        local_nnz: rows.values.len(),
        local_owned_bytes: std::mem::size_of_val(rows.owned_globals.as_slice())
            + std::mem::size_of_val(rows.row_ptr.as_slice())
            + std::mem::size_of_val(rows.global_col_idx.as_slice())
            + std::mem::size_of_val(rows.values.as_slice()),
        scatter_ns: begin.elapsed().as_nanos(),
        root_full_csr_before_drop: rank == 0,
    };
    Ok((assignment, rows, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix() -> Csr32Matrix {
        Csr32Matrix::new(
            5,
            5,
            vec![0, 2, 4, 7, 9, 10],
            vec![0, 1, 1, 2, 0, 2, 3, 1, 3, 4],
            vec![4., 1., 5., 2., 3., 6., 1., 1., 7., 9.],
        )
        .unwrap()
    }

    #[test]
    fn root_chunks_preserve_noncontiguous_rows_and_coefficients() {
        let m = matrix();
        let a = PartitionAssignment::from_owners(2, vec![0, 1, 0, 1, 0]).unwrap();
        let chunks = prepare_chunks(&m, &a).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].owned_globals, [0, 2, 4]);
        assert_eq!(chunks[1].owned_globals, [1, 3]);
        assert_eq!(chunks[0].values, [4., 1., 3., 6., 1., 9.]);
        assert_eq!(chunks[1].values, [5., 2., 1., 7.]);
        assert_eq!(
            chunks.iter().map(|c| c.values.len()).sum::<usize>(),
            m.nnz()
        );
    }

    #[test]
    fn root_chunks_reject_shape_mismatch() {
        let m = matrix();
        let a = PartitionAssignment::from_owners(2, vec![0, 1, 0, 1, 0]).unwrap();
        let bad =
            Csr32Matrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1., 2., 3.]).unwrap();
        assert!(prepare_chunks(&bad, &a).is_err());
        assert!(prepare_chunks(&m, &a).is_ok());
    }
}
