//! H1-C3K7-A: bounded-memory root-only CSR scatter for noncontiguous owner maps.
//!
//! Only root supplies the global CSR and owner assignment. Other ranks receive
//! a replicated O(global DOF) owner map and their own CSR rows. After this call,
//! the caller can release root's full matrix; numerical SpMV uses only local
//! rows, packed ghosts, and MPI_Alltoallv through mpi_assignment.
//!
//! This is **not** parallel file I/O, nor an owner-map-free fully distributed
//! partitioner. Root preflights just O(ranks) payload counts, then builds, sends,
//! and drops one destination CSR chunk at a time. Rank 0 builds its own chunk
//! last. Pre-transfer failures are checked collectively before MPI sends start.

use crate::PartitionAssignment;
use crate::mpi_assignment::AssignmentOwnedCsrRows;
use crate::mpi_backend::MpiRuntime;
use hybit_matrix::Csr32Matrix;
use mpi::traits::*;
use std::io;
use std::time::Instant;

fn invalid(msg: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

/// Only O(ranks) metadata is allocated before the first point-to-point send.
/// No per-rank CSR payload is built here. This separates all possible input /
/// MPI Count errors from streaming, where early rank-local failure could hang
/// other ranks in blocking receives.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ChunkCounts {
    pub(crate) rows: usize,
    pub(crate) nnz: usize,
}

pub(crate) fn chunk_payload_bytes(c: ChunkCounts) -> io::Result<u64> {
    let rows = u64::try_from(c.rows).map_err(|_| invalid("owned row count overflows u64"))?;
    let nnz = u64::try_from(c.nnz).map_err(|_| invalid("local nnz count overflows u64"))?;
    rows.checked_mul(8)
        .and_then(|v| v.checked_add(rows.checked_add(1)?.checked_mul(4)?))
        .and_then(|v| v.checked_add(nnz.checked_mul(16)?))
        .ok_or_else(|| invalid("owned chunk byte estimate overflow"))
}

pub(crate) fn preflight_chunk_counts(
    matrix: &Csr32Matrix,
    assignment: &PartitionAssignment,
) -> io::Result<Vec<ChunkCounts>> {
    let n = matrix.nrows();
    let ranks = assignment.rank_count() as usize;
    if n == 0
        || n != matrix.ncols()
        || n as u64 != assignment.global_dofs()
        || ranks == 0
        || n > i32::MAX as usize
        || matrix.nnz() > i32::MAX as usize
        || assignment.owners().len() != n
        || matrix.col_idx().iter().any(|&col| col as usize >= n)
        || matrix.values().iter().any(|value| !value.is_finite())
    {
        return Err(invalid("invalid root CSR or partition dimensions/values"));
    }
    let mut counts = vec![ChunkCounts::default(); ranks];
    for (row, &owner) in assignment.owners().iter().enumerate() {
        let c = counts
            .get_mut(owner as usize)
            .ok_or_else(|| invalid("owner rank outside MPI world"))?;
        c.rows = c
            .rows
            .checked_add(1)
            .ok_or_else(|| invalid("owned row overflow"))?;
        let width = (matrix.row_ptr()[row + 1] - matrix.row_ptr()[row]) as usize;
        c.nnz = c
            .nnz
            .checked_add(width)
            .ok_or_else(|| invalid("owned nnz overflow"))?;
    }
    if counts
        .iter()
        .any(|c| c.rows == 0 || c.rows >= i32::MAX as usize || c.nnz > i32::MAX as usize)
        || counts.iter().map(|c| c.nnz).sum::<usize>() != matrix.nnz()
    {
        return Err(invalid("invalid or oversized per-rank CSR payload"));
    }
    for &c in &counts {
        chunk_payload_bytes(c)?;
    }
    Ok(counts)
}

pub(crate) fn build_checked_chunk(
    matrix: &Csr32Matrix,
    assignment: &PartitionAssignment,
    rank: u32,
    expected: ChunkCounts,
) -> io::Result<AssignmentOwnedCsrRows> {
    let chunk = AssignmentOwnedCsrRows::from_global(matrix, assignment, rank)?;
    if chunk.owned_globals.len() != expected.rows || chunk.values.len() != expected.nnz {
        return Err(invalid(
            "built CSR disagrees with collective preflight counts",
        ));
    }
    Ok(chunk)
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
    /// Root-only estimate of the largest one-at-a-time transient CSR chunk.
    /// Excludes global input CSR, owner maps, allocator overhead and parser RSS.
    pub root_peak_streamed_chunk_bytes_estimate: u64,
    /// Root-only estimate of the simultaneous chunk allocation in K6.
    pub root_k6_all_chunks_bytes_estimate: u64,
    /// Number of temporary per-destination CSR chunks held at once by root.
    pub max_concurrent_root_chunks: u32,
}

/// Distribute root's global CSR according to a noncontiguous owner map.
/// Collective contract: root alone supplies matrix and assignment; all others
/// supply None. The root preflights all per-rank lengths and numeric values
/// before transfer; per-target CSR materialization is strictly sequential.
/// All ranks participate in an error collective before *each* point-to-point
/// transfer so a failed chunk construction cannot strand a receiver.
/// This changes no production contiguous MPI input or solver routing.
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
    let started = Instant::now();
    let wrong_args = if rank == 0 {
        root_matrix.is_none() || root_assignment.is_none()
    } else {
        root_matrix.is_some() || root_assignment.is_some()
    };
    if mpi.all_reduce_sum_u64(u64::from(wrong_args)) != 0 {
        return Err(invalid(
            "root-only CSR and ownership argument contract violated",
        ));
    }
    // Root checks all payload sizes, owner labels, finite values and columns
    // once. This allocates only O(ranks) counts, never a vector of CSR chunks.
    let preflight = if rank == 0 {
        Some((|| -> io::Result<_> {
            let matrix = root_matrix.ok_or_else(|| invalid("missing root matrix"))?;
            let owners = root_assignment.ok_or_else(|| invalid("missing root assignment"))?;
            if owners.rank_count() != size as u32 {
                return Err(invalid("assignment MPI rank count mismatch"));
            }
            let counts = preflight_chunk_counts(matrix, owners)?;
            Ok((counts, matrix.nnz() as u64))
        })())
    } else {
        None
    };
    if mpi.all_reduce_sum_u64(u64::from(preflight.as_ref().is_some_and(Result::is_err))) != 0 {
        if let Some(Err(err)) = &preflight {
            eprintln!("K7-A root CSR preflight failed: {err}");
        }
        return Err(invalid("root streamed CSR preflight failed"));
    }
    let root_info = preflight.transpose()?;
    let mut header = [0u64; 2];
    if rank == 0 {
        let owners = root_assignment.ok_or_else(|| invalid("missing root assignment"))?;
        header = [
            owners.global_dofs(),
            root_info
                .as_ref()
                .ok_or_else(|| invalid("missing root preflight"))?
                .1,
        ];
    }
    world.process_at_rank(0).broadcast_into(&mut header[..]);
    let n = usize::try_from(header[0]).map_err(|_| invalid("global DOF usize overflow"))?;
    if n == 0 || n > i32::MAX as usize || header[1] > i32::MAX as u64 {
        return Err(invalid("global CSR exceeds root-scatter MPI count limits"));
    }
    let mut owners = vec![0u32; n];
    if let Some(root_owners) = root_assignment {
        owners.copy_from_slice(root_owners.owners());
    }
    world.process_at_rank(0).broadcast_into(&mut owners[..]);
    let assignment = PartitionAssignment::from_owners(size as u32, owners)
        .map_err(|_| invalid("invalid broadcast owner map"));
    if mpi.all_reduce_sum_u64(u64::from(assignment.is_err())) != 0 {
        return Err(invalid("invalid broadcast owner map on at least one rank"));
    }
    let assignment = assignment?;

    // Only O(ranks) nnz lengths are broadcast. Every receiver can size buffers
    // without trusting a potentially malformed point-to-point header.
    let mut announced_nnz = vec![0u64; size as usize];
    if let Some((counts, _)) = &root_info {
        for (dst, &c) in announced_nnz.iter_mut().zip(counts) {
            *dst = c.nnz as u64;
        }
    }
    world
        .process_at_rank(0)
        .broadcast_into(&mut announced_nnz[..]);
    let owned_count = assignment
        .owners()
        .iter()
        .filter(|&&r| r == rank as u32)
        .count();
    let totals_ok = announced_nnz.iter().copied().sum::<u64>() == header[1];
    let layout_ok = owned_count > 0
        && owned_count < i32::MAX as usize
        && announced_nnz.iter().all(|&count| count <= i32::MAX as u64)
        && totals_ok;
    if mpi.all_reduce_sum_u64(u64::from(!layout_ok)) != 0 {
        return Err(invalid("broadcast per-rank CSR counts rejected"));
    }

    let mut received_rows: Option<AssignmentOwnedCsrRows> = None;
    // One collective gate, one blocking transfer, one collective check per
    // destination. Only its recipient allocates a non-root receive buffer.
    for target in 1..size {
        let outgoing = if rank == 0 {
            let matrix = root_matrix.ok_or_else(|| invalid("missing root matrix"))?;
            let counts = &root_info
                .as_ref()
                .ok_or_else(|| invalid("missing preflight"))?
                .0;
            Some(build_checked_chunk(
                matrix,
                &assignment,
                target as u32,
                counts[target as usize],
            ))
        } else {
            None
        };
        if mpi.all_reduce_sum_u64(u64::from(outgoing.as_ref().is_some_and(Result::is_err))) != 0 {
            if let Some(Err(err)) = &outgoing {
                eprintln!("K7-A per-destination CSR build failed: {err}");
            }
            return Err(invalid("root streaming chunk build rejected"));
        }
        let outgoing = outgoing.transpose()?;
        let mut bad_receive = false;
        if rank == 0 {
            let chunk = outgoing.ok_or_else(|| invalid("missing root streaming chunk"))?;
            let dest = world.process_at_rank(target);
            let counts = [chunk.owned_globals.len() as u64, chunk.values.len() as u64];
            dest.send(&counts[..]);
            dest.send(&chunk.owned_globals[..]);
            dest.send(&chunk.row_ptr[..]);
            dest.send(&chunk.global_col_idx[..]);
            dest.send(&chunk.values[..]);
            // `chunk` drops at the end of this branch BEFORE the next target.
        } else if rank == target {
            let from = world.process_at_rank(0);
            let mut counts = [0u64; 2];
            from.receive_into(&mut counts[..]);
            let nnz = announced_nnz[target as usize] as usize;
            let mut chunk = AssignmentOwnedCsrRows {
                global_dofs: header[0],
                owned_globals: vec![0u64; owned_count],
                row_ptr: vec![0u32; owned_count + 1],
                global_col_idx: vec![0u64; nnz],
                values: vec![0f64; nnz],
            };
            from.receive_into(&mut chunk.owned_globals[..]);
            from.receive_into(&mut chunk.row_ptr[..]);
            from.receive_into(&mut chunk.global_col_idx[..]);
            from.receive_into(&mut chunk.values[..]);
            bad_receive = counts != [owned_count as u64, nnz as u64]
                || chunk.validate(&assignment, target as u32).is_err();
            received_rows = Some(chunk);
        }
        if mpi.all_reduce_sum_u64(u64::from(bad_receive)) != 0 {
            return Err(invalid(
                "streamed CSR transfer failed collective validation",
            ));
        }
    }

    // Root's own rows are materialized LAST: it never holds its own chunk
    // simultaneously with a destination chunk.
    let root_chunk = if rank == 0 {
        let matrix = root_matrix.ok_or_else(|| invalid("missing root matrix"))?;
        let counts = &root_info
            .as_ref()
            .ok_or_else(|| invalid("missing preflight"))?
            .0;
        Some(build_checked_chunk(matrix, &assignment, 0, counts[0]))
    } else {
        None
    };
    if mpi.all_reduce_sum_u64(u64::from(root_chunk.as_ref().is_some_and(Result::is_err))) != 0 {
        return Err(invalid("root owned CSR build rejected"));
    }
    let rows = if rank == 0 {
        root_chunk
            .transpose()?
            .ok_or_else(|| invalid("root rows missing"))?
    } else {
        received_rows.ok_or_else(|| invalid("rank rows missing"))?
    };
    if mpi.all_reduce_sum_u64(u64::from(rows.validate(&assignment, rank as u32).is_err())) != 0 {
        return Err(invalid("distributed noncontiguous CSR validation failed"));
    }
    let (root_peak, root_total) = if let Some((counts, _)) = &root_info {
        let mut max_bytes = 0u64;
        let mut sum_bytes = 0u64;
        for &c in counts {
            let bytes = chunk_payload_bytes(c)?;
            max_bytes = max_bytes.max(bytes);
            sum_bytes = sum_bytes
                .checked_add(bytes)
                .ok_or_else(|| invalid("root payload sum overflow"))?;
        }
        (max_bytes, sum_bytes)
    } else {
        (0, 0)
    };
    let stats = AssignmentScatterStats {
        global_nnz: header[1],
        local_nnz: rows.values.len(),
        local_owned_bytes: std::mem::size_of_val(rows.owned_globals.as_slice())
            + std::mem::size_of_val(rows.row_ptr.as_slice())
            + std::mem::size_of_val(rows.global_col_idx.as_slice())
            + std::mem::size_of_val(rows.values.as_slice()),
        scatter_ns: started.elapsed().as_nanos(),
        root_full_csr_before_drop: rank == 0,
        root_peak_streamed_chunk_bytes_estimate: root_peak,
        root_k6_all_chunks_bytes_estimate: root_total,
        max_concurrent_root_chunks: u32::from(rank == 0),
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
    fn counted_chunks_match_exact_noncontiguous_rows_and_original_coefficients() {
        let matrix = matrix();
        let owners = PartitionAssignment::from_owners(2, vec![0, 1, 0, 1, 0]).unwrap();
        let counts = preflight_chunk_counts(&matrix, &owners).unwrap();
        assert_eq!((counts[0].rows, counts[0].nnz), (3, 6));
        assert_eq!((counts[1].rows, counts[1].nnz), (2, 4));
        let root = build_checked_chunk(&matrix, &owners, 0, counts[0]).unwrap();
        let remote = build_checked_chunk(&matrix, &owners, 1, counts[1]).unwrap();
        assert_eq!(root.owned_globals, [0, 2, 4]);
        assert_eq!(remote.owned_globals, [1, 3]);
        assert_eq!(root.values, [4., 1., 3., 6., 1., 9.]);
        assert_eq!(remote.values, [5., 2., 1., 7.]);
        assert_eq!(root.values.len() + remote.values.len(), matrix.nnz());
    }

    #[test]
    fn streaming_peak_chunk_is_smaller_than_k6_all_chunk_footprint() {
        let counts = preflight_chunk_counts(
            &matrix(),
            &PartitionAssignment::from_owners(2, vec![0, 1, 0, 1, 0]).unwrap(),
        )
        .unwrap();
        let amounts = counts
            .iter()
            .copied()
            .map(chunk_payload_bytes)
            .collect::<io::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(amounts, [8 * 3 + 4 * 4 + 16 * 6, 8 * 2 + 4 * 3 + 16 * 4]);
        assert!(amounts.iter().sum::<u64>() > amounts.iter().copied().max().unwrap());
    }

    #[test]
    fn preflight_rejects_mismatched_matrix_and_assignment() {
        let matrix = matrix();
        let owners = PartitionAssignment::from_owners(2, vec![0, 1, 0, 1, 0]).unwrap();
        let bad =
            Csr32Matrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1., 2., 3.]).unwrap();
        assert!(preflight_chunk_counts(&bad, &owners).is_err());
        assert!(preflight_chunk_counts(&matrix, &owners).is_ok());
    }
}
