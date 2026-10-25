//! H1-C3K7-B: optional, exact reference verification of noncontiguous owned CSR.
//!
//! Rank zero alone retains the ORIGINAL CSR, which is the authoritative source.
//! It independently regenerates expected owned row IDs, row pointers, global
//! columns, and the f64 IEEE-754 bit patterns for each destination. The owner
//! rank compares these references with its *already distributed* CSR. This is
//! an exact comparison, not a probabilistic fingerprint or a SpMV self-check.
//!
//! Verification is diagnostic and collective. It must be called before the
//! root drops the original matrix. All MPI payloads are bounded by FRAME_LEN.
//! No production solver/preconditioner path is modified by this module.

use crate::PartitionAssignment;
use crate::mpi_assignment::AssignmentOwnedCsrRows;
use crate::mpi_backend::MpiRuntime;
use hybit_matrix::Csr32Matrix;
use mpi::traits::*;
use std::io;
use std::time::Instant;

const FRAME_LEN: usize = 16_384;

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Exact comparison to ORIGINAL CSR in the original row/column traversal order.
/// No `from_global` or scatter construction helper is invoked by the reference.
fn compare_local_exact(
    matrix: &Csr32Matrix,
    assignment: &PartitionAssignment,
    rank: u32,
    received: &AssignmentOwnedCsrRows,
) -> io::Result<()> {
    received.validate(assignment, rank)?;
    if matrix.nrows() != matrix.ncols() || matrix.nrows() as u64 != assignment.global_dofs() {
        return Err(invalid("reference matrix and ownership dimensions differ"));
    }
    let mut row_slot = 0usize;
    let mut flat_slot = 0usize;
    if received.row_ptr[0] != 0 {
        return Err(invalid("owned CSR row pointer must start at zero"));
    }
    for (global, &owner) in assignment.owners().iter().enumerate() {
        if owner != rank {
            continue;
        }
        if received.owned_globals[row_slot] != global as u64
            || received.row_ptr[row_slot] as usize != flat_slot
        {
            return Err(invalid(
                "owned CSR row identity/order/pointer differs from root",
            ));
        }
        let first = matrix.row_ptr()[global] as usize;
        let last = matrix.row_ptr()[global + 1] as usize;
        for position in first..last {
            if received.global_col_idx[flat_slot] != u64::from(matrix.col_idx()[position]) {
                return Err(invalid("owned CSR global column differs from root"));
            }
            if received.values[flat_slot].to_bits() != matrix.values()[position].to_bits() {
                return Err(invalid("owned CSR f64 coefficient bits differ from root"));
            }
            flat_slot += 1;
        }
        row_slot += 1;
        if received.row_ptr[row_slot] as usize != flat_slot {
            return Err(invalid("owned CSR row end pointer differs from root"));
        }
    }
    if row_slot != received.owned_globals.len() || flat_slot != received.values.len() {
        return Err(invalid("root owned row/nnz totals differ from local CSR"));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AssignmentExactVerifyStats {
    pub verified_rows: u64,
    pub verified_nnz: u64,
    /// Exact checking includes column indices and f64::to_bits(), not hashes.
    pub bitwise_coefficients: bool,
    /// Elapsed wall time on each rank; caller may take MPI maximum.
    pub verify_ns: u128,
    /// Maximum single reference frame size (bytes), NOT the process RSS.
    pub reference_frame_bytes: usize,
}

/// All ranks collectively verify scattered CSR against rank 0's original CSR.
///
/// Root must supply `Some(original)` and other ranks must supply `None`.
/// All ranks supply the same already-broadcast owner map and their own rows.
/// The original matrix is not reconstructed from the received data.
///
/// Expected lengths are collectively checked before point-to-point traffic.
/// Root streams independently computed reference data for one rank at a time;
/// each destination checks the received data against the *already held* rows.
/// A mismatch is propagated to all ranks before the next destination begins.
/// This debug/validation facility intentionally adds network and CPU overhead.
/// MPI transport must be called from the initializing thread.
pub fn verify_assignment_scatter_exact(
    mpi: &MpiRuntime,
    root_matrix: Option<&Csr32Matrix>,
    assignment: &PartitionAssignment,
    received: &AssignmentOwnedCsrRows,
) -> io::Result<AssignmentExactVerifyStats> {
    let started = Instant::now();
    let size = mpi.size();
    let rank = mpi.rank();
    if size <= 0 || rank < 0 {
        return Err(invalid("invalid MPI communicator"));
    }
    let world = mpi::topology::SimpleCommunicator::world();
    let shape_bad = assignment.rank_count() != size as u32
        || received.validate(assignment, rank as u32).is_err()
        || (rank == 0) != root_matrix.is_some();
    if mpi.all_reduce_sum_u64(u64::from(shape_bad)) != 0 {
        return Err(invalid("verification input/ownership contract invalid"));
    }
    // K7-A broadcasts the owner map. The diagnostic nevertheless verifies
    // exact consensus independently before any point-to-point reference data.
    let mut global_lengths = vec![0u64; size as usize];
    world.all_gather_into(&assignment.global_dofs(), &mut global_lengths[..]);
    if global_lengths
        .iter()
        .any(|&n| n != assignment.global_dofs())
    {
        return Err(invalid("MPI ranks disagree on global owner-map length"));
    }
    let mut map_mismatch = false;
    for first in (0..assignment.owners().len()).step_by(FRAME_LEN) {
        let end = first
            .saturating_add(FRAME_LEN)
            .min(assignment.owners().len());
        let mut reference = vec![0u32; end - first];
        if rank == 0 {
            reference.copy_from_slice(&assignment.owners()[first..end]);
        }
        world.process_at_rank(0).broadcast_into(&mut reference[..]);
        map_mismatch |= reference.as_slice() != &assignment.owners()[first..end];
    }
    if mpi.all_reduce_sum_u64(u64::from(map_mismatch)) != 0 {
        return Err(invalid("nonidentical owner maps across MPI ranks"));
    }
    let local_lengths = [
        received.owned_globals.len() as u64,
        received.values.len() as u64,
    ];
    let mut lengths = vec![0u64; 2 * size as usize];
    world.all_gather_into(&local_lengths[..], &mut lengths[..]);

    // Independently walk the source CSR to derive expected per-rank counts.
    let root_check = if rank == 0 {
        let matrix = root_matrix.ok_or_else(|| invalid("root matrix missing"))?;
        let mut rows_per_rank = vec![0u64; size as usize];
        let mut nnz_per_rank = vec![0u64; size as usize];
        if matrix.nrows() != matrix.ncols() || matrix.nrows() as u64 != assignment.global_dofs() {
            true
        } else {
            for (global, &owner) in assignment.owners().iter().enumerate() {
                let p = owner as usize;
                rows_per_rank[p] += 1;
                nnz_per_rank[p] +=
                    u64::from(matrix.row_ptr()[global + 1] - matrix.row_ptr()[global]);
            }
            (0..size as usize).any(|p| {
                rows_per_rank[p] != lengths[2 * p] || nnz_per_rank[p] != lengths[2 * p + 1]
            })
        }
    } else {
        false
    };
    if mpi.all_reduce_sum_u64(u64::from(root_check)) != 0 {
        return Err(invalid(
            "original CSR and distributed per-rank row/nnz totals disagree",
        ));
    }

    // Root's own distributed rows are checked without transferring a second copy.
    let root_bad = rank == 0
        && compare_local_exact(
            root_matrix.ok_or_else(|| invalid("root matrix missing"))?,
            assignment,
            0,
            received,
        )
        .is_err();
    if mpi.all_reduce_sum_u64(u64::from(root_bad)) != 0 {
        return Err(invalid("rank-zero owned CSR differs bitwise from original"));
    }

    // Each remote rank verifies its distributed rows. Root creates references
    // DIRECTLY from the source CSR, not from scatter's chunk builder.
    for target in 1..size {
        let mut mismatch = false;
        if rank == 0 {
            let matrix = root_matrix.ok_or_else(|| invalid("root matrix missing"))?;
            let expected_rows: Vec<usize> = assignment
                .owners()
                .iter()
                .enumerate()
                .filter_map(|(row, &owner)| (owner == target as u32).then_some(row))
                .collect();
            let peer = world.process_at_rank(target);

            for group in expected_rows.chunks(FRAME_LEN) {
                let ids: Vec<u64> = group.iter().map(|&row| row as u64).collect();
                peer.send(&ids[..]);
            }
            let mut prefix = 0u32;
            for first in (0..=expected_rows.len()).step_by(FRAME_LEN) {
                let end = first.saturating_add(FRAME_LEN).min(expected_rows.len() + 1);
                let mut ptr = Vec::with_capacity(end - first);
                for slot in first..end {
                    if slot != 0 {
                        let global = expected_rows[slot - 1];
                        prefix += matrix.row_ptr()[global + 1] - matrix.row_ptr()[global];
                    }
                    ptr.push(prefix);
                }
                peer.send(&ptr[..]);
            }
            let columns = expected_rows.iter().flat_map(|&row| {
                let first = matrix.row_ptr()[row] as usize;
                let last = matrix.row_ptr()[row + 1] as usize;
                matrix.col_idx()[first..last].iter().map(|&c| u64::from(c))
            });
            let mut columns = columns;
            for first in (0..lengths[2 * target as usize + 1] as usize).step_by(FRAME_LEN) {
                let count = (lengths[2 * target as usize + 1] as usize - first).min(FRAME_LEN);
                let frame: Vec<u64> = columns.by_ref().take(count).collect();
                peer.send(&frame[..]);
            }
            let bits = expected_rows.iter().flat_map(|&row| {
                let first = matrix.row_ptr()[row] as usize;
                let last = matrix.row_ptr()[row + 1] as usize;
                matrix.values()[first..last]
                    .iter()
                    .map(|value| value.to_bits())
            });
            let mut bits = bits;
            for first in (0..lengths[2 * target as usize + 1] as usize).step_by(FRAME_LEN) {
                let count = (lengths[2 * target as usize + 1] as usize - first).min(FRAME_LEN);
                let frame: Vec<u64> = bits.by_ref().take(count).collect();
                peer.send(&frame[..]);
            }
        } else if rank == target {
            let peer = world.process_at_rank(0);
            let mut frame_u64 = vec![0u64; FRAME_LEN];
            let mut frame_u32 = vec![0u32; FRAME_LEN];
            for (first, chunk) in received.owned_globals.chunks(FRAME_LEN).enumerate() {
                let first = first * FRAME_LEN;
                let len = chunk.len();
                peer.receive_into(&mut frame_u64[..len]);
                mismatch |= frame_u64[..len] != received.owned_globals[first..first + len];
            }
            for (first, chunk) in received.row_ptr.chunks(FRAME_LEN).enumerate() {
                let first = first * FRAME_LEN;
                let len = chunk.len();
                peer.receive_into(&mut frame_u32[..len]);
                mismatch |= frame_u32[..len] != received.row_ptr[first..first + len];
            }
            for (first, chunk) in received.global_col_idx.chunks(FRAME_LEN).enumerate() {
                let first = first * FRAME_LEN;
                let len = chunk.len();
                peer.receive_into(&mut frame_u64[..len]);
                mismatch |= frame_u64[..len] != received.global_col_idx[first..first + len];
            }
            for chunk in received.values.chunks(FRAME_LEN) {
                peer.receive_into(&mut frame_u64[..chunk.len()]);
                mismatch |= frame_u64[..chunk.len()]
                    .iter()
                    .zip(chunk)
                    .any(|(&expected, actual)| expected != actual.to_bits());
            }
        }
        if mpi.all_reduce_sum_u64(u64::from(mismatch)) != 0 {
            return Err(invalid(
                "exact distributed CSR reference verification mismatch",
            ));
        }
    }
    let verified_nnz = lengths.as_chunks::<2>().0.iter().map(|pair| pair[1]).sum();
    Ok(AssignmentExactVerifyStats {
        verified_rows: assignment.global_dofs(),
        verified_nnz,
        bitwise_coefficients: true,
        verify_ns: started.elapsed().as_nanos(),
        reference_frame_bytes: FRAME_LEN * std::mem::size_of::<u64>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> (Csr32Matrix, PartitionAssignment) {
        let matrix = Csr32Matrix::new(
            5,
            5,
            vec![0, 2, 4, 7, 9, 10],
            vec![0, 1, 1, 2, 0, 2, 3, 1, 3, 4],
            vec![4.0, -0.0, 5.0, 2.0, 3.0, 6.0, 1.0, 1.0, 7.0, 9.0],
        )
        .unwrap();
        let owners = PartitionAssignment::from_owners(2, vec![0, 1, 0, 1, 0]).unwrap();
        (matrix, owners)
    }

    #[test]
    fn exact_noncontiguous_original_coefficients_pass() {
        let (matrix, owners) = sample();
        for rank in 0..2 {
            let rows = AssignmentOwnedCsrRows::from_global(&matrix, &owners, rank).unwrap();
            compare_local_exact(&matrix, &owners, rank, &rows).unwrap();
        }
    }

    #[test]
    fn coefficient_single_bit_flip_rejected() {
        let (matrix, owners) = sample();
        let mut rows = AssignmentOwnedCsrRows::from_global(&matrix, &owners, 0).unwrap();
        rows.values[0] = f64::from_bits(rows.values[0].to_bits() ^ 1);
        assert!(rows.validate(&owners, 0).is_ok());
        assert!(compare_local_exact(&matrix, &owners, 0, &rows).is_err());
    }

    #[test]
    fn negative_zero_bit_flip_rejected() {
        let (matrix, owners) = sample();
        let mut rows = AssignmentOwnedCsrRows::from_global(&matrix, &owners, 0).unwrap();
        rows.values[1] = 0.0;
        assert!(rows.validate(&owners, 0).is_ok());
        assert!(compare_local_exact(&matrix, &owners, 0, &rows).is_err());
    }

    #[test]
    fn changed_column_and_row_structure_rejected() {
        let (matrix, owners) = sample();
        let mut rows = AssignmentOwnedCsrRows::from_global(&matrix, &owners, 0).unwrap();
        rows.global_col_idx[0] = 2;
        assert!(compare_local_exact(&matrix, &owners, 0, &rows).is_err());
        let mut rows = AssignmentOwnedCsrRows::from_global(&matrix, &owners, 0).unwrap();
        rows.row_ptr[1] -= 1;
        assert!(compare_local_exact(&matrix, &owners, 0, &rows).is_err());
        let mut rows = AssignmentOwnedCsrRows::from_global(&matrix, &owners, 0).unwrap();
        rows.owned_globals[1] = 3;
        assert!(compare_local_exact(&matrix, &owners, 0, &rows).is_err());
    }
}
