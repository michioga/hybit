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

use crate::mpi_assignment::{AssignmentOwnedCsrRows, SparseOwnerDirectory};
use crate::mpi_backend::MpiRuntime;
use crate::PartitionAssignment;
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

/// K7-C original CSR bit-exact verifier for a sparse-owner MPI layout.
/// Only root sees the global ownership map. Remote ranks compare streamed
/// reference frames against their already-distributed CSR.
pub fn verify_sparse_assignment_scatter_exact(
    mpi: &MpiRuntime,
    root_matrix: Option<&Csr32Matrix>,
    root_assignment: Option<&PartitionAssignment>,
    directory: &SparseOwnerDirectory,
    received: &AssignmentOwnedCsrRows,
) -> io::Result<AssignmentExactVerifyStats> {
    let started = Instant::now();
    let size = mpi.size();
    let rank = mpi.rank();
    if size <= 0 || rank < 0 {
        return Err(invalid("invalid MPI communicator"));
    }
    let world = mpi::topology::SimpleCommunicator::world();
    let shape_bad = directory.ranks != size as u32
        || directory.rank != rank as u32
        || directory.validate(received).is_err()
        || (rank == 0) != root_matrix.is_some()
        || (rank == 0) != root_assignment.is_some();
    if mpi.all_reduce_sum_u64(u64::from(shape_bad)) != 0 {
        return Err(invalid("sparse verification input contract invalid"));
    }
    let mut n_values = vec![0u64; size as usize];
    world.all_gather_into(&directory.global_dofs, &mut n_values[..]);
    if n_values.iter().any(|&n| n != directory.global_dofs) {
        return Err(invalid(
            "sparse directory global dimension disagrees across ranks",
        ));
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
        if matrix.nrows() != matrix.ncols() || matrix.nrows() as u64 != directory.global_dofs {
            true
        } else {
            for (global, &owner) in root_assignment
                .ok_or_else(|| invalid("missing root owners"))?
                .owners()
                .iter()
                .enumerate()
            {
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
            root_assignment.ok_or_else(|| invalid("missing root owners"))?,
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
            let expected_rows: Vec<usize> = root_assignment
                .ok_or_else(|| invalid("missing root owners"))?
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
    // Independently reconstruct remote-column -> owner labels from ORIGINAL
    // CSR and authoritative root assignment, then compare exact bit patterns.
    for target in 1..size {
        let mut mismatched = false;
        if rank == 0 {
            let matrix = root_matrix.ok_or_else(|| invalid("root matrix missing"))?;
            let owners = root_assignment.ok_or_else(|| invalid("root owners missing"))?;
            let mut remote = std::collections::BTreeSet::new();
            for (row, &owner) in owners.owners().iter().enumerate() {
                if owner == target as u32 {
                    for &column in &matrix.col_idx()
                        [matrix.row_ptr()[row] as usize..matrix.row_ptr()[row + 1] as usize]
                    {
                        if owners
                            .owner_of(u64::from(column))
                            .map_err(|_| invalid("invalid original column owner"))?
                            != target as u32
                        {
                            remote.insert(u64::from(column));
                        }
                    }
                }
            }
            let target_peer = world.process_at_rank(target);
            let count = [remote.len() as u64];
            target_peer.send(&count[..]);
            for group in remote.iter().copied().collect::<Vec<_>>().chunks(FRAME_LEN) {
                let labels = group
                    .iter()
                    .map(|&g| owners.owner_of(g).map_err(|_| invalid("missing owner")))
                    .collect::<io::Result<Vec<_>>>()?;
                target_peer.send(group);
                target_peer.send(&labels[..]);
            }
        } else if rank == target {
            let source = world.process_at_rank(0);
            let mut count = [0u64];
            source.receive_into(&mut count[..]);
            let (expected_globals, expected_owners) = directory.remote_entries();
            mismatched |= count[0] != expected_globals.len() as u64;
            let mut slot = 0usize;
            let mut remote_globals = vec![0u64; FRAME_LEN];
            let mut remote_owners = vec![0u32; FRAME_LEN];
            while slot < count[0] as usize {
                let len = ((count[0] as usize) - slot).min(FRAME_LEN);
                source.receive_into(&mut remote_globals[..len]);
                source.receive_into(&mut remote_owners[..len]);
                if slot + len <= expected_globals.len() {
                    mismatched |= remote_globals[..len] != expected_globals[slot..slot + len]
                        || remote_owners[..len] != expected_owners[slot..slot + len];
                } else {
                    mismatched = true;
                }
                slot += len;
            }
        }
        if mpi.all_reduce_sum_u64(u64::from(mismatched)) != 0 {
            return Err(invalid(
                "sparse remote owner directory differs from root original",
            ));
        }
    }
    let verified_nnz = lengths.chunks_exact(2).map(|pair| pair[1]).sum();
    Ok(AssignmentExactVerifyStats {
        verified_rows: directory.global_dofs,
        verified_nnz,
        bitwise_coefficients: true,
        verify_ns: started.elapsed().as_nanos(),
        reference_frame_bytes: FRAME_LEN * std::mem::size_of::<u64>(),
    })
}
