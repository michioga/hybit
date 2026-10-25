//! K7-C: root-only global owner map; nonroot ranks keep only owners of
//! their locally referenced remote columns. Original K6/K7-B path unchanged.
use crate::PartitionAssignment;
use crate::mpi_assignment::{AssignmentOwnedCsrRows, SparseOwnerDirectory};
use crate::mpi_assignment_scatter::{
    AssignmentScatterStats, build_checked_chunk, chunk_payload_bytes, preflight_chunk_counts,
};
use crate::mpi_backend::MpiRuntime;
use hybit_matrix::Csr32Matrix;
use mpi::traits::*;
use std::collections::BTreeMap;
use std::io;
use std::time::Instant;

fn invalid(s: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, s)
}

/// Sorted, unique referenced nonlocal global columns with their owning ranks.
/// The root computes this from its authoritative global assignment; receivers
/// never allocate an O(global_dofs) owner array.
fn remote_directory(
    assignment: &PartitionAssignment,
    rows: &AssignmentOwnedCsrRows,
    rank: u32,
) -> io::Result<(Vec<u64>, Vec<u32>)> {
    let mut remote = BTreeMap::<u64, u32>::new();
    for &column in &rows.global_col_idx {
        let owner = assignment
            .owner_of(column)
            .map_err(|_| invalid("root CSR references unowned column"))?;
        if owner != rank {
            remote.insert(column, owner);
        }
    }
    Ok(remote.into_iter().unzip())
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SparseScatterStats {
    pub distribution: AssignmentScatterStats,
    pub owner_entries: usize,
    pub owner_bytes: usize,
    pub root_owner_bytes: usize,
}

/// Requires original CSR and original global owner assignment only on rank 0.
/// Rank-local retained state: Owned CSR + remote referenced DOF -> rank map.
/// Collective/point-to-point sequence is identical across all ranks. All
/// potential invalid payload lengths are preflighted before blocking receives.
pub fn scatter_sparse_owners_from_root(
    mpi: &MpiRuntime,
    root_matrix: Option<&Csr32Matrix>,
    root_assignment: Option<&PartitionAssignment>,
) -> io::Result<(
    SparseOwnerDirectory,
    AssignmentOwnedCsrRows,
    SparseScatterStats,
)> {
    let rank = mpi.rank();
    let size = mpi.size();
    if size < 1 || rank < 0 {
        return Err(invalid("invalid MPI rank/count"));
    }
    let world = mpi::topology::SimpleCommunicator::world();
    let started = Instant::now();
    let bad_args = (rank == 0) != (root_matrix.is_some() && root_assignment.is_some())
        || (rank != 0 && (root_matrix.is_some() || root_assignment.is_some()));
    if mpi.all_reduce_sum_u64(u64::from(bad_args)) != 0 {
        return Err(invalid("root-only sparse scatter arguments inconsistent"));
    }
    let root_info = if rank == 0 {
        Some((|| -> io::Result<_> {
            let matrix = root_matrix.ok_or_else(|| invalid("missing root matrix"))?;
            let owners = root_assignment.ok_or_else(|| invalid("missing root owners"))?;
            if owners.rank_count() != size as u32 {
                return Err(invalid("global owner count mismatches MPI size"));
            }
            preflight_chunk_counts(matrix, owners)
        })())
    } else {
        None
    };
    if mpi.all_reduce_sum_u64(u64::from(root_info.as_ref().is_some_and(Result::is_err))) != 0 {
        return Err(invalid("sparse root preflight rejected"));
    }
    let root_counts = root_info.transpose()?;
    let mut header = [0u64; 2];
    let mut row_counts = vec![0u64; size as usize];
    let mut nnz_counts = vec![0u64; size as usize];
    if rank == 0 {
        let owners = root_assignment.ok_or_else(|| invalid("root owner map missing"))?;
        let counts = root_counts
            .as_ref()
            .ok_or_else(|| invalid("root counts missing"))?;
        header = [
            owners.global_dofs(),
            root_matrix
                .ok_or_else(|| invalid("root CSR missing"))?
                .nnz() as u64,
        ];
        for (i, c) in counts.iter().enumerate() {
            row_counts[i] = c.rows as u64;
            nnz_counts[i] = c.nnz as u64;
        }
    }
    world.process_at_rank(0).broadcast_into(&mut header[..]);
    world.process_at_rank(0).broadcast_into(&mut row_counts[..]);
    world.process_at_rank(0).broadcast_into(&mut nnz_counts[..]);
    let own_rows = usize::try_from(row_counts[rank as usize]);
    let own_nnz = usize::try_from(nnz_counts[rank as usize]);
    let bad_counts = own_rows.as_ref().is_err()
        || own_nnz.as_ref().is_err()
        || header[0] == 0
        || header[0] > i32::MAX as u64
        || header[1] > i32::MAX as u64
        || row_counts.iter().sum::<u64>() != header[0]
        || nnz_counts.iter().sum::<u64>() != header[1]
        || row_counts.iter().any(|&v| v == 0 || v >= i32::MAX as u64)
        || nnz_counts.iter().any(|&v| v > i32::MAX as u64);
    if mpi.all_reduce_sum_u64(u64::from(bad_counts)) != 0 {
        return Err(invalid("sparse scatter row/nnz announcement invalid"));
    }
    let own_rows = own_rows.map_err(|_| invalid("local row count overflow"))?;
    let own_nnz = own_nnz.map_err(|_| invalid("local nnz count overflow"))?;
    let mut received: Option<(SparseOwnerDirectory, AssignmentOwnedCsrRows)> = None;
    for target in 1..size {
        let prepared = if rank == 0 {
            Some((|| -> io::Result<_> {
                let matrix = root_matrix.ok_or_else(|| invalid("root CSR missing"))?;
                let owners = root_assignment.ok_or_else(|| invalid("root owner map missing"))?;
                let counts = root_counts
                    .as_ref()
                    .ok_or_else(|| invalid("root counts missing"))?;
                let chunk =
                    build_checked_chunk(matrix, owners, target as u32, counts[target as usize])?;
                let (remote, labels) = remote_directory(owners, &chunk, target as u32)?;
                let dir = SparseOwnerDirectory::new(
                    header[0],
                    size as u32,
                    target as u32,
                    &chunk,
                    remote,
                    labels,
                )?;
                Ok((chunk, dir))
            })())
        } else {
            None
        };
        if mpi.all_reduce_sum_u64(u64::from(prepared.as_ref().is_some_and(Result::is_err))) != 0 {
            return Err(invalid("root sparse target preparation rejected"));
        }
        let prepared = prepared.transpose()?;
        // Header first, then a collective gate: a corrupt length can never
        // strand the root inside a blocking payload send.
        let mut local_error = false;
        let mut counts = [0u64; 3];
        if rank == 0 {
            let (_, dir) = prepared
                .as_ref()
                .ok_or_else(|| invalid("root target missing"))?;
            let (chunk, _) = prepared
                .as_ref()
                .ok_or_else(|| invalid("root chunk missing"))?;
            counts = [
                chunk.owned_globals.len() as u64,
                chunk.values.len() as u64,
                dir.entries() as u64,
            ];
            world.process_at_rank(target).send(&counts[..]);
        } else if rank == target {
            world.process_at_rank(0).receive_into(&mut counts[..]);
            local_error = counts[0] != own_rows as u64
                || counts[1] != own_nnz as u64
                || counts[2] > own_nnz as u64
                || counts[2] > i32::MAX as u64;
        }
        if mpi.all_reduce_sum_u64(u64::from(local_error)) != 0 {
            return Err(invalid(
                "sparse CSR/directory announced wire lengths invalid",
            ));
        }
        if rank == 0 {
            let (chunk, dir) = prepared.ok_or_else(|| invalid("missing root prepared chunk"))?;
            let peer = world.process_at_rank(target);
            let remotes = dir.remote_entries();
            peer.send(&chunk.owned_globals[..]);
            peer.send(&chunk.row_ptr[..]);
            peer.send(&chunk.global_col_idx[..]);
            peer.send(&chunk.values[..]);
            peer.send(remotes.0);
            peer.send(remotes.1);
        } else if rank == target {
            let peer = world.process_at_rank(0);
            let ghost_len = counts[2] as usize;
            let mut chunk = AssignmentOwnedCsrRows {
                global_dofs: header[0],
                owned_globals: vec![0u64; own_rows],
                row_ptr: vec![0u32; own_rows + 1],
                global_col_idx: vec![0u64; own_nnz],
                values: vec![0.0f64; own_nnz],
            };
            let mut remote_globals = vec![0u64; ghost_len];
            let mut remote_owners = vec![0u32; ghost_len];
            peer.receive_into(&mut chunk.owned_globals[..]);
            peer.receive_into(&mut chunk.row_ptr[..]);
            peer.receive_into(&mut chunk.global_col_idx[..]);
            peer.receive_into(&mut chunk.values[..]);
            peer.receive_into(&mut remote_globals[..]);
            peer.receive_into(&mut remote_owners[..]);
            match SparseOwnerDirectory::new(
                header[0],
                size as u32,
                rank as u32,
                &chunk,
                remote_globals,
                remote_owners,
            ) {
                Ok(dir) => received = Some((dir, chunk)),
                Err(_) => local_error = true,
            }
        }
        if mpi.all_reduce_sum_u64(u64::from(local_error)) != 0 {
            return Err(invalid("sparse CSR/directory transfer rejected"));
        }
    }
    let own = if rank == 0 {
        Some((|| -> io::Result<_> {
            let matrix = root_matrix.ok_or_else(|| invalid("root CSR missing"))?;
            let owners = root_assignment.ok_or_else(|| invalid("root owner map missing"))?;
            let counts = root_counts
                .as_ref()
                .ok_or_else(|| invalid("root counts missing"))?;
            let chunk = build_checked_chunk(matrix, owners, 0, counts[0])?;
            let (remote, labels) = remote_directory(owners, &chunk, 0)?;
            let dir = SparseOwnerDirectory::new(header[0], size as u32, 0, &chunk, remote, labels)?;
            Ok((dir, chunk))
        })())
    } else {
        None
    };
    if mpi.all_reduce_sum_u64(u64::from(own.as_ref().is_some_and(Result::is_err))) != 0 {
        return Err(invalid("root sparse owned chunk invalid"));
    }
    let (directory, rows) = if rank == 0 {
        own.transpose()?
            .ok_or_else(|| invalid("root sparse output missing"))?
    } else {
        received.ok_or_else(|| invalid("rank sparse output missing"))?
    };
    let local_invalid = directory.validate(&rows).is_err()
        || rows.owned_globals.len() != own_rows
        || rows.values.len() != own_nnz;
    if mpi.all_reduce_sum_u64(u64::from(local_invalid)) != 0 {
        return Err(invalid("sparse owner directory final validation failed"));
    }
    let (peak, all) = if let Some(counts) = &root_counts {
        let mut peak = 0u64;
        let mut sum = 0u64;
        for &c in counts {
            let b = chunk_payload_bytes(c)?;
            peak = peak.max(b);
            sum = sum
                .checked_add(b)
                .ok_or_else(|| invalid("root CSR estimate overflow"))?;
        }
        (peak, sum)
    } else {
        (0, 0)
    };
    let own_bytes = std::mem::size_of_val(rows.owned_globals.as_slice())
        + std::mem::size_of_val(rows.row_ptr.as_slice())
        + std::mem::size_of_val(rows.global_col_idx.as_slice())
        + std::mem::size_of_val(rows.values.as_slice());
    let owner_entries = directory.entries();
    let owner_bytes = directory.payload_bytes();
    Ok((
        directory,
        rows,
        SparseScatterStats {
            distribution: AssignmentScatterStats {
                global_nnz: header[1],
                local_nnz: own_nnz,
                local_owned_bytes: own_bytes,
                scatter_ns: started.elapsed().as_nanos(),
                root_full_csr_before_drop: rank == 0,
                root_peak_streamed_chunk_bytes_estimate: peak,
                root_k6_all_chunks_bytes_estimate: all,
                max_concurrent_root_chunks: u32::from(rank == 0),
            },
            owner_entries,
            owner_bytes,
            root_owner_bytes: if rank == 0 {
                header[0] as usize * std::mem::size_of::<u32>()
            } else {
                0
            },
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ghost_lookup_sparse_is_sorted_and_deduplicated() {
        let m = Csr32Matrix::new(
            4,
            4,
            vec![0, 2, 4, 6, 8],
            vec![0, 1, 1, 2, 2, 3, 0, 3],
            vec![1.; 8],
        )
        .unwrap();
        let assign = PartitionAssignment::from_owners(2, vec![0, 1, 0, 1]).unwrap();
        let rows = AssignmentOwnedCsrRows::from_global(&m, &assign, 0).unwrap();
        let (remote, owners) = remote_directory(&assign, &rows, 0).unwrap();
        assert_eq!(remote, vec![1, 3]);
        assert_eq!(owners, vec![1, 1]);
        let directory = SparseOwnerDirectory::new(4, 2, 0, &rows, remote, owners).unwrap();
        assert_eq!(directory.owner_of(1, &rows).unwrap(), 1);
        assert_eq!(directory.owner_of(2, &rows).unwrap(), 0);
    }
    #[test]
    fn sparse_directory_rejects_missing_and_wrong_owners() {
        let m = Csr32Matrix::new(
            4,
            4,
            vec![0, 2, 4, 6, 8],
            vec![0, 1, 1, 2, 2, 3, 0, 3],
            vec![1.; 8],
        )
        .unwrap();
        let assign = PartitionAssignment::from_owners(2, vec![0, 1, 0, 1]).unwrap();
        let rows = AssignmentOwnedCsrRows::from_global(&m, &assign, 0).unwrap();
        assert!(SparseOwnerDirectory::new(4, 2, 0, &rows, vec![1], vec![1]).is_err());
        assert!(SparseOwnerDirectory::new(4, 2, 0, &rows, vec![1, 3], vec![0, 1]).is_err());
        assert!(SparseOwnerDirectory::new(4, 2, 0, &rows, vec![3, 1], vec![1, 1]).is_err());
    }
}

#[cfg(test)]
mod k7d_tests {
    use super::*;

    #[test]
    fn repeated_columns_produce_one_sorted_remote_entry_each() {
        // Two columns repeat across all six CSR rows; rank 0 owns rows 0,2,4.
        let matrix = Csr32Matrix::new(
            6,
            6,
            vec![0, 2, 4, 6, 8, 10, 12],
            vec![1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5],
            vec![1.0; 12],
        )
        .unwrap();
        let assignment = PartitionAssignment::from_owners(2, vec![0, 1, 0, 1, 0, 1]).unwrap();
        let rows = AssignmentOwnedCsrRows::from_global(&matrix, &assignment, 0).unwrap();
        let (globals, owners) = remote_directory(&assignment, &rows, 0).unwrap();
        assert_eq!(globals, vec![1, 5]);
        assert_eq!(owners, vec![1, 1]);
        SparseOwnerDirectory::new(6, 2, 0, &rows, globals, owners).unwrap();
    }
}
