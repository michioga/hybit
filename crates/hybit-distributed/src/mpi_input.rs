//! G8-C2: construct a rank-local CSR and reciprocal HaloPlan from owned rows.
//!
//! Each rank holds only its own CSR rows (global u64 columns), the O(P)
//! contiguous ownership offsets, and the distinct ghost indices. Requests
//! for remote values are exchanged with MPI_Alltoallv. No replicated global
//! CSR and no global ownership-length vector are required.
//!
//! Collective contract: all ranks call `prepare_owned_rows` together. The
//! partition must be identical on every rank. Input failures are propagated
//! to all ranks before entering MPI_Alltoallv. Large message counts beyond
//! MPI's i32 Count limit are rejected, not silently truncated.

use crate::mpi_backend::MpiRuntime;
use crate::{ContiguousPartition, HaloPeerPlan, HaloPlan, RankLocalCsr};
use mpi::datatype::{Partition, PartitionMut};
use mpi::traits::*;
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::ops::Range;

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

/// CSR rows owned by one rank, with global column DOFs (64-bit).
/// `row_ptr` and `values` describe only `owned.end-owned.start` rows.
#[derive(Clone, Debug)]
pub struct OwnedCsrRows {
    pub global_dofs: u64,
    pub owned: Range<u64>,
    pub row_ptr: Vec<u32>,
    pub global_col_idx: Vec<u64>,
    pub values: Vec<f64>,
}

impl OwnedCsrRows {
    pub fn validate(&self) -> io::Result<()> {
        if self.global_dofs == 0
            || self.owned.start >= self.owned.end
            || self.owned.end > self.global_dofs
        {
            return Err(invalid("invalid locally owned global row range"));
        }
        let rows = usize::try_from(self.owned.end - self.owned.start)
            .map_err(|_| invalid("rank-owned row count exceeds usize"))?;
        if self.row_ptr.len() != rows + 1 || self.row_ptr.first() != Some(&0) {
            return Err(invalid("local CSR row_ptr shape/start mismatch"));
        }
        if self.row_ptr.windows(2).any(|pair| pair[0] > pair[1])
            || self.row_ptr.last().copied() != Some(self.global_col_idx.len() as u32)
            || self.global_col_idx.len() != self.values.len()
        {
            return Err(invalid("invalid local CSR row pointers or value lengths"));
        }
        if self.global_col_idx.len() > u32::MAX as usize {
            return Err(invalid("rank-local CSR nnz exceeds u32"));
        }
        if self.global_col_idx.iter().any(|&c| c >= self.global_dofs)
            || self.values.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "local CSR contains invalid column or nonfinite coefficient",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OwnedInputStats {
    pub owned_rows: usize,
    pub local_nnz: usize,
    pub ghosts: usize,
    pub peer_ranks: usize,
    pub sent_halo_values_per_spmv: usize,
    pub received_halo_values_per_spmv: usize,
    /// Only owned rows/CSR and extended-vector indices; not MPI/process RSS.
    pub operator_and_halo_bytes_estimate: usize,
}

fn displacements(counts: &[i32]) -> io::Result<(Vec<i32>, usize)> {
    let mut next = 0i32;
    let mut offsets = Vec::with_capacity(counts.len());
    for &count in counts {
        if count < 0 {
            return Err(invalid("negative MPI request count"));
        }
        offsets.push(next);
        next = next
            .checked_add(count)
            .ok_or_else(|| invalid("MPI_Alltoallv total exceeds i32 Count range"))?;
    }
    Ok((offsets, next as usize))
}

/// Build the exact B1/B3-compatible halo plan from rank-local global-column CSR.
/// Only `partition.offsets()` (O(ranks)) is common metadata.
/// Communication is collective, including ranks with no ghost dependencies.
pub fn prepare_owned_rows(
    mpi: &MpiRuntime,
    partition: &ContiguousPartition,
    rows: &OwnedCsrRows,
) -> io::Result<(HaloPlan, RankLocalCsr, OwnedInputStats)> {
    let size = usize::try_from(mpi.size()).map_err(|_| invalid("invalid MPI size"))?;
    let rank = usize::try_from(mpi.rank()).map_err(|_| invalid("invalid MPI rank"))?;
    let input_check = rows.validate().and_then(|()| {
        let expected = partition
            .owned_range(rank as u32)
            .map_err(|_| invalid("partition rank invalid"))?;
        if partition.rank_count() as usize != size
            || rows.global_dofs != partition.global_dofs()
            || rows.owned != expected
        {
            Err(invalid("owned CSR metadata disagrees with MPI partition"))
        } else {
            Ok(())
        }
    });
    if mpi.all_reduce_sum_u64(u64::from(input_check.is_err())) != 0 {
        return Err(invalid("invalid input on at least one MPI rank"));
    }
    input_check?;

    // For each owner, request each remote DOF once, even if several rows
    // reference it. Sorting ensures matching B1/B3 send/receive order.
    let mut requests: Vec<BTreeSet<u64>> = vec![BTreeSet::new(); size];
    for &col in &rows.global_col_idx {
        if col < rows.owned.start || col >= rows.owned.end {
            let owner = partition
                .owner_of(col)
                .map_err(|_| invalid("global column has no owner"))?
                as usize;
            requests[owner].insert(col);
        }
    }
    let send_counts: Vec<i32> = requests
        .iter()
        .map(|items| i32::try_from(items.len()).map_err(|_| invalid("peer request count > i32")))
        .collect::<io::Result<_>>()?;
    let (send_displs, total_requests) = displacements(&send_counts)?;
    // No rank may return before the next collective.
    if mpi.all_reduce_sum_u64(u64::from(total_requests > i32::MAX as usize)) != 0 {
        return Err(invalid("MPI request buffer exceeds i32"));
    }
    let outgoing: Vec<u64> = requests.iter().flat_map(|r| r.iter().copied()).collect();
    let world = mpi::topology::SimpleCommunicator::world();
    let mut recv_counts = vec![0i32; size];
    world.all_to_all_into(&send_counts[..], &mut recv_counts[..]);
    let recv_layout = displacements(&recv_counts);
    if mpi.all_reduce_sum_u64(u64::from(recv_layout.is_err())) != 0 {
        return Err(invalid("MPI receive buffer exceeds Count range"));
    }
    let (recv_displs, total_incoming) = recv_layout?;
    let mut incoming = vec![0u64; total_incoming];
    {
        let send = Partition::new(&outgoing[..], &send_counts[..], &send_displs[..]);
        let mut recv = PartitionMut::new(&mut incoming[..], &recv_counts[..], &recv_displs[..]);
        world.all_to_all_varcount_into(&send, &mut recv);
    }

    // Incoming lists are other ranks' requests for our owned DOFs.
    // Assemble peer segments even when all traffic is one-way.
    let built = (|| -> io::Result<(HaloPlan, RankLocalCsr, OwnedInputStats)> {
        let owned_len = usize::try_from(rows.owned.end - rows.owned.start)
            .map_err(|_| invalid("owned row length overflow"))?;
        let mut ghost_globals = Vec::with_capacity(outgoing.len());
        let mut ghost_owners = Vec::with_capacity(outgoing.len());
        let mut ghost_slots = BTreeMap::<u64, u32>::new();
        for (owner, items) in requests.iter().enumerate() {
            for &global in items {
                let slot = u32::try_from(owned_len + ghost_globals.len())
                    .map_err(|_| invalid("extended index exceeds u32"))?;
                ghost_slots.insert(global, slot);
                ghost_globals.push(global);
                ghost_owners.push(owner as u32);
            }
        }
        let mut peers = Vec::new();
        let mut send_values = 0usize;
        for (owner, recv_request) in requests.iter().enumerate() {
            let offset = recv_displs[owner] as usize;
            let count = recv_counts[owner] as usize;
            let wanted_by_peer = &incoming[offset..offset + count];
            if wanted_by_peer.windows(2).any(|w| w[0] >= w[1])
                || wanted_by_peer
                    .iter()
                    .any(|&global| global < rows.owned.start || global >= rows.owned.end)
            {
                return Err(invalid("MPI peer requested out-of-order or non-owned DOFs"));
            }
            let recv_globals: Vec<u64> = recv_request.iter().copied().collect();
            if wanted_by_peer.is_empty() && recv_globals.is_empty() {
                continue;
            }
            let send_owned_indices: Vec<u32> = wanted_by_peer
                .iter()
                .map(|&g| {
                    u32::try_from(g - rows.owned.start).map_err(|_| invalid("send slot > u32"))
                })
                .collect::<io::Result<_>>()?;
            let recv_extended_indices: Vec<u32> = recv_globals
                .iter()
                .map(|g| {
                    ghost_slots
                        .get(g)
                        .copied()
                        .ok_or_else(|| invalid("missing ghost slot"))
                })
                .collect::<io::Result<_>>()?;
            send_values += wanted_by_peer.len();
            peers.push(HaloPeerPlan {
                peer: owner as u32,
                send_globals: wanted_by_peer.to_vec(),
                send_owned_indices,
                recv_globals,
                recv_extended_indices,
            });
        }
        let mut col_idx = Vec::with_capacity(rows.global_col_idx.len());
        for &global in &rows.global_col_idx {
            let local = if global >= rows.owned.start && global < rows.owned.end {
                u32::try_from(global - rows.owned.start)
                    .map_err(|_| invalid("local column index exceeds u32"))?
            } else {
                *ghost_slots
                    .get(&global)
                    .ok_or_else(|| invalid("missing remote column"))?
            };
            col_idx.push(local);
        }
        let extended_len = owned_len + ghost_globals.len();
        let stats = OwnedInputStats {
            owned_rows: owned_len,
            local_nnz: rows.values.len(),
            ghosts: ghost_globals.len(),
            peer_ranks: peers.len(),
            sent_halo_values_per_spmv: send_values,
            received_halo_values_per_spmv: ghost_globals.len(),
            operator_and_halo_bytes_estimate: rows.row_ptr.len() * 4
                + col_idx.len() * 4
                + rows.values.len() * 8
                + ghost_globals.len() * 8
                + ghost_owners.len() * 4
                + extended_len * 8,
        };
        let halo = HaloPlan {
            rank: rank as u32,
            owned: rows.owned.clone(),
            ghost_globals,
            ghost_owners,
            peers,
        };
        let local = RankLocalCsr {
            rank: rank as u32,
            owned: rows.owned.clone(),
            extended_len,
            row_ptr: rows.row_ptr.clone(),
            col_idx,
            values: rows.values.clone(),
        };
        Ok((halo, local, stats))
    })();
    if mpi.all_reduce_sum_u64(u64::from(built.is_err())) != 0 {
        return Err(invalid("inconsistent MPI-owned CSR halo configuration"));
    }
    built
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_global_column_outside_domain() {
        let input = OwnedCsrRows {
            global_dofs: 3,
            owned: 0..2,
            row_ptr: vec![0, 2, 3],
            global_col_idx: vec![0, 3, 1],
            values: vec![2.0, -1.0, 2.0],
        };
        assert!(input.validate().is_err());
    }

    #[test]
    fn checks_local_only_row_pointer_shape() {
        let input = OwnedCsrRows {
            global_dofs: 5,
            owned: 2..4,
            row_ptr: vec![0, 2, 3],
            global_col_idx: vec![2, 1, 3],
            values: vec![2.0, -1.0, 2.0],
        };
        assert!(input.validate().is_ok());
    }
}
