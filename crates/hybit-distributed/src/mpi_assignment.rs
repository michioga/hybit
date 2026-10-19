//! H1-C3K3 experimental MPI SpMV for noncontiguous, per-DOF ownership.
//!
//! Unlike the existing contiguous `OwnedCsrRows`, these owned rows have an
//! ascending list of global row IDs. Halo setup exchanges unique, sorted
//! ghost requests with MPI_Alltoallv. The resulting numeric exchange reuses
//! count/displacement layouts and packs owned-vector slots (not global IDs).
//! This module does not change the production MPI-PCG/contiguous path.
//! The ownership map is intentionally replicated on every rank (O(global n)).

use crate::mpi_backend::MpiRuntime;
use crate::PartitionAssignment;
use hybit_matrix::Csr32Matrix;
use mpi::datatype::{Partition, PartitionMut};
use mpi::traits::*;
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::time::Instant;

fn invalid(msg: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

fn displacements(counts: &[i32]) -> io::Result<(Vec<i32>, usize)> {
    let mut offset = 0i32;
    let mut result = Vec::with_capacity(counts.len());
    for &count in counts {
        if count < 0 {
            return Err(invalid("negative MPI count"));
        }
        result.push(offset);
        offset = offset
            .checked_add(count)
            .ok_or_else(|| invalid("MPI Alltoallv displacement exceeds i32"))?;
    }
    Ok((result, offset as usize))
}

/// Rows held by a rank, ordered by increasing global row ID, but not
/// necessarily consecutive. Column IDs are always global.
#[derive(Clone, Debug)]
pub struct AssignmentOwnedCsrRows {
    pub global_dofs: u64,
    pub owned_globals: Vec<u64>,
    pub row_ptr: Vec<u32>,
    pub global_col_idx: Vec<u64>,
    pub values: Vec<f64>,
}

impl AssignmentOwnedCsrRows {
    /// Bounded replicated-CSR benchmark helper. Do not use as a production
    /// replacement for root/parallel row redistribution.
    pub fn from_global(
        matrix: &Csr32Matrix,
        assignment: &PartitionAssignment,
        rank: u32,
    ) -> io::Result<Self> {
        if matrix.nrows() == 0
            || matrix.nrows() != matrix.ncols()
            || assignment.global_dofs() != matrix.nrows() as u64
            || rank >= assignment.rank_count()
        {
            return Err(invalid("invalid global matrix/assignment/rank"));
        }
        let mut owned_globals = Vec::new();
        let mut row_ptr = vec![0u32];
        let mut global_col_idx = Vec::new();
        let mut values = Vec::new();
        for (row, &owner) in assignment.owners().iter().enumerate() {
            if owner != rank {
                continue;
            }
            owned_globals.push(row as u64);
            let begin = matrix.row_ptr()[row] as usize;
            let end = matrix.row_ptr()[row + 1] as usize;
            global_col_idx.extend(
                matrix.col_idx()[begin..end]
                    .iter()
                    .map(|&col| u64::from(col)),
            );
            values.extend_from_slice(&matrix.values()[begin..end]);
            row_ptr.push(
                u32::try_from(values.len())
                    .map_err(|_| invalid("rank-local CSR nnz exceeds u32"))?,
            );
        }
        let result = Self {
            global_dofs: matrix.nrows() as u64,
            owned_globals,
            row_ptr,
            global_col_idx,
            values,
        };
        result.validate(assignment, rank)?;
        Ok(result)
    }

    pub fn validate(&self, assignment: &PartitionAssignment, rank: u32) -> io::Result<()> {
        if self.global_dofs == 0
            || self.global_dofs != assignment.global_dofs()
            || rank >= assignment.rank_count()
            || self.owned_globals.is_empty()
            || self.row_ptr.len() != self.owned_globals.len() + 1
            || self.row_ptr.first() != Some(&0)
            || self.row_ptr.windows(2).any(|w| w[0] > w[1])
            || self.global_col_idx.len() != self.values.len()
            || self.global_col_idx.len() > u32::MAX as usize
            || self.row_ptr.last().copied() != Some(self.values.len() as u32)
            || self.owned_globals.windows(2).any(|w| w[0] >= w[1])
            || self.global_col_idx.iter().any(|&c| c >= self.global_dofs)
            || self.values.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("invalid noncontiguous rank-local CSR"));
        }
        let expected = assignment
            .owners()
            .iter()
            .enumerate()
            .filter(|(_, owner)| **owner == rank)
            .map(|(i, _)| i as u64);
        if !self.owned_globals.iter().copied().eq(expected) {
            return Err(invalid("owned global row list disagrees with assignment"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AssignmentMpiStats {
    pub owned_rows: usize,
    pub local_nnz: usize,
    pub ghosts: usize,
    pub peer_ranks: usize,
    pub sent_f64: usize,
    pub received_f64: usize,
    /// Excludes replicated owner map and the global input CSR benchmark copy.
    pub rank_operator_halo_bytes: usize,
}

/// Prepared reciprocal numeric exchange; field order is `[owned | ghosts]`.
#[derive(Debug)]
pub struct AssignmentHaloPlan {
    pub owned_globals: Vec<u64>,
    pub ghost_globals: Vec<u64>,
    send_counts: Vec<i32>,
    send_displs: Vec<i32>,
    recv_counts: Vec<i32>,
    recv_displs: Vec<i32>,
    send_owned_slots: Vec<u32>,
}

impl AssignmentHaloPlan {
    pub fn owned_len(&self) -> usize {
        self.owned_globals.len()
    }
    pub fn extended_len(&self) -> usize {
        self.owned_globals.len() + self.ghost_globals.len()
    }
    pub fn workspace(&self) -> AssignmentSpmvWorkspace {
        AssignmentSpmvWorkspace {
            extended: vec![0.0; self.extended_len()],
            send: vec![0.0; self.send_owned_slots.len()],
            receive: vec![0.0; self.ghost_globals.len()],
        }
    }
}

#[derive(Debug)]
pub struct AssignmentLocalCsr {
    row_ptr: Vec<u32>,
    col_idx: Vec<u32>,
    values: Vec<f64>,
    owned_len: usize,
    extended_len: usize,
}

impl AssignmentLocalCsr {
    pub fn owned_len(&self) -> usize {
        self.owned_len
    }
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    pub fn apply_extended(&self, extended: &[f64], output: &mut [f64]) -> io::Result<()> {
        if extended.len() != self.extended_len || output.len() != self.owned_len {
            return Err(invalid("SpMV vector lengths differ from prepared operator"));
        }
        for (row, value) in output.iter_mut().enumerate() {
            let first = self.row_ptr[row] as usize;
            let last = self.row_ptr[row + 1] as usize;
            *value = (first..last)
                .map(|j| self.values[j] * extended[self.col_idx[j] as usize])
                .sum();
        }
        Ok(())
    }
}

/// Caller-allocated reusable scratch storage for repeated native MPI SpMV.
#[derive(Debug)]
pub struct AssignmentSpmvWorkspace {
    pub extended: Vec<f64>,
    send: Vec<f64>,
    receive: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AssignmentSpmvTiming {
    pub halo_ns: u128,
    pub numeric_ns: u128,
}

/// Collectively prepare a noncontiguous rank-local CSR and all-to-all Halo.
/// All ranks must supply an identical `PartitionAssignment` (checked via a
/// rank-zero broadcast). Pre-collective failures are aggregated on every rank.
/// This prepares communication once; per-apply exchanges carry f64 only.
pub fn prepare_assignment_owned_rows(
    mpi: &MpiRuntime,
    assignment: &PartitionAssignment,
    rows: &AssignmentOwnedCsrRows,
) -> io::Result<(AssignmentHaloPlan, AssignmentLocalCsr, AssignmentMpiStats)> {
    let size = usize::try_from(mpi.size()).map_err(|_| invalid("invalid MPI size"))?;
    let rank = u32::try_from(mpi.rank()).map_err(|_| invalid("invalid MPI rank"))?;
    let check = rows.validate(assignment, rank).and_then(|()| {
        if assignment.rank_count() as usize != size {
            Err(invalid("MPI size differs from ownership assignment"))
        } else {
            Ok(())
        }
    });
    if mpi.all_reduce_sum_u64(u64::from(check.is_err())) != 0 {
        return Err(invalid("invalid assignment CSR on one or more ranks"));
    }
    check?;
    let world = mpi::topology::SimpleCommunicator::world();
    let global_dofs = assignment.global_dofs();
    let mut rank_dofs = vec![0u64; size];
    world.all_gather_into(&global_dofs, &mut rank_dofs[..]);
    if rank_dofs.iter().any(|&n| n != global_dofs) {
        return Err(invalid("MPI ranks disagree on global owner-map length"));
    }
    let mut root_owners = if rank == 0 {
        assignment.owners().to_vec()
    } else {
        vec![0u32; assignment.owners().len()]
    };
    world
        .process_at_rank(0)
        .broadcast_into(&mut root_owners[..]);
    if mpi.all_reduce_sum_u64(u64::from(root_owners != assignment.owners())) != 0 {
        return Err(invalid("rank ownership maps are not identical"));
    }

    let mut owned_slots = BTreeMap::new();
    for (slot, &global) in rows.owned_globals.iter().enumerate() {
        owned_slots.insert(
            global,
            u32::try_from(slot).map_err(|_| invalid("owned slot exceeds u32"))?,
        );
    }
    let mut requests = vec![BTreeSet::<u64>::new(); size];
    for &col in &rows.global_col_idx {
        let owner = assignment
            .owner_of(col)
            .map_err(|_| invalid("global CSR column has no owner"))?;
        if owner != rank {
            requests[owner as usize].insert(col);
        }
    }
    let send_counts = requests
        .iter()
        .map(|req| i32::try_from(req.len()).map_err(|_| invalid("peer request count > i32")))
        .collect::<io::Result<Vec<_>>>();
    if mpi.all_reduce_sum_u64(u64::from(send_counts.is_err())) != 0 {
        return Err(invalid("peer request count exceeds MPI i32 Count"));
    }
    let send_counts = send_counts?;
    let local_layout = displacements(&send_counts);
    if mpi.all_reduce_sum_u64(u64::from(local_layout.is_err())) != 0 {
        return Err(invalid("MPI ghost request buffer exceeds i32"));
    }
    let (recv_displs, ghost_len) = local_layout?;
    let recv_counts = send_counts;
    let ghost_globals = requests
        .iter()
        .flat_map(|set| set.iter().copied())
        .collect::<Vec<_>>();
    if ghost_globals.len() != ghost_len {
        return Err(invalid("ghost count/layout mismatch"));
    }

    let mut incoming_counts = vec![0i32; size];
    world.all_to_all_into(&recv_counts[..], &mut incoming_counts[..]);
    let incoming_layout = displacements(&incoming_counts);
    if mpi.all_reduce_sum_u64(u64::from(incoming_layout.is_err())) != 0 {
        return Err(invalid("MPI incoming request buffer exceeds i32"));
    }
    let (send_displs, incoming_len) = incoming_layout?;
    let mut incoming_globals = vec![0u64; incoming_len];
    {
        let send = Partition::new(&ghost_globals[..], &recv_counts[..], &recv_displs[..]);
        let mut recv = PartitionMut::new(
            &mut incoming_globals[..],
            &incoming_counts[..],
            &send_displs[..],
        );
        world.all_to_all_varcount_into(&send, &mut recv);
    }

    let built = (|| -> io::Result<(AssignmentHaloPlan, AssignmentLocalCsr, AssignmentMpiStats)> {
        let mut send_owned_slots = Vec::with_capacity(incoming_globals.len());
        for (peer, &count) in incoming_counts.iter().enumerate() {
            let start = send_displs[peer] as usize;
            let end = start + count as usize;
            let globals = &incoming_globals[start..end];
            if globals.windows(2).any(|w| w[0] >= w[1]) {
                return Err(invalid("peer requested unsorted/duplicate DOFs"));
            }
            for global in globals {
                send_owned_slots.push(
                    *owned_slots
                        .get(global)
                        .ok_or_else(|| invalid("peer requested non-owned DOF"))?,
                );
            }
        }
        let owned_len = rows.owned_globals.len();
        let mut slots = owned_slots;
        for (i, &ghost) in ghost_globals.iter().enumerate() {
            slots.insert(
                ghost,
                u32::try_from(owned_len + i).map_err(|_| invalid("extended index exceeds u32"))?,
            );
        }
        let col_idx = rows
            .global_col_idx
            .iter()
            .map(|c| {
                slots
                    .get(c)
                    .copied()
                    .ok_or_else(|| invalid("local CSR column not in owned/ghost slots"))
            })
            .collect::<io::Result<Vec<_>>>()?;
        let peer_ranks = (0..size)
            .filter(|&peer| recv_counts[peer] > 0 || incoming_counts[peer] > 0)
            .count();
        let stats = AssignmentMpiStats {
            owned_rows: owned_len,
            local_nnz: rows.values.len(),
            ghosts: ghost_len,
            peer_ranks,
            sent_f64: incoming_globals.len(),
            received_f64: ghost_len,
            rank_operator_halo_bytes: rows.row_ptr.len() * 4
                + col_idx.len() * 4
                + rows.values.len() * 8
                + rows.owned_globals.len() * 8
                + ghost_globals.len() * 8
                + (owned_len + ghost_len) * 8
                + send_owned_slots.len() * 4,
        };
        Ok((
            AssignmentHaloPlan {
                owned_globals: rows.owned_globals.clone(),
                ghost_globals,
                send_counts: incoming_counts,
                send_displs,
                recv_counts,
                recv_displs,
                send_owned_slots,
            },
            AssignmentLocalCsr {
                row_ptr: rows.row_ptr.clone(),
                col_idx,
                values: rows.values.clone(),
                owned_len,
                extended_len: owned_len + ghost_len,
            },
            stats,
        ))
    })();
    if mpi.all_reduce_sum_u64(u64::from(built.is_err())) != 0 {
        return Err(invalid("invalid reciprocal noncontiguous MPI Halo setup"));
    }
    built
}

/// Native MPI_Alltoallv f64 exchange followed by local CSR SpMV. All ranks
/// must enter this operation collectively in the same epoch and method order.
/// `AssignmentSpmvTiming` splits numeric exchange from local multiplication.
pub fn spmv_assignment(
    mpi: &MpiRuntime,
    plan: &AssignmentHaloPlan,
    operator: &AssignmentLocalCsr,
    x_owned: &[f64],
    scratch: &mut AssignmentSpmvWorkspace,
    y_owned: &mut [f64],
) -> io::Result<AssignmentSpmvTiming> {
    let local_bad = plan.owned_len() != x_owned.len()
        || plan.owned_len() != operator.owned_len()
        || operator.extended_len != plan.extended_len()
        || scratch.extended.len() != plan.extended_len()
        || scratch.send.len() != plan.send_owned_slots.len()
        || scratch.receive.len() != plan.ghost_globals.len()
        || y_owned.len() != plan.owned_len();
    if mpi.all_reduce_sum_u64(u64::from(local_bad)) != 0 {
        return Err(invalid("incompatible MPI assignment SpMV inputs"));
    }
    let started = Instant::now();
    scratch.extended[..x_owned.len()].copy_from_slice(x_owned);
    for (value, &slot) in scratch.send.iter_mut().zip(&plan.send_owned_slots) {
        *value = x_owned[slot as usize];
    }
    let world = mpi::topology::SimpleCommunicator::world();
    {
        let send = Partition::new(
            &scratch.send[..],
            &plan.send_counts[..],
            &plan.send_displs[..],
        );
        let mut recv = PartitionMut::new(
            &mut scratch.receive[..],
            &plan.recv_counts[..],
            &plan.recv_displs[..],
        );
        world.all_to_all_varcount_into(&send, &mut recv);
    }
    scratch.extended[x_owned.len()..].copy_from_slice(&scratch.receive);
    let halo_ns = started.elapsed().as_nanos();
    let numeric_started = Instant::now();
    operator.apply_extended(&scratch.extended, y_owned)?;
    Ok(AssignmentSpmvTiming {
        halo_ns,
        numeric_ns: numeric_started.elapsed().as_nanos(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix() -> Csr32Matrix {
        Csr32Matrix::new(
            4,
            4,
            vec![0, 2, 4, 7, 9],
            vec![0, 1, 1, 2, 0, 2, 3, 1, 3],
            vec![4., 1., 5., 2., 3., 6., 1., 1., 7.],
        )
        .unwrap()
    }

    #[test]
    fn noncontiguous_owned_rows_preserve_global_sorted_order_and_values() {
        let owners = PartitionAssignment::from_owners(2, vec![0, 1, 0, 1]).unwrap();
        let root = matrix();
        let r0 = AssignmentOwnedCsrRows::from_global(&root, &owners, 0).unwrap();
        assert_eq!(r0.owned_globals, vec![0, 2]);
        assert_eq!(r0.row_ptr, vec![0, 2, 5]);
        assert_eq!(r0.global_col_idx, vec![0, 1, 0, 2, 3]);
        assert_eq!(r0.values, vec![4., 1., 3., 6., 1.]);
        let r1 = AssignmentOwnedCsrRows::from_global(&root, &owners, 1).unwrap();
        assert_eq!(r1.owned_globals, vec![1, 3]);
        assert_eq!(r1.row_ptr, vec![0, 2, 4]);
    }

    #[test]
    fn malformed_owned_rows_are_rejected() {
        let owners = PartitionAssignment::from_owners(2, vec![0, 1, 0, 1]).unwrap();
        let mut rows = AssignmentOwnedCsrRows::from_global(&matrix(), &owners, 0).unwrap();
        rows.owned_globals[1] = 3;
        assert!(rows.validate(&owners, 0).is_err());
        rows.owned_globals[1] = 2;
        rows.global_col_idx[0] = 99;
        assert!(rows.validate(&owners, 0).is_err());
    }

    #[test]
    fn bad_displacement_counts_rejected() {
        assert!(displacements(&[i32::MAX, 1]).is_err());
        assert!(displacements(&[-1, 0]).is_err());
    }
}
