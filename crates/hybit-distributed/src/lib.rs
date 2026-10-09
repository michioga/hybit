#[cfg(feature = "mpi")]
pub mod mpi_backend;
#[cfg(feature = "mpi")]
pub mod mpi_overlap;
#[cfg(feature = "mpi")]
pub mod mpi_pcg;
mod multilevel;
mod partition;

use hybit_matrix::Csr32Matrix;
pub use multilevel::{abtm_multilevel_partition, AbtmMultilevelOptions, AbtmMultilevelStats};
pub use partition::{
    abtm_balanced_multisource_partition, abtm_region_grow_partition,
    partition_telemetry_assignment, AbtmMultisourceStats, AbtmPartitionStats, PartitionAssignment,
};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::ops::Range;

/// Global degree-of-freedom identifier.
///
/// G8 uses 64-bit global identifiers even though the current single-process
/// CSR32 reference matrix stores column indices as `u32`. Rank-local indices
/// remain 32-bit.
pub type GlobalDofId = u64;

/// MPI-style rank identifier without depending on an MPI runtime.
pub type RankId = u32;

/// Rank-local owned/extended-vector index.
pub type LocalDofIndex = u32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DistributedTopologyError {
    InvalidRankCount,
    InvalidPartition(&'static str),
    RankOutOfRange {
        rank: RankId,
        ranks: RankId,
    },
    GlobalDofOutOfRange {
        global: GlobalDofId,
        global_dofs: GlobalDofId,
    },
    MatrixMustBeSquare {
        rows: usize,
        cols: usize,
    },
    MatrixPartitionMismatch {
        matrix_rows: usize,
        global_dofs: GlobalDofId,
    },
    LocalIndexOverflow,
    VectorLengthMismatch {
        expected: usize,
        actual: usize,
    },
}

impl Display for DistributedTopologyError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRankCount => write!(f, "rank count must be nonzero and <= global DOFs"),
            Self::InvalidPartition(msg) => write!(f, "invalid contiguous partition: {msg}"),
            Self::RankOutOfRange { rank, ranks } => {
                write!(f, "rank {rank} is outside 0..{ranks}")
            }
            Self::GlobalDofOutOfRange {
                global,
                global_dofs,
            } => write!(f, "global DOF {global} is outside 0..{global_dofs}"),
            Self::MatrixMustBeSquare { rows, cols } => {
                write!(
                    f,
                    "distributed reference matrix must be square; got {rows}x{cols}"
                )
            }
            Self::MatrixPartitionMismatch {
                matrix_rows,
                global_dofs,
            } => write!(
                f,
                "matrix rows ({matrix_rows}) do not match partition global DOFs ({global_dofs})"
            ),
            Self::LocalIndexOverflow => write!(
                f,
                "rank-local owned/ghost index exceeds the 32-bit local representation"
            ),
            Self::VectorLengthMismatch { expected, actual } => write!(
                f,
                "vector length mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl Error for DistributedTopologyError {}

/// Deterministic contiguous ownership used as the G8 reference partition.
///
/// This is deliberately not presented as a production graph partitioner. It
/// provides an unambiguous ownership contract against which ABTM-assisted and
/// MPI-hosted implementations can be cross-checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContiguousPartition {
    global_dofs: GlobalDofId,
    offsets: Vec<GlobalDofId>,
}

impl ContiguousPartition {
    /// Build a balanced contiguous partition. Earlier ranks receive one extra
    /// DOF when `global_dofs` is not divisible by `ranks`.
    pub fn balanced(
        global_dofs: GlobalDofId,
        ranks: RankId,
    ) -> Result<Self, DistributedTopologyError> {
        if ranks == 0 || global_dofs == 0 || u64::from(ranks) > global_dofs {
            return Err(DistributedTopologyError::InvalidRankCount);
        }

        let ranks_u64 = u64::from(ranks);
        let base = global_dofs / ranks_u64;
        let remainder = global_dofs % ranks_u64;

        let mut offsets = Vec::with_capacity(ranks as usize + 1);
        offsets.push(0);
        let mut next = 0_u64;
        for rank in 0..ranks_u64 {
            next += base + u64::from(rank < remainder);
            offsets.push(next);
        }

        Self::from_offsets(offsets)
    }

    /// Build from explicit contiguous ownership offsets:
    ///
    /// `offsets[r]..offsets[r+1]` is owned by rank `r`.
    pub fn from_offsets(offsets: Vec<GlobalDofId>) -> Result<Self, DistributedTopologyError> {
        if offsets.len() < 2 {
            return Err(DistributedTopologyError::InvalidPartition(
                "at least one rank is required",
            ));
        }
        if offsets[0] != 0 {
            return Err(DistributedTopologyError::InvalidPartition(
                "first offset must be zero",
            ));
        }
        if offsets.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(DistributedTopologyError::InvalidPartition(
                "rank ownership ranges must be nonempty and strictly increasing",
            ));
        }

        let rank_count = offsets.len() - 1;
        if rank_count > u32::MAX as usize {
            return Err(DistributedTopologyError::InvalidRankCount);
        }

        let global_dofs = *offsets
            .last()
            .ok_or(DistributedTopologyError::InvalidPartition(
                "missing final global offset",
            ))?;

        Ok(Self {
            global_dofs,
            offsets,
        })
    }

    pub fn global_dofs(&self) -> GlobalDofId {
        self.global_dofs
    }

    pub fn rank_count(&self) -> RankId {
        (self.offsets.len() - 1) as RankId
    }

    pub fn offsets(&self) -> &[GlobalDofId] {
        &self.offsets
    }

    pub fn owned_range(
        &self,
        rank: RankId,
    ) -> Result<Range<GlobalDofId>, DistributedTopologyError> {
        if rank >= self.rank_count() {
            return Err(DistributedTopologyError::RankOutOfRange {
                rank,
                ranks: self.rank_count(),
            });
        }
        let index = rank as usize;
        Ok(self.offsets[index]..self.offsets[index + 1])
    }

    pub fn owner_of(&self, global: GlobalDofId) -> Result<RankId, DistributedTopologyError> {
        if global >= self.global_dofs {
            return Err(DistributedTopologyError::GlobalDofOutOfRange {
                global,
                global_dofs: self.global_dofs,
            });
        }

        let first_greater = self.offsets.partition_point(|&offset| offset <= global);
        Ok((first_greater - 1) as RankId)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HaloPeerPlan {
    peer: RankId,
    /// Global DOFs owned by this rank and requested by `peer`.
    send_globals: Vec<GlobalDofId>,
    /// Owned-vector indices corresponding to `send_globals`.
    send_owned_indices: Vec<LocalDofIndex>,
    /// Global DOFs owned by `peer` and required by this rank.
    recv_globals: Vec<GlobalDofId>,
    /// Indices in `[owned | ghosts]` where received values are stored.
    recv_extended_indices: Vec<LocalDofIndex>,
}

impl HaloPeerPlan {
    pub fn peer(&self) -> RankId {
        self.peer
    }

    pub fn send_globals(&self) -> &[GlobalDofId] {
        &self.send_globals
    }

    pub fn send_owned_indices(&self) -> &[LocalDofIndex] {
        &self.send_owned_indices
    }

    pub fn recv_globals(&self) -> &[GlobalDofId] {
        &self.recv_globals
    }

    pub fn recv_extended_indices(&self) -> &[LocalDofIndex] {
        &self.recv_extended_indices
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HaloPlan {
    rank: RankId,
    owned: Range<GlobalDofId>,
    ghost_globals: Vec<GlobalDofId>,
    ghost_owners: Vec<RankId>,
    peers: Vec<HaloPeerPlan>,
}

impl HaloPlan {
    pub fn rank(&self) -> RankId {
        self.rank
    }

    pub fn owned_range(&self) -> Range<GlobalDofId> {
        self.owned.clone()
    }

    pub fn owned_len(&self) -> usize {
        (self.owned.end - self.owned.start) as usize
    }

    pub fn ghost_globals(&self) -> &[GlobalDofId] {
        &self.ghost_globals
    }

    pub fn ghost_owners(&self) -> &[RankId] {
        &self.ghost_owners
    }

    pub fn ghost_len(&self) -> usize {
        self.ghost_globals.len()
    }

    pub fn extended_len(&self) -> usize {
        self.owned_len() + self.ghost_len()
    }

    pub fn peers(&self) -> &[HaloPeerPlan] {
        &self.peers
    }

    pub fn peer(&self, rank: RankId) -> Option<&HaloPeerPlan> {
        self.peers.iter().find(|peer| peer.peer == rank)
    }
}

#[derive(Clone, Debug, Default)]
struct PeerAccumulator {
    send_globals: Vec<GlobalDofId>,
    recv_globals: Vec<GlobalDofId>,
}

/// Build deterministic halo plans for every rank from a global CSR reference.
///
/// Semantics:
/// - rows are owned by the same contiguous partition as DOFs;
/// - any column referenced by an owned row but owned by another rank is a
///   receive ghost;
/// - duplicate remote references collapse to one ghost entry;
/// - send lists are the exact reciprocal of other ranks' receive requests;
/// - local extended-vector order is `[owned DOFs | ghosts]`;
/// - ghosts are ordered by `(owner rank, global DOF)`.
///
/// No communication is performed here. G8-A1 establishes the topology
/// contract before an MPI transport is introduced.
pub fn build_contiguous_halo_plans(
    matrix: &Csr32Matrix,
    partition: &ContiguousPartition,
) -> Result<Vec<HaloPlan>, DistributedTopologyError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(DistributedTopologyError::MatrixMustBeSquare {
            rows: matrix.nrows(),
            cols: matrix.ncols(),
        });
    }
    if matrix.nrows() as u64 != partition.global_dofs() {
        return Err(DistributedTopologyError::MatrixPartitionMismatch {
            matrix_rows: matrix.nrows(),
            global_dofs: partition.global_dofs(),
        });
    }

    let ranks = partition.rank_count() as usize;
    let mut recv_requests: Vec<BTreeMap<RankId, BTreeSet<GlobalDofId>>> =
        (0..ranks).map(|_| BTreeMap::new()).collect();

    for (rank_index, rank_requests) in recv_requests.iter_mut().enumerate() {
        let rank = rank_index as RankId;
        let owned = partition.owned_range(rank)?;
        let row_start = usize::try_from(owned.start)
            .map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;
        let row_end =
            usize::try_from(owned.end).map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;

        for row in row_start..row_end {
            let start = matrix.row_ptr()[row] as usize;
            let end = matrix.row_ptr()[row + 1] as usize;
            for position in start..end {
                let global_col = GlobalDofId::from(matrix.col_idx()[position]);
                let owner = partition.owner_of(global_col)?;
                if owner != rank {
                    rank_requests.entry(owner).or_default().insert(global_col);
                }
            }
        }
    }

    let mut accumulators: Vec<BTreeMap<RankId, PeerAccumulator>> =
        (0..ranks).map(|_| BTreeMap::new()).collect();

    for (receiver_index, receiver_requests) in recv_requests.iter().enumerate() {
        let receiver = receiver_index as RankId;
        for (&owner, globals) in receiver_requests {
            let ids: Vec<_> = globals.iter().copied().collect();

            accumulators[receiver_index]
                .entry(owner)
                .or_default()
                .recv_globals = ids.clone();

            accumulators[owner as usize]
                .entry(receiver)
                .or_default()
                .send_globals = ids;
        }
    }

    let mut plans = Vec::with_capacity(ranks);

    for (rank_index, peer_map) in accumulators.iter().enumerate() {
        let rank = rank_index as RankId;
        let owned = partition.owned_range(rank)?;
        let owned_len_u64 = owned.end - owned.start;
        let owned_len = u32::try_from(owned_len_u64)
            .map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;

        let mut ghost_globals = Vec::new();
        let mut ghost_owners = Vec::new();
        let mut ghost_slot_by_global = BTreeMap::new();

        for (&peer, accumulator) in peer_map {
            for &global in &accumulator.recv_globals {
                let slot = u32::try_from(ghost_globals.len())
                    .map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;
                let extended = owned_len
                    .checked_add(slot)
                    .ok_or(DistributedTopologyError::LocalIndexOverflow)?;

                ghost_slot_by_global.insert(global, extended);
                ghost_globals.push(global);
                ghost_owners.push(peer);
            }
        }

        let mut peers = Vec::with_capacity(peer_map.len());

        for (&peer, accumulator) in peer_map {
            let mut send_owned_indices = Vec::with_capacity(accumulator.send_globals.len());
            for &global in &accumulator.send_globals {
                if global < owned.start || global >= owned.end {
                    return Err(DistributedTopologyError::InvalidPartition(
                        "send DOF is not owned by the sending rank",
                    ));
                }
                let local = u32::try_from(global - owned.start)
                    .map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;
                send_owned_indices.push(local);
            }

            let mut recv_extended_indices = Vec::with_capacity(accumulator.recv_globals.len());
            for &global in &accumulator.recv_globals {
                let extended = *ghost_slot_by_global.get(&global).ok_or(
                    DistributedTopologyError::InvalidPartition(
                        "receive ghost is missing from the extended-vector map",
                    ),
                )?;
                recv_extended_indices.push(extended);
            }

            peers.push(HaloPeerPlan {
                peer,
                send_globals: accumulator.send_globals.clone(),
                send_owned_indices,
                recv_globals: accumulator.recv_globals.clone(),
                recv_extended_indices,
            });
        }

        plans.push(HaloPlan {
            rank,
            owned,
            ghost_globals,
            ghost_owners,
            peers,
        });
    }

    Ok(plans)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain_matrix(n: usize) -> Csr32Matrix {
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);

        for row in 0..n {
            if row > 0 {
                col_idx.push((row - 1) as u32);
                values.push(-1.0);
            }
            col_idx.push(row as u32);
            values.push(2.0);
            if row + 1 < n {
                col_idx.push((row + 1) as u32);
                values.push(-1.0);
            }
            row_ptr.push(col_idx.len() as u32);
        }

        Csr32Matrix::new(n, n, row_ptr, col_idx, values).unwrap()
    }

    #[test]
    fn balanced_partition_is_deterministic() {
        let partition = ContiguousPartition::balanced(10, 3).unwrap();
        assert_eq!(partition.offsets(), &[0, 4, 7, 10]);
        assert_eq!(partition.owned_range(0).unwrap(), 0..4);
        assert_eq!(partition.owned_range(1).unwrap(), 4..7);
        assert_eq!(partition.owned_range(2).unwrap(), 7..10);
        assert_eq!(partition.owner_of(0).unwrap(), 0);
        assert_eq!(partition.owner_of(3).unwrap(), 0);
        assert_eq!(partition.owner_of(4).unwrap(), 1);
        assert_eq!(partition.owner_of(9).unwrap(), 2);
    }

    #[test]
    fn two_rank_chain_builds_reciprocal_one_dof_halo() {
        let matrix = chain_matrix(6);
        let partition = ContiguousPartition::balanced(6, 2).unwrap();
        let plans = build_contiguous_halo_plans(&matrix, &partition).unwrap();

        let rank0 = &plans[0];
        assert_eq!(rank0.owned_range(), 0..3);
        assert_eq!(rank0.ghost_globals(), &[3]);
        assert_eq!(rank0.ghost_owners(), &[1]);
        assert_eq!(rank0.extended_len(), 4);

        let peer1 = rank0.peer(1).unwrap();
        assert_eq!(peer1.send_globals(), &[2]);
        assert_eq!(peer1.send_owned_indices(), &[2]);
        assert_eq!(peer1.recv_globals(), &[3]);
        assert_eq!(peer1.recv_extended_indices(), &[3]);

        let rank1 = &plans[1];
        assert_eq!(rank1.owned_range(), 3..6);
        assert_eq!(rank1.ghost_globals(), &[2]);

        let peer0 = rank1.peer(0).unwrap();
        assert_eq!(peer0.send_globals(), &[3]);
        assert_eq!(peer0.send_owned_indices(), &[0]);
        assert_eq!(peer0.recv_globals(), &[2]);
        assert_eq!(peer0.recv_extended_indices(), &[3]);
    }

    #[test]
    fn duplicate_remote_columns_collapse_to_one_ghost() {
        let matrix = Csr32Matrix::new(
            4,
            4,
            vec![0, 1, 5, 8, 9],
            vec![0, 0, 1, 2, 2, 1, 2, 3, 3],
            vec![1.0; 9],
        )
        .unwrap();
        let partition = ContiguousPartition::balanced(4, 2).unwrap();
        let plans = build_contiguous_halo_plans(&matrix, &partition).unwrap();

        assert_eq!(plans[0].ghost_globals(), &[2]);
        assert_eq!(plans[0].peer(1).unwrap().recv_globals(), &[2]);
        assert_eq!(plans[1].peer(0).unwrap().send_globals(), &[2]);
    }

    #[test]
    fn disconnected_blocks_need_no_halo() {
        let matrix = Csr32Matrix::new(
            4,
            4,
            vec![0, 2, 4, 6, 8],
            vec![0, 1, 0, 1, 2, 3, 2, 3],
            vec![1.0; 8],
        )
        .unwrap();
        let partition = ContiguousPartition::balanced(4, 2).unwrap();
        let plans = build_contiguous_halo_plans(&matrix, &partition).unwrap();

        assert!(plans[0].peers().is_empty());
        assert!(plans[1].peers().is_empty());
        assert_eq!(plans[0].ghost_len(), 0);
        assert_eq!(plans[1].ghost_len(), 0);
    }

    #[test]
    fn matrix_and_partition_dimensions_must_match() {
        let matrix = chain_matrix(4);
        let partition = ContiguousPartition::balanced(6, 2).unwrap();
        let error = build_contiguous_halo_plans(&matrix, &partition).unwrap_err();
        assert!(matches!(
            error,
            DistributedTopologyError::MatrixPartitionMismatch { .. }
        ));
    }
}
// -----------------------------------------------------------------------------
// G8-A2: rank-local operator preparation, simulated halo exchange, ABTM
// topology cross-check, and communication-quality telemetry.
// -----------------------------------------------------------------------------

/// CSR rows owned by one rank, with columns renumbered into the rank-local
/// extended vector `[owned | ghosts]`.
#[derive(Clone, Debug)]
pub struct RankLocalCsr {
    rank: RankId,
    owned: Range<GlobalDofId>,
    extended_len: usize,
    row_ptr: Vec<u32>,
    col_idx: Vec<u32>,
    values: Vec<f64>,
}

impl RankLocalCsr {
    pub fn rank(&self) -> RankId {
        self.rank
    }

    pub fn owned_range(&self) -> Range<GlobalDofId> {
        self.owned.clone()
    }

    pub fn owned_len(&self) -> usize {
        (self.owned.end - self.owned.start) as usize
    }

    pub fn extended_len(&self) -> usize {
        self.extended_len
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn row_ptr(&self) -> &[u32] {
        &self.row_ptr
    }

    pub fn col_idx(&self) -> &[u32] {
        &self.col_idx
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Apply the rank-local rows to an already-filled `[owned | ghosts]`
    /// extended vector.
    pub fn apply_extended(
        &self,
        x_extended: &[f64],
        y_owned: &mut [f64],
    ) -> Result<(), DistributedTopologyError> {
        if x_extended.len() != self.extended_len {
            return Err(DistributedTopologyError::VectorLengthMismatch {
                expected: self.extended_len,
                actual: x_extended.len(),
            });
        }
        if y_owned.len() != self.owned_len() {
            return Err(DistributedTopologyError::VectorLengthMismatch {
                expected: self.owned_len(),
                actual: y_owned.len(),
            });
        }

        for (row, out) in y_owned.iter_mut().enumerate() {
            let start = self.row_ptr[row] as usize;
            let end = self.row_ptr[row + 1] as usize;
            let mut sum = 0.0;
            for position in start..end {
                let col = self.col_idx[position] as usize;
                sum += self.values[position] * x_extended[col];
            }
            *out = sum;
        }
        Ok(())
    }
}

/// Prepare one rank-local CSR operator from the global CSR reference and its
/// already-established halo plan.
///
/// Stored entry order is preserved, so the simulated distributed SpMV can be
/// compared bit-for-bit with the serial CSR row evaluation.
pub fn prepare_rank_local_csr(
    matrix: &Csr32Matrix,
    plan: &HaloPlan,
) -> Result<RankLocalCsr, DistributedTopologyError> {
    let owned = plan.owned_range();
    let owned_len = plan.owned_len();

    let row_start =
        usize::try_from(owned.start).map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;
    let row_end =
        usize::try_from(owned.end).map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;

    if row_end > matrix.nrows() {
        return Err(DistributedTopologyError::InvalidPartition(
            "owned row range exceeds the global matrix",
        ));
    }

    let mut ghost_local = BTreeMap::new();
    for (ghost_slot, &global) in plan.ghost_globals().iter().enumerate() {
        let extended = owned_len
            .checked_add(ghost_slot)
            .ok_or(DistributedTopologyError::LocalIndexOverflow)?;
        let extended =
            u32::try_from(extended).map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;
        ghost_local.insert(global, extended);
    }

    let mut row_ptr = Vec::with_capacity(owned_len + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);

    for global_row in row_start..row_end {
        let start = matrix.row_ptr()[global_row] as usize;
        let end = matrix.row_ptr()[global_row + 1] as usize;

        for position in start..end {
            let global_col = GlobalDofId::from(matrix.col_idx()[position]);
            let local_col = if global_col >= owned.start && global_col < owned.end {
                u32::try_from(global_col - owned.start)
                    .map_err(|_| DistributedTopologyError::LocalIndexOverflow)?
            } else {
                *ghost_local
                    .get(&global_col)
                    .ok_or(DistributedTopologyError::InvalidPartition(
                        "remote matrix column is absent from the halo plan",
                    ))?
            };

            col_idx.push(local_col);
            values.push(matrix.values()[position]);
        }

        let next = u32::try_from(col_idx.len())
            .map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;
        row_ptr.push(next);
    }

    Ok(RankLocalCsr {
        rank: plan.rank(),
        owned,
        extended_len: plan.extended_len(),
        row_ptr,
        col_idx,
        values,
    })
}

pub fn prepare_rank_local_csrs(
    matrix: &Csr32Matrix,
    plans: &[HaloPlan],
) -> Result<Vec<RankLocalCsr>, DistributedTopologyError> {
    plans
        .iter()
        .map(|plan| prepare_rank_local_csr(matrix, plan))
        .collect()
}

fn validate_ordered_plans(plans: &[HaloPlan]) -> Result<usize, DistributedTopologyError> {
    if plans.is_empty() {
        return Err(DistributedTopologyError::InvalidPartition(
            "at least one halo plan is required",
        ));
    }

    let mut next_global = 0_u64;
    for (rank_index, plan) in plans.iter().enumerate() {
        if plan.rank() as usize != rank_index {
            return Err(DistributedTopologyError::InvalidPartition(
                "halo plans must be stored in rank order",
            ));
        }
        let owned = plan.owned_range();
        if owned.start != next_global || owned.start >= owned.end {
            return Err(DistributedTopologyError::InvalidPartition(
                "halo plan ownership ranges are not contiguous",
            ));
        }
        next_global = owned.end;
    }

    usize::try_from(next_global).map_err(|_| DistributedTopologyError::LocalIndexOverflow)
}

/// Simulate one halo exchange using the exact reciprocal send/receive index
/// lists created by G8-A1.
///
/// No value is copied directly from a remote global index: ghost slots are
/// populated through the sender's owned-vector index and the receiver's
/// extended-vector index. This makes the routine a transport-free reference
/// for the later MPI adapter.
pub fn simulate_halo_exchange(
    global_x: &[f64],
    plans: &[HaloPlan],
) -> Result<Vec<Vec<f64>>, DistributedTopologyError> {
    let global_len = validate_ordered_plans(plans)?;
    if global_x.len() != global_len {
        return Err(DistributedTopologyError::VectorLengthMismatch {
            expected: global_len,
            actual: global_x.len(),
        });
    }

    let mut extended = Vec::with_capacity(plans.len());
    let mut filled = Vec::with_capacity(plans.len());

    for plan in plans {
        let owned = plan.owned_range();
        let start = usize::try_from(owned.start)
            .map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;
        let end =
            usize::try_from(owned.end).map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;

        let mut local = vec![0.0; plan.extended_len()];
        local[..plan.owned_len()].copy_from_slice(&global_x[start..end]);

        let mut flags = vec![false; plan.extended_len()];
        flags[..plan.owned_len()].fill(true);

        extended.push(local);
        filled.push(flags);
    }

    for (sender_index, sender_plan) in plans.iter().enumerate() {
        for sender_peer in sender_plan.peers() {
            if sender_peer.send_globals().is_empty() {
                continue;
            }

            let receiver_index = sender_peer.peer() as usize;
            let receiver_plan =
                plans
                    .get(receiver_index)
                    .ok_or(DistributedTopologyError::InvalidPartition(
                        "halo peer rank is outside the plan set",
                    ))?;
            let receiver_peer = receiver_plan.peer(sender_plan.rank()).ok_or(
                DistributedTopologyError::InvalidPartition("halo peer relation is not reciprocal"),
            )?;

            if sender_peer.send_globals() != receiver_peer.recv_globals() {
                return Err(DistributedTopologyError::InvalidPartition(
                    "send and reciprocal receive global lists differ",
                ));
            }
            if sender_peer.send_owned_indices().len() != receiver_peer.recv_extended_indices().len()
            {
                return Err(DistributedTopologyError::InvalidPartition(
                    "send and reciprocal receive index counts differ",
                ));
            }

            for (&send_index, &recv_index) in sender_peer
                .send_owned_indices()
                .iter()
                .zip(receiver_peer.recv_extended_indices())
            {
                let send_index = send_index as usize;
                let recv_index = recv_index as usize;

                let value = *extended[sender_index].get(send_index).ok_or(
                    DistributedTopologyError::InvalidPartition(
                        "send index exceeds sender owned vector",
                    ),
                )?;

                let receiver_vector = extended.get_mut(receiver_index).ok_or(
                    DistributedTopologyError::InvalidPartition(
                        "receiver rank is outside extended vectors",
                    ),
                )?;
                let slot = receiver_vector.get_mut(recv_index).ok_or(
                    DistributedTopologyError::InvalidPartition(
                        "receive index exceeds receiver extended vector",
                    ),
                )?;
                *slot = value;

                let receiver_flags = filled.get_mut(receiver_index).ok_or(
                    DistributedTopologyError::InvalidPartition(
                        "receiver rank is outside halo fill flags",
                    ),
                )?;
                let flag = receiver_flags.get_mut(recv_index).ok_or(
                    DistributedTopologyError::InvalidPartition(
                        "receive index exceeds halo fill flags",
                    ),
                )?;
                *flag = true;
            }
        }
    }

    if filled.iter().any(|flags| flags.iter().any(|&flag| !flag)) {
        return Err(DistributedTopologyError::InvalidPartition(
            "at least one ghost slot was not populated by reciprocal halo exchange",
        ));
    }

    Ok(extended)
}

/// Transport-free distributed SpMV reference.
///
/// This intentionally rebuilds topology/operators for clarity. Prepared
/// production execution will separate setup from repeated apply.
pub fn simulated_distributed_spmv(
    matrix: &Csr32Matrix,
    partition: &ContiguousPartition,
    global_x: &[f64],
) -> Result<Vec<f64>, DistributedTopologyError> {
    let plans = build_contiguous_halo_plans(matrix, partition)?;
    let local_operators = prepare_rank_local_csrs(matrix, &plans)?;
    let extended = simulate_halo_exchange(global_x, &plans)?;

    let mut global_y = vec![0.0; matrix.nrows()];

    for ((plan, local), local_x) in plans.iter().zip(&local_operators).zip(&extended) {
        let mut local_y = vec![0.0; plan.owned_len()];
        local.apply_extended(local_x, &mut local_y)?;

        let owned = plan.owned_range();
        let start = usize::try_from(owned.start)
            .map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;
        let end =
            usize::try_from(owned.end).map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;
        global_y[start..end].copy_from_slice(&local_y);
    }

    Ok(global_y)
}

fn finalize_halo_plans_from_recv_requests(
    partition: &ContiguousPartition,
    recv_requests: &[BTreeMap<RankId, BTreeSet<GlobalDofId>>],
) -> Result<Vec<HaloPlan>, DistributedTopologyError> {
    let ranks = partition.rank_count() as usize;
    if recv_requests.len() != ranks {
        return Err(DistributedTopologyError::InvalidPartition(
            "receive-request rank count differs from partition",
        ));
    }

    let mut accumulators: Vec<BTreeMap<RankId, PeerAccumulator>> =
        (0..ranks).map(|_| BTreeMap::new()).collect();

    for (receiver_index, receiver_requests) in recv_requests.iter().enumerate() {
        let receiver = receiver_index as RankId;
        for (&owner, globals) in receiver_requests {
            let ids: Vec<_> = globals.iter().copied().collect();

            accumulators[receiver_index]
                .entry(owner)
                .or_default()
                .recv_globals = ids.clone();

            accumulators[owner as usize]
                .entry(receiver)
                .or_default()
                .send_globals = ids;
        }
    }

    let mut plans = Vec::with_capacity(ranks);

    for (rank_index, peer_map) in accumulators.iter().enumerate() {
        let rank = rank_index as RankId;
        let owned = partition.owned_range(rank)?;
        let owned_len = u32::try_from(owned.end - owned.start)
            .map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;

        let mut ghost_globals = Vec::new();
        let mut ghost_owners = Vec::new();
        let mut ghost_slot_by_global = BTreeMap::new();

        for (&peer, accumulator) in peer_map {
            for &global in &accumulator.recv_globals {
                let slot = u32::try_from(ghost_globals.len())
                    .map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;
                let extended = owned_len
                    .checked_add(slot)
                    .ok_or(DistributedTopologyError::LocalIndexOverflow)?;

                ghost_slot_by_global.insert(global, extended);
                ghost_globals.push(global);
                ghost_owners.push(peer);
            }
        }

        let mut peers = Vec::with_capacity(peer_map.len());

        for (&peer, accumulator) in peer_map {
            let mut send_owned_indices = Vec::with_capacity(accumulator.send_globals.len());
            for &global in &accumulator.send_globals {
                if global < owned.start || global >= owned.end {
                    return Err(DistributedTopologyError::InvalidPartition(
                        "send DOF is not owned by the sending rank",
                    ));
                }
                send_owned_indices.push(
                    u32::try_from(global - owned.start)
                        .map_err(|_| DistributedTopologyError::LocalIndexOverflow)?,
                );
            }

            let mut recv_extended_indices = Vec::with_capacity(accumulator.recv_globals.len());
            for &global in &accumulator.recv_globals {
                recv_extended_indices.push(*ghost_slot_by_global.get(&global).ok_or(
                    DistributedTopologyError::InvalidPartition(
                        "receive ghost is missing from the extended-vector map",
                    ),
                )?);
            }

            peers.push(HaloPeerPlan {
                peer,
                send_globals: accumulator.send_globals.clone(),
                send_owned_indices,
                recv_globals: accumulator.recv_globals.clone(),
                recv_extended_indices,
            });
        }

        plans.push(HaloPlan {
            rank,
            owned,
            ghost_globals,
            ghost_owners,
            peers,
        });
    }

    Ok(plans)
}

/// ABTM-topology implementation of the same G8-A1 halo contract.
///
/// The returned plans must be exactly equal to the CSR reference plans. ABTM is
/// used here only to enumerate structural row support; it is not used as the
/// numerical rank-local SpMV representation.
pub fn build_contiguous_halo_plans_abtm(
    matrix: &Csr32Matrix,
    partition: &ContiguousPartition,
) -> Result<Vec<HaloPlan>, DistributedTopologyError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(DistributedTopologyError::MatrixMustBeSquare {
            rows: matrix.nrows(),
            cols: matrix.ncols(),
        });
    }
    if matrix.nrows() as u64 != partition.global_dofs() {
        return Err(DistributedTopologyError::MatrixPartitionMismatch {
            matrix_rows: matrix.nrows(),
            global_dofs: partition.global_dofs(),
        });
    }

    let topology = hybit_matrix::AbtmTopology::from_csr32(matrix).map_err(|_| {
        DistributedTopologyError::InvalidPartition(
            "ABTM topology construction failed for the validated CSR matrix",
        )
    })?;

    let ranks = partition.rank_count() as usize;
    let mut recv_requests: Vec<BTreeMap<RankId, BTreeSet<GlobalDofId>>> =
        (0..ranks).map(|_| BTreeMap::new()).collect();

    for (rank_index, rank_requests) in recv_requests.iter_mut().enumerate() {
        let rank = rank_index as RankId;
        let owned = partition.owned_range(rank)?;
        let row_start = usize::try_from(owned.start)
            .map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;
        let row_end =
            usize::try_from(owned.end).map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;

        for row_index in row_start..row_end {
            let row = topology.row(row_index).map_err(|_| {
                DistributedTopologyError::InvalidPartition("ABTM topology row lookup failed")
            })?;

            for ordinal in 0..row.popcount() {
                let global_col =
                    row.select(ordinal)
                        .ok_or(DistributedTopologyError::InvalidPartition(
                            "ABTM topology select failed",
                        ))? as GlobalDofId;

                let owner = partition.owner_of(global_col)?;
                if owner != rank {
                    rank_requests.entry(owner).or_default().insert(global_col);
                }
            }
        }
    }

    finalize_halo_plans_from_recv_requests(partition, &recv_requests)
}

/// Partition/communication metrics intended for comparing partition policies.
///
/// `cut_nnz` counts stored CSR references crossing ranks. In contrast,
/// `communication_volume` counts unique receive ghost values per rank and is
/// the number of scalar values transferred by one complete halo refresh under
/// the current plan.
#[derive(Clone, Debug, PartialEq)]
pub struct PartitionTelemetry {
    pub ranks: RankId,
    pub global_dofs: GlobalDofId,
    pub global_nnz: usize,
    pub cut_nnz: usize,
    pub communication_volume: usize,
    pub directional_peer_relations: usize,
    pub max_neighbors: usize,
    pub min_owned_dofs: usize,
    pub max_owned_dofs: usize,
    pub owned_dof_imbalance: f64,
    pub min_local_nnz: usize,
    pub max_local_nnz: usize,
    pub local_nnz_imbalance: f64,
}

pub fn partition_telemetry(
    matrix: &Csr32Matrix,
    partition: &ContiguousPartition,
    plans: &[HaloPlan],
) -> Result<PartitionTelemetry, DistributedTopologyError> {
    if plans.len() != partition.rank_count() as usize {
        return Err(DistributedTopologyError::InvalidPartition(
            "halo plan count differs from partition rank count",
        ));
    }

    let mut cut_nnz = 0usize;
    let mut local_nnz = Vec::with_capacity(plans.len());
    let mut owned_counts = Vec::with_capacity(plans.len());

    for plan in plans {
        let rank = plan.rank();
        let owned = plan.owned_range();
        let row_start = usize::try_from(owned.start)
            .map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;
        let row_end =
            usize::try_from(owned.end).map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;

        owned_counts.push(plan.owned_len());

        let mut rank_nnz = 0usize;
        for row in row_start..row_end {
            let start = matrix.row_ptr()[row] as usize;
            let end = matrix.row_ptr()[row + 1] as usize;
            rank_nnz += end - start;

            for &col in &matrix.col_idx()[start..end] {
                if partition.owner_of(GlobalDofId::from(col))? != rank {
                    cut_nnz += 1;
                }
            }
        }
        local_nnz.push(rank_nnz);
    }

    let communication_volume = plans.iter().map(HaloPlan::ghost_len).sum();
    let directional_peer_relations = plans.iter().map(|plan| plan.peers().len()).sum();
    let max_neighbors = plans
        .iter()
        .map(|plan| plan.peers().len())
        .max()
        .unwrap_or(0);

    let min_owned_dofs = *owned_counts.iter().min().unwrap_or(&0);
    let max_owned_dofs = *owned_counts.iter().max().unwrap_or(&0);
    let min_local_nnz = *local_nnz.iter().min().unwrap_or(&0);
    let max_local_nnz = *local_nnz.iter().max().unwrap_or(&0);

    let ranks_f64 = plans.len() as f64;
    let average_owned = partition.global_dofs() as f64 / ranks_f64;
    let average_nnz = matrix.nnz() as f64 / ranks_f64;

    let owned_dof_imbalance = if average_owned > 0.0 {
        max_owned_dofs as f64 / average_owned
    } else {
        1.0
    };
    let local_nnz_imbalance = if average_nnz > 0.0 {
        max_local_nnz as f64 / average_nnz
    } else {
        1.0
    };

    Ok(PartitionTelemetry {
        ranks: partition.rank_count(),
        global_dofs: partition.global_dofs(),
        global_nnz: matrix.nnz(),
        cut_nnz,
        communication_volume,
        directional_peer_relations,
        max_neighbors,
        min_owned_dofs,
        max_owned_dofs,
        owned_dof_imbalance,
        min_local_nnz,
        max_local_nnz,
        local_nnz_imbalance,
    })
}

#[cfg(test)]
mod g8_a2_tests {
    use super::*;

    fn chain_matrix(n: usize) -> Csr32Matrix {
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);

        for row in 0..n {
            if row > 0 {
                col_idx.push((row - 1) as u32);
                values.push(-1.0);
            }
            col_idx.push(row as u32);
            values.push(2.0);
            if row + 1 < n {
                col_idx.push((row + 1) as u32);
                values.push(-1.0);
            }
            row_ptr.push(col_idx.len() as u32);
        }

        Csr32Matrix::new(n, n, row_ptr, col_idx, values).unwrap()
    }

    #[test]
    fn rank_local_csr_preserves_global_row_order_and_values() {
        let matrix = chain_matrix(6);
        let partition = ContiguousPartition::balanced(6, 2).unwrap();
        let plans = build_contiguous_halo_plans(&matrix, &partition).unwrap();
        let local = prepare_rank_local_csrs(&matrix, &plans).unwrap();

        assert_eq!(local[0].owned_range(), 0..3);
        assert_eq!(local[0].extended_len(), 4);
        assert_eq!(local[0].nnz(), 8);

        assert_eq!(local[1].owned_range(), 3..6);
        assert_eq!(local[1].extended_len(), 4);
        assert_eq!(local[1].nnz(), 8);
    }

    #[test]
    fn simulated_halo_spmv_matches_serial_csr_bit_for_bit() {
        let matrix = chain_matrix(11);
        let partition = ContiguousPartition::balanced(11, 3).unwrap();
        let x: Vec<_> = (0..11).map(|i| (i as f64 + 1.0) / 7.0).collect();

        let serial = matrix.spmv(&x).unwrap();
        let distributed = simulated_distributed_spmv(&matrix, &partition, &x).unwrap();

        assert_eq!(distributed, serial);
    }

    #[test]
    fn abtm_halo_extraction_matches_csr_reference_exactly() {
        let matrix = chain_matrix(17);
        let partition = ContiguousPartition::balanced(17, 4).unwrap();

        let csr = build_contiguous_halo_plans(&matrix, &partition).unwrap();
        let abtm = build_contiguous_halo_plans_abtm(&matrix, &partition).unwrap();

        assert_eq!(abtm, csr);
    }

    #[test]
    fn abtm_halo_crosscheck_collapses_duplicate_remote_references() {
        let matrix = Csr32Matrix::new(
            4,
            4,
            vec![0, 1, 5, 8, 9],
            vec![0, 0, 1, 2, 2, 1, 2, 3, 3],
            vec![1.0; 9],
        )
        .unwrap();
        let partition = ContiguousPartition::balanced(4, 2).unwrap();

        let csr = build_contiguous_halo_plans(&matrix, &partition).unwrap();
        let abtm = build_contiguous_halo_plans_abtm(&matrix, &partition).unwrap();

        assert_eq!(abtm, csr);
        assert_eq!(abtm[0].ghost_globals(), &[2]);
    }

    #[test]
    fn telemetry_separates_cut_references_from_unique_communication_volume() {
        let matrix = chain_matrix(6);
        let partition = ContiguousPartition::balanced(6, 2).unwrap();
        let plans = build_contiguous_halo_plans(&matrix, &partition).unwrap();
        let telemetry = partition_telemetry(&matrix, &partition, &plans).unwrap();

        assert_eq!(telemetry.ranks, 2);
        assert_eq!(telemetry.global_dofs, 6);
        assert_eq!(telemetry.global_nnz, 16);
        assert_eq!(telemetry.cut_nnz, 2);
        assert_eq!(telemetry.communication_volume, 2);
        assert_eq!(telemetry.directional_peer_relations, 2);
        assert_eq!(telemetry.max_neighbors, 1);
        assert_eq!(telemetry.min_owned_dofs, 3);
        assert_eq!(telemetry.max_owned_dofs, 3);
        assert_eq!(telemetry.min_local_nnz, 8);
        assert_eq!(telemetry.max_local_nnz, 8);
        assert_eq!(telemetry.owned_dof_imbalance, 1.0);
        assert_eq!(telemetry.local_nnz_imbalance, 1.0);
    }
}
