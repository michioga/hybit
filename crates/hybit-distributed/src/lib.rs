use hybit_matrix::Csr32Matrix;
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
