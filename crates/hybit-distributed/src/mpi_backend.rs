//! G8-B1: optional MPI runtime/collective boundary.
//!
//! Available only with the `mpi` Cargo feature. No MPI calls or native MPI
//! linkage are needed for ordinary HyBIT builds. B1 intentionally does not
//! implement halo exchange or distributed Krylov; those belong to G8-B2/C.
//!
//! MPI is initialized with `MPI_THREAD_SINGLE`. Call all methods consistently
//! from the same thread and in the same collective order across all ranks.

use mpi::collective::SystemOperation;
use mpi::environment::Universe;
use mpi::traits::*;
use std::io;

/// Owns MPI initialization and finalizes MPI when dropped.
///
/// Do not initialize another MPI universe in the same process. The lifetime
/// must enclose every communicator operation, including all rank-local work.
pub struct MpiRuntime {
    universe: Universe,
}

impl MpiRuntime {
    /// Initialize MPI exactly once. An already-initialized process is rejected
    /// instead of falsely claiming ownership of MPI finalization.
    pub fn initialize() -> io::Result<Self> {
        let universe = mpi::initialize().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::AlreadyExists,
                "MPI is already initialized; HyBIT G8-B1 requires ownership of MPI initialization",
            )
        })?;
        Ok(Self { universe })
    }

    /// Rank within MPI_COMM_WORLD (zero-based).
    pub fn rank(&self) -> i32 {
        self.universe.world().rank()
    }

    /// Process count in MPI_COMM_WORLD.
    pub fn size(&self) -> i32 {
        self.universe.world().size()
    }

    /// Synchronize all processes. All ranks must enter this collective.
    pub fn barrier(&self) {
        self.universe.world().barrier();
    }

    /// Collective scalar reduction used for distributed Krylov dot products.
    pub fn all_reduce_sum_f64(&self, local: f64) -> f64 {
        let mut result = 0.0;
        self.universe
            .world()
            .all_reduce_into(&local, &mut result, SystemOperation::sum());
        result
    }

    /// Collective integer reduction for global DOF and instrumentation counts.
    pub fn all_reduce_sum_u64(&self, local: u64) -> u64 {
        let mut result = 0u64;
        self.universe
            .world()
            .all_reduce_into(&local, &mut result, SystemOperation::sum());
        result
    }

    /// Elementwise SUM reduction. All ranks must pass identical buffer sizes.
    ///
    /// Dimension validation here is local; the caller must agree on count
    /// across ranks before entering this collective, or MPI can deadlock.
    pub fn all_reduce_sum_slice_f64(&self, local: &[f64], global: &mut [f64]) -> io::Result<()> {
        if local.len() != global.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "MPI vector allreduce lengths differ",
            ));
        }
        self.universe
            .world()
            .all_reduce_into(local, global, SystemOperation::sum());
        Ok(())
    }

    /// One-step ring exchange: send `value` to next rank, receive from prior.
    ///
    /// Both sides are scheduled in one MPI_Sendrecv operation so no process
    /// waits on a blocking send. This is a B1 transport smoke check, not halo.
    pub fn ring_exchange_i32(&self, value: i32) -> i32 {
        let world = self.universe.world();
        let rank = world.rank();
        let size = world.size();
        let next = (rank + 1) % size;
        let previous = (rank + size - 1) % size;
        let (received, _) = mpi::point_to_point::send_receive(
            &value,
            &world.process_at_rank(next),
            &world.process_at_rank(previous),
        );
        received
    }
}

// -----------------------------------------------------------------------------
// G8-B2: deterministic, synchronous, rank-local halo exchange and CSR SpMV.
// This is a correctness-first transport; G8-B3 will address persistent buffers
// and communication/computation overlap. All ranks must use reciprocal plans.
// -----------------------------------------------------------------------------

use super::{HaloPlan, RankLocalCsr};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default)]
pub struct HaloTraffic {
    /// Number of peer pairs visited by this rank, including one-way traffic.
    pub peers: usize,
    /// f64 payload elements sent (does not include count-handshake messages).
    pub sent_values: usize,
    /// f64 payload elements received.
    pub received_values: usize,
    /// Wall-clock duration of the blocking exchange, including packing.
    pub elapsed_ns: u128,
}

impl HaloTraffic {
    pub fn sent_payload_bytes(self) -> usize {
        self.sent_values * std::mem::size_of::<f64>()
    }

    pub fn received_payload_bytes(self) -> usize {
        self.received_values * std::mem::size_of::<f64>()
    }
}

impl MpiRuntime {
    /// Fill `[owned | ghosts]` using the prepared reciprocal HaloPlan.
    ///
    /// The send/receive order is ascending peer rank on every process.
    /// Each undirected peer edge occurs on both endpoint plans, even if all
    /// numerical values flow in only one direction. For a correct reciprocal
    /// undirected peer graph, the common order avoids a blocking wait cycle.
    /// A count handshake rejects incompatible receive lengths before data.
    ///
    /// This routine assumes all ranks have entered the same SpMV epoch and
    /// hold mutually consistent plans. It is not safe to invoke concurrently
    /// from multiple threads or with unrelated MPI point-to-point traffic.
    pub fn exchange_halo(
        &self,
        plan: &HaloPlan,
        x_owned: &[f64],
        extended: &mut [f64],
    ) -> io::Result<HaloTraffic> {
        let invalid = |message: &'static str| io::Error::new(io::ErrorKind::InvalidInput, message);
        if self.rank() < 0 || self.size() < 1 || plan.rank() != self.rank() as u32 {
            return Err(invalid("MPI rank does not match HaloPlan"));
        }
        if x_owned.len() != plan.owned_len() || extended.len() != plan.extended_len() {
            return Err(invalid(
                "owned/extended vector length differs from HaloPlan",
            ));
        }

        // Perform all local validation before first peer operation.
        let mut filled = vec![false; plan.ghost_len()];
        let mut last_peer: Option<u32> = None;
        for peer in plan.peers() {
            if peer.peer() >= self.size() as u32 || peer.peer() == plan.rank() {
                return Err(invalid("halo peer rank is invalid"));
            }
            if last_peer.is_some_and(|prev| prev >= peer.peer()) {
                return Err(invalid("halo peers must be unique and rank-sorted"));
            }
            last_peer = Some(peer.peer());
            if peer.send_globals().len() != peer.send_owned_indices().len()
                || peer.recv_globals().len() != peer.recv_extended_indices().len()
            {
                return Err(invalid("halo peer value/index counts differ"));
            }
            for &slot in peer.send_owned_indices() {
                if slot as usize >= x_owned.len() {
                    return Err(invalid("halo send index outside locally owned vector"));
                }
            }
            for &slot in peer.recv_extended_indices() {
                let index = slot as usize;
                if index < plan.owned_len() || index >= extended.len() {
                    return Err(invalid("halo receive index outside ghost region"));
                }
                let flag = &mut filled[index - plan.owned_len()];
                if *flag {
                    return Err(invalid("duplicate halo destination index"));
                }
                *flag = true;
            }
        }
        if filled.iter().any(|&flag| !flag) {
            return Err(invalid("halo plan leaves a ghost slot without a sender"));
        }

        let started = Instant::now();
        extended[..x_owned.len()].copy_from_slice(x_owned);
        extended[x_owned.len()..].fill(f64::NAN);
        let world = self.universe.world();
        let mut traffic = HaloTraffic::default();

        for peer in plan.peers() {
            let target = world.process_at_rank(peer.peer() as i32);
            let send_count = u64::try_from(peer.send_owned_indices().len())
                .map_err(|_| invalid("MPI send count exceeds u64"))?;
            let recv_count = u64::try_from(peer.recv_extended_indices().len())
                .map_err(|_| invalid("MPI receive count exceeds u64"))?;
            let (remote_count, _): (u64, _) =
                mpi::point_to_point::send_receive(&send_count, &target, &target);
            if remote_count != recv_count {
                return Err(invalid("non-reciprocal MPI halo send/receive counts"));
            }

            let send_buffer: Vec<f64> = peer
                .send_owned_indices()
                .iter()
                .map(|&i| x_owned[i as usize])
                .collect();
            let mut recv_buffer = vec![0.0; peer.recv_extended_indices().len()];
            mpi::point_to_point::send_receive_into(
                send_buffer.as_slice(),
                &target,
                recv_buffer.as_mut_slice(),
                &target,
            );
            for (&index, &value) in peer.recv_extended_indices().iter().zip(&recv_buffer) {
                extended[index as usize] = value;
            }
            traffic.peers += 1;
            traffic.sent_values += send_buffer.len();
            traffic.received_values += recv_buffer.len();
        }
        traffic.elapsed_ns = started.elapsed().as_nanos();
        Ok(traffic)
    }

    /// MPI-backed rank-local CSR SpMV; `extended` is reusable caller storage.
    /// All MPI ranks must invoke it in the same epoch using reciprocal plans.
    pub fn spmv_local(
        &self,
        plan: &HaloPlan,
        operator: &RankLocalCsr,
        x_owned: &[f64],
        extended: &mut [f64],
        y_owned: &mut [f64],
    ) -> io::Result<HaloTraffic> {
        if operator.rank() != plan.rank()
            || operator.owned_range() != plan.owned_range()
            || operator.extended_len() != plan.extended_len()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "local CSR and halo plan are incompatible",
            ));
        }
        let traffic = self.exchange_halo(plan, x_owned, extended)?;
        operator
            .apply_extended(extended, y_owned)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(traffic)
    }
}
