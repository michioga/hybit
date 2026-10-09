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
