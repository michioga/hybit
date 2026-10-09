//! G8-B3: optional nonblocking MPI halo, reusable message buffers and row overlap.
//!
//! All ranks must collectively construct this object with mutually consistent
//! HaloPlans, then call `spmv_overlap` in matching epochs. `mpi` uses
//! MPI_THREAD_SINGLE: calls must remain on the initializing thread.
//! Nonblocking sends/receives may or may not progress during local computation;
//! benchmark results (not API semantics) decide whether overlap is effective.

use crate::mpi_backend::MpiRuntime;
use crate::{HaloPlan, RankLocalCsr};
use mpi::traits::*;
use std::io;
use std::time::Instant;

fn invalid(what: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, what)
}

#[derive(Debug)]
struct PeerSegment {
    rank: i32,
    send_slots: Vec<u32>,
    recv_slots: Vec<u32>,
}

/// Reusable rank-local MPI buffers and interior/boundary row schedule.
/// Build collectively; the constructor performs one MPI_Alltoall count check.
#[derive(Debug)]
pub struct OverlapSpmv {
    rank: i32,
    owned_len: usize,
    extended: Vec<f64>,
    send_buffer: Vec<f64>,
    recv_buffer: Vec<f64>,
    peers: Vec<PeerSegment>,
    interior_rows: Vec<usize>,
    boundary_rows: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OverlapTiming {
    pub elapsed_ns: u128,
    pub pack_ns: u128,
    pub post_ns: u128,
    pub interior_ns: u128,
    pub wait_ns: u128,
    pub scatter_ns: u128,
    pub boundary_ns: u128,
    pub send_values: usize,
    pub recv_values: usize,
    pub interior_rows: usize,
    pub boundary_rows: usize,
}

fn check_layout(plan: &HaloPlan, local: &RankLocalCsr, rank: i32, size: i32) -> io::Result<()> {
    if rank < 0 || size <= 0 || plan.rank() != rank as u32 {
        return Err(invalid("MPI communicator and HaloPlan rank mismatch"));
    }
    if plan.rank() != local.rank()
        || plan.owned_range() != local.owned_range()
        || plan.extended_len() != local.extended_len()
    {
        return Err(invalid("HaloPlan and rank-local CSR are incompatible"));
    }
    let mut slots = vec![false; plan.ghost_len()];
    let mut prev_peer: Option<u32> = None;
    for peer in plan.peers() {
        if peer.peer() >= size as u32 || peer.peer() == plan.rank() {
            return Err(invalid("invalid remote MPI rank in HaloPlan"));
        }
        if prev_peer.is_some_and(|p| p >= peer.peer()) {
            return Err(invalid("HaloPlan peer list must be sorted and unique"));
        }
        prev_peer = Some(peer.peer());
        if peer.send_globals().len() != peer.send_owned_indices().len()
            || peer.recv_globals().len() != peer.recv_extended_indices().len()
        {
            return Err(invalid("HaloPlan peer global/index lengths differ"));
        }
        for &slot in peer.send_owned_indices() {
            if slot as usize >= plan.owned_len() {
                return Err(invalid("send index out of owned vector range"));
            }
        }
        for &slot in peer.recv_extended_indices() {
            let index = slot as usize;
            if index < plan.owned_len() || index >= plan.extended_len() {
                return Err(invalid("received ghost index out of range"));
            }
            let seen = &mut slots[index - plan.owned_len()];
            if *seen {
                return Err(invalid("duplicate ghost destination index"));
            }
            *seen = true;
        }
    }
    if slots.iter().any(|&seen| !seen) {
        return Err(invalid("an extended-vector ghost slot has no sender"));
    }
    Ok(())
}

/// Matrix-row partition independent of MPI; a row is interior exactly when
/// none of its CSR columns refer to the ghost part of `[owned | ghosts]`.
fn partition_rows(local: &RankLocalCsr) -> (Vec<usize>, Vec<usize>) {
    let mut interior = Vec::new();
    let mut boundary = Vec::new();
    for row in 0..local.owned_len() {
        let start = local.row_ptr()[row] as usize;
        let end = local.row_ptr()[row + 1] as usize;
        if local.col_idx()[start..end]
            .iter()
            .any(|&col| col as usize >= local.owned_len())
        {
            boundary.push(row);
        } else {
            interior.push(row);
        }
    }
    (interior, boundary)
}

fn rows_spmv(local: &RankLocalCsr, rows: &[usize], x: &[f64], y: &mut [f64]) {
    for &row in rows {
        let start = local.row_ptr()[row] as usize;
        let end = local.row_ptr()[row + 1] as usize;
        let mut sum = 0.0;
        for pos in start..end {
            sum += local.values()[pos] * x[local.col_idx()[pos] as usize];
        }
        y[row] = sum;
    }
}

impl OverlapSpmv {
    /// Collective setup. The all-to-all count exchange detects one-way peers
    /// and nonreciprocal message lengths before nonblocking message posting.
    /// As with all MPI collectives, invalid local input must not be supplied
    /// on only a subset of ranks (the application must prevalidate globally).
    pub fn prepare(mpi: &MpiRuntime, plan: &HaloPlan, local: &RankLocalCsr) -> io::Result<Self> {
        check_layout(plan, local, mpi.rank(), mpi.size())?;
        let size = mpi.size() as usize;
        let mut advertised = vec![0u64; size];
        let mut expected = vec![0u64; size];
        let mut peers = Vec::with_capacity(plan.peers().len());
        let mut total_send = 0usize;
        let mut total_recv = 0usize;
        for p in plan.peers() {
            let rank = p.peer() as usize;
            advertised[rank] = p.send_owned_indices().len() as u64;
            expected[rank] = p.recv_extended_indices().len() as u64;
            total_send += p.send_owned_indices().len();
            total_recv += p.recv_extended_indices().len();
            peers.push(PeerSegment {
                rank: p.peer() as i32,
                send_slots: p.send_owned_indices().to_vec(),
                recv_slots: p.recv_extended_indices().to_vec(),
            });
        }
        let mut actual = vec![0u64; size];
        let world = mpi::topology::SimpleCommunicator::world();
        world.all_to_all_into(&advertised[..], &mut actual[..]);
        if actual != expected {
            return Err(invalid("MPI HaloPlan count reciprocity check failed"));
        }
        let (interior_rows, boundary_rows) = partition_rows(local);
        Ok(Self {
            rank: mpi.rank(),
            owned_len: plan.owned_len(),
            extended: vec![0.0; plan.extended_len()],
            send_buffer: vec![0.0; total_send],
            recv_buffer: vec![0.0; total_recv],
            peers,
            interior_rows,
            boundary_rows,
        })
    }

    pub fn interior_len(&self) -> usize {
        self.interior_rows.len()
    }
    pub fn boundary_len(&self) -> usize {
        self.boundary_rows.len()
    }
    pub fn extended(&self) -> &[f64] {
        &self.extended
    }

    /// Post all receives, post all sends, compute interior rows, wait for every
    /// request, scatter ghosts and compute boundary rows. Buffers are reused
    /// across calls; all nonblocking requests are completed before returning.
    pub fn spmv_overlap(
        &mut self,
        mpi: &MpiRuntime,
        local: &RankLocalCsr,
        owned: &[f64],
        y: &mut [f64],
    ) -> io::Result<OverlapTiming> {
        if mpi.rank() != self.rank
            || local.rank() as i32 != self.rank
            || owned.len() != self.owned_len
            || y.len() != self.owned_len
            || local.extended_len() != self.extended.len()
        {
            return Err(invalid("overlap SpMV rank or vector dimensions mismatch"));
        }
        let total = Instant::now();
        self.extended[..self.owned_len].copy_from_slice(owned);
        self.extended[self.owned_len..].fill(f64::NAN);
        let t_pack = Instant::now();
        let mut cursor = 0;
        for peer in &self.peers {
            for &i in &peer.send_slots {
                self.send_buffer[cursor] = owned[i as usize];
                cursor += 1;
            }
        }
        let pack_ns = t_pack.elapsed().as_nanos();
        let mut post_ns = 0u128;
        let mut interior_ns = 0u128;
        let mut wait_ns = 0u128;
        let world = mpi::topology::SimpleCommunicator::world();
        mpi::request::scope(|scope| {
            let t_post = Instant::now();
            let mut recv_requests = Vec::new();
            let mut recv_remaining: &mut [f64] = &mut self.recv_buffer;
            for peer in &self.peers {
                let remaining = std::mem::take(&mut recv_remaining);
                let (chunk, tail) = remaining.split_at_mut(peer.recv_slots.len());
                recv_remaining = tail;
                if !chunk.is_empty() {
                    recv_requests.push(
                        world
                            .process_at_rank(peer.rank)
                            .immediate_receive_into(scope, chunk),
                    );
                }
            }
            let mut send_requests = Vec::new();
            let mut send_remaining: &[f64] = &self.send_buffer;
            for peer in &self.peers {
                let (chunk, tail) = send_remaining.split_at(peer.send_slots.len());
                send_remaining = tail;
                if !chunk.is_empty() {
                    send_requests.push(
                        world
                            .process_at_rank(peer.rank)
                            .immediate_send(scope, chunk),
                    );
                }
            }
            post_ns = t_post.elapsed().as_nanos();
            let t_interior = Instant::now();
            rows_spmv(local, &self.interior_rows, &self.extended, y);
            interior_ns = t_interior.elapsed().as_nanos();
            let t_wait = Instant::now();
            for req in recv_requests {
                req.wait();
            }
            for req in send_requests {
                req.wait();
            }
            wait_ns = t_wait.elapsed().as_nanos();
        });
        let t_scatter = Instant::now();
        let mut cursor = 0;
        for peer in &self.peers {
            for &slot in &peer.recv_slots {
                self.extended[slot as usize] = self.recv_buffer[cursor];
                cursor += 1;
            }
        }
        let scatter_ns = t_scatter.elapsed().as_nanos();
        let t_boundary = Instant::now();
        rows_spmv(local, &self.boundary_rows, &self.extended, y);
        let boundary_ns = t_boundary.elapsed().as_nanos();
        Ok(OverlapTiming {
            elapsed_ns: total.elapsed().as_nanos(),
            pack_ns,
            post_ns,
            interior_ns,
            wait_ns,
            scatter_ns,
            boundary_ns,
            send_values: self.send_buffer.len(),
            recv_values: self.recv_buffer.len(),
            interior_rows: self.interior_rows.len(),
            boundary_rows: self.boundary_rows.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_contiguous_halo_plans, prepare_rank_local_csr, ContiguousPartition};
    use hybit_matrix::Csr32Matrix;

    #[test]
    fn split_rows_detects_ghost_dependency() {
        let a = Csr32Matrix::new(
            4,
            4,
            vec![0, 2, 5, 8, 10],
            vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
            vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
        )
        .unwrap();
        let part = ContiguousPartition::balanced(4, 2).unwrap();
        let plans = build_contiguous_halo_plans(&a, &part).unwrap();
        let local = prepare_rank_local_csr(&a, &plans[0]).unwrap();
        let (interior, boundary) = partition_rows(&local);
        assert_eq!(interior, vec![0]);
        assert_eq!(boundary, vec![1]);
        let x = vec![2.0, 3.0, 5.0];
        let mut y = vec![f64::NAN; 2];
        rows_spmv(&local, &interior, &x, &mut y);
        rows_spmv(&local, &boundary, &x, &mut y);
        assert_eq!(y, vec![1.0, -1.0]);
    }
}
