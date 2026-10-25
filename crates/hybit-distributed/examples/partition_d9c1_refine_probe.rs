//! H1-C3K8-D9C1: experimental communication-aware boundary refinement.
//! This is a probe, NOT the production partitioner. Full CSR reference
//! telemetry gates every accepted refinement pass.
use hybit_distributed::{
    AbtmMultilevelOptions, PartitionAssignment, PartitionTelemetry, abtm_multilevel_partition,
    partition_telemetry_assignment,
};
use hybit_matrix::{Csr32Matrix, read_matrix_market};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::time::Instant;

type ProbeResult<T> = Result<T, Box<dyn Error>>;

/// Number of CSR references to a column by rows belonging to a rank.
/// Exact ghost contribution of column c is the number of r != owner[c]
/// with demand[r,c] > 0. This is not the METIS edge-cut objective.
struct BoundaryState<'a> {
    a: &'a Csr32Matrix,
    incoming: Vec<Vec<usize>>,
    owners: Vec<u32>,
    loads: Vec<usize>,
    demand: Vec<u32>, // rank-major: [rank * n + col]
    n: usize,
    ranks: usize,
    floor: usize,
    cap: usize,
}

impl<'a> BoundaryState<'a> {
    fn new(
        a: &'a Csr32Matrix,
        assignment: &PartitionAssignment,
        imbalance: u32,
    ) -> ProbeResult<Self> {
        let n = a.nrows();
        if n != a.ncols() || n == 0 || assignment.global_dofs() != n as u64 {
            return Err("D9C1 requires a nonempty square CSR and matching owners".into());
        }
        let ranks = assignment.rank_count() as usize;
        if ranks == 0 || ranks > 64 || imbalance > 500 {
            return Err("D9C1 ranks must be 1..64 and imbalance <=500 per-mille".into());
        }
        let mut incoming = vec![Vec::<usize>::new(); n];
        let mut demand = vec![
            0u32;
            n.checked_mul(ranks)
                .ok_or("D9C1 demand allocation overflow")?
        ];
        let mut loads = vec![0usize; ranks];
        let owners = assignment.owners().to_vec();
        for (row, &owner) in owners.iter().enumerate() {
            let r = owner as usize;
            if r >= ranks {
                return Err("D9C1 invalid rank label".into());
            }
            loads[r] += 1;
            let mut last_col = None;
            for idx in a.row_ptr()[row] as usize..a.row_ptr()[row + 1] as usize {
                let col = a.col_idx()[idx] as usize;
                if col >= n {
                    return Err("D9C1 invalid CSR column".into());
                }
                if last_col.is_some_and(|prior| col <= prior) {
                    return Err("D9C1 requires unique increasing CSR columns".into());
                }
                last_col = Some(col);
                incoming[col].push(row);
                let key = r * n + col;
                demand[key] = demand[key]
                    .checked_add(1)
                    .ok_or("D9C1 demand count overflow")?;
            }
        }
        let policy_floor =
            (n as u128 * u128::from(1000 - imbalance) / (ranks as u128 * 1000)) as usize;
        let cap = (n as u128 * u128::from(1000 + imbalance) / (ranks as u128 * 1000)) as usize;
        // Current A6 enforces an upper load cap, not a symmetric lower cap.
        // Preserve its existing minimum load as a floor rather than refusing
        // known-good A6 assignments (thermal1, four ranks in particular).
        let floor = policy_floor.min(*loads.iter().min().ok_or("D9C1 no rank loads")?);
        if loads.iter().any(|&x| x == 0 || x > cap) {
            return Err("D9C1 initial ownership violates rank or upper-cap bounds".into());
        }
        Ok(Self {
            a,
            incoming,
            owners,
            loads,
            demand,
            n,
            ranks,
            floor,
            cap,
        })
    }

    fn row_cols(&self, row: usize) -> impl Iterator<Item = usize> + '_ {
        (self.a.row_ptr()[row] as usize..self.a.row_ptr()[row + 1] as usize)
            .map(|idx| self.a.col_idx()[idx] as usize)
    }

    fn feasible(&self, node: usize, target: usize) -> bool {
        let src = self.owners[node] as usize;
        src != target
            && self.loads[src] > self.floor
            && self.loads[src] > 1
            && self.loads[target] < self.cap
    }

    /// Exact delta in HyBIT's directed CSR cut and unique rank-column ghosts.
    /// Does not scan the whole graph; examines one row, its incoming column,
    /// and at most ranks * (row length + 1) demand entries.
    fn delta(&self, node: usize, target: usize) -> ProbeResult<(i64, i64)> {
        let source = self.owners[node] as usize;
        if !self.feasible(node, target) {
            return Err("D9C1 infeasible move".into());
        }
        let mut cut = 0i64;
        for col in self.row_cols(node) {
            if col != node {
                let p = self.owners[col] as usize;
                cut += i64::from(target != p) - i64::from(source != p);
            }
        }
        for &row in &self.incoming[node] {
            if row != node {
                let p = self.owners[row] as usize;
                cut += i64::from(p != target) - i64::from(p != source);
            }
        }
        let mut ghosts = 0i64;
        let mut self_col_seen = false;
        for col in self.row_cols(node) {
            if col == node {
                self_col_seen = true;
            }
            ghosts += self.ghost_delta_for_col(node, col, source, target, true)?;
        }
        if !self_col_seen {
            ghosts += self.ghost_delta_for_col(node, node, source, target, false)?;
        }
        Ok((ghosts, cut))
    }

    fn ghost_delta_for_col(
        &self,
        node: usize,
        col: usize,
        source: usize,
        target: usize,
        row_col: bool,
    ) -> ProbeResult<i64> {
        let old_owner = self.owners[col] as usize;
        let new_owner = if col == node { target } else { old_owner };
        let mut delta = 0i64;
        for r in 0..self.ranks {
            let count = i64::from(self.demand[r * self.n + col]);
            let adjust = if row_col && r == source {
                -1
            } else if row_col && r == target {
                1
            } else {
                0
            };
            let post = count + adjust;
            if post < 0 {
                return Err("D9C1 underflow in ghost delta".into());
            }
            delta += i64::from(post > 0 && r != new_owner) - i64::from(count > 0 && r != old_owner);
        }
        Ok(delta)
    }

    fn move_node(&mut self, node: usize, target: usize) -> ProbeResult<()> {
        let source = self.owners[node] as usize;
        if !self.feasible(node, target) {
            return Err("D9C1 infeasible move application".into());
        }
        for idx in self.a.row_ptr()[node] as usize..self.a.row_ptr()[node + 1] as usize {
            let col = self.a.col_idx()[idx] as usize;
            self.demand[source * self.n + col] -= 1;
            self.demand[target * self.n + col] += 1;
        }
        self.owners[node] = target as u32;
        self.loads[source] -= 1;
        self.loads[target] += 1;
        Ok(())
    }

    fn assignment(&self) -> ProbeResult<PartitionAssignment> {
        Ok(PartitionAssignment::from_owners(
            self.ranks as u32,
            self.owners.clone(),
        )?)
    }
}

#[derive(Default)]
struct RefinementStats {
    passes: usize,
    accepted_moves: usize,
    examined_moves: usize,
}

fn refine(
    matrix: &Csr32Matrix,
    initial: &PartitionAssignment,
    passes: usize,
    max_moves: usize,
    imbalance: u32,
) -> ProbeResult<(PartitionAssignment, RefinementStats)> {
    let mut state = BoundaryState::new(matrix, initial, imbalance)?;
    let mut before = partition_telemetry_assignment(matrix, initial)?;
    let mut stats = RefinementStats::default();
    for _ in 0..passes {
        let mut moves = Vec::<(usize, usize, usize)>::new();
        let mut total_ghost = 0i64;
        let mut total_cut = 0i64;
        let mut best_prefix = 0usize;
        let mut best_score = 0i64;
        let mut targets = vec![false; state.ranks];
        for node in 0..state.n {
            if moves.len() == max_moves {
                break;
            }
            let source = state.owners[node] as usize;
            if state.loads[source] <= state.floor || state.loads[source] <= 1 {
                continue;
            }
            targets.fill(false);
            for col in state.row_cols(node) {
                targets[state.owners[col] as usize] = true;
            }
            for &row in &state.incoming[node] {
                targets[state.owners[row] as usize] = true;
            }
            targets[source] = false;
            let mut best: Option<(i64, i64, i64, usize)> = None;
            for (target, &is_neighbor) in targets.iter().enumerate() {
                if !is_neighbor || !state.feasible(node, target) {
                    continue;
                }
                let (ghost, cut) = state.delta(node, target)?;
                stats.examined_moves += 1;
                let score = 4 * ghost + cut;
                // Bounded temporary uphill moves (FM-style); every accepted
                // prefix must improve BOTH exact ghost volume and cut.
                if score > 6 {
                    continue;
                }
                let key = (score, ghost, cut, target);
                if best.is_none_or(|x| key < x) {
                    best = Some(key);
                }
            }
            if let Some((_, ghost, cut, target)) = best {
                state.move_node(node, target)?;
                moves.push((node, source, target));
                total_ghost += ghost;
                total_cut += cut;
                let score = 4 * total_ghost + total_cut;
                if total_ghost <= 0 && total_cut <= 0 && score < best_score {
                    best_prefix = moves.len();
                    best_score = score;
                }
            }
        }
        for &(node, source, _) in moves[best_prefix..].iter().rev() {
            state.move_node(node, source)?;
        }
        if best_prefix == 0 {
            break;
        }
        let candidate = state.assignment()?;
        let after = partition_telemetry_assignment(matrix, &candidate)?;
        // Independent full CSR telemetry is the acceptance oracle.
        if after.communication_volume > before.communication_volume
            || after.cut_nnz > before.cut_nnz
            || after.directional_peer_relations > before.directional_peer_relations
        {
            for &(node, source, _) in moves[..best_prefix].iter().rev() {
                state.move_node(node, source)?;
            }
            break;
        }
        if after.communication_volume == before.communication_volume
            && after.cut_nnz == before.cut_nnz
        {
            return Err("D9C1 accepted a zero-gain move sequence".into());
        }
        stats.passes += 1;
        stats.accepted_moves += best_prefix;
        before = after;
    }
    Ok((state.assignment()?, stats))
}

fn write_labels(path: &str, assignment: &PartitionAssignment) -> ProbeResult<()> {
    let mut file = BufWriter::new(File::create_new(path)?);
    for &owner in assignment.owners() {
        writeln!(file, "{owner}")?;
    }
    file.flush()?;
    Ok(())
}

fn emit_metrics(writer: &mut impl Write, metrics: &PartitionTelemetry) -> ProbeResult<()> {
    write!(
        writer,
        ",{},{},{},{},{:.9},{:.9}",
        metrics.cut_nnz,
        metrics.communication_volume,
        metrics.directional_peer_relations,
        metrics.max_neighbors,
        metrics.owned_dof_imbalance,
        metrics.local_nnz_imbalance
    )?;
    Ok(())
}

fn run() -> ProbeResult<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 6 {
        return Err("usage: partition_d9c1_refine_probe <matrix.mtx> <4|8> <output.labels> <output.csv> <passes> <max_moves>".into());
    }
    let ranks: u32 = args[1].parse()?;
    let passes: usize = args[4].parse()?;
    let max_moves: usize = args[5].parse()?;
    if !matches!(ranks, 4 | 8) || !(1..=5).contains(&passes) || !(1..=100_000).contains(&max_moves)
    {
        return Err("D9C1 ranks/passes/budget outside safety bounds".into());
    }
    if fs::metadata(&args[0])?.len() > 512 * 1024 * 1024 {
        return Err("D9C1 input matrix exceeds 512 MiB bound".into());
    }
    if fs::metadata(&args[2]).is_ok() || fs::metadata(&args[3]).is_ok() {
        return Err("D9C1 refuses to overwrite labels or CSV".into());
    }
    let (matrix, _) = read_matrix_market(&args[0])?;
    if matrix.nrows() > 1_200_000 || matrix.nnz() > 16_000_000 {
        return Err("D9C1 input exceeds matrix bounds".into());
    }
    let options = AbtmMultilevelOptions::default();
    let start = Instant::now();
    let (initial, _) = abtm_multilevel_partition(&matrix, ranks, options)?;
    let baseline_ms = start.elapsed().as_secs_f64() * 1_000.0;
    let before = partition_telemetry_assignment(&matrix, &initial)?;
    let started = Instant::now();
    let (improved, stats) = refine(
        &matrix,
        &initial,
        passes,
        max_moves,
        options.imbalance_per_mille,
    )?;
    let refine_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let after = partition_telemetry_assignment(&matrix, &improved)?;
    if after.cut_nnz > before.cut_nnz
        || after.communication_volume > before.communication_volume
        || after.directional_peer_relations > before.directional_peer_relations
        || after.owned_dof_imbalance > 1.031
    {
        return Err("D9C1 final monotonic quality gate failed".into());
    }
    write_labels(&args[2], &improved)?;
    let mut csv = BufWriter::new(File::create_new(&args[3])?);
    writeln!(
        csv,
        "matrix,ranks,n,nnz,baseline_cut,baseline_ghosts,baseline_peers,baseline_max_neighbors,baseline_dof_imbalance,baseline_nnz_imbalance,refined_cut,refined_ghosts,refined_peers,refined_max_neighbors,refined_dof_imbalance,refined_nnz_imbalance,baseline_ms,refine_ms,accepted_passes,accepted_moves,examined_moves,labels_path,verified"
    )?;
    write!(
        csv,
        "{},{},{},{}",
        args[0],
        ranks,
        matrix.nrows(),
        matrix.nnz()
    )?;
    emit_metrics(&mut csv, &before)?;
    emit_metrics(&mut csv, &after)?;
    writeln!(
        csv,
        ",{:.6},{:.6},{},{},{},{},true",
        baseline_ms, refine_ms, stats.passes, stats.accepted_moves, stats.examined_moves, args[2]
    )?;
    csv.flush()?;
    println!(
        "D9C1 PASS ranks={} baseline_cut={} refined_cut={} baseline_ghosts={} refined_ghosts={} baseline_peers={} refined_peers={} A6_ms={:.3} refine_ms={:.3} moves={}",
        ranks,
        before.cut_nnz,
        after.cut_nnz,
        before.communication_volume,
        after.communication_volume,
        before.directional_peer_relations,
        after.directional_peer_relations,
        baseline_ms,
        refine_ms,
        stats.accepted_moves
    );
    Ok(())
}
fn main() -> ProbeResult<()> {
    run()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tiny() -> Csr32Matrix {
        let cols = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5];
        let rows = vec![0, 2, 5, 8, 11, 14, 16];
        Csr32Matrix::new(6, 6, rows, cols, vec![1.0; 16]).unwrap()
    }
    fn asymmetric() -> Csr32Matrix {
        let cols = vec![0, 2, 1, 4, 0, 2, 3, 1, 3, 2, 4, 5, 0, 5];
        let rows = vec![0, 2, 4, 7, 9, 12, 14];
        Csr32Matrix::new(6, 6, rows, cols, vec![1.0; 14]).unwrap()
    }
    #[test]
    fn exact_incremental_delta_matches_hybit_for_symmetric_and_directed_csr() {
        for a in [tiny(), asymmetric()] {
            let p = PartitionAssignment::from_owners(3, vec![0, 0, 0, 1, 1, 2]).unwrap();
            for node in 0..a.nrows() {
                for target in 0..3usize {
                    let mut state = BoundaryState::new(&a, &p, 500).unwrap();
                    if !state.feasible(node, target) {
                        continue;
                    }
                    let orig = partition_telemetry_assignment(&a, &p).unwrap();
                    let (dg, dc) = state.delta(node, target).unwrap();
                    state.move_node(node, target).unwrap();
                    let next =
                        partition_telemetry_assignment(&a, &state.assignment().unwrap()).unwrap();
                    assert_eq!(
                        dg,
                        next.communication_volume as i64 - orig.communication_volume as i64
                    );
                    assert_eq!(dc, next.cut_nnz as i64 - orig.cut_nnz as i64);
                    state.move_node(node, p.owners()[node] as usize).unwrap();
                    assert_eq!(state.owners, p.owners());
                }
            }
        }
    }
    #[test]
    fn refinement_never_worsens_reference_cut_or_ghost_or_peers() {
        let a = tiny();
        let p = PartitionAssignment::from_owners(3, vec![0, 0, 0, 1, 1, 2]).unwrap();
        let (next, _) = refine(&a, &p, 2, 20, 500).unwrap();
        let old = partition_telemetry_assignment(&a, &p).unwrap();
        let new = partition_telemetry_assignment(&a, &next).unwrap();
        assert!(new.cut_nnz <= old.cut_nnz);
        assert!(new.communication_volume <= old.communication_volume);
        assert!(new.directional_peer_relations <= old.directional_peer_relations);
    }
}
