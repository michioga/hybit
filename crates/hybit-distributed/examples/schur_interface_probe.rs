//! HyBIT 0.9 Schur S1: topology-only interior/interface feasibility probe.
//!
//! No Schur matrix is formed.  Rank boundary labels are defined from the union
//! of nonzero A(i,j) and A(j,i) across different owners, so the classification
//! is meaningful for symmetric and unsymmetric input matrices alike.

use hybit_distributed::{
    abtm_multilevel_partition, partition_telemetry_assignment, AbtmMultilevelOptions,
    ContiguousPartition, PartitionAssignment,
};
use hybit_matrix::{read_matrix_market, Csr32Matrix};
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BlockNnz {
    ii: u64,
    ig: u64,
    gi: u64,
    gg: u64,
}

impl BlockNnz {
    fn total(self) -> u64 {
        self.ii + self.ig + self.gi + self.gg
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RankSummary {
    owned: usize,
    interior: usize,
    interface: usize,
    nnz: BlockNnz,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct InterfaceSummary {
    interface_flags: Vec<bool>,
    ranks: Vec<RankSummary>,
    nnz: BlockNnz,
    cut_nnz: usize,
    interior: usize,
    interface: usize,
}

fn analyze_interface(
    matrix: &Csr32Matrix,
    assignment: &PartitionAssignment,
) -> Result<InterfaceSummary, Box<dyn Error>> {
    let n = matrix.nrows();
    if matrix.ncols() != n || assignment.owners().len() != n {
        return Err("matrix must be square and ownership must cover every row".into());
    }
    let owners = assignment.owners();
    let mut interface_flags = vec![false; n];
    let mut cut_nnz = 0usize;

    // Mark both endpoints of every cross-owner nonzero, which gives the
    // A union A^T separator without explicitly storing a dual adjacency.
    for row in 0..n {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        for &column in &matrix.col_idx()[start..end] {
            let col = column as usize;
            if owners[row] != owners[col] {
                interface_flags[row] = true;
                interface_flags[col] = true;
                cut_nnz += 1;
            }
        }
    }

    let mut ranks = vec![RankSummary::default(); assignment.rank_count() as usize];
    for (node, &owner) in owners.iter().enumerate() {
        let rank = &mut ranks[owner as usize];
        rank.owned += 1;
        if interface_flags[node] {
            rank.interface += 1;
        } else {
            rank.interior += 1;
        }
    }

    let mut nnz = BlockNnz::default();
    for row in 0..n {
        let row_is_interface = interface_flags[row];
        let block = &mut ranks[owners[row] as usize].nnz;
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        for &column in &matrix.col_idx()[start..end] {
            let col_is_interface = interface_flags[column as usize];
            match (row_is_interface, col_is_interface) {
                (false, false) => {
                    nnz.ii += 1;
                    block.ii += 1;
                }
                (false, true) => {
                    nnz.ig += 1;
                    block.ig += 1;
                }
                (true, false) => {
                    nnz.gi += 1;
                    block.gi += 1;
                }
                (true, true) => {
                    nnz.gg += 1;
                    block.gg += 1;
                }
            }
        }
    }

    let interior = ranks.iter().map(|r| r.interior).sum();
    let interface = ranks.iter().map(|r| r.interface).sum();
    if interior + interface != n || nnz.total() != matrix.nnz() as u64 {
        return Err("Schur S1 block accounting failed".into());
    }
    // By construction all off-rank edges touch interfaces on both ends.
    // Consequently A_II is block diagonal by rank.
    for row in 0..n {
        if interface_flags[row] {
            continue;
        }
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        for &column in &matrix.col_idx()[start..end] {
            if owners[row] != owners[column as usize] {
                return Err("cross-rank nonzero encountered in interior row".into());
            }
        }
    }

    Ok(InterfaceSummary {
        interface_flags,
        ranks,
        nnz,
        cut_nnz,
        interior,
        interface,
    })
}

fn owner_fnv64(owners: &[u32]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &owner in owners {
        for byte in owner.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

fn dense_bytes(n: usize) -> u128 {
    let n = n as u128;
    n * n * 8
}

fn packed_cholesky_bytes(n: usize) -> u128 {
    let n = n as u128;
    n * (n + 1) / 2 * 8
}

fn read_owner_labels(path: &str, ranks: u32) -> Result<PartitionAssignment, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut owners = Vec::new();
    for line in text.lines() {
        let content = line.split(['#', '%']).next().unwrap_or("");
        for token in content.split_whitespace() {
            owners.push(token.parse::<u32>()?);
        }
    }
    Ok(PartitionAssignment::from_owners(ranks, owners)?)
}

fn run_mode(
    mode: &str,
    case_id: &str,
    matrix: &Csr32Matrix,
    assignment: &PartitionAssignment,
    partition_ms: f64,
    summary_writer: &mut impl Write,
    ranks_writer: &mut impl Write,
) -> Result<(), Box<dyn Error>> {
    let timer = Instant::now();
    let s = analyze_interface(matrix, assignment)?;
    let scan_ms = timer.elapsed().as_secs_f64() * 1e3;
    if s.interface_flags.iter().filter(|&&flag| flag).count() != s.interface {
        return Err("interface flag count does not match summary".into());
    }
    let telemetry = partition_telemetry_assignment(matrix, assignment)?;
    if s.cut_nnz != telemetry.cut_nnz {
        return Err("Schur S1 cut_nnz differs from HyBIT telemetry".into());
    }

    let packed_local_sum: u128 = s
        .ranks
        .iter()
        .map(|r| packed_cholesky_bytes(r.interior))
        .sum();
    let packed_local_max = s
        .ranks
        .iter()
        .map(|r| packed_cholesky_bytes(r.interior))
        .max()
        .unwrap_or(0);
    let rank_dense_gamma_sum: u128 = s.ranks.iter().map(|r| dense_bytes(r.interface)).sum();
    let global_dense_gamma = dense_bytes(s.interface);
    let fraction = if matrix.nrows() == 0 {
        0.0
    } else {
        s.interface as f64 / matrix.nrows() as f64
    };
    let hash = owner_fnv64(assignment.owners());

    writeln!(
        summary_writer,
        "{case_id},{mode},{},{},{},{},{},{fraction:.8},{},{},{},{},{},{},{},{},{},{},{},{},{partition_ms:.6},{scan_ms:.6}",
        assignment.rank_count(),
        matrix.nrows(),
        matrix.nnz(),
        s.interior,
        s.interface,
        s.cut_nnz,
        telemetry.communication_volume,
        s.nnz.ii,
        s.nnz.ig,
        s.nnz.gi,
        s.nnz.gg,
        global_dense_gamma,
        rank_dense_gamma_sum,
        packed_local_sum,
        packed_local_max,
        format_args!("{hash:016x}"),
        telemetry.owned_dof_imbalance,
    )?;

    for (index, r) in s.ranks.iter().enumerate() {
        writeln!(
            ranks_writer,
            "{case_id},{mode},{index},{},{},{},{},{},{},{},{},{},{}",
            r.owned,
            r.interior,
            r.interface,
            r.nnz.ii,
            r.nnz.ig,
            r.nnz.gi,
            r.nnz.gg,
            dense_bytes(r.interface),
            packed_cholesky_bytes(r.interior),
            if r.owned == 0 {
                0.0
            } else {
                r.interface as f64 / r.owned as f64
            },
        )?;
    }

    println!(
        "PASS {case_id} {mode}: I={} Gamma={} ({:.2}%) II/IG/GI/GG={}/{}/{}/{} cut={} comm={} owner={:016x} partition={:.2}ms classify={:.2}ms",
        s.interior,
        s.interface,
        100.0 * fraction,
        s.nnz.ii,
        s.nnz.ig,
        s.nnz.gi,
        s.nnz.gg,
        s.cut_nnz,
        telemetry.communication_volume,
        hash,
        partition_ms,
        scan_ms
    );
    Ok(())
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    match env::var(name) {
        Ok(text) if !text.is_empty() => Ok(text.parse::<usize>()?),
        Ok(_) => Err(format!("{name} must not be empty").into()),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(e) => Err(e.into()),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 5 && args.len() != 6 {
        eprintln!("usage: schur_interface_probe <matrix.mtx> <ranks> <summary.csv> <per_rank.csv> [external_owners.txt]");
        std::process::exit(2);
    }
    let ranks: u32 = args[2].parse()?;
    let (matrix, _) = read_matrix_market(&args[1])?;
    let refinement_passes = env_usize("HYBIT_S1_REFINEMENT_PASSES", 2)?;
    let case_id = format!(
        "{}_r{ranks}_p{refinement_passes}",
        std::path::Path::new(&args[1])
            .file_stem()
            .ok_or("matrix filename has no stem")?
            .to_string_lossy()
    );
    let mut summary_writer = BufWriter::new(File::create(&args[3])?);
    let mut ranks_writer = BufWriter::new(File::create(&args[4])?);
    writeln!(summary_writer, "case,mode,ranks,n,nnz,interior,interface,interface_fraction,cut_nnz,communication_volume,nnz_ii,nnz_ig,nnz_gi,nnz_gg,global_dense_schur_bytes,per_rank_dense_schur_bytes_sum,local_dense_cholesky_bytes_sum,local_dense_cholesky_bytes_max,owner_fnv64,dof_imbalance,partition_ms,classification_ms")?;
    writeln!(ranks_writer, "case,mode,rank,owned,interior,interface,nnz_ii,nnz_ig,nnz_gi,nnz_gg,dense_interface_bytes,dense_local_cholesky_bytes,interface_fraction")?;

    let t_contiguous = Instant::now();
    let contiguous = ContiguousPartition::balanced(matrix.nrows() as u64, ranks)?;
    let contiguous_assignment = PartitionAssignment::from_contiguous(&contiguous)?;
    run_mode(
        "contiguous",
        &case_id,
        &matrix,
        &contiguous_assignment,
        t_contiguous.elapsed().as_secs_f64() * 1e3,
        &mut summary_writer,
        &mut ranks_writer,
    )?;

    let options = AbtmMultilevelOptions {
        max_levels: env_usize("HYBIT_S1_MAX_LEVELS", 5)?,
        coarse_vertices_per_rank: env_usize("HYBIT_S1_COARSE_PER_RANK", 64)?,
        refinement_passes,
        ..AbtmMultilevelOptions::default()
    };
    let t_abtm = Instant::now();
    let (abtm_assignment, _) = abtm_multilevel_partition(&matrix, ranks, options)?;
    run_mode(
        "abtm_a6",
        &case_id,
        &matrix,
        &abtm_assignment,
        t_abtm.elapsed().as_secs_f64() * 1e3,
        &mut summary_writer,
        &mut ranks_writer,
    )?;

    if args.len() == 6 {
        let t_external = Instant::now();
        let assignment = read_owner_labels(&args[5], ranks)?;
        run_mode(
            "external",
            &case_id,
            &matrix,
            &assignment,
            t_external.elapsed().as_secs_f64() * 1e3,
            &mut summary_writer,
            &mut ranks_writer,
        )?;
    }
    summary_writer.flush()?;
    ranks_writer.flush()?;
    println!("=== SCHUR S1 TOPOLOGY PROBE PASS ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix_from_rows(rows: &[&[u32]]) -> Csr32Matrix {
        let n = rows.len();
        let mut ptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::new();
        ptr.push(0);
        for row in rows {
            indices.extend_from_slice(row);
            ptr.push(indices.len() as u32);
        }
        Csr32Matrix::new(n, n, ptr, indices.clone(), vec![1.0; indices.len()]).unwrap()
    }

    #[test]
    fn symmetric_chain_separator_and_blocks() {
        let matrix = matrix_from_rows(&[&[0, 1], &[0, 1, 2], &[1, 2, 3], &[2, 3, 4], &[3, 4]]);
        let owners = PartitionAssignment::from_owners(2, vec![0, 0, 0, 1, 1]).unwrap();
        let s = analyze_interface(&matrix, &owners).unwrap();
        assert_eq!(s.interface_flags, vec![false, false, true, true, false]);
        assert_eq!((s.interior, s.interface, s.cut_nnz), (3, 2, 2));
        assert_eq!(
            s.nnz,
            BlockNnz {
                ii: 5,
                ig: 2,
                gi: 2,
                gg: 4
            }
        );
        assert_eq!(s.nnz.total(), matrix.nnz() as u64);
        assert_eq!((s.ranks[0].interior, s.ranks[0].interface), (2, 1));
        assert_eq!((s.ranks[1].interior, s.ranks[1].interface), (1, 1));
    }

    #[test]
    fn nonsymmetric_arc_marks_both_endpoints() {
        let matrix = matrix_from_rows(&[&[0, 2], &[1], &[2], &[3]]);
        let owners = PartitionAssignment::from_owners(2, vec![0, 0, 1, 1]).unwrap();
        let s = analyze_interface(&matrix, &owners).unwrap();
        assert_eq!(s.interface_flags, vec![true, false, true, false]);
        assert_eq!((s.interior, s.interface, s.cut_nnz), (2, 2, 1));
        assert_eq!(
            s.nnz,
            BlockNnz {
                ii: 2,
                ig: 0,
                gi: 0,
                gg: 3
            }
        );
    }

    #[test]
    fn disconnected_ranks_have_no_interface() {
        let matrix = matrix_from_rows(&[&[0, 1], &[0, 1], &[2, 3], &[2, 3]]);
        let owners = PartitionAssignment::from_owners(2, vec![0, 0, 1, 1]).unwrap();
        let s = analyze_interface(&matrix, &owners).unwrap();
        assert_eq!(s.interface, 0);
        assert_eq!(s.interior, 4);
        assert_eq!(s.nnz.ii, 8);
        assert_eq!(dense_bytes(s.interface), 0);
    }
}
