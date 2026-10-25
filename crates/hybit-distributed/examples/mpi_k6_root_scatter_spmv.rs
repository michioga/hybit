//! H1-C3K6: root-only Matrix Market -> noncontiguous owned CSR -> native MPI SpMV.
//! Real Matrix Market coefficients; no PCG/automatic solver routing.
//! Only root reads/stores the full matrix and computes the ownership policy.

use hybit_distributed::mpi_assignment::{prepare_assignment_owned_rows, spmv_assignment};
use hybit_distributed::mpi_assignment_scatter::scatter_assignment_from_root;
use hybit_distributed::mpi_assignment_verify::verify_assignment_scatter_exact;
use hybit_distributed::mpi_backend::MpiRuntime;
use hybit_distributed::{
    ContiguousPartition, PartitionAssignment, abtm_balanced_multisource_partition,
    abtm_region_grow_partition,
};
use hybit_matrix::{Csr32Matrix, read_matrix_market};
use mpi::collective::SystemOperation;
use mpi::traits::*;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

type BenchResult<T> = Result<T, Box<dyn Error>>;
const MAX_INPUT_BYTES: u64 = 384 * 1024 * 1024;
const MAX_DOFS: usize = 1_200_000;
const MAX_NNZ: usize = 16_000_000;

#[derive(Clone, Copy, Debug)]
enum Policy {
    Equal,
    Region,
    Multisource,
}
impl Policy {
    fn parse(s: &str) -> BenchResult<Self> {
        match s {
            "equal" => Ok(Self::Equal),
            "region" => Ok(Self::Region),
            "multisource" => Ok(Self::Multisource),
            _ => Err("policy must be equal, region, or multisource".into()),
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Equal => "equal",
            Self::Region => "abtm_region_growth",
            Self::Multisource => "abtm_multisource",
        }
    }
}

fn partition(matrix: &Csr32Matrix, ranks: u32, policy: Policy) -> BenchResult<PartitionAssignment> {
    Ok(match policy {
        Policy::Equal => PartitionAssignment::from_contiguous(&ContiguousPartition::balanced(
            matrix.nrows() as u64,
            ranks,
        )?)?,
        Policy::Region => abtm_region_grow_partition(matrix, ranks)?.0,
        Policy::Multisource => abtm_balanced_multisource_partition(matrix, ranks)?.0,
    })
}

fn world_max(value: f64) -> f64 {
    let world = mpi::topology::SimpleCommunicator::world();
    let mut out = 0.0f64;
    world.all_reduce_into(&value, &mut out, SystemOperation::max());
    out
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let i = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[i - 1] + sorted[i])
    } else {
        sorted[i]
    }
}

fn x_value(global: u64, seed: usize) -> f64 {
    let id = global as f64 + 1.0;
    if seed == 0 {
        (id * 0.017).sin()
    } else {
        (id * 0.0081).cos()
    }
}

fn main() -> BenchResult<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err("usage: mpi_k6_root_scatter_spmv <ranks> <matrix.mtx> <equal|region|multisource> <repeats> <output.csv>".into());
    }
    let ranks: i32 = args[0].parse()?;
    let path = Path::new(&args[1]);
    let policy = Policy::parse(&args[2])?;
    let repeats: usize = args[3].parse()?;
    let output = Path::new(&args[4]);
    if !matches!(ranks, 1 | 2 | 4 | 8) || !(3..=512).contains(&repeats) {
        return Err("invalid ranks or repeats".into());
    }
    let mpi = MpiRuntime::initialize()?;
    if mpi.size() != ranks {
        return Err("MPI communicator differs from requested ranks".into());
    }
    if mpi.all_reduce_sum_u64(u64::from(mpi.rank() == 0 && output.exists())) != 0 {
        return Err("output already exists (refuse overwrite)".into());
    }

    mpi.barrier();
    let started = Instant::now();
    // Root alone opens the file; the remaining ranks allocate no global CSR.
    let loaded = if mpi.rank() == 0 {
        Some((|| -> BenchResult<_> {
            if std::fs::metadata(path)?.len() > MAX_INPUT_BYTES {
                return Err("Matrix Market file exceeds K6 bounded root-input limit".into());
            }
            let (matrix, _info) = read_matrix_market(path)?;
            if matrix.nrows() == 0
                || matrix.nrows() != matrix.ncols()
                || matrix.nrows() > MAX_DOFS
                || matrix.nrows() < ranks as usize
                || matrix.nnz() > MAX_NNZ
            {
                return Err("root input outside bounded matrix dimensions".into());
            }
            Ok(matrix)
        })())
    } else {
        None
    };
    if mpi.all_reduce_sum_u64(u64::from(loaded.as_ref().is_some_and(Result::is_err))) != 0 {
        if let Some(Err(err)) = &loaded {
            eprintln!("K6 root Matrix Market read error: {err}");
        }
        return Err("root-only Matrix Market read failed".into());
    }
    let matrix = loaded.transpose()?;
    let read_ms = world_max(started.elapsed().as_secs_f64() * 1000.0);
    let n = mpi.all_reduce_sum_u64(matrix.as_ref().map_or(0, |m| m.nrows() as u64));
    let nnz = mpi.all_reduce_sum_u64(matrix.as_ref().map_or(0, |m| m.nnz() as u64));
    let global_csr_bytes = mpi.all_reduce_sum_u64(matrix.as_ref().map_or(0, |m| {
        (std::mem::size_of_val(m.row_ptr())
            + std::mem::size_of_val(m.col_idx())
            + std::mem::size_of_val(m.values())) as u64
    }));

    mpi.barrier();
    let started = Instant::now();
    let selected = matrix.as_ref().map(|m| partition(m, ranks as u32, policy));
    if mpi.all_reduce_sum_u64(u64::from(selected.as_ref().is_some_and(Result::is_err))) != 0 {
        if let Some(Err(err)) = &selected {
            eprintln!("K6 root partition error: {err}");
        }
        return Err("root-only partition selection failed".into());
    }
    let selected = selected.transpose()?;
    let partition_ms = world_max(started.elapsed().as_secs_f64() * 1000.0);

    mpi.barrier();
    let (assignment, rows, scatter_stats) =
        scatter_assignment_from_root(&mpi, matrix.as_ref(), selected.as_ref())?;
    let scatter_ms = world_max(scatter_stats.scatter_ns as f64 * 1e-6);
    // K7-B: independent exact comparison against root's ORIGINAL coefficients.
    // This is outside measured scatter and SpMV timing; it is intentionally
    // diagnostic and needs the root CSR to remain alive until completion.
    mpi.barrier();
    let verification = verify_assignment_scatter_exact(&mpi, matrix.as_ref(), &assignment, &rows)?;
    let verify_ms = world_max(verification.verify_ns as f64 * 1e-6);
    if verification.verified_rows != n
        || verification.verified_nnz != nnz
        || !verification.bitwise_coefficients
    {
        return Err("K7-B root original CSR exact verification failed".into());
    }
    // Explicitly release the only global CSR and root's owner-selection copy.
    // All subsequent operations use only locally owned CSR and the common owner map.
    drop(matrix);
    drop(selected);
    let root_stream_chunk_bytes =
        mpi.all_reduce_sum_u64(scatter_stats.root_peak_streamed_chunk_bytes_estimate);
    let root_k6_chunks_bytes =
        mpi.all_reduce_sum_u64(scatter_stats.root_k6_all_chunks_bytes_estimate);
    let root_concurrent_chunks =
        mpi.all_reduce_sum_u64(u64::from(scatter_stats.max_concurrent_root_chunks));
    if root_stream_chunk_bytes > root_k6_chunks_bytes || root_concurrent_chunks != 1 {
        return Err("K7-A bounded root chunk telemetry invalid".into());
    }
    let local_bytes_sum = mpi.all_reduce_sum_u64(scatter_stats.local_owned_bytes as u64);
    let local_nnz_sum = mpi.all_reduce_sum_u64(scatter_stats.local_nnz as u64);
    if local_nnz_sum != nnz || scatter_stats.global_nnz != nnz || assignment.global_dofs() != n {
        return Err("K6 distributed row accounting mismatch".into());
    }

    mpi.barrier();
    let started = Instant::now();
    let (halo, local, halo_stats) = prepare_assignment_owned_rows(&mpi, &assignment, &rows)?;
    let halo_setup_ms = world_max(started.elapsed().as_secs_f64() * 1000.0);
    let ghosts = mpi.all_reduce_sum_u64(halo_stats.ghosts as u64);
    let operator_bytes_sum = mpi.all_reduce_sum_u64(halo_stats.rank_operator_halo_bytes as u64);
    let sent = mpi.all_reduce_sum_u64(halo_stats.sent_f64 as u64);
    let received = mpi.all_reduce_sum_u64(halo_stats.received_f64 as u64);
    if sent != received || sent != ghosts || halo.owned_len() != rows.owned_globals.len() {
        return Err("K6 noncontiguous Halo handshake mismatch".into());
    }

    let mut scratch = halo.workspace();
    let mut y = vec![0.0_f64; halo.owned_len()];
    let mut samples = Vec::with_capacity(repeats);
    let mut halo_samples = Vec::with_capacity(repeats);
    let mut numeric_samples = Vec::with_capacity(repeats);
    let mut worst_rel = 0.0f64;
    let mut worst_abs = 0.0f64;
    for trial in 0..repeats + 3 {
        let seed = trial % 2;
        let owned_x: Vec<f64> = halo
            .owned_globals
            .iter()
            .map(|&g| x_value(g, seed))
            .collect();
        mpi.barrier();
        let started = Instant::now();
        let timings = spmv_assignment(&mpi, &halo, &local, &owned_x, &mut scratch, &mut y)?;
        let wall_ms = world_max(started.elapsed().as_secs_f64() * 1000.0);
        let halo_ms = world_max(timings.halo_ns as f64 * 1e-6);
        let numeric_ms = world_max(timings.numeric_ns as f64 * 1e-6);
        if trial < 2 || trial == repeats + 2 {
            let mut local_abs = 0.0f64;
            let mut local_scale = 1.0f64;
            for (i, &actual) in y.iter().enumerate() {
                let first = rows.row_ptr[i] as usize;
                let last = rows.row_ptr[i + 1] as usize;
                let reference: f64 = (first..last)
                    .map(|j| rows.values[j] * x_value(rows.global_col_idx[j], seed))
                    .sum();
                local_abs = local_abs.max((actual - reference).abs());
                local_scale = local_scale.max(reference.abs());
            }
            for (i, &g) in halo.ghost_globals.iter().enumerate() {
                let slot = halo.owned_len() + i;
                local_abs = local_abs.max((scratch.extended[slot] - x_value(g, seed)).abs());
            }
            let error = world_max(local_abs);
            let scale = world_max(local_scale);
            let relative = error / scale;
            if !relative.is_finite() || relative > 1e-11 {
                return Err("K6 numerical SpMV or exchanged ghost verification failed".into());
            }
            worst_abs = worst_abs.max(error);
            worst_rel = worst_rel.max(relative);
        }
        if trial >= 3 {
            samples.push(wall_ms);
            halo_samples.push(halo_ms);
            numeric_samples.push(numeric_ms);
        }
    }
    let spmv_ms = median(&samples);
    let halo_ms = median(&halo_samples);
    let numeric_ms = median(&numeric_samples);
    if mpi.rank() == 0 {
        println!(
            "K6 ROOT-SCATTER NATIVE PASS method={} ranks={} n={} nnz={} ghosts={} read_ms={:.3} partition_ms={:.3} scatter_ms={:.3} halo_setup_ms={:.3} median_spmv_ms={:.6} median_halo_ms={:.6} median_numeric_ms={:.6} sum_owned_bytes={} sum_operator_bytes={} max_rel={:.3e} root_only=true root_stream_chunk_bytes={} root_k6_all_chunks_bytes={} concurrent_chunks={} exact_original=true verify_ms={:.3} verified_rows={} verified_nnz={}",
            policy.label(),
            ranks,
            n,
            nnz,
            ghosts,
            read_ms,
            partition_ms,
            scatter_ms,
            halo_setup_ms,
            spmv_ms,
            halo_ms,
            numeric_ms,
            local_bytes_sum,
            operator_bytes_sum,
            worst_rel,
            root_stream_chunk_bytes,
            root_k6_chunks_bytes,
            root_concurrent_chunks,
            verify_ms,
            verification.verified_rows,
            verification.verified_nnz
        );
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output)?;
        writeln!(
            file,
            "matrix,ranks,method,n,nnz,repeats,read_root_max_ms,partition_root_max_ms,scatter_max_ms,halo_setup_max_ms,median_spmv_max_ms,median_halo_max_ms,median_numeric_max_ms,ghosts,sent_f64,local_owned_bytes_sum,operator_halo_bytes_sum,root_input_csr_bytes,worst_abs,worst_rel,root_stream_chunk_bytes_est,root_k6_all_chunks_bytes_est,max_concurrent_root_chunks,exact_original_verified,verify_original_max_ms,verified_original_rows,verified_original_nnz"
        )?;
        writeln!(
            file,
            "{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{:.12e},{:.12e},{},{},{},true,{:.6},{},{}",
            path.display(),
            ranks,
            policy.label(),
            n,
            nnz,
            repeats,
            read_ms,
            partition_ms,
            scatter_ms,
            halo_setup_ms,
            spmv_ms,
            halo_ms,
            numeric_ms,
            ghosts,
            sent,
            local_bytes_sum,
            operator_bytes_sum,
            global_csr_bytes,
            worst_abs,
            worst_rel,
            root_stream_chunk_bytes,
            root_k6_chunks_bytes,
            root_concurrent_chunks,
            verify_ms,
            verification.verified_rows,
            verification.verified_nnz
        )?;
    }
    mpi.barrier();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn x_values_finite_and_nonidentical() {
        assert!(x_value(42, 0).is_finite());
        assert_ne!(x_value(42, 0), x_value(42, 1));
    }
    #[test]
    fn median_even_odd() {
        assert_eq!(median(&[4.0, 1.0, 3.0]), 3.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }
}
