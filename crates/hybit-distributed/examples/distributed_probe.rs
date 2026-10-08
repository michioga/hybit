use hybit_distributed::{
    build_contiguous_halo_plans, build_contiguous_halo_plans_abtm, partition_telemetry,
    prepare_rank_local_csrs, simulated_distributed_spmv, ContiguousPartition,
};
use hybit_matrix::read_matrix_market;
use std::env;
use std::error::Error;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: distributed_probe <matrix.mtx> <ranks>");
        std::process::exit(2);
    }

    let ranks: u32 = args[2].parse()?;
    let (matrix, info) = read_matrix_market(&args[1])?;
    let partition = ContiguousPartition::balanced(matrix.nrows() as u64, ranks)?;

    let t0 = Instant::now();
    let csr_plans = build_contiguous_halo_plans(&matrix, &partition)?;
    let csr_halo_ms = t0.elapsed().as_secs_f64() * 1.0e3;

    let t1 = Instant::now();
    let abtm_plans = build_contiguous_halo_plans_abtm(&matrix, &partition)?;
    let abtm_halo_ms = t1.elapsed().as_secs_f64() * 1.0e3;

    if csr_plans != abtm_plans {
        return Err("ABTM halo extraction differs from CSR reference".into());
    }

    let t2 = Instant::now();
    let locals = prepare_rank_local_csrs(&matrix, &csr_plans)?;
    let local_prepare_ms = t2.elapsed().as_secs_f64() * 1.0e3;

    let telemetry = partition_telemetry(&matrix, &partition, &csr_plans)?;

    let x: Vec<_> = (0..matrix.ncols())
        .map(|i| ((i % 127) as f64 + 1.0) / 127.0)
        .collect();
    let serial = matrix.spmv(&x)?;
    let distributed = simulated_distributed_spmv(&matrix, &partition, &x)?;
    let spmv_exact = serial == distributed;

    println!("HyBIT G8-A2 distributed topology probe");
    println!("matrix                 : {}", args[1]);
    println!("shape                  : {} x {}", info.nrows, info.ncols);
    println!("nnz                    : {}", matrix.nnz());
    println!("ranks                  : {}", ranks);
    println!("CSR halo prepare ms    : {:.6}", csr_halo_ms);
    println!("ABTM halo prepare ms   : {:.6}", abtm_halo_ms);
    println!("ABTM == CSR halo       : {}", csr_plans == abtm_plans);
    println!("local CSR prepare ms   : {:.6}", local_prepare_ms);
    println!(
        "local CSR nnz total    : {}",
        locals.iter().map(|local| local.nnz()).sum::<usize>()
    );
    println!("cut nnz                : {}", telemetry.cut_nnz);
    println!(
        "communication volume   : {}",
        telemetry.communication_volume
    );
    println!(
        "peer relations         : {}",
        telemetry.directional_peer_relations
    );
    println!("max neighbors          : {}", telemetry.max_neighbors);
    println!(
        "owned DOF min/max      : {} / {}",
        telemetry.min_owned_dofs, telemetry.max_owned_dofs
    );
    println!(
        "owned DOF imbalance    : {:.6}",
        telemetry.owned_dof_imbalance
    );
    println!(
        "local nnz min/max      : {} / {}",
        telemetry.min_local_nnz, telemetry.max_local_nnz
    );
    println!(
        "local nnz imbalance    : {:.6}",
        telemetry.local_nnz_imbalance
    );
    println!("simulated SpMV exact   : {}", spmv_exact);

    if !spmv_exact {
        return Err("simulated distributed SpMV differs from serial CSR".into());
    }

    Ok(())
}
