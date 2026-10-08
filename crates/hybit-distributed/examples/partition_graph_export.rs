use hybit_matrix::{read_matrix_market, AbtmDualTopology};
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};

fn neighbors(topology: &AbtmDualTopology, node: usize) -> Result<Vec<usize>, Box<dyn Error>> {
    let mut result = Vec::new();

    let row = topology.row(node)?;
    for word in row.words() {
        let mut bits = word.mask();
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let neighbor = word.base_col() + bit;
            if neighbor != node {
                result.push(neighbor);
            }
        }
    }

    let column = topology.column(node)?;
    for word in column.words() {
        let mut bits = word.mask();
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let neighbor = word.base_col() + bit;
            if neighbor != node {
                result.push(neighbor);
            }
        }
    }

    result.sort_unstable();
    result.dedup();
    Ok(result)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: partition_graph_export <matrix.mtx> <output-prefix>");
        std::process::exit(2);
    }

    let matrix_path = &args[1];
    let prefix = &args[2];

    let (matrix, info) = read_matrix_market(matrix_path)?;
    if matrix.nrows() != matrix.ncols() {
        return Err("external partition graph export requires a square matrix".into());
    }

    let topology = AbtmDualTopology::from_csr32(&matrix)?;

    let mut degree_sum = 0usize;
    let mut max_degree = 0usize;
    for node in 0..matrix.nrows() {
        let degree = neighbors(&topology, node)?.len();
        degree_sum = degree_sum
            .checked_add(degree)
            .ok_or("undirected adjacency count overflow")?;
        max_degree = max_degree.max(degree);
    }

    if degree_sum % 2 != 0 {
        return Err("A union A^T produced an odd directed adjacency count".into());
    }

    let undirected_edges = degree_sum / 2;
    let metis_path = format!("{prefix}.metis.graph");
    let scotch_path = format!("{prefix}.scotch.grf");

    {
        let file = File::create(&metis_path)?;
        let mut writer = BufWriter::new(file);

        // METIS unweighted undirected graph format:
        // n_vertices n_edges
        // followed by one 1-based adjacency list per vertex.
        writeln!(writer, "{} {}", matrix.nrows(), undirected_edges)?;

        for node in 0..matrix.nrows() {
            let adjacency = neighbors(&topology, node)?;
            for (index, neighbor) in adjacency.iter().enumerate() {
                if index != 0 {
                    write!(writer, " ")?;
                }
                write!(writer, "{}", neighbor + 1)?;
            }
            writeln!(writer)?;
        }
    }

    {
        let file = File::create(&scotch_path)?;
        let mut writer = BufWriter::new(file);

        // SCOTCH graph format, version 0, base 0, no labels/weights:
        // 0
        // vertnbr edgenbr(arcs)
        // 0 000
        // degree neighbor0 neighbor1 ...
        writeln!(writer, "0")?;
        writeln!(writer, "{} {}", matrix.nrows(), degree_sum)?;
        writeln!(writer, "0 000")?;

        for node in 0..matrix.nrows() {
            let adjacency = neighbors(&topology, node)?;
            write!(writer, "{}", adjacency.len())?;
            for neighbor in adjacency {
                write!(writer, " {neighbor}")?;
            }
            writeln!(writer)?;
        }
    }

    println!("HyBIT G8-A5 external graph export");
    println!("matrix            : {matrix_path}");
    println!("shape             : {} x {}", info.nrows, info.ncols);
    println!("matrix nnz        : {}", matrix.nnz());
    println!("graph vertices    : {}", matrix.nrows());
    println!("graph edges       : {undirected_edges}");
    println!("graph arcs        : {degree_sum}");
    println!("max degree        : {max_degree}");
    println!("METIS graph       : {metis_path}");
    println!("SCOTCH graph      : {scotch_path}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hybit_matrix::Csr32Matrix;

    #[test]
    fn undirected_union_removes_diagonal_and_duplicates() {
        let matrix = Csr32Matrix::new(
            3,
            3,
            vec![0, 3, 5, 7],
            vec![0, 1, 1, 1, 2, 0, 2],
            vec![1.0; 7],
        )
        .unwrap();

        let topology = AbtmDualTopology::from_csr32(&matrix).unwrap();

        assert_eq!(neighbors(&topology, 0).unwrap(), vec![1, 2]);
        assert_eq!(neighbors(&topology, 1).unwrap(), vec![0, 2]);
        assert_eq!(neighbors(&topology, 2).unwrap(), vec![0, 1]);
    }
}
