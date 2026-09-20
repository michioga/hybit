# Exporting an assembled CSR matrix

If an FEM program already owns a zero-based CSR matrix, it can use HyBIT's
Matrix Market writer for one-time benchmark export:

```rust
use hybit::{write_matrix_market_general, Csr32Matrix};

let a = Csr32Matrix::new(n, n, row_ptr_u32, col_idx_u32, values_f64)?;
write_matrix_market_general("K.mtx", &a)?;
```

The exported file uses 1-based Matrix Market indices as required by the format.
Apply essential boundary conditions before export so the matrix passed to the
current PCG path is SPD. For large matrices this text format is intended for
validation/interchange rather than production runtime I/O.
