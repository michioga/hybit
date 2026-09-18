#include "hybit.hpp"
#include <iostream>
#include <vector>

int main() {
    std::vector<std::uint32_t> row_ptr{0, 2, 5, 7};
    std::vector<std::uint32_t> col_idx{0, 1, 0, 1, 2, 1, 2};
    std::vector<double> values{2, -1, -1, 2, -1, -1, 2};
    std::vector<double> b1{1, 0, 1};
    std::vector<double> b2{2, 0, 2};
    std::vector<double> x1(3, 0.0), x2(3, 0.0);

    auto matrix = hybit::Matrix::from_csr(3, 3, row_ptr, col_idx, values);
    hybit::Solver solver;
    auto prepared = solver.prepare(matrix);
    auto r1 = prepared.solve(matrix, b1, x1);
    auto r2 = prepared.solve(matrix, b2, x2);

    std::cout << "HyBIT C++ prepared example: x1 = [" << x1[0] << ", " << x1[1] << ", " << x1[2] << "]\n";
    std::cout << "  solve-sequence=" << r1.solve_sequence << " -> " << r2.solve_sequence
              << ", workspace=" << r2.krylov_workspace_bytes << " bytes"
              << ", reused=" << r2.preconditioner_reused << "\n";
}
