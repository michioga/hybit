#ifndef HYBIT_HPP
#define HYBIT_HPP

#include "hybit.h"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hybit {

inline std::string last_error() {
    const std::size_t n = hybit_last_error_message(nullptr, 0);
    std::vector<char> buffer(n ? n : 1);
    hybit_last_error_message(buffer.data(), buffer.size());
    return std::string(buffer.data());
}

inline void check(int status) {
    if (status != HYBIT_OK) throw std::runtime_error(last_error());
}
#ifdef _OPENMP
inline void sync_openmp_threads() {
    check(hybit_sync_openmp_threads());
}
#endif

class Matrix {
public:
    Matrix() = default;
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
    Matrix(Matrix&& other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)) {}
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = std::exchange(other.ptr_, nullptr);
        }
        return *this;
    }
    ~Matrix() { reset(); }

    static Matrix from_csr(
        std::uint32_t nrows,
        std::uint32_t ncols,
        const std::vector<std::uint32_t>& row_ptr,
        const std::vector<std::uint32_t>& col_idx,
        const std::vector<double>& values,
        int index_base = 0) {
        if (col_idx.size() != values.size()) throw std::invalid_argument("col_idx/values size mismatch");
        Matrix m;
        check(hybit_matrix_create_csr_f64(nrows, ncols, static_cast<std::uint32_t>(values.size()), row_ptr.data(), col_idx.data(), values.data(), index_base, &m.ptr_));
        return m;
    }

    hybit_matrix_t* get() const noexcept { return ptr_; }

private:
    void reset() noexcept { if (ptr_) hybit_matrix_destroy(ptr_); ptr_ = nullptr; }
    hybit_matrix_t* ptr_ = nullptr;
};

/// RAII wrapper for field-aware DOF metadata (no changes to solver defaults).
class DofLayout {
public:
    explicit DofLayout(const std::vector<hybit_dof_block_t>& blocks) {
        check(hybit_dof_layout_create(blocks.data(), blocks.size(), &ptr_));
    }
    DofLayout(const DofLayout&) = delete;
    DofLayout& operator=(const DofLayout&) = delete;
    DofLayout(DofLayout&& other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)) {}
    DofLayout& operator=(DofLayout&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = std::exchange(other.ptr_, nullptr);
        }
        return *this;
    }
    ~DofLayout() { reset(); }

    std::uint64_t num_dofs() const {
        std::uint64_t n = 0;
        check(hybit_dof_layout_num_dofs(ptr_, &n));
        return n;
    }
    std::pair<std::uint64_t, std::uint64_t> block_range(std::size_t index) const {
        hybit_dof_block_t block{};
        std::uint64_t start = 0, end = 0;
        check(hybit_dof_layout_block(ptr_, index, &block, &start, &end));
        return {start, end};
    }
    std::pair<std::size_t, std::uint32_t> locate(std::uint64_t dof) const {
        std::size_t block_index = 0;
        std::uint32_t component = 0;
        check(hybit_dof_layout_locate(ptr_, dof, &block_index, &component));
        return {block_index, component};
    }

    /// Packed physical DOF mask. No numeric solve is performed.
    std::vector<std::uint64_t> expand_blocks(const std::vector<std::uint64_t>& seeds) const {
        const std::size_t words = static_cast<std::size_t>((num_dofs() + 63) / 64);
        std::vector<std::uint64_t> bits(words);
        check(hybit_dof_layout_expand_blocks(ptr_, seeds.empty() ? nullptr : seeds.data(),
                                              seeds.size(), bits.data(), bits.size()));
        return bits;
    }
    /// Structural ABTM hops plus field-block closure. Preparation is per call.
    std::vector<std::uint64_t> grow_region(
        const Matrix& matrix, const std::vector<std::uint64_t>& seeds,
        std::size_t hops) const {
        const std::size_t words = static_cast<std::size_t>((num_dofs() + 63) / 64);
        std::vector<std::uint64_t> bits(words);
        check(hybit_dof_layout_grow_region(ptr_, matrix.get(),
                                           seeds.empty() ? nullptr : seeds.data(),
                                           seeds.size(), hops, bits.data(), bits.size()));
        return bits;
    }

private:
    void reset() noexcept { if (ptr_) hybit_dof_layout_destroy(ptr_); ptr_ = nullptr; }
    hybit_dof_layout_t* ptr_ = nullptr;
};

class Prepared {
public:
    Prepared() = default;
    Prepared(const Prepared&) = delete;
    Prepared& operator=(const Prepared&) = delete;
    Prepared(Prepared&& other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)) {}
    Prepared& operator=(Prepared&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = std::exchange(other.ptr_, nullptr);
        }
        return *this;
    }
    ~Prepared() { reset(); }

    hybit_solve_report_t solve(const Matrix& matrix, const std::vector<double>& b, std::vector<double>& x) {
        if (b.size() != x.size()) throw std::invalid_argument("b/x size mismatch for square HyBIT SPD systems");
        hybit_solve_report_t report{};
        check(hybit_solve_prepared(ptr_, matrix.get(), b.data(), x.data(), &report));
        return report;
    }

private:
    explicit Prepared(hybit_prepared_t* ptr) : ptr_(ptr) {}
    void reset() noexcept { if (ptr_) hybit_prepared_destroy(ptr_); ptr_ = nullptr; }
    hybit_prepared_t* ptr_ = nullptr;
    friend class Solver;
};

class Solver {
public:
    Solver() { check(hybit_solver_create(&ptr_)); }
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;
    Solver(Solver&& other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)) {}
    Solver& operator=(Solver&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = std::exchange(other.ptr_, nullptr);
        }
        return *this;
    }
    ~Solver() { reset(); }

    void tolerances(double rtol, double atol = 0.0) { check(hybit_solver_set_tolerances(ptr_, rtol, atol)); }
    void max_iterations(std::uint64_t n) { check(hybit_solver_set_max_iterations(ptr_, n)); }
    void backend(int backend) { check(hybit_solver_set_backend(ptr_, backend)); }
    void hybrid_enabled(bool enabled) { check(hybit_solver_set_hybrid_enabled(ptr_, enabled ? 1 : 0)); }
    void overlap_layers(std::uint64_t layers) { check(hybit_solver_set_overlap_layers(ptr_, layers)); }

    Prepared prepare(const Matrix& matrix) {
        hybit_prepared_t* prepared = nullptr;
        check(hybit_prepare(ptr_, matrix.get(), &prepared));
        return Prepared(prepared);
    }

    hybit_solve_report_t solve(const Matrix& matrix, const std::vector<double>& b, std::vector<double>& x) {
        if (b.size() != x.size()) throw std::invalid_argument("b/x size mismatch for square HyBIT SPD systems");
        hybit_solve_report_t report{};
        check(hybit_solve(ptr_, matrix.get(), b.data(), x.data(), &report));
        return report;
    }

private:
    void reset() noexcept { if (ptr_) hybit_solver_destroy(ptr_); ptr_ = nullptr; }
    hybit_solver_t* ptr_ = nullptr;
};

} // namespace hybit

#endif
