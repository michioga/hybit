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
