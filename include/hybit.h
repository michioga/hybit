#ifndef HYBIT_H
#define HYBIT_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
  #ifdef HYBIT_BUILD_DLL
    #define HYBIT_API __declspec(dllexport)
  #else
    #define HYBIT_API __declspec(dllimport)
  #endif
#else
  #define HYBIT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HybitSolverHandle hybit_solver_t;
typedef struct HybitMatrixHandle hybit_matrix_t;
typedef struct HybitPreparedHandle hybit_prepared_t;

enum {
    HYBIT_OK = 0,
    HYBIT_INVALID_ARGUMENT = 1,
    HYBIT_INVALID_MATRIX = 2,
    HYBIT_NUMERICAL_FAILURE = 3,
    HYBIT_NOT_CONVERGED = 4,
    HYBIT_PANIC = 100
};

enum {
    HYBIT_BACKEND_AUTO = 0,
    HYBIT_BACKEND_CSR32 = 1,
    HYBIT_BACKEND_ABTM = 2
};

typedef struct hybit_solve_report {
    int32_t status;
    int32_t solver_kind;
    int32_t preconditioner_kind;
    int32_t backend;
    uint64_t iterations;
    double initial_residual;
    double final_residual;
    double relative_residual;
    double setup_seconds;
    double solve_seconds;
    double analysis_seconds;
    double prepare_seconds;
    double probe_seconds;
    double diagnostics_seconds;
    double local_factor_seconds;
    double restart_seconds;
    uint64_t escalations;
    uint64_t probe_iterations;
    double probe_final_residual;
    uint64_t hard_dofs;
    uint64_t local_direct_regions;
    uint64_t largest_local_region;
    uint64_t local_factor_dofs;
    uint64_t unique_local_factor_dofs;
    uint64_t local_factor_bytes;
    uint64_t overlap_layers;
    int32_t preconditioner_reused;
    uint64_t solve_sequence;
    uint64_t krylov_workspace_bytes;
} hybit_solve_report_t;

HYBIT_API uint32_t hybit_version_major(void);
HYBIT_API uint32_t hybit_version_minor(void);
HYBIT_API uint32_t hybit_version_patch(void);
HYBIT_API size_t hybit_last_error_message(char *buffer, size_t capacity);

HYBIT_API int32_t hybit_solver_create(hybit_solver_t **out_solver);
HYBIT_API void hybit_solver_destroy(hybit_solver_t *solver);
HYBIT_API int32_t hybit_solver_set_tolerances(hybit_solver_t *solver, double relative_tolerance, double absolute_tolerance);
HYBIT_API int32_t hybit_solver_set_max_iterations(hybit_solver_t *solver, uint64_t max_iterations);
HYBIT_API int32_t hybit_solver_set_backend(hybit_solver_t *solver, int32_t backend);
HYBIT_API int32_t hybit_solver_set_hybrid_enabled(hybit_solver_t *solver, int32_t enabled);
HYBIT_API int32_t hybit_solver_set_overlap_layers(hybit_solver_t *solver, uint64_t layers);

/* index_base may be 0 (C/C++) or 1 (Fortran-style CSR). HyBIT copies the input. */
HYBIT_API int32_t hybit_matrix_create_csr_f64(
    uint32_t nrows,
    uint32_t ncols,
    uint32_t nnz,
    const uint32_t *row_ptr,
    const uint32_t *col_idx,
    const double *values,
    int32_t index_base,
    hybit_matrix_t **out_matrix);
HYBIT_API void hybit_matrix_destroy(hybit_matrix_t *matrix);

/* Build reusable matrix-dependent setup for repeated RHS solves. The prepared
 * handle does not own matrix; pass the same unchanged matrix to solve_prepared. */
HYBIT_API int32_t hybit_prepare(
    hybit_solver_t *solver,
    const hybit_matrix_t *matrix,
    hybit_prepared_t **out_prepared);
HYBIT_API void hybit_prepared_destroy(hybit_prepared_t *prepared);
HYBIT_API int32_t hybit_solve_prepared(
    hybit_prepared_t *prepared,
    const hybit_matrix_t *matrix,
    const double *b,
    double *x,
    hybit_solve_report_t *report);

HYBIT_API int32_t hybit_solve(
    hybit_solver_t *solver,
    const hybit_matrix_t *matrix,
    const double *b,
    double *x,
    hybit_solve_report_t *report);

#ifdef __cplusplus
}
#endif

#endif
