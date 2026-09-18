#include "hybit.h"
#include <stdio.h>
#include <stdlib.h>

static void fail(int code) {
    char message[512];
    hybit_last_error_message(message, sizeof(message));
    fprintf(stderr, "HyBIT error %d: %s\n", code, message);
    exit(EXIT_FAILURE);
}

int main(void) {
    const uint32_t row_ptr[] = {0, 2, 5, 7};
    const uint32_t col_idx[] = {0, 1, 0, 1, 2, 1, 2};
    const double values[] = {2, -1, -1, 2, -1, -1, 2};
    const double b1[] = {1, 0, 1};
    const double b2[] = {2, 0, 2};
    double x1[] = {0, 0, 0};
    double x2[] = {0, 0, 0};

    hybit_solver_t *solver = NULL;
    hybit_matrix_t *matrix = NULL;
    hybit_prepared_t *prepared = NULL;
    hybit_solve_report_t r1 = {0}, r2 = {0};
    int rc;

    if ((rc = hybit_solver_create(&solver)) != HYBIT_OK) fail(rc);
    if ((rc = hybit_matrix_create_csr_f64(3, 3, 7, row_ptr, col_idx, values, 0, &matrix)) != HYBIT_OK) fail(rc);
    if ((rc = hybit_prepare(solver, matrix, &prepared)) != HYBIT_OK) fail(rc);
    if ((rc = hybit_solve_prepared(prepared, matrix, b1, x1, &r1)) != HYBIT_OK) fail(rc);
    if ((rc = hybit_solve_prepared(prepared, matrix, b2, x2, &r2)) != HYBIT_OK) fail(rc);

    printf("HyBIT C prepared example: x1 = [%.12g, %.12g, %.12g]\n", x1[0], x1[1], x1[2]);
    printf("  solve-sequence=%llu -> %llu, workspace=%llu bytes, reused=%d\n",
           (unsigned long long)r1.solve_sequence,
           (unsigned long long)r2.solve_sequence,
           (unsigned long long)r2.krylov_workspace_bytes,
           r2.preconditioner_reused);

    hybit_prepared_destroy(prepared);
    hybit_matrix_destroy(matrix);
    hybit_solver_destroy(solver);
    return 0;
}
