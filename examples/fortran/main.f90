program hybit_fortran_example
    use, intrinsic :: iso_c_binding
    use hybit
    implicit none

    type(c_ptr) :: solver, matrix, prepared
    type(hybit_solve_report) :: r1, r2
    integer(c_int) :: rc
    integer(c_int32_t), target :: row_ptr(4) = [1, 3, 6, 8]
    integer(c_int32_t), target :: col_idx(7) = [1, 2, 1, 2, 3, 2, 3]
    real(c_double), target :: values(7) = [2.0d0, -1.0d0, -1.0d0, 2.0d0, -1.0d0, -1.0d0, 2.0d0]
    real(c_double), target :: b1(3) = [1.0d0, 0.0d0, 1.0d0]
    real(c_double), target :: b2(3) = [2.0d0, 0.0d0, 2.0d0]
    real(c_double), target :: x1(3) = 0.0d0
    real(c_double), target :: x2(3) = 0.0d0

    rc = hybit_solver_create(solver)
    if (rc /= HYBIT_OK) stop "hybit_solver_create failed"

    rc = hybit_matrix_create_csr_f64(3_c_int32_t, 3_c_int32_t, 7_c_int32_t, &
         row_ptr, col_idx, values, 1_c_int, matrix)
    if (rc /= HYBIT_OK) stop "hybit_matrix_create_csr_f64 failed"

    rc = hybit_prepare(solver, matrix, prepared)
    if (rc /= HYBIT_OK) stop "hybit_prepare failed"

    rc = hybit_solve_prepared(prepared, matrix, b1, x1, r1)
    if (rc /= HYBIT_OK) stop "hybit_solve_prepared #1 failed"
    rc = hybit_solve_prepared(prepared, matrix, b2, x2, r2)
    if (rc /= HYBIT_OK) stop "hybit_solve_prepared #2 failed"

    print '(A,3F12.6)', 'HyBIT Fortran prepared example x1 = ', x1
    print '(A,I0,A,I0,A,I0)', 'solve-sequence=', r1%solve_sequence, ' -> ', r2%solve_sequence, &
         ' workspace-bytes=', r2%krylov_workspace_bytes

    call hybit_prepared_destroy(prepared)
    call hybit_matrix_destroy(matrix)
    call hybit_solver_destroy(solver)
end program hybit_fortran_example
