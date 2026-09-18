module hybit
    use, intrinsic :: iso_c_binding
    implicit none
    private

    integer(c_int), parameter, public :: HYBIT_OK = 0
    integer(c_int), parameter, public :: HYBIT_BACKEND_AUTO = 0
    integer(c_int), parameter, public :: HYBIT_BACKEND_CSR32 = 1
    integer(c_int), parameter, public :: HYBIT_BACKEND_ABTM = 2

    type, bind(C), public :: hybit_solve_report
        integer(c_int) :: status
        integer(c_int) :: solver_kind
        integer(c_int) :: preconditioner_kind
        integer(c_int) :: backend
        integer(c_int64_t) :: iterations
        real(c_double) :: initial_residual
        real(c_double) :: final_residual
        real(c_double) :: relative_residual
        real(c_double) :: setup_seconds
        real(c_double) :: solve_seconds
        real(c_double) :: analysis_seconds
        real(c_double) :: prepare_seconds
        real(c_double) :: probe_seconds
        real(c_double) :: diagnostics_seconds
        real(c_double) :: local_factor_seconds
        real(c_double) :: restart_seconds
        integer(c_int64_t) :: escalations
        integer(c_int64_t) :: probe_iterations
        real(c_double) :: probe_final_residual
        integer(c_int64_t) :: hard_dofs
        integer(c_int64_t) :: local_direct_regions
        integer(c_int64_t) :: largest_local_region
        integer(c_int64_t) :: local_factor_dofs
        integer(c_int64_t) :: unique_local_factor_dofs
        integer(c_int64_t) :: local_factor_bytes
        integer(c_int64_t) :: overlap_layers
        integer(c_int) :: preconditioner_reused
        integer(c_int64_t) :: solve_sequence
        integer(c_int64_t) :: krylov_workspace_bytes
    end type

    public :: hybit_solver_create, hybit_solver_destroy
    public :: hybit_solver_set_tolerances, hybit_solver_set_max_iterations, hybit_solver_set_backend
    public :: hybit_solver_set_hybrid_enabled, hybit_solver_set_overlap_layers
    public :: hybit_matrix_create_csr_f64, hybit_matrix_destroy, hybit_solve
    public :: hybit_prepare, hybit_prepared_destroy, hybit_solve_prepared

    interface
        integer(c_int) function hybit_solver_create(out_solver) bind(C, name="hybit_solver_create")
            import :: c_int, c_ptr
            type(c_ptr), intent(out) :: out_solver
        end function

        subroutine hybit_solver_destroy(solver) bind(C, name="hybit_solver_destroy")
            import :: c_ptr
            type(c_ptr), value :: solver
        end subroutine

        integer(c_int) function hybit_solver_set_tolerances(solver, rtol, atol) bind(C, name="hybit_solver_set_tolerances")
            import :: c_int, c_ptr, c_double
            type(c_ptr), value :: solver
            real(c_double), value :: rtol, atol
        end function

        integer(c_int) function hybit_solver_set_max_iterations(solver, max_iterations) &
            bind(C, name="hybit_solver_set_max_iterations")
            import :: c_int, c_ptr, c_int64_t
            type(c_ptr), value :: solver
            integer(c_int64_t), value :: max_iterations
        end function

        integer(c_int) function hybit_solver_set_backend(solver, backend) bind(C, name="hybit_solver_set_backend")
            import :: c_int, c_ptr
            type(c_ptr), value :: solver
            integer(c_int), value :: backend
        end function

        integer(c_int) function hybit_solver_set_hybrid_enabled(solver, enabled) bind(C, name="hybit_solver_set_hybrid_enabled")
            import :: c_int, c_ptr
            type(c_ptr), value :: solver
            integer(c_int), value :: enabled
        end function

        integer(c_int) function hybit_solver_set_overlap_layers(solver, layers) bind(C, name="hybit_solver_set_overlap_layers")
            import :: c_int, c_ptr, c_int64_t
            type(c_ptr), value :: solver
            integer(c_int64_t), value :: layers
        end function

        integer(c_int) function hybit_matrix_create_csr_f64( &
            nrows, ncols, nnz, row_ptr, col_idx, values, index_base, out_matrix) &
            bind(C, name="hybit_matrix_create_csr_f64")
            import :: c_int, c_int32_t, c_double, c_ptr
            integer(c_int32_t), value :: nrows, ncols, nnz
            integer(c_int32_t), intent(in) :: row_ptr(*)
            integer(c_int32_t), intent(in) :: col_idx(*)
            real(c_double), intent(in) :: values(*)
            integer(c_int), value :: index_base
            type(c_ptr), intent(out) :: out_matrix
        end function

        subroutine hybit_matrix_destroy(matrix) bind(C, name="hybit_matrix_destroy")
            import :: c_ptr
            type(c_ptr), value :: matrix
        end subroutine


        integer(c_int) function hybit_prepare(solver, matrix, out_prepared) bind(C, name="hybit_prepare")
            import :: c_int, c_ptr
            type(c_ptr), value :: solver, matrix
            type(c_ptr), intent(out) :: out_prepared
        end function

        subroutine hybit_prepared_destroy(prepared) bind(C, name="hybit_prepared_destroy")
            import :: c_ptr
            type(c_ptr), value :: prepared
        end subroutine

        integer(c_int) function hybit_solve_prepared(prepared, matrix, b, x, report) &
            bind(C, name="hybit_solve_prepared")
            import :: c_int, c_ptr, c_double, hybit_solve_report
            type(c_ptr), value :: prepared, matrix
            real(c_double), intent(in) :: b(*)
            real(c_double), intent(inout) :: x(*)
            type(hybit_solve_report), intent(out) :: report
        end function

        integer(c_int) function hybit_solve(solver, matrix, b, x, report) bind(C, name="hybit_solve")
            import :: c_int, c_ptr, c_double, hybit_solve_report
            type(c_ptr), value :: solver, matrix
            real(c_double), intent(in) :: b(*)
            real(c_double), intent(inout) :: x(*)
            type(hybit_solve_report), intent(out) :: report
        end function
    end interface
end module hybit
