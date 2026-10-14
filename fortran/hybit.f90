module hybit
    use, intrinsic :: iso_c_binding
    implicit none
    private

    integer(c_int), parameter, public :: HYBIT_OK = 0
    integer(c_int), parameter, public :: HYBIT_BACKEND_AUTO = 0
    integer(c_int), parameter, public :: HYBIT_BACKEND_CSR32 = 1
    integer(c_int), parameter, public :: HYBIT_BACKEND_ABTM = 2

    ! Entity-kind codes and field IDs are independent of the existing solver ABI.
    integer(c_int32_t), parameter, public :: HYBIT_ENTITY_NODE = 1
    integer(c_int32_t), parameter, public :: HYBIT_ENTITY_EDGE = 2
    integer(c_int32_t), parameter, public :: HYBIT_ENTITY_FACE = 3
    integer(c_int32_t), parameter, public :: HYBIT_ENTITY_ELEMENT = 4
    integer(c_int32_t), parameter, public :: HYBIT_ENTITY_MULTIPLIER = 5
    integer(c_int32_t), parameter, public :: HYBIT_ENTITY_CONSTRAINT = 6
    integer(c_int32_t), parameter, public :: HYBIT_ENTITY_OTHER = 255

    ! BIND(C) matches the padding/alignment of hybit_dof_block_t.
    type, bind(C), public :: hybit_dof_block
        integer(c_int32_t) :: entity_kind
        integer(c_int64_t) :: entity_id
        integer(c_int32_t) :: field_id
        integer(c_int32_t) :: components
    end type

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

    public :: hybit_set_num_threads, hybit_num_threads
    public :: hybit_solver_create, hybit_solver_destroy
    public :: hybit_solver_set_tolerances, hybit_solver_set_max_iterations, hybit_solver_set_backend
    public :: hybit_solver_set_hybrid_enabled, hybit_solver_set_overlap_layers
    public :: hybit_matrix_create_csr_f64, hybit_matrix_destroy, hybit_solve
    public :: hybit_prepare, hybit_prepared_destroy, hybit_solve_prepared
    public :: hybit_dof_layout_create, hybit_dof_layout_destroy
    public :: hybit_dof_layout_num_dofs, hybit_dof_layout_block, hybit_dof_layout_locate
    public :: hybit_dof_layout_expand_blocks, hybit_dof_layout_grow_region

    interface
        integer(c_int) function hybit_set_num_threads(threads) bind(C, name="hybit_set_num_threads")
            import :: c_int, c_int32_t
            integer(c_int32_t), value :: threads
        end function

        integer(c_int32_t) function hybit_num_threads() bind(C, name="hybit_num_threads")
            import :: c_int32_t
        end function
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


        ! Block indices, components and DOF ranges are zero-based, also here.
        integer(c_int) function hybit_dof_layout_create(blocks, nblocks, out_layout) &
            bind(C, name="hybit_dof_layout_create")
            import :: c_int, c_size_t, c_ptr, hybit_dof_block
            type(hybit_dof_block), intent(in) :: blocks(*)
            integer(c_size_t), value :: nblocks
            type(c_ptr), intent(out) :: out_layout
        end function

        subroutine hybit_dof_layout_destroy(layout) bind(C, name="hybit_dof_layout_destroy")
            import :: c_ptr
            type(c_ptr), value :: layout
        end subroutine

        integer(c_int) function hybit_dof_layout_num_dofs(layout, out_dofs) &
            bind(C, name="hybit_dof_layout_num_dofs")
            import :: c_int, c_ptr, c_int64_t
            type(c_ptr), value :: layout
            integer(c_int64_t), intent(out) :: out_dofs
        end function

        integer(c_int) function hybit_dof_layout_block(layout, index, out_block, out_start, out_end) &
            bind(C, name="hybit_dof_layout_block")
            import :: c_int, c_ptr, c_size_t, c_int64_t, hybit_dof_block
            type(c_ptr), value :: layout
            integer(c_size_t), value :: index
            type(hybit_dof_block), intent(out) :: out_block
            integer(c_int64_t), intent(out) :: out_start, out_end
        end function

        integer(c_int) function hybit_dof_layout_locate(layout, dof, block_index, component) &
            bind(C, name="hybit_dof_layout_locate")
            import :: c_int, c_ptr, c_size_t, c_int64_t, c_int32_t
            type(c_ptr), value :: layout
            integer(c_int64_t), value :: dof
            integer(c_size_t), intent(out) :: block_index
            integer(c_int32_t), intent(out) :: component
        end function

        ! Packed output bit k represents zero-based DOF k (use BTEST).
        ! For empty seeds, caller may provide a dummy array with seed_count=0.
        integer(c_int) function hybit_dof_layout_expand_blocks(layout, seeds, seed_count, words, word_count) &
            bind(C, name="hybit_dof_layout_expand_blocks")
            import :: c_int, c_ptr, c_int64_t, c_size_t
            type(c_ptr), value :: layout
            integer(c_int64_t), intent(in) :: seeds(*)
            integer(c_size_t), value :: seed_count, word_count
            integer(c_int64_t), intent(out) :: words(*)
        end function

        integer(c_int) function hybit_dof_layout_grow_region( &
            layout, matrix, seeds, seed_count, hops, words, word_count) &
            bind(C, name="hybit_dof_layout_grow_region")
            import :: c_int, c_ptr, c_int64_t, c_size_t
            type(c_ptr), value :: layout, matrix
            integer(c_int64_t), intent(in) :: seeds(*)
            integer(c_size_t), value :: seed_count, hops, word_count
            integer(c_int64_t), intent(out) :: words(*)
        end function

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
