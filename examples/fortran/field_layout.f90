! H1 ISO_C_BINDING metadata smoke example. DOF indices are zero-based.
program field_layout
    use, intrinsic :: iso_c_binding
    use hybit
    implicit none
    type(hybit_dof_block) :: blocks(3), metadata
    type(c_ptr) :: handle
    integer(c_int) :: status
    integer(c_int64_t) :: n, first, last
    integer(c_size_t) :: which
    integer(c_int32_t) :: component
    blocks(1) = hybit_dof_block(HYBIT_ENTITY_NODE, 1_c_int64_t, 10_c_int32_t, 3_c_int32_t)
    blocks(2) = hybit_dof_block(HYBIT_ENTITY_NODE, 1_c_int64_t, 20_c_int32_t, 1_c_int32_t)
    blocks(3) = hybit_dof_block(HYBIT_ENTITY_MULTIPLIER, 5_c_int64_t, 30_c_int32_t, 3_c_int32_t)
    status = hybit_dof_layout_create(blocks, 3_c_size_t, handle)
    if (status /= HYBIT_OK) stop 1
    status = hybit_dof_layout_num_dofs(handle, n)
    if (status /= HYBIT_OK) stop 2
    status = hybit_dof_layout_block(handle, 2_c_size_t, metadata, first, last)
    if (status /= HYBIT_OK) stop 3
    status = hybit_dof_layout_locate(handle, 5_c_int64_t, which, component)
    if (status /= HYBIT_OK) stop 4
    print *, 'H1 Fortran: dofs=', n, 'block=', which, 'component=', component
    call hybit_dof_layout_destroy(handle)
    if (n /= 7 .or. first /= 4 .or. last /= 7) stop 5
    if (which /= 2 .or. component /= 1) stop 6
end program
