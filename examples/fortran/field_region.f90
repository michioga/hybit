program field_region
    use, intrinsic :: iso_c_binding
    use hybit
    implicit none
    type(hybit_dof_block) :: blocks(3)
    type(c_ptr) :: layout, matrix
    integer(c_int32_t) :: rp(8), ci(8)
    real(c_double) :: av(8)
    integer(c_int64_t) :: seeds(1), words(1)
    integer(c_int) :: status

    blocks(1) = hybit_dof_block(HYBIT_ENTITY_NODE,1_c_int64_t,1_c_int32_t,3_c_int32_t)
    blocks(2) = hybit_dof_block(HYBIT_ENTITY_NODE,1_c_int64_t,2_c_int32_t,1_c_int32_t)
    blocks(3) = hybit_dof_block(HYBIT_ENTITY_MULTIPLIER,9_c_int64_t,3_c_int32_t,3_c_int32_t)
    rp = [0,2,3,4,5,6,7,8]
    ci = [0,4,1,2,3,4,5,6]
    av = 1.0_c_double
    seeds(1) = 0_c_int64_t
    words(1) = 0_c_int64_t
    status = hybit_dof_layout_create(blocks,3_c_size_t,layout)
    if (status /= HYBIT_OK) stop 1
    status = hybit_matrix_create_csr_f64(7,7,8,rp,ci,av,0,matrix)
    if (status /= HYBIT_OK) stop 2
    status = hybit_dof_layout_expand_blocks(layout,seeds,1_c_size_t,words,1_c_size_t)
    if (status /= HYBIT_OK .or. words(1) /= 7) stop 3
    status = hybit_dof_layout_grow_region(layout,matrix,seeds,1_c_size_t,1_c_size_t,words,1_c_size_t)
    if (status /= HYBIT_OK .or. words(1) /= int(z'77',c_int64_t)) stop 4
    print *, 'H1-B Fortran region:', words(1)
    call hybit_matrix_destroy(matrix)
    call hybit_dof_layout_destroy(layout)
end program
