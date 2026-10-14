/* H1-B: ABI smoke test for undirected ABTM + mixed-field block closure. */
#include "hybit.h"
#include <inttypes.h>
#include <stdio.h>
int main(void) {
    const hybit_dof_block_t blocks[] = {
        {HYBIT_ENTITY_NODE,1,1,3},
        {HYBIT_ENTITY_NODE,1,2,1}, /* same node; different field */
        {HYBIT_ENTITY_MULTIPLIER,9,3,3}
    };
    const uint32_t rp[] = {0,2,3,4,5,6,7,8};
    const uint32_t ci[] = {0,4,1,2,3,4,5,6};
    const double av[] = {1,1,1,1,1,1,1,1};
    hybit_dof_layout_t *layout = NULL;
    hybit_matrix_t *matrix = NULL;
    const uint64_t seeds[] = {0};
    uint64_t mask[] = {0};
    int rc = hybit_dof_layout_create(blocks,3,&layout);
    if (rc != HYBIT_OK) return 1;
    rc = hybit_matrix_create_csr_f64(7,7,8,rp,ci,av,0,&matrix);
    if (rc == HYBIT_OK) rc = hybit_dof_layout_expand_blocks(layout,seeds,1,mask,1);
    if (rc == HYBIT_OK && mask[0] != UINT64_C(0x07)) rc = -1;
    if (rc == HYBIT_OK) rc = hybit_dof_layout_grow_region(layout,matrix,seeds,1,1,mask,1);
    if (rc == HYBIT_OK) printf("H1-B C region: 0x%02" PRIx64 "\n", mask[0]);
    hybit_matrix_destroy(matrix);
    hybit_dof_layout_destroy(layout);
    return rc == HYBIT_OK && mask[0] == UINT64_C(0x77) ? 0 : 1;
}
