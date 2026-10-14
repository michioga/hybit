/* H1 metadata smoke example. Compile/link with existing HyBIT C ABI. */
#include "hybit.h"
#include <inttypes.h>
#include <stdio.h>
int main(void) {
    const hybit_dof_block_t blocks[] = {
        {HYBIT_ENTITY_NODE, 1, 10, 3}, /* displacement */
        {HYBIT_ENTITY_NODE, 1, 20, 1}, /* temperature */
        {HYBIT_ENTITY_MULTIPLIER, 5, 30, 3}, /* contact multipliers */
    };
    hybit_dof_layout_t *layout = NULL;
    uint64_t dofs = 0, start = 0, end = 0;
    hybit_dof_block_t out = {0};
    if (hybit_dof_layout_create(blocks, 3, &layout) != HYBIT_OK) return 1;
    int status = hybit_dof_layout_num_dofs(layout, &dofs);
    if (status == HYBIT_OK) status = hybit_dof_layout_block(layout, 2, &out, &start, &end);
    if (status == HYBIT_OK) printf("H1 C: dofs=%" PRIu64 " multiplier=[%" PRIu64 ",%" PRIu64 ")\n", dofs, start, end);
    hybit_dof_layout_destroy(layout);
    return status == HYBIT_OK && dofs == 7 && start == 4 && end == 7 ? 0 : 1;
}
