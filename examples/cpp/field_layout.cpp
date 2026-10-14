// H1 C++ RAII metadata smoke example.
#include "hybit.hpp"
#include <iostream>
#include <vector>
int main() {
    const std::vector<hybit_dof_block_t> blocks = {
        {HYBIT_ENTITY_NODE, 1, 10, 3},
        {HYBIT_ENTITY_NODE, 1, 20, 1},
        {HYBIT_ENTITY_MULTIPLIER, 5, 30, 3},
    };
    hybit::DofLayout layout(blocks);
    const auto address = layout.locate(5);
    std::cout << "H1 C++: dofs=" << layout.num_dofs()
              << " block=" << address.first << " component=" << address.second << '\n';
    return (layout.num_dofs() == 7 && address.first == 2 && address.second == 1) ? 0 : 1;
}
