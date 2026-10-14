#include "hybit.hpp"
#include <iostream>
int main() {
    const hybit::DofLayout layout({
        {HYBIT_ENTITY_NODE,1,1,3},
        {HYBIT_ENTITY_NODE,1,2,1},
        {HYBIT_ENTITY_MULTIPLIER,9,3,3}
    });
    const auto a = hybit::Matrix::from_csr(7,7,
        {0,2,3,4,5,6,7,8}, {0,4,1,2,3,4,5,6}, {1,1,1,1,1,1,1,1});
    const auto initial = layout.expand_blocks({0});
    const auto region = layout.grow_region(a,{0},1);
    std::cout << "H1-B C++ region: " << region.at(0) << '\n';
    return initial.at(0) == 7 && region.at(0) == 0x77 ? 0 : 1;
}
