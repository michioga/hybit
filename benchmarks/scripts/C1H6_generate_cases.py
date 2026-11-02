#!/usr/bin/env python3
"""Generate the two independent C1-H5 SPD pilot matrices for C1-H6 Auto tests."""
from __future__ import annotations
import argparse
import math
from pathlib import Path

N = 160
LOCAL = 96
WEAK = 1e-4


def generate(global_chain: bool, destination: Path) -> None:
    rows = [dict() for _ in range(N)]
    diagonal = [2.0] * N
    for i in range(N-1):
        if global_chain or i < LOCAL - 1:
            coupling = 1.0
        elif i == LOCAL - 1:
            coupling = WEAK
            diagonal[i] += coupling
            diagonal[i+1] += coupling
        else:
            continue
        rows[i][i+1] = -coupling
        rows[i+1][i] = -coupling
    for i in range(N):
        rows[i][i] = diagonal[i]
    xref = []
    for i in range(N):
        if global_chain:
            x = math.sin(math.pi*(i+1)/(N+1)) + 0.11*math.sin(0.21*i)
        elif i < LOCAL:
            x = math.sin(math.pi*(i+1)/(LOCAL+1))
        else:
            x = 0.7 + 0.2 * math.sin(0.13*i)
        xref.append(x)
    b = [sum(v*xref[j] for j, v in row.items()) for row in rows]
    name = 'global_chain' if global_chain else 'localized_chain'
    base = destination / ('c1h6_' + name)
    entries = [(i+1, j+1, v) for i, row in enumerate(rows) for j, v in sorted(row.items())]
    base.with_suffix('.mtx').write_text(
        '%%MatrixMarket matrix coordinate real general\n'
        '% C1-H6 controlled SPD research matrix\n'
        f'{N} {N} {len(entries)}\n'
        + ''.join(f'{i} {j} {v:.17g}\n' for i,j,v in entries), encoding='utf-8'
    )
    (destination / (base.name+'.rhs.txt')).write_text(''.join(f'{v:.17g}\n' for v in b), encoding='utf-8')
    (destination / (base.name+'.xref.txt')).write_text(''.join(f'{v:.17g}\n' for v in xref), encoding='utf-8')
    # Auditable SPD construction and expected shape, not arbitrary random data.
    assert all(rows[i][j] == rows[j][i] for i in range(N) for j in rows[i])
    assert all(diagonal[i] > 0 for i in range(N))
    print(f'{name}: n={N} nnz={len(entries)}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    generate(False, args.output)
    generate(True, args.output)

if __name__ == '__main__':
    main()
