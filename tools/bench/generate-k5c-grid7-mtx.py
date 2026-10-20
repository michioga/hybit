#!/usr/bin/env python3
"""Generate a deterministic SPD 3D 7-point grid in Matrix Market symmetric form.

Writes only lower-triangle entries to limit input size, which expands to a full
CSR in the Matrix Market reader. No dependency beyond the Python standard library.
"""
import argparse
from pathlib import Path


def counts(nx: int, ny: int, nz: int) -> tuple[int, int, int]:
    n = nx * ny * nz
    lower = (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1)
    return n, n + lower, n + 2 * lower


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nx', type=int, default=120)
    parser.add_argument('--ny', type=int, default=100)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dims = (args.nx, args.ny, args.nz)
    if any(d < 2 for d in dims):
        parser.error('dimensions must all be >= 2')
    n, lines, nnz = counts(*dims)
    if n > 1_200_000 or nnz > 16_000_000:
        parser.error('grid exceeds K5-C DOF or nnz safety cap')
    if args.output.exists():
        parser.error(f'refusing overwrite: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    xy = args.nx * args.ny
    # Buffer writes with a 1 MiB user-space buffer; no full COO in RAM.
    with args.output.open('w', encoding='ascii', buffering=1024 * 1024, newline='\n') as out:
        out.write('%%MatrixMarket matrix coordinate real symmetric\n')
        out.write('% K5-C deterministic SPD 3D grid7, diagonal = degree + 1\n')
        out.write(f'{n} {n} {lines}\n')
        for z in range(args.nz):
            for y in range(args.ny):
                for x in range(args.nx):
                    idx = 1 + x + args.nx * (y + args.ny * z)
                    degree = ((x > 0) + (x + 1 < args.nx)
                              + (y > 0) + (y + 1 < args.ny)
                              + (z > 0) + (z + 1 < args.nz))
                    out.write(f'{idx} {idx} {degree+1}.0\n')
                    if x > 0:
                        out.write(f'{idx} {idx-1} -1.0\n')
                    if y > 0:
                        out.write(f'{idx} {idx-args.nx} -1.0\n')
                    if z > 0:
                        out.write(f'{idx} {idx-xy} -1.0\n')
    size = args.output.stat().st_size
    print(f'K5C GRID PASS: n={n}, mtx_entries={lines}, expanded_nnz={nnz}, bytes={size}, path={args.output}')
    if size > 384 * 1024 * 1024:
        raise SystemExit('generated file exceeds K5-C 384 MiB input limit')


if __name__ == '__main__':
    main()
