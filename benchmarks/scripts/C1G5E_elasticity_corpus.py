#!/usr/bin/env python3
"""Independent synthetic triangular P1 plane-strain elasticity FEM family.

Development/corpus tool only. Not invoked by the production HyBIT solver.
K = sum_t area_t B_t^T D_t B_t, with full Dirichlet displacement on x=0.
For E>0, -1<nu<0.5, positive areas, connected triangulation and clamped
edge, the assembled bilinear form is strictly positive for every nonzero
free-DOF vector. The output is symmetric Matrix Market, not a claim that a
Matrix Market header is an SPD certificate for arbitrary third-party files.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

NX, NY = 41, 41
CASES = ('elastic_uniform', 'elastic_stiff_lens', 'elastic_shear_layers')
FAMILY = 'tri_p1_plane_strain_elasticity_2dof_clamped'


def geometry():
    coords = []
    for j in range(NY):
        for i in range(NX):
            x = i / (NX - 1)
            y = j / (NY - 1)
            # Perturb interior nodes only; uniform boundaries and positive area.
            shape = math.sin(math.pi * x) * math.sin(math.pi * y)
            dx = 0.08 * shape * math.sin(3.1 * math.pi * y) / (NX - 1)
            dy = 0.08 * shape * math.sin(2.3 * math.pi * x) / (NY - 1)
            coords.append((x + dx, y + dy))
    def node(i, j):
        return j * NX + i
    triangles = []
    for j in range(NY - 1):
        for i in range(NX - 1):
            a, b, c, d = node(i, j), node(i + 1, j), node(i, j + 1), node(i + 1, j + 1)
            if (i + 2*j) % 2:
                triangles += [(a, b, c), (b, d, c)]
            else:
                triangles += [(a, b, d), (a, d, c)]
    return coords, triangles


def elasticity_tensor(e, nu):
    if not math.isfinite(e) or e <= 0 or not (-1 < nu < 0.5):
        raise ValueError('invalid elasticity material (E>0, -1<nu<0.5 required)')
    lam = e * nu / ((1 + nu) * (1 - 2 * nu))
    mu = e / (2 * (1 + nu))
    if not (lam + mu > 0 and mu > 0):
        raise ValueError('strain-energy positivity violated')
    return ((lam + 2*mu, lam, 0.0), (lam, lam + 2*mu, 0.0), (0.0, 0.0, mu))


def material(case, x, y):
    if case == 'elastic_uniform':
        return 1.0, 0.30
    if case == 'elastic_stiff_lens':
        lens = ((x-0.62)/0.19)**2 + ((y-0.48)/0.25)**2
        return (500.0 if lens < 1.0 else 1.0), 0.30
    if case == 'elastic_shear_layers':
        return (120.0 if (int(9*y) % 3) == 1 else 1.0), 0.44
    raise ValueError('unrecognized FEM case')


def assemble(case, nx=NX, ny=NY):
    if nx != NX or ny != NY:
        raise ValueError('only version-locked 41x41 mesh is allowed')
    coords, triangles = geometry()
    # Fixed x=0 boundary; all u,v DOFs there are eliminated exactly.
    free = {}
    for idx in range(len(coords)):
        if idx % NX == 0:
            continue
        free[2*idx] = len(free)
        free[2*idx + 1] = len(free)
    n = len(free)
    entries = {}
    min_twice_area = math.inf
    max_twice_area = 0.0
    material_bounds = [math.inf, 0.0]
    for verts in triangles:
        (xa, ya), (xb, yb), (xc, yc) = (coords[k] for k in verts)
        twice = (xb-xa)*(yc-ya) - (xc-xa)*(yb-ya)
        if not math.isfinite(twice) or twice <= 0:
            raise ValueError('degenerate or inverted element')
        min_twice_area = min(min_twice_area, twice)
        max_twice_area = max(max_twice_area, twice)
        gx = ((yb-yc)/twice, (yc-ya)/twice, (ya-yb)/twice)
        gy = ((xc-xb)/twice, (xa-xc)/twice, (xb-xa)/twice)
        B = [[0.0]*6 for _ in range(3)]
        for t in range(3):
            B[0][2*t] = gx[t]
            B[1][2*t+1] = gy[t]
            B[2][2*t] = gy[t]
            B[2][2*t+1] = gx[t]
        xc0 = (xa+xb+xc)/3
        yc0 = (ya+yb+yc)/3
        e, nu = material(case, xc0, yc0)
        material_bounds[0] = min(material_bounds[0], e)
        material_bounds[1] = max(material_bounds[1], e)
        D = elasticity_tensor(e, nu)
        DB = [[sum(D[r][k]*B[k][c] for k in range(3)) for c in range(6)] for r in range(3)]
        dofs = [2*k+t for k in verts for t in (0,1)]
        for p in range(6):
            row = free.get(dofs[p])
            if row is None:
                continue
            for q in range(p, 6):
                col = free.get(dofs[q])
                if col is None:
                    continue
                v = (twice/2)*sum(B[k][p]*DB[k][q] for k in range(3))
                key = (min(row,col),max(row,col))
                entries[key] = entries.get(key, 0.0) + v
    expected = 2*(NX-1)*NY
    if n != expected:
        raise ValueError('unexpected number of unconstrained degrees of freedom')
    diagonal = [entries.get((i,i), 0.0) for i in range(n)]
    if not all(math.isfinite(v) and v > 0 for v in diagonal):
        raise ValueError('nonpositive / nonfinite assembled diagonal')
    # Every free node is edge-connected to a clamped node in this conforming mesh.
    # This plus positive strain-energy density rules out all rigid-body modes.
    header = (f'%%MatrixMarket matrix coordinate real symmetric\n'
              f'% C1-G5E {FAMILY} case={case} clamped_left=ux,uy\n'
              f'{n} {n} {len(entries)}\n')
    payload = io.StringIO()
    payload.write(header)
    for (i,j),v in sorted(entries.items()):
        if not math.isfinite(v):
            raise ValueError('nonfinite stiffness entry')
        payload.write(f'{i+1} {j+1} {v:.17e}\n')
    data = payload.getvalue().encode('ascii')
    meta = dict(family_id=FAMILY, case=case, mesh='perturbed 41x41 P1 triangles; alternating diagonals',
                domain='unit square', physics='plane-strain linear elasticity', ndof=n,
                element_count=len(triangles), free_nodes=n//2, fixed_nodes=NY,
                min_double_area=min_twice_area, max_double_area=max_twice_area,
                min_young=material_bounds[0],max_young=material_bounds[1],
                spd_evidence='K=sum(area*B^T*D*B), E>0, -1<nu<0.5; non-inverted conforming mesh; full ux,uy clamped on x=0; eliminates rigid modes',
                rhs='Rust manufactured/index-smooth/seeded-random/point-load; no physical external loads',
                matrix_sha256=hashlib.sha256(data).hexdigest())
    return data, meta


def generate(out: Path, check: bool):
    out = out.resolve()
    data_list = [(case,*assemble(case)) for case in CASES]
    inventory = dict(schema='C1G5E_TRI_P1_V1', family_id=FAMILY, generator='C1G5E_elasticity_corpus.py',
                     cases=[meta for _,_,meta in data_list], independent_from='C1-G4 structured scalar diffusion; different PDE and 2 DOF/node')
    files = {f'{case}.mtx': data for case,data,_ in data_list}
    files['inventory.json'] = (json.dumps(inventory,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if check:
        for name,content in files.items():
            p=out/name
            if not p.is_file() or p.read_bytes()!=content:
                raise ValueError('corpus missing or modified: '+str(p))
        print(f'C1-G5E CORPUS CHECK PASS: {len(CASES)} matrices, 1 independent FEM generator family')
    else:
        out.mkdir(parents=True,exist_ok=True)
        for name,content in files.items():
            p=out/name
            if p.exists() and p.read_bytes()!=content:
                raise ValueError('refusing to overwrite divergent corpus: '+str(p))
            if not p.exists():
                p.write_bytes(content)
        print(f'C1-G5E CORPUS GENERATED: {out}, {len(CASES)} symmetric FEM stiffness matrices')
    for _,_,meta in data_list:
        print(meta['case'],meta['ndof'],meta['matrix_sha256'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--check',action='store_true')
    a=p.parse_args()
    generate(a.out,a.check)
if __name__=='__main__':main()
