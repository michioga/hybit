"""C1-G5I: independently compare two exact 12-row raw-reconstructable evidence sets.

The full fail-closed raw validation is in the adjacent C1G5I_control.py and
C1G5I_p2_control.py modules. This module performs the matched comparison.
"""
from __future__ import annotations
from pathlib import Path
import C1G5I_control as off
import C1G5I_p2_control as on
import C1G5I_compare_p2 as compare


def audit(repo:Path,default:Path,p2:Path):
    before,_,_=off.reconstruct(repo,default)
    after,_,_=on.reconstruct(repo,p2)
    if default.joinpath('results.csv').read_bytes()!=before:
        raise ValueError('P2-off raw and saved data mismatch')
    if p2.joinpath('results.csv').read_bytes()!=after:
        raise ValueError('P2-on raw and saved data mismatch')
    return compare.compare(compare.read(default/'results.csv',compare.PRIOR_SHA),
                           compare.read(p2/'results.csv'))
