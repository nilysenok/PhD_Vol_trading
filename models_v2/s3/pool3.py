"""Параллельный счёт дизайнов (spawn, <= 6 процессов, nthread=1 в каждом)."""
from __future__ import annotations

import multiprocessing as mp

from models_v2.s3 import gen

_D = None


def _init(D):
    global _D
    _D = D


def _job(a):
    key, dname, spec, depth, delta, T, d_idx, variant = a
    return key, gen.run_design(_D[dname], spec, depth, delta, T, d_idx, variant)


def map_jobs(D, jobs, procs: int = 6) -> list:
    """jobs: (key, имя дизайна, spec, depth, delta, T, d_idx, variant) -> [(key, результат)]."""
    with mp.get_context("spawn").Pool(procs, initializer=_init, initargs=(D,)) as p:
        return list(p.imap_unordered(_job, jobs, chunksize=2))
