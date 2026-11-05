"""Параллельный счёт блоков (spawn, <= 6 процессов, nthread=1 в каждом)."""
from __future__ import annotations

import multiprocessing as mp

from models_v2.s2 import boost

_F = None


def _init(F):
    global _F
    _F = F


def _job(a):
    key, spec, depth, delta, T, d_idx, variant = a
    return key, boost.run_block(_F, spec, depth, delta, T, d_idx, variant)


def map_jobs(F, jobs, procs: int = 6) -> list:
    """jobs: (key, spec, depth, delta, T, d_idx, variant) -> [(key, результат)]."""
    with mp.get_context("spawn").Pool(procs, initializer=_init, initargs=(F,)) as p:
        return list(p.imap_unordered(_job, jobs, chunksize=2))
