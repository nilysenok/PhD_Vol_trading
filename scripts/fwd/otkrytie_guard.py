"""Л6: сторож песочницы — доказуемо не читать реальные новые файлы и не трогать реальный замок.
На время `guard` подменяются `pd.read_parquet`, `pd.read_csv`, `builtins.open`, `io.open`, `os.open`: путь под `root`, попавший в FORBIDDEN, — исключение `Forbidden`
ДО открытия файла; остальные пути под `root/data` пишутся в журнал. Метаданные (stat/exists) и pyarrow-схема не блокируются (значений не читают).
"""
from __future__ import annotations

import builtins
import fnmatch
import io
import os
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

FORBIDDEN = ("data/fwd/sber_m10/*_fwd*", "data/fwd/mkt_m10/*_fwd*", "data/fwd/models/OPENED_HARJ_FWD*", "data/fwd/models/new/*")


class Forbidden(RuntimeError):
    """Попытка открыть реальный запечатанный файл, реальные новые бары или реальный замок."""


def is_forbidden(p, root) -> bool:
    try:
        rel = Path(os.fspath(p)).resolve().relative_to(Path(root).resolve()).as_posix()
    except (ValueError, TypeError, OSError):
        return False
    return any(fnmatch.fnmatch(rel, pat) for pat in FORBIDDEN)


@contextmanager
def guard(root, log: list):
    orig = dict(rp=pd.read_parquet, rc=pd.read_csv, op=builtins.open, io=io.open, os=os.open)

    def wrap(f):
        def w(p, *a, **k):
            if isinstance(p, (str, os.PathLike)):
                if is_forbidden(p, root):
                    raise Forbidden(f"чтение/запись запрещённого реального пути: {p}")
                if "/data/" in os.fspath(p):
                    log.append(os.fspath(p))
            return f(p, *a, **k)
        return w

    pd.read_parquet, pd.read_csv, builtins.open, io.open, os.open = (wrap(orig[k]) for k in ("rp", "rc", "op", "io", "os"))
    try:
        yield log
    finally:
        pd.read_parquet, pd.read_csv, builtins.open, io.open, os.open = (orig[k] for k in ("rp", "rc", "op", "io", "os"))
