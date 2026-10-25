"""Старый код c22a4b3 (V4, V41, S5R, S6R, SW) — только для проверки эквивалентности.

load_legacy() извлекает файлы `git show c22a4b3:moex_volatility/scripts/<f>` в
data/fastbt/legacy_c22a4b3/ (или берёт каталог из FASTBT_LEGACY_DIR) и импортирует.
Возвращает пространство имён L с модулями: L.sw, L.v4, L.v41, L.s5r, L.s6r.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

COMMIT = "c22a4b3"
FILES = ["strategies_walkforward.py", "strategies_walkforward_v4.py",
         "strategies_walkforward_v4_1.py", "s5_rerun.py", "s5s6_rerun.py"]
ROOT = Path(__file__).resolve().parent.parent
_L = None


def legacy_dir() -> Path:
    env = os.environ.get("FASTBT_LEGACY_DIR")
    if env:
        return Path(env)
    d = ROOT / "data" / "fastbt" / f"legacy_{COMMIT}"
    if not all((d / f).exists() for f in FILES):
        d.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            src = subprocess.run(["git", "show", f"{COMMIT}:moex_volatility/scripts/{f}"],
                                 cwd=ROOT, check=True, capture_output=True).stdout
            (d / f).write_bytes(src)
    return d


def load_legacy():
    global _L
    if _L is not None:
        return _L
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    d = str(legacy_dir())
    if d not in sys.path:
        sys.path.insert(0, d)
    mods = {}
    for key, name in [("sw", "strategies_walkforward"), ("s5r", "s5_rerun"),
                      ("s6r", "s5s6_rerun"), ("v4", "strategies_walkforward_v4"),
                      ("v41", "strategies_walkforward_v4_1")]:
        m = sys.modules.get(name)
        if m is not None and not str(getattr(m, "__file__", "")).startswith(d):
            raise RuntimeError(f"модуль {name} уже импортирован не из {d}")
        mods[key] = importlib.import_module(name)
    _L = SimpleNamespace(**mods)
    return _L
