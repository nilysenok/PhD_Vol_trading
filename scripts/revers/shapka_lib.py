"""Шапка таблиц реверса: коммит кода и входы с путями и sha256 (proverka.md п. 4; Д5). Один писатель на все листы.
Путь входа печатается относительно корня (основной копии), не только имя: sigma_hyb_h1_dev.parquet из ch4_dev (dev) и из зоны Z2 — разные файлы."""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def kommit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def put(p: Path) -> str:
    """Путь относительно корня этой копии или основной копии (данные лежат там); иначе абсолютный."""
    from fastbt.cache import MAIN
    p = Path(p).resolve()
    for base in (ROOT, Path(MAIN).resolve()):
        try:
            return p.relative_to(base).as_posix()
        except ValueError:
            continue
    return p.as_posix()


def vhod(p: Path) -> str:
    return f"# вход: {put(p)} sha256={sha(p)}\n"


def vhody_test1() -> list[Path]:
    """Входы счёта test1: σ̂ dev и Z2, дневные бары и IMOEX Z2 (если заданы), таблицы дивидендов (если не NET)."""
    from fastbt import dividendy, sigma_z2, zona
    vh = [sigma_z2.dev_path(h) for h in sigma_z2.HORIZONS] + [sigma_z2.z2_path(h) for h in sigma_z2.HORIZONS]
    vh += [Path(os.environ[k]) for k in ("FASTBT_Z2_DAILY", "FASTBT_Z2_IMOEX") if os.environ.get(k)]
    return vh + ([] if zona.div_net() else list(dividendy._tables()))


def blok(period: str, extra: list[Path] | None = None) -> str:
    """«# коммит: …» и «# вход: …»; входы счёта — только для test1 (у train4 данные вне зоны Z2)."""
    vh = (vhody_test1() if period == "test1" else []) + list(extra or [])
    return f"# коммит: {kommit()}\n" + "".join(vhod(p) for p in vh)
