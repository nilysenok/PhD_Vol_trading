# -*- coding: utf-8 -*-
"""Уборка data/fastbt/numba_cache/<хэш кода>/: оставляет текущий хэш и N последних по времени.
По умолчанию dry-run; удаление только с --da. Удаляет только внутри каталога кэша (симлинки и `..` — отказ)."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KESH = ROOT / "data" / "fastbt" / "numba_cache"


def _razmer(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def _bezopasen(p: Path, kesh: Path) -> bool:
    """Каталог — прямой потомок kesh, не симлинк, без `..`."""
    if p.is_symlink() or not p.is_dir() or ".." in p.parts:
        return False
    return p.resolve().parent == kesh.resolve()


def ubrat(kesh: Path, tekushchiy: str | None, n: int = 5, da: bool = False) -> tuple[list[Path], int]:
    """Возвращает (удаляемые каталоги, байт). Удаляет только при da=True и после проверки всех путей."""
    if n < 0:
        raise ValueError("n < 0")
    if not kesh.is_dir() or kesh.is_symlink():
        raise ValueError(f"не каталог кэша: {kesh}")
    dirs = [p for p in kesh.iterdir() if p.is_dir() or p.is_symlink()]
    dirs.sort(key=lambda p: p.lstat().st_mtime, reverse=True)
    ostav = {p.name for p in dirs[:n]}
    if tekushchiy:
        ostav.add(tekushchiy)
    lishnie = [p for p in dirs if p.name not in ostav]
    for p in lishnie:
        if not _bezopasen(p, kesh):
            raise ValueError(f"путь вне каталога кэша или симлинк: {p}")
    byte = sum(_razmer(p) for p in lishnie)
    if da:
        for p in lishnie:
            shutil.rmtree(p)
    return lishnie, byte


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=5, help="сколько последних оставить (кроме текущего)")
    ap.add_argument("--da", action="store_true", help="реально удалить (иначе dry-run)")
    ap.add_argument("--kesh", type=Path, default=KESH)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from fastbt import cache

    tek = cache.code_hash()[:16]
    lishnie, byte = ubrat(a.kesh, tek, a.n, a.da)
    glagol = "удалено" if a.da else "dry-run, было бы удалено"
    print(f"текущий хэш {tek}; {glagol}: {len(lishnie)} каталогов, {byte / 1e6:.1f} МБ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
