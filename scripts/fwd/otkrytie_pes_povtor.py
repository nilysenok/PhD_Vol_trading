"""Л7: песочница ПОВТОРА открытия (`otkrytie_harj_fwd.py --pesochnica <tmp> --povtor`). Всё в tmp: поддельный первый замок «сбой», FREEZE v1 и v2 (настоящий писатель `freeze_v2.write`),
замок r2, выходы. Реальные первый замок, FREEZE.json, FREEZE_v2.json, замок r2 и новые файлы не открываются (сторож `otkrytie_guard`) и не меняются (sha и mtime до и после).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import freeze_v2 as FV
import otkrytie_povtor as PT

ROOT = FV.ROOT
LOCK2_NAME = Path(PT.LOCK2).name
GATES_L7 = "## Л7. лист\n\n| Гейт | EVIDENCE |\n|---|---|\n| Л7.1 | passed |\n\n"


def real_state() -> dict:
    """{путь: (sha256, mtime_ns) или None} реальных защищаемых файлов + списки реальных выходов; значения не разбираются, замок и v1 читаются как байты для sha."""
    st = {}
    for p in (FV.LOCK1, FV.V1, FV.OUT, ROOT / PT.LOCK2):
        st[str(p.relative_to(ROOT))] = (PT.sha(p), p.stat().st_mtime_ns) if p.exists() else None
    st["FWD_*"] = sorted(x.name for x in FV.REP.glob("FWD_*"))
    new = ROOT / "data/fwd/models/new"
    st["new/*"] = sorted(x.name for x in new.glob("*")) if new.exists() else []
    return st


def fake_first_lock(tmp_repo: Path) -> Path:
    """Поддельный первый замок в tmp: статус «сбой», только для проверок повтора."""
    p = tmp_repo / "data/fwd/models/OPENED_HARJ_FWD.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(status=PT.LOCK1_STATUS, prichina="KeyError: 'date' (песочница)", trassa="—"), ensure_ascii=False, indent=1), encoding="utf-8")
    os.chmod(p, 0o444)
    return p


def write_v2(rep: Path, lock1: Path, lock2: Path) -> Path:
    """FREEZE_v2.json в tmp настоящим писателем; вызывается внутри подмен разделов FREEZE (`otkrytie_pesochnica.freeze_patches`)."""
    out = rep / "FREEZE_v2.json"
    FV.write(path=out, v1_path=rep / "FREEZE.json", lock1_path=lock1, fix_commit="песочница", root=ROOT, lock2=PT.rel(lock2, ROOT))
    return out
