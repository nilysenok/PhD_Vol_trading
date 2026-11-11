"""Л7.5 (быстрая часть): песочница повтора не может открыть реальные первый замок, v1, FREEZE_v2, замок r2 и новые файлы; поддельный первый замок и состояние «до/после» работают.
Полный сквозной прогон — `otkrytie_harj_fwd.py --pesochnica <tmp> --povtor` (минуты, см. GATES_FWD, Л7.5)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/fwd"))
import otkrytie_guard as GD  # noqa: E402
import otkrytie_pes_povtor as PP  # noqa: E402
import otkrytie_pesochnica as PS  # noqa: E402
import otkrytie_povtor as PT  # noqa: E402

REAL = [ROOT / "data/fwd/models/OPENED_HARJ_FWD.json", ROOT / "data/fwd/models/OPENED_HARJ_FWD_r2.json", ROOT / "data/fwd/models/new/x1.parquet"]


@pytest.mark.parametrize("p", REAL, ids=lambda p: p.name)
def test_storozh_blokiruet_realnye_zamki_i_vykhody_povtora(p):
    log = []
    with GD.guard(ROOT, log):
        for fn in (pd.read_parquet, pd.read_csv, open):
            with pytest.raises(GD.Forbidden):
                fn(p)
        with pytest.raises(GD.Forbidden):
            open(str(p), "w")
    assert log == []


def test_mutatsiya_storozha_zamok_r2_ne_zashchishchyon(monkeypatch):
    monkeypatch.setattr(GD, "FORBIDDEN", ("data/fwd/sber_m10/*_fwd*", "data/fwd/models/OPENED_HARJ_FWD.json"))  # мутант: шаблон без звёздочки
    assert not GD.is_forbidden(ROOT / PT.LOCK2, ROOT) and GD.is_forbidden(REAL[0], ROOT)


def test_puti_peschochnicy_povtora_vne_repozitoriya_i_fake_zamok_v_tmp(tmp_path):
    lock1 = PP.fake_first_lock(tmp_path / "repo")
    assert tmp_path in lock1.parents and lock1.read_text(encoding="utf-8").count(PT.LOCK1_STATUS) == 1 and not (lock1.stat().st_mode & 0o222)
    real = type("P", (), dict(bars_dir=tmp_path, imoex_archive=tmp_path / "x", new_sber=tmp_path / "y", new_mkt_dir=tmp_path, sealed_rv=tmp_path / "z", lock=ROOT / PT.LOCK2))()
    assert PS.paths_ok(real)  # мутант: замок r2 песочницы — реальный путь, ловится
    assert PP.LOCK2_NAME == "OPENED_HARJ_FWD_r2.json"


def test_real_state_fiksiruet_sha_i_mtime_realnykh_faylov_bez_izmenenij():
    a, b = PP.real_state(), PP.real_state()
    assert a == b and set(a) >= {"data/fwd/models/OPENED_HARJ_FWD.json", "docs/otchety/harj_fwd/FREEZE.json", "docs/otchety/harj_fwd/FREEZE_v2.json", PT.LOCK2, "FWD_*", "new/*"}
