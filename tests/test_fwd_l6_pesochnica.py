"""Л6: песочница открытия — сторож путей и подготовка данных (быстрая часть). Полный сквозной прогон песочницы — `otkrytie_harj_fwd.py --pesochnica <tmp>` (минуты, см. GATES_FWD).
Доказуемо: ни реальные новые файлы, ни реальный `rv_fwd_SEALED`, ни реальный замок не открываются; реальный замок не создаётся и не меняется."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/fwd"))
import otkrytie_guard as GD  # noqa: E402
import otkrytie_pesochnica as PS  # noqa: E402

REAL = [ROOT / "data/fwd/sber_m10/SBER_m10_fwd.parquet", ROOT / "data/fwd/mkt_m10/IMOEX_m10_fwd.parquet", ROOT / "data/fwd/sber_m10/rv_fwd_SEALED.parquet",
        ROOT / "data/fwd/models/OPENED_HARJ_FWD.json", ROOT / "data/fwd/models/new/x1.parquet", ROOT / "data/fwd/mkt_m10/AFLT_m10_fwd.parquet"]


@pytest.mark.parametrize("p", REAL, ids=lambda p: p.name)
def test_storozh_blokiruet_realnye_puti_do_otkrytiya(p):
    log = []
    with GD.guard(ROOT, log):
        for fn in (pd.read_parquet, pd.read_csv, open):
            with pytest.raises(GD.Forbidden):
                fn(p)
        with pytest.raises(GD.Forbidden):
            open(str(p), "w")
    assert log == []


def test_storozh_propuskaet_arkhiv_i_tmp_i_vosstanavlivaet_funktsii(tmp_path):
    orig = (pd.read_parquet, pd.read_csv, open)
    f = tmp_path / "a.csv"
    f.write_text("x\n1\n", encoding="utf-8")
    with GD.guard(ROOT, []):
        assert pd.read_csv(f)["x"].iloc[0] == 1
        assert not GD.is_forbidden(ROOT / "data/iss_check/bars_10m_full/SBER.parquet", ROOT) and not GD.is_forbidden(ROOT / "data/fwd/sber_m10/SBER_m10_overlap.parquet", ROOT)
    assert (pd.read_parquet, pd.read_csv, open) == orig


def test_mutatsiya_storozha_realnyj_zamok_ne_zashchishchyon(monkeypatch):
    monkeypatch.setattr(GD, "FORBIDDEN", ("data/fwd/sber_m10/*_fwd*",))  # мутант: замок убран из списка запрещённых
    assert not GD.is_forbidden(ROOT / "data/fwd/models/OPENED_HARJ_FWD.json", ROOT) and GD.is_forbidden(REAL[0], ROOT)


def test_podgotovka_dannykh_v_tmp_bez_chteniya_realnykh_novykh(tmp_path):
    real_lock = ROOT / "data/fwd/models/OPENED_HARJ_FWD.json"
    before = (real_lock.exists(), real_lock.stat().st_mtime_ns if real_lock.exists() else None)
    log = []
    with GD.guard(ROOT, log):
        p, models = PS.prepare_data(tmp_path)
    assert PS.paths_ok(p) == [] and tmp_path in Path(p.lock).parents and tmp_path in Path(p.new_sber).parents and tmp_path in Path(p.sealed_rv).parents
    real_reads = [x for x in log if x.startswith(str(ROOT))]
    assert real_reads and not any("_fwd" in x or "OPENED" in x for x in real_reads)  # читался только реальный архив; реальные новые, запечатанный, замок — нет
    arch, new = pd.read_parquet(p.bars_dir / "SBER.parquet"), pd.read_parquet(p.new_sber)
    assert arch["begin"].max() <= PS.AE + pd.Timedelta(hours=23, minutes=59) < new["begin"].min() and new["begin"].min() >= PS.AE + pd.Timedelta(days=1)
    assert new["begin"].max() <= PS.LAST + pd.Timedelta(hours=23, minutes=59) and len(list(p.new_mkt_dir.glob("*_m10_fwd.parquet"))) == 17
    sealed = pd.read_parquet(p.sealed_rv)
    assert sealed.index.name == "date" and sealed.index.min() >= PS.AE
    for name, col in PS.MODELS:
        assert (pd.read_parquet(models / f"{name}.parquet")[col] <= PS.AE).all()
    assert (real_lock.exists(), real_lock.stat().st_mtime_ns if real_lock.exists() else None) == before  # реальный замок не тронут
    assert PS.paths_ok(type("P", (), dict(bars_dir=ROOT / "data/iss_check", imoex_archive=ROOT / "x", new_sber=REAL[0], new_mkt_dir=ROOT / "y", sealed_rv=REAL[2], lock=REAL[3]))())  # мутант: пути в репозитории ловятся
