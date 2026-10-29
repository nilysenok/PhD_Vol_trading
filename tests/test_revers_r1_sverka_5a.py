"""Реверс р.1: разложение шага 5а (scripts/revers/r1_sverka_5a.py, docs/tables/revers_r1/sverka_5a.csv).
Сумма шагов = 5а (остаток < 1e-3), 5а и E0 = сверка C/D, утечка меняет позиции. python3 -m pytest tests/test_revers_r1_sverka_5a.py -q"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "docs/tables/revers_r1"
KOL = ["delta", "delta_2022", "delta_2023", "delta_2024"]
SHAGI = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "E5"]
M = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading")


def _t(path=TAB / "sverka_5a.csv"):
    return pd.read_csv(path, comment="#")


def nesoshlos(t: pd.DataFrame) -> list:
    """(объект, столбец), где ΣΔ шагов P0…E5 ≠ строка 5а или Δ шага ≠ разность Шарпов соседних шагов."""
    osh = []
    for ob, g in t.groupby("obekt"):
        s = g.set_index("shag")
        for k in KOL:
            sh = s.loc[["E0", *SHAGI], k.replace("delta", "sharpe")].to_numpy()
            if not abs(s.loc[SHAGI, k].sum() - s.loc["5а", k]) < 1e-3 or not np.allclose(np.diff(sh), s.loc[SHAGI, k], atol=1e-12):
                osh.append((ob, k))
    return osh


def test_summa_shagov_ravna_5a_i_ostatok():
    t = _t()
    assert set(t["obekt"]) == {"A", "C", "D"} and nesoshlos(t) == []
    ost = t[t["shag"] == "ост"][KOL].to_numpy()
    assert np.abs(ost).max() < 1e-3


def _v(t, ob, shag, kol):
    r = t[(t["obekt"] == ob) & (t["shag"] == shag)]
    assert len(r) == 1, (ob, shag)
    return float(r[kol].iloc[0])


def test_5a_i_e0_kak_v_sverke_cd():
    t, ref = _t(), _t(TAB / "sverka_predzashchita_CD.csv")
    for ob in ("C", "D"):
        for k in KOL:
            assert abs(_v(t, ob, "5а", k) - _v(ref, ob, "5а", k)) < 1e-3
            sk = k.replace("delta", "sharpe")
            assert abs(_v(t, ob, "E0", sk) - _v(ref, ob, "4", sk)) < 1e-3


def test_mutatsiya_delta_ot_e0_vmesto_predydushchego_lovitsya():
    """Мутация: Δ шага P4 взята от E0, а не от P3 — сумма шагов ≠ 5а, проверка ловит."""
    t = _t()
    m = (t["obekt"] == "D") & (t["shag"] == "P4")
    e0 = t.loc[(t["obekt"] == "D") & (t["shag"] == "E0"), "sharpe"].iloc[0]
    t.loc[m, "delta"] = t.loc[m, "sharpe"] - e0
    assert ("D", "delta") in nesoshlos(t)


def test_utechka_menyaet_pozicii_v_tablice():
    t = _t()
    for ob in ("A", "C", "D"):
        assert _v(t, ob, "P1", "dolya_izm_poz") > 0.01 and abs(_v(t, ob, "P1", "delta")) > 0.1


@pytest.mark.skipif(not (M / "data/regen/test1_2022_2024/daily/ohlcv_daily_1850.parquet").exists(), reason="нет баров Z2")
def test_utechka_pereklyuchatel_menyaet_pozicii_progon(monkeypatch):
    """Прогон одной единицы (SBER, S1, A) в зоне Z2: маска [1] и [0] дают разные позиции 2022–2023."""
    sys.path.insert(0, str(ROOT / "scripts" / "revers"))
    import dataclasses
    import r1_sverka_5a as S
    from fastbt import data, pipeline, zona
    for k, v in S.konfig("P0")[1].items():
        if k != "FASTBT_CACHE" or "FASTBT_CACHE" not in os.environ:
            monkeypatch.setenv(k, v)
    with zona.zone_ctx("test1"):
        data.load_ticker.cache_clear()
        poz = {}
        for mo in (True, False):
            cfg = pipeline.z2_config(dataclasses.replace(S.BASE, mask_old=mo, approaches="A", years_a=(2022, 2023)))
            df, _ = pipeline.run_unit("SBER", "S1", cfg)
            poz[mo] = df[df["approach"] == "A"].set_index(["date", "test_year"])["position"]
    assert pd.to_datetime(poz[False].index.get_level_values(0)).max() < zona.HARD
    a, b = poz[True].align(poz[False])
    assert len(a) > 400 and (a - b).abs().max() > 0
