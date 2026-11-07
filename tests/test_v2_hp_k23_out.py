"""P1a, G7: SBER побитно с components.XJ/LF (реальные входы, пропуск при их отсутствии), таблицы QLIKE на общих датах, проверка записанного выхода (проверка должна уметь падать)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2 import run_fix
from models_v2.s10 import hp_k23 as K, hp_k23_calc as KC, hp_k23_out as KO, hp_k23_world as KW, hp_paths as P

REAL = P.bars_path("SBER").exists() and P.COMP.exists() and P.IMOEX.exists()


@pytest.mark.skipif(not REAL, reason="нет реальных входов SBER/components")
def test_sber_pobitno_components_i_feats_kesh():
    bars, imo = K.read_bars("SBER"), K.read_bars("IMOEX")
    w = KW.build_world(bars, imo, KW.pool_papers(K.read_bars))
    old = pd.read_parquet(P.ROOT / "data/regen/v2_sber_h1_deep_v1/b2/feats_b2.parquet")
    assert old.equals(w["feats"]) and len(w["grid"]) == 2271          # признаки F1–F6 побитно как в кэше SBER
    i2 = run_fix.load_i2("i2_choice_g132")
    rows = w["grid"][::250]
    fr = KC.forecast_frame(w, i2, rows, 2)
    d = KO.sber_diff(fr.assign(ticker="SBER", rv_tplus=fr["rv"]), pd.read_parquet(P.COMP))
    assert d["n"] == len(rows) and d["max_abs_diff_xj"] == 0.0 and d["max_abs_diff_lf"] == 0.0 and d["max_abs_diff_rv"] == 0.0


def _frame(n: int = 6) -> pd.DataFrame:
    d = pd.date_range("2018-01-01", periods=n)
    return pd.DataFrame({"d": d, "target_date": d + pd.Timedelta(days=1), "rv": np.arange(1.0, n + 1), "XJ": np.arange(2.0, n + 2), "LF": np.arange(3.0, n + 3)})


def test_sber_diff_lovit_otklonenie():
    comp = _frame()
    s = pd.DataFrame({"date": comp["d"], "t_plus": comp["target_date"], "rv_tplus": comp["rv"], "sigma2_k2": comp["XJ"], "sigma2_k3": comp["LF"]})
    ok = KO.sber_diff(s, comp)
    assert ok["n"] == 6 and ok["max_abs_diff_xj"] == ok["max_abs_diff_lf"] == 0.0
    bad = s.copy()
    bad.loc[3, "sigma2_k2"] += 1e-12
    assert KO.sber_diff(bad, comp)["max_abs_diff_xj"] > 0.0
    bad = s.copy()
    bad.loc[2, "sigma2_k3"] *= 1 + 1e-15
    assert KO.sber_diff(bad, comp)["max_abs_diff_lf"] > 0.0


def _panel(t: str, n: int = 8) -> pd.DataFrame:
    d = pd.date_range("2018-01-01", periods=n)
    rv = np.arange(1.0, n + 1)
    return pd.DataFrame({"ticker": t, "date": d, "t_plus": d + pd.Timedelta(days=1), "rv_tplus": rv, "sigma2_k1": rv * 2, "gap": False, "cross_seg": False,
                         "sigma2_k2": rv * 0.5, "sigma2_k3": rv, "hp_year": 2018.0})


def test_qlike_tablica_na_obshchih_datah_znanyj_otvet():
    """K1 = 2·RV, K2 = RV/2, K3 = RV: QLIKE r − ln r − 1 при r = 1/2, 2, 1; строки без K3 в общие даты не входят."""
    df = _panel("AFLT")
    df.loc[0:1, "sigma2_k3"] = np.nan
    t = KO.qlike_table(df)
    a = t[t["year"] == "all"].iloc[0]
    q = lambda r: r - np.log(r) - 1.0  # noqa: E731
    assert a["n"] == 6 and np.isclose(a["qlike_k1"], q(0.5)) and np.isclose(a["qlike_k2"], q(2.0)) and a["qlike_k3"] == 0.0
    s = KO.summary(t)
    r = s[(s["group"] == "M16") & (s["period"] == "all")].iloc[0]
    assert r["share_k2_lt_k1"] == 0.0 and r["share_k3_lt_k1"] == 1.0           # q(2) > q(0.5); q(1) = 0 < q(0.5)


def test_verify_lovit_nulevye_i_rannie_prognozy():
    """Проверка записанного падает: прогноз раньше первого года У-3; перепутанная бумага (t⁺ не как в панели)."""
    t = ["AFLT"]
    df, panel = _panel("AFLT"), _panel("AFLT")[["ticker", "date", "t_plus", "rv_tplus"]]
    meta = pd.DataFrame({"ticker": t, "y0_used": [2019]})
    comp = _frame()
    bad = KO.verify(df, panel, meta, comp)
    assert any("раньше первого года" in b for b in bad) and any("бумаг 1" in b for b in bad)
    df2 = df.copy()
    df2["t_plus"] = df2["t_plus"] + pd.Timedelta(days=1)
    assert any("не равны panel_base" in b for b in KO.verify(df2, panel, meta, comp))
