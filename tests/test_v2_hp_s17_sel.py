"""P3-17, G17 (Ж-220, У-12/У-22): выбор на результатах с обучением на 17 через боевой путь `hp_s17_sel.run` на синтетике (сеть и данные проекта не читаются).
Знание ответа: σ² ряда s = RV·exp(sd·z) — чем меньше sd, тем ниже QLIKE; на M16 лучший xgb_A1, на E11 — xgb_A2. Мутации: E11 в критерии выбора; N объявлено как число строк / повторно другим числом; K6/K7 пересчитаны не из k4567."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from models_v2.s10 import hp_paths as P, hp_s17_sel as SEL, hp_stats_sel as S

M16T, E11T = ["AFLT", "ALRS", "HYDR", "IRAO"], ["GCHE", "MRKC"]
SD_M16 = {"k1": 0.50, "k4": 0.56, "k5": 0.56, "xgb_A1": 0.30, "xgb_A2": 0.45, "lgbm_A1": 0.40}
SD_E11 = {"k1": 0.50, "k4": 0.56, "k5": 0.56, "xgb_A1": 0.90, "xgb_A2": 0.20}


def make_k(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", periods=n)
    parts = []
    for t in M16T + ["SBER"] + E11T:
        grp = "M16" if t in M16T else ("SBER" if t == "SBER" else "E11")
        sd = SD_E11 if grp == "E11" else SD_M16
        rv = np.exp(rng.normal(0, 0.3, n))
        d = pd.DataFrame({"ticker": t, "date": dates, "t_plus": dates + pd.offsets.BDay(1), "group": grp, "gap": False, "rv_tplus": rv})
        for s in S.model_cols():
            d[s] = rv * np.exp(sd.get(s, 0.6) * rng.normal(size=n))
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


@pytest.fixture
def env(monkeypatch, tmp_path):
    out, rep = tmp_path / "out", tmp_path / "rep"
    out.mkdir()
    rep.mkdir()
    monkeypatch.setattr(P, "OUT", out)
    monkeypatch.setattr(P, "REP", rep)
    monkeypatch.setattr(SEL, "D17", rep / "statistics_p17")
    monkeypatch.setattr(SEL, "HASHES17", {})
    k = make_k()
    base = k[SEL.KEY + ["t_plus", "group", "gap", "rv_tplus"]]
    k[SEL.KEY + ["t_plus", "group", "gap", "rv_tplus", "k1", "k4", "k5"]].rename(columns={"k1": "sigma2_k1", "k4": "sigma2_k4", "k5": "sigma2_k5"}).to_parquet(out / "k45_p17.parquet")
    k[SEL.KEY + ["k6", "k7"]].rename(columns={"k6": "sigma2_k6", "k7": "sigma2_k7"}).to_parquet(out / "k4567.parquet")
    for m in ("xgb", "lgbm"):
        h = base[SEL.KEY].copy()
        for n in S.NAMES:
            h["sigma2_" + n] = k["%s_%s" % (m, n)]
        h.to_parquet(out / ("hyb_%s_p17.parquet" % m))
    (rep / "CHECKPOINT.json").write_text(json.dumps({"N_declared_before_stats": 608, "loto_declared": ["xgb_A1"], "used": 20}))
    return rep


def test_vybor_po_M16_a_ne_po_E11_i_ob_yavlenie_N(env, monkeypatch):
    sel = SEL.run(lambda m: None)
    assert sel["finalists"]["xgb"] == "xgb_A1" and sel["positions"]["A"]["best"] in ("xgb_A1", "lgbm_A1") and sel["u16"]
    cp = json.loads((env / "CHECKPOINT.json").read_text())
    n_loto = len(S.loto_list(sel))
    assert cp["N_declared_before_stats_p17"] == 608 + 20 + n_loto == json.loads((env / "statistics_p17" / "SELECTION.json").read_text())["N_declared"]
    assert cp["N_declared_before_stats"] == 608 and cp["loto_declared"] == ["xgb_A1"]          # объявление на 28 не тронуто
    sc = pd.read_csv(env / "statistics_p17" / "SELECTION.csv", comment="#")
    assert len(sc) == 25 and (sc["n_tickers"] == 4).all()                                         # M16: четыре бумаги; SBER и E11 в критерий не входят
    monkeypatch.setattr(S, "crit_window", lambda k: k[(k["t_plus"].dt.year >= 2017) & (k["t_plus"].dt.year <= 2021)])      # мутант: E11 и SBER в критерии
    bad = S.select(S.crit_panel(SEL.load17(), S.model_cols()))
    assert bad["finalists"]["xgb"] == "xgb_A2"                                                    # E11 в критерии меняет финалиста — тест поймал бы


def test_N_ob_yavlennoe_ne_chislo_strok_i_povtor_otkaz(env):
    sel = SEL.run(lambda m: None)
    n = json.loads((env / "CHECKPOINT.json").read_text())["N_declared_before_stats_p17"]
    assert n == 608 + 20 + len(S.loto_list(sel)) and n > 628                                      # 628 — без LOTO; N не равно числу строк таблицы выбора (25)
    assert SEL.declare17(sel, len(S.loto_list(sel))) == n                                         # то же число — идемпотентно
    with pytest.raises(SystemExit):
        SEL.declare17(sel, len(S.loto_list(sel)) + 1)                                             # другое N после просмотра — отказ


def test_load17_k6_k7_iz_k4567_a_k45_p17_dlya_k1_k4_k5(env):
    k = SEL.load17()
    assert {"k1", "k4", "k5", "k6", "k7", "xgb_A1", "lgbm_D2"} <= set(k.columns) and len(k) == 7 * 1500
    assert k["k6"].notna().all() and set(k["group"]) == {"M16", "SBER", "E11"}
