"""G15: сравнения с K2/K3 на 28 бумагах в статистике P3 (`hp_stats_cmp.with_k23`, `hp_stats_tst.k23_rows`, `hp_stats_val.levels`) на синтетике (сеть и данные проекта не читаются).
Знание ответа: σ² ряда = RV·exp(sd·z), меньше sd — лучше QLIKE. Мутации: подключение K2/K3 только к четырём бумагам; уровень 2 без K2/K3; знак Δ к K2 перепутан."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s10 import hp_paths as P, hp_stats_cmp as CM, hp_stats_sel as S, hp_stats_tst as T, hp_stats_val as V

TICKS = {"AFLT": "M16", "ALRS": "M16", "HYDR": "M16", "IRAO": "M16", "SBER": "SBER", "GCHE": "E11", "MRKC": "E11"}
USED = ["k1", "k4", "k5", "xgb_A1"]


def make_k(sd: dict, n: int = 2400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", periods=n)
    parts = []
    for t, g in TICKS.items():
        rv = np.exp(rng.normal(0, 0.3, n))
        d = pd.DataFrame({"ticker": t, "date": dates, "t_plus": dates + pd.offsets.BDay(1), "group": g, "gap": False, "rv_tplus": rv})
        for s in USED + ["k2", "k3"]:
            d[s] = rv * np.exp(sd.get(s, 0.6) * rng.normal(size=n))
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(CM, "SIGS", USED)
    monkeypatch.setattr(T, "HYB", ["xgb_A1"])
    monkeypatch.setattr(T, "WINS", ("all", "p2017_2021", "y2022", "p2023_2025", "y2026", "confirm_2022_2026", "no_240222", "no_top5_adv"))
    monkeypatch.setattr(P, "OUT", tmp_path)
    return tmp_path


def _write_k23(k: pd.DataFrame, path) -> None:
    k[S.KEY + ["k2", "k3"]].rename(columns={"k2": "sigma2_k2", "k3": "sigma2_k3"}).to_parquet(path / "k23.parquet")


def test_with_k23_chitaet_vse_28_bumag_a_mutaciya_chetyre_lovitsya(env, monkeypatch):
    k = make_k({})
    _write_k23(k, env)
    base = k.drop(columns=["k2", "k3"])
    assert CM.K23 == P.TICKERS and len(CM.K23) == 28
    x = CM.with_k23(base)
    assert x["k2"].notna().all() and x["k3"].notna().all() and len(x) == len(base)
    monkeypatch.setattr(CM, "K23", ["SBER", "AFLT", "ALRS", "GCHE"])        # мутант: старый список четырёх бумаг
    y = CM.with_k23(base)[base["ticker"].isin(CM.K23).to_numpy()]
    assert set(y["ticker"]) == {"SBER", "AFLT", "ALRS", "GCHE"} != set(x["ticker"])


def _lvl(k: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    c = S.common(k, CM.SIGS)
    pt = CM.per_ticker(c, CM.SIGS, CM.slice_masks(c))
    k2 = S.common(k, CM.SIGS + ["k2", "k3"])
    pt2 = CM.per_ticker(k2, CM.SIGS + ["k2", "k3"], {CM.K23_PREF + n: m for n, m in CM.slice_masks(k2).items()})
    return V.levels(pd.concat([pt, pt2], ignore_index=True), tests, set())


def _run(k: pd.DataFrame) -> pd.DataFrame:
    _write_k23(k, P.OUT)
    loto = k[S.KEY + ["xgb_A1"]].rename(columns={"xgb_A1": "sigma2"}).assign(config="xgb_A1")
    return T.run_tests(k.drop(columns=["k2", "k3"]), loto, 608, log=lambda m: None)       # как в бою: k4567 + гибриды без K2/K3, они подключаются из k23.parquet


def test_k23_stroki_oblasti_okna_i_znak(env):
    k = make_k({"xgb_A1": 0.3, "k2": 0.6, "k3": 0.2})                 # гибрид лучше K2, хуже K3
    tests = _run(k)
    r = tests[tests["comparator"].isin(["k2", "k3"])]
    assert set(r["scope"]) == {"M16", "SBER", "E11", "all28"} and all(w.startswith("k23_") for w in r["window"])
    main = r[r["kind"] == "main"]
    assert set(zip(main["scope"], main["window"])) == {("M16", "k23_all"), ("M16", "k23_confirm_2022_2026")} and len(main) == 4
    d = lambda c, s: float(r[(r["comparator"] == c) & (r["scope"] == s) & (r["window"] == "k23_all")]["delta_mean"].iloc[0])
    assert d("k2", "M16") < -0.05 and d("k3", "M16") > 0.01 and d("k2", "E11") < -0.05 and d("k3", "SBER") > 0.01     # знак: Δ < 0 — гибрид лучше
    assert (main["p_holm"] >= main["dm_p"] - 1e-12).all() and main["p_holm"].notna().all()


def test_uroven2_trebuet_k2_k3_i_mutaciya_lovitsya(env, monkeypatch):
    good = {"xgb_A1": 0.3, "k1": 0.6, "k4": 0.6, "k5": 0.6}
    k = make_k({**good, "k2": 0.2, "k3": 0.6})                         # K2 лучше гибрида -> уровень 2 недостижим
    tests = _run(k)
    lv = _lvl(k, tests)
    assert lv["l1_ok"].all() and lv["l2_k4"].all() and lv["l2_k5"].all() and lv["l2_k3"].all() and not lv["l2_k2"].any() and (lv["level"] == 1).all()
    orig = V.l1
    monkeypatch.setattr(V, "l1", lambda *a, **kw: dict(ok=True, raw=True) if a[2] == "k2" else orig(*a, **kw))     # мутант: K2 не проверяется
    assert (_lvl(k, tests)["level"] >= 2).all()
    monkeypatch.setattr(V, "l1", orig)
    both = make_k({**good, "k2": 0.6, "k3": 0.6})                      # K2, K3 хуже гибрида -> уровень 2 выполнен
    lv2 = _lvl(both, _run(both))
    assert lv2["l2_k2"].all() and lv2["l2_k3"].all() and (lv2["level"] >= 2).all()


def test_obshchie_daty_k2_k3_propuski_ne_zapolnyayutsya(env):
    """Бумага без K2/K3 на части дат (NaN): сравнение — только на датах, где все ряды определены; число строк срезов не растёт."""
    k = make_k({"xgb_A1": 0.3})
    k.loc[(k["ticker"] == "IRAO") & (k["date"] < pd.Timestamp("2019-01-01")), ["k2", "k3"]] = np.nan
    c2 = S.common(k, CM.SIGS + ["k2", "k3"])
    ir = c2[c2["ticker"] == "IRAO"]
    assert ir["date"].min() >= pd.Timestamp("2019-01-01") and np.isfinite(ir[["k2", "k3"]]).all().all()
    assert len(S.common(k, CM.SIGS)) > len(c2)
