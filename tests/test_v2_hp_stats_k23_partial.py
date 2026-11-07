"""P3-17, починка (б): сравнения с K2/K3 — по бумагам, где есть `k23_<T>.parquet`, без собранного `k23.parquet` (`hp_stats_cmp.with_k23`, `k23_tickers`); число бумаг с K2/K3 — в каждой таблице.
Синтетика (сеть и данные проекта не читаются). Без правки тест падает: старый `with_k23` читал только `k23.parquet` (FileNotFoundError)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s10 import hp_paths as P, hp_stats_cmp as CM, hp_stats_sel as S, hp_stats_tst as T, hp_stats_val as V

TICKS = {"AFLT": "M16", "ALRS": "M16", "HYDR": "M16", "IRAO": "M16", "SBER": "SBER", "GCHE": "E11", "MRKC": "E11"}
WITH_K23 = ["AFLT", "ALRS", "SBER", "GCHE"]
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


def write_partial(k: pd.DataFrame, path, tickers=WITH_K23) -> None:
    for t in tickers:
        k[k["ticker"] == t][S.KEY + ["k2", "k3"]].rename(columns={"k2": "sigma2_k2", "k3": "sigma2_k3"}).to_parquet(path / ("k23_%s.parquet" % t))


def test_with_k23_po_bumagam_bez_sobrannogo_fajla(env):
    k = make_k({})
    write_partial(k, env)
    assert not (env / "k23.parquet").exists()
    base = k.drop(columns=["k2", "k3"])
    assert CM.k23_tickers() == sorted(WITH_K23)
    x = CM.with_k23(base)
    have = x[x["ticker"].isin(WITH_K23)]
    assert have["k2"].notna().all() and have["k3"].notna().all() and len(x) == len(base)
    assert x[~x["ticker"].isin(WITH_K23)][["k2", "k3"]].isna().all().all()            # у бумаг без файла K2/K3 не подставляются (ни ffill, ни ноль, Д4)
    pd.testing.assert_series_equal(have["k2"].reset_index(drop=True), k[k["ticker"].isin(WITH_K23)]["k2"].reset_index(drop=True), check_names=False)


def run_tests(k: pd.DataFrame) -> pd.DataFrame:
    write_partial(k, P.OUT)
    loto = k[S.KEY + ["xgb_A1"]].rename(columns={"xgb_A1": "sigma2"}).assign(config="xgb_A1")
    return T.run_tests(k.drop(columns=["k2", "k3"]), loto, 608, log=lambda m: None)


def test_tablicy_tolko_po_bumagam_s_k23_i_chislo_bumag_v_kazhdoj(env):
    k = make_k({"xgb_A1": 0.3, "k2": 0.6, "k3": 0.2})
    tests = run_tests(k)
    r = tests[tests["comparator"].isin(["k2", "k3"])]
    n = r.groupby("scope")["n_k23_tickers"].agg(lambda s: sorted(set(s)))
    assert n["M16"] == [2] and n["SBER"] == [1] and n["E11"] == [1] and n["all28"] == [4]          # AFLT, ALRS | SBER | GCHE | четыре бумаги
    pt, cmp_, c = CM.tables(k.drop(columns=["k2", "k3"]))
    assert set(pt[pt["period"].str.startswith("k23_")]["ticker"]) == set(WITH_K23)
    cm = cmp_[cmp_["scope"].str.startswith("K23:")]
    assert cm["n_k23_tickers"].notna().all() and cm[cm["scope"] == "K23:all28"]["n_k23_tickers"].eq(4).all()
    assert cmp_[~cmp_["scope"].str.startswith("K23:")]["n_k23_tickers"].isna().all()
    lv = V.levels(pt, tests, set())
    assert (lv["n_k23_tickers_m16"] == 2).all()


def test_net_k23_voobshche_tablicy_bez_k2_k3_ne_padayut(env):
    k = make_k({"xgb_A1": 0.3}).drop(columns=["k2", "k3"])
    assert CM.k23_tickers() == []
    pt, cmp_, c = CM.tables(k)
    assert not pt["period"].str.startswith("k23_").any() and not cmp_["scope"].str.startswith("K23:").any()


def test_mutaciya_trebovanie_k23_parquet_lovitsya(env, monkeypatch):
    """Мутант — старое поведение (читать только собранный `k23.parquet`): при частичном наборе файлов падает."""
    k = make_k({})
    write_partial(k, env)
    monkeypatch.setattr(CM, "k23_frame", lambda: pd.read_parquet(P.OUT / "k23.parquet"))
    with pytest.raises(FileNotFoundError):
        CM.with_k23(k.drop(columns=["k2", "k3"]))
