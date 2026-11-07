"""P3-17, G17 (Ж-220 п. 9, У-22): статистика РАЗДЕЛЬНО для «17» (M16, SBER, 17) и «E11» (перенос) через боевой путь `hp_s17_stats` на синтетике (сеть и данные проекта не читаются).
Мутации: статистика E11 смешана с 17 (`population` отдаёт все строки); Холм с N = числу строк (№ 103); область all28 не переименована."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s5.d import stats_d as SD
from models_v2.s10 import hp_paths as P, hp_s17_sel as SEL, hp_s17_stats as ST, hp_stats_cmp as CM, hp_stats_sel as S, hp_stats_tst as T

TICKS = {"AFLT": "M16", "ALRS": "M16", "SBER": "SBER", "GCHE": "E11", "MRKC": "E11"}
USED = ["k1", "k4", "k5", "k6", "k7", "xgb_A1"]


def make_k(sd: dict, n: int = 1500, seed: int = 0, spike17: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", periods=n)
    parts = []
    for t, g in TICKS.items():
        rv = np.exp(rng.normal(0, 0.3, n))
        d = pd.DataFrame({"ticker": t, "date": dates, "t_plus": dates + pd.offsets.BDay(1), "group": g, "gap": False, "rv_tplus": rv})
        for s in USED:
            d[s] = rv * np.exp(sd.get(s, 0.6) * rng.normal(size=n))
        if spike17 and g != "E11":
            d.loc[d["date"] == dates[700], "k1"] = rv[700] * 1e-3                    # огромный QLIKE K1 в одну дату только у 17
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(CM, "SIGS", USED)
    monkeypatch.setattr(T, "HYB", ["xgb_A1"])
    monkeypatch.setattr(T, "WINS", ("all", "p2017_2021", "y2022", "p2023_2025", "y2026", "confirm_2022_2026", "no_240222", "no_top5_adv"))
    monkeypatch.setattr(P, "OUT", tmp_path)
    monkeypatch.setattr(P, "REP", tmp_path)
    monkeypatch.setattr(SEL, "D17", tmp_path / "statistics_p17")
    return tmp_path


def loto_of(k: pd.DataFrame) -> pd.DataFrame:
    m = k["group"].isin(["M16", "SBER"])
    return k[m][S.KEY + ["xgb_A1"]].rename(columns={"xgb_A1": "sigma2"}).assign(config="xgb_A1")


def test_razdelno_17_i_E11_bez_smeshivaniya(env):
    k = make_k({"xgb_A1": 0.3, "k1": 0.6})
    a17, aE = ST.compute_pop(k, "17", loto_of(k), 631, lambda m: None), ST.compute_pop(k, "E11", None, 631, lambda m: None)
    assert set(a17["pt"]["ticker"]) == {"AFLT", "ALRS", "SBER"} and set(aE["pt"]["ticker"]) == {"GCHE", "MRKC"}
    assert set(a17["c"]["group"]) == {"M16", "SBER"} and set(aE["c"]["group"]) == {"E11"}
    assert set(a17["tests"]["kind"]) >= {"main"} and "transfer" not in set(a17["tests"]["kind"]) and set(aE["tests"]["kind"]) >= {"transfer"} and "main" not in set(aE["tests"]["kind"])
    r = ST.relabel(a17["tests"], "17")
    assert "17" in set(r["scope"]) and not r["scope"].str.contains("all28").any() and set(r["population"]) == {"17"}
    e = ST.relabel(aE["tests"], "E11")
    assert set(e["scope"]) == {"E11"} and set(e["population"]) == {"E11"}


def test_E11_ne_zavisit_ot_dannyh_17_i_mutaciya_smeshivaniya_lovitsya(env, monkeypatch):
    """Подмена данных 17 (спайк K1 в одну дату) не меняет ничего в статистике E11; мутант (`population` отдаёт все строки) меняет срез «без 5 дат с наибольшим QLIKE K1»."""
    a, b = make_k({"xgb_A1": 0.3, "k1": 0.6}), make_k({"xgb_A1": 0.3, "k1": 0.6}, spike17=True)
    ra, rb = (ST.compute_pop(x, "E11", None, 631, lambda m: None) for x in (a, b))
    pick = lambda r: r["pt"][r["pt"]["period"] == "no_top5dates_k1"].sort_values(["ticker", "model"])["qlike"].to_numpy()
    assert len(pick(ra)) > 0 and np.array_equal(pick(ra), pick(rb))
    assert ra["tests"].drop(columns="p_holm").equals(rb["tests"].drop(columns="p_holm"))
    monkeypatch.setattr(ST, "population", lambda k, pop: k)                      # мутант: E11 считается на всех строках
    ma, mb = (ST.compute_pop(x, "E11", None, 631, lambda m: None) for x in (a, b))
    assert not np.array_equal(pick(ma), pick(mb))


def test_holm_n_ob_yavlennoe_dlya_main_i_transfer(env):
    k = make_k({"xgb_A1": 0.3, "k1": 0.6})
    for pop, kind in (("17", "main"), ("E11", "transfer")):
        t = ST.compute_pop(k, pop, loto_of(k) if pop == "17" else None, 631, lambda m: None)["tests"]
        m = t[t["kind"] == kind]
        assert len(m) > 0 and np.allclose(m["p_holm"], SD.holm_n(m["dm_p"].to_numpy(), 631))
        small = ST.holm(t.copy(), len(m))
        assert (small.loc[small["kind"] == kind, "p_holm"].to_numpy() < 0.9 * m["p_holm"].to_numpy()).any()          # N = число строк (№ 103) даёт меньшие p — тест равенства с N = 631 его ловит
        with pytest.raises(ValueError):
            ST.holm(t.copy(), len(m) - 1)


def test_sborka_tablic_oba_naboro_i_urovni(env):
    k = make_k({"xgb_A1": 0.3, "k1": 0.6, "k4": 0.6, "k5": 0.6})
    ST.save("17", ST.compute_pop(k, "17", loto_of(k), 631, lambda m: None))
    ST.save("E11", ST.compute_pop(k, "E11", None, 631, lambda m: None))
    ST.assemble(631, lambda m: None)
    d = env / "statistics_p17"
    for f in ("COMPARISON", "TESTS", "MCS", "PER_TICKER", "HARJ_VALUE", "LEVELS", "LOTO"):
        assert (d / (f + ".csv")).exists(), f
    pt = pd.read_csv(d / "PER_TICKER.csv", comment="#")
    assert set(pt[pt["population"] == "17"]["ticker"]).isdisjoint(set(pt[pt["population"] == "E11"]["ticker"]))
    lv = pd.read_csv(d / "LEVELS.csv", comment="#")
    assert len(lv) == 2 and lv["l4_e11"].all()                                    # гибрид лучше K1 на E11 и в LOTO достижим только при переносе и LOTO; здесь E11 — уровень 4 не проверяется без M16-ворот
    t = pd.read_csv(d / "TESTS.csv", comment="#")
    assert not t["scope"].astype(str).str.contains("all28").any()
