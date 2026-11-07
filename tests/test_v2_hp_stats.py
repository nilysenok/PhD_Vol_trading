"""P3, G11: выбор (R9, У-12), Холм (У-14), DM / бутстрап дат (R13) через боевой путь `hp_stats_sel` / `hp_stats_tst` на синтетике (сеть и данные проекта не читаются).
Мутации: финалист по QLIKE 2022+ вместо crit_panel M16 × 2017–2021; Холм с N = числу строк (№ 103); бутстрап без сохранения всех бумаг дня (ячейки независимо)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s5.d import stats_d as SD
from models_v2.s10 import hp_stats_cmp as CM, hp_stats_sel as S, hp_stats_tst as T

M16T = ["AFLT", "ALRS", "HYDR", "IRAO"]


def make_k(sd_early: dict, sd_late: dict, n: int = 2200, tickers=M16T, seed: int = 0, shock: float = 0.0) -> pd.DataFrame:
    """Строка (бумага, дата): σ² ряда s = RV·exp(sd·z): чем меньше sd, тем ниже QLIKE; sd зависит от периода (≤ 2021 / 2022+). shock — общая по дате добавка к z (панельная зависимость)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", periods=n)
    common = rng.normal(size=n) * shock
    parts = []
    for t in tickers:
        rv = np.exp(rng.normal(0, 0.3, n))
        d = pd.DataFrame({"ticker": t, "date": dates, "t_plus": dates + pd.offsets.BDay(1), "group": "M16", "gap": False, "rv_tplus": rv})
        late = (d["t_plus"].dt.year >= 2022).to_numpy()
        for s in S.model_cols():
            sd = np.where(late, sd_late.get(s, 0.6), sd_early.get(s, 0.6))
            d[s] = rv * np.exp(sd * (rng.normal(size=n) + common))
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def test_vybor_po_crit_panel_a_ne_po_2022(monkeypatch):
    early = {"k1": 0.5, "k4": 0.55, "k5": 0.55, "xgb_A1": 0.3, "xgb_A2": 0.45, "xgb_C1": 0.5, "xgb_C2": 0.5}
    late = {"k1": 0.5, "k4": 0.55, "k5": 0.55, "xgb_A1": 0.8, "xgb_A2": 0.2}
    k = make_k(early, late)
    k["xgb_C1"], k["xgb_C2"] = k["k1"], k["k1"]               # точные копии K1: разницы нет вообще
    sel = S.select(S.crit_panel(k, S.model_cols()))
    assert sel["finalists"]["xgb"] == "xgb_A1"               # лучший 2017–2021, хотя 2022+ он хуже A2
    assert sel["positions"]["A"]["best"] == "xgb_A1" and sel["triple"][0] == "A"
    assert not sel["positions"]["C"]["passes"]             # C1/C2 = уровню K1: порог R9 K1 − 0,0005 не пройден → позиция пуста
    assert sel["u16"] and "xgb_A1" in sel["u16_configs"]
    monkeypatch.setattr(S, "CRIT_YEARS", (2022, 2026))      # мутация: выбор по 2022+
    bad = S.select(S.crit_panel(k, S.model_cols()))
    assert bad["finalists"]["xgb"] == "xgb_A2" != sel["finalists"]["xgb"]


def test_porog_k1_minus_0005():
    """Гибрид лучше K1 меньше чем на 0,0005 в crit_panel — позиция пуста; чуть лучше K4/K5 — не «превосходит ML»."""
    sc = pd.DataFrame({"series": S.model_cols(), "crit_panel": 0.30})
    sc.loc[sc["series"] == "k1", "crit_panel"] = 0.2800
    sc.loc[sc["series"] == "xgb_A1", "crit_panel"] = 0.2800 - 0.0004        # лучше K1 на 0,0004 < 0,0005
    sc.loc[sc["series"] == "xgb_B1", "crit_panel"] = 0.2800 - 0.0006        # лучше на 0,0006
    sel = S.select(sc)
    assert sel["positions"]["A"]["passes"] is False and sel["positions"]["B"]["passes"] is True
    assert sel["triple"] == ["B"]
    assert sel["beats_k45"]["xgb_B1"] is True               # K4 = 0,30


def test_loto_spisok_i_N():
    sel = dict(triple_cfgs=["xgb_A1", "lgbm_B1", "xgb_C1", "lgbm_D1"])
    assert S.loto_list(sel) == ["xgb_A1", "lgbm_A1", "lgbm_B1", "xgb_C1", "lgbm_D1"][:5]


def _tests_env(monkeypatch, k):
    monkeypatch.setattr(T, "HYB", ["xgb_A1", "xgb_A2"])
    monkeypatch.setattr(T, "WINS", ("all", "confirm_2022_2026", "y2022"))
    monkeypatch.setattr(CM, "SIGS", [s for s in CM.SIGS if s in ("k1", "k4", "k5", "k6", "k7", "xgb_A1", "xgb_A2")])
    monkeypatch.setattr(CM, "with_k23", lambda x: x.assign(k2=x["k1"], k3=x["k1"]))
    monkeypatch.setattr(CM, "K23", ["AFLT", "ALRS"])
    return k.assign(k2=k["k1"], k3=k["k1"])


def test_holm_n_ob_javlennoe_ne_chislo_strok(monkeypatch):
    k = make_k({"xgb_A1": 0.4, "xgb_A2": 0.45, "k1": 0.5}, {"xgb_A1": 0.4, "xgb_A2": 0.45, "k1": 0.5})
    k = _tests_env(monkeypatch, k)
    loto = k[S.KEY + ["xgb_A1"]].rename(columns={"xgb_A1": "sigma2"}).assign(config="xgb_A1")
    big = T.run_tests(k, loto, 608, log=lambda m: None)
    m = big[big["kind"] == "main"]
    exp = SD.holm_n(m["dm_p"].to_numpy(), 608)
    assert len(m) > 0 and np.allclose(m["p_holm"], exp)
    small = T.holm_main(big.copy(), len(m))
    assert (small.loc[small["kind"] == "main", "p_holm"] <= m["p_holm"] + 1e-15).all() and (small.loc[small["kind"] == "main", "p_holm"] < m["p_holm"] - 1e-12).any()   # N = число строк даёт меньшие p (№ 103)
    with pytest.raises(ValueError):
        T.holm_main(big.copy(), len(m) - 1)


def test_mutaciya_holm_N_chislo_strok_lovitsya(monkeypatch):
    """Подмена: Холм без добивки до N объявленного (holm_n игнорирует N) — тест равенства p_holm = holm_n(p, 608) падает."""
    k = _tests_env(monkeypatch, make_k({"xgb_A1": 0.4, "xgb_A2": 0.45, "k1": 0.5}, {"xgb_A1": 0.4, "xgb_A2": 0.45, "k1": 0.5}))
    loto = k[S.KEY + ["xgb_A1"]].rename(columns={"xgb_A1": "sigma2"}).assign(config="xgb_A1")
    good = T.run_tests(k, loto, 608, log=lambda m: None)
    monkeypatch.setattr(SD, "holm_n", lambda p, N: T.dmod.holm(np.asarray(p, float)))
    bad = T.run_tests(k, loto, 608, log=lambda m: None)
    gm, bm = good[good["kind"] == "main"], bad[bad["kind"] == "main"]
    assert not np.allclose(gm["p_holm"], bm["p_holm"])


def test_dm_znak_i_nulevoj_effekt():
    rng = np.random.default_rng(1)
    dl = -0.05 + rng.normal(0, 0.1, 600)                    # гибрид лучше: Δ < 0
    z, p = T.dm_panel(dl)
    assert z > 3 and p < 0.01
    z0, p0 = T.dm_panel(rng.normal(0, 0.1, 600))
    assert p0 > 0.05
    assert T.dm_panel(-dl)[0] < -3                          # знак: Δ > 0 — модель хуже


def _ci_width(k):
    c = S.common(k, S.model_cols())
    r = T.rows_for(c, S.model_cols(), [("xgb_A1", "k1")], {"M16": np.ones(len(c), bool)}, {"all": np.ones(len(c), bool)}, "aux")[0]
    return r["boot_hi"] - r["boot_lo"]


def test_butstrap_sohranyaet_vsekh_bumag_dnya_i_mutaciya_lovitsya(monkeypatch):
    """Общий шок даты: Δ_d сильно связаны между бумагами дня — ДИ по датам широк; мутация (ячейки (бумага, дата) независимо, даты строк перемешаны) даёт заметно более узкий ДИ."""
    k = make_k({"xgb_A1": 0.45, "k1": 0.5}, {"xgb_A1": 0.45, "k1": 0.5}, n=700, shock=3.0)
    good = _ci_width(k)
    orig = T.date_tab

    def shuffled(c, Q, sm, wm):
        c2 = c.assign(date=np.random.default_rng(3).permutation(c["date"].to_numpy()))
        return orig(c2, Q, sm, wm)
    monkeypatch.setattr(T, "date_tab", shuffled)
    bad = _ci_width(k)
    assert good > 1.3 * bad
