"""P2, G9/G10: архитектура D (hp_hyb_d) — Ḡ, λ_{i,t}, D1, D2 (логистика), бэггинг на синтетике (сеть и данные проекта не читаются).
Правила: Ḡ только по закрытым парам (мутант «с незакрытым днём» ловится); λ = 0 ≡ K1 побитно; бэггинг D2 — тем же модулем, что конфигурация (мутант «другой модуль» ловится);
обучение w — пары t⁺ < 01.01.Y; вырожденная логистика — явная ошибка с причиной."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s10 import hp_engine as E, hp_hyb_core as C, hp_hyb_d as D

HP = dict(lr=0.2, rounds=15, patience=5, min_leaf=10, l2=1.0, es_dates=20, min_train=100)


def one_ticker(n: int = 120):
    dates = pd.bdate_range("2016-01-04", periods=n + 1)
    return pd.DataFrame({"ticker": "A", "date": dates[:-1], "t_plus": dates[1:], "rv_tplus": 1.0})


def test_gbar_tolko_po_zakrytym_i_mutaciya_nezakrytyj_den(monkeypatch):
    p = one_ticker()
    gn = np.arange(len(p), dtype=float)                       # выигрыш пары r = r; пара r закрыта только к дате r+1
    gb = D.gbar(p, gn)
    r = 100
    assert gb[r] == np.mean(gn[r - 63:r]) and np.isnan(gb[0])      # окно r−63…r−1: пара самой строки (t⁺ = d+1) не входит
    monkeypatch.setattr(C, "closed_counts", lambda tp, dates: np.searchsorted(tp, dates, side="right") + 1)
    assert D.gbar(p, gn)[r] == np.mean(gn[r - 62:r + 1]) != gb[r]


def test_gain_znak_i_otsutstvie_g_dayot_nan():
    rv, k1 = np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0])
    f1 = np.array([2.0, 0.5, np.nan])
    g = D.gain(rv, k1, f1)
    assert g[0] > 0 and g[1] < 0 and np.isnan(g[2])            # f1 = RV — выигрыш; f1 дальше K1 — проигрыш
    p = one_ticker(3)
    gg = D.gain_a1(p, np.ones(3), np.array([np.nan, 0.5, 0.5]), np.array([1.0, 1.0, 1.0]))
    assert np.isnan(gg[0]) and np.isfinite(gg[1:]).all()       # нет g (модель не обучена) — NaN, а не нулевой выигрыш


def test_d1_lambda_nol_ravno_k1_pobajtno(monkeypatch):
    p = pd.concat([one_ticker(600).assign(ticker=t) for t in "AB"], ignore_index=True)
    rng = np.random.default_rng(2)
    k1, g, c = np.exp(rng.normal(size=len(p))), rng.normal(size=len(p)), np.full(len(p), 1.0)
    p["rv_tplus"] = np.exp(rng.normal(size=len(p)))
    gb = D.gbar(p, D.gain_a1(p, k1, g, c))
    monkeypatch.setattr(D, "A_GRID", (0.0,))
    monkeypatch.setattr(D, "B_GRID", (0.0,))
    s, ch = D.finalize_d1(p, k1, g, c, gb, years=[2017])
    m = (pd.DatetimeIndex(p["date"]).year == 2017)
    assert m.sum() > 300 and np.array_equal(s[m], k1[m])
    monkeypatch.setattr(D, "A_GRID", (0.5,))
    assert not np.array_equal(D.finalize_d1(p, k1, g, c, gb, years=[2017])[0][m], k1[m])


def test_d1_vybor_ab_po_zakrytym_proshlogo_goda():
    p = pd.concat([one_ticker(700).assign(ticker=t) for t in "AB"], ignore_index=True)
    rng = np.random.default_rng(4)
    k1, g, c = np.exp(rng.normal(size=len(p))), rng.normal(size=len(p)), np.full(len(p), 1.0)
    p["rv_tplus"] = k1 * np.exp(0.8 * np.clip(g, -1, 1) + 0.1 * rng.normal(size=len(p)))
    gb = D.gbar(p, D.gain_a1(p, k1, g, c))
    a = D.select_ab(p, k1, g, c, gb, 2017)
    q = p.copy()
    q.loc[q["t_plus"] >= "2017-01-01", "rv_tplus"] = 9.0
    assert D.select_ab(q, k1, g, c, gb, 2017) == a and a["source"] == "closed"
    assert D.select_ab(p, k1, g, c, gb, 2016)["source"] == "default"


def test_lam_d2_nol_daet_k1_pobajtno_i_logistika_uchitsya():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(2000, 5))
    y = (Z[:, 0] + 0.3 * rng.normal(size=2000) > 0).astype(int)
    w = D.fit_w(Z, y)
    lam = D.lam_d2(Z, w)
    assert 0 < lam.min() and lam.max() < 1 and np.corrcoef(lam, Z[:, 0])[0, 1] > 0.8            # λ растёт с Ḡ
    k1, g = np.exp(rng.normal(size=2000)), rng.normal(size=2000)
    with np.errstate(over="ignore"):
        z0 = D.lam_d2(Z, {**w, "intercept": -1e4, "coef": np.zeros(5)})
    assert (z0 == 0.0).all() and np.array_equal(C.apply_lam(k1, g, np.full(2000, 1.0), z0), k1)


def test_vyrozhdennaya_logistika_yavnaya_oshibka():
    Z = np.random.default_rng(0).normal(size=(800, 5))
    with pytest.raises(ValueError, match="вырожденное"):
        D.fit_w(Z, np.ones(800, int))
    with pytest.raises(ValueError, match="постоянный признак"):
        D.fit_w(np.column_stack([Z[:, :4], np.ones(800)]), (Z[:, 0] > 0).astype(int))


def test_d2_obuchenie_tolko_na_parah_zakrytyh_k_nachalu_goda_i_mutaciya(monkeypatch):
    p = pd.concat([one_ticker(800).assign(ticker=t) for t in "ABC"], ignore_index=True)
    rng = np.random.default_rng(7)
    n = len(p)
    Z, gn = rng.normal(size=(n, 5)), rng.normal(size=n)
    gn += 1.5 * Z[:, 0]
    k1, g, c = np.exp(rng.normal(size=n)), rng.normal(size=n), np.full(n, 1.0)
    m = pd.DatetimeIndex(p["date"]).year == 2017
    s, _ = D.finalize_d2(p, k1, g, c, Z, gn, years=[2017])
    gn2 = gn.copy()
    gn2[(p["t_plus"] >= "2017-01-01").to_numpy()] = -gn[(p["t_plus"] >= "2017-01-01").to_numpy()] * 5       # цели года Y и позже
    s2, _ = D.finalize_d2(p, k1, g, c, Z, gn2, years=[2017])
    assert m.sum() > 200 and np.array_equal(s[m], s2[m])
    monkeypatch.setattr(D, "train_mask", lambda panel, Y: pd.DatetimeIndex(panel["t_plus"]) < pd.Timestamp(Y + 1, 1, 1))   # мутант: обучение захватывает год Y
    assert not np.array_equal(D.finalize_d2(p, k1, g, c, Z, gn2, years=[2017])[0][m], s[m])
    monkeypatch.undo()
    assert D.finalize_d2(p, k1, g, c, Z, gn, years=[2016])[1]["source"].iloc[0] == "default"        # нет закрытых пар — λ = 1 (R8)


@pytest.mark.parametrize("module", ["xgb", "lgbm"])
def test_begging_tem_zhe_modulem_i_mutaciya_drugoj_modul(module, monkeypatch, tmp_path):
    from models_v2.s10 import hp_paths as P
    monkeypatch.setattr(P, "OUT", tmp_path)
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=500)
    p = pd.concat([pd.DataFrame({"ticker": t, "date": dates[:-1], "t_plus": dates[1:], "rv_tplus": np.exp(rng.normal(size=499)), "f1": rng.normal(size=499)}) for t in "ABC"], ignore_index=True)
    p = p.sort_values(["date", "ticker"]).reset_index(drop=True)
    mods = []
    orig = E.fit_model
    monkeypatch.setattr(E, "fit_model", lambda m, *a: (mods.append(m), orig(m, *a))[1])
    D.bag_runs(module, p, ["f1"], np.ones(len(p)), "syn", [2], 2, None, HP)
    assert set(mods) == {module} and len(mods) > 20
    mods.clear()
    other = "lgbm" if module == "xgb" else "xgb"
    monkeypatch.setattr(D, "bag_module", lambda m: other)
    D.bag_runs(module, p, ["f1"], np.ones(len(p)), "syn", [2], 2, None, HP)
    assert set(mods) == {other}                                                        # мутант виден: модуль копий ≠ модуля конфигурации


def test_razbros_i_dolya_znaka():
    g = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    gb = [g + d for d in (0.1, -0.1, 0.2, -0.2)]
    disp = D.dispersion(g, gb)
    assert np.allclose(disp, np.std([0, 0.1, -0.1, 0.2, -0.2], ddof=1))
    assert np.isnan(D.dispersion(g, [np.where(np.arange(6) == 2, np.nan, g)] * 4)[2])      # нет g у копии — NaN
    p = one_ticker(30)
    sh = D.sign_share(p, np.where(np.arange(30) % 2 == 0, 1.0, -1.0))
    assert np.isnan(sh[:4]).all() and np.allclose(sh[10:], 0.5, atol=0.05)
    assert (D.sign_share(p, np.ones(30))[10:] == 1.0).all()
