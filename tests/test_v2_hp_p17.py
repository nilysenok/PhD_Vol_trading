"""P17, G16 (Ж-220, У-19…У-21): правила панели на 17 — поперечные признаки (боевой путь бары -> hp_daily -> hp_har -> hp_p17.features_all), выбор параметров года по 17 (`hp_p17.finalize`),
λ = 0 ≡ K1 / K7 побитно на 28 строках, реестр `Ж-220-p17:*` (used не меняется). Мутации — подмена крючков hp_p17 / правил hp_hyb: поперечные по 28, выбор по 28, λ ≠ 0 при λ = 0."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from models_v2.s10 import hp_daily as HD, hp_engine as E, hp_har as HH, hp_hyb_core as C, hp_hyb_out as O, hp_p17 as X, hp_p17_hyb as H17, hp_paths as P, hp_sync as SY, hp_synth as SS

T = ("A", "B", "C", "D", "E", "F", "G", "H")
T17, T11 = T[:6], T[6:]                                  # «17» — шесть бумаг (поперечные величины требуют >= 5), «E11» — G, H


def test_sostav_17_ravno_fastbt_i_trenirovka_isklyuchaet_E11():
    from fastbt.data import TICKERS
    assert sorted(X.TRAIN) == sorted(TICKERS) and len(X.TRAIN) == 17 and "SBER" in X.TRAIN
    assert X.train_exclude(P.TICKERS) == sorted(P.E11) and len(P.E11) == 11
    assert X.cross_members(P.TICKERS) == [t for t in P.TICKERS if t in P.MAIN17] and X.sel_mask(np.array(["SBER", "GCHE"])).tolist() == [True, False]


# --- поперечные признаки ---
@pytest.fixture(scope="module")
def world():
    bars, imo, cal = SS.world(900, T)
    return bars, imo, cal


def prep(bars: dict, imoex: pd.DataFrame) -> tuple:
    daily = {t: HD.ticker_daily(b, imoex) for t, b in bars.items()}
    return daily, HD.index_daily(imoex), {t: HH.series(d) for t, d in daily.items()}


@pytest.fixture(scope="module")
def feats(world):
    bars, imo, _ = world
    daily, imd, k1 = prep(bars, imo)
    return daily, imd, k1


def test_poperechnye_po_17_i_E11_ne_vliyaet(feats, monkeypatch):
    monkeypatch.setattr(X, "TRAIN", list(T17))
    daily, imd, k1 = feats
    full = X.features_all(daily, imd, k1)
    sub = X.features_all({t: daily[t] for t in T17}, imd, {t: k1[t] for t in T17})
    for t in T17:
        pd.testing.assert_frame_equal(full[t], sub[t], check_exact=True)                 # удаление E11 не меняет признаки 17 (в т. ч. ранг и медиану дня)
    t = full["A"].dropna(subset=["m_med"]).index[300]
    want = float(np.median([np.log(daily[s].loc[t, "rv"] / daily[s].loc[t, "rv_m"]) for s in T17]))
    assert full["A"].loc[t, "m_med"] == pytest.approx(want, rel=1e-12) and full["G"].loc[t, "m_med"] == full["A"].loc[t, "m_med"]   # медиана дня — по 17, у E11 та же
    assert full["G"]["c_rank"].notna().sum() > 500 and ((full["G"]["c_rank"].dropna() > 0) & (full["G"]["c_rank"].dropna() <= 1)).all()


def test_mutaciya_E11_v_poperechnyh_priznakah_lovitsya(feats, monkeypatch):
    monkeypatch.setattr(X, "TRAIN", list(T17))
    monkeypatch.setattr(X, "cross_members", lambda tickers: list(tickers))               # ошибка: поперечные по всем бумагам
    daily, imd, k1 = feats
    full = X.features_all(daily, imd, k1)
    sub = X.features_all({t: daily[t] for t in T17}, imd, {t: k1[t] for t in T17})
    assert any(not full[t].equals(sub[t]) for t in T17)                                  # E11 влияет на признаки 17 — тест инвариантности падает


def test_rank_vne_17_znanyj_otvet():
    ix = pd.bdate_range("2020-01-01", periods=2)
    L = pd.DataFrame({c: [1.0, 1.0] for c in "abcde"}, index=ix)
    L["e"] = [5.0, 5.0]
    L["a"] = [0.0, np.nan]                                                                # 2-й день: 4 бумаги (< MIN_CROSS = 5) — NaN
    med = L.median(axis=1)
    x = pd.Series([2.0, 2.0], index=ix)
    r, dev = X.rank_outside(L, med, x)
    assert r.iloc[0] == pytest.approx((4 + 1) / (5 + 1)) and dev.iloc[0] == pytest.approx(2.0 - 1.0)   # меньше 2.0: четыре из пяти
    assert np.isnan(r.iloc[1]) and np.isnan(dev.iloc[1]) and SY.MIN_CROSS == 5
    r2, _ = X.rank_outside(L, med, pd.Series([np.nan, 2.0], index=ix))
    assert np.isnan(r2.iloc[0])


# --- выбор года по 17 ---
def _runs_and_panel():
    """База 1, RV = e^0.5: глубина 2 идеальна на 17 (g = 0.5) и плоха на E11 (g = −2); глубина 4 — чуть хуже на 17 (g = 0.3) и идеальна на E11 (g = 0.5)."""
    dates = pd.bdate_range("2015-01-01", periods=600)
    rows = [(t, d) for d in dates[:-1] for t in T]
    p = pd.DataFrame(rows, columns=["ticker", "date"])
    p["t_plus"] = p["date"].map(dict(zip(dates[:-1], dates[1:])))
    p["rv_tplus"] = np.exp(0.5)
    p["sigma2_k1"] = 1.0
    is17 = p["ticker"].isin(T17).to_numpy()
    mk = lambda g17, g11: pd.DataFrame({"g": np.where(is17, g17, g11), "s2g": 0.0, "n_train": 1.0, "n_trees": 1.0, "fit_sec": 0.0})
    return p, {2: mk(0.5, -2.0), 4: mk(0.3, 0.5)}


def test_vybor_goda_po_17_a_mutaciya_po_28_lovitsya(monkeypatch):
    monkeypatch.setattr(X, "TRAIN", list(T17))
    p, runs = _runs_and_panel()
    fc, ch = X.finalize(p, p["sigma2_k1"].to_numpy(), runs)
    sel = ch[ch["source"] == "closed 2015"]
    assert len(sel) >= 1 and (ch.loc[ch["source"] != "default", "depth"] == 2).all()    # по 17 побеждает глубина 2
    e11 = fc[fc["ticker"].isin(T11)]
    assert len(e11) > 0 and (e11["depth"] == 2).all()                                    # E11 получает выбор, сделанный по 17 (перенос)
    monkeypatch.setattr(X, "sel_mask", lambda tk: np.ones(len(tk), bool))               # ошибка: выбор по всем 28
    _, ch2 = X.finalize(p, p["sigma2_k1"].to_numpy(), runs)
    assert (ch2.loc[ch2["source"] != "default", "depth"] == 4).any()                    # мутант выбирает другую глубину — различим


def test_lambda_nol_ravno_k1_pobitno_na_vseh_strokah(monkeypatch):
    monkeypatch.setattr(X, "TRAIN", list(T17))
    p, runs = _runs_and_panel()
    k1 = np.exp(np.random.default_rng(1).normal(0, 0.3, len(p)))
    fc, _ = X.finalize(p, k1, runs, lams=(0.0,))
    m = p.reset_index().merge(fc[["ticker", "date", "sigma2"]], on=["ticker", "date"])
    assert len(m) > 1000 and set(m["ticker"]) == set(T) and np.array_equal(m["sigma2"].to_numpy(), k1[m["index"].to_numpy()])   # λ = 0 ≡ K1, включая E11


def test_lambda_nol_ravno_k7_dlya_C_pobitno_i_mutaciya(monkeypatch):
    monkeypatch.setattr(X, "TRAIN", list(T17))
    p, runs = _runs_and_panel()
    hm, hu = (np.exp(np.random.default_rng(s).normal(0, 0.3, len(p))) for s in (2, 3))
    out, _ = H17.finalize_c(p, hm, hu, runs, runs, lams=(0.0,))
    ok = np.isfinite(out)
    assert ok.sum() > 1000 and np.array_equal(out[ok], (hm + hu)[ok])                    # λ = 0 ≡ K7 (Ĥ^m + Ĥ^u) побитно
    real = C.apply_lam
    monkeypatch.setattr(C, "apply_lam", lambda base, g, c, lam: real(base, g, c, np.asarray(lam, float) + 1e-9))   # ошибка: λ = 1e-9 вместо 0
    bad, _ = H17.finalize_c(p, hm, hu, runs, runs, lams=(0.0,))
    assert not np.array_equal(bad[ok], (hm + hu)[ok])


# --- реестр ---
def test_reestr_p17_odna_stroka_used_ne_menyaetsya(monkeypatch, tmp_path):
    reg, cp = tmp_path / "ALL_TRIALS.csv", tmp_path / "CHECKPOINT.json"
    reg.write_text("key,program,source,direction,config,kind\nЖ-217:xgb:A1:aa,Ж-217,s,d,\"{}\",trial\n", encoding="utf-8")
    cp.write_text(json.dumps({"used": 20, "leaf": "P3"}))
    for name, val in (("REG", reg), ("CP", cp), ("LOCK", tmp_path / "lock")):
        monkeypatch.setattr(O, name, val)
    assert H17.register("xgb", "A1", "panel_global", {"n_rows": 5}) and not H17.register("xgb", "A1", "panel_global", {"n_rows": 6})
    rows = reg.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3 and rows[2].startswith("Ж-220-p17:xgb:A1:") and rows[1].startswith("Ж-217:xgb:A1:aa")
    assert json.loads(cp.read_text())["used"] == 20                                      # register не трогает used (У-21)
    monkeypatch.setattr(P, "REP", tmp_path)
    from models_v2.s10 import hp_p17_out as OUT
    assert OUT.write_status() == 2 and pd.read_csv(tmp_path / "ALL_TRIALS_STATUS.csv")["status"].tolist() == [OUT.S28, OUT.S17]
