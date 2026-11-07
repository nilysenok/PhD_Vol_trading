"""P2, G9/G10: архитектура B (hp_hyb_b) — боевые функции forecast_b / select_kappa на синтетической панели (сеть и данные проекта не читаются, ML нет: B зависит только от f_A1 и RV).
Правила: λ = 0 ≡ K1 побитно (нулевая поправка α, γ при κ → ∞); α по закрытым парам t⁺ <= d (№ 102, «+1» ловится); обрезка будущего панели после d не меняет прогноз на d; κ года Y — по закрытым Y−1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s10 import hp_engine as E, hp_hyb_b as B, hp_hyb_core as C


def make(seed: int = 0, n: int = 900, tickers=("A", "B", "C"), shift_2016: float = 0.0):
    """Строка (бумага, d): t⁺ — следующая дата; f_A1 = base, RV = base·exp(shift·[год 2016] + шум); z — признак ln RV_d/RV_m."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    parts, fs = [], []
    for t in tickers:
        base = np.exp(rng.normal(0, 0.2, n))
        z = rng.normal(size=n)
        rv = base * np.exp(shift_2016 * (dates.year == 2016) + 0.05 * rng.normal(size=n) + 0.2 * z)
        parts.append(pd.DataFrame({"ticker": t, "date": dates[:-1], "t_plus": dates[1:], "rv_tplus": rv[1:], "z_x_d_m": z[:-1]}))
        fs.append(base[:-1])
    p = pd.concat(parts, ignore_index=True)
    f = np.concatenate(fs)
    o = np.argsort(p["date"].to_numpy(), kind="stable")
    return p.iloc[o].reset_index(drop=True), f[o]


def corrupt(p: pd.DataFrame, d0, seed: int = 9) -> pd.DataFrame:
    """Будущее после d0: RV цели всех пар с t⁺ > d0 и признак строк после d0 — другие значения."""
    q, rng = p.copy(), np.random.default_rng(seed)
    q.loc[q["t_plus"] > d0, "rv_tplus"] = 50 * np.exp(rng.normal(size=int((q["t_plus"] > d0).sum())))
    q.loc[q["date"] > d0, "z_x_d_m"] = rng.normal(size=int((q["date"] > d0).sum())) + 3
    return q


@pytest.mark.parametrize("kind", ["b1", "b2"])
def test_lambda_nol_ravno_baze_pobajtno_i_mutaciya_bez_szhatiya_lovitsya(kind, monkeypatch):
    p, f = make(shift_2016=0.3)
    out, _ = B.forecast_b(p, f, kind, kappas=(1e30,))
    ok = np.isfinite(out)
    assert ok.sum() > 1000 and np.array_equal(out[ok], f[ok])                      # κ → ∞: α = γ = 0, exp(0) = 1.0
    if kind == "b1":
        monkeypatch.setattr(B, "adj_b1", lambda ahat, n, kappa: ahat)               # мутация: α не сжимается
    else:
        orig = B.adj_b2
        monkeypatch.setattr(B, "adj_b2", lambda panel, f_a1, kappa, win=B.WIN: orig(panel, f_a1, 1e-3, win))
    bad, _ = B.forecast_b(p, f, kind, kappas=(1e30,))
    assert not np.array_equal(bad[ok], f[ok])


def test_alpha_znaet_otvet_na_postoyannom_smeshchenii():
    p, f = make(shift_2016=0.0)
    p["rv_tplus"] = f * np.exp(0.4)                                                  # ln(RV/f) ≡ 0.4 -> α̂ = 0.4 при n >= 1
    ahat, n = B.alpha_b1(p, f)
    late = pd.DatetimeIndex(p["date"]) > "2016-06-01"
    assert np.allclose(ahat[late], 0.4) and (n[late] == 252).all()
    assert np.allclose(B.adj_b1(ahat, n, 252.0)[late], 0.2)                           # n/(n + κ) = 1/2


@pytest.mark.parametrize("kind", ["b1", "b2"])
def test_obrezka_budushchego_i_mutaciya_zakrytye_plus_odin(kind, monkeypatch):
    p, f = make(shift_2016=0.3)
    d0 = pd.Timestamp("2017-06-30")
    base, _ = B.forecast_b(p, f, kind)
    cut, _ = B.forecast_b(corrupt(p, d0), f, kind)
    past = (p["date"] <= d0).to_numpy() & np.isfinite(base)
    assert past.sum() > 1000 and np.array_equal(base[past], cut[past])              # прогноз на d не видит пар с t⁺ > d и RV после d
    monkeypatch.setattr(C, "closed_counts", lambda tp, dates: np.searchsorted(tp, dates, side="right") + 1)    # мутант: пара с t⁺ = d+1 «закрыта»
    leak, _ = B.forecast_b(corrupt(p, d0), f, kind)
    assert not np.array_equal(leak[past], base[past])


def test_kappa_goda_po_zakrytym_proshlogo_goda_i_mutaciya_togo_zhe_goda(monkeypatch):
    p, f = make(shift_2016=0.5)                                                      # 2016: систематический сдвиг (мало сжатия лучше); 2017: сдвига нет (лучше сжатие)
    kap = lambda: B.forecast_b(p, f, "b1", years=[2017])[1]["kappa"].iloc[0]
    assert kap() == 50.0
    monkeypatch.setattr(E, "closed_mask", lambda date, tp, Y: pd.DatetimeIndex(date).year == Y)     # мутант: выбор по самому году Y
    assert kap() == 1000.0
    p2 = corrupt(p, pd.Timestamp("2016-12-31"))
    monkeypatch.undo()
    assert B.forecast_b(p2, f, "b1", years=[2017])[1]["kappa"].iloc[0] == 50.0        # данные 2017+ выбор на 2017 не меняют


def test_pervyj_god_bez_zakrytoj_validacii_maksimum_szhatiya():
    p, f = make()
    _, ch = B.forecast_b(p, f, "b2", years=[2015])
    assert ch["source"].iloc[0] == "default" and ch["kappa"].iloc[0] == 1000.0
