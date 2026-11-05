"""И3 (Ж-200/Ж-201): рыночные признаки на синтетике с известным ответом; бумага с днём d+1 (M6) ловится."""
import numpy as np
import pandas as pd
import pytest

from models_v2 import daily
from models_v2.s4 import mkt, synth

N = 40
DT = synth.days(N)
SEG = np.zeros(N, int)


def rets(seed, vol=0.001):
    return synth.random_returns(N, seed, vol)


def paper_of(R, gaps=None, drop_day=None):
    b = synth.from_returns(DT, R, gaps)
    if drop_day is not None:
        b = b[b["begin"].dt.normalize() != DT[drop_day]]
    return mkt.paper(b)


def sber_ln(p):
    return np.log(p["day"][["rv", "rv_w", "rv_m"]].reindex(DT).to_numpy(float))


def build(papers, imoex=None):
    return mkt.block(None, sber_ln(papers[0]), SEG, DT, papers, imoex)


def rv_pct(R):
    return ((100.0 * R) ** 2).sum(axis=1)


def test_proxy_rv_is_equal_weight_portfolio_rv():
    RA, RB = rets(1), rets(2)
    out, _ = build([paper_of(RA), paper_of(RB)])
    assert np.allclose(out["proxy_d"], np.log(rv_pct((RA + RB) / 2)), rtol=1e-9)
    w = np.log(pd.Series(rv_pct((RA + RB) / 2)).rolling(5).mean())
    assert np.allclose(out["proxy_w"].dropna(), w.dropna(), rtol=1e-9) and out["proxy_w"].isna().sum() == 4


def test_average_log_rv_and_relative_sber():
    RA, RB = rets(1), rets(2)
    out, _ = build([paper_of(RA), paper_of(RB)])
    assert np.allclose(out["avg_d"], (np.log(rv_pct(RA)) + np.log(rv_pct(RB))) / 2, rtol=1e-9)
    assert np.allclose(out["rel_d"], np.log(rv_pct(RA)) - out["proxy_d"], rtol=1e-9)


def test_median_gap_squared_over_rv_m():
    RA, RB = rets(1), rets(2)
    gA, gB = 0.01 * np.ones(N), 0.03 * np.ones(N)
    out, _ = build([paper_of(RA, gA), paper_of(RB, gB)])
    i = 30
    exp = np.median([(100 * g[i]) ** 2 / rv_pct(R)[i - 21:i + 1].mean() for g, R in ((gA, RA), (gB, RB))])
    assert out["gap_med"].iloc[i] == pytest.approx(exp, rel=1e-6)


def test_jump_share_half_on_jump_day_and_ht_formula():
    a = 0.001
    smooth = np.tile([a, -a], N * 51)[:N * 51].reshape(N, 51)
    jumpy = smooth.copy()
    jumpy[10, 25] = 0.03
    out, _ = build([paper_of(smooth), paper_of(jumpy)])
    assert out["jump_share"].iloc[10] == 0.5 and out["jump_share"].drop(DT[10]).max() == 0.0
    r = pd.Series(jumpy[10])
    day = pd.Series(np.zeros(51, int))
    z = mkt.jump_z(r, day)["z"].iloc[0]
    n = 51
    rv, bv = (r ** 2).sum(), np.pi / 2 * (r.abs() * r.abs().shift(1)).sum()
    a43 = (r.abs() ** (4 / 3)).to_numpy()
    tp = n * mkt.MU43 ** -3 * n / (n - 2) * sum(a43[k] * a43[k - 1] * a43[k - 2] for k in range(2, n))
    ref = (rv - bv) / rv / np.sqrt(mkt.HT_C / n * max(1.0, tp / bv ** 2))
    assert z == pytest.approx(ref, rel=1e-9) and z > mkt.Z99


def test_paper_without_day_is_skipped_in_mean():
    RA, RB = rets(1), rets(2)
    out, _ = build([paper_of(RA), paper_of(RB, drop_day=10)])
    assert out["avg_d"].iloc[10] == pytest.approx(np.log(rv_pct(RA)[10]), rel=1e-9)
    assert out["proxy_d"].iloc[10] == pytest.approx(np.log(rv_pct(RA)[10]), rel=1e-9)   # портфель из одной бумаги


def test_imoex_parameter_with_proxy_fallback_before_first_date():
    RA, RB, RI = rets(1), rets(2), rets(3)
    start = 8
    imoex = synth.from_returns(DT[start:], RI[start:])
    out, meta = build([paper_of(RA), paper_of(RB)], imoex)
    first = start + 21                          # RV_m IMOEX определён с 22-го дня
    assert np.allclose(out["imoex_d"].iloc[first:], np.log(rv_pct(RI[start:]))[first - start:], rtol=1e-9)
    assert out["imoex_d"].iloc[:start].isna().all() and out["imoex_m"].iloc[:first].isna().all()   # Ж-201: без прокси-фолбэка
    assert out["imoex_d"].iloc[start:].notna().all() and meta["n_nan_imoex_d"] == start
    assert meta["imoex_from"] == str(DT[start].date())
    assert np.allclose(out["relx_d"], sber_ln(paper_of(RA))[:, 0] - out["imoex_d"], equal_nan=True)
    assert out["proxy_d"].notna().all()                     # прокси не затронут


def invariant(builder, d=25) -> bool:
    """Строки <= d не меняются при порче баров после d (и баров >= 18:50 дня d)."""
    RA, RB = rets(1), rets(2)
    bA, bB = synth.from_returns(DT, RA), synth.from_returns(DT, RB)
    o1 = builder([mkt.paper(bA), mkt.paper(bB)])
    o2 = builder([mkt.paper(synth.damage_after(bA, DT[d])), mkt.paper(synth.damage_after(bB, DT[d]))])
    return o1.iloc[:d + 1].to_numpy().tobytes() == o2.iloc[:d + 1].to_numpy().tobytes()


def future_builder(papers):
    """M6: бумага с днём d+1 (признаки сдвинуты на −1 строку)."""
    for p in papers:
        p["day"] = p["day"].shift(-1)
    return build(papers)[0]


def test_main_no_future_and_mutation_m6_caught():
    assert invariant(lambda ps: build(ps)[0])
    assert not invariant(future_builder)
