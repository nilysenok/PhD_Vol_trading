"""Л6: оценка семейства H1…H7 на синтетике (никаких новых данных и реального замка).
Известный ответ (независимая NW-реализация), всегда 7 строк, Холм по 7 (№ 103), «подтверждено» только при p Холма < 0,05 (№ 158), знак Δ (№ 125/140)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from models_v2 import metrics
from models_v2.s17 import fwd_eval as E

AE, LAST = E.ARCHIVE_END, E.LAST_SESSION


def make_frame(n=180, seed=3, noise=None, tp0="2026-03-02") -> pd.DataFrame:
    """Синтетика: σ̂² = rv·exp(−u), u ~ N(0, σ_ряд); у X1 и ZB шум меньше, чем у контролей."""
    rng = np.random.default_rng(seed)
    noise = noise or dict(K1=0.9, K2=0.6, K3=0.6, X1=0.4, ZB=0.35)
    tp = pd.bdate_range(tp0, periods=n)
    rv = np.exp(rng.normal(0, 0.5, n))
    fr = dict(d=tp.shift(-1, freq="B"), t_plus=tp, rv=rv)
    ser = {k: pd.Series(rv * np.exp(rng.normal(0, s, n)), index=fr["d"]) for k, s in noise.items()}
    base = pd.DataFrame(fr)
    extra = dict(e=pd.Series((rng.random(n) < 0.2).astype(float), index=fr["d"]), ZB0=ser["ZB"] * 1.01, M=ser["X1"] * 1.02, cfg8=ser["X1"] * 1.03)
    return E.common_frame(base, ser, extra, start=AE, end=tp[-1])


def nw_p(d: np.ndarray, lag: int = 5) -> tuple:
    """Независимая реализация: Ньюи–Уэст (Бартлетт) и односторонний p «плюс = A лучше»."""
    n, u = len(d), d - d.mean()
    v = float(u @ u) / n
    for l in range(1, lag + 1):
        v += 2 * (1 - l / (lag + 1)) * float(u[l:] @ u[:-l]) / n
    z = d.mean() / math.sqrt(v / n)
    return z, 0.5 * math.erfc(z / math.sqrt(2))


def expected(fr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, a, b in E.HYP:
        qa, qb = metrics.qlike(fr["rv"], fr[a]), metrics.qlike(fr["rv"], fr[b])
        z, p = nw_p(qb - qa)
        rows.append(dict(hyp=h, delta=100 * (qa.mean() - qb.mean()) / qb.mean(), stat=z, p=p))
    return pd.DataFrame(rows)


def matches(fam: pd.DataFrame, fr: pd.DataFrame) -> bool:
    ex = expected(fr)
    return bool(np.allclose(fam["delta_pct"], ex["delta"], atol=1e-9) and np.allclose(fam["dm_stat"], ex["stat"], atol=1e-9) and np.allclose(fam["p_one"], ex["p"], atol=1e-12))


def test_izvestnyj_otvet_delta_dm_i_poryadok():
    fr = make_frame()
    fam = E.family_table(fr)
    assert list(fam["hyp"]) == ["H1", "H2", "H3", "H4", "H5", "H6", "H7"] and list(fam.columns) == E.FAM_COLS
    assert (fam["n"] == len(fr)).all() and matches(fam, fr)
    assert fam.loc[fam["hyp"] == "H6", "delta_pct"].iloc[0] < 0  # X1 (меньше шума) лучше K1: Δ < 0 в % QLIKE базы
    assert (fam["lo90_pct"] <= fam["delta_pct"]).all() and (fam["delta_pct"] <= fam["hi90_pct"]).all()
    assert E.check_family(fam) == []


def test_pary_obshchie_i_okno_dat():
    fr = make_frame()
    assert fr["t_plus"].min() > AE and fr["t_plus"].max() <= LAST
    rv = pd.DataFrame(dict(d=pd.to_datetime(["2026-02-26", "2026-02-27", "2026-03-02", "2026-03-03", "2026-11-10", "2026-11-11"]),
                           t_plus=pd.to_datetime(["2026-02-27", "2026-03-02", "2026-03-03", "2026-03-04", "2026-11-11", "2026-11-12"]), rv=[1.0, 1, 1, np.nan, 1, 1]))
    s = pd.Series([1.0] * 6, index=rv["d"])
    fr2 = E.common_frame(rv, {k: s for k in E.SERIES}, start=AE, end=LAST)
    assert list(fr2["t_plus"].dt.strftime("%m-%d")) == ["03-02", "03-03"]  # пара с t⁺ = 02-27, пара с NaN rv и пары за 11-10 — вне окна
    s_bad = s.copy()
    s_bad.iloc[2] = np.nan  # дыра в одном из пяти рядов — дата выпадает у всех
    assert len(E.common_frame(rv, {**{k: s for k in E.SERIES}, "ZB": s_bad}, start=AE, end=LAST)) == 1


@pytest.mark.parametrize("fr", [make_frame(n=1, seed=1), pd.DataFrame({c: pd.Series(dtype=float) for c in ["d", "t_plus", "rv", *E.SERIES, "e"]})], ids=["n1", "pusto"])
def test_vsegda_sem_strok_i_otchet_stroitsya(fr):
    from models_v2.s17 import fwd_doc
    fr = fr.copy()
    for c in ("d", "t_plus"):
        fr[c] = pd.to_datetime(fr[c])
    t = E.all_tables(fr)
    fam = t["family"]
    assert len(fam) == 7 and list(fam["hyp"]) == [h for h, _, _ in E.HYP]
    assert not fam["verdict"].str.startswith(E.CONF).any() and fam["verdict"].str.contains("n = ").all()
    text = fwd_doc.render({k: v for k, v in t.items() if k != "pairs"})
    assert text.count("\n- H") == 7 and fwd_doc.verify(text, t) == []


def test_holm_po_semi_izvestnyj_otvet_mutatsiya_N_ravno_stroki_lovitsya():
    p = np.array([0.001, 0.01, 0.02, 0.03, 0.04, 0.2, 0.5])
    assert np.allclose(E.holm7(p), [0.007, 0.06, 0.1, 0.12, 0.12, 0.4, 0.5])
    with pytest.raises(ValueError):
        E.holm7(p[:5])  # N = числу строк запрещён
    fr = make_frame()
    fam = E.family_table(fr)
    from models_v2.s3 import dm as dmod
    bad = fam.copy()
    p = fam["p_one"].to_numpy(float)
    bad["p_holm"] = np.concatenate([dmod.holm(p[:5]), p[5:]])  # мутант: поправка по пяти гипотезам (N = числу строк подтаблицы)
    assert E.check_family(fam) == [] and any("N" in m for m in E.check_family(bad))


def fake_stats(p_list):
    it = iter(p_list)

    def pair_stats(fr, a, b, idx=None):
        p = next(it)
        return dict(A=a, B=b, n=len(fr), **E._dates(fr), qlike_A=0.2, qlike_B=0.25, delta_pct=-20.0, lo90_pct=-30.0, hi90_pct=-10.0, dm_stat=1.6, p_one=p)
    return pair_stats


def test_podtverzhdeno_tolko_pri_p_holma_menshe_005(monkeypatch):
    fr = make_frame()
    monkeypatch.setattr(E, "pair_stats", fake_stats([0.04, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    fam = E.family_table(fr)  # p_one = 0,04 < 0,05, но p Холма = 7·0,04 = 0,28
    assert fam["p_holm"].iloc[0] == pytest.approx(0.28) and not fam["verdict"].str.startswith(E.CONF).any()
    monkeypatch.setattr(E, "pair_stats", fake_stats([0.004, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    fam2 = E.family_table(fr)  # 7·0,004 = 0,028 < 0,05 — только H1
    assert list(fam2["verdict"].str.startswith(E.CONF)) == [True] + [False] * 6
    mut = fam.copy()
    mut.loc[0, "verdict"] = E.CONF  # мутант: вывод по сырому p
    assert any("вывод" in m for m in E.check_family(mut))
    assert E.verdict(0.05, -1.0, 9).startswith(E.NOT_CONF) and E.verdict(0.049, -1.0, 9) == E.CONF and E.verdict(0.01, +1.0, 9).startswith(E.NOT_CONF)


def test_mutatsiya_znak_delta_i_pereputannye_ryady_lovyatsya(monkeypatch):
    fr = make_frame()
    real = E.pair_stats
    monkeypatch.setattr(E, "pair_stats", lambda f, a, b, idx=None: real(f, b, a, idx) | dict(A=a, B=b))  # A и B поменяны местами внутри счёта
    assert not matches(E.family_table(fr), fr)
    monkeypatch.setattr(E, "pair_stats", real)
    fam = E.family_table(fr)
    flip = fam.copy()
    flip["delta_pct"] = -flip["delta_pct"]
    assert any("Δ" in m for m in E.check_family(flip))
    assert not matches(flip, fr) and matches(fam, fr)


def test_blok_10_bootstrap_i_parametry_semejstva():
    assert (E.BLOCK, E.B, E.SEED, E.LAG, E.CI, E.ALPHA, E.M_FAMILY) == (10, 2000, 0, 5, 90, 0.05, 7)
    fr = make_frame()
    assert E.family_table(fr).equals(E.family_table(fr))  # seed 0: повтор тождественен


def test_opisanie_mcs_mesyatsy_sostoyaniya_zb_zb0():
    fr = make_frame(n=120)
    t = E.all_tables(fr)
    assert list(t["mcs"]["series"]) == list(E.SERIES) and t["mcs"]["in_mcs"].any() and t["mcs"]["p_mcs"].between(0, 1).all()
    assert t["months"]["n"].sum() == len(fr) and set(t["states"]["state"]) == {"E", "N"} and t["states"]["n"].sum() == len(fr)
    assert int(t["states"].loc[t["states"]["state"] == "E", "n"].iloc[0]) == int(fr["e"].sum())
    assert list(zip(t["descr"]["A"], t["descr"]["B"])) == [("ZB", "ZB0"), ("X1", "M"), ("X1", "cfg8")]
    for k in ("series", "months", "states", "mcs"):
        assert {"n", "tp_first", "tp_last"} <= set(t[k].columns) and (t[k]["n"] > 0).all()
    assert t["series"]["qlike"].between(0, 5).all()
    assert np.isfinite(t["descr"][["qlike_A", "qlike_B", "delta_pct", "p_one"]].to_numpy(float)).all()
