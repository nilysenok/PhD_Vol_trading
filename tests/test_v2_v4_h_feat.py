"""V4-H (G39): PIT-признаки новых источников (ZH): секторный сюрприз без своей бумаги, поперечный набор только из 17, обрезка (признаки строки d не зависят от данных после d), мутация «признак дня d+1», известные ответы.
Данные проекта не читаются; бумаги — настоящие тикеры (сектора из `hp_paths.SECTOR`)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s11 import v4_cld_brk as BK, v4_h_feat as HF, v4_z_feat as ZF

TK = ("LKOH", "NVTK", "TATN", "PHOR", "SBER")          # нефть ×3; PHOR — один в секторе; SBER — один в наборе


def synth_z(n=400, seed=0, tickers=TK):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", periods=n + 1)
    rows = []
    for t in tickers:
        ell = rng.normal(0, 0.2, n)
        rows.append(pd.DataFrame({"ticker": t, "date": dates[:-1], "t_plus": dates[1:], "ell": ell, "abs_ell": np.abs(ell), "lerr_h": rng.normal(-0.2, 0.5, n), "lrv": rng.normal(-1.0, 0.4, n),
                                  "jshare": rng.uniform(0, 0.4, n), "lasth": rng.uniform(0, 0.3, n), "herf": rng.uniform(0.03, 0.3, n)}))
    zl = ZF.recross(pd.concat(rows, ignore_index=True), list(tickers))
    fh = pd.concat([pd.DataFrame({"ticker": t, "date": dates[:-1], "firsth": rng.uniform(0.1, 0.4, n)}) for t in tickers], ignore_index=True)
    return zl, fh


def test_sector_surprise_excludes_own_ticker_and_falls_back_to_market():
    zl, fh = synth_z()
    out = HF.build(zl, fh, TK)
    L = zl.pivot(index="date", columns="ticker", values="lerr_h")
    o = out.set_index(["ticker", "date"])
    d = L.index[100]
    assert o.loc[("LKOH", d), "sur_sec"] == pytest.approx(np.median([L.loc[d, "NVTK"], L.loc[d, "TATN"]]), rel=1e-12)
    assert o.loc[("TATN", d), "sur_sec"] == pytest.approx(np.median([L.loc[d, "LKOH"], L.loc[d, "NVTK"]]), rel=1e-12)
    sm = zl.drop_duplicates("date").set_index("date")["sur_mkt"]
    assert o.loc[("PHOR", d), "sur_sec"] == sm[d]                       # один в секторе — рыночный s_d


def test_mutation_sector_median_with_own_ticker_is_caught():
    """Секторный сюрприз со своей бумагой отличается от боевого: подмена lerr_h своей бумаги меняет мутанта, но не боевой путь."""
    zl, fh = synth_z()
    base = HF.build(zl, fh, TK).set_index(["ticker", "date"])["sur_sec"]
    z2 = zl.copy()
    z2.loc[z2["ticker"] == "LKOH", "lerr_h"] += 5.0
    chg = HF.build(z2, fh, TK).set_index(["ticker", "date"])["sur_sec"]
    assert np.array_equal(base.xs("LKOH").to_numpy(), chg.xs("LKOH").to_numpy())        # свой ряд на свой сюрприз не влияет
    assert not np.array_equal(base.xs("NVTK").to_numpy(), chg.xs("NVTK").to_numpy())     # соседи — влияет
    own_med = zl[zl["ticker"].isin(["LKOH", "NVTK", "TATN"])].groupby("date")["lerr_h"].median()          # мутант: со своей бумагой
    assert not np.allclose(own_med.to_numpy(), base.xs("LKOH").to_numpy(), equal_nan=True)


def test_loto_excluded_ticker_not_in_sector_peers():
    """Набор 16 (без NVTK): соседи LKOH — только TATN; подмена lerr_h исключённой бумаги не меняет признаки остальных."""
    zl, fh = synth_z()
    tk = [t for t in TK if t != "NVTK"]
    a = HF.build(ZF.recross(zl, tk), fh, tk).set_index(["ticker", "date"])
    z2 = zl.copy()
    z2.loc[z2["ticker"] == "NVTK", ["lerr_h", "lrv"]] += 5.0
    b = HF.build(ZF.recross(z2, tk), fh, tk).set_index(["ticker", "date"])
    for t in tk:
        assert np.array_equal(a.xs(t)[["sur_sec", "sur5", "brk", "vol20"]].to_numpy(), b.xs(t)[["sur_sec", "sur5", "brk", "vol20"]].to_numpy(), equal_nan=True)
    d = zl["date"].iloc[100]
    assert a.loc[("LKOH", d), "sur_sec"] == zl[(zl["ticker"] == "TATN") & (zl["date"] == d)]["lerr_h"].iloc[0]


def test_cross_set_rejects_e11():
    zl, fh = synth_z(60)
    with pytest.raises(ValueError):
        HF.build(zl, fh, list(TK) + ["GCHE"])


def truncated_equal(lead: int, cut=200) -> bool:
    """Признаки строк до `cut` не меняются при подмене данных после них (боевой путь `build`; поперечные медианы пересчитываются по подменённым данным)."""
    zl, fh = synth_z()
    full = HF.build(zl, fh, TK, lead=lead)
    day = zl["date"].drop_duplicates().iloc[cut]
    z2 = zl.copy()
    late = z2["date"] > day
    z2.loc[late, ["lerr_h", "lrv", "jshare", "lasth", "herf", "ell", "abs_ell"]] += 1.7
    f2 = fh.copy()
    f2.loc[f2["date"] > day, "firsth"] += 0.2
    cut_ = HF.build(ZF.recross(z2.drop(columns=["med_lrv", "sur_mkt"]), list(TK)), f2, TK, lead=lead)
    a, b = (x[x["date"] <= day].sort_values(["ticker", "date"])[HF.ALL].to_numpy() for x in (full, cut_))
    return bool(np.array_equal(a, b, equal_nan=True))


def test_features_of_day_d_do_not_depend_on_later_data():
    assert truncated_equal(0)


def test_mutation_feature_of_day_d_plus_one_is_caught():
    assert not truncated_equal(1)


def test_jump_probability_uses_only_past_window_for_percentile():
    """Известный ответ: плоский ряд доли скачка — индикатор «выше p90 прошлого окна» равен 0 (строгое «выше», а не «не ниже»); до 60 прошлых наблюдений он не определён; первое значение выше прошлого — индикатор 1."""
    n = 120
    js = np.full(n, 0.1)
    js[100] = 0.11
    f = HF._own(pd.DataFrame({"jshare": js, "abs_ell": np.zeros(n), "ell": np.zeros(n)}))
    jp = f["jp_ewma"]
    assert jp.iloc[:60].isna().all() and (jp.iloc[60:100] == 0.0).all()
    assert jp.iloc[100] > 0.0 and jp.iloc[99] == 0.0


def test_flip5_and_ell_dev_known_answer():
    ell = np.array([0.1, 0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.1])
    g = pd.DataFrame({"jshare": np.full(10, 0.1), "abs_ell": np.abs(ell), "ell": ell})
    f = HF._own(g)
    fl = f["flip5"].to_numpy()
    assert np.isnan(fl[0]) and list(fl[1:]) == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]      # смена знака в строках 2 и 9; окно 5 строк (включая день d)
    assert np.allclose(f["ell_dev"].to_numpy(), 0.0)                     # |ℓ| постоянен -> отклонение от EWMA нулевое


def test_brk_and_vol20_are_computed_from_median_only_and_match_cusum():
    zl, _ = synth_z()
    dl = HF.date_level(zl)
    x = BK.median_series(zl)
    assert np.array_equal(dl["brk"].to_numpy(), BK.cusum(x)[0].to_numpy())
    assert dl["vol20"].iloc[:20].isna().all() and dl["vol20"].iloc[20] == pytest.approx(x.diff().iloc[1:21].std(), rel=1e-12)
