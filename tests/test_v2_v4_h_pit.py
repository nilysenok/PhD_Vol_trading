"""V4-H (G39): PIT боевого пути H (признаки из Z + проекция + оценка + прогноз): обрезка, мутации «закрыто + 1», «признак дня d+1», «ортогонализация по всей выборке», тождество θ ≡ 0 -> K8, исключённая бумага в LOTO,
известный ответ, ключ кэша (Д5), чтение без записи, хэш прежних прогнозов. Данные проекта не читаются."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s11 import (v4_b_run as BR, v4_h_cache as HC, v4_h_feat as HF, v4_h_fit as HFIT, v4_h_run as R, v4_h_stage as ST, v4_rqc_cache as RC, v4_rqc_calib as CAL, v4_trials as TR,
                           v4_z_feat as ZF)
from tests.test_v2_v4_h_feat import TK, synth_z

DAY = "2017-09-01"          # первый торговый день месяца: к нему закрыты пары 2017-01…08


def synth(n=330, seed=0, gate=0.0):
    """Панель + Z + доля первого часа. gate — истинный наклон ln(RV/K8) по herf (известный ответ для H3)."""
    zl, fh = synth_z(n, seed)
    rng = np.random.default_rng(seed + 100)
    rows = []
    for t in TK:
        g = zl[zl["ticker"] == t].reset_index(drop=True)
        h, k2, k3 = (np.exp(rng.normal(-1.0, 0.2, len(g))) for _ in range(3))
        hz = (g["herf"] - g["herf"].mean()) / g["herf"].std()
        rows.append(pd.DataFrame({"ticker": t, "date": g["date"], "t_plus": g["t_plus"], "rv": CAL.k8(h, k2, k3) * np.exp(gate * hz + rng.normal(0, 0.25, len(g))), "h": h, "k2": k2, "k3": k3}))
    return pd.concat(rows, ignore_index=True), zl, fh


@pytest.fixture(scope="module")
def data():
    return synth()


def first_day(P, m):
    return P["date"][P["date"].dt.to_period("M") == pd.Period(m)].min()


@pytest.mark.parametrize("cfg", ["H1", "H3", "H7", "H8"])
def test_cut_future_data_does_not_change_forecast(data, cfg):
    P, zl, fh = data
    assert not R.pit_violation(cfg, P, zl, fh, DAY)


@pytest.mark.parametrize("cfg", ["H1", "H7"])
def test_mutation_closed_plus_one_is_caught_via_real_path(data, monkeypatch, cfg):
    P, zl, fh = data
    monkeypatch.setattr(BR, "closed_mask", lambda t_plus, d0: np.asarray(t_plus) <= np.datetime64(d0) + np.timedelta64(3, "D"))
    assert R.pit_violation(cfg, P, zl, fh, DAY)


def test_mutation_feature_of_day_d_plus_one_is_caught_via_real_path(data):
    P, zl, fh = data
    assert R.pit_violation("H7", P, zl, fh, DAY, lead=1)


def test_mutation_orthogonalization_on_whole_sample_is_caught(data, monkeypatch):
    """Проекция z на базу по ВСЕЙ выборке (включая будущие строки), а не по закрытому окну: подмена данных после d меняет прогноз d — обрезка боевого пути ловит."""
    P, zl, fh = data
    stash = {}
    orig_prep, orig_fit = R.prepare, HFIT.orthogonalize_fit

    def prep(*a, **k):
        r = orig_prep(*a, **k)
        stash["X"], stash["B"] = r[1], r[2]
        return r
    monkeypatch.setattr(R, "prepare", prep)
    monkeypatch.setattr(HFIT, "orthogonalize_fit", lambda Zr, B, lam=None: orig_fit(stash["X"][np.isfinite(stash["X"]).all(1)], stash["B"][np.isfinite(stash["X"]).all(1)]))
    assert R.pit_violation("H7", P, zl, fh, DAY)


def test_identity_theta_zero_gives_k8_bitwise_and_prior_before_60_pairs(data):
    P, zl, fh = data
    Zh = HF.build(zl, fh, TK)
    out = R.forecast("H7", P, Zh, zero_theta=True)[0]
    Ps = P.sort_values(["date", "ticker"]).reset_index(drop=True)
    assert np.array_equal(out["F"].to_numpy(), CAL.k8(Ps["h"], Ps["k2"], Ps["k3"]))
    real = R.forecast("H7", P, Zh)[0]
    first = real[real["date"] < first_day(P, "2017-02")]            # январь: закрытых пар < 60 -> g = 0 -> F = K8
    assert (first["g"] == 0.0).all() and (real["g"] != 0.0).any()


def test_loto_excluded_ticker_does_not_enter_estimation(data):
    """Исключённая бумага не входит в оценку: подмена её RV не меняет прогнозы остальных (побитно) при тех же признаках; без исключения — меняет (мутация)."""
    P, zl, fh = data
    tk = [t for t in TK if t != "NVTK"]
    Zh16 = HF.build(ZF.recross(zl, tk), fh, tk)
    out = R.forecast("H8", P, Zh16, exclude="NVTK")[0]
    assert ST.zero_ok("H8", P, Zh16, "NVTK", out)
    Q = P.copy()
    Q.loc[Q["ticker"] == "NVTK", "rv"] *= 5.0
    a, b = R.forecast("H8", P, Zh16)[0], R.forecast("H8", Q, Zh16)[0]
    f = lambda x: x[x["ticker"] != "NVTK"].sort_values(["date", "ticker"])["F"].to_numpy()
    assert not np.array_equal(f(a), f(b))


def test_known_answer_gate_on_herf_is_recovered_and_beats_k8():
    P, zl, fh = synth(n=600, seed=2, gate=0.3)
    Zh = HF.build(zl, fh, TK)
    out, co = R.forecast("H3", P, Zh)
    late = out[out["date"] >= first_day(out, "2018-06")]
    hz = late[["ticker", "date"]].merge(Zh[["ticker", "date", "herf"]], on=["ticker", "date"], how="left")["herf"].to_numpy()
    assert co["th_herf"].iloc[-1] > 0.1 and np.corrcoef(late["g"].to_numpy(), hz)[0, 1] > 0.9
    q = lambda f: np.mean(late["rv"] / f - np.log(late["rv"] / f) - 1)
    assert q(late["F"]) < q(CAL.k8(late["h"], P.set_index(["ticker", "date"]).loc[list(zip(late["ticker"], late["date"])), "k2"].to_numpy(), P.set_index(["ticker", "date"]).loc[list(zip(late["ticker"], late["date"])), "k3"].to_numpy()))
    assert co["converged"].all()


def test_key_changes_with_input_code_and_constants(tmp_path, monkeypatch):
    a, c = tmp_path / "a.txt", tmp_path / "c.py"
    a.write_text("1")
    c.write_text("x")
    k0 = HC.keys([a], [c])["key"]
    a.write_text("2")
    assert HC.keys([a], [c])["key"] != k0
    a.write_text("1")
    c.write_text("y")
    assert HC.keys([a], [c])["key"] != k0
    c.write_text("x")
    assert HC.keys([a], [c])["key"] == k0
    monkeypatch.setattr(HFIT, "CLIP", 0.4)
    assert HC.keys([a], [c])["key"] != k0


def test_cached_readonly_never_writes_and_fails_on_key_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(HC, "OUT", tmp_path)
    df = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(RuntimeError):
        HC.cached("H1", "k", lambda: {"": df}, write=False)
    assert list(tmp_path.iterdir()) == []
    HC.cached("H1", "k", lambda: {"": df}, write=True)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert HC.cached("H1", "k", lambda: (_ for _ in ()).throw(AssertionError("пересчёт")), write=False)[""].equals(df)
    with pytest.raises(RuntimeError):
        HC.cached("H1", "другой ключ", lambda: {"": df}, write=False)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_prior_sha_ignores_stage_h_files_and_sees_others(tmp_path, monkeypatch):
    for mod in (HC, RC):
        monkeypatch.setattr(mod, "OUT", tmp_path)
    (tmp_path / "C1.parquet").write_bytes(b"c1")
    s0 = HC.prior_sha()
    for n in ("H1.parquet", "H8_coef.parquet", "ZH_SBER.parquet", "H_loto_H8.parquet"):
        (tmp_path / n).write_bytes(b"x")
    assert HC.prior_sha() == s0 == RC.prior_sha()
    (tmp_path / "C1.parquet").write_bytes(b"c1-changed")
    assert HC.prior_sha() != s0


def test_trials_rows_replace_keep_others_and_no_duplicates(tmp_path):
    f = tmp_path / "ALL_TRIALS.csv"
    f.write_text(TR.HEADER + "Ж-222:RQC1:aa,p,s,d,cfg,trial\nЖ-222:E1:bb,p,s,d,cfg,trial\n")
    for _ in range(2):
        TR.replace_rows(["Ж-222:%s:" % c for c in ST.CFGS], ST.trials_rows("k" * 64), path=f)
    keys = [x.split(",")[0] for x in f.read_text().splitlines()[1:]]
    assert sorted(k.split(":")[1] for k in keys) == sorted(["RQC1", "E1"] + ST.CFGS)
