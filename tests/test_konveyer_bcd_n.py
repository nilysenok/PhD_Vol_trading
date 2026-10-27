"""Конвейер B/C/D: условие 3 у B (Ж-009), N проб (DSR, Холм), граница периода. Синтетика, без данных и сети."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_konveyer_bcd import bases, eng  # noqa: E402
from test_s9_shag_c import _mutated  # noqa: E402

from fastbt import konveyer_bcd as K  # noqa: E402
from fastbt import konveyer_bcd_n as N  # noqa: E402
from fastbt import shag_c as C  # noqa: E402
from fastbt import sweep_drivers as SD  # noqa: E402

QUIET = dict(log=lambda *_: None)


def chk_b_cond3(monkeypatch):
    """usl1 и usl2 выполнены, usl3 (к2) нет → B не проходит С8 v2 (Ж-009: для B условие 3 применяется)."""
    ns = eng()
    fake = dict(usl1_c8=True, usl2_k3=True, usl3_k2=False, proshla_c8v2=False)
    monkeypatch.setattr(C, "evaluate", lambda b, g, reps, **kw: dict(fake))
    out = K.eval_point(bases(ns), np.ones(len(ns.A_row)), "B", 10)
    assert not any(r["proshla_c8v2"] for r, _ in out.values())


def test_b_condition3_applied(monkeypatch):
    chk_b_cond3(monkeypatch)


def test_b_without_k2_mutation_caught(monkeypatch):
    new = 'r.update(msr=me.msr(ex), sortino=me.sortino_e(ex), **sr_stats(ex))\n        r["proshla_c8v2"] = bool(r["usl1_c8"] and r["usl2_k3"])'
    old = "r.update(msr=me.msr(ex), sortino=me.sortino_e(ex), **sr_stats(ex))"
    monkeypatch.setattr(K, "eval_point", _mutated(K.eval_point, old, new, K.__dict__))
    with pytest.raises(AssertionError):
        chk_b_cond3(monkeypatch)


def test_n_declared_c_with_const():
    y = yaml.safe_load((K.SWEEP / "C_setka_e3.yaml").read_text(encoding="utf-8"))
    full = C.load_grid()
    n = sum(len(C.plan(s, full)) for s in ("naive", "harj", "hyb"))
    assert y["tochek_razlichnyh"] == y["tochek_vsego"] == n + len(C.plan("naive", full)) == 360


def test_n_includes_screened_points_dsr_holm():
    ns = eng()
    ns.calib_closed = lambda G: 0.9 if G.mean() > 0.97 else 0.3        # часть масок выпадает по экрану
    df, _ = K.run_approach(ns, "D", {}, bases(ns), 100, **QUIET)
    assert (~df["screened_in"]).any() and df["screened_in"].any()
    d = N.add_n(df)
    for c, sub in d.groupby("comm_pct"):
        assert (sub["n_prob"] == len(sub)).all() and len(sub) == len(df) // 2     # все пробы, вкл. выпавшие
        trials = sub["sr_d"].tolist()
        r = sub[sub["screened_in"]].iloc[0]
        assert r["dsr"] == pytest.approx(SD.dsr(r["sr_d"], trials, int(r["T"]), r["skew"], r["kurt"])["dsr"])
        assert r["dsr"] < SD.psr(r["sr_d"], 0.0, int(r["T"]), r["skew"], r["kurt"]) + 1e-12   # DSR ≤ PSR(0): N > 1 штрафует
        assert (sub["holm_p_vs_A"].dropna() >= sub["p_vs_A"].reindex(sub["holm_p_vs_A"].dropna().index).fillna(1)).all()
        assert (sub.loc[~sub["screened_in"], "holm_p_vs_A"] == 1.0).all()           # невычисленные p = 1, но в m
    with pytest.raises(AssertionError):
        N.add_n(df, declared=len(df) // 2 - 1)                                       # объявленное N ≠ посчитанному


def test_n_mutation_only_evaluated_trials_changes_dsr():
    ns = eng()
    ns.calib_closed = lambda G: 0.9 if G.mean() > 0.97 else 0.3
    d = N.add_n(K.run_approach(ns, "D", {}, bases(ns), 100, **QUIET)[0])
    assert d["n_prob"].iloc[0] > d["screened_in"].sum() / 2        # мутация «N = только прошедшие экран» была бы меньше


