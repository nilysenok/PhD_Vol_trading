"""V7 лист 2a, G27 (счёт 4–7) через РЕАЛЬНЫЙ путь `run_cfg` на синтетике. Состав входа маршрутизатора (z в A, Z в C, B без z), одинаковые эксперты и решётка у 5–7 (мутация «разная решётка у C»),
«закрыто + 1» и стандартизация Z по всей выборке, g ≡ 0 ⇒ F = H, F > 0, известный ответ среднего по seeds, подключение стадии."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from models_v2.s14 import v7_1m_feat as FE, v7_1m_forms as FM, v7_1m_stage as ST, v7_1m_syn as SY, v7_1m_wf as WF, v7_2_cfg as C2, v7_2_run as R2

SEEDS = (0, 1)
S = SY.make(330, 0)
Z_OF = lambda s: S["z"]


@pytest.fixture(scope="module")
def runs():
    return {c: C2.run_cfg(c, S["tab"], S["Zg"], Z_OF, SEEDS, keep=True) for c in C2.IDS}


def test_declared_experts_and_grid_are_one_for_5_6_7_and_params_match_forms():
    assert C2.declared_problems() == []
    c = FM.counts()
    assert R2.PARAMS == {"4": c["flat"], "5": c["router"]["A"] + c["experts"], "6": c["router"]["B"] + c["experts"], "7": c["router"]["C"] + c["experts"]}
    assert C2.GRIDS["4"] == ((0.1, 0.0), (1.0, 0.0)) and len(C2.GRIDS["5"]) == 4


def test_mutation_different_grid_at_c_or_different_experts_is_caught_on_real_path(monkeypatch):
    monkeypatch.setitem(C2.GRIDS, "7", ((0.1, 0.0), (1.0, 0.0)))
    assert C2.declared_problems()
    with pytest.raises(ValueError, match="решётка"):
        C2.run_cfg("7", S["tab"], S["Zg"], Z_OF, (0,))
    monkeypatch.setitem(C2.GRIDS, "7", C2.GRID_MOE)
    monkeypatch.setitem(C2.SPECS, "7", dict(kind="moe", router="C", K=3))
    with pytest.raises(ValueError, match="эксперты"):
        C2.run_cfg("7", S["tab"], S["Zg"], Z_OF, (0,))


def dependence(models_batch) -> tuple:
    """(π зависит от Z, π зависит от z) для обученной модели реального пути — возмущением входа."""
    models, bt = models_batch[0]
    m = models[0]
    base = m.pi(bt["Zg"], bt["z"])
    with torch.no_grad():
        return (not torch.equal(base, m.pi(bt["Zg"] + 1.0, bt["z"])), not torch.equal(base, m.pi(bt["Zg"], bt["z"] + 1.0)))


def test_router_inputs_on_real_path_z_in_A_Z_in_C_B_has_both(runs):
    got = {c: dependence([r for r in runs[c][1][0]["raw"] if r][-1:]) for c in "567"}
    assert got == {"5": (True, False), "6": (True, True), "7": (False, True)}


ORIG = FM.router_input


@pytest.mark.parametrize("name,mut", [("z в A", lambda k, Zg, z: torch.cat([z, z, z[:, :1]], 1) if k == "A" else ORIG(k, Zg, z)),
                                     ("Z в C", lambda k, Zg, z: Zg[:, :4] if k == "C" else ORIG(k, Zg, z)),
                                     ("B без z", lambda k, Zg, z: torch.cat([Zg, torch.zeros_like(z)], 1) if k == "B" else ORIG(k, Zg, z))])
def test_mutation_router_input_composition_caught(runs, monkeypatch, name, mut):
    monkeypatch.setattr(FM, "router_input", mut)
    got = {c: dependence([r for r in runs[c][1][0]["raw"] if r][-1:]) for c in "567"}
    assert got != {"5": (True, False), "6": (True, True), "7": (False, True)}, name


def test_output_columns_weights_in_simplex_and_experts_bounded(runs):
    for c in "567":
        df, ok = runs[c][0], runs[c][0]["fitted"] > 0
        assert list(df.columns) == ["d", "t_plus", "rv", "H", "G", "F", "fitted", "pi0", "pi1", "pi2", "g1", "g2"] and ok.sum() > 50
        assert np.allclose(df.loc[ok, list(C2.PI_COLS)].sum(axis=1), 1.0, atol=1e-12) and (df.loc[ok, list(C2.GK_COLS)].abs() <= 1.0).all().all()
        assert df.loc[~ok, list(C2.PI_COLS)].isna().all().all()
    assert list(runs["4"][0].columns) == ["d", "t_plus", "rv", "H", "G", "F", "fitted"]
    assert str(runs["6"][0]["d"].iloc[0].date()) == "2018-01-03"


def test_G_is_expectation_of_pi_times_experts_for_one_seed():
    df, per = C2.run_cfg("6", S["tab"], S["Zg"], Z_OF, (0,))
    ok = df["fitted"] > 0
    g = df.loc[ok, "pi1"] * df.loc[ok, "g1"] + df.loc[ok, "pi2"] * df.loc[ok, "g2"]
    assert np.allclose(g, df.loc[ok, "G"], atol=1e-12)                                                   # G = Σ π_k g_k (g₀ ≡ 0)


def test_assemble_known_answer_mean_over_seeds_and_F_equals_H_exp_G():
    tab = S["tab"].iloc[:3]
    mk = lambda g, p: dict(G=np.array([np.nan, g, g]), fitted=np.array([0.0, 1.0, 1.0]), pi=np.full((3, 3), p), gk=np.full((3, 2), g))
    df = C2.assemble("6", tab, [mk(0.2, 0.2), mk(0.4, 0.6)])
    assert len(df) == 2 and np.allclose(df["G"], 0.3) and np.allclose(df["pi0"], 0.4) and np.allclose(df["g1"], 0.3)
    assert np.array_equal(df["F"].to_numpy(), tab["H"].to_numpy()[1:] * np.exp(df["G"].to_numpy()))


def test_g_zero_gives_F_equals_H_bitwise_and_F_positive_on_real_path(monkeypatch, runs):
    monkeypatch.setattr(WF, "MIN_TRAIN", 10 ** 6)
    df, _ = C2.run_cfg("6", S["tab"], S["Zg"], Z_OF, (0,))
    assert (df["G"] == 0).all() and np.array_equal(df["F"].to_numpy(), df["H"].to_numpy()) and (df["fitted"] == 0).all()
    assert all((runs[c][0]["F"] > 0).all() for c in C2.IDS)


def leaks(monkeypatch, cfg="6", patch=None, per_run=None):
    if patch:
        patch(monkeypatch)
    a = SY.make(330, 0)
    J = int(np.flatnonzero(a["tab"].index >= pd.Timestamp("2018-02-01"))[0])
    b = SY.perturb_future(a, J)
    out = []
    for s in (a, b):
        if per_run:
            per_run(monkeypatch, s)
        out.append(C2.run_cfg(cfg, s["tab"], s["Zg"], lambda sd, s=s: s["z"], (0,))[0].set_index("d")["F"])
    idx = a["tab"].index
    same = out[0].loc[:idx[J]].equals(out[1].loc[:idx[J]])
    assert not out[0].loc[idx[J + 1]:].equals(out[1].loc[idx[J + 1]:])                                 # возмущение видно позже: тест не пустой
    return not same


@pytest.mark.parametrize("cfg", ["4", "6"])
def test_honest_real_path_has_no_future_dependence(monkeypatch, cfg):
    assert leaks(monkeypatch, cfg) is False


def test_mutation_closed_plus_one_caught(monkeypatch):
    def plus_one(tp, d0):
        return np.flatnonzero(tp <= np.sort(tp[tp > np.datetime64(d0)])[0])
    assert leaks(monkeypatch, "6", lambda mp: mp.setattr(WF, "closed_rows", plus_one)) is True


def test_mutation_Z_standardization_on_whole_sample_and_current_pair_caught(monkeypatch):
    def whole(mp, s):
        A = np.hstack([s["Zg"], s["z"]])
        A = A[np.isfinite(A).all(axis=1)]
        sd = A.std(axis=0)
        mp.setattr(WF, "norm_stats", lambda X: (A.mean(axis=0), np.where(sd > 1e-12, sd, 1.0)))
    assert leaks(monkeypatch, "6", per_run=whole) is True
    assert leaks(monkeypatch, "4", lambda mp: mp.setattr(FE, "last_closed", lambda tp, d: np.arange(len(tp)))) is True


def test_connect_does_not_replace_handler_and_check_branch_calls_2a(monkeypatch):
    rn = {"5": "кто-то"}
    R2.connect(rn)
    assert rn["5"] == "кто-то" and set(rn) == set(C2.IDS)
    called = []
    monkeypatch.setattr(ST.RG, "gate", lambda cfg, *a, **k: {})
    monkeypatch.setattr(R2, "check_run", lambda ids: called.append(list(ids)) or 7)
    assert ST.run_configs("2", None, True) == 7 and called == [["4", "5", "6", "7"]]
    monkeypatch.setattr(ST.RG, "gate", lambda cfg, *a, **k: (_ for _ in ()).throw(ST.RG.Refusal("x")) if cfg == "6" else {})
    assert ST.run_configs("2", None, True) == 3 and len(called) == 1
