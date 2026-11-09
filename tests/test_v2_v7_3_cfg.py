"""V7 лист 3, G42: конфигурация 8 (MoE B + якорь на M) через РЕАЛЬНЫЙ путь `run_cfg` на синтетике.
Известные ответы: w = 0 и G = 0 ⇒ F = H побитно; якорь находит w > 0, когда M = истинный прогноз; штраф λ_g включает w·ℓ (мутация «только MoE-часть»); ℓ даты d + 1 вместо d (мутация) ловится инвариантом «будущее не меняет прошлое»;
решётка и эксперты как у 6, 78 параметров (мутации K = 3, без якоря, другая решётка). Файлы SBER не читаются."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from models_v2.s14 import v7_1m_feat as FE, v7_1m_fit as FT, v7_1m_forms as FM, v7_1m_loss as LS, v7_1m_syn as SY, v7_1m_wf as WF, v7_2_cfg as C2, v7_3_cfg as C8

SEEDS = (0, 1)
S = SY.make(330, 0)
Z_OF = lambda s: S["z"]


@pytest.fixture(scope="module")
def run():
    return C8.run_cfg(S["tab"], S["Zg"], Z_OF, SEEDS)


def test_declared_params_spec_and_grid_equal_to_config_6():
    assert C8.declared_problems() == [] and C8.PARAMS8 == FM.counts()["anchor_b"] == 78
    assert C8.GRID8 == C2.GRIDS["6"] and C8.SPEC8 == dict(C2.SPECS["6"], kind="anchor")                         # эксперты, K, маршрутизатор — как у 6; отличие только якорь


@pytest.mark.parametrize("spec,grid,word", [(dict(kind="anchor", router="B", K=3), C8.GRID8, "якорь"), (dict(kind="moe", router="B", K=2), C8.GRID8, "якорь"),
                                            (dict(kind="anchor", router="A", K=2), C8.GRID8, "якорь"), (C8.SPEC8, ((0.1, 0.0), (1.0, 0.0)), "решётка")])
def test_mutation_other_spec_or_grid_is_refused_on_real_path(monkeypatch, spec, grid, word):
    monkeypatch.setattr(C8, "SPEC8", spec)
    monkeypatch.setattr(C8, "GRID8", grid)
    assert C8.declared_problems()
    with pytest.raises(ValueError, match=word):
        C8.run_cfg(S["tab"], S["Zg"], Z_OF, (0,))


def test_output_columns_pit_ell_weights_in_simplex_and_first_date(run):
    df = run[0]
    ok = df["fitted"] > 0
    assert list(df.columns) == C8.COLS and ok.sum() > 50 and str(df["d"].iloc[0].date()) == "2018-01-03"
    tab = S["tab"].loc[pd.DatetimeIndex(df["d"])]
    assert np.array_equal(df["ell"].to_numpy(), np.log(np.sqrt(tab["K2"] * tab["K3"]) / tab["H"]).to_numpy())            # ℓ ДАТЫ d
    assert np.allclose(df.loc[ok, list(C2.PI_COLS)].sum(axis=1), 1.0, atol=1e-12) and np.isfinite(df.loc[ok, "w"]).all() and (df["F"] > 0).all()
    assert np.array_equal(df["F"].to_numpy(), df["H"].to_numpy() * np.exp(df["G"].to_numpy()))


def test_w_zero_and_G_zero_give_F_equals_H_bitwise_fresh_model_and_real_path(monkeypatch):
    H, ell = torch.exp(torch.randn(64, dtype=torch.float64)), 0.3 * torch.randn(64, dtype=torch.float64)
    m = FM.ResMoE("B", 2, anchor=True, seed=3)
    with FM.deterministic():
        G, _ = m(torch.randn(64, 9, dtype=torch.float64), torch.randn(64, 4, dtype=torch.float64), ell)
    assert float(m.w) == 0.0 and torch.equal(G, torch.zeros_like(G)) and torch.equal(FM.forecast(H, G), H)
    monkeypatch.setattr(WF, "MIN_TRAIN", 10 ** 6)                                                              # ни одной переоценки: G = 0 всюду
    df, _ = C8.run_cfg(S["tab"], S["Zg"], Z_OF, (0,))
    assert (df["G"] == 0).all() and np.array_equal(df["F"].to_numpy(), df["H"].to_numpy()) and (df["fitted"] == 0).all() and df["w"].isna().all()


def test_assemble_known_answer_mean_over_seeds_w_included():
    tab = S["tab"].iloc[:3]
    mk = lambda g, w, p: dict(G=np.array([np.nan, g, g]), fitted=np.array([0.0, 1.0, 1.0]), pi=np.full((3, 3), p), gk=np.full((3, 2), g), w=np.array([np.nan, w, w]))
    df = C8.assemble(tab, [mk(0.2, 0.5, 0.2), mk(0.4, 1.0, 0.6)])
    assert len(df) == 2 and np.allclose(df["G"], 0.3) and np.allclose(df["w"], 0.75) and np.allclose(df["pi0"], 0.4)
    assert np.array_equal(df["F"].to_numpy(), tab["H"].to_numpy()[1:] * np.exp(df["G"].to_numpy()))


def anchor_batch(n=40):
    g = torch.Generator().manual_seed(1)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    return dict(Zg=r(n, 9), z=r(n, 4), ell=0.4 * r(n), H=torch.exp(r(n)), rv=torch.exp(r(n)))


def penalty_covers_anchor() -> bool:
    """λ_g·mean(G²) должен включать w·ℓ: при нулевых экспертах и w = 0,5 штраф = λ_g·mean((w·ℓ)²)."""
    m, b = FM.ResMoE("B", 2, anchor=True, seed=0), anchor_batch()
    with torch.no_grad():
        m.w.fill_(0.5)
    with FM.deterministic():
        d = LS.loss(m, b, 1.0, 0.0) - LS.loss(m, b, 0.0, 0.0)
    return bool(torch.isclose(d, torch.mean((0.5 * b["ell"]) ** 2)))


def test_penalty_lambda_g_includes_anchor_term_and_is_used_by_training(monkeypatch):
    assert penalty_covers_anchor()
    real, calls = LS.loss, []
    monkeypatch.setattr(LS, "loss", lambda *a, **k: calls.append(1) or real(*a, **k))
    FT.train_one(C8.SPEC8, 0, 1.0, 0.0, FT.take(anchor_batch(), range(30)), None, 3)
    assert len(calls) == 3                                                                                    # обучение берёт потери из LS.loss (подмена видна)


def test_mutation_penalty_without_anchor_term_is_caught(monkeypatch):
    real = LS.loss

    def moe_only(model, b, lam_g, lam_lb, reg=LS.REG):
        G = model(b["Zg"], b["z"], b["ell"])[0] - model.w * b["ell"]                                          # штраф только на MoE-часть
        return real(model, b, 0.0, lam_lb, reg) + lam_g * torch.mean(G ** 2)
    monkeypatch.setattr(LS, "loss", moe_only)
    assert not penalty_covers_anchor()


def test_anchor_finds_w_when_M_is_the_true_forecast():
    """Известный ответ: K2 = K3 = истинный F ⇒ ℓ = G_true, w → заметно > 0; QLIKE 8 < QLIKE H на оцениваемых датах."""
    s = SY.make(330, 0)
    tab = s["tab"].copy()
    tab["K2"] = tab["K3"] = tab["H"] * np.exp(s["G_true"])
    df, _ = C8.run_cfg(tab, FE.build_Z(tab).to_numpy(float), lambda sd: s["z"], SEEDS)
    ok = df["fitted"] > 0
    q = lambda F: float(np.mean(df["rv"][ok] / F[ok] - np.log(df["rv"][ok] / F[ok]) - 1))
    assert df.loc[ok, "w"].mean() > 0.4 and q(df["F"]) < q(df["H"]) - 0.01


def leaks(monkeypatch, patch=None) -> bool:
    """Меняем K2, K3 (то есть ℓ) строк после J: прогнозы строк ≤ J обязаны остаться прежними."""
    if patch:
        patch(monkeypatch)
    tab = S["tab"]
    J = int(np.flatnonzero(tab.index >= pd.Timestamp("2018-02-01"))[0])
    tab2 = tab.copy()
    tab2.iloc[J + 1:, [tab.columns.get_loc("K2"), tab.columns.get_loc("K3")]] *= 1.5
    out = [C8.run_cfg(t, FE.build_Z(t).to_numpy(float), Z_OF, (0,))[0].set_index("d")["F"] for t in (tab, tab2)]
    idx = tab.index
    assert not out[0].loc[idx[J + 1]:].equals(out[1].loc[idx[J + 1]:])                                        # возмущение видно позже: тест не пустой
    return not out[0].loc[:idx[J]].equals(out[1].loc[:idx[J]])


def ell_next_day(mp):
    """Мутант: в `_refit` ℓ строки d берётся от строки d + 1 (прогноз бустингов завтрашнего дня)."""
    orig = WF._refit
    mp.setattr(WF, "_refit", lambda tc, Zg, z, spec, d0, lo, b, seeds, grid: orig((tc[0], tc[1], tc[2], np.r_[tc[3][1:], tc[3][-1]]), Zg, z, spec, d0, lo, b, seeds, grid))


def test_honest_real_path_ell_of_later_dates_does_not_change_earlier_forecasts(monkeypatch):
    assert leaks(monkeypatch) is False


def test_mutation_ell_of_day_d_plus_1_is_caught_on_real_path(monkeypatch):
    assert leaks(monkeypatch, ell_next_day) is True
