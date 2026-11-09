"""V7 лист 1a-M, G17 (известный ответ и выбор по последним 126 закрытым парам). Синтетика с двумя режимами: правильная поправка зависит от режима. Маршрутизатор восстанавливает режим;
MoE-B не хуже плоского корректора в пределах объявленного допуска; ранняя остановка и решётка выбираются по валидации; детерминизм побитно. Данных SBER нет."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from models_v2 import metrics
from models_v2.s14 import v7_1m_fit as FT, v7_1m_forms as FM, v7_1m_loss as LS, v7_1m_syn as SY

TOL_FLAT = 0.002                         # объявленный допуск: QLIKE(MoE-B) ≤ QLIKE(плоский) + 0,002 на отложенных датах


def split(s, n_train=700):
    tab, Zg, z = s["tab"], s["Zg"], s["z"]
    H, rv = tab["H"].to_numpy(), tab["rv_next"].to_numpy()
    ell = np.log(np.sqrt(tab["K2"].to_numpy() * tab["K3"].to_numpy()) / H)
    idx = np.flatnonzero(np.isfinite(Zg).all(axis=1))
    a, te = idx[:n_train], idx[n_train:]
    mu, sd, mz, sz = Zg[a].mean(0), Zg[a].std(0), z[a].mean(0), z[a].std(0)
    prep = lambda ii: FT.batch((Zg[ii] - mu) / sd, (z[ii] - mz) / sz, ell[ii], H[ii], rv[ii])
    return prep(a[:-126]), prep(a[-126:]), prep(te), te


def qlike_of(models, te, s, idx):
    return float(metrics.qlike(s["tab"]["rv_next"].to_numpy()[idx], np.exp(FT.ensemble_lnF(models, te).numpy())).mean())


@pytest.fixture(scope="module")
def known():
    s = SY.make(1100, 0)
    tr, va, te, idx = split(s)
    out = {"s": s, "te": te, "idx": idx, "reg": s["regime"][idx]}
    for name, spec in (("B", dict(kind="moe", router="B")), ("C", dict(kind="moe", router="C")), ("flat", dict(kind="flat"))):
        out[name] = FT.fit_config(spec, tr, va)
    return out


def pi_by_regime(r, te, reg):
    with torch.no_grad(), FM.deterministic():
        pi = np.mean([m.pi(te["Zg"], te["z"]).numpy() for m in r["models"]], axis=0)
    return pi[reg == 0].mean(axis=0), pi[reg == 1].mean(axis=0)


def test_known_answer_router_recovers_regime(known):
    p0, p1 = pi_by_regime(known["B"], known["te"], known["reg"])
    assert abs(p0[1] - p1[1]) > 0.3 and abs(p0[2] - p1[2]) > 0.3             # оба эксперта специализируются по режимам


def test_known_answer_moe_not_worse_than_flat_within_declared_tolerance(known):
    s, te, idx = known["s"], known["te"], known["idx"]
    qb, qf = qlike_of(known["B"]["models"], te, s, idx), qlike_of(known["flat"]["models"], te, s, idx)
    q_h = float(metrics.qlike(s["tab"]["rv_next"].to_numpy()[idx], s["tab"]["H"].to_numpy()[idx]).mean())
    q_true = float(metrics.qlike(s["tab"]["rv_next"].to_numpy()[idx], (s["tab"]["H"].to_numpy() * np.exp(s["G_true"]))[idx]).mean())
    assert qb <= qf + TOL_FLAT and qb < 0.5 * q_h and qb < q_true + 0.03        # MoE близок к истинной поправке и не хуже плоского


def test_known_answer_router_without_regime_input_fails(known):
    """Маршрутизатор C видит только z (режима там нет): режим не восстановлен, QLIKE заметно хуже B — различие входов проверяется на известном ответе."""
    s, te, idx = known["s"], known["te"], known["idx"]
    assert qlike_of(known["C"]["models"], te, s, idx) > qlike_of(known["B"]["models"], te, s, idx) + 0.02
    p0, p1 = pi_by_regime(known["C"], te, known["reg"])
    assert abs(p0[1] - p1[1]) < 0.15


def test_mutation_b_without_regime_features_caught(monkeypatch):
    s = SY.make(1100, 0)
    tr, va, te, idx = split(s)
    monkeypatch.setattr(FM, "router_input", lambda k, Zg, z: z if k == "B" else Zg)           # мутант: B теряет Z
    monkeypatch.setitem(FM.IN_DIM, "B", 4)
    r = FT.fit_config(dict(kind="moe", router="B"), tr, va)
    p0, p1 = pi_by_regime(r, te, s["regime"][idx])
    assert not (abs(p0[1] - p1[1]) > 0.3 and abs(p0[2] - p1[2]) > 0.3)


def test_early_stopping_epoch_is_validation_optimum_and_refit_reproduces_best_state():
    s = SY.make(700, 1)
    tr, va, te, idx = split(s, 600)
    spec = dict(kind="moe", router="B")
    with FM.deterministic():
        m, ep, q = FT.train_one(spec, 0, 0.1, 0.0, tr, va)
        m2, _, _ = FT.train_one(spec, 0, 0.1, 0.0, tr, None, ep)               # без валидации, ровно ep шагов -> то же состояние
        qs = []
        for e in range(1, min(FT.MAX_EP, ep + FT.PATIENCE) + 1):
            mm, _, _ = FT.train_one(spec, 0, 0.1, 0.0, tr, None, e)
            qs.append(float(LS.qlike_np(va["rv"], va["H"] * torch.exp(mm(va["Zg"], va["z"], va["ell"])[0]))))
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))
    assert q == pytest.approx(qs[ep - 1], rel=1e-12) and int(np.argmin(qs)) + 1 == ep           # лучшая эпоха = минимум QLIKE на валидации


def test_grid_choice_is_argmin_of_validation_qlike_and_uses_all_four_points():
    s = SY.make(700, 2)
    tr, va, te, idx = split(s, 600)
    r = FT.fit_config(dict(kind="moe", router="B"), tr, va, seeds=(0, 1))
    assert [(x["lam_g"], x["lam_lb"]) for x in r["val"]] == list(FT.GRID)
    best = min(r["val"], key=lambda x: x["q"])
    assert r["lam"] == (best["lam_g"], best["lam_lb"]) and len(r["models"]) == 2


def test_training_is_deterministic_bitwise_and_seeds_differ():
    s = SY.make(500, 3)
    tr, va, te, idx = split(s, 400)
    a = FT.fit_config(dict(kind="moe", router="A"), tr, va, seeds=(0, 1), grid=FT.GRID[:1])
    b = FT.fit_config(dict(kind="moe", router="A"), tr, va, seeds=(0, 1), grid=FT.GRID[:1])
    assert torch.equal(FT.ensemble_lnF(a["models"], te), FT.ensemble_lnF(b["models"], te))
    assert not torch.equal(FT.lnF(a["models"][0], te), FT.lnF(a["models"][1], te))


def test_ensemble_forecast_is_mean_of_ln_F_over_seeds():
    s = SY.make(500, 3)
    tr, va, te, idx = split(s, 400)
    r = FT.fit_config(dict(kind="moe", router="A"), tr, va, seeds=(0, 1, 2), grid=FT.GRID[:1])
    manual = sum(FT.lnF(m, te) for m in r["models"]) / 3
    assert torch.allclose(FT.ensemble_lnF(r["models"], te), manual, rtol=0, atol=1e-14)
