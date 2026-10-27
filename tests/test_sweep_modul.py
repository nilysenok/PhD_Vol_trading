"""Ж-035: приёмка модуля владельца `fastbt/sweep_drivers.py` (OAT, плато, PBO, DSR, Моррис, Optuna) и сквозной
прогон оценщик -> oat_sweep. Детекторы `ok_*` — True, если свойство выполнено; мутации — `test_sweep_modul_mut.py`.
`xfail(strict=True)` — найденное расхождение с контрактом/спецификацией (красное, названо в отчёте; починка — решение владельца).
Тесты v1.1 (PBO, Моррис, Optuna) — `@pytest.mark.v11`."""
import random

import numpy as np
import pandas as pd
import pytest

from fastbt import metriki_e as me
from fastbt import sweep_drivers as sd
from fastbt import sweep_stat as ss
from fastbt.sweep_synth import SynthEvaluator
from test_sweep_engine import Tiny

v11 = pytest.mark.v11
GRID = {"a": [3, 5, 10], "w": [0.5, 1.0, 2.0]}
BASE = {"a": 5, "w": 1.0}
PBO_TOL = 0.10          # объявлено до счёта: |среднее PBO − 0,5| ≤ 0,10 по 12 матрицам 400×20 (seed 0..11)


def make_eval(ev, years_of=None, comm=0.0004):
    """Адаптер оценщика к контракту модуля: evaluate(params, period) -> {sharpe, by_year, net}; период -> годы."""
    years_of = years_of or {"train": (2018, 2019)}

    def evaluate(params, period):
        yrs = years_of[period]
        sigs = ev.signals({k: v for k, v in params.items() if k in ev.sig_names})
        df = ev.pnl(sigs, params, params.get("comm", comm), pd.Timestamp(f"{max(yrs) + 1}-01-01"))
        df = ss.year_slice(df, yrs)
        return {"sharpe": float(me.sharpe(df["excess"].values)), "by_year": ss.yearly_sharpes(df, yrs),
                "net": float(df["excess"].sum())}
    return evaluate


def lin(p, period):                       # аналитика: Шарп линеен по x1, x2; x3 не влияет
    v = 3 * p["x1"] - p["x2"]
    return {"sharpe": v, "by_year": {2018: v, 2019: v}}


def spike_eval(p, period):                # плато вокруг (20; 0,6), шумовой пик в (5; 0,9)
    v = -((p["w"] - 20) / 15) ** 2 - ((p["p"] - 0.6) / 0.3) ** 2 + (2.5 if (p["w"], p["p"]) == (5, 0.9) else 0.0)
    return {"sharpe": v, "by_year": {2018: v + .05, 2019: v - .05}}


SG = {"w": [5, 10, 15, 20, 25, 30, 35], "p": [0.3, 0.45, 0.6, 0.75, 0.9]}


def ok_plateau_not_peak() -> bool:
    top = sd.pick_by_plateau(sd.factorial_sweep(spike_eval, {}, SG, "train"), k=1)[0]
    return (top["w"], top["p"]) == (20, 0.6)


def ok_param_moves_result(ev=None) -> bool:
    o = sd.oat_sweep(make_eval(ev or SynthEvaluator()), BASE, GRID, "train")
    return o["tornado"]["a"]["range"] > 0 and o["tornado"]["w"]["range"] > 0


def ok_validation_not_in_train() -> bool:
    objs = [[r["objective"] for r in sd.oat_sweep(make_eval(SynthEvaluator(late=k)), BASE, GRID, "train")["rows"]] for k in (1.0, 6.0)]
    return objs[0] == objs[1]


def ok_commission_rows() -> bool:
    o = sd.oat_sweep(make_eval(Tiny()), {"a": 1, "comm": 0.0004}, {"comm": [0.0, 0.00035, 0.0004]}, "train")
    return all(abs(r["net"] + 4 * r["comm"]) < 1e-12 for r in o["rows"])


def ok_morris_mu_star() -> bool:
    lv = {k: [0, 1, 2, 3, 4] for k in ("x1", "x2", "x3")}
    m = sd.morris_screen(lin, lv, "train", n_paths=10)["effects"]
    return bool(np.allclose([m["x1"]["mu_star"], m["x2"]["mu_star"], m["x3"]["mu_star"]], [3, 1, 0]))


def pbo_random_mean() -> float:
    out = []
    for s in range(12):
        rnd = random.Random(s)
        out.append(sd.pbo_cscv([[rnd.gauss(0, 1) for _ in range(20)] for _ in range(400)])["pbo"])
    return float(np.mean(out))


def pbo_skill() -> float:
    rnd = random.Random(99)
    return sd.pbo_cscv([[rnd.gauss(0.3 if j == 3 else 0, 1) for j in range(20)] for _ in range(400)])["pbo"]


def ok_dsr_n1_grows() -> bool:
    r = random.Random(1)
    return sd.dsr(0.08, [0.08], T=500)["dsr"] > sd.dsr(0.08, [r.gauss(0, 0.05) for _ in range(100)], T=500)["dsr"]


def ok_objective_penalty() -> bool:
    return sd.objective_stable({"by_year": {2018: 1.0, 2019: 0.0}}) < 0.5 and sd.objective_stable({"by_year": {2018: 1.0, 2019: 1.0}}) == 1.0


# ---------- Ж-031: приёмка ----------
def test_param_menyaet_rezultat():
    assert ok_param_moves_result()


def test_validation_ne_vliyaet_na_tselevuyu_train():
    assert ok_validation_not_in_train()
    leak = make_eval(SynthEvaluator(late=1.0), {"train": (2018, 2019, 2020)})       # подмешали 2020 в train
    leak6 = make_eval(SynthEvaluator(late=6.0), {"train": (2018, 2019, 2020)})
    assert sd.objective_stable(leak(BASE, "train")) != sd.objective_stable(leak6(BASE, "train"))


def test_granitsy_periodov_ryad_train_2018_2019():
    r = make_eval(SynthEvaluator())(BASE, "train")
    assert set(r["by_year"]) == {2018, 2019}


def test_komissiya_obe_storony_cherez_oat():
    assert ok_commission_rows()


@v11
def test_pbo_shum_okolo_polovinu():
    assert abs(pbo_random_mean() - 0.5) <= PBO_TOL


@v11
def test_pbo_signal_nizkiy():
    assert pbo_skill() < 0.2


def test_dsr_n1_rastet():
    assert ok_dsr_n1_grows()
    assert sd.dsr(0.08, [0.08], T=500)["n_trials"] == 1 and sd.dsr(0.08, [0.08], T=500)["sr_star"] == 0.0


def test_plato_ne_na_pike():
    assert ok_plateau_not_peak()
    f = sd.factorial_sweep(spike_eval, {}, SG, "train")
    peak = max(f["rows"], key=lambda r: r["objective"])
    assert (peak["w"], peak["p"]) == (5, 0.9)               # пик действительно шумовой


def test_plato_srednee_sosedey_vklyuchaya_tochku():
    rows = [{"x": 1, "objective": 1.0}, {"x": 2, "objective": 5.0}, {"x": 3, "objective": 9.0}]
    g = {"x": [1, 2, 3]}
    assert sd.plateau_score(rows, ["x"], g, {"x": 2}) == (5.0, 3)
    assert sd.plateau_score(rows, ["x"], g, {"x": 1}) == (3.0, 2)        # край: 2 соседа


# ---------- OAT ----------
def test_oat_chislo_prob_po_povedeniyu_modulya():
    """Модуль: n_trials = 1 + Σ|сетка_i| (базовая точка, если она узел сетки, считается повторно), не Σ(|g|−1)+1."""
    calls = []
    ev = make_eval(SynthEvaluator())
    o = sd.oat_sweep(lambda p, per: calls.append(dict(p)) or ev(p, per), BASE, GRID, "train")
    assert o["n_trials"] == 1 + 3 + 3 == len(o["rows"]) == len(calls)
    assert all(sum(p[k] != BASE[k] for k in BASE) <= 1 for p in calls)       # остальные фиксированы


# ---------- ddof (Д7) и λ ----------
def test_tselevaya_lambda_polovina_i_shtraf():
    assert ok_objective_penalty()


@pytest.mark.xfail(strict=True, reason="расхождение: sweep_drivers.objective_stable считает σ с ddof=0; контракт и ss.objective — ddof=1 (Д7)")
def test_tselevaya_sigma_ddof1():
    assert sd.objective_stable({"by_year": {2018: 1.0, 2019: 0.0}}) == pytest.approx(ss.objective({2018: 1.0, 2019: 0.0}))


# ---------- Моррис, Optuna (v1.1) ----------
@v11
def test_morris_analiticheskaya_funkciya_mu_star():
    assert ok_morris_mu_star()


@v11
@pytest.mark.xfail(strict=True, reason="расхождение: эффект не нормирован по направлению шага (вверх/вниз), σ>0 даже у линейной функции")
def test_morris_sigma_nulevaya_u_lineynoy():
    lv = {k: [0, 1, 2, 3, 4] for k in ("x1", "x2")}
    assert sd.morris_screen(lin, lv, "train", n_paths=10)["effects"]["x1"]["sigma"] == pytest.approx(0.0, abs=1e-12)


def _opt(seed=7, **kw):
    pytest.importorskip("optuna")
    return sd.optuna_search(make_eval(SynthEvaluator()), BASE, GRID, "train", n_trials=15, seed=seed, **kw)


@v11
def test_optuna_seed_vosproizvodim_i_znacheniya_iz_setki():
    a, b = _opt(), _opt()
    assert a["rows"] == b["rows"] and a["n_trials"] == 15
    assert all(r["a"] in GRID["a"] and r["w"] in GRID["w"] for r in a["rows"])
    assert a["best_objective"] == max(r["objective"] for r in a["rows"])


@v11
@pytest.mark.xfail(strict=True, reason="расхождение с Ж-031: optuna_search не ограничивает n_jobs ≤ 8")
def test_optuna_n_jobs_ogranichen():
    with pytest.raises(ValueError):
        _opt(n_jobs=9)


# ---------- сквозной: оценщик -> oat_sweep ----------
def test_skvoznoy_oat_ot_otsenshchika():
    o = sd.oat_sweep(make_eval(SynthEvaluator()), BASE, GRID, "train")
    assert o["n_trials"] == 7 and {"sharpe_2018", "sharpe_2019", "net"} <= set(o["rows"][1])
    base_row = o["rows"][0]
    assert base_row["objective"] == o["base_objective"] == pytest.approx(sd.objective_stable(make_eval(SynthEvaluator())(BASE, "train")))
