"""B1-5 (Ж-204): главный тест обрезки для новых строк B1 — прогноз на d побитно одинаков при порче баров после 18:50 дня d.
Мутации: № 1 (признаки +1 день), «остаток к HAR на всей выборке» (M-B1-7)."""
import numpy as np
import pandas as pd
import pytest

from models_v2 import har
from models_v2.s2 import features as ft
from models_v2.s3 import resid
from models_v2.s4 import design, fit, mkt, synth
from models_v2.s5 import pit
from models_v2.s5.b1 import designs, jobs

HP = fit.HP(3, 10.0, 2.0, 0.1, 0.8, 1.0, 80)
D = 560
FAMILIES = ["obj_l1n", "obj_l1raw", "obj_l3x", "obj_l3l", "obj_l4", "obj_l5a", "obj_l5c", "har_h2_l4", "har_h2_l4m", "har_h1_s", "har_h1_raw",
            "reg_r1a", "reg_r1b", "reg_r1c"]


def build(bars_a, bars_b, full_lb: bool = False):
    F = ft.build(bars_a)
    M, _ = mkt.block(None, F.X[:, [0, 1, 2]], F.seg, F.dates, [mkt.paper(bars_a), mkt.paper(bars_b)])
    pan = har.Panel(F.dates, F.seg, F.rv, F.X[:, :4])
    lb = resid.harj_ab(pan, full=full_lb)[1]
    Mp = design.columns(M, "proxy")
    DZ = {"m_mkx": design.make(F, "m", Mp), "resid_mkx": design.make(F, "resid", Mp, lb), "level_mkx": design.make(F, "level", Mp)}
    specs = designs.build_specs(F, DZ, lb)
    jobs.init(F, specs, {y: HP for y in range(2018, 2032)}, pit.eps_floor(F.rv))
    return F, specs


@pytest.fixture(scope="module")
def bars():
    return synth.random_bars(700, 21), synth.random_bars(700, 22)


def dmg(F, a, b):
    d = pd.Timestamp(F.dates[D])
    return synth.damage_after(a, d), synth.damage_after(b, d)


def forecast(name):
    k, r = jobs.block_job((name, D))
    return np.array([r["f"], r["s2"]])


@pytest.mark.parametrize("name", FAMILIES)
def test_forecast_bitwise_invariant_to_bars_after_d(bars, name):
    a, b = bars
    F, _ = build(a, b)
    f1 = forecast(name)
    build(*dmg(F, a, b))
    f2 = forecast(name)
    assert f1.tobytes() == f2.tobytes()


@pytest.mark.parametrize("name", ["reg_r2a", "reg_r2b"])
def test_r2_experts_bitwise_invariant(bars, name, monkeypatch):
    monkeypatch.setattr(designs, "N_MIN", 50)
    a, b = bars
    F, _ = build(a, b)
    o1 = jobs.experts_job((name, D))[1]
    build(*dmg(F, a, b))
    o2 = jobs.experts_job((name, D))[1]
    assert np.array(o1, float).tobytes() == np.array(o2, float).tobytes()
    assert any(np.isfinite(x[0]) for x in o1)          # хотя бы один эксперт реально обучен


def test_mutation_features_plus_one_day_caught(bars):
    a, b = bars
    F, specs = build(a, b)
    specs["obj_l4"].X = np.vstack([specs["obj_l4"].X[1:], specs["obj_l4"].X[-1:]])
    f1 = forecast("obj_l4")
    F2, specs2 = build(*dmg(F, a, b))
    specs2["obj_l4"].X = np.vstack([specs2["obj_l4"].X[1:], specs2["obj_l4"].X[-1:]])
    assert f1.tobytes() != forecast("obj_l4").tobytes()


def test_mutation_residual_to_har_fitted_on_full_sample_caught(bars):
    a, b = bars
    F, _ = build(a, b, full_lb=True)
    f1 = forecast("har_h1_s")
    build(*dmg(F, a, b), full_lb=True)
    assert f1.tobytes() != forecast("har_h1_s").tobytes()


def test_unmutated_h1_s_is_invariant_control(bars):
    a, b = bars
    F, _ = build(a, b)
    f1 = forecast("har_h1_s")
    build(*dmg(F, a, b))
    assert f1.tobytes() == forecast("har_h1_s").tobytes()
