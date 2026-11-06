"""B5 (Ж-206, волна B, поток XGB): внутренний критерий crit_pen = mean_Y [mean_y q + 0,25·(max_y q − mean_y q)], y — только ПРОШЛЫЕ годы (< Y).
Известный ответ считается вручную циклами независимо от кода. Мутации: mean вместо max, штраф 1, будущие фолды в pen_Y, перепутанный знак."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s6.xgb import critx


def q_of(Y, y):
    return 0.1 + 0.01 * (y - 2014) + 0.003 * (Y - 2016) * ((y * 7) % 5)


def folds(extra=()):
    rec = [{"Y": Y, "y": y, "q": q_of(Y, y)} for Y in range(2017, 2027) for y in range(2015, Y)]
    return pd.DataFrame(rec + list(extra))


def expected(penalty=0.25):
    pens, means = [], []
    for Y in range(2017, 2027):
        qs = [q_of(Y, y) for y in range(2015, Y)]
        m = sum(qs) / len(qs)
        means.append(m)
        pens.append(m + penalty * (max(qs) - m))
    return sum(means) / 10, sum(pens) / 10


def test_known_answer():
    r = critx.crit(folds())
    em, ep = expected()
    assert r["crit_mean"] == pytest.approx(em, abs=1e-12) and r["crit_pen"] == pytest.approx(ep, abs=1e-12)
    assert r["crit_pen"] > r["crit_mean"]


def test_future_and_nan_folds_do_not_enter_pen():
    extra = [{"Y": 2020, "y": 2022, "q": 99.0}, {"Y": 2020, "y": 2020, "q": 50.0}, {"Y": 2019, "y": 2016, "q": np.nan}]
    assert critx.crit(folds(extra)) == critx.crit(folds())


def test_mutation_mean_instead_of_max_is_caught():
    e = expected()[1]
    f = folds()
    g = f.groupby("Y")["q"]
    mutant = float((g.mean() + 0.25 * (g.mean() - g.mean())).mean())
    assert abs(mutant - e) > 1e-4                                       # критерий без худшего года отличается — тест это видит


def test_mutation_penalty_one_and_sign_are_caught():
    assert abs(critx.crit(folds(), penalty=1.0)["crit_pen"] - expected()[1]) > 1e-4
    assert abs(critx.crit(folds(), penalty=-0.25)["crit_pen"] - expected()[1]) > 1e-4
    assert critx.crit(folds(), penalty=0.0)["crit_pen"] == pytest.approx(expected()[0], abs=1e-12)


def test_mutation_future_folds_in_year_are_caught():
    f = folds()
    leaked = pd.concat([f, pd.DataFrame([{"Y": 2020, "y": 2021, "q": 9.0}])])   # фолд 2021 в критерии года 2020 — будущее
    assert critx.crit(leaked)["crit_pen"] == critx.crit(f)["crit_pen"]          # код отбрасывает y >= Y
    raw = leaked.groupby("Y")["q"].max().mean()                                  # мутант без отбрасывания видит 9.0
    assert raw > f.groupby("Y")["q"].max().mean() + 0.5


def test_missing_year_is_error():
    with pytest.raises(ValueError):
        critx.crit(folds()[lambda d: d["Y"] != 2023])


def test_perspective_rule_and_better_tolerance():
    assert critx.perspective(0.150, 0.160, -0.001, False)
    assert not critx.perspective(0.150, 0.160, 0.001, False)            # хуже HAR-J(б) без 24.02.2022
    assert not critx.perspective(0.150, 0.160, -0.001, True)            # нестабильна
    assert not critx.perspective(0.160, 0.160, -0.001, False)           # ничья не прирост
    assert not critx.perspective(0.170, 0.160, -0.001, False)           # перепутанный знак сравнения
    assert not critx.better(0.16 - 1e-13, 0.16)
