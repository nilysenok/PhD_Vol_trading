"""Главный тест утечки Ж-200: прогноз на d побитно одинаков при порче баров после 18:50 дня d — для всех конфигураций."""
import itertools

import numpy as np
import pandas as pd
import pytest

from models_v2 import har
from models_v2.s2 import features as ft
from models_v2.s3 import resid
from models_v2.s4 import design, fit, mkt, runner, synth

HP = fit.HP(3, 10.0, 2.0, 0.1, 0.8)
D = 560


def world(bars_a, bars_b):
    F = ft.build(bars_a)
    papers = [mkt.paper(bars_a), mkt.paper(bars_b)]
    M, _ = mkt.block(None, F.X[:, [0, 1, 2]], F.seg, F.dates, papers)
    lb = resid.harj_ab(har.Panel(F.dates, F.seg, F.rv, F.X[:, :4]))[1]
    return F, M, lb


@pytest.fixture(scope="module")
def bars():
    return synth.random_bars(700, 11), synth.random_bars(700, 12)


def predict(F, M, lb, form, with_m, w, d):
    dz = design.make(F, form, M if with_m else None, lb if form == "resid" else None)
    return runner.block_fit(F, dz, HP, d, [d], w, None, "work")["f"]


@pytest.mark.parametrize("form,with_m,w", list(itertools.product(design.FORMS, (False, True), (False, True))))
def test_forecast_bitwise_invariant_to_bars_after_d(bars, form, with_m, w):
    a, b = bars
    base = world(a, b)
    dmg = world(synth.damage_after(a, pd.Timestamp(base[0].dates[D])), synth.damage_after(b, pd.Timestamp(base[0].dates[D])))
    f1, f2 = predict(*base, form, with_m, w, D), predict(*dmg, form, with_m, w, D)
    assert f1.tobytes() == f2.tobytes()


def test_mutation_features_plus_one_day_caught(bars):
    a, b = bars
    F, M, lb = world(a, b)
    dmg = world(synth.damage_after(a, pd.Timestamp(F.dates[D])), synth.damage_after(b, pd.Timestamp(F.dates[D])))
    for (G, _, _) in (dmg, (F, M, lb)):
        G.X = np.vstack([G.X[1:], G.X[-1:]])        # признаки строки d видят d+1
    assert predict(F, M, lb, "level", False, False, D).tobytes() != predict(*dmg, "level", False, False, D).tobytes()
