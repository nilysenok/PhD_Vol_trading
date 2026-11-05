"""G2 B2: определения F1–F6 равны GATES_B2 §1 — имена, число столбцов, значения против независимой прямой формулы, сегменты."""
import math

import numpy as np
import pandas as pd
import pytest

from models_v2 import daily
from models_v2.s2 import features as ft
from models_v2.s5.b2 import feat, feat_day, synth_b2

N = {"F1": 9, "F2": 8, "F3": 8, "F4": 8, "F5": 8, "F6": 10}


@pytest.fixture(scope="module")
def w():
    a, b, c = synth_b2.pair(520, 3, "2021-09-01")   # пересекает остановку 28.02–24.03.2022
    return a, b, ft.build(a), feat.build(ft.build(a), a, b)


def test_group_sizes_and_names():
    assert {g: len(c) for g, c in feat.GROUPS.items()} == N
    assert len(feat.ALL) == len(set(feat.ALL)) == 51


def test_all_columns_present_and_not_empty(w):
    _, _, F, df = w
    assert list(df.columns) == list(feat.ALL)
    assert (df.notna().sum() > 100).all(), df.notna().sum()[df.notna().sum() <= 100]


def test_segments_not_glued(w):
    _, _, F, df = w
    k = int(np.flatnonzero(F.seg == 1)[0])
    assert np.isnan(df["f1_lag1"].iloc[k]) and np.isnan(df["f1_chg"].iloc[k])
    assert df["f1_m63"].iloc[k:k + 62].isna().all() and np.isfinite(df["f1_m63"].iloc[k + 62])
    assert np.isnan(df["f5_beta22"].iloc[k:k + 21]).all()


def _day(b, d):
    m = daily.main_session(b)
    m = m[m["begin"].dt.normalize() == d].sort_values("begin")
    r = np.log(m["close"].to_numpy() / m["open"].to_numpy())
    c = np.log(m["close"].to_numpy())
    r = np.concatenate([[r[0]], np.diff(c)])
    return m, r


def test_f4_values_against_direct_formula(w):
    a, _, F, df = w
    i = 300
    d = pd.Timestamp(F.dates[i])
    m, r = _day(a, d)
    n, lS = len(r), F.X[i, 2]
    rq = n / 3 * np.sum((100 * r) ** 4)
    med = [np.median(np.abs(r[j - 1:j + 2])) ** 2 for j in range(1, n - 1)]
    medrv = math.pi / (6 - 4 * math.sqrt(3) + math.pi) * n / (n - 2) * sum(med) * 1e4
    park = sum(math.log(h / l) ** 2 for h, l in zip(m["high"], m["low"])) / (4 * math.log(2)) * 1e4
    rng2 = (100 * math.log(m["high"].max() / m["low"].min())) ** 2
    hh = pd.Series(r ** 2).groupby(m["begin"].dt.hour.to_numpy()).sum()
    row = df.iloc[i]
    assert row["f4_rq"] == pytest.approx(math.log(rq) - 2 * lS, abs=1e-10)
    assert row["f4_rq_rel"] == pytest.approx(math.log(rq) - 2 * math.log(F.rv[i]), abs=1e-10)
    assert row["f4_medrv"] == pytest.approx(math.log(medrv) - lS, abs=1e-10)
    assert row["f4_park"] == pytest.approx(math.log(park) - lS, abs=1e-10)
    assert row["f4_range"] == pytest.approx(math.log(rng2) - lS, abs=1e-10)
    assert row["f4_hhi"] == pytest.approx(float(((hh / hh.sum()) ** 2).sum()), abs=1e-12)


def test_f5_corr_and_f3_ret_against_direct_formula(w):
    a, b, F, df = w
    i = 300
    d = pd.Timestamp(F.dates[i])
    ma, ra = _day(a, d)
    mb, rb = _day(b, d)
    both = pd.Series(ra, index=ma["begin"]).to_frame("s").join(pd.Series(rb, index=mb["begin"]).rename("m"), how="inner")
    exp = (both.s * both.m).sum() / math.sqrt((both.s ** 2).sum() * (both.m ** 2).sum())
    assert df["f5_corr_d"].iloc[i] == pytest.approx(exp, abs=1e-12)
    cl = feat_day.day_stats(a, F.dates)["close"].to_numpy()
    assert df["f3_ret1"].iloc[i] == pytest.approx(100 * math.log(cl[i] / cl[i - 1]), abs=1e-12)


def test_f1_f6_values_against_direct_formula(w):
    _, _, F, df = w
    i = 480   # >= 252 строк после остановки
    lr, lS = np.log(F.rv), F.X[i, 2]
    assert df["f1_lag2"].iloc[i] == pytest.approx(lr[i - 2] - lS, abs=1e-12)
    assert df["f1_m10"].iloc[i] == pytest.approx(math.log(F.rv[i - 9:i + 1].mean()) - lS, abs=1e-12)
    assert df["f1_dev63"].iloc[i] == pytest.approx(lr[i] - math.log(F.rv[i - 63:i].mean()), abs=1e-12)
    assert df["f1_std22"].iloc[i] == pytest.approx(np.std(lr[i - 21:i + 1], ddof=1), abs=1e-12)
    w252 = np.exp(F.X[i - 251:i + 1, 1])
    assert df["f6_pct"].iloc[i] == pytest.approx(float((w252 <= w252[-1]).mean()), abs=1e-12)
    assert df["f6_dev252"].iloc[i] == pytest.approx(F.X[i, 1] - math.log(F.rv[i - 251:i + 1].mean()), abs=1e-12)
