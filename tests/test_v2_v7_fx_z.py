"""V7 лист FX, G32–G33: согласованные координаты z. При переоценке в году Y z ВСЕХ обучающих пар и прогнозов блока — от энкодера года Y (того же seed); пары до 2018 тоже получают z.
Красный на старом `z_matrix` (строка даты года y — от энкодера года y: обучающие пары прошлых лет в чужих координатах). Мутации: «z пары от энкодера её года», «энкодер года Y+1».
Синтетика: артефакт Stage A (kind, год, seed) — z для всех дней ≤ конца года; столбец 3 = (год энкодера − 2018)·0,5 — по нему видно, чей это энкодер."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s14 import v7_1b_cfg as CFG, v7_1m_wf as WF

YEARS, SEEDS = (2018, 2019, 2020), (0, 1)


def make_tab(seed=0):
    d = pd.bdate_range("2017-01-02", "2020-12-31")
    n, g = len(d), np.random.default_rng(seed)
    H = np.exp(g.normal(0, 0.3, n))
    rv = H * np.exp(0.7 * np.tanh(2 * np.sin(np.arange(n) / 9.0)) + g.normal(0, 0.1, n))
    return pd.DataFrame({"H": H, "K2": H, "K3": H, "rv_next": rv, "t_plus": list(d[1:]) + [d[-1] + pd.Timedelta(days=1)]}, index=d)


TAB = make_tab()


def fake_load(path):
    _, _, year, s = Path(path).name.removesuffix(".npz").split("_")
    year = int(year)
    days = TAB.index[TAB.index < pd.Timestamp(year + 1, 1, 1)]
    i = np.arange(len(days))
    z = np.column_stack([np.sin(i / 9.0), np.cos(i / 13.0), 0.01 * int(s[1:]) * np.ones(len(i)), 0.5 * (year - 2018) * np.ones(len(i))])
    return dict(days=days.to_numpy("datetime64[ns]").astype(np.int64), z=z)


def run_spied(monkeypatch):
    """Боевой `run_kind` со шпионом на `WF._refit`, поставленным ДО входа в `run_kind` (видит z, с которым реально учится корректор). -> [(d0, z)]."""
    seen, orig = [], WF._refit

    def rec(tab_cols, Zg, z, spec, d0, lo, b, seeds, grid):
        seen.append((d0, np.array(z, float)))
        return orig(tab_cols, Zg, z, spec, d0, lo, b, seeds, grid)
    monkeypatch.setattr(WF, "_refit", rec)
    CFG.run_kind("tcn", TAB, load=fake_load, seeds=SEEDS[:1], years=YEARS)
    monkeypatch.setattr(WF, "_refit", orig)
    return seen


def violations(seen) -> list:
    """Переоценки (с 2018), где у конечных строк z столбец 3 ≠ метка энкодера года d0."""
    out = []
    for d0, z in seen:
        fin = np.isfinite(z).all(axis=1)
        if d0.year >= 2018 and fin.any() and not np.allclose(z[fin, 3], 0.5 * (d0.year - 2018)):
            out.append(d0)
    return out


def test_refit_in_year_Y_all_rows_from_encoder_Y(monkeypatch):
    seen = run_spied(monkeypatch)
    assert any(d0.year == 2020 for d0, _ in seen) and violations(seen) == []
    d0, z = next((d0, z) for d0, z in seen if d0 >= pd.Timestamp("2018-01-02"))
    fin = np.isfinite(z).all(axis=1)
    assert fin[:5].all() and TAB.index[fin][0] == TAB.index[0]              # пары 2017 года получают z (энкодер 2018 был известен на 01.01.2018)
    assert not fin[TAB.index > pd.Timestamp("2018-12-31")].any()           # дат после конца года Y в z нет


def test_mutation_pair_z_from_encoder_of_its_own_year_is_caught(monkeypatch):
    def mixed(kind, seed, index, year, load, window):
        z = np.full((len(index), 4), np.nan)
        for y in YEARS:
            a = fake_load("%s_w5_%d_s%d.npz" % (kind, y, seed))
            days = pd.DatetimeIndex(a["days"].astype("datetime64[ns]"))
            use = days.year == y
            z[index.get_indexer(days[use])] = a["z"][use]
        return z
    monkeypatch.setattr(CFG, "z_of_year", mixed)
    assert violations(run_spied(monkeypatch)) != []


def test_mutation_encoder_of_year_plus_one_is_caught(monkeypatch):
    monkeypatch.setattr(CFG, "enc_year", lambda y: y + 1)
    assert violations(run_spied(monkeypatch)) != []


def test_warm_pairs_get_z_enter_training_and_output_dates_unchanged(monkeypatch):
    from models_v2.s14 import v7_fx_warm as WM
    wd = pd.bdate_range("2016-10-03", "2016-12-29")
    g = np.random.default_rng(1)
    Hw = np.exp(g.normal(0, 0.3, len(wd)))
    warm = pd.DataFrame({"H": Hw, "K2": np.nan, "K3": np.nan, "rv_next": Hw * np.exp(0.7 * np.tanh(2 * np.sin(np.arange(len(wd)) / 9.0))), "t_plus": list(wd[1:]) + [pd.Timestamp("2016-12-30")]}, index=wd)
    full = pd.concat([warm, TAB])

    def load(path):
        year = int(Path(path).name.removesuffix(".npz").split("_")[2])
        days = full.index[full.index < pd.Timestamp(year + 1, 1, 1)]
        i = np.arange(len(days))
        return dict(days=days.to_numpy("datetime64[ns]").astype(np.int64), z=np.column_stack([np.sin(i / 9.0), np.cos(i / 13.0), np.zeros(len(i)), 0.5 * (year - 2018) * np.ones(len(i))]))
    seen, orig = [], WF._refit
    monkeypatch.setattr(WF, "_refit", lambda tc, Zg, z, spec, d0, lo, b, seeds, grid: (seen.append((d0, np.isfinite(np.array(z, float)).all(axis=1), len(tc[0]))), orig(tc, Zg, z, spec, d0, lo, b, seeds, grid))[1])
    df = CFG.run_kind("tcn", TAB, load=load, seeds=(0,), years=YEARS, warm=warm)
    monkeypatch.setattr(WF, "_refit", orig)
    d0, fin, n = next(s for s in seen if s[0] >= pd.Timestamp("2018-01-02"))
    assert n == len(full) and fin[:len(warm)].all()                         # прогрев: пары с z входят в таблицу переоценки
    assert df["d"].iloc[0] >= pd.Timestamp("2018-01-03") and df["d"].isin(TAB.index).all()
    assert CFG.refit_block("tcn", TAB, int(np.flatnonzero(TAB.index >= pd.Timestamp("2019-03-01"))[0]), load, (0,), YEARS, warm=warm).shape[0] > 0
    real = WM.load_warm()                                                    # реальный прогрев: 377 дат с 2015-07-07, до 2016-12-29
    assert len(real) == 377 and real.index[0] == pd.Timestamp("2015-07-07") and real.index[-1] < pd.Timestamp("2016-12-30")


def test_pca_of_encoder_year_uses_only_dates_before_year_and_mutation_is_caught(monkeypatch):
    from models_v2.s14 import v7_1e_rep as REP
    d = pd.bdate_range("2017-01-02", "2018-12-31")
    z = np.random.default_rng(3).normal(size=(len(d), 16))
    z2 = z.copy()
    z2[d >= pd.Timestamp("2018-01-01")] += 5.0 * np.random.default_rng(4).normal(size=(int((d >= pd.Timestamp("2018-01-01")).sum()), 16))
    a, b = REP.fit_pca_past(z, d, 2018), REP.fit_pca_past(z2, d, 2018)
    assert np.array_equal(a["comps"], b["comps"]) and np.array_equal(a["mu"], b["mu"])      # данные года Y и позже на главные компоненты не влияют
    monkeypatch.setattr(REP, "_past_mask", lambda dates, year: np.asarray(pd.DatetimeIndex(dates) < pd.Timestamp(year + 1, 1, 1)))   # мутант: компоненты по датам ≥ Y
    assert not np.array_equal(REP.fit_pca_past(z, d, 2018)["comps"], REP.fit_pca_past(z2, d, 2018)["comps"])


def test_mixed_coordinates_are_worse_than_consistent_known_answer():
    """Известный ответ: rv = H·exp(0,7·tanh(2·z₀)). Энкодер 2019 даёт z₀ с обратным знаком: учёба на смеси (2018 — свои, 2019 — перевёрнутые) хуже учёбы в одной системе."""
    n, yr = len(TAB), TAB.index.year.to_numpy()
    z_ok = np.column_stack([np.sin(np.arange(n) / 9.0), np.cos(np.arange(n) / 13.0), np.zeros(n), np.zeros(n)])
    z_mix = z_ok.copy()
    z_mix[yr == 2019, 0] *= -1.0                                           # координата 1 в 2019 перевёрнута (другой знак главной компоненты)
    q = {}
    with CFG.lin5_patch():
        for name, z in (("ok", z_ok), ("mixed", z_mix)):
            G = WF.walk_forward(TAB, np.zeros((n, 1)), z, CFG.SPEC, seeds=(0,), grid=CFG.GRID)["G"].to_numpy(float)
            late = (TAB.index >= pd.Timestamp("2020-01-02")) & np.isfinite(G)
            F = TAB["H"].to_numpy(float) * np.exp(G)
            q[name] = metrics.qlike(TAB["rv_next"].to_numpy(float)[late], F[late]).mean()
    assert q["ok"] < 0.8 * q["mixed"]
