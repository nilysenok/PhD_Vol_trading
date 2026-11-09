"""V7 лист 1b-B, G24 (конфигурации 1–2): корректор 5 параметров, F > 0, g ≡ 0 ⇒ F = H, z энкодера года d, PIT walk-forward на реальном пути `run_kind`, одна переоценка побитно.
Синтетика: известный ответ rv = H·exp(0.7·tanh(2·z₀)·…) и z = детерминированная функция даты; артефакты Stage A подменены загрузчиком. Мутации: энкодер года d + 1; «закрыто + 1»."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from models_v2 import metrics
from models_v2.s13 import v6_1_wf as W6
from models_v2.s14 import v7_1b_cfg as CFG, v7_1m_fit as FT, v7_1m_forms as FM, v7_1m_wf as WF

YEARS, SEEDS = (2018, 2019, 2020), (0, 1)


def make_tab(seed=0):
    d = pd.bdate_range("2017-01-02", "2020-12-31")
    n, g = len(d), np.random.default_rng(seed)
    H = np.exp(g.normal(0, 0.3, n))
    z0 = np.sin(np.arange(n) / 9.0)
    rv = H * np.exp(0.7 * np.tanh(2 * z0) + g.normal(0, 0.1, n))
    return pd.DataFrame({"H": H, "K2": H, "K3": H, "rv_next": rv, "t_plus": list(d[1:]) + [d[-1] + pd.Timedelta(days=1)]}, index=d)


def fake_load(path, tab=None):
    """Артефакт (kind, год, seed): z для всех дней ≤ конца года; столбец 3 = (год энкодера − 2018)·0,5 — по нему видно, чей это энкодер."""
    _, _, year, s = path.name.removesuffix(".npz").split("_")
    year, tab = int(year), tab if tab is not None else TAB
    days = tab.index[tab.index < pd.Timestamp(year + 1, 1, 1)]
    i = np.arange(len(days))
    z = np.column_stack([np.sin(i / 9.0), np.cos(i / 13.0), 0.01 * int(s[1:]) * np.ones(len(i)), 0.5 * (year - 2018) * np.ones(len(i))])
    return dict(days=days.to_numpy("datetime64[ns]").astype(np.int64), z=z)


TAB = make_tab()


def run(tab=None, **kw):
    tab = TAB if tab is None else tab
    return CFG.run_kind("tcn", tab, load=lambda p: fake_load(p, tab), seeds=SEEDS, years=YEARS, **kw)


@pytest.fixture(scope="module")
def honest():
    return run()


def test_lin5_has_five_parameters_and_zero_start_gives_F_equal_H_bitwise():
    m = CFG.Lin5()
    assert FM.n_params(m) == 5
    g = torch.Generator().manual_seed(0)
    H, z = torch.exp(torch.randn(32, generator=g, dtype=torch.float64)), torch.randn(32, 4, generator=g, dtype=torch.float64)
    G, pi = m(None, z, None)
    assert pi is None and torch.equal(FM.forecast(H, G), H)
    m.a.data.fill_(1e3)
    m.b.data.fill_(1e3)
    G = m(None, 1e30 * z, None)[0]
    assert bool((FM.forecast(H, G) > 0).all()) and float(G.abs().max()) <= FM.C


def test_run_kind_real_path_F_positive_g0_gives_H_and_signal_is_learned(honest):
    df = honest
    assert df["d"].iloc[0] == pd.Timestamp("2018-01-02") or df["d"].iloc[0] >= pd.Timestamp("2018-01-03")
    assert (df["F"] > 0).all() and np.isfinite(df["g"]).all()
    z = df["g"] == 0
    assert np.array_equal(df["F"][z], df["H"][z])                             # g = 0 ⇒ F = H побитно
    assert not z.any()                                                        # FX (версия 2): пары 2017 года получают z энкодера 2018 — корректор учится с первой оцениваемой даты, F = H до конца 2018 нет
    late = df["d"] >= pd.Timestamp("2019-07-01")
    q_f = metrics.qlike(df["rv"][late].to_numpy(), df["F"][late].to_numpy()).mean()
    q_h = metrics.qlike(df["rv"][late].to_numpy(), df["H"][late].to_numpy()).mean()
    assert q_f < 0.8 * q_h                                                    # известный ответ: поправка выучена


def test_lambda_lb_and_lambda_s_not_applied_and_grid_is_declared():
    assert CFG.GRID == ((0.1, 0.0), (1.0, 0.0)) and all(lb == 0.0 for _, lb in CFG.GRID)
    m = CFG.Lin5()
    b = FT.batch(np.zeros((8, 1)), np.random.default_rng(0).normal(size=(8, 4)), np.zeros(8), np.ones(8), np.ones(8))
    from models_v2.s14 import v7_1m_loss as LS
    assert float(LS.loss(m, b, 0.1, 0.1)) == float(LS.loss(m, b, 0.1, 0.0))   # λ_LB без маршрутизатора не влияет


def test_lin5_patch_restores_make_and_passes_other_specs():
    orig = FT.make
    with CFG.lin5_patch():
        assert isinstance(FT.make(CFG.SPEC, 0), CFG.Lin5) and isinstance(FT.make(dict(kind="flat", z_only=True), 0), FM.Flat)
    assert FT.make is orig


def test_z_of_refit_year_comes_from_encoder_of_that_year_and_mutation_year_plus_one_is_caught(monkeypatch):
    # FX (версия 2): вместо «строка даты года y — энкодер года y» (z_matrix) — матрица энкодера года Y на всех датах ≤ 31.12.Y
    idx = TAB.index
    z = CFG.z_by_year("tcn", 0, idx, lambda p: fake_load(p), YEARS)
    for y in YEARS:
        upto = idx <= pd.Timestamp(y, 12, 31)
        assert np.allclose(z[y][upto, 3], 0.5 * (y - 2018)) and np.isnan(z[y][~upto]).all()
    monkeypatch.setattr(CFG, "enc_year", lambda y: y + 1)
    bad = CFG.z_by_year("tcn", 0, idx, lambda p: fake_load(p), YEARS[:-1])
    assert not np.allclose(bad[2018][idx <= pd.Timestamp(2018, 12, 31), 3], 0.0)   # мутант: z 2018 от энкодера 2019 — ловится
    monkeypatch.undo()


def test_mutation_encoder_of_year_plus_one_changes_z_of_refit(honest, monkeypatch):
    # FX: столбец 3 постоянен внутри переоценки (стандартизация его обнуляет), поэтому различие мутанта — в самом z; на прогноз через столбец 3 не влияет (см. tests/test_v2_v7_fx_z.py — шпион на WF._refit)
    base = CFG.z_by_year("tcn", 0, TAB.index, lambda p: fake_load(p), YEARS)
    monkeypatch.setattr(CFG, "enc_year", lambda y: min(y + 1, 2020))
    mutant = CFG.z_by_year("tcn", 0, TAB.index, lambda p: fake_load(p), YEARS)
    assert not np.array_equal(np.nan_to_num(mutant[2018]), np.nan_to_num(base[2018]))


def test_pit_future_targets_do_not_change_forecasts_and_closed_plus_one_mutant_is_caught(honest, monkeypatch):
    pos = W6.refit_positions(TAB.index, WF.FIRST_REFIT)
    a = next(p for p in pos if TAB.index[p] >= pd.Timestamp("2019-03-01"))
    b = pos[pos.index(a) + 1]
    tab2 = TAB.copy()
    tab2.iloc[a:, tab2.columns.get_loc("rv_next")] *= 5.0                      # цели с позиции a (пара a закрывается после d0) — не должны менять прогнозы блока [a, b)
    base = honest.set_index("d")["g"]
    same = run(tab2).set_index("d")["g"]
    rows = TAB.index[a:b]
    assert np.array_equal(base.reindex(rows).to_numpy(), same.reindex(rows).to_numpy())
    monkeypatch.setattr(WF, "closed_rows", lambda tp, d0: np.flatnonzero(np.asarray(tp, "datetime64[ns]") <= np.datetime64(d0) + np.timedelta64(4, "D")))   # «закрыто + 1»
    leak = run(tab2).set_index("d")["g"]
    assert not np.array_equal(base.reindex(rows).to_numpy(), leak.reindex(rows).to_numpy())


def test_one_refit_recomputed_bitwise_equals_saved_block(honest):
    pos = W6.refit_positions(TAB.index, WF.FIRST_REFIT)
    a = next(p for p in pos if TAB.index[p] >= pd.Timestamp("2019-07-01"))
    got = CFG.refit_block("tcn", TAB, a, lambda p: fake_load(p), SEEDS, YEARS)
    want = honest.set_index("d")["g"].reindex(TAB.index[a:a + len(got)]).to_numpy()
    assert np.array_equal(got, want)
    assert not np.array_equal(got + 1e-12, want)                               # сверка побитовая, не «почти»
