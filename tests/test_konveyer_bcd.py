"""Конвейер B/C/D (fastbt/konveyer_bcd.py) на синтетике: данные не читаются, в сеть не ходим.
Мутации в памяти (`_mutated`): № 9 (к2 с лагом 0), № 10 (к3 с невыключенной S5/S6), № 1 (σ̂ на +1 день), Д5 (ключ кэша)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_s9_shag_c import _mutated, mini_grid, world  # noqa: E402

from fastbt import fast_kontrol as fk  # noqa: E402
from fastbt import fast_signals, s9, shag0  # noqa: E402
from fastbt import konveyer_bcd as K  # noqa: E402
from fastbt import shag_c as C  # noqa: E402

GB = yaml.safe_load((K.SWEEP / "B_setka_e3.yaml").read_text(encoding="utf-8"))
GD = yaml.safe_load((K.SWEEP / "D_setka_e3.yaml").read_text(encoding="utf-8"))
BAND = dict(cap=[1.0, 2.0], band=[0.0, 0.1])


def grid_c():
    g = mini_grid()
    g["sigmy"] = {k: {"h": [1] if k == "naive" else [1, 5]} for k in ("naive", "harj", "hyb")}
    return g


GRIDS = {"B": BAND, "C": grid_c(), "D": {}}


def eng(seed=3, noise_after=None):
    """Синтетический движок: σ-ряд зависит от метки; noise_after=t — после бара t σ и RVI заменены шумом."""
    ns = world(seed)
    rvi = np.exp(np.random.RandomState(9).normal(0, 0.3, ns.sigma("x").shape))

    def sig(lab):
        S = np.exp(np.cumsum(np.random.RandomState(sum(map(ord, lab))).normal(0, 0.05, rvi.shape), axis=1) * 0.3)
        return S if noise_after is None else _noisy(S, noise_after)
    ns.sigma = sig
    ns.rvi = lambda lag=0: _noisy(rvi, noise_after) if noise_after else rvi
    ns.calib_closed = lambda G: 0.3
    return ns


def _noisy(X, t):
    X = X.copy()
    X[:, t + 1:] = np.random.RandomState(1).lognormal(0, 1, X[:, t + 1:].shape)
    return X


def bases(ns, n=20):
    return {c: C.Base(ns, c, n) for c in K.COMMS}


def chk_causal(ap, t=600):
    """σ после бара t заменена шумом → затворы и веса на барах ≤ t не меняются (σ̂_{d+1} в решении дня d не участвует)."""
    a = K.PTS[ap](eng(), "hyb", 1, GRIDS[ap])
    b = K.PTS[ap](eng(noise_after=t), "hyb", 1, GRIDS[ap])
    assert len(a) == len(b) > 0
    assert all(np.array_equal(x[1][..., :t + 1], y[1][..., :t + 1], equal_nan=True) for x, y in zip(a, b))


def chk_k2_k3():
    ns = eng()
    for c, base in bases(ns).items():
        g = ns.g_row(K.pts_c(ns, "hyb", 1, GRIDS["C"])[0][1])
        pt = s9.sharpe(ns.port(base.A * g, comm=c)["excess"].values)
        assert abs(base.kon.identity_sharpe(g) - pt) < 1e-9          # к2: лаг 0 даёт исходный ряд
        assert base.kon.LAG.min() >= fk.MIN_LAG
        drop = np.isin(shag0.sid_row(ns), ["S5", "S6"])
        assert np.all(base.k3[drop] == 0) and np.array_equal(base.k3[~drop], ns.A_row[~drop])   # к3: ровно S5/S6


@pytest.mark.parametrize("ap", ["B", "C", "D"])
def test_sigma_causal_ok(ap):
    chk_causal(ap)


@pytest.mark.parametrize("ap,fn,owner", [("B", "b_weights", s9.S9), ("D", "d_points", s9.S9), ("C", "gates", C)])
def test_sigma_plus_one_day_caught(monkeypatch, ap, fn, owner):
    var = "eng" if owner is C else "e"
    old = f"S = {var}.sigma(src)"
    new = f"S = np.concatenate([{var}.sigma(src)[:, 1:], np.full(({var}.sigma(src).shape[0], 1), np.nan)], axis=1)"
    target = getattr(owner, fn)
    monkeypatch.setattr(owner, fn, _mutated(target, old, new, (C if owner is C else s9).__dict__))
    with pytest.raises(AssertionError):
        chk_causal(ap)


def test_k2_k3_ok():
    chk_k2_k3()


def test_k2_lag_zero_caught(monkeypatch):
    monkeypatch.setattr(fk, "draw_lags", lambda t, n=1000, seed=0, min_lag=21: np.zeros((n, len(t)), np.int64))
    with pytest.raises((AssertionError, ValueError)):
        chk_k2_k3()


def test_k3_with_s6_left_caught(monkeypatch):
    monkeypatch.setattr(fk, "k3_sids", lambda: ("S6",))
    with pytest.raises(AssertionError):
        chk_k2_k3()


def test_setki_n_declared_before_count():
    for y in (GB, GD):
        assert y["tochek_vsego"] == y["tochek_na_ryad_sigma"] * len(y["ryady_sigma"])
        assert y["tochek_razlichnyh"] == y["tochek_vsego"] - y["tochek_na_ryad_sigma"]      # naive_h5 ≡ naive_h1
        assert y["tochek_razlichnyh"] == y["tochek_vsego_296"]                             # 6 различных рядов × точек на ряд
    assert (GB["tochek_vsego_296"], GD["tochek_vsego_296"]) == (54, 396)                     # #296
    assert len(GB["cap"]) * len(GB["band"]) == GB["tochek_na_ryad_sigma"] == 9
    assert sum(f["tochek"] for f in GD["semyi"].values()) == GD["tochek_na_ryad_sigma"] == 66
    assert 9 * 6 == GB["tochek_vsego_296"] and 66 * 6 == GD["tochek_vsego_296"]


def test_setki_counts_match_code(monkeypatch):
    monkeypatch.setattr(s9, "V2", True)                                  # окна D 126/252/504, как в сетке
    ns = eng()
    d = K.pts_d(ns, "hyb", 1, GD)
    for f, spec in GD["semyi"].items():
        assert sum(m["family"] == f for m, _, _ in d) == spec["tochek"], f
    assert len(K.pts_b(ns, "hyb", 1, GB)) == GB["tochek_na_ryad_sigma"]
    full = C.load_grid()
    assert len([p for p in C.plan("naive", full)]) + sum(len(C.plan(s, full)) for s in ("harj", "hyb")) == 324


def test_klyuch_changes_with_a_yaml(tmp_path):
    a1, a2, g = tmp_path / "a1.yaml", tmp_path / "a2.yaml", tmp_path / "g.yaml"
    a1.write_text("ens_k: 10\n")
    a2.write_text("ens_k: 9\n")
    g.write_text("cap: [1]\n")
    kw = dict(grid_paths=[g], comm=0.0004, series="s", code="c")
    assert K.cache_key(a1, **kw) == K.cache_key(a1, **kw)
    assert K.cache_key(a1, **kw) != K.cache_key(a2, **kw)
    assert K.cache_key(a1, **kw) != K.cache_key(a1, **{**kw, "comm": 0.00035})
    with K.dev_train(a1):
        c1 = fast_signals._ctx()[-1]
    with K.dev_train(a2):
        assert fast_signals._ctx()[-1] != c1                              # кэш сигналов тоже привязан к yaml A


def test_klyuch_mutation_ignoring_yaml_caught(monkeypatch, tmp_path):
    a1, a2, g = tmp_path / "a1.yaml", tmp_path / "a2.yaml", tmp_path / "g.yaml"
    a1.write_text("x: 1\n")
    a2.write_text("x: 2\n")
    g.write_text("y: 1\n")
    monkeypatch.setattr(K, "sha", lambda p: "const")
    kw = dict(grid_paths=[g], comm=0.0004, series="s", code="c")
    assert K.cache_key(a1, **kw) == K.cache_key(a2, **kw)                # мутация: ключ не видит A-yaml — тест выше упал бы


def test_train_only_period(tmp_path):
    a = tmp_path / "a.yaml"
    a.write_text("x: 1\n")
    before = (s9.YEARS, shag0.YEARS, s9.CUT, s9.MAIN_YAML)
    with K.dev_train(a):
        assert s9.YEARS == shag0.YEARS == (2018, 2019) and str(s9.CUT.date()) == "2019-12-31" and s9.MAIN_YAML == a
    assert (s9.YEARS, shag0.YEARS, s9.CUT, s9.MAIN_YAML) == before
    for bad in ((2018, 2019, 2020), (2022,), (2018, 2021)):
        with pytest.raises(AssertionError):
            with K.dev_train(a, bad):
                pass


@pytest.mark.parametrize("ap", ["B", "C", "D"])
def test_konveyer_columns_ladder_no_selection(ap, tmp_path):
    ns, grid = eng(), GRIDS[ap]
    with K.dev_train(tmp_path / "a.yaml"):
        df, ports = K.run_approach(ns, ap, grid, bases(ns), 100, log=lambda *_: None)
        lad = K.lestnitsa(ports, 100)
    need = {"sharpe", "msr", "sortino", "sharpe_k2_med", "p_vs_k2", "sharpe_k3", "d_vs_A", "p_vs_A", "d_vs_k3", "p_vs_k3",
            "mdd_pct", "usl1_c8", "usl2_k3", "usl3_k2", "proshla_c8v2", "rung", "comm_pct", "znak_2018", "znak_2019"}
    assert need <= set(df.columns) and "znak_2020" not in df.columns
    assert not {"vybor", "best", "rank", "vybrano"} & set(df.columns)     # выбора точек нет
    assert set(df["rung"]) == set(K.LADDER) and set(df["comm_pct"].dropna()) == {0.035, 0.04}
    assert set(df["rung"][df["h"] == 0]) == set(K.HLESS)
    steps = set(zip(lad["ot"], lad["k"]))
    assert {("harj", "hyb")} <= steps
    if ap != "C":
        assert {("const", "naive"), ("naive", "harj")} <= steps
    else:
        assert not ((lad["family"] == "C*") & (lad["ot"] == "naive")).any()   # у const/naive нет C*: цепочка harj → hyb


def test_konveyer_b_ignores_k2_in_passed():
    ns = eng()
    df, _ = K.run_approach(ns, "B", BAND, bases(ns), 100, log=lambda *_: None)
    assert (df["proshla_c8v2"] == (df["usl1_c8"] & df["usl2_k3"])).all()


def test_konveyer_d_screen_drops_points():
    ns = eng()
    ns.calib_closed = lambda G: 0.9                                       # экран [0,10; 0,50]: все маски выпали
    df, _ = K.run_approach(ns, "D", {}, bases(ns), 100, log=lambda *_: None)
    masks = df[df["family"] != "D7"]
    assert not masks["screened_in"].any() and masks["sharpe"].isna().all()
    assert df[df["family"] == "D7"]["screened_in"].all()                  # D7 (масштаб) — без экрана
