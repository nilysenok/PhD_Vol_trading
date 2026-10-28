"""Шаг 4 (v3), боевая проводка на синтетическом Engine: цель QLIKE без окна за 2021, отказ без флага/shag_c,
luchshiy / odna_tochka / svyazka / run / feature_matrix, знак Δ по годам, comm_pct, valid в share_diff.
Данные не читаются: parquet — во временной папке, значения 2022 — заведомо «ядовитые» (777).
python3 -m pytest tests/test_s9_shag4_boy.py -q"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "scripts" / "analysis"))
from test_s9_kontrol import _engine  # noqa: E402

import s9_shag4 as sc  # noqa: E402
from fastbt import s9, shag4, shag4_gl3 as g3, shag4_td as tdl  # noqa: E402

ENGINE = s9.Engine                                         # настоящий класс (s9.Engine в тестах подменяется)
LAST = pd.Timestamp("2021-12-31")
ND = 1100
T3 = {"C_naive": ("C1:W126_p70", "naive_h1"), "C_harj": ("C1:W126_p70", "harj_h1"),
      "C_hyb": ("C1:W126_p70", "hyb_h5")}


# ---------------- цель: окно t+h не выходит за 2021-12-31 ----------------

def _frame(n=30):
    dts = pd.bdate_range(end=LAST, periods=n)
    return pd.DataFrame(dict(date=np.tile(dts, 2), ticker=np.repeat(["A", "B"], n),
                             sigma=np.arange(1.0, 2 * n + 1)))


def _check_cut(fn):
    d = _frame()
    for h in (1, 5):
        out = fn(d, h, LAST)
        for _, g in out.groupby("ticker"):
            s = g.sort_values("date")["sigma"].values
            assert np.isnan(s[-h:]).all() and np.isfinite(s[:-h]).all(), f"h={h}"


def test_cut_target_okno_za_granitsey_nan():
    _check_cut(g3.cut_target)


def test_cut_target_mutatsii_lovyatsya():
    for mutant in (lambda d, h, last: d,                              # без обрезки окна цели
                   lambda d, h, last: g3.cut_target(d, 1, last),      # h=5 обрезается как h=1
                   lambda d, h, last: g3.cut_target(d, h - 1, last)):
        with pytest.raises(AssertionError):
            _check_cut(mutant)


# ---------------- синтетический Engine ----------------

def _rv(seed=0):
    rng = np.random.default_rng(seed)
    lr = np.zeros((3, ND))
    for t in range(1, ND):
        lr[:, t] = 0.8 * lr[:, t - 1] + rng.normal(0, 0.3, 3)
    return np.exp(lr) * 1e-4


def _eng(seed=1):
    ns = _engine(Ds=(ND,) * 3, drift=0.0004, seed=seed)
    dates = ns.dates
    RV = _rv(seed)
    prev = np.concatenate([RV[:, :1], RV[:, :-1]], axis=1)
    ns.RV = RV
    ns.tickers = ["T0", "T1", "T2"]
    ns.nmax, ns.n_t = ND, np.array([ND] * 3)
    ns.cache = {t: {"dates": dates.values} for t in ns.tickers}
    ns.r["bar"], ns.r["cat"] = ns.r["date"], ns.r["pair"] % 3

    def sigma(src):
        rng = np.random.default_rng(sum(map(ord, src)))
        return np.sqrt((0.7 * RV + 0.3 * prev) * np.exp(rng.normal(0, 0.1, RV.shape)))

    ns.sigma = sigma
    ns.rvw = lambda: RV
    ns.g_row = lambda G: ENGINE.g_row(ns, G)
    return ns


def _parquet(eng, root):
    """features_pit_v3: до 2021 — RV и цели (≈RV), 2022 — 777 (читаться не должно)."""
    dts = pd.DatetimeIndex(eng.dates)
    rows = []
    for ti, t in enumerate(eng.tickers):
        bad = dts > LAST
        rv = np.where(bad, 777.0, eng.RV[ti])
        rows.append(pd.DataFrame(dict(date=dts, ticker=t, rv=rv, rv_target_h1=rv, rv_target_mean_h5=rv)))
    p = root / "features_pit_v3.parquet"
    pd.concat(rows).to_parquet(p)
    return p


@pytest.fixture
def stend(tmp_path, monkeypatch):
    eng = _eng()
    _parquet(eng, tmp_path)
    monkeypatch.setattr(s9, "REGEN", tmp_path)

    def gates(e, src, grid, naive):
        G = (e.sigma(src) > np.median(e.sigma(src))).astype(float)
        return {("C1", "W126_p70"): np.stack([G, G, 1 - G])}

    mod = types.ModuleType("fastbt.shag_c")
    mod.gates, mod.load_grid = gates, lambda: {}
    monkeypatch.setitem(sys.modules, "fastbt.shag_c", mod)
    monkeypatch.setattr(shag4, "_has_shag_c", lambda: True)
    return eng


def test_feature_matrix_cel_obrezana_i_bez_2022(stend):
    for col, h in (("rv_target_h1", 1), ("rv_target_mean_h5", 5)):
        Y = sc.feature_matrix(stend, col, h)
        last = int(np.flatnonzero(pd.DatetimeIndex(stend.dates) <= LAST).max())
        assert np.isnan(Y[:, last - h + 1:]).all() and np.isfinite(Y[:, :last - h + 1]).all()
        assert not (Y == 777).any()
    rv = sc.feature_matrix(stend, "rv")                    # не цель — не обрезается
    assert np.isfinite(rv[:, last]).all() and not (rv == 777).any()


# ---------------- отказ ----------------

def test_otkaz_bez_flaga(stend):
    with pytest.raises(PermissionError):
        shag4.portfel(stend, "C1:W126_p70", "hyb_h1", 0.0004)
    with pytest.raises(PermissionError):
        sc.run(T3, 20)
    assert shag4.portfel(stend, "C1:W126_p70", "hyb_h1", 0.0004, True) is not None


def test_otkaz_bez_shag_c(stend, monkeypatch):
    monkeypatch.setattr(shag4, "_has_shag_c", lambda: False)
    with pytest.raises(RuntimeError):
        shag4.portfel(stend, "C1:W126_p70", "hyb_h1", 0.0004, True)
    with pytest.raises(RuntimeError):
        sc.run(T3, 20, True)


def test_gate_cherez_shag_c(stend):
    G = shag4.gate(stend, "C1:W126_p70", "hyb_h1", True)
    assert G.shape == (3, 3, ND) and set(np.unique(G)) <= {0.0, 1.0}
    with pytest.raises(ValueError):
        shag4.gate(stend, "C*1:W126_p70", "naive_h1", True)      # C* на наивной не строится
    with pytest.raises(ValueError):
        shag4.gate(stend, "C1:W999_p70", "hyb_h1", True)         # точки нет в сетке shag_c


# ---------------- 4.D на Engine ----------------

def test_svyazka_okno_i_chislo_nan(stend):
    rows, ref = sc.svyazka(stend, T3, True)
    q = {r["gruppa"]: r for r in rows if r.get("period") == "2018–2021" and r.get("baza") == "HAR-J (б)"}
    ndev = len(pd.bdate_range("2018-01-01", LAST))
    assert q["все"]["n_nan_celi"] == 3 * 5                 # h5: последние 5 дней × 3 тикера
    assert q["все"]["n"] + q["все"]["n_bez_ranga"] == 3 * (ndev - 5)
    assert q["открыт"]["n"] + q["закрыт"]["n"] == q["все"]["n"]
    assert q["открыт"]["dolya"] + q["закрыт"]["dolya"] == pytest.approx(1)
    assert {r["rezhim"] for r in rows} == {"корреляция σ", "решения фильтра", tdl.REZ}
    assert {r["rezhim"] for r in ref} >= {tdl.REZ, "4.D портфельное: затвор тренда"}


def test_svyazka_gody_summiruyutsya(stend):
    rows, _ = sc.svyazka(stend, T3, True)
    q = [r for r in rows if r.get("baza") == "HAR-J (б)" and r["gruppa"] == "все"]
    assert sum(r["n"] for r in q if r["period"] != "2018–2021") == next(r["n"] for r in q if r["period"] == "2018–2021")
    assert {r["period"] for r in q} == {"2018", "2019", "2020", "2021", "2018–2021"}


def test_sdvig_boevoy_put_i_sigma_plus_odin(stend, monkeypatch):
    sc.sdvig(stend, T3); true_sigma = stend.sigma  # noqa: E702
    monkeypatch.setattr(stend, "sigma", lambda src: g3.shifted(true_sigma(src), 1), raising=False)
    with pytest.raises(AssertionError):
        sc.sdvig(stend, T3)


# ---------------- run целиком ----------------

def test_run_pishet_chetyre_tablitsy(stend, monkeypatch):
    out = {}
    monkeypatch.setattr(sc, "write", lambda rows, name: out.__setitem__(name, rows))
    monkeypatch.setattr(s9, "use_v2", lambda c: None)
    monkeypatch.setattr(s9, "load_c_dop", lambda: None)
    monkeypatch.setattr(s9, "load_cache", lambda: None)

    class _E(ENGINE):                                      # методы настоящего Engine (port, _finish) остаются
        def __new__(cls, cache):
            return stend

    monkeypatch.setattr(s9, "Engine", _E)
    sc.run(T3, 20, True)
    assert set(out) == set(sc.TABLES)
    for name in sc.TABLES[:3]:
        assert {r["comm_pct"] for r in out[name]} == {0.035, 0.04}
    assert len(out[sc.TABLES[0]]) == len(out[sc.TABLES[1]]) == 2 * 9 and len(out[sc.TABLES[2]]) == 2 * len(shag4.STEPS)
    assert all(r["pometka"] == shag4.POMETKA for r in out[sc.TABLES[2]])

