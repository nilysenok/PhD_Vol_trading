"""Ж-047, тесты 2.9–2.12 (контракт §2): нет прогнозных рядов, сдвиг σ, point-in-time событий, режим.
Синтетика. Интерфейсы — из веток zh047-k1 (sigma_real) и zh047-k3 (rezhim_e1, sobytiya_e1). Мутации — monkeypatch."""
import ast
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_zh047_pomoshch import wait

ROOT = Path(__file__).resolve().parent.parent / "fastbt"
FORBID_IMPORT = ("sigma_e", "vpred", "ch4_dev")
FORBID_WORD = ("sigma_pred", "hyb", "harj")
MODULES = ("etap1", "sigma_real", "rezhim_e1", "sobytiya_e1", "exec_e")


def narusheniya(src: str) -> list:
    """Импорты запрещённых модулей и запрещённые слова в именах/строках кода."""
    bad = []
    for n in ast.walk(ast.parse(src)):
        mods = ([a.name for a in n.names] if isinstance(n, ast.Import) else
                [n.module or ""] + [a.name for a in n.names] if isinstance(n, ast.ImportFrom) else [])
        bad += [m for m in mods for f in FORBID_IMPORT if f in m.split(".")[-1] or f in m]
        words = [n.id] if isinstance(n, ast.Name) else [n.value] if isinstance(n, ast.Constant) and isinstance(n.value, str) else []
        bad += [w for w in words for f in FORBID_WORD if f in w.lower()]
    return bad


@pytest.mark.parametrize("name", MODULES)
def test_29_no_forecast_imports(name):                             # 2.9
    p = ROOT / f"{name}.py"
    if not p.exists():
        pytest.xfail(f"ждёт К: {name}.py ещё нет (strict-ожидание ниже через test_29_modules_exist)")
    assert narusheniya(p.read_text()) == []


@pytest.mark.xfail(any(not (ROOT / f"{m}.py").exists() for m in MODULES), strict=True, reason="ждёт К1/К3/К4: модулей A′ нет")
def test_29_modules_exist():
    assert [m for m in MODULES if not (ROOT / f"{m}.py").exists()] == []


@pytest.mark.parametrize("src", ["from fastbt.sigma_e import load", "import fastbt.ch4_dev.x as q",
                                 "from fastbt import vpred", "x = 'sigma_pred_h1'", "hyb = 1"])
def test_29_mutation_caught(src):                                  # подложенный запрещённый импорт/слово
    assert narusheniya(src) != []


# --- 2.10 сдвиг σ ---
def rv_series(n=400, seed=7):
    r = np.random.default_rng(seed).lognormal(-9.0, 0.6, n)
    return pd.Series(r, index=pd.bdate_range("2017-01-02", periods=n), name="rv")


def sdvig_ok_all(sr):
    rv = rv_series()
    out = {"rv5": sr.sigma_rv(rv, 5), "rv22": sr.sigma_rv(rv, 22), "rv66": sr.sigma_rv(rv, 66), "ewma": sr.sigma_ewma(rv)}
    for n, s in out.items():
        assert sr.proverka_sdvig(s, rv)["ok"], f"{n}: пик корреляции не на последнем известном дне"
        pre = sr.sigma_rv(rv.iloc[:300], 22) if n == "rv22" else None   # σ_d не зависит от RV после d
        if pre is not None:
            pd.testing.assert_series_equal(pre.dropna(), out[n].iloc[:300].dropna())


@wait(1)
def test_210_sigma_proverka_sdvig():
    sdvig_ok_all(importlib.import_module("fastbt.sigma_real"))


@wait(1)
def test_210_mutation_shift_plus1_caught(monkeypatch):             # mutations.md № 1
    sr = importlib.import_module("fastbt.sigma_real")
    orig_rv, orig_ew = sr.sigma_rv, sr.sigma_ewma
    monkeypatch.setattr(sr, "sigma_rv", lambda rv, w: orig_rv(rv, w).shift(-1))
    monkeypatch.setattr(sr, "sigma_ewma", lambda rv, lam=0.94: orig_ew(rv, lam).shift(-1))
    with pytest.raises(AssertionError):
        sdvig_ok_all(sr)


@wait(1)
def test_210_mutation_centered_window_caught(monkeypatch):         # окно, заходящее вперёд
    sr = importlib.import_module("fastbt.sigma_real")
    monkeypatch.setattr(sr, "sigma_rv", lambda rv, w: np.sqrt(rv.rolling(w, center=True, min_periods=w).mean()))
    with pytest.raises(AssertionError):
        sdvig_ok_all(sr)


# --- 2.11 события, point-in-time ---
DATES = pd.bdate_range("2019-03-01", periods=40)


def cbr_case(se, known_from):
    return se.cbr_block(DATES, pd.DataFrame({"date": [DATES[10]], "known_from": [known_from]}))


def div_case(se, announce):
    d = pd.DataFrame({"ticker": ["SBER"], "last_cum_date": [DATES[20]], "announce_date": [announce]})
    return se.div_block(DATES, d, "SBER")


def sc_events(se):
    assert not cbr_case(se, DATES[10]).any(), "known_from = d+1: на d заседание ещё неизвестно — вход не блокирован"
    b = cbr_case(se, DATES[9])
    assert b[9] and b.sum() == 1, "known_from ≤ d: вход по close d запрещён"
    assert div_case(se, DATES[15])[20] and div_case(se, DATES[15]).sum() == 1
    assert not div_case(se, DATES[15] + pd.Timedelta(days=15)).any(), "announce_date +15 дн. (после отсечки) меняет исход"


@wait(3)
def test_211_point_in_time():
    sc_events(importlib.import_module("fastbt.sobytiya_e1"))


@wait(3)
def test_211_mutation_ignore_known_from(monkeypatch):              # фильтр по date без known_from / announce_date
    se = importlib.import_module("fastbt.sobytiya_e1")
    o1, o2 = se.cbr_block, se.div_block
    old = pd.Timestamp("1900-01-01")
    monkeypatch.setattr(se, "cbr_block", lambda d, c: o1(d, c.assign(known_from=old)))
    monkeypatch.setattr(se, "div_block", lambda d, x, t: o2(d, x.assign(announce_date=old), t))
    with pytest.raises(AssertionError):
        sc_events(se)


# --- 2.12 режим: MA200 и ранг только по прошлому ---
def sc_rezhim(rz):
    c = np.full(260, 100.0)
    c[229] = 100.5
    f0 = rz.flag_ma200(c)
    c2 = c.copy()
    c2[230] = 1e9
    assert (f0[:230] == rz.flag_ma200(c2)[:230]).all(), "MA200 зависит от дней после d"
    assert f0[229], "close 100.5 > MA200 → вход разрешён"
    s = np.r_[np.full(202, 0.5), np.full(50, 2.0), 1.0]         # 202/252 = 0,8016 ≥ 0,8 → запрет; с днём d в окне 0,798 → разрешено
    assert not rz.flag_rank(s)[252], "ранг среди d−252…d−1: 0,8016 ≥ 0,8 — вход запрещён"
    w = rz.razmer_ves(np.array([0.02, 0.04, 0.005]), "obratnyj", 0.20)
    assert w[0] == pytest.approx(0.20 / (0.02 * np.sqrt(252))) and w[2] == 1.0, "вес σ дня сигнала, потолок L = 1"
    assert w[1] == pytest.approx(w[0] / 2)


@wait(3)
def test_212_rezhim_only_past():
    sc_rezhim(importlib.import_module("fastbt.rezhim_e1"))


@pytest.mark.parametrize("lever,val", [("RANK_LAG", 0), ("MA_LAG", -1), ("WEIGHT_LAG", -1)])
@wait(3)
def test_212_mutation_current_day_caught(monkeypatch, lever, val):  # mutations.md № 8 (текущий день в ранге) и аналоги
    rz = importlib.import_module("fastbt.rezhim_e1")
    monkeypatch.setattr(rz, lever, val)
    with pytest.raises(AssertionError):
        sc_rezhim(rz)
