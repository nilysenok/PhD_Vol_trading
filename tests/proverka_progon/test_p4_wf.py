"""П-4 (проверяющий), Л-8: walk-forward и PBO_wf на синтетике. Контракт: docs/zadaniya/2026-10-28_progon_L8_wf.md.
wf(M, rows, daty, pravilo) -> {"oos_ryad", "oos_by_quarter", ...}; M — T × N; вызов vybor_region перехватывается шпионом (аргументы
kvartaly_idx / radius / tsel). П-7, Ж-086: в klyuch добавлены постоянные оси t_vyhod = open2, vhod = 0.5 (без них ValueError; ожидания те же). Модули импортируются внутри тестов: до кода каждый тест красный (без skip и xfail, Ж-081)."""
import importlib
import inspect
import itertools
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
DATY = pd.bdate_range("2016-01-01", "2019-12-31").values.astype("datetime64[D]")
QI = pd.PeriodIndex(DATY, freq="Q").asi8 - pd.Period("2016Q1").ordinal
KEYS = [f"{y}Q{q}" for y in range(2016, 2020) for q in range(1, 5)]
OOS_KEYS = KEYS[8:]
NAMES = ["s1", "s2", "s3", "s4", "n1"]


def sr(x):
    s = np.nanstd(x, ddof=1)
    return 0.0 if s < 1e-15 else float(np.nanmean(x) / s * np.sqrt(252))


def kub(mu_q, seed=1):
    """4 сигнальные оси × 3 значения + шумовая ×2 = 162 столбца. mu_q(q) -> дневной дрейф плато (все сигнальные ячейки)."""
    pts = list(itertools.product(*([range(3)] * 4 + [range(2)])))
    mu = np.array([mu_q(q) for q in range(16)])
    M = mu[QI][:, None] + np.random.default_rng(seed).normal(0, 0.01, (len(DATY), len(pts)))
    rows = []
    for j, p in enumerate(pts):
        bq = {k: sr(M[QI == i, j]) for i, k in enumerate(KEYS)}
        v = np.array(list(bq.values()))
        rows.append(dict(status="ok", klyuch={**{a: repr(float(i)) for a, i in zip(NAMES, p)}, "t_vyhod": "open2", "vhod": "0.5"}, by_quarter=bq,
                         mediana_kv=float(np.median(v)), dolya_pol_kv=float((v > 0).sum() / 16),
                         sdelok_y={str(y): 60 for y in range(2016, 2020)}, storony_y={str(y): 40 for y in range(2016, 2020)}))
    return M, rows


def sharpe_oos(res):
    return sr(np.asarray(res["oos_ryad"], float)[np.isfinite(res["oos_ryad"])])


@pytest.fixture(autouse=True)
def _dvojnik_l7(monkeypatch):
    """Пока Л-7 нет в базе — двойник по контракту (ряд = среднее всех столбцов; сам регион проверяет test_p4_region.py).
    Подмена только на время теста, чтобы тесты региона двойник не видели; progon_wf на каждый тест импортируется заново."""
    try:
        importlib.import_module("fastbt.progon_region")
    except ImportError:
        m = types.ModuleType("fastbt.progon_region")
        m.vybor_region = lambda M, rows, daty, kvartaly_idx=None, radius=1, tsel="mediana": dict(
            centr=None, chleny=[], ryad=np.asarray(M).mean(axis=1), score=0.0, osi=[])
        monkeypatch.setitem(sys.modules, "fastbt.progon_region", m)
    monkeypatch.delitem(sys.modules, "fastbt.progon_wf", raising=False)
    import fastbt
    monkeypatch.delattr(fastbt, "progon_wf", raising=False)
    yield
    sys.modules.pop("fastbt.progon_wf", None)


@pytest.fixture
def shpion(monkeypatch):
    """Оборачивает vybor_region (в progon_region и в progon_wf) и пишет (q-независимые) аргументы вызовов."""
    from fastbt import progon_region as R, progon_wf as W
    calls, real = [], R.vybor_region
    sig = inspect.signature(real)

    def spy(*a, **k):
        b = sig.bind(*a, **k)
        b.apply_defaults()
        calls.append(dict(kv=None if b.arguments["kvartaly_idx"] is None else list(b.arguments["kvartaly_idx"]),
                          radius=b.arguments["radius"], tsel=b.arguments["tsel"]))
        return real(*a, **k)
    monkeypatch.setattr(R, "vybor_region", spy)
    if hasattr(W, "vybor_region"):
        monkeypatch.setattr(W, "vybor_region", spy)
    return calls


def test_wf_ne_vidit_oos_kvartal(shpion):
    """Для OOS-квартала q выбор — по W кварталам строго до q: kv == [q-W, q-1]. Мутация «окно включает q» ловится."""
    from fastbt import progon_wf as W
    M, rows = kub(lambda q: 0.0008)
    for pr in W.PRAVILA:
        shpion.clear()
        W.wf(M, rows, DATY, pr)
        wnd, r, t = pr
        assert len(shpion) == 8, "по одному выбору на каждый из 8 OOS-кварталов"
        for q, c in zip(range(8, 16), shpion):
            assert c["kv"] == list(range(q - wnd, q)) and max(c["kv"]) < q, (pr, q, c["kv"])
            assert c["radius"] == r and c["tsel"] == t


def test_okna_w4_i_w8_razlichayutsya(shpion):
    from fastbt import progon_wf as W
    M, rows = kub(lambda q: 0.0008)
    shpion.clear()
    W.wf(M, rows, DATY, (4, 1, "mediana"))
    k4 = [c["kv"] for c in shpion]
    shpion.clear()
    W.wf(M, rows, DATY, (8, 1, "mediana"))
    k8 = [c["kv"] for c in shpion]
    assert k4[0] == [4, 5, 6, 7] and k8[0] == list(range(8)) and k4 != k8


def test_obshchij_oos_u_vseh_8_pravil():
    """Общий OOS 2018Q1–2019Q4 у всех 8 правил: одинаковые ключи и длина ряда (мутация: W=4 начинает с 2017Q1)."""
    from fastbt import progon_wf as W
    M, rows = kub(lambda q: 0.0008)
    assert len(W.PRAVILA) == 8 and len(set(W.PRAVILA)) == 8
    n_oos = int((QI >= 8).sum())
    for pr in W.PRAVILA:
        res = W.wf(M, rows, DATY, pr)
        assert list(res["oos_by_quarter"]) == OOS_KEYS, pr
        assert len(res["oos_ryad"]) == n_oos, pr


def test_signal_tolko_2016_2017_oos_okolo_nulya():
    from fastbt import progon_wf as W
    M, rows = kub(lambda q: 0.002 if q < 8 else 0.0)
    assert abs(sharpe_oos(W.wf(M, rows, DATY, W.OSNOVNOE))) < 2.5          # OOS без сигнала: N(0, ~0,7), 2,5 ≈ 3,5σ


def test_statsionarnyj_signal_oos_polozhitelen():
    from fastbt import progon_wf as W
    M, rows = kub(lambda q: 0.002)
    res = W.wf(M, rows, DATY, W.OSNOVNOE)
    assert sharpe_oos(res) > 1.0 and np.mean([v > 0 for v in res["oos_by_quarter"].values()]) >= 0.75


def test_osnovnoe_ne_menyaetsya_posle_scheta():
    from fastbt import progon_wf as W
    assert W.OSNOVNOE == (8, 1, "mediana") and W.OSNOVNOE in W.PRAVILA
    M, rows = kub(lambda q: 0.002 if q < 8 else 0.0)
    snap = (W.OSNOVNOE, list(W.PRAVILA))
    W.tablica(M, rows, DATY)
    assert (W.OSNOVNOE, list(W.PRAVILA)) == snap == ((8, 1, "mediana"), snap[1])


def test_tablica_8_strok():
    from fastbt import progon_wf as W
    M, rows = kub(lambda q: 0.002)
    t = W.tablica(M, rows, DATY)
    assert isinstance(t, pd.DataFrame) and len(t) == 8


def test_pbo_wf_izvestnyj_otvet():
    """T_oos × 8: правило 3 доминирует во всех блоках -> PBO = 0; iid-правила -> около 0,5; 7 правил — отказ."""
    from fastbt import progon_wf as W
    rng = np.random.default_rng(11)
    T = int((QI >= 8).sum())
    oos = rng.normal(0, 0.01, (T, 8))
    oos[:, 3] += 0.01
    assert W.pbo_wf(oos) == 0.0
    assert 0.2 <= W.pbo_wf(rng.normal(0, 0.01, (T, 8))) <= 0.8
    with pytest.raises((ValueError, AssertionError)):
        W.pbo_wf(oos[:, :7])                                                 # мутация «выпало правило»
