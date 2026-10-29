"""Ж-113 пп. 1, 6 / Ж-114 пп. 1–2: период v2 (train 2015–2019, прогрев с 26.08.2014), путь A без σ.
Синтетика (быстро) + реальные данные (нужен FASTBT_BARS1M; нет — пропуск реальной части).
Якорь: 18 точек Sk (S1–S6 × 3) + 2 точки A (0,04 и 0,035 %) = 20, годы 2016–2019, допуск 1e-12. Дивиденды 2014 не подключены.
python3 -m pytest tests/test_period_v2.py -q"""
import pickle

import numpy as np
import pandas as pd
import pytest

from fastbt import bars5m, ekviv_s9, period_v2, s9
from fastbt.signals.interface import TickerData

TOL, PER, GODY = 1e-12, "v2", (2016, 2017, 2018, 2019)
real = pytest.mark.skipif(not bars5m.SRC.exists(), reason="нет 1-минуток D25 (FASTBT_BARS1M)")


@pytest.fixture(autouse=True)
def _restore():
    old = s9.PERIOD, s9.YEARS, s9.VAL_START
    yield
    s9.PERIOD, s9.YEARS, s9.VAL_START = old


# ---------------- синтетика ----------------
def test_umolchanie_i_train4_ne_izmeneny():
    assert s9.PERIODS_S9["train"] == ((2018, 2019, 2020, 2021), 2017)
    assert s9.PERIODS_S9["train4"] == ((2016, 2017, 2018, 2019), 2015)
    assert s9.PERIODS_S9["v2"] == ((2015, 2016, 2017, 2018, 2019), 2014)
    assert ekviv_s9.windows() is ekviv_s9.WINDOWS and ekviv_s9.windows("train4") is ekviv_s9.WINDOWS_TRAIN4
    k_train = s9.cache_key()
    s9.set_period("train4")
    k4 = s9.cache_key()
    s9.set_period("v2")
    kv = s9.cache_key()
    s9.set_period("train")
    assert len({k_train, k4, kv}) == 3 and s9.cache_key() == k_train
    s9.set_period("train4")
    assert s9.cache_key() == k4                      # ключ train4 не зависит от добавок v2


def test_v2_period_ctx_okna_i_B_otkaz():
    with s9.period_ctx("v2"):
        assert s9.YEARS == (2015, 2016, 2017, 2018, 2019) and s9.VAL_START == 2014
        with pytest.raises(NotImplementedError, match="σ-источников нет"):
            s9.need_sigma("B")
        assert s9._latest_name().endswith("_v2.txt")
    assert s9.PERIOD == "train"
    assert ekviv_s9.windows("v2")[2019] == (2015, 2016, 2017, 2018, 2019)


def test_mutaciya_v2_ne_v_klyuche_lovitsya(monkeypatch):
    """Подмена: ключ не видит период (PERIOD=train) — ключи совпали бы; на правке v2 ≠ train."""
    k_train = s9.cache_key()
    s9.set_period("v2")
    monkeypatch.setattr(s9, "PERIOD", "train")
    assert s9.cache_key() == k_train                 # мутация: период в ключе отсутствует
    monkeypatch.setattr(s9, "PERIOD", "v2")
    assert s9.cache_key() != k_train


def _td(dates):
    n = len(dates)
    a = np.arange(n, dtype=float) + 1
    return TickerData("X", np.asarray(dates, "datetime64[ns]"), a, a, a, a, a, a, {"h1": a})


def test_trim_td_s_26_08_2014():
    d = pd.bdate_range("2014-06-09", "2014-12-31").values
    t = period_v2.trim_td(_td(d))
    assert pd.Timestamp(t.dates[0]) == pd.Timestamp("2014-08-26") and t.n == len(t.dates) == len(t.log_ret) == len(t.sigma["h1"])
    assert np.array_equal(t.close, _td(d).close[len(d) - t.n:])


def test_mutaciya_progrev_ne_26_08_2014_lovitsya(monkeypatch):
    d = pd.bdate_range("2014-06-09", "2014-12-31").values
    monkeypatch.setattr(period_v2, "START", pd.Timestamp("2014-06-09"))
    assert pd.Timestamp(period_v2.trim_td(_td(d)).dates[0]) != pd.Timestamp("2014-08-26")


def test_hold_invalid_derzhit_pozitsiyu():
    pos = np.array([0, 1, 1, -1, -1, 0, 1.0])
    ok = np.array([True, True, False, False, True, True, False])
    assert hold_ref(pos, ok) == period_v2.hold_invalid(pos, ok).tolist()
    assert period_v2.hold_invalid(pos, np.zeros(7, bool)).tolist() == [0.0] * 7      # все невалидны: держится первый день


def hold_ref(pos, ok):
    out, last = [], pos[0]
    for i, p in enumerate(pos):
        last = p if (ok[i] or i == 0) else last
        out.append(last)
    return out


def test_mutaciya_nevalidnyy_den_dayot_signal_lovitsya(monkeypatch):
    """Мутация: все дни валидны (правило не подключено) — сигнал на невалидном дне проходит; тест это видит."""
    pos = np.array([0, 1, 1, -1, -1, 0, 1.0])
    ok = np.array([True, True, False, False, True, True, False])
    class K:  # noqa: E301
        valid = pd.DataFrame({"X": ok}, index=pd.bdate_range("2016-01-04", periods=7))
    d = K.valid.index.values
    a = period_v2.hold_years({2016: pos}, "X", d, K)[2016]
    assert a[3] == 1 and a[2] == 1 and a[6] == 0
    K.valid = pd.DataFrame({"X": np.ones(7, bool)}, index=K.valid.index)               # мутация
    assert period_v2.hold_years({2016: pos}, "X", d, K)[2016][3] == -1                 # сигнал прошёл: тест выше упал бы
    K.valid = pd.DataFrame({"X": ok}, index=K.valid.index)
    assert period_v2.valid_vec("X", pd.bdate_range("2017-01-02", periods=2), K).tolist() == [False, False]   # вне календаря


# ---------------- реальные данные ----------------
@pytest.fixture(scope="module")
def evrs():
    cache = {}

    def mk(sid):
        if sid not in cache:
            from fastbt import etap1
            from fastbt.etap1_eval import Evaluator
            inp = etap1.build_inputs(sid, period=PER)
            cache[sid] = Evaluator(sid, inp, etap1.base_point(sid), 0.0004)
        return cache[sid]
    return mk


def deltas(evr, sid, sigp):
    from fastbt import etap1
    ev, p = evr.sk, {**etap1.base_point(sid), **sigp}
    sp = {k: v for k, v in p.items() if k in ev.sig_names}
    sigs = ev.signals(sp)
    bo = {t: ekviv_s9.boevoy_pos(sid, p, t, *ev._td[t]) for t in sigs}
    td = {t: ev._td[t][0] for t in sigs}
    ys = ekviv_s9.windows(PER)[2019]
    with s9.period_ctx(PER):
        eng = ekviv_s9.s9_engine(ekviv_s9.sk_cache(sid, bo, td, ys), ys, PER)
        ref = eng.port(eng.A_row, comm=0.0004)["excess"]
    mine = evr.excess(sp, PER)[0]["excess"]
    out = {}
    for y in GODY:
        r, m = ref[ref.index.year == y], mine[mine.index.year == y]
        assert len(r) > 200 and np.abs(r.to_numpy()).max() > 0, (sid, y)
        out[y] = float(np.abs(m.to_numpy() - r.to_numpy()).max()) if m.index.equals(r.index) else float("inf")
    return out


@real
@pytest.mark.parametrize("i", range(18))
def test_v2_yakor_Sk_ravno_s9(i, evrs):
    from test_yakor_v1 import POINTS
    sid, sigp = POINTS[i]
    d = deltas(evrs(sid), sid, sigp)
    assert set(d) == set(GODY) and max(d.values()) <= TOL, (sid, sigp, d)


@real
@pytest.mark.parametrize("sid", [f"S{k}" for k in range(1, 7)])
def test_v2_anchor_s9_diff_2016_2019(sid, evrs):
    from fastbt import etap1
    d = etap1.anchor_s9_diff(sid, evrs(sid), 0.0004, PER)
    assert set(d) == set(GODY) and all(np.isfinite(v) and v <= TOL for v in d.values()), d


@pytest.fixture(scope="module")
def a_cache(tmp_path_factory):
    old = s9.OUT
    s9.OUT = tmp_path_factory.mktemp("s9_v2")
    try:
        with s9.period_ctx(PER):
            res = pickle.loads(s9.build_cache(workers=3).read_bytes())
    finally:
        s9.OUT = old
    return res


@real
def test_v2_kesh_A_progrev_i_hold(a_cache):
    assert len(a_cache) == 17
    for t, c in a_cache.items():
        assert pd.Timestamp(c["dates"][0]) >= period_v2.START and pd.Timestamp(c["dates"][0]) < pd.Timestamp("2014-09-05"), t
        # A выбирает топ-10 по данным до 1 января года и требует WARMUP = 200 баров (approach_a.year): от 26.08.2014 до
        # 1.01.2015 их ~86, поэтому у A нет 2015 (молча, как есть в движке); 2015 есть у Sk с фиксированной точкой.
        assert set(c["A"]["S1"]) == {2016, 2017, 2018, 2019} and c["C"] == {}
    ok = period_v2.valid_vec("LSRG", a_cache["LSRG"]["dates"])
    assert (~ok).any()
    for sid, byY in a_cache["LSRG"]["A"].items():
        for Y, pos in byY.items():
            bad = np.flatnonzero(~ok)
            bad = bad[bad > 0]
            assert np.array_equal(pos[bad], pos[bad - 1]), (sid, Y)


@real
@pytest.mark.parametrize("comm", (0.0004, 0.00035))
def test_v2_A_ravna_s9(comm, a_cache):
    for y in GODY:
        pt = dict(id=f"A_{y}", klass="A_ishodnaya", god=y)
        with s9.period_ctx(PER):
            r = ekviv_s9.run_a(pt, a_cache, (comm,), PER)[0]
        assert r["max_dpnl"] <= TOL and r["sharpe_s9"] != 0, (y, comm, r["max_dpnl"])
