"""Ж-075 п. 1 (К-як): оценщик этапа 1 на train4 (В1: 2016–2019, прогрев 2015) против боевого s9 на РЕАЛЬНЫХ данных.
18 точек Sk (S1–S6 × 3: P1 итерации 1 и две соседние по сетке zh047) × годы 2016, 2017, 2018, 2019 -> 72 сверки,
допуск max |Δ excess| ≤ 1e-12; комиссия 0,04 %. Точка A (боевой ансамбль) на 2016 не считается — см. отчёт (dev.py:26).
Нужны 1-минутки (FASTBT_BARS1M) и данные загрузчиков; нет данных — пропуск. Только чтение.
Мутации (обязаны ронять сверку): etap1 без дивидендов 2015–2016; прогрев не 2015.
python3 -m pytest tests/test_yakor_v1.py -q"""
import numpy as np
import pandas as pd
import pytest

from fastbt import bars5m

pytestmark = pytest.mark.skipif(not bars5m.SRC.exists(), reason="нет 1-минуток D25 (FASTBT_BARS1M)")
TOL, COMM, PER = 1e-12, 0.0004, "train4"
GODY = (2016, 2017, 2018, 2019)
# (sid, сигнальные параметры): P1 — из шапки docs/tables/diag/S{k}_sdelki_posle370.csv, соседи — шаг по сетке zh047_S{k}.yaml
POINTS = [
    ("S1", dict(ma_window=13, z_entry=1.67, z_exit=0.5)), ("S1", dict(ma_window=17, z_entry=1.67, z_exit=0.5)),
    ("S1", dict(ma_window=13, z_entry=2.0, z_exit=0.5)),
    ("S2", dict(bb_window=20, bb_std=1.67)), ("S2", dict(bb_window=17, bb_std=1.67)), ("S2", dict(bb_window=20, bb_std=1.33)),
    ("S3", dict(dc_window=13)), ("S3", dict(dc_window=10)), ("S3", dict(dc_window=17)),
    ("S4", dict(atr_period=14, multiplier=1.5)), ("S4", dict(atr_period=12, multiplier=1.5)),
    ("S4", dict(atr_period=14, multiplier=2.0)),
    ("S5", dict(pivot_period="weekly", pivot_type="fibonacci", filter_set=1)),
    ("S5", dict(pivot_period="weekly", pivot_type="fibonacci", filter_set=2)),
    ("S5", dict(pivot_period="weekly", pivot_type="classic", filter_set=1)),
    ("S6", dict(vwap_window=30, dev_mult=1.5, filter_set=1, dir_bias="OFF")),
    ("S6", dict(vwap_window=27, dev_mult=1.5, filter_set=1, dir_bias="OFF")),
    ("S6", dict(vwap_window=30, dev_mult=1.25, filter_set=1, dir_bias="OFF")),
]
IDS = [f"{s}-{'-'.join(f'{k}={v}' for k, v in p.items())}" for s, p in POINTS]


def deltas(evr, sid: str, sigp: dict) -> dict:
    """{год: max |Δ excess|} оценщик (режим «якорь», сигнал = sigp, остальное — база S{k}_a) против SkEvaluator.pnl (s9)."""
    from fastbt import ekviv_s9, etap1, s9
    ev, p = evr.sk, {**etap1.base_point(sid), **sigp}
    sp = {k: v for k, v in p.items() if k in ev.sig_names}
    assert set(sigp) <= set(sp), f"{sid}: {set(sigp) - set(sp)} не сигнальные параметры"
    sigs = ev.signals(sp)
    # эталон — боевой путь s9: Unit + ApproachA.raw_pos + exec_layer -> Engine.port (SkEvaluator.pnl не годится: START_YEAR=2018)
    bo = {t: ekviv_s9.boevoy_pos(sid, p, t, *ev._td[t]) for t in sigs}
    td = {t: ev._td[t][0] for t in sigs}
    ys = ekviv_s9.windows(PER)[2019]
    old = s9.PERIOD, s9.YEARS, s9.VAL_START
    try:
        s9.set_period(PER)
        eng = ekviv_s9.s9_engine(ekviv_s9.sk_cache(sid, bo, td, ys), ys, PER)
        ref = eng.port(eng.A_row, comm=COMM)["excess"]
    finally:
        s9.PERIOD, s9.YEARS, s9.VAL_START = old
    mine = evr.excess(sp, PER)[0]["excess"]
    out = {}
    for y in GODY:
        r, m = ref[ref.index.year == y], mine[mine.index.year == y]
        assert len(r) > 200, f"{sid} {y}: ряд s9 пуст/короток ({len(r)})"
        out[y] = float(np.abs(m.to_numpy() - r.to_numpy()).max()) if m.index.equals(r.index) else float("inf")
    return out


def make_evr(sid: str, ev=None):
    from fastbt import etap1
    from fastbt.etap1_eval import Evaluator
    inp = etap1.build_inputs(sid, ev=ev, period=PER)
    return Evaluator(sid, inp, etap1.base_point(sid), COMM)


@pytest.fixture(scope="module")
def evrs():
    cache = {}
    return lambda sid: cache.setdefault(sid, make_evr(sid))


@pytest.mark.parametrize("sid,sigp", POINTS, ids=IDS)
def test_train4_anchor_equals_s9(sid, sigp, evrs):
    d = deltas(evrs(sid), sid, sigp)
    assert set(d) == set(GODY) and max(d.values()) <= TOL, d


def test_20_tochek_mutatsii_ne_menyayut_spisok():
    assert len(POINTS) == 18 and len({(s, tuple(sorted(p.items()))) for s, p in POINTS}) == 18
    assert {s for s, _ in POINTS} == {f"S{k}" for k in range(1, 7)}


def test_mutation_etap1_bez_dividendov_2015_2016_caught(monkeypatch):
    """etap1 строит div без дивидендов до 2017-01-01 (s9-сторона собрана до подмены): якорь обязан разойтись."""
    from fastbt import dividendy, etap1
    from fastbt.sweep_sk import SkEvaluator  # noqa: F401
    ev = etap1.sk_evaluator("S1")                       # эталон s9 — с дивидендами
    orig = dividendy.gap_divs

    def bad(t, dates):
        d = orig(t, dates).copy()
        d[np.asarray(dates) < np.datetime64("2017-01-01")] = 0.0
        return d
    monkeypatch.setattr(etap1.dividendy, "gap_divs", bad)
    evr = make_evr("S1", ev=ev)
    monkeypatch.setattr(etap1.dividendy, "gap_divs", orig)
    d = deltas(evr, "S1", POINTS[0][1])
    assert max(d[2016], d[2017]) > TOL, d


def test_mutation_progrev_ne_2015_caught(monkeypatch):
    """Прогрев начинается 2016-01-01, а не 2015: состояние сигналов/выходов на 2016 другое — якорь обязан разойтись."""
    from fastbt import etap1
    ev = etap1.sk_evaluator("S1")
    monkeypatch.setattr(etap1, "START_V1", pd.Timestamp("2016-01-01"))
    evr = make_evr("S1", ev=ev)
    monkeypatch.undo()
    try:
        d = deltas(evr, "S1", POINTS[0][1])
    except AssertionError:
        return                                       # ряд не сошёлся по индексу/длине — тоже падение сверки
    assert max(d.values()) > TOL, d
