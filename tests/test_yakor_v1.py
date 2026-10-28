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
        assert np.abs(r.to_numpy()).max() > 0, f"{sid} {y}: ряд s9 нулевой — сверка тривиальна"
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


def test_spisok_tochek_18_sk_po_3_na_strategiyu():
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
    monkeypatch.setattr(etap1, "START_V1", pd.Timestamp("2017-01-01"))      # окно оценщика не покрывает 2016
    evr = make_evr("S1", ev=ev)
    monkeypatch.undo()
    try:
        d = deltas(evr, "S1", POINTS[0][1])
    except ValueError:                       # оценщик на смещённом окне ломается громко (длины рядов) — сверка не проходит
        return
    assert max(d.values()) > TOL, d


def test_mutation_progrev_s9_ne_2015_caught(monkeypatch):
    """Прогрев s9 (VAL_START train4) = 2016: окно валидации 2016 пустое, боевой путь отказывает, а не считает молча."""
    from fastbt import s9
    monkeypatch.setitem(s9.PERIODS_S9, "train4", ((2016, 2017, 2018, 2019), 2016))
    with pytest.raises(ValueError, match="пустое окно"):
        deltas(make_evr("S1"), "S1", POINTS[0][1])


# ---------------- A на train4 (Ж-075 п. 1, «20 точек» Ж-040): контракт ветки s9-a-train4 ----------------
# Две точки A (портфель «все семейства», комиссия 0,04 и 0,035 %) × годы 2016–2019: Engine.port на боевом кэше A (train4)
# против сборки оценщика A (`sweep_a_core.assemble`, ekviv_s9.run_a), max |Δ| ≤ 1e-12; плюс lr кэша = `etap1.load_ticker_nf`.
# Контракт кода: при set_period("train4") период входит в cache_key(), кэш A лежит в cache_<key>.pkl, годы A — с 2015 (прогрев).
COMMS_A = (0.0004, 0.00035)


def _key(period: str):
    from fastbt import s9
    old = s9.PERIOD, s9.YEARS, s9.VAL_START
    try:
        s9.set_period(period)
        return s9.cache_key(), s9.OUT
    finally:
        s9.PERIOD, s9.YEARS, s9.VAL_START = old


def a4_cache() -> dict:
    from fastbt import s9
    with s9.period_ctx("train4"):                       # указатель cache_latest_train4.txt (s9-a-train4)
        res = s9.load_cache()
    return {t: dict(dates=c["dates"], lr=c["lr"], A={s: dict(v) for s, v in c["A"].items()}) for t, c in res.items()}


def warmup_ok(cache: dict, val_start: int) -> bool:
    """Годы A кэша — ровно 2016–2019, начало окна валидации (прогрев) — 2015."""
    ys = {Y for c in cache.values() for v in c["A"].values() for Y in v}
    return ys == set(GODY) and val_start == 2015


def lr_delta(cache_lr: dict, ref_lr: dict, y: int) -> float:
    """max |lr кэша − lr эталона| на датах года y по всем тикерам (даты и длины обязаны совпасть)."""
    out = 0.0
    for t, (dates, lr) in cache_lr.items():
        rd, rl = ref_lr[t]
        if len(rl) != len(lr) or not np.array_equal(rd, dates):
            return float("inf")
        m = pd.DatetimeIndex(dates).year == y
        out = max(out, float(np.abs(lr[m] - rl[m]).max()))
    return out


def ref_lr_nf(tickers) -> dict:
    from fastbt import etap1
    return {t: (td.dates, td.log_ret) for t in tickers for td in [etap1.load_ticker_nf(t)]}


@pytest.mark.parametrize("comm", COMMS_A)
def test_a_train4_equals_s9(comm):
    from fastbt import ekviv_s9
    cache = a4_cache()
    from fastbt import s9
    assert warmup_ok(cache, s9.PERIODS_S9[PER][1])
    ref = ref_lr_nf(cache)
    for y in GODY:
        pt = dict(id=f"A_{y}", klass="A_ishodnaya", god=y)
        old = ekviv_s9.s9.PERIOD, ekviv_s9.s9.YEARS, ekviv_s9.s9.VAL_START
        try:
            ekviv_s9.s9.set_period(PER)
            r = ekviv_s9.run_a(pt, cache, (comm,), PER)[0]
        finally:
            ekviv_s9.s9.PERIOD, ekviv_s9.s9.YEARS, ekviv_s9.s9.VAL_START = old
        dl = lr_delta({t: (c["dates"], c["lr"]) for t, c in cache.items()}, ref, y)
        assert r["max_dpnl"] <= TOL and r["sharpe_s9"] != 0 and dl <= TOL, (y, comm, r["max_dpnl"], dl)


def test_a_train4_period_v_kluche_kesha():
    """Кэш A по ключу `train` при запросе train4 недопустим: период обязан входить в cache_key."""
    assert _key("train4")[0] != _key("train")[0]


def test_mutation_a_bez_dividendov_2015_2016_caught(monkeypatch):
    """Эталон lr без дивидендов до 2017: lr_delta 2016 обязана превысить допуск (кэш A — с дивидендами)."""
    from fastbt import dividendy
    good = ref_lr_nf(("SBER", "LKOH"))
    orig = dividendy.adjust_lr

    def bad(t, dates, c, lr):
        out = np.array(orig(t, dates, c, lr), float)
        keep = np.asarray(dates) < np.datetime64("2017-01-01")
        out[keep] = lr[keep]
        return out
    monkeypatch.setattr(dividendy, "adjust_lr", bad)
    mut = ref_lr_nf(("SBER", "LKOH"))
    assert lr_delta(good, mut, 2016) > TOL, "мутация не ловится"
    assert lr_delta(good, good, 2016) == 0.0


# ---------------- Ж-080: anchor_s9_diff(period="train4") конечен и ≤ 1e-12 (S1–S6; A — тест выше) ----------------
def _anchor_d(sid: str) -> dict:
    from fastbt import etap1
    return etap1.anchor_s9_diff(sid, make_evr(sid), COMM, PER)


@pytest.mark.parametrize("sid", [f"S{k}" for k in range(1, 7)])
def test_anchor_s9_diff_train4_konechen(sid):
    d = _anchor_d(sid)
    assert set(d) == set(GODY) and all(np.isfinite(v) and v <= TOL for v in d.values()), d


def test_mutation_start_year_2018_konstantoy_caught(monkeypatch):
    """START_YEAR = 2018 константой: ряд эталона 2016–2017 пуст, anchor_s9_diff даёт inf — проверка обязана это видеть."""
    from fastbt import sweep_sk
    orig = sweep_sk.SkEvaluator.pnl
    monkeypatch.setattr(sweep_sk.SkEvaluator, "pnl", lambda self, sigs, p, comm, end, start_year=None: orig(self, sigs, p, comm, end))
    d = _anchor_d("S1")
    assert not (np.isfinite(d[2016]) and np.isfinite(d[2017])), d


def test_mutation_a_progrev_ne_2015_caught():
    """Прогрев не 2015 (val_start 2016) или лишний/пропавший год в кэше A — warmup_ok обязан отвергнуть."""
    mk = lambda ys: {"T": {"A": {"S1": {Y: np.zeros(3) for Y in ys}}}}      # noqa: E731
    assert warmup_ok(mk(GODY), 2015)
    assert not warmup_ok(mk(GODY), 2016) and not warmup_ok(mk(range(2017, 2020)), 2015)
