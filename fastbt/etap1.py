"""Движок этапа 1 (К4, Ж-047): входы, ключ кэша (Д5), сборка из данных, фабрика для драйвера К5.

evaluate(params, period) → строка v1.0 (fastbt/etap1_eval.py); `fabrika(sid)` → (evaluate, s9_check) для
`python -m fastbt.etap1_poisk --fabrika fastbt.etap1:fabrika`. Только 2017…2019 (train); validation не открывается.
Расхождения с контрактом §1: `evaluate_e1(...)` заменён методом `Evaluator.evaluate(point, period)` (K5 так просит);
`sdelok_y` — словарь по годам (К5), не число.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fastbt import cache, dividendy, ekviv_s9, zona
from fastbt.etap1_data import LIMIT, START, START_V1, START_Z2, align_f, build_days, load_events, load_rezhim, load_sigma
from fastbt import period_v2
from fastbt.etap1_eval import PERIODS, YAKOR, Evaluator, years_of

END_TRAIN = pd.Timestamp("2020-01-01")                 # окно train — до 2019-12-31; строка b−1 берёт close 2020-01-09 (Н5)


def _flat(x):
    """Всё, что влияет на результат, → список «частей» для cache.key."""
    if isinstance(x, dict):
        return [p for k in sorted(x, key=str) for p in ([str(k)] + _flat(x[k]))]
    if isinstance(x, (list, tuple)):
        return [p for v in x for p in _flat(v)]
    if isinstance(x, pd.Series):
        return [np.asarray(x.to_numpy()), np.asarray(x.index.astype("int64"))] if x.dtype != object else [str(x.tolist())]
    if hasattr(x, "o") and hasattr(x, "pusto"):                     # Day
        out = [np.stack([x.o, x.h, x.l, x.c]) if len(x.c) else np.empty(0), bool(x.pusto)]
        return out + ([np.ascontiguousarray(x.t)] if getattr(x, "t", None) is not None else [])   # Д5: время баров в ключе
    if isinstance(x, np.ndarray):
        return [np.ascontiguousarray(x)]
    return [x]


@dataclass
class E1Inputs:
    sk: object                       # SkEvaluator (сигналы Sk) или None
    sigma: dict                      # {имя: {тикер: σ по датам тикера s0…i1}}
    bars: dict                       # {тикер: [Day]}
    close_n6: dict                   # {тикер: close Н6 по датам}
    rezhim: dict                     # {имя: флаг входа}
    sobytiya: dict                   # {имя: {тикер: bool «вход запрещён»}}
    tick: dict                       # {тикер: шаг цены}
    rate: pd.Series
    validation_open: bool = False
    meta: dict = field(default_factory=dict)      # {тикер: dates, s0, i1, a18}
    div: dict = field(default_factory=dict)       # {тикер: полный D (руб.) на день гэпа по датам s0…i1}, Ж-057 п. 1
    commission: float = 0.0004                    # Д5: в digest; 0,04 % — умолчание, хэш прежний

    def digest(self) -> str:
        """Д5: хэш всех входов (σ, бары, close, режим, события, тик, ставка, даты, дивиденды, ряды Sk) и кода."""
        parts = _flat([self.sigma, self.bars, self.close_n6, self.rezhim, self.sobytiya, self.tick,
                       self.rate, {t: m["dates"].astype("int64") for t, m in self.meta.items()},
                       self.div, float(dividendy.DIV_NALOG)] + ([] if self.commission == 0.0004 else ["komissiya", self.commission]))
        return cache.key("etap1_inputs", *parts, self.sk.series_digest() if self.sk is not None else "")


def load_ticker_nf(ticker: str):
    """`data.load_ticker` без прогнозов σ: дневной ряд Н6 и всё; vpred_aligned.parquet не открывается (Ж-047 п. 1)."""
    from fastbt import data
    from fastbt.signals.interface import TickerData
    x = data.daily_df()
    x = x[x["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    c, h, l, o, v = [x[k].values.astype(np.float64) for k in ("close", "high", "low", "open", "volume")]
    lr = np.zeros(len(c))
    lr[:-1] = np.log(c[1:] / np.maximum(c[:-1], 1e-12))
    dates = pd.to_datetime(x["date"].values).values
    lr = dividendy.adjust_lr(ticker, dates, c, lr)      # Ж-057 п. 1: тот же модуль, что в data.load_ticker
    o, h, l, c = dividendy.adjust_ohlc(ticker, dates, c, o, h, l, c)   # Ж-059 п. 2: сигналы — на ряду с поправкой
    return TickerData(ticker, dates, o, h, l, c, v, lr, {})


def real_close(ticker: str) -> np.ndarray:
    """Реальное (не скорректированное) close Н6 тикера: исполнение идёт по реальным ценам (Ж-059 п. 2)."""
    from fastbt import data
    x = data.daily_df()
    return x[x["ticker"] == ticker].sort_values("date")["close"].values.astype(np.float64)


def sk_evaluator(sid: str):
    """SkEvaluator без прогнозных рядов: тот же класс, `load_ticker` заменён на `load_ticker_nf` на время сборки."""
    from fastbt import sweep_sk
    old = sweep_sk.load_ticker
    sweep_sk.load_ticker = load_ticker_nf
    try:
        return sweep_sk.SkEvaluator(sid)
    finally:
        sweep_sk.load_ticker = old


def window_idx(dates, start, y0: int, end) -> tuple:
    """(s0, a0, i1): начало прогрева, первый день метрик (1 января y0), конец окна. a0 — граница ex-ante для h (❓1)."""
    f = lambda d: int(np.searchsorted(dates, np.datetime64(d)))      # noqa: E731
    return f(start), f(f"{y0}-01-01"), min(f(end), len(dates) - 1)    # ряд кончается раньше end (wf: 02-27 при 02-28) — i1 = его последний день


def _konec_z2(ev) -> pd.Timestamp:
    """Последний день ряда зоны Z2 — общий у всех тикеров (иначе ошибка), строго < 2025-01-01."""
    last = {pd.Timestamp(ev._td[t][0].dates.max()) for t in ev.tickers}
    if len(last) != 1:
        raise ValueError(f"тест-1: ряды тикеров кончаются в разные дни: {sorted(last)}")
    zona.check(next(iter(last)), "тест-1")
    return last.pop()


def build_inputs(sid: str, ev=None, end: pd.Timestamp = END_TRAIN, period: str = "train") -> E1Inputs:
    """period: «train» (2018–2019, прогрев с 2017) или «train4» (В1: 2016–2019, прогрев с 2015)."""
    if period == "test1":                                   # реверс раунд 1: зона Z2, окно 2022–2024, прогрев 2021
        with zona.zone_ctx("test1"):
            return _build(sid, ev, None, period)
    if period == "wf":                                      # walk-forward (Ж-136): до отметки владельца — отказ; конец — 2026-02-28
        period_v2.proverit_chtenie("wf")
        return _build(sid, ev, period_v2.PERIODY["wf"].konec + pd.Timedelta(days=1), period)
    if end >= LIMIT or end > END_TRAIN:
        raise PermissionError("этап 1: только окна до 2019-12-31; validation открывает дирижёр")
    return _build(sid, ev, end, period)


def _build(sid: str, ev, end, period: str) -> E1Inputs:
    """end=None (test1): окно кончается последним днём ряда зоны Z2 (< 2025-01-01); он же — строка Н5 (закрытия 2025 нет)."""
    from fastbt import msp
    y0, y1 = PERIODS[period]
    start = {"train": START, "train4": START_V1, "v2": period_v2.START, "test1": START_Z2,
             "wf": period_v2.PERIODY["wf"].progrev}[period]
    ev = ev or sk_evaluator(sid)
    if end is None:
        end = _konec_z2(ev)
    meta, bars, close, tick, div = {}, {}, {}, {}, {}
    for t in ev.tickers:
        td = ev._td[t][0]
        s0, a18, i1 = window_idx(td.dates, start, y0, end)           # ключ a18 = начало метрик (2018; в В1 — 2016)
        meta[t] = dict(dates=td.dates[s0:i1 + 1], s0=s0, i1=i1, a18=a18)
        rc = real_close(t)                                  # исполнение — по реальным ценам; td.close скорректирован
        close[t] = rc[s0:i1 + 1].astype(float)
        div[t] = dividendy.gap_divs(t, td.dates)[s0:i1 + 1]
        bars[t], tick[t] = build_days(t, td.dates, rc, s0, i1, end, start=start, last_year=y1)
    sg = load_sigma(ev.tickers, end)
    sigma = {n: {t: align_f(sg[n][t], meta[t]["dates"]) for t in ev.tickers} for n in sg}
    ev_ = load_events({t: pd.DatetimeIndex(meta[t]["dates"]) for t in ev.tickers}, end)
    return E1Inputs(ev, sigma, bars, close, load_rezhim(end), ev_, tick, msp.rate_cbr(), meta=meta, div=div)


def base_point(sid: str) -> dict:
    """Плоские параметры центра сетки: ekviv_20.yaml#S{k}_a (поле fiks в zh047_S{k}.yaml)."""
    return next(p["p"] for p in ekviv_s9.load_points()["tochki"] if p["id"] == f"{sid}_a")


def anchor_s9_diff(sid: str, evr: Evaluator, comm: float = 0.0004, period: str = "train") -> dict:
    """Якорь: макс |Δ excess| режима «якорь» против SkEvaluator.pnl (= ekviv_s9.run_sk), годы периода
    (train: 2018 и 2019; train4: 2016–2019 одним окном с прогревом 2015)."""
    ev, p = evr.sk, base_point(sid)
    sp = {k: v for k, v in p.items() if k in ev.sig_names}
    sigs = ev.signals(sp)
    out = {}
    pairs = ((2018, "tune2018"), (2019, "train")) if period == "train" else \
        tuple((g, period) for g in years_of(period))
    for god, per in pairs:
        if period == "v2" and god < 2016:      # Ж-113 п. 6: сверка на пересечении 2016–2019 (эталон SkEvaluator — с начала ряда)
            continue
        end = PERIODS[per][1] + 1 if period != "train" else god + 1
        kw = {} if period == "train" else dict(start_year=PERIODS[per][0])      # train — прежний START_YEAR
        ref = ev.pnl(sigs, p, comm, pd.Timestamp(f"{end}-01-01"), **kw)["excess"]
        mine = evr.excess(sp, per)[0]["excess"]
        m = ref.index.year == god
        ref, mine = ref[m], mine[mine.index.year == god]
        out[god] = float(np.abs(mine.to_numpy() - ref.to_numpy()).max()) if mine.index.equals(ref.index) else float("inf")
    return out


def fabrika(sid: str, commission: float = 0.0004, period: str = "train"):
    """sid → (evaluate(params, period), s9_check(point)); s9_check — якорь ≤ 1e-12 к s9 (считается один раз на sid).
    period — окно входов и якоря (train4 — В1)."""
    inp = build_inputs(sid, period=period)
    inp.commission = commission
    evr = Evaluator(sid, inp, base_point(sid), commission)
    state = {}

    def evaluate(params: dict, period: str) -> dict:
        return evr.evaluate({**YAKOR, **params}, period)
    evaluate.comm = commission                                    # Ledger(evaluate) берёт комиссию в ключ (Д5)

    def s9_check(point: dict) -> bool:
        if "ok" not in state:
            d = anchor_s9_diff(sid, evr, commission, period)
            state["ok"] = bool(max(d.values()) <= 1e-12)
        return state["ok"]
    return evaluate, s9_check
