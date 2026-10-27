"""Оценщик семей C и E под контракт sweep_evaluate v1.1: evaluate(obj, params, period, commission) → row.

obj: {kind: "E"|"C", sigma: имя ряда σ̂, u: "all"|"top10"}; params E: q, k, f, a, b, tau, s; C: W, p | (p1, p2), star.
База — текущий A (не A′): позиции пар (тикер, sid) в Inputs. E берёт длинную сторону A (sig = pos > 0, ↺ шорт не
исполняется: §2 описывает только покупку) и считает exec_e по 5-мин барам; C — A × затвор g по рангу σ̂ (как s9).
Портфель — EW по парам, ставка с днями [t, след. торговый день) (msp.cal_days), без bfill. Период: tune2018,
ctrl2019, train, validation (только inp.validation_open = True); даты ≥ 2022 — PermissionError. Прогрев читает
историю до периода, PnL — только дни периода. Кэшируются только маски сигнала, ключ = fastbt.cache.key (Д5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fastbt import cache, metriki_e as me, msp
from fastbt.exec_e import Day, Params, q_mask, run

PERIODS = {"tune2018": (2018, 2018), "ctrl2019": (2019, 2019), "train": (2018, 2019), "validation": (2020, 2021)}
ALIAS = {"setup": "tune2018", "control": "ctrl2019"}
HONEST = {"tune2018": "net", "ctrl2019": "kontrol", "train": "net", "validation": "validation"}
LAST = pd.Timestamp("2021-12-31")
MINP = {126: 50, 252: 50, 504: 252}
SIG_KEYS = ("q",)                              # параметры, меняющие маску сигнала (входят в ключ кэша масок)
_MASKS: dict = {}


@dataclass
class Pair:
    ticker: str
    sid: str
    cat: int                                   # 0 тренд, 1 контртренд, 2 диапазон (s9.CATS)
    dates: pd.DatetimeIndex
    pos: np.ndarray                            # позиция A ∈ {−1, 0, 1} на close t
    lr: np.ndarray                             # ln(close t+1 / close t), для C
    strength: np.ndarray | None = None
    days: list | None = None                   # 5-мин бары по торговым дням (Day), для E


@dataclass
class Inputs:
    pairs: list
    sigma: dict                                # имя → {тикер: ряд по датам тикера}
    rate: pd.Series                            # ключевая ставка, % годовых/100, индекс — даты
    top10: dict = field(default_factory=dict)  # год → список тикеров (оборот прошлого года)
    rvw: dict | None = None                    # √RV_w по тикерам, для C*
    validation_open: bool = False

    def digest(self) -> str:
        """Хэш всех входов: позиции, бары, сила, σ̂, ставка, состав top10."""
        parts = [p.dates.values.astype("int64") for p in self.pairs] + [p.pos for p in self.pairs]
        parts += [p.strength for p in self.pairs if p.strength is not None]
        parts += [np.concatenate([np.stack([d.o, d.h, d.l, d.c]).ravel() for d in p.days]) for p in self.pairs if p.days]
        for n in sorted(self.sigma):
            for t in sorted(self.sigma[n]):
                s = self.sigma[n][t]
                parts += [np.asarray(s, float)] + ([np.asarray(s.index.astype(str), "U40")] if isinstance(s, pd.Series) else [])
        return cache.key("sem_e_inputs", *parts, self.rate.values, {y: sorted(v) for y, v in self.top10.items()},
                         [(p.ticker, p.sid) for p in self.pairs])


def cache_key(obj: dict, params: dict, inp: Inputs, comm: float | None = None) -> str:
    """Д5: хэш объекта, параметров, всех входов и кода (cache.key включает code_hash)."""
    return cache.key("sem_e", obj, sorted(params.items()), inp.digest(), comm)


def _window(period: str, inp: Inputs):
    period = ALIAS.get(period, period)
    if period not in PERIODS:
        raise ValueError(f"период {period!r}: допустимо {sorted(PERIODS)}")
    y0, y1 = PERIODS[period]
    if y1 >= 2022:
        raise PermissionError("даты ≥ 2022 не читаются")
    if period == "validation" and not inp.validation_open:
        raise PermissionError("validation открывает только дирижёр (inp.validation_open)")
    if any(p.dates.max() > LAST for p in inp.pairs):
        raise PermissionError("во входах есть даты ≥ 2022")
    return period, pd.Timestamp(y0, 1, 1), pd.Timestamp(y1, 12, 31)


def _rf_cal(dates, rate) -> np.ndarray:
    r = rate.reindex(rate.index.union(dates)).sort_index().ffill().reindex(dates)
    if r.isna().any():
        raise ValueError("ставка не покрывает начало периода (bfill запрещён, Д4)")
    return r.values / 365.0 * msp.cal_days(dates)


def _u_mask(p: Pair, inp: Inputs, u: str) -> np.ndarray:
    if u == "all":
        return np.ones(len(p.dates), int)
    return np.array([p.ticker in inp.top10.get(y - 1, ()) for y in p.dates.year], int)   # оборот прошлого года


def _sig_e(p: Pair, inp: Inputs, obj: dict, q: float) -> np.ndarray:
    sig = (p.pos > 0).astype(int) * _u_mask(p, inp, obj.get("u", "all"))
    if q == 0:
        return sig
    if p.strength is None:
        raise ValueError(f"{p.ticker}/{p.sid}: нет силы сигнала, q > 0 невозможна (❓ определение силы для ансамбля A)")
    k = cache.key("sem_e_qmask", sig, p.strength, q)
    if k not in _MASKS:
        _MASKS[k] = q_mask(p.strength, sig, q)
    return _MASKS[k]


def pair_e(p: Pair, inp: Inputs, obj: dict, par: dict, comm: float, end, t0=None):
    """Одна пара: длинная сторона A — через exec_e (σ-правила E), короткая — как в A (close→close, без правил E:
    толкование Ж-036 T2 п. 3). → (избыточная серия, позиция ±1 по дням, число сторон в окне [t0; end])."""
    pr = Params(**{k: par[k] for k in ("k", "f", "a", "b", "tau", "s") if k in par}, comm=comm / 100)
    n = int(p.dates.searchsorted(end, side="right"))
    d, rf = p.dates[:n], _rf_cal(p.dates[:n], inp.rate)
    r = run(p.ticker, p.days[:n], _sig_e(p, inp, obj, par.get("q", 0.0))[:n],
            inp.sigma[obj["sigma"]][p.ticker][:n], pr, rf_cal=rf)
    sh = -1.0 * (p.pos[:n] < 0) * _u_mask(p, inp, obj.get("u", "all"))[:n]   # короткая позиция A: −1 (подвыборка u — та же)
    sides = np.abs(np.diff(sh, prepend=0.0))
    sides[-1] += abs(sh[-1])                                   # Н5: выход на конце окна
    ex = r.pnl + r.fdr - rf + sh * p.lr[:n] - sides * pr.comm - np.abs(sh) * rf
    m = d >= (d[0] if t0 is None else t0)
    cnt = sides[m].sum() + sum((d[t.entry_day] >= d[m][0]) + (t.exit_day is None or d[t.exit_day] >= d[m][0])
                               for t in r.trades) if m.any() else 0.0
    return pd.Series(ex, d), r.pos + sh, float(cnt)


def _excess_e(inp: Inputs, obj: dict, par: dict, comm: float, t0, end):
    cols, expo, sd = {}, {}, 0.0
    for p in inp.pairs:
        ex, pos, c = pair_e(p, inp, obj, par, comm, end, t0)
        cols[(p.ticker, p.sid)], expo[(p.ticker, p.sid)] = ex, pd.Series(np.abs(pos), ex.index)
        sd += c
    return pd.DataFrame(cols).mean(axis=1), pd.DataFrame(expo).mean(axis=1), sd, sd   # сторона = единица позиции


def _gate(sg: np.ndarray, par: dict, cat: int) -> np.ndarray:
    from fastbt.s9 import _c_gate
    from fastbt.s9_stats import past_rank
    R = past_rank(sg.copy(), par["W"], MINP[par["W"]])
    return _c_gate(R[:, None], (par["p1"], par["p2"]) if "p1" in par else par["p"])[cat][:, 0]


def _excess_c(inp: Inputs, obj: dict, par: dict, comm: float, t0, end):
    rows, ob, sd = [], 0.0, 0.0
    for p in inp.pairs:
        n = int(p.dates.searchsorted(end, side="right"))
        sg = inp.sigma[obj["sigma"]][p.ticker][:n]
        sg = sg / inp.rvw[p.ticker][:n] if par.get("star") else sg
        pos = p.pos[:n] * _gate(sg, par, p.cat)
        dp = np.abs(np.diff(pos, prepend=0.0))
        dp[-1] += abs(pos[-1])
        w = dp[p.dates[:n] >= t0]                              # прогрев в счёт не входит
        ob, sd = ob + w.sum(), sd + (w > 0).sum()
        rows.append(pd.DataFrame(dict(date=p.dates[:n], ticker=p.ticker, strategy=p.sid, position=pos,
                                      daily_gross_return=pos * p.lr[:n])))
    port = msp.portfolio(pd.concat(rows), inp.rate, comm / 100)
    return port["excess"], port["exposure"], ob, sd


def evaluate(obj: dict, params: dict, period: str, commission: float, inp: Inputs) -> dict:
    """commission — % на сторону (0,04 / 0,035). Возвращает строку контракта v1.1."""
    period, t0, t1 = _window(period, inp)
    f = {"E": _excess_e, "C": _excess_c}.get(obj["kind"])
    if f is None:
        raise ValueError(f"kind {obj['kind']!r}: допустимо E, C")
    ex, expo, ob, sd = f(inp, obj, params, commission, t0, t1)
    ex, expo = ex[(ex.index >= t0) & (ex.index <= t1)], expo[(expo.index >= t0) & (expo.index <= t1)]
    if ex.empty or not np.isfinite(ex.values).all():
        raise ValueError("пустой или нечисловой ряд избыточной доходности")
    x, yrs = ex.values, ex.index.year
    yr = (t1.year - t0.year + 1)
    return {"v": "1.1", "sharpe": me.sr_dnevnoy(x) * np.sqrt(252),
            "by_year": {int(y): me.sr_dnevnoy(x[yrs == y]) * np.sqrt(252) for y in sorted(set(yrs))},
            "sr_d": me.sr_dnevnoy(x), "T": len(x), "skew": me.asimmetriya(x), "kurt": me.ekscess(x) + 3.0,
            "msr": me.msr(x), "sortino": me.sortino_e(x), "mdd": me.mdd(x),
            "oborot_y": ob / yr / len(inp.pairs), "dni_v_rynke": float((expo.values > 0).mean()),
            "sdelok_y": sd / yr / len(inp.pairs), "period": period, "okno_s": str(t0.date()),
            "okno_po": str(t1.date()), "chestnaya": HONEST[period], "comm_pct": commission, "excess_d": list(x)}


def point_params(point: dict) -> dict:
    """Точка сетки (оси C_setka_e2 / E_setka_e2.yaml) → params оценщика: p в % → доля, pary → p1, p2, ab → a = b.
    Ось sigma — не параметр, а obj['sigma'] (её берёт вызывающий)."""
    out = {k: v for k, v in point.items() if k not in ("sigma", "pary", "ab", "p")}
    if "p" in point:
        out["p"] = point["p"] / 100
    if "pary" in point:
        out["p1"], out["p2"] = (x / 100 for x in point["pary"])
    if "ab" in point:
        out["a"] = out["b"] = point["ab"]
    return out


def make_evaluate(obj: dict, inp: Inputs, commission: float):
    """Замыкание (params, period) → row для oat_sweep / factorial_sweep (sweep_drivers)."""
    return lambda params, period: evaluate(obj, params, period, commission, inp)
