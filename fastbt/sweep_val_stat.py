"""Статистика кандидата на validation (Ж-031 Э1-д, Ж-037): Δ Шарпа, ЛВ, к2, PBO, DSR(N), ворота. Данных сама не читает.

Кандидат (`Cand`) — пара «версия / исходная» на одном формате: `Sweep` с `AEvaluator` над панелью пар и параметры сборки.
Для стратегии — панель одной стратегии и сборка (равные, net, net); для A′ — панель шести точек и сборка из топ-2.
Ряды validation берутся из `excess_d` строки оценщика (`Sweep.evaluate(…, "validation")`, после `open_validation`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from fastbt import fast_kontrol as fk
from fastbt import metriki_e as me
from fastbt import sweep_drivers as sd
from fastbt import sweep_pbo
from fastbt import sweep_stat as ss
from fastbt.s9_stats import lw_sharpe_diff
from fastbt.sweep_a_core import assemble

SEED, REPS_LW, REPS_K2, COMM, LAG_MIN = 20261027, 10000, 1000, 0.04, 21
END = pd.Timestamp("2022-01-01")
YEARS = (2020, 2021)
GATES = dict(pbo=0.2, dsr=0.9, plato=3, p_lw=0.05, p_k2=0.05)


@dataclass
class Cand:
    id: str
    kind: str                                   # "S" (стратегия) | "A" (сборка)
    sw: object                                  # Sweep версии; orig_sw — исходной (для A — над исходными точками)
    params: dict
    orig_sw: object
    orig_params: dict
    neighbors: int                              # число соседей плато (из таблицы итераций)
    trials: list = field(default_factory=list)  # SR проб накопленного ledger (N для DSR)
    excess: dict = field(default_factory=dict)  # ledger рядов проб (PBO)
    n_total: int = 0                            # >0: N в DSR задан отдельно (N_vsego), var_sr — по trials (толкование 27.10 20:20)


def dsr_n(sr: float, trials: list, n: int, T: int, skew: float, kurt: float) -> dict:
    """DSR: var_sr — по уникальным новым пробам (порядок не влияет: сортировка и fsum), N задан отдельно. Дополнения списка нет."""
    t = sorted(trials)
    m = math.fsum(t) / len(t)
    var = math.fsum((x - m) ** 2 for x in t) / (len(t) - 1) if len(t) > 1 else 0.0
    star = sd.expected_max_sharpe(n, var)
    return dict(n_trials=n, var_sr=var, sr_star=star, dsr=sd.psr(sr, star, T, skew, kurt))


def delta_sharpe(rc: dict, ro: dict) -> float:
    """Δ = разность табличных Шарпов (ddof=1), не Δ бутстрапа ЛВ (одно определение метрики, Д7)."""
    return float(rc["sharpe"] - ro["sharpe"])


def shift_panel(pn, lags: dict):
    """к2: круговой сдвиг сигналов (позиции и сила) каждого тикера на свой лаг; доходности на месте. Лаг 0 — исходный ряд."""
    tick = pn.tick if pn.tick is not None else np.arange(pn.pos.shape[1])
    pos, st = pn.pos.copy(), pn.strength.copy()
    for t, lag in lags.items():
        m = tick == t
        pos[:, m], st[:, m] = np.roll(pn.pos[:, m], lag, axis=0), np.roll(pn.strength[:, m], lag, axis=0)
    return replace(pn, pos=pos * pn.act, strength=st)


def k2_sharpes(src, params: dict, reps: int = REPS_K2, seed: int = SEED, lags=None) -> np.ndarray:
    """Шарпы на validation при перемешанных сигналах: лаг каждого тикера ∈ [21, T−21], reps розыгрышей."""
    pn = src.panel(END)
    T = len(pn.dates)
    tick = np.unique(pn.tick if pn.tick is not None else np.arange(pn.pos.shape[1]))
    rng, wc, out = np.random.default_rng(seed), {}, []
    for i in range(reps):
        lag = lags[i] if lags is not None else {t: int(rng.integers(LAG_MIN, T - LAG_MIN + 1)) for t in tick}
        df = assemble(shift_panel(pn, lag), params, COMM / 100, wc)
        out.append(float(me.sharpe(ss.year_slice(df, YEARS)["excess"].values)))
    return np.array(out)


def evaluate_candidate(c: Cand, reps_lw: int = REPS_LW, reps_k2: int = REPS_K2) -> dict:
    rc = c.sw.evaluate(c.params, "validation", COMM)
    ro = c.orig_sw.evaluate(c.orig_params, "validation", COMM)
    if rc["daty"] != ro["daty"]:
        raise ValueError(f"{c.id}: ряды версии и исходной на разных датах")
    delta = delta_sharpe(rc, ro)
    same = np.array_equal(rc["excess_d"], ro["excess_d"])        # идентичные ряды: Δ = 0, разницы нет — p = 1 (ЛВ вырождается)
    p_lw = 1.0 if same else lw_sharpe_diff(rc["excess_d"], ro["excess_d"], reps_lw, 5, SEED)[2]
    p_k2 = fk.p_perm(rc["sharpe"], k2_sharpes(c.sw.ev.src, c.params, reps_k2))
    mult = c.sw.n_mult()
    dsr = (dsr_n(rc["sr_d"], c.trials, c.n_total * mult, rc["T"], rc["skew"], rc["kurt"]) if c.n_total
           else sd.dsr(rc["sr_d"], list(c.trials) * mult, rc["T"], rc["skew"], rc["kurt"]))
    pbo = sweep_pbo.pbo(c.excess)["pbo"] if c.excess else float("nan")
    ok = dict(delta=delta > 0, pbo=pbo < GATES["pbo"], dsr=dsr["dsr"] >= GATES["dsr"], plato=c.neighbors >= GATES["plato"],
              p_lw=p_lw < GATES["p_lw"], p_k2=p_k2 < GATES["p_k2"])      # nan в сравнении -> False: не прошло
    return dict(id=c.id, kind=c.kind, sharpe=rc["sharpe"], sharpe_orig=ro["sharpe"], delta=delta, p_lw=p_lw, p_k2=p_k2,
                pbo=pbo, dsr=dsr["dsr"], N=dsr["n_trials"], n_mult=mult, plato=c.neighbors, proshel=all(ok.values()),
                **{f"ok_{k}": v for k, v in ok.items()})
