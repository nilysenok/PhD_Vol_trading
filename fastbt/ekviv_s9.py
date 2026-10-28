"""Сверка оценщика перебора (`sweep_sk`/`sweep_post`) с боевым путём s9 на 20 фиксированных точках (Ж-040 п. 2).

Боевой путь одной стратегии в точке собран только из боевых функций, ничего в них не менялось:
сигнал `get_family(sid).generate` (общий с оценщиком: у A он же внутри `sigprov`), RM — `Unit._build_rm`,
сделки и позиции — `ApproachA.raw_pos` (`trades_one` + `pos_from_trades`), исполнение — `execlayer.exec_layer` ВСЕГДА
(в A он вызывается и при тривиальных параметрах; оценщик тогда его пропускает), фильтр оборота — `turnover.tfilter`,
счёт — `s9.Engine.port`. `sweep_post` / `sweep_sk` здесь не вызываются.
Нет боевого аналога (класс «нет эталона»): выходы в √RV, маски после кэша сигналов (не равные зашитым порогам), q_sily,
vyhod_dev ≠ 0,5, значения RM, которых Unit не принимает (trail_n вне TREND_TN, комбо пивотов вне PIV_COMBOS).
Точки A: позиции боевой A из кэша s9 → `Engine.port` против `sweep_a_core.assemble`.
"""
from __future__ import annotations

import pickle
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fastbt import s9
from fastbt.approach_a import ApproachA
from fastbt.data import year_bounds
from fastbt.execlayer import exec_layer
from fastbt.sigprov import SIG_GRIDS
from fastbt.signals.interface import get_family
from fastbt.sweep_a_core import Panel, assemble
from fastbt.turnover import tfilter
from fastbt.unit import Unit

YAML = Path(__file__).parent / "designs" / "sweep" / "ekviv_20.yaml"
MASK_BASE = dict(adx_min=20, vf_porog=0.5, obyom_mnozh_sma20=1.0, vol_regime_max=1.2, svechi_min_iz5=3, q_sily=0.0, vyhod_dev=0.5)
BASE_A = {"vesa": "ravnye", "limit_pozitsiy": "net", "limit_na_strategiyu": "net"}
WINDOWS = {2018: (2018,), 2019: (2018, 2019)}      # год сверки -> годы окна (как end у Sweep: 2019-01-01 / 2020-01-01)
PNL_TOL = 1e-12
PIV_KEYS = ["sl_target", "tp_target", "buffer", "max_hold", "breakeven", "cooldown_bars"]
ATR_KEYS = ["sl_mult", "tp_mult", "max_hold", "breakeven_trigger", "cooldown_bars", "time_decay"]
RM_KEYS = {"pivot": PIV_KEYS, "atr": ATR_KEYS, "vwap": ["sl_dev", "max_hold", "breakeven", "cooldown_bars"]}


class NoRef(Exception):
    """У точки нет боевого аналога: причина в тексте."""


def load_points(path=YAML) -> dict:
    return yaml.safe_load(Path(path).read_text("utf-8"))


def rm_row(sid: str, p: dict) -> dict:
    """RM-строка в формате боевого `grids.rm_grid` из плоских параметров точки."""
    if "sqrt_rv" in (p.get("rm_tip"), p.get("rm_edinitsa")):
        raise NoRef("выходы в √RV: боевого аналога в s9 нет")
    for k, base in MASK_BASE.items():
        if k in p and p[k] != base:
            raise NoRef(f"{k}={p[k]}: маска/квантиль/выход S6 после кэша сигналов, в боевом пути нет")
    if sid in ("S1", "S2"):
        return dict(sl_mult=p["sl_mult"], tp_mult=p["tp_mult"], max_hold=p["max_hold"],
                    breakeven_trigger=p.get("breakeven_trigger"), cooldown_bars=p["cooldown_bars"], partial_exit=False)
    if sid in ("S3", "S4"):
        return dict(initial_sl_mult=p["initial_sl_mult"], trail_type=p["trail_type"], trail_n=p["trail_n"],
                    trail_atr_mult=p["trail_atr_mult"], breakeven_thresh=p.get("breakeven_thresh"),
                    cooldown_bars=p["cooldown_bars"], parabolic_step=p.get("rm_parabolic.parabolic_step", 0.02),
                    parabolic_max=p.get("rm_parabolic.parabolic_max", 0.15))
    t = p["rm_tip"]
    return {**{k: p.get(f"rm_{t}.{k}") for k in RM_KEYS[t]}, "_rm_type": t}


def boevoy_pos(sid: str, p: dict, t: str, td, ind) -> np.ndarray:
    """Позиции [n] точки p по тикеру t боевым путём A (полный ряд, end не задан)."""
    row = rm_row(sid, p)
    u = Unit(t, sid, td, ind, get_family(sid).generate(td, ind, [{k: p[k] for k in SIG_GRIDS[sid]}]))
    u.rm = [row]
    try:
        u._build_rm()
    except (ValueError, KeyError, IndexError) as e:
        raise NoRef(f"RM {row} боевой Unit не принимает: {e!r}")
    a = ApproachA.__new__(ApproachA)
    a.u, a._raw = u, {}
    pos = exec_layer(a.raw_pos(0, 0), u.strength[0], int(p["consec_entry"]), int(p["exec_min_hold"]),
                     int(p["exec_cooldown"]), float(p["min_strength"]), u.n)
    if p["tf_min_hold"] > 0 or p["tf_reentry"] > 0:
        pos = tfilter(pos.astype(np.float64), int(p["tf_min_hold"]), int(p["tf_reentry"]))
    return pos


WINDOWS_TRAIN4 = {y: tuple(range(2016, y + 1)) for y in range(2016, 2020)}     # В1: 2016–2019, прогрев 2015


def windows(period: str = "train") -> dict:
    if period not in ("train", "train4"):
        raise ValueError(f"период {period!r}: train или train4")
    return WINDOWS if period == "train" else WINDOWS_TRAIN4


@contextmanager
def years(ys, val_start=None):
    old = s9.YEARS, s9.VAL_START
    s9.YEARS = tuple(ys)
    if val_start is not None:
        s9.VAL_START = val_start
    try:
        yield
    finally:
        s9.YEARS, s9.VAL_START = old


def s9_engine(cache: dict, ys, period: str = "train") -> "s9.Engine":
    vs = None if period == "train" else s9.PERIODS_S9[period][1]
    with years(ys, vs):
        return s9.Engine(cache)


def sk_cache(sid: str, pos: dict, td: dict, ys) -> dict:
    """Кэш в формате s9: позиция точки одна и та же на каждый год окна, остальные стратегии пусты."""
    return {t: dict(dates=td[t].dates, lr=td[t].log_ret, A={s: ({Y: pos[t] for Y in ys} if s == sid else {}) for s in s9.SIDS})
            for t in pos}


def a_cache(last_year: int = 2019) -> dict:
    """Боевая A из кэша s9 (v2): только dates/lr/A, годы после last_year отброшены сразу (validation не читается)."""
    res = pickle.loads((s9.OUT / (s9.OUT / "cache_latest_v2.txt").read_text().strip()).read_bytes())
    return {t: dict(dates=c["dates"], lr=c["lr"], A={s: {Y: v for Y, v in c["A"][s].items() if Y <= last_year} for s in c["A"]})
            for t, c in res.items()}


def assemble_excess(eng, comm: float) -> pd.Series:
    """`assemble` на тех же позициях, что `Engine.port`: панель из строк движка."""
    D, P = len(eng.dates), len(eng.pairs)
    pos, lr, act = np.zeros((D, P)), np.zeros((D, P)), np.zeros((D, P), bool)
    k, q = eng.r["date"], eng.r["pair"]
    pos[k, q], lr[k, q], act[k, q] = eng.A_row, eng.lr_row, True
    keep = act.any(0)
    pn = Panel(eng.dates, np.array([s for _, s in eng.pairs])[keep], pos[:, keep], lr[:, keep], act[:, keep],
               np.zeros((D, int(keep.sum()))), eng.rf * eng.cal, np.array([t for t, _ in eng.pairs])[keep])
    return assemble(pn, BASE_A, comm)["excess"]


def row(pt, god, comm, ev, ref, npos=0, status=None, why=""):
    """Строка таблицы: ev — ряд excess оценщика (или assemble), ref — ряд s9 (None — нет эталона)."""
    r = dict(tochka=pt["id"], klass=pt["klass"], god=god, comm_pct=comm * 100, etalon="da" if ref is not None else "net")
    r["sharpe_ocenshik"] = s9.sharpe(ev)
    if ref is None:
        return {**r, "sharpe_s9": np.nan, "raznitsa": np.nan, "max_dpnl": np.nan, "bar_poz_razl": 0, "status": "нет эталона", "prichina": why}
    same = ev.index.equals(ref.index)
    dp = float(np.abs(ev.values - ref.values).max()) if same else np.inf
    ok = same and dp <= PNL_TOL and npos == 0
    return {**r, "sharpe_s9": s9.sharpe(ref), "raznitsa": r["sharpe_ocenshik"] - s9.sharpe(ref), "max_dpnl": dp,
            "bar_poz_razl": npos, "status": status or ("совпало" if ok else "РАСХОЖДЕНИЕ"), "prichina": why}


def run_sk(pt: dict, ev, comms, period: str = "train") -> list:
    """Точка Sk: оценщик (`SkEvaluator`) против боевого пути, окна 2018 и 2019, комиссии comms (доли)."""
    from fastbt.sweep_post import build_view, positions
    sid, p = pt["sid"], pt["p"]
    sigs = ev.signals({k: v for k, v in p.items() if k in ev.sig_names})
    try:
        bo = {t: boevoy_pos(sid, p, t, *ev._td[t]) for t in sigs}
        why = ""
    except NoRef as e:
        bo, why = None, str(e)
    td = {t: ev._td[t][0] for t in sigs}
    out = []
    for god, ys in windows(period).items():
        end = pd.Timestamp(f"{god + 1}-01-01")
        npos = 0
        if bo is not None:
            for t, u in sigs.items():
                v, ex = build_view(sid, u, ev._td[t][0], ev._td[t][1], p, ev.sig_names)
                i1 = int(np.searchsorted(u.dates, np.datetime64(end)))
                a, b = year_bounds(u.dates, ys[0])[0], i1
                npos += int((positions(v, ex, i1)[a:b] != bo[t][a:b]).sum())
            eng = s9_engine(sk_cache(sid, bo, td, ys), ys, period)
        for c in comms:
            df = ev.pnl(sigs, p, c, end)["excess"]
            m = df.index.year == god
            ref = eng.port(eng.A_row, comm=c)["excess"] if bo is not None else None
            out.append(row(pt, god, c, df[m], None if ref is None else ref[ref.index.year == god], npos, why=why))
    return out


def run_a(pt: dict, cache: dict, comms, period: str = "train") -> list:
    """Точка A: `Engine.port` против `assemble` на позициях боевой A."""
    god, ys = pt["god"], windows(period)[pt["god"]]
    eng = s9_engine(cache, ys, period)
    out = []
    for c in comms:
        ref = eng.port(eng.A_row, comm=c)["excess"]
        asm = assemble_excess(eng, c)
        out.append(row(pt, god, c, asm[asm.index.year == god], ref[ref.index.year == god]))
    return out


def run_all(ids=None, comms=(0.0004, 0.00035)) -> pd.DataFrame:
    from fastbt.sweep_sk import SkEvaluator
    pts = [x for x in load_points()["tochki"] if ids is None or x["id"] in ids]
    rows, evs, acache = [], {}, None
    for pt in pts:
        if pt["sid"] == "A":
            acache = acache or a_cache()
            rows += run_a(pt, acache, comms)
        else:
            ev = evs.setdefault(pt["sid"], SkEvaluator(pt["sid"]))
            rows += run_sk(pt, ev, comms)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(run_all().to_string(index=False))
