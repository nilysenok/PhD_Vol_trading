"""Л9: разбор точки куба v5 (fastbt/designs/v5/kub_S{k}.yaml, канон fastbt/canon_v5.py) в оси оценщика v4/v2. Данных не читает.

razbor(sid, pt5) → Razbor: pt4 (оси etap1_eval.YAKOR), nov (оси etap1_eval_v2.NOVYE), ext (поля Params поверх
EvaluatorV2._params), filtr (v5_filtr.primenit), sloj (exec_e_v4sloj.filtr_signala), размер и ранг.
Каждая ось точки обязана быть известна разбору — иначе ValueError (ось не игнорируется молча, мутация «ось игнорируется»).
Якорь v5 (все оси = yakor куба) ↔ точка v4: k_s = 2 (σ), вход лимитом k = 0 (контр) / стоп-заявкой (тренд), выход по смене
сигнала, проверка по 5-мин барам, остальное — YAKOR (гейт «v5 на якорях = v4», tests/test_eval_v5.py).
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np

from fastbt.canon_v5 import NET, SIGNAL, TREND, edinica, load_kub
from fastbt.etap1_eval import YAKOR
from fastbt.etap1_eval_v2 import NOVYE
from fastbt.tejk_signal import RODNOY

# h в тейке k_t·σ·√h — по семейству (docs/resheniya/resheniya.md:1835: контр 3, диапазон 5, импульс 2, тренд 10);
# ❓ S2/S6 — контр, S5 — диапазон (умолчание кодера Л9, для дирижёра)
H_TEJK = {"S1": 3.0, "S2": 3.0, "S5": 5.0, "S6": 3.0, "S3": 10.0, "S4": 10.0}
RHO_RISK = 1.0          # ρ, % — только для формы веса risk_sigma (уровень — слой МСП, Ж-129 (5)); Шарп от уровня не зависит, пока min(1,·) не режет
FILTR = ("adx_th", "atr_lo", "vol_th", "rsi_th", "mom_n", "dow", "squeeze_pct")
SLOJ = {"consec": 1, "min_hold": 0, "cooldown": 0, "min_sily": 0.0}      # фаза 3 (не оси Соболя), умолчание — слой выключен
VHOD = ("k_e", "T", "reentry")
VYHOD = ("stop_nach", "tejk", "trail", "bezubytok", "stop_v_plyus", "tau", "chast", "pir", "smena", "razmer_tip", "m_sily",
         "rang_tip")
PT4 = {**YAKOR, "proverka": "bars5m", "sigma": "rv22"}                  # yakorya_progon куба


@dataclass
class Razbor:
    sp: dict
    pt4: dict
    nov: dict
    ext: dict = field(default_factory=dict)
    filtr: dict = field(default_factory=dict)
    sloj: dict = field(default_factory=lambda: dict(SLOJ))
    razmer: str = "ravnyj"
    m_sily: str = "1"
    rang_tip: str = "tikernyj"
    rang: bool = False                     # нужен ли ранг силы (m ≠ 1, тейк от силы, безубыток от ранга)


def rang_nan_u_vseh(sigs, rangi) -> bool:
    """Тренд × m_sily ≠ 1: есть старты сигнала (дни, где сигнал ≠ 0 и ≠ вчерашнему), но ни у одного нет конечного ранга — вес всех сделок 0
    (`_ves`: nan_to_num), Y = 0 при ненулевом числе сделок. Для любого rang_tip (tikernyj, srednee, obshchiy), не только tikernyj."""
    est = False
    for sig, rg in zip(sigs, rangi):
        sig = np.asarray(sig, float)
        start = (sig != 0) & (sig != np.r_[0.0, sig[:-1]])
        est |= bool(start.any())
        if np.isfinite(np.asarray(rg, float)[start]).any():
            return False
    return est


def chisla(v) -> tuple:
    """`atr:2.0` → (2.0,); `sila:0.5:1` → (0.5, 1.0); `1/3:1` → (0.333…, 1.0)."""
    s = str(v).split(":")
    s = s[1:] if s[0][:1].isalpha() else s
    return tuple(float(a) / float(b) if b else float(a) for a, _, b in (x.partition("/") for x in s))


def yakor(sid: str) -> dict:
    """Точка v5 «все новые оси = yakor» (сигнальные оси — без значения, их задаёт вызывающий)."""
    k = load_kub(sid)
    return {n: v["yakor"] for sec in ("filtr", "vhod", "vyhod") for n, v in k.get(sec, {}).items()}


def _vhod(r: Razbor, trend: bool, k_e: float, T: int, reentry: str) -> None:
    if trend:
        r.nov["vhod_tip"] = "stop_zayavka"
        if k_e:
            r.ext["k_e"] = float(k_e)
        return
    r.pt4["vhod"] = float(k_e)                                       # лимит L = P·(1 − k_e·σ̂), k_e = 0 — по цене закрытия
    if int(T) != 1:
        r.ext["T"] = int(T)
    if reentry != NET:
        r.ext["reentry_den"] = True


def _tejk(r: Razbor, sid: str, v) -> None:
    e = edinica(v)
    if e == NET:
        return
    if e == "rr":
        r.pt4["R"], r.ext["h"] = chisla(v)[0], 1.0                   # цель P(1 + R·b·u·√1): RR от стопа
    elif e in ("pct", "atr"):
        r.ext.update(a=chisla(v)[0], ed_t=e)
    elif e == "sigh":
        r.ext["a"] = chisla(v)[0] * math.sqrt(H_TEJK[sid])
    elif e == "rodnoy":
        r.nov["tejk_signal"] = RODNOY
    elif e == "sila":
        r.ext["a_rang"], r.rang = chisla(v), True
    else:
        raise ValueError(f"tejk {v!r}")


def _trail(r: Razbor, v) -> None:
    e = edinica(v)
    if e == "sigma":
        r.pt4["k_tr"] = chisla(v)[0]
    elif e == "atr":
        r.ext["atr_m"] = chisla(v)[0]
    elif e == "step":
        r.ext["step_s"] = chisla(v)[0]
    elif e == "parab":
        r.ext["parab"] = chisla(v)
    elif e != NET:
        raise ValueError(f"trail {v!r}")


def _vyhod(r: Razbor, sid: str, q: dict) -> None:
    u, (b,) = edinica(q["stop_nach"]), chisla(q["stop_nach"])
    r.pt4["k_s"] = b
    if u != "sigma":
        r.ext["ed_s"] = "atr" if u == "chand" else u
    _tejk(r, sid, q.get("tejk", NET))
    _trail(r, q.get("trail", NET))
    bz = edinica(q.get("bezubytok", NET))
    if bz == "atr":
        r.ext["b0"] = chisla(q["bezubytok"])[0]
    elif bz == "rang":
        r.ext["b0_c"], r.rang = chisla(q["bezubytok"])[0], True
    if q.get("stop_v_plyus", NET) != NET:
        r.ext["sp"] = chisla(q["stop_v_plyus"])
    r.pt4["tau"] = NET if q.get("tau", NET) == NET else int(q["tau"])
    if q.get("chast", NET) != NET:
        r.ext["chast"] = chisla(q["chast"])
    if float(q.get("pir", 0)):
        r.ext["pir"] = float(q["pir"])
    if q.get("smena", "da") == NET:
        r.ext["exit_signal"] = False
    r.razmer, r.m_sily, r.rang_tip = q.get("razmer_tip", "ravnyj"), str(q.get("m_sily", "1")), q.get("rang_tip", "tikernyj")
    r.rang = r.rang or r.m_sily != "1"


@functools.lru_cache(maxsize=None)
def _yakorya_progon(sid: str) -> dict:
    return dict(load_kub(sid).get("yakorya_progon", {}))


def razbor(sid: str, pt5: dict, signal=None, rol: str | None = None) -> Razbor:
    """Точка v5 (после canon_v5) → Razbor. signal — имена сигнальных осей (None — семья v5). Неизвестная ось — ValueError."""
    signal, trend = tuple(SIGNAL[sid] if signal is None else signal), sid in TREND
    ya = _yakorya_progon(sid)                             # драйвер (#608) кладёт якоря zh047 в точку: допустимы только на якоре
    spr = {"t_vyhod": "close"} if rol == "spravka" else {}   # Ж-137 п. 6: выход по close — только справочная роль (Ж-129 п. 4 для остальных)
    chuzhie = {k: v for k, v in pt5.items() if k in ya and pt5[k] != ya[k] and spr.get(k) != v}
    pt_close = bool(spr) and pt5.get("t_vyhod") == "close"
    if chuzhie:
        raise ValueError(f"{sid}: оси zh047 {chuzhie} не на якоре — оценщик v5 их не считает")
    pt5 = {k: v for k, v in pt5.items() if k not in ya}
    znaem = set(signal) | set(FILTR) | set(VHOD) | set(VYHOD) | set(SLOJ)
    if set(pt5) - znaem:
        raise ValueError(f"{sid}: оси {sorted(set(pt5) - znaem)} оценщик v5 не знает")
    if set(signal) - set(pt5) or "stop_nach" not in pt5:
        raise ValueError(f"{sid}: в точке нет сигнальных осей {sorted(set(signal) - set(pt5))} или stop_nach")
    r = Razbor({k: pt5[k] for k in signal}, dict(PT4), dict(NOVYE))
    r.nov["t_vyhod"] = "close" if pt_close else ya.get("t_vyhod", r.nov["t_vyhod"])   # Ж-137 п. 6: якорь куба (open2) доходит до исполнения, не молчит
    r.filtr = {k: pt5[k] for k in FILTR if k in pt5 and pt5[k]}
    r.sloj.update({k: pt5[k] for k in SLOJ if k in pt5})
    _vhod(r, trend, pt5.get("k_e", 0), pt5.get("T", 1), pt5.get("reentry", NET))
    _vyhod(r, sid, pt5)
    if r.razmer not in ("ravnyj", "risk_sigma") or r.m_sily not in ("1", "lin", "kvadr"):
        raise ValueError(f"razmer_tip {r.razmer!r}, m_sily {r.m_sily!r}")
    if trend and r.m_sily != "1" and r.rang_tip == "tikernyj":
        raise ValueError(f"{sid}: тренд × m_sily {r.m_sily!r} × rang_tip tikernyj несовместимо — ранга по тикеру нет, вес 0 (Р-4)")
    return r
