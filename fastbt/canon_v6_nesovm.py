"""Несовместимые сочетания осей кубов v6 → каноническая форма None (Ж-137 п. 7б; tolkovaniya.md [30.10 04:10], класс B).

Каждое правило — ровно то условие, по которому оценщик (или движок) отказывает точке: NeSchitana / ValueError. Условия не шире и не уже.
Отказ в оценщике остаётся защитой. Данных не читает. Правила — функции `(sid, q) -> bool` (True — точка несовместима); q — точка до canon_v5.
Правило 1 — eval_v6_ispoln.proverit + exec_e._check_params; 2 — exec_e_parab.parab_check; 3 — eval_v5._rows (a_rang, RunStop);
4 — eval_v6_piv.proverit (pol_puti); 5, 6 — eval_v6_piv.proverit (пивот-стоп); 7 — eval_v6._rows (norm) и eval_v5._rows (rodnoy_ok).
"""
from __future__ import annotations

from fastbt.canon_v5 import NET, SIGNAL, TREND, edinica

_EXT_PIV = ("trail", "bezubytok", "stop_v_plyus", "chast")      # оси, ставящие atr_m/step_s/parab/b0/b0_c/sp/chast в Params (eval_v5_osi._vyhod)


def svernut(q: dict) -> dict:
    """Пивот/VWAP-стоп — единственный: оси ведения стопа (трейл, безубыток, stop_v_plyus), chast (R — расстояние до начального стопа, его нет),
    пирамида (риск добавки — от стопа) и размер risk_sigma (вес от расстояния до стопа) при нём не действуют → «выкл.» (свёртка, как piv_buf),
    а не None. Меняет q на месте. Тейк rr (R от стопа) — содержательная ось тейка: не сворачивается, остаётся отказом (правило 5)."""
    if edinica(q.get("stop_nach")) in ("piv", "vwap"):
        q.update({n: NET for n in _EXT_PIV if n in q})
        q.update({n: v for n, v in (("pir", 0), ("razmer_tip", "ravnyj")) if n in q})
    return q


def _est(q: dict, n: str) -> bool:
    return q.get(n, NET) != NET


def _pir(q: dict) -> bool:
    return bool(float(q.get("pir", 0)))


def vyhod_lim_t_vyhod(sid: str, q: dict) -> bool:
    """1: vyhod_lim = da требует выхода по close; ось t_vyhod вне точки — якорь куба open2."""
    return q.get("vyhod_lim", NET) == "da" and q.get("t_vyhod", "open2") != "close"


def parab_drugie(sid: str, q: dict) -> bool:
    """2: parab несовместим с другими трейлингами, безубытком, stop_v_plyus и пирамидой (трейл — одна ось, остальные трейлы исключены ею)."""
    return edinica(q.get("trail", NET)) == "parab" and (_est(q, "bezubytok") or _est(q, "stop_v_plyus") or _pir(q))


def trend_tejk_sily(sid: str, q: dict) -> bool:
    """3: тренд на стоп-заявке (vhod_tip по умолчанию) не считает тейк от силы (a_rang)."""
    return sid in TREND and edinica(q.get("tejk", NET)) == "sila" and q.get("vhod_tip", "stop_zayavka") == "stop_zayavka"


def pol_puti_ma_tip(sid: str, q: dict) -> bool:
    """4: pol_puti — M = SMA сигнала (S1, S2)."""
    return q.get("tejk") == "pol_puti" and sid in ("S1", "S2") and q.get("ma_tip") != "sma"


def piv_edinstvennyj_stop(sid: str, q: dict) -> bool:
    """5 (остаток после свёртки): пивот- или VWAP-стоп — единственный: тейк rr (R от стопа) невозможен. Остальное сворачивается (svernut)."""
    return edinica(q.get("stop_nach")) in ("piv", "vwap") and (
        _pir(q) or edinica(q.get("tejk", NET)) == "rr" or q.get("razmer_tip") == "risk_sigma")     # pir, risk_sigma — защита, если свёртку обошли


def piv_pol_puti(sid: str, q: dict) -> bool:
    """6: пивот-стоп и pol_puti делят канал tejk_M."""
    return edinica(q.get("stop_nach")) in ("piv", "vwap") and q.get("tejk") == "pol_puti"


def rodnoy_ok(sid: str, q: dict) -> bool:
    """Копия eval_v5.rodnoy_ok (сверяется тестом): родной уровень совпадает с уровнем сигнала v5 только при этих осях."""
    if sid == "S1":
        return q.get("ma_tip") == "sma" and q.get("dev_tip") == "std" and q.get("z_exit") != NET
    return {"S2": q.get("ma_tip") == "sma", "S6": True}.get(sid, False)


def rodnoy_uroven(sid: str, q: dict) -> bool:
    """7: tejk = rodnoy при norm = s252 (масштаб окна, не 252 дней) или при осях, где уровень не совпадает с сигналом."""
    return edinica(q.get("tejk", NET)) == "rodnoy" and (q.get("norm") == "s252" or not rodnoy_ok(sid, {k: q.get(k) for k in SIGNAL[sid]}))


def rodnoy_tau_vyhod_lim(sid: str, q: dict) -> bool:
    """8 (найдено оракулом теста, не в списке дирижёра): rodnoy при τ ≠ net идёт с t_vyhod = open2 (EvaluatorV2._params), а vyhod_lim с t_vyhod ≠ close
    движок отказывает ValueError (exec_e._check_params) → status oshibka → rc = 2. Откат: убрать из PRAVILA."""
    return edinica(q.get("tejk", NET)) == "rodnoy" and q.get("vyhod_lim", NET) == "da" and q.get("tau", NET) != NET


PRAVILA = (vyhod_lim_t_vyhod, parab_drugie, trend_tejk_sily, pol_puti_ma_tip, piv_edinstvennyj_stop, piv_pol_puti, rodnoy_uroven,
           rodnoy_tau_vyhod_lim)


def nesovm(sid: str, q: dict) -> bool:
    return any(p(sid, q) for p in PRAVILA)
