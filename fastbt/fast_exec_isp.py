"""Оси исполнения куба v6 в быстром ядре (Ф-v6-osi): gap = propusk, zhizn (12 | 36 | 60), ed_lim (вход `ed` — единица отступа лимита
по дням) и t_vyhod 1500 / 1700. Построчно как `exec_e2` (gap_over, zhizn_hits, Run2._place / _try_order / _scan / _leave) и `exec_e_vremya`;
результат побайтно равен эталону. lestnica ядра не требует: две ноги — два обычных вызова run (eval_v6_ispoln._legs).
Модуль не импортирует fast_exec (индексы осей — здесь, fast_exec их берёт). Строка UR[6] — ed (pi[P_ED] = 1), иначе единица лимита — σ̂."""
from __future__ import annotations

import numpy as np
from numba import njit

P_GAP, P_ZH, P_ED, P_TH = 34, 35, 36, 37        # pi: gap_propusk; жизнь заявки в барах (0 — весь день); ed подан; hhmm выхода (900 | 1020; 0 — нет)
PENDH, PENDR = 35, 36                            # st: отложенный выход — бар hhmm (1) или 10:00 (0); причина отложенного выхода
NPI, NST = 38, 37
HHMM = {"1500": 900, "1700": 1020}               # = exec_e_vremya.HHMM
OSI = frozenset(("gap_propusk", "zhizn"))        # поля Params, которые знает ядро (+ t_vyhod 1500 / 1700 и вход ed)


def params_isp(p) -> np.ndarray:
    """pi[4]: gap_propusk, zhizn (0 — нет), ed (ставит run_v5), hhmm выхода."""
    return np.array([p.gap_propusk, p.zhizn or 0, 0, HHMM.get(p.t_vyhod, 0)], np.int64)


def cut_hhmm(days, p) -> np.ndarray:
    """Бар hhmm каждого дня (exec_e_vremya.bar_at; −1 — времени нет или день пуст) при t_vyhod 1500 / 1700, иначе пустой массив."""
    hh = HHMM.get(p.t_vyhod)
    if hh is None:
        return np.empty(0, np.int64)
    from fastbt import exec_e_vremya as vr
    return np.array([-1 if d.pusto else (-1 if (b := vr.bar_at(d, hh)) is None else b) for d in days], np.int64)


@njit(cache=True)
def unit(UR, i, sg, pi):
    """Единица отступа лимита дня i (Run2._place: ed[i] или σ̂)."""
    return UR[6, i] if pi[P_ED] else sg


@njit(cache=True)
def gap_over(sd, o0, L, tick, htick):
    """exec_e2.gap_over: open бара 0 уже за лимитом (тик не задан — строго)."""
    if not htick:
        return o0 < L if sd > 0 else o0 > L
    return o0 <= L - tick if sd > 0 else o0 >= L + tick


@njit(cache=True)
def nbars(nb, pi):
    """Бары, на которых заявка может исполниться: b < zhizn (exec_e2.zhizn_hits); 0 — весь день."""
    return nb if pi[P_ZH] == 0 else min(nb, pi[P_ZH])
