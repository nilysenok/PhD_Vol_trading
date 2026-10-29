"""Семья E: внутридневное исполнение по правилам §2 (docs/zadaniya/2026-10-27_semya_E.md).

Один тикер, длина 1 капитал. sig[d], sigma[d] (σ̂₁) известны на close d; заявка на d+1 исполняется
по барам d+1 и позже. Дни pusto выпадают из ленты (test_08_halts): позиция переносится, доходность
паузы — на интервал до неё, в τ не считаются. Плохие бары (пусто/NaN/≤0) — ValueError.

Учёт — соглашение A (msp.portfolio, Н2), логарифмы цен сделок. Строка t = интервал [close t, close
следующего торгового дня): pos[t] — позиция на close t; gross[t] = pos[t]·lr[t] у A; cost[t] — комиссии
исполнений по close t, внутридневные (лимит/стоп/тейк) — в строке начала их интервала; pnl = gross − cost.
Ставка: free[t] = 1 − доля баров следующего дня в рынке (у A 1 − |pos[t]|): позиция, открытая по close d,
ставку за день d не получает; после выхода по close x ставка идёт со строки x. fdr = free·rf_cal,
rf_cal[t] = rf/365·календарные дни [t, след. торговый день) (msp.cal_days); на днях pusto нули.
Избыточная = pnl + fdr − rf_cal.
Сигнал: sig == 0 на close любого дня удержания, включая день лимитного входа → выход по close
(позиция не доживает до d+2); zakrytie_d1 и повтор — только при sig[d+1] == 1. Повторный вход как A:
после stop/take/vremya/signal при sig == 1 — снова на том же close (cooldown A — параметр RM, тут 0).
Н5: позиция, открытая на последней строке окна, платит комиссию выхода в cost последней строки.
↺ A не задаёт неисполненную заявку: вход блокируется до sig == 0. Тейк в баре входа не проверяется.

Ж-047 (docs/otchety/2026-10-27_zh047_kontrakt.md §1.1, правила — exec_e_rules): sig ∈ {−1, 0, +1}, знак — сторона
(шорт: pos = −1, gross = −ln(px/ref)); выход по сигналу, если sig != стороне. Всё новое выключено по умолчанию
(Params): tick, stop_touch, R/h, k_tr, be, check="zakrytie", reentry=False, close (Н6, 18:50) — E не меняется.
Родные выходы Sk (sl/tp/max_hold) живут в самих позициях sig: режим «якорь» — σ-выходы выключены, режим
сетки — родные RM-выходы выключены при построении sig (K4), σ-выходы включены; k_s = нет (b=None) — стопа нет вообще.
"""
from __future__ import annotations

import numpy as np

from fastbt.exec_e2 import Run2
from fastbt.exec_e_parab import parab_check, parab_new
from fastbt.exec_e_rules import atr_series
from fastbt.exec_e_run import Run
from fastbt.exec_e_types import Day, Params, Result, Trade  # noqa: F401
from fastbt.exec_e_v6 import v5_new


def q_mask(strength, sig, q: float, window: int = 252, min_obs: int = 20) -> np.ndarray:
    """E0 (§4): sig' = sig ∧ (сила[d] > квантиль q силы за дни сигнала окна [d−window, d−1]).
    Окно строго прошлое (Н1), как A при min_strength — обнуляет и посреди сделки (spec п. 13).
    q = 0 — sig как есть (побайтно). Истории меньше min_obs — нет сделки (консервативно)."""
    sig, st = np.asarray(sig).astype(int), np.asarray(strength, float)
    if len(sig) != len(st):
        raise ValueError("sig и strength разной длины")
    if q == 0:
        return sig.copy()
    if not 0 < q < 1:
        raise ValueError("q вне [0; 1)")
    out = np.zeros(len(sig), int)
    for d in np.flatnonzero(sig):
        h = st[max(0, d - window):d]
        h = h[(sig[max(0, d - window):d] == 1) & np.isfinite(h)]
        out[d] = int(h.size >= min_obs and np.isfinite(st[d]) and st[d] > np.quantile(h, q))
    return out


def shift_krug(x, lag: int) -> np.ndarray:
    """к2: x'[j] = x[(j − lag) mod T]; lag = 0 — исходный ряд (мутация 8)."""
    x = np.asarray(x)
    return x[(np.arange(len(x)) - lag) % len(x)] if len(x) else x.copy()


def _check_params(p: Params) -> None:
    if p.R is not None and p.a is not None:
        raise ValueError("R и a несовместимы")
    if p.R is not None and p.h is None and p.tau is None:
        raise ValueError("R требует h или tau (❓1)")
    if p.tejk not in ("net", "k_srednemu", "pol_puti", "rodnoy_uroven"):
        raise ValueError("tejk: net | k_srednemu | pol_puti | rodnoy_uroven")
    if p.tejk != "net" and (p.R is not None or p.a is not None):
        raise ValueError("tejk ≠ net несовместим с R и a: тейк один (Ж-084)")
    if p.check not in ("bary", "zakrytie"):
        raise ValueError("check: bary | zakrytie")
    if p.side not in (1, -1):
        raise ValueError("side: +1 | −1")
    if p.tick is not None and not p.tick >= 0:
        raise ValueError("tick должен быть ≥ 0 (0 — касание)")
    if p.vyhod_lim and (p.k is None or p.check != "bary"):
        raise ValueError("vyhod_lim требует k (лимитный вход) и check='bary'")
    if p.t_vyhod not in ("close", "1500", "1700", "open2"):
        raise ValueError("t_vyhod: close | 1500 | 1700 | open2")
    if p.zhizn not in (None, 12, 36, 60):
        raise ValueError("zhizn: None | 12 | 36 | 60")
    if p.bar_plus < 0 or p.stop_tik < 0:
        raise ValueError("bar_plus и stop_tik должны быть ≥ 0")
    if p.stop_tik and p.tick is None:
        raise ValueError("stop_tik требует tick")
    if (p.bar_plus or p.t_vyhod != "close") and (p.check != "bary" or p.vyhod_lim):
        raise ValueError("bar_plus и t_vyhod требуют check='bary' без vyhod_lim")
    if p.atr_m is not None and not p.atr_m > 0:
        raise ValueError("atr_m должен быть > 0")
    if p.chast_1r and (p.b is None or p.check != "bary"):    # chast (доля, r) проверяет exec_e_v6.v5_check
        raise ValueError("chast_1r требует b (R = расстояние до начального стопа) и check='bary'")
    if p.vhod_tip not in ("podtverzhdenie", "stop_zayavka"):
        raise ValueError("vhod_tip: podtverzhdenie | stop_zayavka")
    if p.vhod_tip == "stop_zayavka" and (p.k is not None or p.check != "bary" or not p.stop_touch or p.exit_signal or p.vyhod_lim
                                           or p.zhizn or p.gap_propusk):
        raise ValueError("stop_zayavka: без лимита k, zhizn, gap, vyhod_lim и выхода по сигналу; check='bary', stop_touch (Ж-091)")
    if p.vhod_open and (p.k is not None or p.check != "bary" or p.vyhod_lim):
        raise ValueError("vhod_open — вход рынком (k = None), check='bary', без vyhod_lim")
    if not isinstance(p.T, int) or p.T < 1:
        raise ValueError("T — целое ≥ 1")
    if p.T > 1 and (p.k is None or p.vhod_open or p.vhod_tip != "podtverzhdenie"):
        raise ValueError("T > 1 — заявки по уровню (k задан), без vhod_open и stop_zayavka (Ж-111; стоп-заявка пересчитывается каждый вечер)")
    if p.reentry_den and p.vhod_tip == "podtverzhdenie" and (p.k is None or p.vhod_open):
        raise ValueError("reentry_den — заявки по уровню (k задан) без vhod_open, либо stop_zayavka (Ж-111, V6-11c)")
    if p.reentry_den and p.check != "bary":
        raise ValueError("reentry_den требует check='bary'")
    if p.ed_s not in ("sigma", "atr", "pct") or p.ed_t not in ("sigma", "atr", "pct"):
        raise ValueError("ed_s, ed_t: sigma | atr | pct")
    if p.chand is not None and (not p.chand > 0 or p.b is None or p.check != "bary"):
        raise ValueError("chand > 0 требует b (начальный стоп) и check='bary'")
    parab_check(p)


def unit_series(days, close, kind: str, sigma, atr) -> np.ndarray | None:
    """Ряд единицы уровней u[i], известный на close дня i (Л5): sigma — None (путь прежний); pct — 1;
    atr — ATR(14) по дням ≤ i / close дня i (последнего торгового). Уровень сделки с сигналом дня d берёт u[d]."""
    if kind == "sigma":
        return None
    if kind == "pct":
        return np.ones(len(days))
    cl, last = np.empty(len(days)), float("nan")
    for i, d in enumerate(days):
        last = (float(close[i]) if close is not None else float(d.c[-1])) if not d.pusto else last
        cl[i] = last
    return np.asarray(atr, float) / cl


def _new_axes(p: Params, ed) -> bool:
    return ed is not None or p.zhizn is not None or p.gap_propusk or p.t_vyhod != "close" or bool(p.bar_plus or p.stop_tik) or p.vhod_open \
        or p.T != 1 or p.reentry_den or v5_new(p) or parab_new(p)     # Л6, П3b → медленный путь


def run(ticker, days, sig, sigma, p: Params, rf_cal=None, close=None, div=None, nalog=None, ed=None, tejk_M=None,
        urovni=None, rang=None) -> Result:
    """rang[d] — ранг силы сигнала дня d, [0; 1] (V6-b: Params.b0_c, fastbt/exec_e_v6.py); иначе не нужен.
    urovni — уровни стоп-заявок по дням (vhod_tip = stop_zayavka, fastbt/stop_zayavki.py), обязателен при нём, иначе не принимается.
    tejk_M[d] — уровень M сигнала на день d (Params.tejk ≠ net, Ж-084). rf_cal[t] — ставка/365·календарные дни [t, следующий торговый день) (msp.cal_days, Н2).
    close[t] — закрытие дня по Н6 (18:50); None — close последнего 5-мин бара (E).
    div[i] — полный дивиденд, руб./акция, на день гэпа i (fastbt/dividendy.gap_divs); None — без дивидендов (Ж-057 п. 1).
    В PnL идёт D·nalog (None — dividendy.DIV_NALOG = 0,87), стоп переносимой позиции сдвигается на полный D (Ж-060)."""
    sig, sigma = np.asarray(sig), np.asarray(sigma, float)
    if not (len(days) == len(sig) == len(sigma)) or (close is not None and len(close) != len(days)):
        raise ValueError("days, sig, sigma, close разной длины")
    _check_params(p)
    for i, d in enumerate(days):
        if not d.pusto:
            a = np.stack([d.o, d.h, d.l, d.c]) if len(d.c) else None
            if a is None or not np.isfinite(a).all() or (a <= 0).any():
                raise ValueError(f"день {i}: пустой или нечисловой набор баров (нужен pusto=True)")
            if close is not None and not (np.isfinite(close[i]) and close[i] > 0):
                raise ValueError(f"день {i}: close Н6 нечисловой")
    if div is not None and len(div) != len(days):
        raise ValueError("days, div разной длины")
    if ed is not None and len(ed) != len(days):
        raise ValueError("days, ed разной длины")
    from fastbt import dividendy
    nal = dividendy.DIV_NALOG if nalog is None else nalog
    if p.tejk != "net" and (tejk_M is None or np.shape(tejk_M)[-1] != len(days)
                            or np.ndim(tejk_M) != (2 if p.tejk == "rodnoy_uroven" else 1)):
        raise ValueError("tejk ≠ net требует tejk_M длины days (rodnoy_uroven — [2, days]: лонг, шорт)")
    need_atr = p.atr_m is not None or p.chand is not None or p.k_e != 0 or "atr" in (p.ed_s, p.ed_t)
    atr = atr_series(days, close) if need_atr else None      # Ж-092: ATR(14) на close каждого дня
    un = dict(us=unit_series(days, close, p.ed_s, sigma, atr), ut=unit_series(days, close, p.ed_t, sigma, atr))
    if (p.vhod_tip == "stop_zayavka") != (urovni is not None):
        raise ValueError("urovni нужны ровно при vhod_tip = stop_zayavka")
    if urovni is not None and any(len(urovni[k]) != len(days) for k in ("up", "dn", "xl", "xs")):
        raise ValueError("days, urovni разной длины")
    if urovni is not None:
        from fastbt.exec_e_stop import RunStop
        r = RunStop(ticker, days, sig, sigma, p, close, div, nal, ed=ed, tejk_M=tejk_M, urovni=urovni, atr=atr, rang=rang, **un)
    else:
        r = Run2(ticker, days, sig, sigma, p, close, div, nal, ed=ed, tejk_M=tejk_M, atr=atr, rang=rang, **un) if _new_axes(p, ed) \
            else Run(ticker, days, sig, sigma, p, close, div, nal, tejk_M=tejk_M, atr=atr, rang=rang, **un)
    for j, i in enumerate(r.ix):
        r.step(j, i)
    n = len(days)
    if r.ix and r.tr is not None:                  # Н5: открытая на последней строке окна платит выход
        r.cost[r.ix[-1]] += p.comm * r.w
    gross, free = np.zeros(n), np.zeros(n)
    for j, i in enumerate(r.ix):                   # день i → интервал, начатый предыдущим торговым днём
        if j:
            gross[r.ix[j - 1]] += r.gross[i]
            free[r.ix[j - 1]] = 1.0 - r.used[i] / len(days[i].c)
    if r.ix:
        free[r.ix[-1]] = 1.0 - abs(r.pos[r.ix[-1]])
    for i, d in enumerate(days):
        if d.pusto:
            r.pos[i] = r.pos[i - 1] if i else 0.0
    rf = np.zeros(n) if rf_cal is None else np.asarray(rf_cal, float)
    return Result(r.trades, gross, r.cost, r.pos, free, free * rf, r.n_orders, r.n_fills, r.x_orders, r.x_fills,
                  getattr(r, "x_t", 0), r.n_bez)


def run_e1(*a, **kw):
    """Ж-047 этап 1: строки портфеля (g, sides, expo) с весом на сделку — fastbt/etap1_run.py."""
    from fastbt.etap1_run import run_e1 as f
    return f(*a, **kw)
