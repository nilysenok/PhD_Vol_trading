"""Правила цен и исполнения Ж-047 (контракт §1.1), чистые функции без состояния.

sd = +1 лонг, −1 шорт. Всё выключено по умолчанию (Params): tick=None — строгие сравнения E §2
(тейк h > T, лимит l < L); stop_touch=False — стоп l < S. Худший сценарий при включении:
стоп срабатывает при касании, тейк и лимит — только при проходе на 1 тик, цена — уровень или open при гэпе.
"""
from __future__ import annotations

import math

EPS = 1e-6                                      # допуск на сетку тиков, доля тика


def ceil_tick(x: float, tick: float) -> float:
    return math.ceil(x / tick - EPS) * tick if tick else x          # tick = 0 — без округления (касание)


def floor_tick(x: float, tick: float) -> float:
    return math.floor(x / tick + EPS) * tick if tick else x


def take_mult(p, sg: float) -> float | None:
    """Относительная дистанция цели: a·σ (E) или R·k_s·σ·√h (Ж-047, ❓1: h = τ, если не задано)."""
    if p.R is not None:
        return p.R * p.b * math.sqrt(p.h if p.h is not None else p.tau) * sg if p.b is not None else None
    return p.a * sg if p.a is not None else None


def stop_level(sd, px, sg, p, ext, be_on):
    """Стоп P(1 − sd·k_s σ); трейлинг ужесточает: экстремум с входа ∓ k_tr σ; безубыток — не хуже цены входа.
    k_s = нет (b is None) — стопа нет вообще, ни трейла, ни безубытка.
    Возвращает (уровень, причина): "stop" | "trail" (трейл строже базового стопа) | "bu" (безубыток строже всех)."""
    if p.b is None:
        return None, None
    c = [(px * (1 - sd * p.b * sg), "stop")]
    if p.k_tr is not None:
        c.append((ext * (1 - sd * p.k_tr * sg), "trail"))
    if be_on:
        c.append((px, "bu"))
    best = c[0]
    for x in c[1:]:
        if (x[0] > best[0]) if sd > 0 else (x[0] < best[0]):
            best = x
    return best


def ext_next(sd, ext, hi, lo):
    """Экстремум для трейлинга: max high (лонг) / min low (шорт) с входа; в режиме zakrytie hi = lo = close."""
    return max(ext, hi) if sd > 0 else min(ext, lo)


def be_reached(sd, px, sg, hi, lo):
    """Безубыток: достигнуто +1σ от цены входа (лонг — high ≥ P(1+σ); шорт — low ≤ P(1−σ))."""
    return hi >= px * (1 + sg) if sd > 0 else lo <= px * (1 - sg)


def take_level(sd, px, sg, p):
    m = take_mult(p, sg)
    if m is None:
        return None
    T = px * (1 + sd * m)
    return T if p.tick is None else (ceil_tick(T, p.tick) if sd > 0 else floor_tick(T, p.tick))


def stop_hit(sd, o, h, l, S, touch):
    """(сработал, цена): цена — S или open при гэпе через S. touch: l ≤ S, иначе l < S."""
    if sd > 0:
        return (l <= S if touch else l < S), min(o, S)
    return (h >= S if touch else h > S), max(o, S)


def take_hit(sd, o, h, l, T, tick):
    """T — уровень после округления по тику. Тик не задан — строго сквозь T; иначе — на 1 тик сквозь T."""
    if sd > 0:
        hit = h > T if tick is None else h >= T + tick - tick * EPS
        return hit, max(o, T)
    hit = l < T if tick is None else l <= T - tick + tick * EPS
    return hit, min(o, T)


def limit_level(sd, close, k, sg, tick):
    """Лимит входа close·(1 − sd·k σ); с тиком — округление в сторону, где исполнить труднее."""
    L = close * (1 - sd * k * sg)
    return L if tick is None else (floor_tick(L, tick) if sd > 0 else ceil_tick(L, tick))


def limit_mask(sd, h, l, L, tick):
    """Бары, где лимит исполнен (скаляры или массивы): l < L / l ≤ L − тик; шорт зеркально."""
    if sd > 0:
        return l < L if tick is None else l <= L - tick + tick * EPS
    return h > L if tick is None else h >= L + tick - tick * EPS


def limit_px(sd, o, L):
    """Цена исполнения: L или open при гэпе через L."""
    return min(o, L) if sd > 0 else max(o, L)
