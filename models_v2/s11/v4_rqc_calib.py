"""V4-RQC: K8 и аналитические калибровки множителя F = a·K8 (Ж-225 п. 6; GATES_V4.md «Лист K8-RQC»).
K8 = sqrt(H·M), H = K1, M = sqrt(K2·K3) — равновесная смесь (B2 при w = 0,5), контроль «ничего не знающего» веса. x = RV/K8; a = mean(x) — QLIKE-оптимальный множитель; обрезка a ∈ [0,6; 1,6].
RQC1 — по всем закрытым парам панели (17 бумаг, расширяющееся окно); RQC2 — по закрытым парам панели прошлого календарного года Y−1 (для 2017 a = 1). Пара закрыта для прогноза дня D, если closed_date(пара) ≤ D (t⁺ ≤ D).
До MIN_PAIRS закрытых пар a = 1. Гиперпараметры фиксированы объявлением. Подмены `closed_date`, `ref_year` — только в тестах-мутациях."""
from __future__ import annotations

import numpy as np

A_LO, A_HI, MIN_PAIRS = 0.6, 1.6, 60


def closed_date(t_plus, date):
    """День, с которого пара доступна прогнозу: дата цели t⁺ (тест подменяет на дату прогноза — «закрыто + 1»)."""
    return t_plus


def ref_year(y):
    """Год пар для a_Y: прошлый календарный год Y−1 (тест подменяет на Y — утечка)."""
    return y - 1


def k8(h, k2, k3):
    """K8 = sqrt(H·sqrt(K2·K3))."""
    return np.sqrt(np.asarray(h, float) * np.sqrt(np.asarray(k2, float) * np.asarray(k3, float)))


def year_of(d) -> np.ndarray:
    return np.asarray(d).astype("datetime64[Y]").astype(np.int64) + 1970


def global_a(x, avail, date, member) -> np.ndarray:
    """RQC1: a по строкам = mean(x) по парам панели (member) с avail ≤ дата строки; до MIN_PAIRS пар — 1."""
    order = np.argsort(avail, kind="stable")
    ok = np.asarray(member)[order]
    cs, cn = np.cumsum(np.where(ok, np.asarray(x, float)[order], 0.0)), np.cumsum(ok)
    pos = np.searchsorted(avail[order], np.asarray(date, dtype="datetime64[ns]"), side="right")
    ix = np.maximum(pos - 1, 0)
    s, c = np.where(pos > 0, cs[ix], 0.0), np.where(pos > 0, cn[ix], 0)
    return np.clip(np.where(c >= MIN_PAIRS, s / np.maximum(c, 1), 1.0), A_LO, A_HI)


def yearly_a(x, avail, date, member) -> np.ndarray:
    """RQC2: для строк года Y — mean(x) по парам панели с годом avail = ref_year(Y) (Y−1); нет пар — 1."""
    dy, ay = year_of(date), year_of(avail)
    x, ok, a = np.asarray(x, float), np.asarray(member), np.ones(len(dy))
    for y in np.unique(dy):
        m = ok & (ay == ref_year(y))
        if m.sum() >= MIN_PAIRS:
            a[dy == y] = x[m].mean()
    return np.clip(a, A_LO, A_HI)
