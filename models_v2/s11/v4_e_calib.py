"""V4-E: причинная калибровка смещения F = F0·exp(b_t) экспоненциальными средними (E1–E4, дневной слой E6).
ĉ = Σω·x / Σω, x = RV/F0 (QLIKE-оптимальный множитель при весах ω); b = clip(log ĉ, ±0,5). Пара (прогноз d, цель t⁺) закрыта для прогноза дня D, если closed_date(пара) ≤ D (боевое правило: t⁺ ≤ D).
Режимы: own (бумага + стягивание к панели, E1/E2/E6), panel (общая ĉ_t, E3), two (панель · относительная бумага, E4). До MIN_PAIRS закрытых пар бумаги b = 0.
Панель — бумаги panel_mask (17; в LOTO — 16). Состояние бумаги — только по её закрытым парам. Гиперпараметры фиксированы объявлением."""
from __future__ import annotations

import numpy as np

KAPPA, MIN_PAIRS, B_CLIP = 250.0, 60, 0.5


def closed_date(t_plus, date):
    """День, с которого пара доступна прогнозу: дата цели t⁺ (тест подменяет на дату прогноза — «закрыто + 1»)."""
    return t_plus


def lam(hl: float) -> float:
    return 0.5 ** (1.0 / hl)


class _St:
    """Состояние: на бумагу (num, den, den2, cnt) и панельное (pnum, pden с затуханием по календарным шагам)."""

    def __init__(self, nT):
        self.num, self.den, self.den2, self.cnt = (np.zeros(nT) for _ in range(4))
        self.pnum = self.pden = 0.0

    def add(self, i, x, lo, on_panel, xp=None):
        self.num[i] = lo * self.num[i] + x
        self.den[i] = lo * self.den[i] + 1.0
        self.den2[i] = lo * lo * self.den2[i] + 1.0
        self.cnt[i] += 1
        if on_panel:
            self.pnum += x if xp is None else xp
            self.pden += 1.0


def _own(st, i, cpan, kappa):
    c_i = st.num[i] / st.den[i]
    n = st.den[i] ** 2 / st.den2[i]
    return (n * c_i + kappa * cpan) / (n + kappa)


def ewma_b(rv, F0, tix, date, t_plus, nT, mode, hl_own, hl_panel, panel_mask, kappa=KAPPA, min_pairs=MIN_PAIRS):
    """rv, F0, tix, date (datetime64), t_plus — массивы строк, отсортированных по (date, ticker). -> (b, c_panel) по строкам; c_panel — панельная ĉ дня (до умножения на бумажную часть)."""
    n = len(rv)
    avail = np.asarray(closed_date(np.asarray(t_plus), np.asarray(date)))
    order = np.argsort(avail, kind="stable")
    x = np.asarray(rv, float) / np.asarray(F0, float)
    st, lo, lp = _St(nT), lam(hl_own), lam(hl_panel)
    b, cpn = np.zeros(n), np.ones(n)
    cprow = np.ones(n)             # панельная ĉ, применённая к прогнозу строки (для E4: x_rel)
    days, starts = np.unique(date, return_index=True)
    ends = list(starts[1:]) + [n]
    ptr = 0
    for d, a, e in zip(days, starts, ends):
        st.pnum *= lp
        st.pden *= lp
        while ptr < n and avail[order[ptr]] <= d:
            j = order[ptr]
            ptr += 1
            xr = x[j] / cprow[j] if mode == "two" else x[j]
            st.add(tix[j], xr, lo, bool(panel_mask[tix[j]]), x[j])
        cpan = st.pnum / st.pden if st.pden > 0 else 1.0
        cprow[a:e], cpn[a:e] = cpan, cpan
        for r in range(a, e):
            i = tix[r]
            if st.cnt[i] < min_pairs:
                continue
            c = _c_row(mode, st, i, cpan, kappa)
            b[r] = float(np.clip(np.log(c), -B_CLIP, B_CLIP))
    return b, cpn


def _c_row(mode, st, i, cpan, kappa):
    if mode == "panel":
        return cpan
    if mode == "own":
        return _own(st, i, cpan, kappa)
    return cpan * _own(st, i, 1.0, kappa)            # two: ĉ_i,rel стянута к 1
