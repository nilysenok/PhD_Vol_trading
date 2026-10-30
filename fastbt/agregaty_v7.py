"""Агрегаты точки куба v7 для ledger (Ж-155 п. 3, Ж-156 п. 4): суммы вместо дневных рядов; данных не читает.

Формат: поле строки ledger 2.2 `agr` (расширение 2.2, не новая версия: старые строки без `agr` читаются прежним путём,
agregaty() на них отдаёт None; proverit_stroku_22 и Pisatel не меняются). Поле необязательное.
  {"v": "7.1", "gody": [2017..2026],
   "d": [n, s, q, sd, ex]   — по годам (списки длины len(gody)): n — дней, s = Σx, q = Σx², sd — сделок, ex = Σ|poz| портфеля
                              (валовая доля капитала по дням; средняя экспозиция = ex/n); год — по дате ДНЯ ряда;
   "w": [n, s1, s2, s3, s4] — по годам, недельные суммы x (W-FRI, как metriki_e.nedelnye_summy; неполные края входят),
                              год недели — год её последнего торгового дня в ряду: n недель, Σw, Σw², Σw³, Σw⁴ (для CF-поправки);
   "r": [n, s, q, sd, ex]   — по трём режимам рынка за весь период, порядок REZHIMY = (рост, падение, боковик);
                              режим дня d — компонента trend из fazy._priznaki по IMOEX на d−1 (_LAG = 1); дни без режима не входят}.
x — избыточная дневная доходность портфеля после комиссий. Sharpe — как metriki_e.sharpe (Д7: √252, ddof = 1).
Год недели — год её последнего торгового дня в ряду (неделя через границу года идёт в год своего последнего дня)."""
from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd

from fastbt import fazy
from fastbt.metriki_e import Z95, sharpe, sr_dnevnoy

_REF = np.array([1.0, 2.0, 4.0])
SHKALA = sharpe(_REF) / sr_dnevnoy(_REF)            # годовая шкала проекта (√252), из metriki_e

VERSIYA = "7.1"
GODY_V7 = tuple(range(2017, 2027))                  # 2017…2025 и янв–фев 2026 (справочно)
REZHIMY = ("рост", "падение", "боковик")


def _god(daty: np.ndarray) -> np.ndarray:
    return np.asarray(daty, "datetime64[D]").astype("datetime64[Y]").astype(int) + 1970


def rezhim_dnya(daty, imoex_close: pd.Series) -> np.ndarray:
    """Индекс режима 0/1/2 (REZHIMY) на каждую дату по IMOEX до d−1 включительно; −1 — режима нет (прогрев MA200)."""
    tr = fazy._priznaki(imoex_close.astype(float).sort_index())["trend"]
    tr = tr.reindex(pd.DatetimeIndex(np.asarray(daty, "datetime64[D]")))
    return np.array([REZHIMY.index(v) if isinstance(v, str) else -1 for v in tr], dtype=int)


def _po_klyucham(k: np.ndarray, nk: int, x, poz, sdelki) -> list:
    ok = k >= 0
    f = lambda a: np.bincount(k[ok], weights=np.asarray(a, float)[ok], minlength=nk)   # noqa: E731
    x = np.asarray(x, float)
    return [f(np.ones_like(x)), f(x), f(x * x), f(sdelki), f(np.abs(poz))]


def sobrat(daty, x, poz, sdelki, rezhim, gody: tuple = GODY_V7) -> dict:
    """Поле `agr` по дневным рядам портфеля [T]: x, |poz|, сделок/день; rezhim — rezhim_dnya(daty, IMOEX). Год вне gody — отказ."""
    d, x = np.asarray(daty, "datetime64[D]"), np.asarray(x, float)
    if not (d.shape == x.shape == np.shape(poz) == np.shape(sdelki) == np.shape(rezhim)) or not np.isfinite(x).all():
        raise ValueError("agregaty: daty, x, poz, sdelki, rezhim — один размер, x без nan")
    g = _god(d)
    if not np.isin(g, gody).all():
        raise ValueError(f"agregaty: дата вне годов {gody[0]}–{gody[-1]}")
    ig = g - gody[0]
    nedeli = pd.Series(x, index=pd.DatetimeIndex(d)).groupby(pd.DatetimeIndex(d).to_period("W-FRI"))
    wsum = nedeli.sum()
    wlast = nedeli.apply(lambda s: s.index[-1]).to_numpy("datetime64[D]")   # последний торговый день недели в ряду
    w = wsum.to_numpy(float)
    kw = _god(wlast) - gody[0]
    wk = [np.bincount(kw, weights=w**p, minlength=len(gody)) for p in range(5)]
    sd = lambda a: [float(v) for v in a]    # noqa: E731
    return {"v": VERSIYA, "gody": list(gody), "d": [sd(a) for a in _po_klyucham(ig, len(gody), x, poz, sdelki)],
            "w": [sd(a) for a in wk], "r": [sd(a) for a in _po_klyucham(np.asarray(rezhim, int), 3, x, poz, sdelki)]}


def sharpe_iz_summ(n: float, s: float, q: float) -> float:
    """Sharpe проекта (Д7): metriki_e.sharpe — годовая шкала, ddof = 1, избыточная доходность — из n, Σx, Σx²;
    n < 2 → nan; sd ≤ 1e-12 → 0. Масштаб берётся из metriki_e (отношение sharpe / sr_dnevnoy), не литералом."""
    if n < 2:
        return float("nan")
    var = max((q - s * s / n) / (n - 1), 0.0)
    return float(SHKALA * s / n / sqrt(var)) if sqrt(var) > 1e-12 else 0.0


def msr_iz_summ(n: float, s1: float, s2: float, s3: float, s4: float, g: int = 52) -> float:
    """mSR Фавра–Галеано (формула metriki_e.msr) из n, Σw…Σw⁴; nan при n < 3 или mVaR ≤ 0."""
    if n < 3:
        return float("nan")
    m = s1 / n
    c2 = s2 / n - m * m
    c3 = s3 / n - 3 * m * s2 / n + 2 * m**3
    c4 = s4 / n - 4 * m * s3 / n + 6 * m * m * s2 / n - 3 * m**4
    sk, ku = (c3 / c2**1.5, c4 / c2**2 - 3.0) if c2 > 1e-24 else (0.0, 0.0)
    sd, z = sqrt(max(c2 * n / (n - 1), 0.0)), Z95
    zcf = z + (z**2 - 1) * sk / 6 + (z**3 - 3 * z) * ku / 24 - (2 * z**3 - 5 * z) * sk**2 / 36
    mvar = -(m + zcf * sd) * sqrt(g)
    return float(m * g / mvar) if mvar > 1e-12 else float("nan")


def agregaty(stroka: dict) -> dict | None:
    """Из строки ledger — Sharpe дн. и mSR нед. по годам, итог, Sharpe по режимам, сделки и экспозиция; None, если `agr` нет (2.0–2.2 без агрегатов)."""
    a = stroka.get("agr")
    if not a:
        return None
    gody, (n, s, q, sd, ex), w = a["gody"], a["d"], a["w"]
    sh = lambda i: sharpe_iz_summ(n[i], s[i], q[i])   # noqa: E731
    rn, rs, rq, rsd, rex = a["r"]
    return {"gody": gody,
            "sharpe_gody": {g: sh(i) for i, g in enumerate(gody)},
            "sharpe_itog": sharpe_iz_summ(sum(n), sum(s), sum(q)),
            "msr_ned_gody": {g: msr_iz_summ(*(w[p][i] for p in range(5))) for i, g in enumerate(gody)},
            "msr_ned_itog": msr_iz_summ(*(sum(w[p]) for p in range(5))),
            "sharpe_rezhimy": {r: sharpe_iz_summ(rn[i], rs[i], rq[i]) for i, r in enumerate(REZHIMY)},
            "sdelok_gody": {g: sd[i] for i, g in enumerate(gody)},
            "ekspoziciya_gody": {g: (ex[i] / n[i] if n[i] else float("nan")) for i, g in enumerate(gody)}}
