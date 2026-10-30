"""Диагностические метрики Ж-148 (Д7: одно определение на метрику). Только диагностика:
в пороги, ворота, парные сравнения и предрегистрацию не входят (там Sharpe дн. и mSR нед.).

Sharpe дн. — не копия: `sharpe` из fastbt/metriki_e.py (ddof=1, √252; № 12–13).
Экспозиция — валовая, среднее по дням окна от |лонг| + |шорт| (дни без позиций = 0).
NaN всегда сопровождается причиной: prichina_norm / prichina_sdelok.
"""
from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd

from fastbt.metriki_e import sharpe as sharpe_dn  # noqa: F401  (единственное определение Sharpe дн.)

NAN = float("nan")
GOD = 252.0
K_NORM = "Sharpe норм. (при полной загрузке, диагностика)"
K_SDELOK = "Sharpe по сделкам (диагностика)"
K_CENA = "цена простоев (диагностика)"


def ekspoziciya(poz_dn) -> float:
    """Среднее по дням окна доли капитала в позициях. poz_dn: 1-D — один тикер (|w| по дням);
    2-D или DataFrame «дни × тикеры» — валовая за день Σ|w| по тикерам (не |Σw|)."""
    x = np.abs(np.asarray(poz_dn, float))
    if x.ndim == 2:
        x = x.sum(axis=1)
    return float(x.mean()) if len(x) else 0.0


def prichina_norm(poz_dn) -> str | None:
    return "экспозиция 0: нет позиций в окне" if ekspoziciya(poz_dn) <= 1e-12 else None


def sharpe_norm(r_dn, poz_dn) -> float:
    """Sharpe дн. / экспозиция; при экспозиции 0 — NaN (причина: prichina_norm)."""
    if prichina_norm(poz_dn):
        return NAN
    return float(sharpe_dn(r_dn) / ekspoziciya(poz_dn))


def cena_prostoev(r_dn, poz_dn) -> float:
    """Sharpe норм. − Sharpe дн."""
    return float(sharpe_norm(r_dn, poz_dn) - sharpe_dn(r_dn))


def prichina_sdelok(pnl_sdelok_izb, let_okna) -> str | None:
    x = np.asarray(pnl_sdelok_izb, float)
    if len(x) < 2:
        return "сделок < 2"
    if let_okna <= 0:
        return "длина окна 0 лет"
    if np.std(x, ddof=1) <= 1e-12:
        return "СКО PnL сделок 0"
    return None


def sharpe_sdelok(pnl_sdelok_izb, let_okna) -> float:
    """mean/std(ddof=1) × √(N/лет). Вход — ГОТОВОЕ избыточное PnL сделки (после комиссии входа/выхода
    и платы за шорт; избыточное над ставкой ЦБ за дни удержания); здесь это не считается.
    Откуда брать: исполнитель fastbt/golaya_v7_isp.py / движок. N < 2 — NaN (причина: prichina_sdelok)."""
    if prichina_sdelok(pnl_sdelok_izb, let_okna):
        return NAN
    x = np.asarray(pnl_sdelok_izb, float)
    return float(x.mean() / np.std(x, ddof=1) * sqrt(len(x) / let_okna))


def sdelok_v_god(n_sdelok, let_okna) -> float:
    return float(n_sdelok / let_okna) if let_okna > 0 else NAN


def uderzhanie_srednee(dni_uderzh) -> float:
    x = np.asarray(dni_uderzh, float)
    return float(x.mean()) if len(x) else NAN


def _stroka(r, poz, god_s, pnl, ud) -> dict:
    let = len(r) / GOD
    pr = [p for p in (prichina_norm(poz), prichina_sdelok(pnl, let)) if p]
    return {
        "Sharpe дн.": float(sharpe_dn(r)) if len(r) else NAN,
        K_NORM: sharpe_norm(r, poz) if len(r) else NAN,
        K_SDELOK: sharpe_sdelok(pnl, let),
        K_CENA: cena_prostoev(r, poz) if len(r) else NAN,
        "N сделок в год": sdelok_v_god(len(pnl), let),
        "удержание, дн.": uderzhanie_srednee(ud),
        "причина NaN": "; ".join(pr),
    }


def panel_diag(daty, r_dn, poz_dn, god_sdelok, pnl_sdelok_izb, dni_uderzh) -> pd.DataFrame:
    """Панель по годам и «итого». god_sdelok — год входа каждой сделки; pnl_sdelok_izb — см. sharpe_sdelok;
    длина года = число дней окна в нём / 252."""
    gody = pd.DatetimeIndex(daty).year.values
    r, poz = np.asarray(r_dn, float), np.asarray(poz_dn, float)
    gs, pnl, ud = np.asarray(god_sdelok), np.asarray(pnl_sdelok_izb, float), np.asarray(dni_uderzh, float)
    rows = {}
    for g in sorted(set(gody)):
        m, ms = gody == g, gs == g
        rows[int(g)] = _stroka(r[m], poz[m], gs[ms], pnl[ms], ud[ms])
    rows["итого"] = _stroka(r, poz, gs, pnl, ud)
    return pd.DataFrame.from_dict(rows, orient="index")
