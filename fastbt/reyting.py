"""МСП, лист «рейтинг» (spec Ф3 = §7, §3; Ж-099 п. 4–5, Ж-109, Ж-114 п. 8): score = сторона · ранг силы · w_λ · g.

Всё по данным к 18:50 дня d; фазы — готовые `fazy.faza_rynka` / `fazy.faza_tikera` (лаг 1 внутри); терцили f1/f2 —
готовый `intraday_priznaki.tercile_cell` (окно d−252…d−1). Данные ≥ 2020 не читаются (ValueError).

Отступления от «вызывать готовое» (модулей нет в базе; заменить вызовом, когда появятся):
  * ранг силы — `rangi_v6.rang_obshchiy` (обёртка `rang_sily(sila[date×тикер]) -> [date×тикер]`, пул по тикерам);
  * таблица весов §2.1 — `ves` (лист Ф1 `fazy_vesa`), g_нов / g_ликв — `uroven_v3`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt import fazy, rangi_v6
from fastbt.daily_1850 import CUT_MIN
from fastbt.intraday_priznaki import tercile_cell

END = pd.Timestamp("2020-01-01")
OKNO_RANG, MIN_NABL = 252, 20
KAPITAL, DOLYA_OBOROTA = 200e6, 0.05          # ❓ Д-1 v3: капитал 200 млн; ставка ≤ 5 % медианного оборота
SEMYA = {"S1": "kt", "S2": "kt", "S3": "tr", "S4": "tr", "S5": "dp", "S6": "dp", "S7": "ks", "S8": "ks", "S9": "ks",
         "S10": "tr", "S11": "tr", "S12": "tr"}   # контртренд / тренд (+ импульс) / диапазон / кросс-секция


def _do_2020(idx) -> None:
    if (pd.DatetimeIndex(idx) >= END).any():
        raise ValueError("данные ≥ 2020 не читаются")


def _w_baz(sem: str, faza: str) -> float:
    trend, vol = faza.split("_")
    if sem == "kt":
        return 0.3 if vol == "стресс" else (1.0 if trend == "боковик" else 0.6)
    if sem == "tr":
        return 0.7 if vol == "стресс" else (0.3 if trend == "боковик" else 1.0)
    if sem == "dp":
        return 1.0 if (trend == "боковик" and vol == "спокойно") else 0.4
    return 1.0


def _s_storona(sem: str, trend: str, side: float) -> float:
    protiv = (side > 0 and trend == "падение") or (side < 0 and trend == "рост")
    if sem == "ks" or not protiv:
        return 1.0
    if sem == "kt":
        return 0.3 if side > 0 else 0.0           # лонг в П ×0,3; шорт в Р ×0 (Ж-103)
    return 0.0 if sem == "tr" else (1.0 if side > 0 else 0.0)   # диапазон: лонг в П ×1, шорт в Р ×0


def zhestkost(w: float, lam: float) -> float:
    """w_λ = 1 − λ(1 − w): λ = 0 — фазы выключены, λ = 1 — полный вес."""
    return 1.0 - lam * (1.0 - w)


def ves(sk: str, faza_r, faza_s, side: float, lam: float) -> float:
    """w_λ = zhestkost(W · S · M · Q) для Sk. faza_* — «тренд_волатильность» из fazy; NaN → вес 1 (Ж-114 п. 3)."""
    sem = SEMYA[sk]
    if not isinstance(faza_r, str) or not isinstance(faza_s, str):
        return 1.0
    tr_r, tr_s = faza_r.split("_")[0], faza_s.split("_")[0]
    w = _w_baz(sem, faza_r) * _s_storona(sem, tr_r, side)
    if sem != "ks":
        q = 0.5 if (side > 0 and tr_s == "падение") or (side < 0 and tr_s == "рост") else 1.0
        w *= (1.0 if faza_s == faza_r else 0.7) * q
    return zhestkost(w, lam)


def fazy_dnya(tikery, daty, dannye: dict) -> tuple[pd.Series, pd.DataFrame]:
    """(фаза рынка [d], фаза сектора [d × тикер]) — готовые функции fazy (по ≤ d−1)."""
    _do_2020(daty)
    fr = fazy.faza_rynka(daty, dannye)["faza"]
    fs = pd.DataFrame({t: fazy.faza_tikera(t, daty, dannye)["faza"] for t in tikery})
    return fr, fs


def ves_panel(sk: str, storona: pd.DataFrame, faza_r: pd.Series, faza_s: pd.DataFrame, lam: float) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=storona.index, columns=storona.columns)
    for t in storona.columns:
        out[t] = [ves(sk, fr, fs, s, lam) for fr, fs, s in zip(faza_r.reindex(storona.index),
                                                                      faza_s[t].reindex(storona.index), storona[t])]
    return out


def rang_sily(sila: pd.DataFrame, okno: int = OKNO_RANG, min_nabl: int = MIN_NABL) -> pd.DataFrame:
    """Ранг пулом по тикерам (Д7: `rangi_v6.rang_obshchiy`; окно d−okno…d−1, строго меньше, < min_nabl — NaN)."""
    r = rangi_v6.rang_obshchiy(sila.to_numpy(float).T, okno, min_nabl).T
    return pd.DataFrame(r, index=sila.index, columns=sila.columns)


def g_novost(r: pd.Series, sigma_prev: pd.Series, obyom: pd.Series) -> pd.Series:
    """1, если |r_d| < 2,5·σ̂_{d−1} и объём_d < 2·mean(объём d−20…d−1); иначе 0. σ̂_{d−1} подаётся уже сдвинутой."""
    sr = obyom.rolling(20).mean().shift(1)
    g = ((r.abs() < 2.5 * sigma_prev) & (obyom < 2 * sr)).astype(float)
    return g.where(r.notna() & sigma_prev.notna() & sr.notna())


def oborot_dnya(bars: pd.DataFrame) -> pd.Series:
    """Оборот дня из 1-мин баров [timestamp, close, volume]: бары с началом ≥ 18:50 не входят."""
    _do_2020(bars["timestamp"])
    ts = bars["timestamp"]
    b = bars[(ts.dt.hour * 60 + ts.dt.minute) < CUT_MIN]
    return (b["close"] * b["volume"]).groupby(b["timestamp"].dt.normalize()).sum()


def g_likvidnost(razmer_rub: pd.Series, oborot: pd.Series) -> pd.Series:
    """1, если ставка при m = 1 ≤ 5 % медианы оборота за d−19…d; иначе 0."""
    med = oborot.rolling(20).median()
    return (razmer_rub.abs() <= DOLYA_OBOROTA * med).astype(float).where(med.notna() & razmer_rub.notna())


def g_f(storona: pd.Series, f: pd.Series, znak: int) -> pd.Series:
    """g по признаку: терциль d по порогам d−252…d−1; dir = znak в верхнем, −znak в нижнем, 0 в среднем;
    a = сторона · dir; g = 0,5 при a = −1, иначе 1. znak: f2 → +1, f1 → −1. Нет окна → 1."""
    napr = (tercile_cell(f) - 1.0) * znak
    return pd.Series(np.where(storona * napr == -1, 0.5, 1.0), index=f.index)


def skor(storona, rang, w, g):
    """score = сторона · ранг · w_λ · g (ранг ∈ [0; 1], |score| ≤ 1)."""
    return storona * rang * w * g


def reyting(sk: str, storona: pd.DataFrame, sila: pd.DataFrame, g: pd.DataFrame, faza_r: pd.Series,
            faza_s: pd.DataFrame, lam: float, porog: float = 0.0) -> pd.DataFrame:
    """Панель score [d × тикер] для одной Sk; |score| < порога → NaN (кандидата нет)."""
    s = skor(storona, rang_sily(sila), ves_panel(sk, storona, faza_r, faza_s, lam), g)
    return s.where(s.abs() >= porog) if porog > 0 else s
