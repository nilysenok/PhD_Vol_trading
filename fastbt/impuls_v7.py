"""Л1.2 (Ж-146): сигналы S10–S12 из 10-минуток — импульс на 1ч и 2ч (определения: tolkovaniya [31.10 00:04], в1).

S10 — пробой канала из KANAL полных 1ч-баров на последнем полном баре дня; S11 — то же на 2ч (эталон — Ж-107,
`golaya_tf_v5.impuls_signal`, при t_sig = 18:30 совпадает побайтно); S12 — согласие: S10 и S11 в один день в одну сторону.
Бар полный, если он закончился ≤ t_sig (момент решения T): бар с концом позже t_sig в сигнал не входит, дни после d тоже.
Вход — не раньше d+1 (Д2) или в T: это дело исполнителя (Л1.3), здесь только сигнал дня. Читает только переданную ленту;
лента с днями ≥ period_v2.chtenie_konec(период ленты) отвергается: по умолчанию train4 (2020-01-01), polnaya (2025-01-01 до акта) — явно, `attrs["period"]`. Движок не трогает. Статус: разработка, до аудита.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt import bars_agg as ba, period_v2

KANAL = 20
ATR_OKNO = 14      # баров того же ТФ для ATR силы пробоя (V7-5)
TF = {"S10": "1h", "S11": "2h"}
PERIOD_UMOLCH = "train4"      # период ленты без отметки: разработка, < 2020-01-01 (как было); polnaya ленте выставляет загрузчик (`b10.attrs["period"]`)


def granitsa(b10: pd.DataFrame) -> pd.Timestamp:
    """Граница чтения ленты (исключительно) — period_v2.chtenie_konec(период ленты); одно определение с периодами проекта."""
    return period_v2.chtenie_konec(b10.attrs.get("period", PERIOD_UMOLCH))


def _minuty(t_sig: str) -> int:
    """'ЧЧ:ММ' → минуты от полуночи; вне [10:00; 18:50] — ValueError."""
    try:
        h, m = (int(x) for x in str(t_sig).split(":"))
    except ValueError:
        raise ValueError(f"t_sig {t_sig!r}: ожидается 'ЧЧ:ММ'") from None
    t = h * 60 + m
    if not ba.OPEN_MIN <= t <= ba.CLOSE_MIN:
        raise ValueError(f"t_sig {t_sig!r} вне окна сигналов 10:00–18:50")
    return t


def proboj(b10: pd.DataFrame, tf: str, t_sig: str, kanal: int = KANAL) -> pd.Series:
    """Сигнал дня d (±1/0) = пробой канала kanal полных баров tf на последнем полном баре дня (закрыт ≤ t_sig)."""
    k = ba.K[tf]
    n_full = (_minuty(t_sig) - ba.OPEN_MIN) // (ba.STEP * k)       # полных баров tf в дне к t_sig
    f = ba.resample(b10, k)
    f = f[f["slot"] < n_full].sort_values(["date", "slot"]).reset_index(drop=True)
    hi = f["high"].shift(1).rolling(kanal).max()
    lo = f["low"].shift(1).rolling(kanal).min()
    s = np.where(f["close"] > hi, 1, np.where(f["close"] < lo, -1, 0))
    s = pd.Series(s, index=f["date"].to_numpy())
    return s[f["slot"].to_numpy() == n_full - 1]


def sila_proboja(b10: pd.DataFrame, tf: str, t_sig: str, kanal: int = KANAL, atr_okno: int = ATR_OKNO) -> pd.Series:
    """Знаковая сила пробоя дня d: (close − граница канала) / ATR(tf); лонг > 0 (над верхней), шорт < 0 (под нижней), иначе 0.

    ATR — среднее истинного диапазона atr_okno баров того же ТФ, строго до бара решения (shift(1), М-d1)."""
    k = ba.K[tf]
    n_full = (_minuty(t_sig) - ba.OPEN_MIN) // (ba.STEP * k)
    f = ba.resample(b10, k)
    f = f[f["slot"] < n_full].sort_values(["date", "slot"]).reset_index(drop=True)
    pc = f["close"].shift(1)
    tr = pd.concat([f["high"] - f["low"], (f["high"] - pc).abs(), (f["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_okno).mean().shift(1)
    hi = f["high"].shift(1).rolling(kanal).max()
    lo = f["low"].shift(1).rolling(kanal).min()
    raz = np.where(f["close"] > hi, f["close"] - hi, np.where(f["close"] < lo, f["close"] - lo, 0.0))
    ok = (atr > 0).to_numpy()
    # пробой при ATR не определён или 0: сила не определена = NaN (Д4: без подстановки); нет пробоя — 0
    s = pd.Series(np.where(ok, raz / atr.where(atr > 0, 1.0), np.where(raz != 0, np.nan, 0.0)), index=f["date"].to_numpy())
    return s[f["slot"].to_numpy() == n_full - 1]


def sila_dnya(b10: pd.DataFrame, sid: str, t_sig: str = "18:30") -> pd.Series:
    """Знаковая сила дня для sid ∈ {S10, S11, S12}; S12 — сторона согласия × среднее |силы| S10 и S11."""
    signal_dnya(b10, sid, t_sig)                                    # те же проверки sid, t_sig, границы 2020
    if sid != "S12":
        return sila_proboja(b10, TF[sid], t_sig)
    a, b = (sila_proboja(b10, TF[x], t_sig) for x in ("S10", "S11"))
    side = signal_dnya(b10, "S12", t_sig)                           # сторона прежняя; сила NaN, если у одной ноги не определена
    mag = (a.abs().reindex(side.index) + b.abs().reindex(side.index)) / 2
    return (side * mag).where(side != 0, 0.0)


def soglasie(a: pd.Series, b: pd.Series) -> pd.Series:
    """Согласие двух сигналов: сторона, если обе одинаковы и не нулевые, иначе 0 (день вне одного из рядов = 0)."""
    idx = a.index.union(b.index)
    x, y = a.reindex(idx).fillna(0).astype(int), b.reindex(idx).fillna(0).astype(int)
    return x.where(x == y, 0)


def signal_dnya(b10: pd.DataFrame, sid: str, t_sig: str = "18:30") -> pd.Series:
    """Сигнал дня ±1/0 для sid ∈ {S10, S11, S12}; b10 — 10-минутки (date, slot, OHLCV), как после `bars_agg.prep_10m`."""
    if sid not in ("S10", "S11", "S12"):
        raise ValueError(f"sid {sid!r}: ожидается S10, S11 или S12")
    _minuty(t_sig)
    gr = granitsa(b10)
    if len(b10) and pd.Timestamp(b10["date"].max()) >= gr:
        raise ValueError(f"лента с днями ≥ {gr:%Y-%m-%d}: чтение за границей периода {b10.attrs.get('period', PERIOD_UMOLCH)} (2020 для train4) запрещено")
    if sid == "S12":
        return soglasie(proboj(b10, TF["S10"], t_sig), proboj(b10, TF["S11"], t_sig))
    return proboj(b10, TF[sid], t_sig)
