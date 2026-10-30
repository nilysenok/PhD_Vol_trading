"""События этапа 1 (Ж-047 п. 3, контракт К3): фильтры входа point-in-time.

Массив bool по торговым датам: True = вход по close d запрещён. Действуют только на новый вход
(открытую позицию не снимают, контракт §1.1).
- cbr: заседание ЦБ приходится на следующий торговый день d+1 и known_from ≤ d.
- div: last_cum_date = d (последний день с правом на дивиденд) и announce_date ≤ d, по тикеру.
Ось дивидендов включается только при div_verified=True; флаг означает «таблица отфильтрована по Д8»: в ней лишь годные строки
свода (`load_div_svod`, ❓5 закрыт построчно). Без свода sobytiya() отдаёт только net и cbr. Даты ≥ 2022 → PermissionError.
Окна v6 (К18 sob_do, К19 sob_posle): по умолчанию do=1, posle=0 — побайтно v5.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

END = pd.Timestamp("2022-01-01")
CBR_CSV = Path("data/calendar/cbr.csv")
DIV_CSV = Path("docs/issledovaniya/2026-10-27_dividendy_2017_2021.csv")
SVOD_CSV = Path(__file__).resolve().parent.parent / "docs/issledovaniya/div_sverka_svod.csv"


def _dates(dates) -> pd.DatetimeIndex:
    d = pd.DatetimeIndex(dates)
    from fastbt import zona
    if len(d) and d.max() >= zona.end():
        raise PermissionError(f"события: даты ≥ {zona.end().date()}")
    return d


def _okno(out: np.ndarray, d: pd.DatetimeIndex, a: int, known, do: int, posle: int) -> None:
    """Якорь a — торговый день перед событием (v5: блок только на a). Окно: a−do+1 … a+posle
    (do — дней до события включая якорь; posle — дней события и следом). Point-in-time:
    день k блокируется, только если дата известности ≤ d[k] (задним числом не блокируем)."""
    for k in range(max(a - do + 1, 0), min(a + posle, len(d) - 1) + 1):
        if known <= d[k]:
            out[k] = True


def _proverka(do: int, posle: int) -> None:
    if do < 1 or posle < 0:
        raise ValueError(f"sob_do ≥ 1, sob_posle ≥ 0: получено {do}, {posle}")


def cbr_block(dates, cbr: pd.DataFrame, do: int = 1, posle: int = 0) -> np.ndarray:
    """cbr: колонки date, known_from. v5 (do=1, posle=0): блок на d, если заседание — на d+1."""
    _proverka(do, posle)
    d = _dates(dates)
    out = np.zeros(len(d), bool)
    if len(d) < 2:
        return out
    for m, kf in zip(pd.to_datetime(cbr["date"]), pd.to_datetime(cbr["known_from"])):
        j = int(d.searchsorted(m, side="left"))     # первый торговый день ≥ даты заседания
        if j == 0 or j >= len(d):
            continue
        _okno(out, d, j - 1, kf, do, posle)
    return out


def div_block(dates, div: pd.DataFrame, ticker: str, do: int = 1, posle: int = 0) -> np.ndarray:
    """div: колонки ticker, last_cum_date, announce_date. v5: блок на d = last_cum_date, если announce ≤ d."""
    _proverka(do, posle)
    d = _dates(dates)
    out = np.zeros(len(d), bool)
    x = div[div["ticker"] == ticker]
    for lc, an in zip(pd.to_datetime(x["last_cum_date"]), pd.to_datetime(x["announce_date"])):
        i = int(d.searchsorted(lc, side="right")) - 1   # last_cum_date не торговый → предыдущий торговый
        if i >= 0 and an <= d[i]:
            _okno(out, d, i, an, do, posle)
    return out


def load_cbr(path=CBR_CSV) -> pd.DataFrame:
    c = pd.read_csv(path, parse_dates=["date", "known_from"])
    from fastbt import zona
    return c[c["date"] < zona.end()]


def load_div(path=DIV_CSV) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["record_date", "last_cum_date", "announce_date"])


def load_div_svod(path=SVOD_CSV) -> pd.DataFrame | None:
    """Д8: только годные строки свода (scripts/div_svod.py), announce_date = announce_pit (более поздняя дата источников),
    last_cum_date — по правилу dividendy.last_cum (T+2/T+1 по барам). Свода нет — None (ось дивидендов остаётся закрытой)."""
    path = Path(path)
    if not path.exists():
        return None
    s = pd.read_csv(path, dtype={"godna": int})
    s = s[s["godna"] == 1].reset_index(drop=True)
    from fastbt import dividendy
    rec = pd.to_datetime(s["record_date"])
    return pd.DataFrame({"ticker": s["ticker"], "record_date": rec, "last_cum_date": dividendy.last_cum(rec),
                         "announce_date": pd.to_datetime(s["announce_pit"]), "dps_rub": s["dps_rub"]})


def trebovat_svod(path=None) -> None:
    """Старт прогона v6: нет свода Д8 — отказ ДО счёта (иначе точки с sob_tip ∋ div падают ValueError посреди цепочки)."""
    p = Path(path or SVOD_CSV)
    if not p.exists():
        raise SystemExit(f"нет свода Д8 {p}: ось дивидендов v6 закрыта, точки sob_tip ∋ div дадут ошибку. На сервере свод едет "
                         "усечённой копией манифеста (data/server_2021/docs/issledovaniya/div_sverka_svod.csv -> накладка в корень "
                         "кода razvernut.sh); пересоберите perenos/ с коммита, где свод есть в docs/server/manifest_dannyh_2021.tsv")


def sobytiya(dates, tickers, cbr: pd.DataFrame, div: pd.DataFrame | None = None,
             div_verified: bool = False, do: int = 1, posle: int = 0) -> dict:
    """{имя: {тикер: bool}}; net — ничего не запрещает. div/cbr_div — только при div_verified (таблица по Д8)."""
    d = _dates(dates)
    c = cbr_block(d, cbr, do, posle)
    out = {"net": {t: np.zeros(len(d), bool) for t in tickers},
           "cbr": {t: c.copy() for t in tickers}}
    if div_verified:
        if div is None:
            raise ValueError("div_verified=True требует таблицу дивидендов")
        dv = {t: div_block(d, div, t, do, posle) for t in tickers}
        out["div"] = dv
        out["cbr_div"] = {t: dv[t] | c for t in tickers}
    return out
