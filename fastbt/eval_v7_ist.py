"""Источники куба v7 для `sreda` (V7-8b, Ж-155 п. 7): 10-минутки S10–S12, панель сечения S8/S9, ряды `otnositelno` S1–S4.

Как `golaya_v7.iz_barov` (#881): корпсобытия — ДО агрегации 10-минуток; сигнальные ряды скорректированы на дивиденды
(множитель дня дневки × 10-минутки, `_korr10`, mutations № 33); исполнение на реальных ценах — дело слоя исполнения.
Панель: дни, где есть ≥ 8 тикеров (или все, если тикеров меньше); sigma = std лог-доходностей за 22 дня (≥ 15); ok — есть open 10:00
следующего дня сетки (LAG). Ряды otnositelno: `otnositelno_v7.fon_ryad` на закрытиях панели для sektor и imoex.
`sha` (Д5, № 7/38) — все ряды: 10-минутки каждого тикера, панель (закрытия, IMOEX, метки, σ, ok), фон otnositelno; входит в ключ точки.
Чтение файлов — `zagruzit(period, tikery)`: polnaya = границы wf (2014-06…2026-02-28), только при отметке снятого замка
(`period_v2.proverit_chtenie`); граница чтения строго < конец + 1 день. Сборка `sobrat` файлов не читает (синтетика в тестах).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import bars_agg, daily_1850 as d18, dividendy, kross_seksiya as ks, otnositelno_v7 as otn, period_v2
from fastbt.eval_v7 import Ist7
from fastbt.golaya_v7_isp import korpsobytiya
from fastbt.signals.v7_kross import Panel

POLNAYA = "polnaya"
PERIOD_FAJL = {POLNAYA: "wf"}          # polnaya читается по границам и замку wf
REZHIMY_OTN = ("sektor", "imoex")
MIN_TIK = 8


@dataclass
class IstochnikiV7:
    """b10 {тикер: 10-минутки prep_10m} — S10–S12; panel (v7_kross.Panel) — S8, S9; fon {rezhim: (fon, flag)} — otnositelno S1–S4."""
    b10: dict
    panel: Panel
    fon: dict
    sha: str = ""                  # Д5: хэш всех рядов; входит в ключ точки (как IstochnikiV6.sha)

    def ist7(self) -> Ist7:
        return Ist7(b10=self.b10, panel=self.panel)


def korr10(b: pd.DataFrame, f: pd.Series) -> pd.DataFrame:
    """10-минутки × множитель дня дневки (нет дня в f — 1.0 только для дней вне дневки: они не попадают в сигнал)."""
    k = f.reindex(pd.DatetimeIndex(b["timestamp"].dt.normalize())).fillna(1.0).to_numpy()
    return b.assign(**{c: b[c] * k for c in ("open", "high", "low", "close")})


def _mnozhitel(t: str, dts, close: np.ndarray) -> np.ndarray:
    return dividendy.adjust_ohlc(t, dts, close, np.ones(len(close)))[0]


def _h(h, x) -> None:
    if isinstance(x, (pd.DataFrame, pd.Series)):
        h.update(pd.util.hash_pandas_object(x, index=True).to_numpy().tobytes())
        h.update(str(list(x.columns) if hasattr(x, "columns") else [x.name]).encode())
    else:
        a = np.asarray(x)
        h.update(pd.util.hash_array(a.ravel()).tobytes() if a.dtype == object else np.ascontiguousarray(a).tobytes())
        h.update(str(a.shape).encode())


def sha_istochnikov(b10: dict, panel: Panel, fon: dict) -> str:
    """Д5: sha256 значений, индексов и имён всех рядов источников. Подмена любого значения меняет хэш."""
    h = hashlib.sha256()
    for t in sorted(b10):
        h.update(b"b10:" + t.encode())
        _h(h, b10[t])
    h.update(b"panel:" + ",".join(panel.tikery).encode())
    for x in (panel.c, panel.imoex, panel.daty, panel.metki, panel.sigma, panel.ok):
        _h(h, x)
    for r in sorted(fon):
        h.update(b"fon:" + r.encode())
        _h(h, fon[r][0])
        _h(h, fon[r][1])
    return h.hexdigest()


def sobrat(raw: dict, imoex: pd.Series, karta: pd.DataFrame, tikery, nachalo, konec, korp=korpsobytiya, mnozh=_mnozhitel) -> IstochnikiV7:
    """raw {тикер: 10-минутки (timestamp, OHLCV)}, imoex — close 18:50 по дням, karta (ticker, indeks), [nachalo; konec) — окно чтения."""
    tk, nachalo, konec = list(tikery), pd.Timestamp(nachalo), pd.Timestamp(konec)
    cl, op, b10 = {}, {}, {}
    for t in tk:
        b = korp(raw[t][lambda x: (x["timestamp"] >= nachalo) & (x["timestamp"] < konec)], t)     # корпсобытия до агрегации
        dd = d18.aggregate(b, t).set_index("date")
        f = pd.Series(mnozh(t, dd.index.values, dd["close"].to_numpy()), dd.index)
        cl[t] = dd["close"] * f
        bo = b[b["timestamp"].dt.hour * 60 + b["timestamp"].dt.minute == d18.OPEN_MIN]
        op[t] = bo.groupby(bo["timestamp"].dt.normalize())["open"].first()
        b10[t] = bars_agg.prep_10m(korr10(b, f))
    c = pd.DataFrame(cl)[tk]
    c = c[c.notna().sum(axis=1) >= min(MIN_TIK, len(tk))]
    daty, im = pd.DatetimeIndex(c.index), imoex[imoex.index < konec].reindex(c.index)
    o = pd.DataFrame(op)[tk].reindex(c.index)
    metki = ks.sektor_na_d(karta.assign(s=pd.Timestamp("2000-01-01")), daty, tk)
    sigma = np.log(c).diff().rolling(22, min_periods=15).std().to_numpy()
    pn = Panel(daty=daty.values, tikery=tk, c=c, imoex=im, metki=metki, sigma=sigma,
               ok=o.notna().shift(-ks.LAG, fill_value=False).to_numpy())
    fon = {r: otn.fon_ryad(c, im, metki, r) for r in REZHIMY_OTN}
    return IstochnikiV7(b10=b10, panel=pn, fon=fon, sha=sha_istochnikov(b10, pn, fon))


def zagruzit(period: str, tikery, main: Path | None = None, fajly=None) -> IstochnikiV7:
    """Источники v7 из файлов (< конец + 1 день). period = polnaya | wf; замок wf — `proverit_chtenie`; иначе отказ."""
    from fastbt import istochniki_v6 as iv6
    if period not in PERIOD_FAJL and period != "wf":
        raise NotImplementedError(f"период {period!r}: источники v7 читаются только для polnaya (wf)")
    p = PERIOD_FAJL.get(period, period)
    f = fajly or iv6.Fajly(iv6.konec(p), main, p)
    raw = {t: f.akcii(t) for t in tikery}
    b = f.indeks("imoex")
    if b is None:
        raise FileNotFoundError("IMOEX: нет файла баров")
    im = d18.aggregate(b, "IMOEX").set_index("date")["close"].astype(float)
    karta = pd.read_csv(otn.KARTA_CSV, usecols=["ticker", "indeks"])
    return sobrat(raw, im, karta, tikery, period_v2.PERIODY[p].progrev, f.end)
