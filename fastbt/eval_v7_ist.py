"""Источники куба v7 для `sreda` (V7-8b, Ж-155 п. 7): 10-минутки S10–S12, панель сечения S8/S9, ряды `otnositelno` S1–S4.

Как `golaya_v7.iz_barov` (#881): корпсобытия — ДО агрегации 10-минуток; сигнальные ряды скорректированы на дивиденды
(множитель дня дневки × 10-минутки, `_korr10`, mutations № 33); исполнение на реальных ценах — дело слоя исполнения.
Панель: дни, где есть ≥ 8 тикеров (или все, если тикеров меньше); sigma = std лог-доходностей за 22 дня (≥ 15); ok — есть open 10:00
следующего дня сетки (LAG). Ряды otnositelno: `otnositelno_v7.fon_ryad` на закрытиях панели для sektor и imoex.
`sha` (Д5, № 7/38) — все ряды: 10-минутки каждого тикера, панель (закрытия, IMOEX, метки, σ, ok), фон otnositelno; входит в ключ точки.
Чтение файлов — `zagruzit(period, tikery)`: polnaya — только при отметке снятого замка (`period_v2.proverit_chtenie`); граница чтения
строго < min(конец + 1 день, DO_AKTA = 2025-01-01): до акта владельца (Ж-155) строки ≥ 2025 не читаются, в том числе индексы. Сборка `sobrat` файлов не читает (синтетика в тестах).
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
PERIOD_FAJL = {POLNAYA: "wf"}          # замок polnaya — отметка владельца, как у wf
REZHIMY_OTN = ("sektor", "imoex")
MIN_TIK = 8


@dataclass
class IstochnikiV7:
    """b10 {тикер: 10-минутки prep_10m} — S10–S12; panel (v7_kross.Panel) — S8, S9; fon {rezhim: (fon, flag)} — otnositelno S1–S4."""
    b10: dict
    panel: Panel
    fon: dict
    sha: str = ""                  # Д5: хэш всех рядов; входит в ключ точки (как IstochnikiV6.sha)
    b10_real: dict | None = None   # 10-минутки без поправки на дивиденды: цена исполнения cena_T (#370)

    def ist7(self) -> Ist7:
        from fastbt.eval_v7_otn import OtnIst
        return Ist7(b10=self.b10, panel=self.panel, otn=OtnIst(imoex=self.panel.imoex.astype(float)), b10_real=self.b10_real)     # карта секторов проекта (karta=None)


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


def sha_istochnikov(b10: dict, panel: Panel, fon: dict, b10_real: dict | None = None) -> str:
    """Д5: sha256 значений, индексов и имён всех рядов источников (в т.ч. реальных 10-минуток). Подмена любого значения меняет хэш."""
    h = hashlib.sha256()
    for t in sorted(b10):
        h.update(b"b10:" + t.encode())
        _h(h, b10[t])
    for t in sorted(b10_real or {}):
        h.update(b"real:" + t.encode())
        _h(h, b10_real[t])
    h.update(b"panel:" + ",".join(panel.tikery).encode())
    for x in (panel.c, panel.imoex, panel.daty, panel.metki, panel.sigma, panel.ok):
        _h(h, x)
    for r in sorted(fon):
        h.update(b"fon:" + r.encode())
        _h(h, fon[r][0])
        _h(h, fon[r][1])
    return h.hexdigest()


def sobrat(raw: dict, imoex: pd.Series, karta: pd.DataFrame, tikery, nachalo, konec, korp=korpsobytiya, mnozh=_mnozhitel, ryady_s=None, ryady_e=None) -> IstochnikiV7:
    """raw {тикер: 10-минутки (timestamp, OHLCV)}, imoex — close 18:50 по дням, karta (ticker, indeks), [nachalo; konec) — окно чтения.
    ryady_e — последняя дата рядов оценщика (после границы чтения — строки NaN).
    ryady_s — первая дата рядов оценщика: панель покрывает все её дни (до первой 10-минутки — строки NaN), иначе S8/S9 дают KeyError «дат вне панели»."""
    tk, nachalo, konec = list(tikery), pd.Timestamp(nachalo), pd.Timestamp(konec)
    cl, op, b10, b10r = {}, {}, {}, {}
    for t in tk:
        b = korp(raw[t][lambda x: (x["timestamp"] >= nachalo) & (x["timestamp"] < konec)], t)     # корпсобытия до агрегации
        dd = d18.aggregate(b, t).set_index("date")
        f = pd.Series(mnozh(t, dd.index.values, dd["close"].to_numpy()), dd.index)
        cl[t] = dd["close"] * f
        bo = b[b["timestamp"].dt.hour * 60 + b["timestamp"].dt.minute == d18.OPEN_MIN]
        op[t] = bo.groupby(bo["timestamp"].dt.normalize())["open"].first()
        b10[t] = bars_agg.prep_10m(korr10(b, f))
        b10r[t] = bars_agg.prep_10m(b)                       # реальные цены (после корпсобытий, без дивидендной поправки) — для cena_T
    c = pd.DataFrame(cl)[tk]
    c = c[c.notna().sum(axis=1) >= min(MIN_TIK, len(tk))]
    if ryady_s is not None and pd.Timestamp(ryady_s) < c.index[0]:       # дни рядов оценщика до первой 10-минутки: строки NaN (сигнал в них — прогрев, ok = False)
        c = c.reindex(pd.date_range(ryady_s, c.index[0] - pd.Timedelta(days=1)).append(c.index))
    if ryady_e is not None and pd.Timestamp(ryady_e) > c.index[-1]:     # дни рядов оценщика после границы чтения: строки NaN (сигнал 0, вход закрыт)
        c = c.reindex(c.index.append(pd.date_range(c.index[-1] + pd.Timedelta(days=1), ryady_e)))      # все календарные дни: торгуются и субботы
    daty, im = pd.DatetimeIndex(c.index), imoex[imoex.index < konec].reindex(c.index)
    o = pd.DataFrame(op)[tk].reindex(c.index)
    metki = ks.sektor_na_d(karta.assign(s=pd.Timestamp("2000-01-01")), daty, tk)
    sigma = np.log(c).diff().rolling(22, min_periods=15).std().to_numpy()
    pn = Panel(daty=daty.values, tikery=tk, c=c, imoex=im, metki=metki, sigma=sigma,
               ok=o.notna().shift(-ks.LAG, fill_value=False).to_numpy())
    fon = {r: otn.fon_ryad(c, im, metki, r) for r in REZHIMY_OTN}
    return IstochnikiV7(b10=b10, panel=pn, fon=fon, sha=sha_istochnikov(b10, pn, fon, b10r), b10_real=b10r)


def zagruzit(period: str, tikery, main: Path | None = None, fajly=None, *, dev: bool = False) -> IstochnikiV7:
    """Источники v7 из файлов (< конец + 1 день). period = polnaya | wf; замок wf — `proverit_chtenie`; иначе отказ.
    dev=True допускает train4 (сухой прогон progon_v7): граница чтения — iv6.konec (dev-зона < 2022). Панель — с START (первая дата рядов оценщика)."""
    from fastbt import istochniki_v6 as iv6
    if period not in PERIOD_FAJL and period != "wf" and not (dev and period == "train4"):
        raise NotImplementedError(f"период {period!r}: источники v7 читаются только для polnaya (wf); train4 — с dev=True")
    if period == POLNAYA:              # до акта владельца (Ж-155) — строго < 2025-01-01 на чтении, а не по границам wf (2026-03-01)
        period_v2.proverit_chtenie(POLNAYA, otmetka=period_v2.OTMETKA)
        end = period_v2.chtenie_konec(POLNAYA)
        f = fajly or iv6.Fajly(end, main, None)      # period=None: indeks() читает по f.end, а не по granitsa("wf")
        p, ryady_e = POLNAYA, period_v2.PERIODY["wf"].konec           # ряды оценщика идут до конца зоны wf; панель после границы чтения — NaN
    else:
        p = PERIOD_FAJL.get(period, period)
        ryady_e = iv6.konec(p) - pd.Timedelta(days=1)                 # последний день рядов оценщика (train4: dev-зона < 2022)
        f = fajly or iv6.Fajly(period_v2.chtenie_konec(p) if period == "train4" else iv6.konec(p), main, p)    # train4: чтение < 2020-01-01
    raw = {t: f.akcii(t) for t in tikery}
    b = f.indeks("imoex")
    if b is None:
        raise FileNotFoundError("IMOEX: нет файла баров")
    im = d18.aggregate(b, "IMOEX").set_index("date")["close"].astype(float)
    karta = pd.read_csv(otn.KARTA_CSV, usecols=["ticker", "indeks"])
    s = sobrat(raw, im, karta, tikery, period_v2.PERIODY[p].progrev, f.end,
               ryady_s=period_v2.PERIODY["wf"].progrev, ryady_e=ryady_e)     # панель — на все дни рядов оценщика
    for b in (*s.b10.values(), *(s.b10_real or {}).values()):
        b.attrs["period"] = p                          # граница ленты S10–S12 — chtenie_konec(p), явно (impuls_v7.granitsa)
    return s
