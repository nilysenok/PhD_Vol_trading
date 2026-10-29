"""Боевой загрузчик рядов оценщика v6: `zagruzit(period, tikery) -> IstochnikiV6` (лист Ист-v6, Ж-135: только код и малые проверки).

Собирает ряды для осей v6 из существующих загрузчиков, ничего не придумывая:
  dannye(t, td) → kontekst_v6.Dannye на td.dates: σ̂ rv5/rv22/rv66 (`sigma_real`), Z рынка (`t_obshchee.Z_ryad`), режим IMOEX
    (`rezhim_e1.rezhim_arrays`), сектор К9–К12 (`e1_sektor.Ctx`), К13–К16 (`e1_rel_sektor.Ctx`, bench сектор/рынок);
  rel(t, rezhim) → дневной td_rel (`rel_ryad.td_rel_rezhim`, сектор PIT из sostav_sektorov); vn(t, f) → (z, β) (`vneshnie_uslovie`);
  cbr — `sobytiya_e1.load_cbr`; div — таблица, но div_verified=False (❓5): ось дивидендов оценщик отказывает.
Период — через `period_v2.proverit_chtenie` (wf до отметки владельца — отказ); читается строго < конец периода + 1 день;
конец позже `zona.end()` (2022) — NotImplementedError (загрузчики ≥ 2022 не подключены), молча не обрезается.
Д3: значение дня d — из баров < 18:50 d, а Z/σ̂ берутся на d−1 внутри модулей. Д4: дыры остаются NaN (в `Dannye` — вход
запрещён модулем контекста; в td_rel — ValueError оценщика), ffill нет. Чтение файлов — только класс `Fajly` (в тестах подменяется).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import daily_1850 as d18, e1_rel_sektor as rels, e1_sektor as sek, filtr_e1, kontekst_v6, period_v2, rel_ryad
from fastbt import rezhim_e1, sigma_real, sobytiya_e1, t_obshchee, vneshnie_1850 as v18, vneshnie_uslovie as vu, zona
from fastbt.eval_v6 import IstochnikiV6

ROOT = Path(__file__).resolve().parent.parent
SIGMY = (("sig5", "rv5"), ("sig22", "rv22"), ("sig66", "rv66"))


def _al(s: pd.Series, ix: pd.DatetimeIndex) -> np.ndarray:
    """Ряд на даты тикера; нет значения — NaN (без подстановки, Д4)."""
    return s.reindex(ix).to_numpy(float)


class Fajly:
    """Чтение файлов (только < end). Нет файла индекса/фактора — None (решает вызывающий)."""

    def __init__(self, end: pd.Timestamp, main: Path | None = None):
        self.end, self.main = end, Path(main) if main else d18.default_main()

    def akcii(self, t: str) -> pd.DataFrame:
        return d18.read_bars(self.main / "moex_discovery" / "data" / "raw" / "candles_10m" / f"{t}.parquet", self.end)

    def indeks(self, name: str):
        key = name.lower()
        p = self.main / (v18.PATHS[key] if key in v18.PATHS else f"{sek.IDX_DIR}/{name}.parquet")
        return v18.read_bars(p, self.end) if p.exists() else None

    def zakrytie(self, t: str) -> pd.Series:
        """Дневной close Н6 тикера (с поправкой на дивиденды, как у оценщика)."""
        from fastbt import data
        td = data.load_ticker(t)
        s = pd.Series(td.close, index=pd.DatetimeIndex(td.dates))
        return s[s.index < self.end]

    def brent(self) -> tuple:
        """({secid: бары}, таблица контрактов). Нет файла контракта — его нет в словаре (дыра остаётся дырой)."""
        table = v18.contract_table(self.main)
        out = {}
        for s in table["secid"]:
            p = self.main / v18.BR_DIR / f"{s}.parquet"
            p = p if p.exists() else self.main / v18.BR_DIR / f"{s.split('_')[0]}.parquet"
            if p.exists() and len(b := v18.read_bars(p, self.end)):
                out[s] = b
        return out, table

    def ryady(self, tikery):
        """Все входные ряды источников [(имя, DataFrame | Series | None)] для хэша: бары, индексы, факторы, Brent, close, календарь."""
        names = {n for rows in rels.sektor_pit(ROOT / rels.SEKTORY_CSV).values() for _, n in rows if n}
        out = [(f"akcii:{t}", self.akcii(t)) for t in tikery] + [(f"close:{t}", self.zakrytie(t)) for t in tikery]
        out += [(f"indeks:{n}", self.indeks(n)) for n in sorted(names | {"imoex", "rvi", "usdrub"})]
        bars, table = self.brent()
        out += [(f"brent:{k}", b) for k, b in sorted(bars.items())] + [("brent_tablica", table)]
        return out + list(zip(("cbr", "div"), self.kalendar()))

    def kalendar(self) -> tuple:
        return sobytiya_e1.load_cbr(ROOT / sobytiya_e1.CBR_CSV), sobytiya_e1.load_div(ROOT / sobytiya_e1.DIV_CSV)


class _Sbor:
    def __init__(self, f: Fajly, tikery):
        self.f, self.tikery, self._k = f, list(tikery), {}

    def _once(self, key, fn):
        if key not in self._k:
            self._k[key] = fn()
        return self._k[key]

    def sigma(self, t: str) -> dict:
        def go():
            rv = sigma_real.rv_1850(self.f.akcii(t))
            return sigma_real.sigma_all(rv[rv.index < self.f.end])
        return self._once(("sigma", t), go)

    def imoex(self) -> pd.DataFrame:
        def go():
            b = self.f.indeks("imoex")
            if b is None:
                raise FileNotFoundError("IMOEX: нет файла баров")
            m = b["timestamp"].dt.hour * 60 + b["timestamp"].dt.minute
            nb = b[(m >= d18.OPEN_MIN) & (m < d18.CUT_MIN)].groupby(b["timestamp"].dt.normalize()).size()
            d = pd.DataFrame({"close": v18.close_1850(b, "IMOEX"), "sig_mkt": sigma_real.sigma_rv(sigma_real.rv_1850(b), 22),
                              "nbars": nb}).dropna()
            return d[d.index < self.f.end]
        return self._once("imoex", go)

    def pit(self) -> dict:
        return self._once("pit", lambda: rels.sektor_pit(ROOT / rels.SEKTORY_CSV))

    def konteksty(self) -> tuple:
        """(sektor, rel_sektor, rel_rynok): общие ряды всех тикеров universe."""
        def go():
            names = {n for rows in self.pit().values() for _, n in rows if n}
            idx = {n: v18.close_1850(b, n) for n in sorted(names) if (b := self.f.indeks(n)) is not None}
            close = {t: self.f.zakrytie(t) for t in self.tikery}
            sg = {t: self.sigma(t)["rv22"] for t in self.tikery}
            smap = sek.sektor_map(ROOT / sek.SEKTORY_CSV, idx)
            return (sek.Ctx(idx, close, smap), rels.Ctx(idx, close, sg, self.pit()),
                    kontekst_v6.rynok_ctx(self.imoex()["close"], close, sg))
        return self._once("ctx", go)

    def dannye(self, t: str, td) -> kontekst_v6.Dannye:
        ix, im, sg = pd.DatetimeIndex(td.dates), self.imoex(), self.sigma(t)
        rz = rezhim_e1.rezhim_arrays(im["close"].to_numpy(float), im["sig_mkt"].to_numpy(float), im["nbars"].to_numpy())
        rzv = {k: (pd.Series(v, index=im.index).astype(float).reindex(ix).fillna(0.0).to_numpy() > 0.5) for k, v in rz.items()}
        s, r, q = self.konteksty()
        return kontekst_v6.Dannye(t, td.dates, close=td.close, high=td.high, low=td.low, volume=td.volume,
                                  **{a: _al(sg[b], ix) for a, b in SIGMY}, Z=t_obshchee.Z_ryad(_al(im["close"], ix), _al(im["sig_mkt"], ix)),
                                  sektor=s, rel_sektor=r, rel_rynok=q, rezhim=rzv)

    def rel(self, t: str, rezhim: str) -> pd.DataFrame:
        rows = self.pit().get(t, [])
        ind = {n: b for n in {n for _, n in rows if n} if (b := self.f.indeks(n)) is not None}
        return rel_ryad.td_rel_rezhim(rezhim, self.f.akcii(t), t, self.f.indeks("imoex"), ind, rows)[0]

    def faktor(self, f: str) -> tuple:
        """(dx, σ_f): изменение фактора за день и его rv22 (Brent — склейка по доходностям, σ активного контракта)."""
        def go():
            if f == "brent":
                bars, table = self.f.brent()
                br = v18.splice_brent({s: v18.close_1850(b, s) for s, b in bars.items()}, table)
                rv = {s: sigma_real.rv_1850(b) for s, b in bars.items()}
                r = pd.Series([rv[s].get(d, np.nan) for d, s in zip(br["date"], br["secid"])], index=pd.DatetimeIndex(br["date"]), dtype=float)
                return br.set_index("date")["ret"], sigma_real.sigma_rv(r, 22)
            b = self.f.indeks(f)
            if b is None:
                raise FileNotFoundError(f"внешний фактор {f}: нет файла баров")
            return v18.change(f, v18.close_1850(b, f)), sigma_real.sigma_rv(sigma_real.rv_1850(b), 22)
        return self._once(("faktor", f), go)

    def vn(self, t: str, f: str) -> tuple:
        if f not in vu.FAKTORY:
            raise ValueError(f"vn_faktor {f!r} вне {vu.FAKTORY}")
        dx, sg = self.faktor(f)
        c = self.f.zakrytie(t)
        return vu.z_f(dx, sg), vu.beta_f(pd.Series(filtr_e1.ret(c.to_numpy(float)), index=c.index), dx)


def konec(period: str) -> pd.Timestamp:
    """Граница чтения (строго <): `zona.end()` (2022-01-01). Сетка td оценщика (sk._td) идёт до конца dev-ряда (2021), а не до конца
    периода, поэтому ряды читаются на всю dev-зону; метрики периода их не используют. Замок ≥ 2022 — `proverit_chtenie`;
    загрузчики ≥ 2022 (wf после снятия замка) не подключены — NotImplementedError, молчаливой обрезки нет."""
    period_v2.proverit_chtenie(period, otmetka=period_v2.OTMETKA)
    if period_v2.PERIODY[period].konec >= zona.end():
        raise NotImplementedError(f"период {period!r}: файлы баров лежат по тем же путям, но загрузчики (zona.end(), e1_sektor.END, "
                                  f"daily_1850) режут чтение на {zona.end().date()}; расширять границы — отдельное решение, не здесь")
    return zona.end()


def sha_ryadov(f: Fajly, tikery) -> str:
    """Д5: sha256 всех рядов источников (значения, индексы, имена колонок; нет ряда — отметка «net»). Подмена значения меняет хэш."""
    h = hashlib.sha256()
    for name, x in f.ryady(list(tikery)):
        h.update(name.encode())
        if x is None:
            h.update(b"net")
        else:
            h.update(pd.util.hash_pandas_object(x, index=True).to_numpy().tobytes())
            h.update(str(list(x.columns) if hasattr(x, "columns") else [x.name]).encode())
    return h.hexdigest()


def zagruzit(period: str, tikery, main: Path | None = None, fajly: Fajly | None = None) -> IstochnikiV6:
    """Ряды для EvaluatorV6(ist=...) на период `period` (train4 | v2 | wf) и universe `tikery`."""
    f = fajly or Fajly(konec(period), main)
    sbor = _Sbor(f, tikery)
    cbr, div = f.kalendar()
    return IstochnikiV6(dannye=sbor.dannye, rel=sbor.rel, vn=sbor.vn, cbr=cbr, div=div, div_verified=False, rel_progrev=True,
                        sha=sha_ryadov(f, tikery))
