"""Оценщик этапа 1, строка ledger v2.0 (Л-5, контракт docs/otchety/2026-10-28_progon_etap1.md §1).

`EvaluatorV2(Evaluator)`: оси NOVYE, возмущения `vozm`, ряды `excess` портфеля (float64 [T]) и тикеров (float64 [17, T]).
Все новые оси = умолчания и vozm пуст → путь `Evaluator._rows` побайтно. `etap1_eval.py` не правится (Ж-083).
Зависимости от соседей — ленивые и по имени поля: `Params.zhizn/gap/t_vyhod/bar_plus/stop_tik` и `exec_e.run(..., ed=)` (Л-2),
`filtr_e1_rynok.razresheno(имя, inp, тикер, sig) -> bool[d] | None` (Л-4, ❓ сигнатура).
"""
from __future__ import annotations

import importlib
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from fastbt import metriki_e as me
from fastbt.etap1_data import _one_bar, align_b, gate_trades
from fastbt.etap1_eval import NET, PERIODS, REZHIM, Evaluator, params_of, years_of
from fastbt.etap1_run import E1Rows, run_e1
from fastbt.sweep_post import build_view, positions

NOVYE = dict(ed_lim="sigma", zhizn="den", lestnica="net", gap="vhod", t_vyhod="close", k_x="net", storona="obe", filtr="net")
POROG = {"S1": "z_entry", "S2": "bb_std", "S3": "dc_window", "S4": "multiplier", "S6": "dev_mult"}   # §1.1 storona
ZAEM = (5, 10)                                                                 # % годовых, колонки by_quarter_zaem
_ID = {"zhizn": "den", "gap": "vhod", "t_vyhod": "close"}
SQ = np.sqrt(252)


class NeSchitana(Exception):
    """Фильтр не определён на данных (razresheno → None): строка ne_schitana, а не молча net."""


def kvartaly(daty, x, years) -> dict:
    """{"2016Q1": SR годовой шкалы (ddof=1), …}; квартал без дней — nan."""
    i = pd.DatetimeIndex(daty)
    qq = i.year * 10 + i.quarter
    return {f"{y}Q{q}": (me.sr_dnevnoy(x[qq == y * 10 + q]) * SQ if (qq == y * 10 + q).any() else float("nan"))
            for y in years for q in range(1, 5)}


def tselevaya(bq: dict) -> tuple:
    """(медиана, доля кварталов с SR > 0); есть nan — медиана nan, nan не положителен."""
    v = np.array(list(bq.values()), float)
    return (float("nan") if np.isnan(v).any() else float(np.median(v))), float((v > 0).sum() / len(v))


def sum_rows(rs: list, half: bool = False) -> E1Rows:
    """Строки g, sides, expo, noch, short — суммой (run_e1 линеен по весу); счёт сделок — среднее половин."""
    f = lambda k: sum(getattr(r, k) for r in rs)                             # noqa: E731
    noch = None if rs[0].noch is None else f("noch")
    return E1Rows(f("g"), f("sides"), f("expo"), f("cnt") / len(rs), [t for r in rs for t in r.trades],
                  f("n_orders"), f("n_fills"), noch, f("x_orders"), f("x_fills"), f("short"))


def _carry(eng, inv) -> np.ndarray:
    """rf · cal по строкам: ставка на неиспользованный капитал за календарные дни интервала (вычитается как u·rf·cal)."""
    return eng.rf[inv] * eng.cal[inv]


def mediana_goda(sg: np.ndarray, dates) -> np.ndarray:
    """σ лимита = медиана σ̂ календарного года дня (контроль, Л-9); nan не считаются."""
    yr = pd.DatetimeIndex(dates).year.to_numpy()
    out = np.full(len(sg), np.nan)
    for y in np.unique(yr):
        out[yr == y] = np.nanmedian(sg[yr == y])
    return out


class EvaluatorV2(Evaluator):
    _ctx = dict(nov=NOVYE, vozm={}, default=True)
    _kray = False

    # ---- строки одного тикера ----
    def _rows(self, pt, period, anchor):
        c = self._ctx
        return super()._rows(pt, period, anchor) if c["default"] else self._rows2(pt, period, anchor, c)

    def _shift(self, sp: dict) -> dict:
        """Пороговая ось на 1 шаг строже (больше); на краю сетки — прежнее значение, флаг kray (§1.1)."""
        from fastbt.etap1_poisk import load_setka
        ax = POROG.get(self.sid)
        if ax is None:
            raise ValueError(f"{self.sid}: у стратегии нет пороговой оси (short_shag1 → obe в canon_v2)")
        up = [v for v in sorted(load_setka(self.sid)["axes"][ax]) if v > sp[ax]]
        self._kray = self._kray or not up
        return {**sp, ax: up[0] if up else sp[ax]}

    def _sig(self, t, sp, q, anchor, s0, b, st):
        pos = self._positions(t, sp, q, anchor, s0 + b)[s0:s0 + b + 1].astype(float)
        if st == "obe":
            return pos
        if st == "long":
            return np.maximum(pos, 0.0)
        pos2 = self._positions(t, self._shift(sp), q, anchor, s0 + b)[s0:s0 + b + 1].astype(float)
        return np.where(pos > 0, pos, np.minimum(pos2, 0.0))               # short_shag1: лонг из точки, шорт — из сдвинутой

    def _allow(self, pt, t, dates, sig, f):
        inp = self.inp
        allow = ~inp.sobytiya["cbr" if f == "cbr" else pt["sobytiya"]][t]
        if f not in ("net", "cbr"):
            a = importlib.import_module("fastbt.filtr_e1_rynok").razresheno(f, inp, t, sig)
            if a is None:
                raise NeSchitana(f"filtr {f}: условие не определено на данных")
            allow = allow & np.asarray(a, bool)
        if pt["rezhim"] != NET:
            allow = allow & align_b(inp.rezhim[REZHIM.get(pt["rezhim"], pt["rezhim"])], dates)
        return allow

    def _ed(self, t, e, sg, dates, s0, b):
        if e == "sigma":
            return None
        td, ind = self.sk._td[t]
        c = td.close[s0:s0 + b + 1]
        return {"atr14": lambda: ind["atr14"][s0:s0 + b + 1] / c, "diap": lambda: (td.high[s0:s0 + b + 1] - td.low[s0:s0 + b + 1]) / c,
                "pct": lambda: np.full(b + 1, 0.01), "mediana": lambda: mediana_goda(sg, dates)}[e]()

    def _params(self, pt, t, h, anchor, nov, vz):
        p = params_of(pt, self.inp.tick[t], h, anchor, self.comm)
        if nov["k_x"] != NET and p.b is None and not anchor:
            p = replace(p, a=float(nov["k_x"]))                             # тейк без стопа; при k_s ≠ net k_x не действует
        ext = {k: v for k, v in nov.items() if k in _ID and v != _ID[k]}
        ext.update({k: vz[k] for k in ("bar_plus", "stop_tik") if k in vz})
        return replace(p, **ext) if ext else p

    def _rows2(self, pt, period, anchor, c):
        nov, vz = c["nov"], c["vozm"]
        sp, q = {k: pt[k] for k in self.sk.sig_names}, float(pt.get("q_sily", 0.0))
        h = 5.0 if (anchor or pt["tau"] != NET or pt["R"] == NET) else self._h(sp, q)
        out = []
        for ti, t in enumerate(self.sk.tickers):
            m, inp = self.inp.meta[t], self.inp
            s0, b = m["s0"], self._b(t, period)
            dates = m["dates"][:b + 1]
            a = int(np.searchsorted(dates, np.datetime64(f"{PERIODS[period][0]}-01-01")))
            sig = self._sig(t, sp, q, anchor, s0, b, nov["storona"])
            sig[-1] = 0.0
            if vz.get("sig_fn"):
                sig = np.asarray(vz["sig_fn"](t, sig, dates), float)
            days = inp.bars[t][:b] + [_one_bar(float(inp.close_n6[t][b]))]
            sg = np.ones(len(sig)) if anchor else inp.sigma[pt["sigma"]][t][:b + 1] * float(vz.get("sigma_mult", 1.0))
            if not anchor:
                sig = gate_trades(sig, self._allow(pt, t, dates, sig, nov["filtr"]))
            w = np.ones(len(sig))
            if not anchor and pt["razmer"] != "ravnyy":
                from fastbt import rezhim_e1
                w = rezhim_e1.razmer_ves(sg, "obratnyj", float(pt["razmer"]) / 100.0)
            ed = self._ed(t, "mediana" if vz.get("sigma_lim_mediana") else nov["ed_lim"], sg, dates, s0, b)
            p = self._params(pt, t, h, anchor, nov, vz)
            kw = dict(close=inp.close_n6[t][:b + 1], win=(a, b), div=inp.div[t][:b + 1], ed=ed, **self.fdr_kw())
            out.append((ti, self._ladder(t, days, sig, sg, p, w, nov["lestnica"], kw), a, b))
        return out

    @staticmethod
    def _ladder(t, days, sig, sg, p, w, lest, kw):
        if lest == NET:
            return run_e1(t, days, sig, sg, p, w=w, **kw)
        if p.k is None:
            raise ValueError("lestnica требует лимитного входа (vhod ≠ rynok)")
        return sum_rows([run_e1(t, days, sig, sg, pp, w=w / 2, **kw) for pp in (p, replace(p, k=p.k + float(lest)))])

    # ---- ряды и строка ----
    def excess(self, pt, period):
        r = super().excess(pt, period)
        self._last = r
        return r

    def _series(self, period, comm):
        df, inv, eng, _, rs = self._last
        yrs = years_of(period)
        keep = np.isin(df.index.year, yrs)
        col = np.cumsum(keep) - 1
        ex_all = df["excess"].to_numpy()
        cat = lambda f: np.concatenate([f(r) for r in rs])                       # noqa: E731
        u = cat(lambda r: r.noch) if self.fdr is not None else cat(lambda r: r.expo)
        rc = _carry(eng, inv)
        ex_row = cat(lambda r: r.g) - cat(lambda r: r.sides) * comm - u * rc
        tik = np.full((len(rs), int(keep.sum())), np.nan)
        lo = 0
        for ti, r in enumerate(rs):
            sl = slice(lo, lo + len(r.g))
            lo = sl.stop
            ok = keep[inv[sl]]
            tik[ti, col[inv[sl]][ok]] = ex_row[sl][ok]
        sh = np.bincount(inv, cat(lambda r: r.short), len(eng.dates)) / eng.cnt     # доля шорт-экспозиции портфеля по датам
        return ex_all[keep], tik, df.index[keep], {s: (ex_all - s / 100.0 * sh * eng.cal / 365.0)[keep] for s in ZAEM}, \
            cat(lambda r: r.cnt), pd.DatetimeIndex(eng.dates[inv]), keep

    def evaluate_v2(self, pt: dict, period: str, vozm: dict | None = None):
        t0 = time.perf_counter()
        nov, vz = {k: pt.get(k, v) for k, v in NOVYE.items()}, dict(vozm or {})
        self._ctx, self._kray, comm0 = dict(nov=nov, vozm=vz, default=nov == NOVYE and not vz), False, self.comm
        self.comm = float(vz.get("komissiya", comm0))
        try:
            v1 = self.evaluate({k: v for k, v in pt.items() if k not in NOVYE}, period)
        except NeSchitana as e:
            v1 = {"ne_schitana": str(e)}
        finally:
            self.comm, comm = comm0, self.comm
        yrs, y0, y1 = years_of(period), *PERIODS[period]
        head = {"v": "2.0", "format": "etap1-ledger", "period": period, "okno_s": f"{y0}-01-01", "okno_po": f"{y1}-12-31",
                "progrev_s": f"{y0 - 1}-01-01", "zaderzhka": "d+1", "chestnaya": v1.get("chestnaya", "net"), "komissiya": comm,
                "fdr": self.fdr, "vozm": vz or None}
        bq = {f"{y}Q{q}": float("nan") for y in yrs for q in range(1, 5)}
        if "ne_schitana" in v1:
            row = {**head, "status": "ne_schitana", "prichina": v1["ne_schitana"], "by_quarter": bq, "mediana_kv": float("nan"),
                   "dolya_pol_kv": float("nan"), "sharpe": float("nan"), "msr": float("nan"), "flagi": {"vyrozhdena": True}}
            return row, None, None
        x, tik, dts, zaem, cnt, rdates, keep = self._series(period, comm)
        bq = kvartaly(dts, x, yrs)
        med, dolya = tselevaya(bq)
        qr = (rdates.year * 10 + rdates.quarter).to_numpy()
        sk = lambda d: {str(k): v for k, v in d.items()}                         # noqa: E731
        row = {**head, "status": "ok", "prichina": None, "kod": v1["kod"], "vhody": v1["vhody"],
               "by_year": sk(v1["by_year"]), "msr_by_year": sk(v1["msr_by_year"]), "by_quarter": bq, "mediana_kv": med,
               "dolya_pol_kv": dolya, "by_quarter_zaem": {str(s): kvartaly(dts, z, yrs) for s, z in zaem.items()},
               "sdelok_y": sk(v1["sdelok_y"]), "storony_y": sk(v1["storony_y"]),
               "sdelok_kv": {k: float(cnt[(qr == int(k[:4]) * 10 + int(k[5]))].sum()) / 2 for k in bq},
               "flagi": {"vyrozhdena": v1["vyrozhdena"], "kray": [], "short_shag1_kray": self._kray}, "t_sek": time.perf_counter() - t0,
               **{k: v1[k] for k in ("sharpe", "msr", "sortino", "kalmar", "mdd", "T", "skew", "kurt", "prichiny",
                                     "dolya_ispolneniya_limita")}}
        return row, x, tik
