"""Л9: оценщик точки куба v5 (Ж-127/Ж-129 п. 8): точка → семья v5 → фильтры v5 → слой v4 → автомат v2 → ряды.

`EvaluatorV5(EvaluatorV2)`, контракт выхода — как у EvaluatorV2.evaluate_v2 (строка ledger v2.0, excess портфеля float64 [T],
excess тикеров float64 [n_tik, T]); |pos| и сделки по (тикер, день) — `poz`, как у EvaluatorV2 (#600). Драйвер
progon_sobol.ocenshchik (#608) создаёт его как EvaluatorV5(sid, inp, base, komissiya, fdr=<куб>); fdr вне None | fdr.REZHIMY —
ValueError в EvaluatorV2.__init__. Цепочка на тикер:
  get_family(sid, setka="v5")(td, ind, [sp]) → v5_filtr.primenit (маска дня d: вход на close d, исполнение ≥ d+1) →
  sweep_post.build_view/positions (родные RM выключены, RM_OFF) → exec_e_v4sloj.filtr_signala (фаза 3, min_hold_rezhim
  znak_vhoda) → события/режим (net) → exec_e.run(Params: k, T, reentry_den, ed_s/ed_t, a_rang, b0/b0_c, sp, step_s, atr_m, parab,
  chast, pir, exit_signal) — оси точки разбирает fastbt/eval_v5_osi.py.
semya = "v4" — тот же путь с сигналом семьи v4 (сигнальные оси v4): гейт «v5 на якорях = v4» против EvaluatorV2.
"""
from __future__ import annotations

import importlib
from dataclasses import replace

import numpy as np
import pandas as pd

from fastbt import etap1_run, exec_e, rangi_v6, stop_zayavki, tejk_signal
from fastbt.canon_v5 import NET, SIGNAL, TREND, tip
from fastbt.etap1_data import _one_bar, gate_trades
from fastbt.etap1_eval import PERIODS, RM_OFF
from fastbt.etap1_eval_v2 import NOVYE, EvaluatorV2, NeSchitana
from fastbt.eval_v5_osi import RHO_RISK, razbor
from fastbt.exec_e_v4sloj import filtr_signala
from fastbt.fast_filtry_v5 import calc_adx_nb, maska_atr_expand_nb, rang_nb
from fastbt.fdr import OSNOVNOY
from fastbt.indicators import calc_atr
from fastbt.signals.interface import get_family
from fastbt.signals.v5_filtr import primenit
from fastbt.sweep_post import EXEC, MASKS, build_view, positions
from fastbt.unit import Unit

MODUL = {"S1": "v5_s12", "S2": "v5_s12", "S3": "v5_s34", "S4": "v5_s34", "S5": "v5_s56", "S6": "v5_s56"}
M_MAX = {"1": 1.0, "lin": 1.5, "kvadr": 2.0}                      # max rangi_v6.mnozhitel по типу
UROVNI = {"S3": {"okno": "dc_window"}, "S4": {"period": "atr_period", "mnozh": "multiplier"}}   # оси v5 → stop_zayavki.urovni


def maska(td, tip_: str, filtr: dict, sig) -> np.ndarray:
    """bool[n]: вход на close d разрешён фильтрами v5 (данные ≤ d)."""
    df = pd.DataFrame({"high": td.high, "low": td.low, "close": td.close, "volume": td.volume}, index=pd.to_datetime(td.dates))
    f = dict(filtr)
    at, al = f.pop("adx_th", 0), f.pop("atr_lo", 0)
    m = primenit(df, tip_, f, sig=np.asarray(sig))                  # прочие фильтры; ADX и ATR-процентиль — numba (#629)
    hlc = [np.asarray(x, float) for x in (td.high, td.low, td.close)]
    if at > 0:
        adx = calc_adx_nb(*hlc, 14)
        with np.errstate(invalid="ignore"):
            m &= ~np.isnan(adx) & ((adx > at) if tip_ == "trend" else (adx < at))
    if al > 0:
        m &= maska_atr_expand_nb(calc_atr(*hlc, 14), al)
    return m


def rodnoy_ok(sid: str, sp: dict) -> bool:
    """tejk rodnoy: уровень tejk_signal.rodnoy_uroven (S1 SMA ∓ z_exit·STD, S2 SMA, S6 VWAP, S5 пивоты v4) — тот же, что у
    сигнала v5, только при этих осях; иначе уровень чужой — строка ne_schitana, а не молча другой тейк."""
    if sid == "S1":
        return sp.get("ma_tip") == "sma" and sp.get("dev_tip") == "std" and sp.get("z_exit") != NET
    return {"S2": sp.get("ma_tip") == "sma", "S6": True}.get(sid, False)


def rang_starta(sig, rg):
    """Ранг дня старта ТЕКУЩЕГО сигнала на каждом дне, пока сигнал активен (тот же знак подряд); конец сигнала и смена знака
    обнуляют перенос. Ранг дня старта считался по данным ≤ дня старта; позже не берётся (Р-4, умолчание дирижёра 29.10)."""
    sig, rg = np.asarray(sig, float), np.asarray(rg, float)
    out, cur, prev = np.full(len(sig), np.nan), np.nan, 0.0
    for d in range(len(sig)):
        if sig[d] == 0:
            cur = np.nan
        elif sig[d] != prev:
            cur = rg[d]
        out[d], prev = cur, sig[d]
    return out


def run_rows(t, days, sig, sg, p, rang=None, w=None, win=None, fdr=None, **kw):
    """etap1_run.run_e1; при rang (тейк/безубыток от силы) — копия его сборки строк с rang в exec_e.run (run_e1 rang не
    принимает, etap1_run.py вне владения Л9; равенство копии — tests/test_eval_v5.py::test_run_rows_rang_kopiya_run_e1)."""
    if rang is None:
        return etap1_run.run_e1(t, days, sig, sg, p, w=w, win=win, fdr=fdr, **kw)
    n = len(days)
    a, b = (0, n) if win is None else win
    w = np.ones(n) if w is None else np.asarray(w, float)
    xk = {k: v for k, v in kw.items() if k in ("ed", "tejk_M", "urovni", "dates") and v is not None}
    res = exec_e.run(t, days, sig, sg, p, close=kw.get("close"), div=kw.get("div"), nalog=kw.get("nalog"), rang=rang, **xk)
    ix = [i for i, d in enumerate(days) if not d.pusto]
    wr, sides, cnt, sh = np.zeros(n), np.zeros(b - a), np.zeros(b - a), np.zeros(n)
    for tr, r0, last_g, er in etap1_run._spans(res, ix, p, n):
        wt = float(w[tr.d_sig])
        wr[r0:last_g + 1], sh[r0:last_g + 1] = wt, float(tr.side < 0)
        if r0 < b and last_g >= a:
            for r in (max(r0, a), min(er, b - 1)):
                sides[r - a] += wt
                cnt[r - a] += 1.0
    g, expo = res.gross * wr, (1.0 - res.free) * wr
    noch = None
    if fdr is not None:
        from fastbt import fdr as fdr_mod
        noch = fdr_mod.noch_iz_sdelok(res.trades, {tr.d_sig: float(w[tr.d_sig]) for tr in res.trades}, n, fdr)[a:b]
    return etap1_run.E1Rows(g[a:b], sides, expo[a:b], cnt, res.trades, res.n_orders, res.n_fills, noch,
                            res.x_orders, res.x_fills, ((1.0 - res.free) * wr * sh)[a:b], res.n_bez_tejka)


class EvaluatorV5(EvaluatorV2):
    STROGAYA = True

    def __init__(self, sid, inp, base_p, commission: float = 0.0004, fdr=OSNOVNOY, semya: str = "v5", rol: str | None = None):
        super().__init__(sid, inp, base_p, commission, fdr)
        if semya not in ("v5", "v4"):
            raise ValueError(f"semya: v5 | v4, получено {semya!r}")
        self.semya, self.r5, self._fkey, self._days5, self.rol = semya, None, (), {}, rol
        importlib.import_module(f"fastbt.signals.{MODUL[sid]}")          # регистрация семей v5
        names = set(self.sk.sig_names) | set(SIGNAL[sid])
        self._names5 = tuple(sorted(names))
        self._base5 = {k: v for k, v in self.base.items() if k not in names and k.rpartition(".")[2] not in MASKS | set(EXEC)}

    def _signals5(self, sp: dict) -> dict:
        fam, out = get_family(self.sid, setka="v5"), {}
        for t, (td, ind) in self.sk._td.items():
            ss = fam(td, ind, [dict(sp)])
            if self.r5.filtr:
                ss.sig[0] = np.where(maska(td, tip(self.sid), self.r5.filtr, ss.sig[0]), ss.sig[0], 0).astype(ss.sig.dtype)
            out[t] = Unit(t, self.sid, td, ind, ss)
        return out

    def _view(self, t, sp, q):
        sk = (tuple(sorted(sp.items())), self._fkey)
        if sk not in self._sigs:
            self._sigs[sk] = self._signals5(sp)
        td, ind = self.sk._td[t]
        return build_view(self.sid, self._sigs[sk][t], td, ind, {**self._base5, **sp, "q_sily": q, **RM_OFF[self.sid]}, self._names5)

    def _positions(self, t, sp, q, anchor, gi1):
        if self.semya == "v4":
            return super()._positions(t, sp, q, anchor, gi1)
        k = (t, tuple(sorted(sp.items())), self._fkey, q, gi1)
        if k not in self._pos:
            self._pos[k] = positions(*self._view(t, sp, q), gi1)
        return self._pos[k]

    def _sig5(self, t, s0, b):
        r = self.r5
        pos = self._positions(t, r.sp, 0.0, False, s0 + b)[s0:s0 + b + 1].astype(float)
        st = None
        if r.sloj["min_sily"] > 0:
            st = np.abs(np.asarray(self._view(t, r.sp, 0.0)[0].strength[0], float))[s0:s0 + b + 1]
        sig = np.asarray(filtr_signala(pos, st, **r.sloj), float).copy()
        sig[-1] = 0.0
        return sig

    def _urovni(self, t, s0, b, al):
        sp = {UROVNI[self.sid].get(k, k): v for k, v in self.r5.sp.items()} if self.semya == "v5" else self.r5.sp
        td, ind = self.sk._td[t]
        u = stop_zayavki.urovni(self.sid, sp, td, ind, "obe")
        if self.r5.filtr:                        # вход лонг/шорт — маска фильтров по своему направлению
            for k, z in (("up", 1), ("dn", -1)):
                u[k] = np.where(maska(td, "trend", self.r5.filtr, np.full(td.n, z)), u[k], np.nan)
        return stop_zayavki.okno(u, s0, b, al)

    def _start(self, period):
        """{тикер: (s0, b, a, dates, sig, sg, allow)} до размера и ранга."""
        r, y0, out = self.r5, PERIODS[period][0], {}
        for t in self.sk.tickers:
            m = self.inp.meta[t]
            s0, b = m["s0"], self._b(t, period)
            dates = m["dates"][:b + 1]
            sig = self._sig5(t, s0, b)
            al = self._allow(r.pt4, t, dates, sig, "net")
            sg = self.inp.sigma[r.pt4["sigma"]][t][:b + 1]
            out[t] = (s0, b, int(np.searchsorted(dates, np.datetime64(f"{y0}-01-01"))), dates, gate_trades(sig, al), sg, al)
        return out

    def _ranki(self, info) -> dict:
        """{тикер: (ранг[d] на день старта сделки, allow[d])} — как rangi_v6.ves_tikerov, по силе семьи v5."""
        tk = self.sk.tickers
        cal = np.unique(np.concatenate([info[t][3] for t in tk]))
        M, ix = np.full((len(tk), len(cal)), np.nan), {}
        for k, t in enumerate(tk):
            s0, b, _, dates, sig, _, _ = info[t]
            start = (sig != 0) & (sig != np.r_[0.0, sig[:-1]])
            ix[t] = np.searchsorted(cal, dates)
            sila = np.abs(np.asarray(self._view(t, self.r5.sp, 0.0)[0].strength[0], float))[s0:s0 + b + 1]
            M[k, ix[t]] = np.where(start, sila, np.nan)
        R = rang_nb(M, self.r5.rang_tip)
        return {t: (R[k, ix[t]], ~(np.isfinite(M[k, ix[t]]) & ~np.isfinite(R[k, ix[t]]))) for k, t in enumerate(tk)}

    def _ves(self, sg, rg):
        """Вес сделки ≤ 1/(1 + 2·pir): занятость с добавками пирамиды ≤ 1 (ФДР t0d требует expo, noch ∈ [0; 1]).
        ravnyj при m ≠ 1 — m/max m (lin 1,5, kvadr 2,0); risk_sigma — rangi_v6.ves (min(1, ·) внутри)."""
        r = self.r5
        m = np.ones(len(sg)) if rg is None else rangi_v6.mnozhitel(rg, r.m_sily)
        if r.razmer == "risk_sigma":
            w = rangi_v6.ves(sg, RHO_RISK, float(r.pt4["k_s"]), m)
        else:
            w = np.nan_to_num(m / M_MAX[r.m_sily], nan=0.0)
        return w / (1.0 + 2.0 * r.ext.get("pir", 0.0))

    def _tm(self, t, s0, b):
        """tejk_M для exec_e.run: rodnoy — [2, b+1]; нет tejk_signal — None. Хук EvaluatorV6 (pol_puti, PivUrovni)."""
        r = self.r5
        return None if r.nov["tejk_signal"] == NET else tejk_signal.m_tikera(self, r.sp, t, s0, b, True)

    def _rows(self, pt, period, anchor):
        if self.r5 is None:
            return super()._rows(pt, period, anchor)
        r, inp, out = self.r5, self.inp, []
        if r.nov["vhod_tip"] == "stop_zayavka" and "a_rang" in r.ext:         # b0_c: RunStop получает rang с V6-11a (#666)
            raise NeSchitana("тренд: тейк от силы (a_rang) на RunStop не считается")
        if r.nov["tejk_signal"] != NET and not rodnoy_ok(self.sid, r.sp):
            raise NeSchitana(f"{self.sid}: родной уровень tejk_signal.rodnoy_uroven — формула v4, с осями {r.sp} не совпадает")
        info = self._start(period)
        rk =self._ranki(info) if r.rang else None
        for ti, t in enumerate(self.sk.tickers):
            s0, b, a, dates, sig, sg, al = info[t]
            rg = None
            if rk is not None:
                rg, sig = rk[t][0], gate_trades(sig, rk[t][1])      # кандидат без ранга (< 20 наблюдений) не входит
            days = inp.bars[t][:b] + [_one_bar(float(inp.close_n6[t][b]))]
            p = replace(self._params(r.pt4, t, 5.0, False, r.nov, {}), **r.ext)
            tm = self._tm(t, s0, b)
            uv = {} if r.nov["vhod_tip"] == "podtverzhdenie" else {"urovni": self._urovni(t, s0, b, al)}
            kw = dict(close=inp.close_n6[t][:b + 1], win=(a, b), div=inp.div[t][:b + 1], dates=dates, ed=self._ed(t, "sigma", sg, dates, s0, b),
                      tejk_M=tm, **self.fdr_kw(), **uv)
            rang = rg if (p.a_rang is not None or p.b0_c is not None) else None
            # тренд: d_sig сделки — любой вечер постановки пары, не день старта → вес по рангу дня старта текущего сигнала
            rw = rang_starta(info[t][4], rg) if rg is not None and r.nov["vhod_tip"] == "stop_zayavka" else rg
            self._days5[ti] = days
            out.append((ti, run_rows(t, days, sig, sg, p, w=self._ves(sg, rw), rang=rang, **kw), a, b))
        return out

    def evaluate_v2(self, pt: dict, period: str, vozm: dict | None = None):
        """Точка v5 (после canon_v5) → (строка ledger v2.0, excess портфеля, excess тикеров) — контракт EvaluatorV2."""
        if vozm:
            raise ValueError("vozm у оценщика v5 не поддержан")
        self.r5 = r = razbor(self.sid, pt, SIGNAL[self.sid] if self.semya == "v5" else self.sk.sig_names, self.rol)
        if self.semya == "v4" and (r.rang or r.filtr or self.sid in TREND and r.ext.get("k_e")):
            raise ValueError("semya v4 — только для сверки якоря: без ранга, фильтров v5 и k_e")
        self._fkey = tuple(sorted(r.filtr.items()))
        nov = {k: v for k, v in r.nov.items() if v != NOVYE[k]}
        return super().evaluate_v2({**r.pt4, **r.sp, **nov}, period)


def vne_bara(evr: EvaluatorV5, rel: float = 1e-9) -> dict:
    """Инвариант исполнения (#604): цена входа/выхода сделки внутри [low; high] своего бара. {"vsego", "vne", "primery"}."""
    n, vne, pr = 0, 0, []
    for ti, r in enumerate(evr._last[4]):
        days = evr._days5[ti]
        for tr in r.trades:
            for kind, d, j, px in (("vhod", tr.entry_day, tr.entry_bar, tr.entry_px), ("vyhod", tr.exit_day, tr.exit_bar, tr.exit_px)):
                if d is None or j is None or px is None or days[d].pusto or not 0 <= j < len(days[d].c):
                    continue
                n += 1
                lo, hi = float(days[d].l[j]), float(days[d].h[j])
                if px < lo * (1 - rel) or px > hi * (1 + rel):
                    vne += 1
                    pr.append((evr.sk.tickers[ti], kind, tr.reason, int(d), int(j), float(px), lo, hi))
    return {"vsego": n, "vne": vne, "primery": pr[:5]}
