"""Полный путь точки S8–S12 в EvaluatorV7 (лист s8_s12_ocenka, Ж-153, Ж-156 п. 3): `SemyaMixin`.

Дефект: evaluate_v2 бросал ValueError для S8–S12. Причины: (1) оценщик v5/v6 знает сигнальные оси S1–S6, а у S10–S12 это tau/vhod/T, у S8/S9 —
porog/bufer/cap/rebal; `tau` и `T` в razbor — оси ВЫХОДА и ВХОДА v5, не сигнала; (2) вход в T требует cena_T и Params.vhod_T (exec_e_t), а
run_rows (eval_v5) cena_T не передаёт; (3) у точек S8/S9 нет stop_nach, а razbor его требует.
Решение: сигнальные оси точки снимаются ДО разбора и хранятся в `_sg`; на время расчёта sid оценщика = прокси S3 (категория тренд, общие оси
выхода и риска — куб S3), в точку подставляются S3-заглушки сигнальных осей (семья S3 не вызывается — сигнал даёт `_signals5` по `_sg`).
Вход — рынком: Params.k = None, vhod_tip = podtverzhdenie (без стоп-заявок и уровней); vhod = T — Params.vhod_T + cena_T (open 10-мин бара T,
18:40 — close) из Ist7.b10, день без бара T — NaN (сделки в этот день нет, Д4). Выход — флаги SignalSet (close дня i+τ) через positions.
↺ S8/S9: stop_nach в кубе нет — берётся якорь S3 («sigma:2.0»), откат — убрать подстановку `STOP_S89`. ❓ для дирижёра.
Ключ кэша (Д5): к ключу семьи добавляются сигнальные оси точки (`_sg`)."""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd

from fastbt import eval_v5
from fastbt.canon_v5 import SIGNAL
from fastbt.eval_v5 import run_rows as _run_rows
from fastbt.t_reshenie_v7 import AUKCION, _minuty

IMPULS, KROSS = ("S10", "S11", "S12"), ("S8", "S9")
SIG_OSI = {s: ("tau", "vhod", "T") for s in IMPULS} | {s: ("porog", "bufer", "cap", "rebal") for s in KROSS}
STOP_S89 = "sigma:2.0"
ZAGLUSHKA = {"okno": 20, "podtv": "none"}            # сигнальные оси S3: семья S3 не вызывается, нужны razbor и ключу
SLOT0, STEP10 = 10 * 60, 10


def rasschepit(sid: str, pt: dict) -> tuple[dict, dict]:
    """(точка без сигнальных осей семьи + заглушка S3, сигнальные оси для семьи v7). Не хватает оси или лишняя — ValueError."""
    osi = SIG_OSI[sid]
    if set(osi) - set(pt):
        raise ValueError(f"{sid}: в точке нет сигнальных осей {sorted(set(osi) - set(pt))}")
    sg = {k: pt[k] for k in osi}
    rest = {k: v for k, v in pt.items() if k not in osi}
    if sid in KROSS:
        sg = {"rezhim": sg.pop("rebal"), **sg}
        rest.setdefault("stop_nach", STOP_S89)
    assert set(ZAGLUSHKA) == set(SIGNAL["S3"])
    return {**rest, **ZAGLUSHKA}, sg


def cena_t(b10: pd.DataFrame, dates, T: str) -> np.ndarray:
    """Цена входа в T по датам `dates`: open 10-мин бара, начинающегося в T (18:40 — close); нет бара — NaN (без ffill, Д4)."""
    t = _minuty(T)
    b = b10[b10["slot"] == (t - SLOT0) // STEP10]
    c = pd.Series(b["close" if t == AUKCION else "open"].to_numpy(float), index=pd.DatetimeIndex(b["date"]).normalize())
    return c[~c.index.duplicated(keep="last")].reindex(pd.DatetimeIndex(dates).normalize()).to_numpy(float)


class SemyaMixin:
    _sem = None          # реальный sid S8–S12 во время расчёта точки (self.sid = прокси S3)
    _sg = None           # сигнальные оси точки для семьи v7

    def _semya_v2(self, pt: dict, period: str, vozm):
        real, (q, sg) = self.sid, rasschepit(self.sid, pt)
        self._sem, self._sg, self.sid = real, sg, self.proxy
        try:
            return super().evaluate_v2(q, period, vozm)
        finally:
            self.sid, self._sem, self._sg = real, None, None

    def _rows(self, pt, period, anchor):
        if self._sem is None or self.r5 is None:
            return super()._rows(pt, period, anchor)
        r = self.r5
        r.nov["vhod_tip"] = "podtverzhdenie"
        r.ext.pop("k_e", None)
        r.ext["k"] = None
        T = self._sg.get("T") if self._sem in IMPULS and self._sg.get("vhod") == "T" else None
        r.ext["vhod_T"] = T
        with _cena_t(self, T):
            return super()._rows(pt, period, anchor)


@contextmanager
def _cena_t(evr, T):
    """На время _rows run_rows получает cena_T тикера (V5._rows её не передаёт); вне блока — прежний run_rows."""
    if T is None:
        yield
        return
    def run_rows(t, days, sig, sg, p, **kw):
        return _rows_cena(t, days, sig, sg, p, cena_t(evr.ist7.b10[t], kw["dates"], T), **kw)
    eval_v5.run_rows = run_rows
    try:
        yield
    finally:
        eval_v5.run_rows = _run_rows


def _rows_cena(t, days, sig, sg, p, cena, rang=None, w=None, win=None, fdr=None, **kw):
    """run_rows с cena_T: тот же разбор строк, что у eval_v5.run_rows (rang=None → нужен exec_e.run c cena_T)."""
    from fastbt import etap1_run, exec_e
    n = len(days)
    a, b = (0, n) if win is None else win
    w = np.ones(n) if w is None else np.asarray(w, float)
    xk = {k: v for k, v in kw.items() if k in ("ed", "tejk_M", "urovni", "dates") and v is not None}
    res = exec_e.run(t, days, sig, sg, p, close=kw.get("close"), div=kw.get("div"), nalog=kw.get("nalog"), rang=rang, cena_T=cena, **xk)
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
