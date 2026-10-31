"""Полный путь точки S8–S12 в EvaluatorV7 (лист s8_s12_ocenka, Ж-153, Ж-156 п. 3): `SemyaMixin`.

Дефект: evaluate_v2 бросал ValueError для S8–S12. Причины: (1) оценщик v5/v6 знает сигнальные оси S1–S6, а у S10–S12 это tau/vhod/T, у S8/S9 —
porog/bufer/cap/rebal; `tau` и `T` в razbor — оси ВЫХОДА и ВХОДА v5, не сигнала; (2) вход в T требует cena_T и Params.vhod_T (exec_e_t), а
run_rows (eval_v5) cena_T не передавал; (3) у точек S8/S9 нет stop_nach, а razbor его требует.
Решение: сигнальные оси точки снимаются ДО разбора и хранятся в `_sg`; на время расчёта sid оценщика = прокси S3 (категория тренд, общие оси
выхода и риска — куб S3), в точку подставляются S3-заглушки сигнальных осей (семья S3 не вызывается — сигнал даёт `_signals5` по `_sg`).
Вход — рынком (Params.k = None, vhod_tip = podtverzhdenie, без стоп-заявок и уровней): vhod = d1 и S8/S9 — по open 10:00 дня d+1
(Params.vhod_open, Д2: не раньше d+1; без него k = None входит по close дня сигнала); vhod = T — Params.vhod_T + cena_T (open 10-мин бара T,
18:40 — close) из Ist7.b10, день без бара T — NaN (сделки в этот день нет, Д4). cena_T идёт в run_rows явным параметром через `fdr_kw`
(подмены eval_v5.run_rows нет — класс № 43). Выход — флаги SignalSet (close дня i+τ) через positions.
↺ S8/S9: stop_nach в кубе нет — берётся якорь S3 («sigma:2.0»), откат — убрать подстановку `STOP_S89`. ❓ для дирижёра.
Ключ кэша (Д5): к ключу семьи добавляются сигнальные оси точки (`_sg`)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt.canon_v5 import SIGNAL
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
    _cena_T = None       # 'ЧЧ:ММ' входа в T на время _rows; None — cena_T в run_rows не передаётся

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
        T = self._sg.get("T") if self._sem in IMPULS and self._sg.get("vhod") == "T" else None
        r.ext.update(k=None, vhod_T=T, vhod_open=T is None)        # d1 и сечение: open d+1 (Д2); T: цена T того же дня
        self._cena_T = T
        try:
            return super()._rows(pt, period, anchor)
        finally:
            self._cena_T = None

    def fdr_kw(self) -> dict:
        """kwargs run_rows: к прежним (fdr) добавляется cena_T(тикер, dates) только на время точки со входом в T."""
        kw = super().fdr_kw()
        T = self._cena_T
        return kw if T is None else {**kw, "cena_T": lambda t, dates: cena_t(self.ist7.b10[t], dates, T)}
