"""V6-12c: оси исполнения куба v6 для EvaluatorV6: lestnica, gap, t_vyhod, vyhod_lim, f_neisp (inventarizaciya §1.2: И4, И5, И6, И7, И8).

gap = propusk → r.nov["gap"] (Params.gap_propusk), t_vyhod → r.nov["t_vyhod"] (close | 1500 | 1700 | open2), vyhod_lim = da →
Params.vyhod_lim (выход по signal/vremya лимитом на d+1; нужен лимитный вход), f_neisp = zakrytie_d1 → Params.f.
lestnica: две ноги по w/2 — лимит на уровне k_e и на k_e + lestnica (как EvaluatorV2._ladder), ряды складывает etap1_run.sum_rows.
Несовместимое (движок отказал бы ValueError) — NeSchitana: vyhod_lim с t_vyhod ≠ close; gap/vyhod_lim/lestnica на стоп-заявке (тренд).
"""
from __future__ import annotations

from fastbt.etap1_eval import NET
from fastbt.etap1_eval_v2 import NeSchitana
from fastbt.etap1_run import sum_rows

ISPOLN = ("lestnica", "gap", "t_vyhod", "vyhod_lim", "f_neisp")


def primenit(r, v6: dict) -> None:
    """Оси исполнения → разбор точки v5 `r` (eval_v5_osi.Razbor): nov (gap, t_vyhod), ext (vyhod_lim, f). lestnica — в IspolnMixin._legs."""
    for k in ("gap", "t_vyhod"):
        if k in v6:
            r.nov[k] = v6[k]
    if v6.get("vyhod_lim") == "da":
        r.ext["vyhod_lim"] = True
    if v6.get("f_neisp") == "zakrytie_d1":
        r.ext["f"] = "zakrytie_d1"


def proverit(r, v6: dict) -> None:
    if v6.get("vyhod_lim") == "da" and r.nov["t_vyhod"] != "close":
        raise NeSchitana("vyhod_lim и t_vyhod ≠ close несовместимы (exec_e: t_vyhod требует выхода без лимитной заявки)")
    if r.nov["vhod_tip"] == "stop_zayavka" and any(k in v6 for k in ("gap", "vyhod_lim", "lestnica")):
        raise NeSchitana("стоп-заявка (тренд): gap, vyhod_lim и лесенка недопустимы (exec_e, Ж-091)")


class IspolnMixin:
    """Подмешивается перед EvaluatorV5: вес ноги лесенки /2 и сборка ног (super() — EvaluatorV5._rows)."""
    _lest = False

    def _ves(self, sg, rg):
        w = super()._ves(sg, rg)
        return w / 2 if self._lest else w

    def _legs(self, pt, period, anchor):
        lest = self.v6.get("lestnica", NET)
        if lest == NET:
            return super()._rows(pt, period, anchor)
        r, k0 = self.r5, self.r5.pt4["vhod"]
        legs = []
        try:
            self._lest = True
            for k in (k0, k0 + float(lest)):
                r.pt4["vhod"] = k
                legs.append(super()._rows(pt, period, anchor))
        finally:
            r.pt4["vhod"], self._lest = k0, False
        return [(a[0], sum_rows([a[1], b[1]]), a[2], a[3]) for a, b in zip(*legs)]
