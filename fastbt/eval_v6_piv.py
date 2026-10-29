"""V6-12d: пивот S5 / VWAP S6 (stop_nach piv:next|half, vwap:m; tejk piv:R|half) и тейк pol_puti в EvaluatorV6 (inventarizaciya §1.2: И11, И12, И13).

Значения осей v5 stop_nach / tejk с префиксом piv:, vwap: и значение pol_puti razbor v5 не знает: `razobrat` заменяет их заглушкой (σ-стоп, net) и
возвращает поля Params (piv_sl, vwap_m, piv_tp, tejk=pol_puti). Тогда `proverit` отказывает (NeSchitana) там, где пивотному стопу нужен b или
второй тейк, а `tm` отдаёт exec_e.run канал tejk_M: PivUrovni (пивот / VWAP) либо M по дням (pol_puti).
Данные: уровни — из того же семейства v5, что сигнал (SignalSet.extra), по РЕАЛЬНОМУ td (цены стопа реальные; td_rel — только сигнал).
Пивот бара d — уровни прошлого периода (≤ d−1, v5_s56.pivots), VWAP/dev строки d_sig — на закрытии d_sig (exec_e_piv: строка d_sig, не d_sig+1).
"""
from __future__ import annotations

import numpy as np

from fastbt.canon_v5 import NET, edinica
from fastbt.etap1_eval_v2 import NeSchitana
from fastbt.exec_e_piv import PivUrovni
from fastbt.signals.interface import get_family

ZAGLUSHKA = "sigma:2.0"        # стоп-заглушка для razbor; k_s затем = net
PIV_TP = {"piv:R": "R", "piv:half": "half"}


def razobrat(pt5: dict) -> tuple:
    """(pt5 без пивотных значений, ext поля Params). Нет пивота и pol_puti — pt5 как есть, ext пуст."""
    sn, tj, ext, pt = pt5.get("stop_nach"), pt5.get("tejk", NET), {}, dict(pt5)      # razbor сам откажет, если stop_nach нет
    e = edinica(sn)
    if e == "piv":
        ext["piv_sl"], pt["stop_nach"] = sn.split(":")[1], ZAGLUSHKA
    elif e == "vwap":
        ext["piv_sl"], ext["vwap_m"], pt["stop_nach"] = "vwap", float(sn.split(":")[1]), ZAGLUSHKA
    if tj in PIV_TP:
        ext["piv_tp"], pt["tejk"] = PIV_TP[tj], NET
    elif tj == "pol_puti":
        ext["tejk"], pt["tejk"] = "pol_puti", NET
    return pt, ext


def proverit(r, ext: dict, sid: str) -> None:
    """Несовместимое (exec_e / exec_e_piv отказали бы ValueError) — NeSchitana; piv_buf — только у стопа next | half."""
    if "piv_sl" in ext:
        if "tejk" in ext:
            raise NeSchitana("пивотный стоп и pol_puti: канал tejk_M один (PivUrovni либо M)")
        bad = [k for k in ("atr_m", "step_s", "parab", "b0", "b0_c", "sp", "chast", "pir", "chand") if k in r.ext]
        if r.pt4["k_tr"] != NET or r.pt4["R"] != NET or r.razmer == "risk_sigma" or bad:
            raise NeSchitana(f"{sid}: piv_sl — единственный стоп: без трейла, безубытка, пирамиды, частичного выхода, rr и risk_sigma {bad}")
    if ext.get("tejk") == "pol_puti" and sid in ("S1", "S2") and r.sp.get("ma_tip") != "sma":
        raise NeSchitana(f"{sid}: pol_puti — M = SMA сигнала; ma_tip {r.sp.get('ma_tip')!r} не sma")


def primenit(r, ext: dict, v6: dict) -> None:
    """ext → r.ext (поля Params); piv_buf (ось v6) — только при piv_sl next | half; стоп без b: k_s = net."""
    r.ext.update(ext)
    if "piv_sl" in ext:
        r.pt4["k_s"] = NET
        r.ext["piv_buf"] = float(v6.get("piv_buf", 0.0)) if ext["piv_sl"] in ("next", "half") else 0.0


def _extra(evr, t: str) -> dict:
    """{ключ: ряд [день]} семейства v5 на реальном td (кэш по тикеру и осям сигнала)."""
    key = (t, tuple(sorted({**evr.r5.sp, **({"norm": evr.v6["norm"]} if "norm" in evr.v6 else {})}.items())))
    cache = evr.__dict__.setdefault("_piv_kesh", {})
    if key not in cache:
        td, ind = evr.sk._td[t]
        sp = dict(key[1])
        ss = get_family(evr.sid, setka="v5")(td, ind, [sp])
        cache[key] = {k: np.asarray(a[0], float) for k, a in ss.extra.items()}
    return cache[key]


def tm(evr, ext: dict, t: str, s0: int, b: int):
    """tejk_M на дни [s0; s0+b] или None, если пивота и pol_puti нет."""
    sl = slice(s0, s0 + b + 1)
    if "piv_sl" in ext or "piv_tp" in ext:
        x = _extra(evr, t)
        return PivUrovni(**{k: x[k][sl] for k in ("P", "S1", "S2", "R1", "R2", "vwap", "dev") if k in x})
    if ext.get("tejk") == "pol_puti":
        if evr.sid in ("S5", "S6"):
            return _extra(evr, t)["P" if evr.sid == "S5" else "vwap"][sl]
        from fastbt import tejk_signal
        return tejk_signal.uroven(evr.sid, evr.r5.sp, evr.sk._td[t][0])[sl]
    return None
