"""V6-12d: оси выхода тренда на стоп-заявке (S3/S4: bezubytok, stop_v_plyus, pir, reentry; inventarizaciya §1.2 И16, V6-11a–c #666/#667/#671).

В кубе v6 тренда эти оси v5 стоят в разделе ispolnenie (новые), и razdelit уводит их в v6. `vernut` возвращает bezubytok, stop_v_plyus, pir
в точку v5 перед razbor (он их разбирает как у контртренда: b0/b0_c, sp, pir), reentry = da — Params.reentry_den.
Ось T (срок заявки) для тренда не подключена: T > 1 — заявки по уровню (k), у стоп-заявки и входа по закрытию их нет (exec_e._check_params).
Ось smena не подключена: у стоп-заявки сигнал не выходит никогда (Ж-091), а vhod_tip = podtverzhdenie в куб не открыт.
reentry при входе по подтверждению (k нет) движок отвергает → NeSchitana; свёртка в canon_v6.
"""
from __future__ import annotations

from fastbt.etap1_eval_v2 import NeSchitana

OSI = ("bezubytok", "stop_v_plyus", "pir")


def vernut(sid_trend: bool, pt5: dict, v6: dict) -> dict:
    """Точка v5 + оси выхода тренда из v6 (только у тренда; у контртренда эти оси уже в pt5)."""
    return {**pt5, **{k: v6[k] for k in OSI if k in v6}} if sid_trend else pt5


def primenit(sid_trend: bool, r, v6: dict) -> None:
    if not sid_trend or v6.get("reentry") != "da":
        return
    if r.nov["vhod_tip"] != "stop_zayavka":
        raise NeSchitana("reentry (тренд): повторный вход — только на стоп-заявке (при входе по подтверждению заявки по уровню нет)")
    r.ext["reentry_den"] = True
