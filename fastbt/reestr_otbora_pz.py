"""Реестр отбора A, дополнение перезаморозки 2 (вариант (а)). Только дописывает строки (otbor_statistika п. 4), старые не трогает.

Новые строки (источник `zamorozka_2`, ключ — (стратегия, источник, ключ), повторный запуск ничего не добавляет):
* проба sweep_e1v2 без боевого эталона (`ekviv_filtr.prichina_net`) — `otbroshena_net_etalona`;
* бывший кандидат с эталоном, не вошедший в новый топ-3 — `otbroshena_rang`;
* новый кандидат — `kandidat_zamorozki`; в причине бывших кандидатов отмечен дубль по подписи (S5: ранги 2–3 прежнего списка — одна
  торговля, в фабрике рабочим был ранг 4);
* соседи, достроенные для плато воспроизводимых точек (источник `sweep_e1v2_dostroyka`): `otbroshena_net_etalona` или `otbroshena_rang`.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastbt import ekviv_filtr as ef
from fastbt import reestr_otbora as rs
from fastbt import sweep_setka as sk

ZAM, DOSTR = "zamorozka_2", "sweep_e1v2_dostroyka"
S = "net_etalona"


def tochka(sid: str, klyuch: str, root=".") -> dict:
    """Параметры пробы из ключа реестра (неактивные и None — из базы сетки)."""
    base = sk.expand(Path(root) / "fastbt" / "designs" / "sweep" / f"{sid}_setka_e1b.yaml")["base"]
    return {**base, **{a: rs.norm(b) for a, b in (kv.split("=", 1) for kv in klyuch.split(";") if kv)}}


def _podpis(r) -> tuple:
    return (r["strategiya"], *(round(float(r[c]), 9) if r.get(c) not in ("", None) else None for c in ("tune_004", "ctrl_004", "storon_god")))


def novaya(r: dict, istochnik: str, status: str, prichina: str, ekviv: str) -> dict:
    return {**r, "istochnik": istochnik, "status": status, "prichina": prichina, "ekviv_s9": ekviv}


def stroki(starye: list, res: dict, kand, root=".") -> list:
    """Дописываемые строки: по старым строкам sweep_e1v2, достроенным пробам и новым кандидатам."""
    out, novye, vidno = [], {(r.strategiya, r.klyuch) for r in kand.itertuples()}, set()
    for r in starye:
        if r["istochnik"] != "sweep_e1v2":
            continue
        s, w = r["strategiya"], ef.prichina_net(r["strategiya"], tochka(r["strategiya"], r["klyuch"], root))
        note = ""
        if r["status"] == "kandidat_zamorozki":          # S5: ранги 2–3 старого списка — одна торговля, фабрика брала ранг 4
            note = " (бывший кандидат" + ("; дубль подписи ранга выше, в фабрике пропущен)" if _podpis(r) in vidno else ")")
            vidno.add(_podpis(r))
        if w:
            out.append(novaya(r, ZAM, "otbroshena_" + S, "нет боевого эталона: " + w + note, "net"))
        elif r["status"] == "kandidat_zamorozki" and (s, r["klyuch"]) not in novye:
            out.append(novaya(r, ZAM, "otbroshena_rang", "есть эталон, но не в новом топ-3 по плато среди воспроизводимых", "da"))
    for r in kand.itertuples():
        d = {"strategiya": r.strategiya, "klyuch": r.klyuch, "rezhim": "plato_vosp", "tune_004": rs.fmt(r.tune_004), "ctrl_004": rs.fmt(r.ctrl_004),
             "tune_0035": rs.fmt(r.tune_sharpe_c0035), "ctrl_0035": rs.fmt(r.ctrl_sharpe_c0035), "storon_god": rs.fmt(r.storon_y),
             "plato": rs.fmt(r.plateau), "sosedey": rs.fmt(r.neighbors), "aktivnost_ok": "1", "plato_ok": "1", "vyrozhdena": "0"}
        out.append(novaya(d, ZAM, "kandidat_zamorozki", f"топ-3 по плато среди воспроизводимых (вариант (а)), место {r.rank} в пуле; N_vsego {r.N_vsego}", "da"))
    for s, x in res.items():
        eksist = {rs.klyuch_str(rs.kanon_ledger(k)) for k in json.load(open(f"{rs.NOVYE}/_ledger/{s}/ledger.json", encoding="utf-8"))}
        for k, sr in x["ledger"].items():
            kl = rs.klyuch_str(rs.kanon_ledger(k))
            if kl in eksist:
                continue
            w = ef.prichina_net(s, tochka(s, kl, root))
            d = {"strategiya": s, "klyuch": kl, "rezhim": "ledger", "tune_004": rs.fmt(rs.godovoy(sr))}
            out.append(novaya(d, DOSTR, "otbroshena_" + S if w else "otbroshena_rang",
                              ("нет боевого эталона: " + w) if w else "сосед, достроенный для плато воспроизводимых точек", "net" if w else "da"))
    return out


def dopisat(res: dict, kand, root="."):
    starye = rs.chitat(rs.VYHOD)
    novye = stroki(starye, res, kand, root)
    rs.zapisat(rs.sliyanie(starye, novye))
