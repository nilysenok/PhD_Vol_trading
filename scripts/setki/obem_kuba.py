#!/usr/bin/env python3
"""Объём куба сеток zh126 (Ж-127 п. 1): сигнал × РМ × фильтры типа.

Читает каталог с `*obshchee*.yaml` и `*_S{k}.yaml` (k = 1..6). Объём стратегии
= точки сигнала (с зависимыми осями) × точки РМ × точки фильтров её типа.
Время полного перебора — при 0,02 с на точку, на 1 и на 10 процессов.

  python3 scripts/setki/obem_kuba.py                  # fastbt/designs/zh126
  python3 scripts/setki/obem_kuba.py --v5 <каталог>   # тот же расчёт для сетки v5
"""
import argparse
import re
import sys
from pathlib import Path

import yaml

KOREN = Path(__file__).resolve().parents[2]
KATALOG = KOREN / "fastbt" / "designs" / "zh126"
SEK_NA_TOCHKU = 0.02
ZAPRET = {"best", "sharpe", "params", "tochki", "rezultat"}
S_FAYL = re.compile(r"_(S[1-6])\.ya?ml$")


def chislo_znacheniy(spec):
    """Число значений оси: список, {znacheniya: [...]} или {ot, do, shag}."""
    if isinstance(spec, list):
        return len(spec)
    if not isinstance(spec, dict):
        raise ValueError(f"ось не разобрана: {spec!r}")
    if "znacheniya" in spec:
        return len(spec["znacheniya"])
    if all(k in spec for k in ("ot", "do", "shag")):
        shagov = (spec["do"] - spec["ot"]) / spec["shag"]
        if shagov < 0 or abs(shagov - round(shagov)) > 1e-9:
            raise ValueError(f"ot/do/shag не делятся нацело: {spec!r}")
        return int(round(shagov)) + 1
    raise ValueError(f"у оси нет znacheniya или ot/do/shag: {spec!r}")


def spisok_znacheniy(spec):
    """Явный список значений оси-родителя."""
    if isinstance(spec, list):
        return spec
    if isinstance(spec, dict) and "znacheniya" in spec:
        return spec["znacheniya"]
    raise ValueError(f"у оси-родителя нет списка znacheniya: {spec!r}")


def normalizovat_zavisimye(zav):
    """Список {os, roditel, znacheniya_po_roditelyu}; допускается и словарь по имени оси."""
    if not zav:
        return []
    if isinstance(zav, dict):
        zav = [dict(v, os=k) for k, v in zav.items()]
    for z in zav:
        if not {"os", "roditel", "znacheniya_po_roditelyu"} <= set(z):
            raise ValueError(f"зависимая ось без os/roditel/znacheniya_po_roditelyu: {z!r}")
    return zav


def obem_bloka(osi, zav):
    """Произведение независимых осей × по каждому родителю сумма по его значениям
    произведения чисел значений его зависимых осей."""
    zav = normalizovat_zavisimye(zav)
    po_roditelyu = {}
    for z in zav:
        po_roditelyu.setdefault(z["roditel"], []).append(z)
    zanyatye = set(po_roditelyu) | {z["os"] for z in zav}
    obem = 1
    for imya, spec in (osi or {}).items():
        if imya not in zanyatye:
            obem *= chislo_znacheniy(spec)
    for roditel, deti in po_roditelyu.items():
        if roditel in (osi or {}):
            znach_rod = spisok_znacheniy(osi[roditel])
        else:
            znach_rod = list(deti[0]["znacheniya_po_roditelyu"])
        summa = 0
        for zr in znach_rod:
            proizv = 1
            for d in deti:
                proizv *= chislo_znacheniy(d["znacheniya_po_roditelyu"].get(zr, []))
            summa += proizv
        obem *= summa
    return obem


def obem_rm(obsh):
    rm = obsh["rm"]
    return obem_bloka(rm.get("osi"), rm.get("zavisimye_osi"))


def obem_filtrov(obsh, tip, imena=None):
    """Произведение чисел значений фильтров типа; `imena` — список из S-файла."""
    imena = imena if imena is not None else obsh["filtry_po_tipu"][tip]
    obem = 1
    for imya in imena:
        f = obsh["filtry"][imya]
        obem *= chislo_znacheniy(f["znacheniya_po_tipu"][tip] if "znacheniya_po_tipu" in f else f)
    return obem


def zapreshchennye_klyuchi(obj, put=""):
    """Пути ключей точек/результатов (best, sharpe, params, tochki, rezultat)."""
    naydeno = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{put}.{k}" if put else str(k)
            if str(k).lower() in ZAPRET:
                naydeno.append(p)
            naydeno += zapreshchennye_klyuchi(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            naydeno += zapreshchennye_klyuchi(v, f"{put}[{i}]")
    return naydeno


def chitat(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def zagruzit(katalog):
    """(общий yaml, {Sk: (имя файла, данные)})."""
    katalog = Path(katalog)
    obshchie = sorted(katalog.glob("*obshchee*.yaml"))
    if not obshchie:
        raise FileNotFoundError(f"нет *obshchee*.yaml в {katalog}")
    s_fayly = {}
    for p in sorted(katalog.glob("*.y*ml")):
        m = S_FAYL.search(p.name)
        if m:
            s_fayly[m.group(1)] = (p.name, chitat(p))
    return chitat(obshchie[0]), s_fayly


def stroka_strategii(sk, dannye, obsh, rm):
    """(объём, текст, расхождение) для одной стратегии."""
    tip = dannye.get("tip") or obsh["tip_strategii"][sk]
    sig = obem_bloka(dannye.get("signal"), dannye.get("zavisimye_osi"))
    filt = obem_filtrov(obsh, tip, dannye.get("filtry_tipa"))
    obem = sig * rm * filt
    zayavleno = dannye.get("obem_signal")
    rashozhdenie = zayavleno is not None and int(zayavleno) != sig
    metka = f"  РАСХОЖДЕНИЕ: obem_signal={zayavleno}" if rashozhdenie else ""
    tekst = f"{sk} ({tip}): сигнал {sig:,} × РМ {rm:,} × фильтры {filt:,} = {obem:,}{metka}"
    return obem, tekst, rashozhdenie


def chasy(obem):
    h = obem * SEK_NA_TOCHKU / 3600
    return f"{h:,.1f} ч на 1 процесс, {h / 10:,.1f} ч на 10 процессов"


def otchet(katalog, metka="zh126"):
    """Печатает объёмы; возвращает код выхода (1 — запрещённые ключи или расхождение)."""
    obsh, s_fayly = zagruzit(katalog)
    kod = 0
    zapret = zapreshchennye_klyuchi(obsh)
    if zapret:
        print("ОШИБКА: в общем yaml ключи точек/результатов:", ", ".join(zapret))
        kod = 1
    rm = obem_rm(obsh)
    print(f"[{metka}] каталог: {katalog}")
    print(f"РМ: {rm:,} точек")
    for tip in ("kontr", "trend"):
        print(f"фильтры {tip}: {obem_filtrov(obsh, tip):,} точек")
    if not s_fayly:
        print("S-файлов нет: объём по стратегиям = сигнал × строки выше")
        return kod
    itog = 0
    for sk in sorted(s_fayly):
        imya, dannye = s_fayly[sk]
        obem, tekst, rash = stroka_strategii(sk, dannye, obsh, rm)
        print(f"{tekst}  [{imya}]  ({chasy(obem)})")
        itog += obem
        kod = 1 if rash else kod
    print(f"ИТОГО: {itog:,} точек ({chasy(itog)})")
    return kod


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--katalog", default=str(KATALOG), help="каталог сеток (по умолчанию zh126)")
    ap.add_argument("--v5", metavar="КАТАЛОГ", help="тот же расчёт для сетки v5")
    a = ap.parse_args(argv)
    if a.v5:
        return otchet(Path(a.v5), "v5")
    return otchet(Path(a.katalog))


if __name__ == "__main__":
    sys.exit(main())
