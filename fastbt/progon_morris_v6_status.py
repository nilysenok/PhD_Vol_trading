"""Y одной точки Морриса v6 и её статус (Ж-137 п. 2). Данных сам не читает (оценщик — sr.evr). Выделено из progon_morris_v6 (≤ 200 строк)."""
from __future__ import annotations

import importlib

import numpy as np

from fastbt import morris_v6_zamorozka as mz, progon_sobol as ps
from fastbt.metriki_e import sr_dnevnoy

DO = dict(zip(mz.OKNA, ("2016-12-31", "2017-12-31", "2018-12-31")))
NAN3 = [float("nan")] * 3
STATUSY = ("ok", "ne_schitana", "none_kanon", "oshibka")


def status_okna(sr, pt, period: str, fast: bool = False) -> tuple:
    """(Y трёх окон, статус) для точки; статус: ok / ne_schitana / none_kanon / oshibka (status оценщика не ok и не ne_schitana).
    None-канон → NaN; excess None (нет сделок) → 0 (план §3.3: эффект оси, не пропуск). fast — быстрый слой как в progon_sobol
    (fast_etap1.patched() для покрытых точек): Y Морриса и Соболя идут одним путём; исключение оценщика идёт наверх."""
    if pt is None:
        return list(NAN3), "none_kanon"
    if fast and ps.fast_ok(pt):
        with importlib.import_module("fastbt.fast_etap1").patched():
            r, ex_p, _ = sr.evr.evaluate_v2(pt, period)
    else:
        r, ex_p, _ = sr.evr.evaluate_v2(pt, period)
    st = r.get("status", "ok")
    if st != "ok":
        return list(NAN3), "ne_schitana" if st == "ne_schitana" else "oshibka"
    if ex_p is None:
        return [0.0] * 3, "ok"
    x = np.asarray(ex_p, dtype=np.float64)
    if len(x) != len(sr.daty):
        raise ValueError(f"excess {len(x)} дней, дат {len(sr.daty)}")
    out = []
    for d in DO.values():
        w = x[sr.daty <= np.datetime64(d)]
        out.append(sr_dnevnoy(w) if w.size > 1 and np.isfinite(w).all() else float("nan"))
    return out, "ok"


def nol_sdelok(Y, st: dict) -> tuple:
    """(bool по точкам, сводка {chislo, ok, dolya}): точка ok с Y целиком 0 = 0 сделок во всех окнах Y (Д4). Доля — от точек ok."""
    nol = (np.asarray(Y, dtype=np.float64) == 0).all(axis=1)
    return nol, {"chislo": int(nol.sum()), "ok": int(st["ok"]), "dolya": round(float(nol.sum()) / st["ok"], 4) if st["ok"] else 0.0}


def baza_bez_sdelok(nol, nom) -> dict:
    """База траектории = её первая точка (nom[:, 0] — номера уникальных точек): {chislo, r, dolya} баз с Y целиком 0 (страж «база даёт сделки»)."""
    b = np.asarray(nol, dtype=bool)[np.asarray(nom)[:, 0]]
    return {"chislo": int(b.sum()), "r": int(b.size), "dolya": round(float(b.mean()), 4) if b.size else 0.0}


def itog_nol(sid: str, nl: dict, bz: dict) -> str:
    """Строки итога драйвера: счётчики и «ВНИМАНИЕ» (nol_sdelok > 50 %; baza_bez_sdelok > 0); rc не меняется."""
    s = [f"  nol_sdelok: {nl}", f"  baza_bez_sdelok: {bz}", vnimanie(sid, nl)]
    if bz["chislo"] > 0:
        s.append(f"ВНИМАНИЕ: {sid}: {bz['chislo']} из {bz['r']} баз траекторий без сделок — эффекты осей на этой траектории не измерены")
    return "\n".join(x for x in s if x)


def vnimanie(sid: str, nl: dict) -> str | None:
    """Строка «ВНИМАНИЕ», если у Sk доля точек без сделок > mz.DOLYA_NOL (rc драйвера не меняется)."""
    if nl["dolya"] <= mz.DOLYA_NOL:
        return None
    return f"ВНИМАНИЕ: {sid}: {nl['chislo']} из {nl['ok']} точек ok без сделок во всех окнах Y ({nl['dolya']:.0%} > {mz.DOLYA_NOL:.0%}) — Y по Sk малоинформативен"


def y_okna(sr, pt, period: str, fast: bool = False) -> list:
    return status_okna(sr, pt, period, fast)[0]
