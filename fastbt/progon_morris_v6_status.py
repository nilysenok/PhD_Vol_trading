"""Y одной точки Морриса v6 и её статус (Ж-137 п. 2). Данных сам не читает (оценщик — sr.evr). Выделено из progon_morris_v6 (≤ 200 строк)."""
from __future__ import annotations

import importlib

import numpy as np

from fastbt import morris_v6_zamorozka as mz, progon_sobol as ps
from fastbt.metriki_e import sr_dnevnoy

DO = dict(zip(mz.OKNA, ("2016-12-31", "2017-12-31", "2018-12-31")))
NAN3 = [float("nan")] * 3
STATUSY = ("ok", "ne_schitana", "none_kanon", "oshibka")


def sdelok_vsego(r: dict) -> float:
    """Число сделок точки по `sdelok_y` строки ledger ({год: сделок}); нет поля → NaN (неизвестно)."""
    sy = r.get("sdelok_y")
    return float(sum(sy.values())) if isinstance(sy, dict) and sy else float("nan")


def status_okna(sr, pt, period: str, fast: bool = False) -> tuple:
    """(Y трёх окон, статус) для точки — см. `ocenka`."""
    return ocenka(sr, pt, period, fast)[:2]


def ocenka(sr, pt, period: str, fast: bool = False) -> tuple:
    """(Y трёх окон, статус, число сделок) для точки; статус: ok / ne_schitana / none_kanon / oshibka (status оценщика не ok и не ne_schitana).
    Число сделок — NaN, если точка не ok или оценщик его не дал. None-канон → NaN; excess None (нет сделок) → 0 (план §3.3: эффект оси, не пропуск).
    fast — быстрый слой как в progon_sobol (fast_etap1.patched() для покрытых точек): Y Морриса и Соболя идут одним путём;
    исключение оценщика идёт наверх."""
    if pt is None:
        return list(NAN3), "none_kanon", float("nan")
    if fast and ps.fast_ok(pt):
        with importlib.import_module("fastbt.fast_etap1").patched():
            r, ex_p, _ = sr.evr.evaluate_v2(pt, period)
    else:
        r, ex_p, _ = sr.evr.evaluate_v2(pt, period)
    st = r.get("status", "ok")
    if st != "ok":
        return list(NAN3), "ne_schitana" if st == "ne_schitana" else "oshibka", float("nan")
    ns = sdelok_vsego(r)
    if ex_p is None:
        return [0.0] * 3, "ok", ns
    x = np.asarray(ex_p, dtype=np.float64)
    if len(x) != len(sr.daty):
        raise ValueError(f"excess {len(x)} дней, дат {len(sr.daty)}")
    out = []
    for d in DO.values():
        w = x[sr.daty <= np.datetime64(d)]
        out.append(sr_dnevnoy(w) if w.size > 1 and np.isfinite(w).all() else float("nan"))
    return out, "ok", ns


def razdelit_nol(nol, ns) -> dict:
    """Точки с Y целиком 0 по числу сделок ns (NaN — неизвестно, не считается): {bez_sdelok: сделок 0, ves0: сделки есть, а вес/Y 0}."""
    nol, ns = np.asarray(nol, dtype=bool), np.asarray(ns, dtype=np.float64)
    return {"bez_sdelok": int((nol & (ns == 0)).sum()), "ves0": int((nol & (ns > 0)).sum())}


def nol_sdelok(Y, st: dict, ns=None) -> tuple:
    """(bool по точкам, сводка {chislo, ok, dolya}): точка ok с Y целиком 0 (Д4). Доля — от точек ok. С ns (число сделок по точкам) в сводке
    ещё два счётчика: bez_sdelok (сделок 0) и ves0 (сделки есть, Y = 0: вес 0 — ранг NaN и т. п.); chislo = их сумма + точки с неизвестным ns."""
    nol = (np.asarray(Y, dtype=np.float64) == 0).all(axis=1)
    nl = {"chislo": int(nol.sum()), "ok": int(st["ok"]), "dolya": round(float(nol.sum()) / st["ok"], 4) if st["ok"] else 0.0}
    return nol, nl if ns is None else {**nl, **razdelit_nol(nol, ns)}


def baza_bez_sdelok(nol, nom, ns=None) -> dict:
    """База траектории = её первая точка (nom[:, 0] — номера уникальных точек): {chislo, r, dolya} баз с Y целиком 0 (страж «база даёт сделки»);
    с ns — ещё bez_sdelok / ves0 (см. nol_sdelok)."""
    i = np.asarray(nom)[:, 0]
    b = np.asarray(nol, dtype=bool)[i]
    d = {"chislo": int(b.sum()), "r": int(b.size), "dolya": round(float(b.mean()), 4) if b.size else 0.0}
    return d if ns is None else {**d, **razdelit_nol(b, np.asarray(ns, dtype=np.float64)[i])}


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
