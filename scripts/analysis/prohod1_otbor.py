"""Разбор прохода 1: регион (Л-7), walk-forward (Л-8), PBO по правилу (Ж-080), ворота train и исход (Ж-088 п. 1).
Только вызовы fastbt.*; код отбора не правится. Разработка 2016–2019 (train4), потолок."""
from __future__ import annotations

import math

import numpy as np

import prohod1_lib as L  # noqa: F401  (путь и FASTBT_MAIN)
from fastbt.pbo_pravilo import cscv, msr_oos, msr_stolbcy, vybor_argmax
from fastbt.progon_region import (K_OSI, MIN_CHLENOV, NK, RADIUS, _aktivna, _cel_stolbcov, _lvl, _pt, kodirovat, osi_rows, rang_osey, regiony,
                                  ryad_regiona, select_fn_region, vybor_region)
from fastbt.progon_wf import OSNOVNOE, PRAVILA, _sr, pbo_wf, wf
from fastbt.tselevaya import GODA_TRAIN4, msr_i_gody

PBO_MAX, MIN_SDELOK, MIN_PLATO, POROG = 0.2, 30, 3, 1.5
NAN = float("nan")


def tsentr_stroka(res: dict, rows: list) -> int:
    """Строка центра: в клетке центра (по ведущим осям) допустимая по активности с лучшей целевой; индекс в rows."""
    X = kodirovat(rows, res["osi"])
    kand = np.flatnonzero((X == np.array(res["centr"])).all(axis=1))
    akt = [int(i) for i in kand if _aktivna(rows[i], None)] or [int(i) for i in kand]
    t = res["tstolb"]
    return max(akt, key=lambda i: t[i] if math.isfinite(t[i]) else -math.inf)


def urovni_tsentra(rows: list, res: dict) -> dict:
    pts = [_pt(r) for r in rows]
    return {a: sorted({p.get(a) for p in pts}, key=_lvl)[c] for a, c in zip(res["osi"], res["centr"])}


def msr_ryada(x: np.ndarray, daty: np.ndarray) -> dict:
    m, py, ust, ys = msr_i_gody(np.asarray(x, float)[:, None], daty, GODA_TRAIN4)
    return dict(mSR=float(m[0]), ustojchiv=bool(ust[0]), **{f"mSR_{y}": float(py[0, i]) for i, y in enumerate(ys)})


def region_stroka(res, rows, daty, M5) -> tuple:
    """Шаг 3: (строка таблицы, индекс центра | None). Ансамбль — среднее рядов членов (ряд региона из vybor_region)."""
    if res is None:
        return dict(region="net", chlenov=0), None
    i = tsentr_stroka(res, rows)
    t = res["tstolb"][res["chleny"]]
    ens, cen = msr_ryada(res["ryad"], daty), msr_ryada(M5[:, i], daty)
    return dict(region="est", osi=";".join(res["osi"]), centr=";".join(f"{a}={v}" for a, v in urovni_tsentra(rows, res).items()),
                chlenov=len(res["chleny"]), chlenov_ustojchivyh=int(np.isfinite(t).sum()), score_msr=res["score"],
                **{f"ansambl_{k}": v for k, v in ens.items()}, tsentr_klyuch_sha=rows[i]["klyuch_sha"],
                **{f"tsentr_{k}": v for k, v in cen.items()}), i


def diagnostika_regiona(M5, rows, daty) -> dict:
    """Справочно: почему региона нет. Повторяет шаги vybor_region (оси по индексу Соболя, регионы радиуса 1) только как счётчики,
    плюс регион по ослабленной целевой «dolya» (≥ 75 % годов с mSR > 0) — для сравнения, в ворота не идёт."""
    tcol = _cel_stolbcov(M5, daty, list(range(NK)), "msr")
    osi_all = osi_rows(rows)
    st = rang_osey(rows, tcol, osi_all)
    osi = [osi_all[i] for i in np.argsort(-np.asarray(st), kind="stable")[:K_OSI]]
    X = kodirovat(rows, osi)
    reg = regiony(rows, osi, RADIUS)
    big = [(c, idx) for c, idx in reg.items() if len(idx) >= MIN_CHLENOV]
    akt = [(c, idx) for c, idx in big if any(_aktivna(rows[i], None) for i in np.flatnonzero((X == np.array(c)).all(axis=1)))]
    ust = [c for c, idx in big if math.isfinite(float(_cel_stolbcov(ryad_regiona(M5, idx)[:, None], daty, list(range(NK)), "msr")[0]))]
    rd = vybor_region(M5, rows, daty, tsel="dolya")
    a = np.array([_aktivna(r, None) for r in rows])
    return dict(aktivnyh_strok_sdelok_ge30=int(a.sum()), aktivnyh_i_ustojchivyh=int((a & np.isfinite(tcol)).sum()),
                osi_top=";".join(osi), regionov_vsego=len(reg), s_chlenov_ge10=len(big), s_aktivnym_tsentrom=len(akt),
                s_ustojchivym_ryadom=len(ust), ustojchivyh_strok_vsego=int(np.isfinite(tcol).sum()),
                dolya_sprav_region="est" if rd else "net", dolya_sprav_chlenov=len(rd["chleny"]) if rd else 0,
                dolya_sprav_ansambl_mSR=msr_ryada(rd["ryad"], daty)["mSR"] if rd else NAN)


def wf_tablica(M5, rows, daty) -> tuple:
    """Шаг 4: 8 правил WF, OOS 2018Q1–2019Q4; PBO_wf — pbo_progon_wf (CSCV по 8 правилам, argmax внутри — как в Л-8)."""
    rs = {p: wf(M5, rows, daty, p) for p in PRAVILA}
    oos = np.column_stack([rs[p]["oos_ryad"] for p in PRAVILA])
    tab = []
    for p, r in rs.items():
        v = np.array(list(r["oos_by_quarter"].values()), float)
        tab.append(dict(W=p[0], radius=p[1], tsel=p[2], osnovnoe=int(p == OSNOVNOE), kv_bez_regiona=sum(c is None for c in r["region"].values()),
                        kv_vsego=len(v), mediana_kv=float(np.nanmedian(v)), sharpe_oos=_sr(r["oos_ryad"]),
                        mSR_oos=float(msr_stolbcy(r["oos_ryad"][:, None])[0]), dney_oos=len(r["oos_ryad"])))
    return tab, float(pbo_wf(oos))


def pbo_regiona(M5, rows, daty) -> dict:
    """PBO (Ж-080): внутри каждого из 70 разбиений CSCV выбор — правило региона; справочно — argmax, тот же ранг OOS."""
    coords = [_pt(r) for r in rows]
    a = cscv(M5.T, coords, select_fn_region(rows, daty), 8, daty, msr_oos)
    b = cscv(M5.T, coords, vybor_argmax, 8, daty, msr_oos)
    return dict(pbo=a["pbo"], razbienij=a["n"], bez_kandidata=a["n_propusk"], pbo_argmax_sprav=b["pbo"], razbienij_argmax=b["n"])


def vorota(reg: dict, pbo: dict, row: dict | None) -> dict:
    """Ворота train (Ж-044): PBO ≤ 0,2; плато (≥ 3 устойчивых членов региона — ↺ определение для региона);
    сделки ≥ 30 в каждый год у центра; s9 — н/д (якоря в ledger прохода 1 нет, сверка кода — IG-8)."""
    sd = (row or {}).get("sdelok_y") or {}
    g1 = bool(pbo["razbienij"] and pbo["pbo"] <= PBO_MAX)
    g2 = reg.get("region") == "est" and reg["chlenov_ustojchivyh"] >= MIN_PLATO
    g3 = bool(sd) and all(int(y) in map(int, sd) for y in GODA_TRAIN4) and all(v >= MIN_SDELOK for v in sd.values())
    return dict(G1_pbo=g1, G2_plato=bool(g2), G3_sdelki=g3, G4_s9="nd", sdelok_min=min(sd.values()) if sd else NAN,
                vorota_bez_s9=g1 and bool(g2) and g3)


def iskhod(reg: dict, v: dict) -> str:
    """Ж-088 п. 1 для одной Sk: а — ворота и порог 1,5; б — ворота, порог не достигнут; в — ворота не пройдены (s9 не проверен)."""
    if reg.get("region") != "est" or not v["vorota_bez_s9"]:
        return "в"
    return "а" if reg["ansambl_mSR"] >= POROG else "б"


def otbor_sid(d: dict) -> dict:
    """Шаги 3–5 по загруженной Sk: возвращает таблицы и индекс центра."""
    rows, daty = d["rows"], d["daty"]
    M5 = d["M"] - L.STAVKA_SHORT * d["Z"]
    res = vybor_region(M5, rows, daty)
    reg, ic = region_stroka(res, rows, daty, M5)
    reg.update(diagnostika_regiona(M5, rows, daty))
    wft, pbo_w = wf_tablica(M5, rows, daty)
    pbo = pbo_regiona(M5, rows, daty)
    v = vorota(reg, pbo, None if ic is None else rows[ic])
    return dict(region=reg, ic=ic, wf=wft, pbo_wf=pbo_w, pbo=pbo, vorota=v, iskhod=iskhod(reg, v), res=res, M5=M5)
