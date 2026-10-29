"""Лист R-в реверса р.1 для A_zashch = A (s9) (Ж-124 п. 1): овернайт против дня, test1. Вызывается из r1_v.py --obekt A_zashch. Справочно, без выбора.
Определение: позиция строки d (решение по close 18:50 дня d) держится close d → close d+1 (lr из кэша s9, дивиденд ex-дня внутри).
Внутридневная часть I[d] = ln(close[d+1]/open[d+1]) (open 10:00), ночная N[d] = lr[d] − I[d] = ln(open[d+1]/close[d]) + дивиденд.
Части × позиция усредняются по числу пар на дату — арифметика Engine.port; I + N = gross порта (проверка 1e-12). Плата за шорт 5 % вычитается из x
(как в r1_a_plata), в части не входит. Строка таблицы = день решения d (как у A_strogaya). Позиции — eng.A_row (путь r1_a_cd: ApproachA → turnover.apply)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts" / "analysis", ROOT / "scripts" / "revers"):
    sys.path.insert(0, str(p))
import overnajt as O  # noqa: E402
import overnajt_lib as L  # noqa: E402
from fastbt import a_strogaya as AS, metriki_e as me  # noqa: E402


def vnutridnevnaya(o, c) -> np.ndarray:
    """I[i] = ln(c[i+1]/o[i+1]) — день i+1: open 10:00 → close 18:50. Нет open/close дня i+1 (NaN, ≤ 0) и последняя строка — 0:
    вся доходность уходит в ночную часть (как kniga_razlozhenie)."""
    I = np.zeros(len(c))
    if len(c) > 1:
        o1, c1 = np.asarray(o, float)[1:], np.asarray(c, float)[1:]
        ok = (o1 > 0) & (c1 > 0)
        I[:-1][ok] = np.log(c1[ok] / o1[ok])
    return I


def chasti_dnya(lr_row, I_row, pos, date_idx, cnt):
    """(I, N) по датам: pos × часть lr, среднее по числу пар на дату (cnt) — арифметика Engine.port; N = lr − I."""
    nd = len(cnt)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.bincount(date_idx, pos * I_row, nd) / cnt, np.bincount(date_idx, pos * (lr_row - I_row), nd) / cnt)


def pod_dvizhok(eng, keep):
    """Копия Engine на подмножестве строк (пары одной категории): первый бар пары, число пар на дату, lr — пересчитаны."""
    e = copy.copy(eng)
    e.r = {k: v[keep] for k, v in eng.r.items()}
    e.first = np.r_[True, e.r["pair"][1:] != e.r["pair"][:-1]]
    e.lr_row = eng.lr_row[keep]
    e.cnt = np.bincount(e.r["date"], minlength=len(eng.dates)).astype(float)
    return e


def doli_ravny(I, N, gross, atol: float = 1e-12) -> float:
    """Макс. |I + N − gross порта|; больше atol — отказ (части не складываются в доходность движка)."""
    ecc = float(np.nanmax(np.abs(I + N - gross)))
    assert ecc <= atol, f"A_zashch: части ≠ gross порта s9 ({ecc:.2e})"
    return ecc


def stroki(eng, comm: float, plata: float, baza_fn, god: float = 3.0) -> tuple:
    """(строки, x0 всего A_zashch без платы, ecc): Sk по отдельности и категории как в stroki_a (+ недельные колонки в конце, Ж-124 п. 3).
    x = excess порта s9 − плата × база шорта; god = 3 (2022–2024)."""
    from fastbt.data import load_ticker
    Ik = np.zeros_like(eng.lr)
    for ti, t in enumerate(eng.tickers):
        td = load_ticker(t, False)
        assert np.array_equal(td.dates, eng.cache[t]["dates"]), f"{t}: даты load_ticker ≠ кэш s9"
        Ik[ti, :len(td.dates)] = vnutridnevnaya(td.o, td.c)
    I_row, sid_row = Ik[eng.r["tick"], eng.r["bar"]], np.array([eng.pairs[p][1] for p in eng.r["pair"]])
    rows, x0, ecc = [], None, 0.0
    for imya, sids in [(s, (s,)) for s in AS.SIDS] + [("A_zashch (S1–S6)", AS.SIDS)] + O.KATEGORII[1:5]:
        keep = np.isin(sid_row, sids)
        e, pos = pod_dvizhok(eng, keep), eng.A_row[keep]
        port = e.port(pos, comm=comm)
        I, N = chasti_dnya(e.lr_row, I_row[keep], pos, e.r["date"], e.cnt)
        ecc = max(ecc, doli_ravny(I, N, port["gross"].to_numpy()))
        x = port["excess"].to_numpy(float)
        x0 = x if len(sids) == len(AS.SIDS) else x0
        xp = x - plata * baza_fn(pos, eng.cal[e.r["date"]], e.r["date"], e.cnt)
        r = L.stroka(imya, ",".join(sids), I, N, xp, port["exposure"].to_numpy(float) > 0, god)
        r.update(me.stroka_ned(xp, e.dates.to_numpy().astype("datetime64[D]")))
        rows.append(r)
    return rows, x0, ecc


def sverka_a_cd(x0, a_cd: Path, tol: float = 1e-10) -> float:
    """Sharpe A_zashch (полные, плата 0) против a_cd_test1.csv (A (s9), «(а) полные»); больше tol — отказ."""
    ob = pd.read_csv(a_cd, comment="#")
    ref = float(ob[ob.portfel.str.startswith("A (s9") & (ob.stroka == "(а) полные")].sharpe.iloc[0])
    d = abs(L.sharpe(x0) - ref)
    if not d <= tol:
        raise SystemExit(f"A_zashch: Sharpe {L.sharpe(x0)} ≠ {a_cd.name} {ref} (|Δ| = {d:.2e} > {tol:.0e})")
    print(f"сверка с {a_cd.name}: Sharpe A_zashch (плата 0) |Δ| = {d:.2e}", flush=True)
    return d


def main(out: Path | None) -> None:
    import r1_a_cd as CD
    import r1_a_plata as PL
    from fastbt import s9, zona
    CD.ustanovit_period("test1")
    eng = s9.Engine(CD.kesh("test1"))
    rows, x0, ecc = stroki(eng, AS.COMM, AS.PLATA, PL.baza_shorta)
    base = ROOT / "docs" / "tables" / "revers_r1" / ("_net" if zona.div_net() else "")
    sverka_a_cd(x0, base / "a_cd_test1.csv")
    df = pd.DataFrame(rows)
    out = out or base / "v_overnajt_a_zashch.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    note = f"# объект: A_zashch = A (s9); I = ln(close[d+1]/open[d+1]), N = lr − I; max|I+N−gross порта| = {ecc:.1e}; плата за шорт 5 % в x\n"
    out.write_text(CD.shapka("test1") + note + df.to_csv(index=False), encoding="utf-8")
    print(df.round(3).to_string())
