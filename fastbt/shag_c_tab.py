"""Шаги 1–3 главы 4 (v3): отбор С8 v2, устойчивость, разложение, комиссии, проверка сдвига σ — по таблице сетки
`fastbt.shag_c.run_setka`. Чистые функции над DataFrame/движком; пометка — «разработка 2018–2021, после аудита»."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt import s9, shag0
from fastbt.shag_c import COMM_MAIN, COMM_REF, COMM_SEL, Base, evaluate, lw, tag

KEY = ["sigma", "h", "family", "W", "p1", "p2", "point"]


def pick(df: pd.DataFrame, c: float):
    """Максимум Шарпа среди прошедших С8 v2 при комиссии c; None — не прошёл никто."""
    t = tag(c)
    ok = df[df[f"proshla_c8v2_{t}"].astype(bool)]
    return None if ok.empty else ok.loc[ok[f"sharpe_{t}"].idxmax()]


def _line(r, c: float, rol: str) -> dict:
    t = tag(c)
    out = {k: r[k] for k in KEY}
    return dict(pometka=shag0.POMETKA, rol=rol, comm_pct=c * 100, sharpe=r[f"sharpe_{t}"],
                usl1_c8=r[f"usl1_c8_{t}"], usl2_k3=r[f"usl2_k3_{t}"], usl3_k2=r[f"usl3_k2_{t}"],
                itog_c8v2=r[f"proshla_c8v2_{t}"], p_vs_k2=r[f"p_vs_k2_{t}"], **out)


def otbor(df: pd.DataFrame) -> tuple:
    """(таблица, выбранная строка). Шапка: выбор при 0,04; та же точка при 0,035; справочно выбор при 0,05
    (отдельная строка, если точка другая). Дальше все точки с флагами условий 1, 2, 3 и итогом при 0,04.
    Нет прошедших — строка-результат, а Шарп-лидер идёт справочно (не отобран)."""
    best, ref = pick(df, COMM_SEL), pick(df, COMM_REF)
    rows = []
    if best is None:
        rows.append(dict(pometka=shag0.POMETKA, rol="никто не проходит С8 v2 при 0,04 (результат шага)"))
    else:
        rows += [_line(best, COMM_SEL, "ВЫБРАНА при 0,04"), _line(best, COMM_MAIN[0], "та же точка при 0,035")]
    if ref is None:
        rows.append(dict(pometka=shag0.POMETKA, rol="справочно: при 0,05 не проходит никто"))
    elif best is None or ref["point"] != best["point"] or ref["h"] != best["h"]:
        rows.append(_line(ref, COMM_REF, "справочно: выбор при 0,05 — ДРУГАЯ точка"))
    else:
        rows.append(_line(ref, COMM_REF, "справочно: выбор при 0,05 — та же точка"))
    rows += [_line(r, COMM_SEL, "кандидат") for _, r in df.iterrows()]
    return pd.DataFrame(rows), best


def ustoychivost(df: pd.DataFrame, best, c: float = COMM_SEL, tol: float = 0.1) -> pd.DataFrame:
    """Соседи ±1 по W и по p (p1 или p2 на 10 п.; та же семья, тот же h): число соседей в пределах `tol` Шарпа
    и вывод: край (p50 — по построению, либо соседей нет) / плато (≥ половины соседей в пределах tol) / пик."""
    t, ws = tag(c), sorted(df["W"].unique())
    rows = []
    for _, r in df.iterrows():
        s = df[(df["family"] == r["family"]) & (df["h"] == r["h"])]
        dw = (s["W"].map(ws.index) - ws.index(r["W"])).abs()
        dp = (s["p1"] - r["p1"]).abs() + (s["p2"] - r["p2"]).abs()
        nb = s[((s["W"] == r["W"]) & (dp == 10)) | ((dp == 0) & (dw == 1))]
        near = int((np.abs(nb[f"sharpe_{t}"] - r[f"sharpe_{t}"]) <= tol).sum())
        verdikt = ("край" if r["p1"] == 50 or len(nb) == 0 else "плато" if near * 2 >= len(nb) else "пик")
        rows.append(dict(pometka=shag0.POMETKA, **{k: r[k] for k in KEY}, comm_pct=c * 100, sharpe=r[f"sharpe_{t}"],
                         n_sosedey=len(nb), v_predelah_tol=near, tol_sharpe=tol, vyvod=verdikt,
                         vybrana=bool(best is not None and r["point"] == best["point"] and r["h"] == best["h"])))
    return pd.DataFrame(rows)


def razlozhenie(eng, bases: dict, g: np.ndarray, reps: int) -> list:
    """Δ к A = (к3 − A) + (C − к3), обе части с p ЛВ; экспозиция и оборот против A и к3; доля закрытых дней
    A по каждой семье S1–S6 (A ≠ 0 и затвор 0) и средний недобор веса (1 − g)."""
    rows = []
    for c in COMM_MAIN:
        b = bases[c]
        pC = eng.port(b.A * g, comm=c)
        sA, sK, sC = (s9.sharpe(x["excess"].values) for x in (b.pA, b.pK, pC))
        (_, p1), (_, p2), (_, p3) = lw(b.pK, b.pA, reps), lw(pC, b.pK, reps), lw(pC, b.pA, reps)
        rows.append(dict(rezhim="разложение", comm_pct=c * 100, d_vs_A=sC - sA, p_vs_A=p3, chast1_k3_minus_A=sK - sA,
                         p_chast1=p1, chast2_C_minus_k3=sC - sK, p_chast2=p2,
                         summa_chastey_minus_d=(sK - sA) + (sC - sK) - (sC - sA)))
        for lab, p in (("A", b.pA), ("к3", b.pK), ("точка C", pC)):
            st = shag0.stats(p, np.ones(len(p), bool))
            rows.append(dict(rezhim="экспозиция/оборот", comm_pct=c * 100, row=lab, sharpe=st["sharpe"],
                             expo_pct=st["expo_pct"], turn_y=st["turn_y"], dni_v_rynke_pct=st["dni_v_rynke_pct"]))
    sidr, A = shag0.sid_row(eng), np.asarray(eng.A_row, float)
    for s in shag0.SIDS:
        m = (sidr == s) & (A != 0)
        rows.append(dict(rezhim="закрыто фильтром", semya=s, dnei_s_poziciey_A=int(m.sum()),
                         zakryto_dnei_pct=float(np.mean(g[m] == 0) * 100) if m.any() else np.nan,
                         nedobor_vesa_pct=float(np.mean(1 - g[m]) * 100) if m.any() else np.nan))
    return [dict(pometka=shag0.POMETKA, **r) for r in rows]


def komissiya(eng, g: np.ndarray, reps: int, comms=(0.00035, 0.0004, 0.0005, 0.0006)) -> list:
    """Выбранная точка, A и к3 при 0,035 / 0,04 / 0,05 / 0,06 (0,05 и 0,06 — приложение)."""
    rows = []
    for c in comms:
        b = Base(eng, c)
        full = evaluate(b, g, reps)
        for lab, p in (("A", b.pA), ("к3", b.pK)):
            rows.append(dict(row=lab, **shag0.stats(p, np.ones(len(p), bool))))
            rows[-1]["comm_pct"] = c * 100
        rows.append(dict(row="точка C", comm_pct=c * 100, **full))
        for r in rows[-3:]:
            r["prilozhenie"] = c > COMM_SEL
    return [dict(pometka=shag0.POMETKA, **r) for r in rows]


def sdvig(csv_path, sigma: str) -> pd.DataFrame:
    """Строки data/regen/ch4_dev/proverka_sdvig.csv по рядам σ шага (без контролей): наивная — `naive`,
    HAR-J — `harj_b_h*`, гибрид — `hyb_h*`. Нужны столбцы ryad, ok (формат ch4_dev_sigma.shift_row)."""
    d = pd.read_csv(csv_path, sep=";", decimal=",", encoding="utf-8-sig")
    pref = {"naive": "naive", "harj": "harj_b", "hyb": "hyb"}[sigma]
    keep = d["ryad"].astype(str).str.startswith(pref)
    for col in ("rol", "tip"):
        if col in d:
            keep &= ~d[col].astype(str).str.contains("контрол", case=False)
    return d[keep]


def sdvig_line(d: pd.DataFrame) -> str:
    """Первая строка журнала: итог проверки сдвига (§0.1)."""
    ok = d["ok"].astype(str).str.lower().isin(["true", "1"]) if len(d) else pd.Series(dtype=bool)
    if not len(d) or not ok.all():
        return "ПРОВЕРКА СДВИГА σ: НЕ ПРОЙДЕНА или нет строк — счёт запрещён"
    return f"проверка сдвига σ (k = −2…+1, максимум на последнем известном дне): {len(d)} рядов, все допущены"
