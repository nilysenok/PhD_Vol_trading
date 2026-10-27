"""Прогон сборок Э1-г на train (Ж-031): 36 сборок на топ-1 точках S1–S6 (лучшая по плато), якорь и A на новых точках.

Цель — настройка 2018; контроль 2019 отдельно; комиссии 0,04 и 0,035. Validation и ≥ 2022 не читаются (`Sweep`).
Выход `A_sborki.csv`: строки kind = sborka (36) | S_centr_ravnye (НЕ A: S1–S6 в центрах сеток, равные веса, без лимитов; прежнее имя
A_ishodnaya) | A_boevaya_s9 (боевая A через `s9.Engine.port`, Ж-039 п. 1; база сравнения A′) | A_novye_ravnye (топ-1 точки, равные веса, без лимитов — одна из 36); колонка top2 = 1/2 у двух лучших по цели при плато ≥ 3.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from fastbt import sweep_drivers as sd
from fastbt.sweep_a import BASE, COMMS, SkPanelSource, make_sweep, run_sborki
from fastbt.sweep_val_baza import S9Baza, baza_ref
from fastbt.sweep_val_kand import SIDS, TABLES, kandidaty

STATUS = "разработка, train 2018 (tune) / 2019 (ctrl), до взгляда на validation"
KEYS = list(BASE)


def anchor_row(sw, kind: str) -> dict:
    """Строка в колонках `run_sborki` для сборки BASE на панели sw."""
    row = dict(kind=kind, **BASE)
    res = sw.evaluate(BASE, "setup", COMMS[0])
    row.update(objective=sd.objective_stable(res), plateau_neighbors=0)
    for c in COMMS:
        for per, tag in (("setup", "tune"), ("control", "ctrl")):
            x = sw.evaluate(BASE, per, c)
            row.update({f"{tag}_{m}_c{c:g}": x[m] for m in ("sharpe", "msr", "sortino", "mdd", "oborot_y", "dni_v_rynke")})
    row["dsr"] = sd.dsr(res["sr_d"], sw.trials_sr(), res["T"], res["skew"], res["kurt"])["dsr"]
    return row


def s9_row(baza, kind: str) -> dict:
    """Строка боевой A (`s9.Engine.port`, Ж-039 п. 1) в колонках `run_sborki`; objective и dsr у неё нет (не сборка, проб нет)."""
    row = dict(kind=kind, **BASE, plateau_neighbors=0)
    for c in COMMS:
        for per, tag in (("setup", "tune"), ("control", "ctrl")):
            x = baza.evaluate({}, per, c)
            row.update({f"{tag}_{m}_c{c:g}": x[m] for m in ("sharpe", "msr", "sortino", "mdd", "oborot_y", "dni_v_rynke")})
    return row


def mark_top2(t: pd.DataFrame, top2: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["top2"] = 0
    for i, r in enumerate(top2.itertuples(), 1):
        t.loc[(t.vesa == r.vesa) & (t.limit_pozitsiy.astype(str) == str(r.limit_pozitsiy))
              & (t.limit_na_strategiyu.astype(str) == str(r.limit_na_strategiyu)) & (t.kind == "sborka"), "top2"] = i
    return t


def run(root=".", out=None) -> pd.DataFrame:
    kd = {s: kandidaty(s, root) for s in SIDS}
    new, old = {s: kd[s]["cands"][0]["point"] for s in SIDS}, {s: kd[s]["base"] for s in SIDS}
    sw_new, sw_old = make_sweep(SkPanelSource(new)), make_sweep(SkPanelSource(old))
    res = run_sborki(sw_new)
    t = res["table"].copy()
    t.insert(0, "kind", "sborka")
    ravn = anchor_row(sw_new, "A_novye_ravnye")
    anc = dict(anchor_row(sw_old, "S_centr_ravnye"), pometka="не A: S1–S6 в центрах сеток, равные веса")
    a9 = dict(s9_row(S9Baza(baza_ref("A")), "A_boevaya_s9"), pometka="боевая A, s9.Engine.port; база сравнения A′")
    t = mark_top2(pd.concat([t, pd.DataFrame([anc, a9, ravn])], ignore_index=True), res["top2"]).fillna({"top2": 0})
    t["top2"] = t["top2"].astype(int)
    h = hashlib.sha1(json.dumps(new, sort_keys=True, default=str).encode()).hexdigest()[:12]
    head = (f"# статус: {STATUS}; объект A_sborki; {datetime.now():%d.%m.%Y %H:%M}\n"
            f"# точки (топ-1 по плато из S{{k}}_top3.csv) {h}; N сборок {res['n']} (модуль {res['n_modul']}); S_centr_ravnye — исходные точки Sk, НЕ A; A_boevaya_s9 — боевая A\n"
            f"# top2 — две лучшие по цели setup при плато ≥ 3 соседей; нетто «net» = лимитов нет\n")
    f = Path(out or Path(root) / TABLES / "A_sborki.csv")
    f.write_text(head + t.to_csv(index=False), "utf-8")
    return t


if __name__ == "__main__":
    tab = run(sys.argv[1] if len(sys.argv) > 1 else ".")
    print(tab[["kind", "vesa", "limit_pozitsiy", "limit_na_strategiyu", "objective", "tune_sharpe_c0.04", "ctrl_sharpe_c0.04",
               "plateau_neighbors", "top2"]].round(3).to_string())
