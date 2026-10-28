# -*- coding: utf-8 -*-
"""Свод диагностики по Sk: точность (IR сделки) базовых точек против контролей к2 и плацебо Л-9, p по перестановкам; терцили σ.

p Sk-уровня: для каждого повтора контроля — среднее IR сделки по точкам группы; p = (1 + #{повтор ≥ база}) / (n + 1) (progon_kontrol.p_perm).
Группа: ust = устойчивые точки (mSR каждого года > 0, Ж-090), dobor = добор неустойчивыми (отступление, справочно).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SIDS = [f"S{k}" for k in range(1, 7)]


def _csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, comment="#") if p.exists() else pd.DataFrame()


def _p(fakt: float, vals) -> float:
    from fastbt.progon_kontrol import p_perm
    return p_perm(fakt, vals)


def svod_ir(out: Path) -> pd.DataFrame:
    rows = []
    for sid in SIDS:
        b, k = _csv(out / f"diag_tochki_{sid}.csv"), _csv(out / f"diag_kontrol_{sid}.csv")
        for ust, gr in b.groupby("ustojchiva"):
            kk = k[(k.sobol_i.isin(gr.sobol_i))]
            base = float(gr.ir_sdelki.median())                              # медианы по точкам: выбросы с горсткой сделок не тянут
            r = dict(sid=sid, gruppa="ust" if ust else "dobor", tochek=len(gr), sdelok_v_god=gr.sdelok_v_god.median(),
                     ekspoziciya=gr.ekspoziciya.median(), sr_sdelki=gr.sr_sdelki.median(), ir_baza=base,
                     sharpe_rasch=gr.sharpe_rasch.median(), sharpe_fakt=gr.sharpe_fakt.median(), skew=gr["skew"].median(),
                     kurt=gr["kurt"].median(), msr=gr.msr.median(), msr_zaem=gr.msr_zaem.median(),
                     msr_minus_sharpe=(gr.msr - gr.sharpe_fakt).median())
            for tip in ("placebo", "k2"):
                kt = kk[kk.kontrol.str.startswith(tip)]
                rep = kt.groupby("kontrol").ir_sdelki.median()              # медиана по точкам на повтор
                r.update({f"ir_{tip}": float(rep.median()), f"p_{tip}": _p(base, rep.values), f"povtorov_{tip}": len(rep),
                          f"sdelok_v_god_{tip}": float(kt.sdelok_v_god.median()),
                          f"tochek_ir_vyshe_vseh_{tip}": int(sum(float(p.ir_sdelki) > kt[kt.sobol_i == p.sobol_i].ir_sdelki.max() for p in gr.itertuples()))})
            rows.append(r)
    return pd.DataFrame(rows)


def svod_terciles(out: Path) -> pd.DataFrame:
    t = pd.concat([_csv(out / f"diag_terciles_{s}.csv") for s in SIDS], ignore_index=True)
    t["gruppa"] = np.where(t.ustojchiva, "ust", "dobor")
    t["s"] = t.sr_sdelki * t.sdelok
    t["w"] = t.dolya_pribyl * t.sdelok
    g = t.groupby(["sid", "gruppa", "terc"])
    r = pd.DataFrame(dict(sdelok=g.sdelok.sum(), sr_sdelki=g.s.sum() / g.sdelok.sum(), dolya_pribyl=g.w.sum() / g.sdelok.sum()))
    return r.reset_index()


def svod(out: Path) -> None:
    from prohod1_diagnostika import zapis
    a, b = svod_ir(out), svod_terciles(out)
    zapis(a, "diag_svod.csv")
    zapis(b, "diag_terciles_svod.csv")
    print(a.round(4).T.to_string())
    print(b.round(4).to_string())
