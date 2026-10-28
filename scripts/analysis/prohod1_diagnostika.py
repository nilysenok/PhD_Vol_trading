# -*- coding: utf-8 -*-
"""Диагностика прохода 1 «редко и метко» (Ж-091 п. 2–3; docs/otchety/2026-10-28_prohod1_diagnostika.md). Справочно, train4, потолок.

Запуск (один процесс на Sk):  python3 scripts/analysis/prohod1_diagnostika.py schet S3 [N_placebo N_k2]
Свод по всем Sk:             python3 scripts/analysis/prohod1_diagnostika.py svod
Ledger — только чтение. Контроли: fastbt.progon_kontrol (placebo_sig_fn, k2_sig_fn, p_perm) через evaluate_v2 с sig_fn — то же,
что run_kandidat делает внутри, но без k1/k3/возмущений (они не нужны этой диагностике; код контролей не менялся).
Фильтров по σ-режиму нет: терцили — только описание сделок базовых точек.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prohod1_diag_metriki import metriki, sdelki, terciles  # noqa: E402
from prohod1_diag_vybor import vybor  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LEDGER = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/docs/tables/progon_e1/_ledger_v2")
OUT = ROOT / "docs/tables/progon_e1"
SHAPKA = "# разработка train4 (2016–2019), справочно, потолок: выбор и счёт на одних данных; в отбор не идёт\n"
N_PLACEBO, N_K2 = 10, 20        # как в progon_kontrol (объявленные N_PLACEBO, N_K2)


def zapis(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(SHAPKA + df.to_csv(index=False), encoding="utf-8")


def schet(sid: str, n_pl: int = N_PLACEBO, n_k2: int = N_K2) -> None:
    from fastbt import progon_kontrol as pk, progon_sobol as ps
    top, info = vybor(LEDGER, sid)
    print(sid, info, flush=True)
    ev = ps.fabrika_real(sid).evr
    lags = pk.k2_lags(20)[:: max(20 // n_k2, 1)][:n_k2]            # равномерная выборка из 20 объявленных лагов к2
    tochki, kontroli, terc = [], [], []
    for _, p in top.iterrows():
        shared = dict(sid=sid, sobol_i=p.sobol_i, ustojchiva=p.ustojchiva)
        runs = [("baza", "", {})] + [("placebo", f"placebo:{s}", {"sig_fn": pk.placebo_sig_fn(s)}) for s in range(n_pl)] \
            + [("k2", f"k2:lag{g}", {"sig_fn": pk.k2_sig_fn(g)}) for g in lags]
        for tip, nm, vz in runs:
            row, _, _ = ev.evaluate_v2(p.pt, "train4", vz)
            df = sdelki(ev) if row["status"] == "ok" else pd.DataFrame()
            m = metriki(row, df) if row["status"] == "ok" else {}
            (tochki if tip == "baza" else kontroli).append({**shared, "kontrol": nm, **m, **({"msr_zaem": p.msr_vsego} if tip == "baza" else {})})
            if tip == "baza" and len(df):
                terc.append(terciles(df).assign(**shared))
        print(sid, int(p.sobol_i), "ok", flush=True)
    zapis(pd.DataFrame(tochki), f"diag_tochki_{sid}.csv")
    zapis(pd.DataFrame(kontroli), f"diag_kontrol_{sid}.csv")
    zapis(pd.concat(terc) if terc else pd.DataFrame(), f"diag_terciles_{sid}.csv")


def main(argv: list) -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if argv[0] == "schet":
        schet(argv[1], *[int(x) for x in argv[2:4]])
    elif argv[0] == "svod":
        from prohod1_diag_svod import svod
        svod(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
