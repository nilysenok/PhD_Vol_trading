# -*- coding: utf-8 -*-
"""pbo_pravilo_sprav.py — справочный пересчёт PBO итераций 1, 2, 4 из старого ledger без нового счёта (Ж-080).

Ledger (до дивидендов, архив) читается только на чтение, sha256 сверяется с SHA256SUMS.txt до использования
(zh051_kriteriy.load_ledger). Вердикты итераций 1–10 не пересматриваются; таблица — справочная.
Запуск: FASTBT_MAIN=<своя копия> python3 -m fastbt.pbo_pravilo_sprav [--ledger <папка _ledger>]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fastbt import etap1_poisk as P
from fastbt import etap1_poisk_vorota as V
from fastbt import pbo_pravilo as B
from fastbt import zh051_kriteriy as K

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/tables/zh051/pbo_pravilo_it124.csv"
SHAPKA = ("# статус: справочно, разработка train 2018–2019; ledger ДО дивидендов (#370), в архиве, нового счёта нет; "
          "вердикты итераций 1–10 не пересматриваются (Ж-080)\n")
ITER = {1: ("it2", "pbo_it1"), 2: ("it2", "pbo_obshchaya"), 4: ("it4", "pbo_obshchaya")}


def nayti_ledger() -> Path | None:
    """_ledger в этой копии или в основной (предок папки .claude)."""
    for d in (ROOT, *ROOT.parents):
        p = d / "docs/tables/zh051/_ledger"
        if (p / "SHA256SUMS.txt").exists() and (p / "S1/ledger_it1.pkl").exists():
            return p
    return None


def plan_base(sid: str, S: dict) -> tuple:
    d = ROOT / "docs/tables/zh047"
    f = d / f"{sid}_itog_v2.csv"
    row = pd.read_csv(f if f.exists() else d / f"{sid}_itog.csv", comment="#").iloc[-1]
    plan = str(row["osi_plana"]).split()
    return plan, {a: P._coerce(str(row[f"p_{a}"]), S["axes"][a]) for a in S["axes"] if a not in plan}


def staroe(sid: str, it: int) -> float:
    f, col = ITER[it]
    return float(pd.read_csv(ROOT / f"docs/tables/zh051/{f}/{sid}_itog.csv", comment="#").iloc[-1][col])


def schitat(sid: str, it: int, led_dir: Path) -> dict:
    S = P.load_setka(sid)
    name = f"ledger_it{it}.pkl"
    L = K.load_ledger(sid, name, None, led_dir)
    plan, base = plan_base(sid, S)
    pr = B.pravilo_etapa(L, S["axes"], plan, base, S["yakor"])
    M, kw = V.matrix(L).T, pr["select_fn"].keywords
    assert M.shape[0] == len(pr["coords"]), "столбцы pravilo_etapa не совпали с V.matrix"
    a = B.cscv(M, pr["coords"], B.vybor_argmax, dates=pr["dates"])
    c = B.cscv(M, pr["coords"], pr["select_fn"], dates=pr["dates"])
    c2 = B.cscv(M, pr["coords"], pr["select_fn"], dates=pr["dates"], oos_fn=pr["oos_fn"])
    old = staroe(sid, it)
    return dict(strategiya=sid, iteraciya=it, N_tochek=M.shape[0], T_dney=M.shape[1], N_v_kube=int((kw["kub"].col >= 0).sum()),
                N_dopustimyh=int(kw["dop"].sum()), pbo_argmax=a["pbo"], pbo_staroe=old, argmax_sovpal=bool(abs(a["pbo"] - old) < 1e-12),
                pbo_centr_plato_oos_msr=c2["pbo"], pbo_centr_plato_oos_sredn=c["pbo"], razbienij=c["n"], propusk_centr=c["n_propusk"],
                osi_plana=" ".join(plan), ledger=name)


def main(argv=None) -> list:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, default=nayti_ledger())
    a = ap.parse_args(argv)
    rows = [schitat(f"S{k}", it, a.ledger) for k in range(1, 7) for it in ITER]
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"{OUT}: реестр только дополняется — файл уже есть")
    OUT.write_text(SHAPKA, encoding="utf-8")
    df.to_csv(OUT, mode="a", index=False, float_format="%.6f")
    print(df.to_string())
    return rows


if __name__ == "__main__":
    main()
