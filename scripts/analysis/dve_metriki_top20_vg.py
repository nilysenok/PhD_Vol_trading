# -*- coding: utf-8 -*-
"""Две метрики, п. 2 (в, г) Ж-102: топ-20 точек каждой Sk прохода 1 — как есть / стоп 2σ / размер 1/σ_d. Справочно, train4, потолок.

Точки — те же, что в диагностике прохода 1 (prohod1_diag_vybor.vybor: ledger только чтение). Каждая точка пересчитывается через
EvaluatorV2 (строгое исполнение, fabrika_real, комиссия 0,04 %), от excess вычитается плата за шорт 5 % (база zaem_v2) — как целевая Ж-090.
  kak_est — точка без изменений (пересчёт, не копия ledger; сверка mSR с diag_tochki — колонка msr_diag).
  stop2s  — k_s = 2,0 (если стоп был — заменён). По lm.canon_v2 при k_s ≠ net ось k_x схлопывается в net (тейк «k_x» без стопа
            не действует); R, k_tr, bezubytok остаются как у точки (у точки без стопа они уже net: стопа не было — тейка R тоже нет).
  razmer  — razmer = 20 (дневная ставка равного риска 1/σ_d, цель 20 % годовых, потолок веса 1) вместо «ravnyy»; если у точки
            уже числовой razmer — он сохраняется (такая точка уже в режиме 1/σ_d, варианты kak_est и razmer совпадают).
  ravnyy  — контраст: razmer = ravnyy (равный вес); у точек с «ravnyy» совпадает с kak_est. Показывает, что даёт 1/σ_d точкам,
            у которых он уже был в «как есть».
Свод: группа vse; bez_stopa_isx (k_s исходно net — чистый эффект добавленного стопа); ravnyy_isx (razmer исходно ravnyy — чистый
эффект перехода на 1/σ_d).
Sharpe = mean/sd(ddof=1)·√252, mSR, skew, excess kurt — fastbt.metriki_e на ряду excess − 0,05·zaem. Разрыв = mSR − Sharpe.

Запуск (один процесс на Sk):  python3 scripts/analysis/dve_metriki_top20_vg.py schet S3 ЧАСТИ_ДИР [N]
Свод:                        python3 scripts/analysis/dve_metriki_top20_vg.py svod ЧАСТИ_ДИР
Нужно FASTBT_BARS1M=…/data/regen/D25/bars1m; FASTBT_CACHE — копия кэша (не симлинк).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

LEDGER = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/docs/tables/progon_e1/_ledger_v2")
DIAG = ROOT / "docs/tables/progon_e1"
OUT = ROOT / "docs/tables/dve_metriki"
SHAPKA = ("# разработка train4 (2016–2019), справочно, потолок: топ-20 выбран и считан на одних данных; в отбор не идёт. "
          "Sharpe/mSR с платой за шорт 5 %; Ж-102 п. 2 (в, г)\n")
STAVKA = 0.05
STOP_K, RAZMER = 2.0, 20
VARIANTY = ("kak_est", "stop2s", "razmer", "ravnyy")
METRIKI = ("sharpe", "msr", "razryv", "skew", "kurt")


def variant_pt(sid: str, pt: dict, nazv: str) -> dict:
    """Точка варианта. canon_v2 применяется после замены оси (схлопывание k_x/R/k_tr/bezubytok — его правило)."""
    from fastbt import ledger_v2 as lm
    if nazv == "stop2s":
        return lm.canon_v2(sid, {**pt, "k_s": STOP_K})
    if nazv == "razmer":
        return lm.canon_v2(sid, {**pt, "razmer": RAZMER if pt["razmer"] == "ravnyy" else pt["razmer"]})
    if nazv == "ravnyy":
        return lm.canon_v2(sid, {**pt, "razmer": "ravnyy"})
    return dict(pt)


def metriki_ryada(x: np.ndarray) -> dict:
    """Sharpe, mSR, разрыв, skew, kurt (метрики канона: fastbt.metriki_e) по дневному ряду excess после платы за шорт."""
    from fastbt import metriki_e as me
    x = np.asarray(x, float)
    if not np.isfinite(x).all():
        raise ValueError("в ряду excess есть nan/inf")
    sh, ms = me.sharpe(x), me.msr(x)
    return dict(sharpe=sh, msr=ms, razryv=ms - sh, skew=me.asimmetriya(x), kurt=me.ekscess(x), T=len(x))


def schet_tochki(ev, sid: str, p, nazv: str) -> dict:
    pt = variant_pt(sid, p.pt, nazv)
    row, x, _ = ev.evaluate_v2(pt, "train4", {})
    base = dict(sid=sid, sobol_i=int(p.sobol_i), ustojchiva=bool(p.ustojchiva), variant=nazv,
                k_s=str(pt.get("k_s")), razmer=str(pt.get("razmer")), k_x=str(pt.get("k_x")), R=str(pt.get("R")),
                k_s_isx=str(p.pt.get("k_s")), razmer_isx=str(p.pt.get("razmer")), status=row["status"])
    if row["status"] != "ok":
        return {**base, **{k: float("nan") for k in METRIKI}, "T": 0}
    return {**base, **metriki_ryada(np.asarray(x, float) - STAVKA * np.asarray(ev.zaem, float))}


def schet(sid: str, outdir: Path, n: int | None) -> None:
    from fastbt import progon_sobol as ps
    from prohod1_diag_vybor import vybor
    top, info = vybor(LEDGER, sid)
    top = top.head(n) if n else top
    print(sid, info, "точек", len(top), flush=True)
    ev = ps.fabrika_real(sid).evr
    diag = pd.read_csv(DIAG / f"diag_tochki_{sid}.csv", comment="#").set_index("sobol_i").msr_zaem        # mSR с платой 5 % (msr без платы — не та колонка)
    rows = []
    for _, p in top.iterrows():
        for nazv in VARIANTY:
            r = schet_tochki(ev, sid, p, nazv)
            r["msr_diag"] = float(diag.get(int(p.sobol_i), float("nan"))) if nazv == "kak_est" else float("nan")
            rows.append(r)
        print(sid, int(p.sobol_i), "ok", flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"chast_{sid}.csv").write_text(pd.DataFrame(rows).to_csv(index=False), encoding="utf-8")


def svod(partdir: Path) -> None:
    df = pd.concat([pd.read_csv(f) for f in sorted(partdir.glob("chast_S*.csv"))], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "top20_stop_razmer.csv").write_text(SHAPKA + df.to_csv(index=False), encoding="utf-8")
    grupp = {"vse": df, "bez_stopa_isx": df[df.k_s_isx == "net"], "ravnyy_isx": df[df.razmer_isx == "ravnyy"]}
    parts = []
    for gname, d in grupp.items():
        g = d.groupby(["sid", "variant"], sort=False)
        s = g[list(METRIKI)].median().add_prefix("med_").join(g.size().rename("tochek")).join(g.msr.apply(lambda x: int(x.isna().sum())).rename("msr_nan"))
        parts.append(s.reset_index().assign(gruppa=gname))
    sv = pd.concat(parts, ignore_index=True)
    sv["variant"] = pd.Categorical(sv.variant, VARIANTY, ordered=True)
    sv = sv.sort_values(["gruppa", "sid", "variant"])[["gruppa", "sid", "variant", "tochek", "msr_nan"] + ["med_" + m for m in METRIKI]]
    (OUT / "top20_stop_razmer_svod.csv").write_text(SHAPKA + "# медианы по точкам Sk (nan в медиане пропускается, msr_nan — сколько)\n" + sv.to_csv(index=False), encoding="utf-8")
    kb = df[df.variant == "kak_est"].dropna(subset=["msr_diag"])
    print(sv.to_string(index=False))
    bad = kb[~np.isclose(kb.msr, kb.msr_diag, rtol=1e-6, atol=1e-9, equal_nan=True)]
    print(f"сверка kak_est с diag_tochki (#467): расхождений {len(bad)} из {len(kb)}")
    print(bad[["sid", "sobol_i", "msr", "msr_diag", "sharpe"]].to_string(index=False))


def main(argv: list) -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if argv[0] == "schet":
        schet(argv[1], Path(argv[2]), int(argv[3]) if len(argv) > 3 else None)
    elif argv[0] == "svod":
        svod(Path(argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
