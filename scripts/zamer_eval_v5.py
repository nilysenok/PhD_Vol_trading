"""Л9: замер оценщика v5 (медленный путь) — N точек Соболя куба v5 на Sk, train4, все тикеры. Метрик не печатает.

Печатает на Sk: точек, ошибок (исключений), с/точку (медиана, среднее), исполнений вне [low; high] бара (#604).
Запуск: FASTBT_MAIN=<своя копия> python3 scripts/zamer_eval_v5.py [--n 20] [--sid S1 ...]
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastbt import canon_v5  # noqa: E402
from fastbt import progon_sobol as ps  # noqa: E402
from fastbt.eval_v5 import EvaluatorV5, vne_bara  # noqa: E402

PERIOD = "train4"


def zamer(sid: str, n: int) -> dict:
    """Через драйвер (#608): ps.fabrika_real → ps.ocenshchik → EvaluatorV5(…, fdr куба); точки ps.tochki (якоря ∪ канон)."""
    rz = ps.Rezhim("v5", PERIOD, "seed", False, canon_v5.load_kub(sid)["fdr"])
    sr = ps.fabrika_real(sid, rz)
    assert isinstance(sr.evr, EvaluatorV5) and sr.evr.fdr == rz.fdr
    pts = ps.tochki(ps.load_kub(sid, rz), 0, n, rz)
    ts, osh, vsego, vne, pr, ns = [], [], 0, 0, [], []
    for i, pt in pts:
        t0 = time.perf_counter()
        try:
            row, x, tik = ps.schitat_tochku(sr, sid, i, pt, "zamer_l9", False, False, rz=rz)
            if row["status"] == "ne_schitana":
                ns.append((i, row["prichina"]))
                continue
            if row["status"] != "ok" or x is None or not np.isfinite(x).all() or tik.shape[0] != len(sr.tikery) \
                    or sr.evr.poz is None or sr.evr.poz[1].shape != tik.shape:
                raise ValueError(f"status {row['status']}: {row.get('prichina')}")
            evr = sr.evr
            v = vne_bara(evr)
            vsego, vne, pr = vsego + v["vsego"], vne + v["vne"], pr + v["primery"]
        except Exception as e:                   # noqa: BLE001 — замер считает ошибки, не прерывается
            osh.append((i, f"{type(e).__name__}: {e}"[:300], traceback.format_exc()[-700:]))
        ts.append(time.perf_counter() - t0)                              # с/точку — только посчитанные (ok или ошибка)
    return dict(sid=sid, tochek=len(pts), oshibok=len(osh), ne_schitano=len(ns), med=float(np.median(ts)) if ts else float("nan"),
                sred=float(np.mean(ts)) if ts else float("nan"),
                ispolneniy=vsego, vne_bara=vne, primery=pr[:3], osh=osh[:3])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--sid", nargs="*", default=canon_v5.SIDS)
    a = ap.parse_args(argv)
    print("sid | точек | ошибок | ne_schitana | с/точку медиана | с/точку среднее | исполнений | вне [low; high]")
    bad = 0
    for sid in a.sid:
        z = zamer(sid, a.n)
        bad += z["oshibok"]
        print(f"{sid} | {z['tochek']} | {z['oshibok']} | {z['ne_schitano']} | {z['med']:.2f} | {z['sred']:.2f} | {z['ispolneniy']} | {z['vne_bara']}")
        for o in z["osh"]:
            print("  ошибка", o[0], o[1], "\n", o[2])
        for p in z["primery"]:
            print("  вне бара", p)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
