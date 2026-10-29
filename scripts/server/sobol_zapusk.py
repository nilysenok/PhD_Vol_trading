#!/usr/bin/env python3
"""Счёт Соболя на сервере через существующий fastbt.progon_sobol (замена, пока fastbt.progon_server не слит; Ж-133 (г)).

Только train4 (progon_sobol считает только его, Ж-082). Одна стратегия — свой каталог <root>/<Sk>/ (Pisatel). Блоки по 5 000
точек: после каждого — строка состояния по каждой стратегии; ledger пишется кусками по 100 точек с хэшем кода и входов, возобновление —
по klyuch_sha закрытых точек (повторный запуск досчитывает). Процессов — не больше progon_sobol.MAX_PROC (10). Метрик не печатает.
    python3 sobol_zapusk.py --root ledger/train4 --procs N [--strategii S1,S2] [--tochek 50] [--kod-slit]
    python3 sobol_zapusk.py --root ledger/train4 --sostoyanie
"""
import argparse
import sys
import time
from pathlib import Path

BLOK = 5000


def predel(c: dict) -> tuple:
    sb = c["sobol"]
    p1 = int(sb.get("prohod1", sb.get("N1")))
    return p1, int(sb.get("prohod2") or p1)


def bloki(n1: int, p1: int) -> list:
    """[(a, b, ist)]: по 5 000, граница прохода 1 (p1) — граница блока."""
    out, a = [], 0
    while a < n1:
        b = min(a + BLOK, n1, p1 if a < p1 else n1)
        out.append((a, b, "sobol1" if a < p1 else "sobol2"))
        a = b
    return out


def sostoyanie(root: Path, sids: list, cel: dict) -> list:
    from fastbt import progon_sobol as ps
    out = []
    for s in sids:
        n = len(ps.zakrytye(root, s)) if (root / s / "manifest.json").exists() else 0
        out.append(f"{s}: закрыто {n} из {cel.get(s, '?')} ({time.strftime('%F %T')})")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--strategii", default="")
    ap.add_argument("--tochek", type=int, help="точек на стратегию (по умолчанию prohod2 куба)")
    ap.add_argument("--kod-slit", action="store_true")
    ap.add_argument("--sostoyanie", action="store_true")
    a = ap.parse_args(argv)
    from fastbt import ledger_v2 as lm
    from fastbt import progon_sobol as ps
    sids = a.strategii.split(",") if a.strategii else ps.SIDS
    kubs = {s: ps.load_kub(s) for s in sids}
    [ps.proverit_kub(c) for c in kubs.values()]
    cel = {s: a.tochek or predel(c)[1] for s, c in kubs.items()}
    if a.sostoyanie:
        print("\n".join(sostoyanie(a.root, sids, cel)))
        return 0
    procs = min(a.procs, ps.MAX_PROC)
    print(f"sobol: {','.join(sids)}, точек {cel}, процессов {procs} (ядра − 4 = {a.procs}; потолок progon_sobol {ps.MAX_PROC}), "
          f"root {a.root}", flush=True)
    ps.proverit_disk(sum(cel.values()), a.root, ps.BAIT_V21)
    B = {s: bloki(cel[s], predel(kubs[s])[0]) for s in sids}
    oshibok = 0
    for k in range(max(len(b) for b in B.values())):
        t0 = time.time()
        for s in [s for s in sids if k < len(B[s])]:
            x0, x1, ist = B[s][k]
            itog = ps.schet([s], x0, x1, ist, a.root, procs, a.kod_slit, False, lm.V21)
            oshibok += itog[s]["oshibok"]
        print(f"чекпоинт {k + 1}: блок {k * BLOK}…{(k + 1) * BLOK - 1} за {time.time() - t0:.0f} с, "
              f"ошибок оценщика {oshibok}", flush=True)
        for line in sostoyanie(a.root, sids, cel):
            print("  " + line, flush=True)
    print("sobol: готово" + (f"; точек со status = oshibka {oshibok}" if oshibok else ""), flush=True)
    return 2 if oshibok else 0


if __name__ == "__main__":
    sys.exit(main())
