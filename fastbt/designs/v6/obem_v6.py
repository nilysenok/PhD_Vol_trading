"""Объём куба v6 по стратегиям (V6-13): номинал, без оси tf и после заморозки осей (формула §3.2 инвентаризации).

Запуск: `FASTBT_MAIN=<копия> python3 fastbt/designs/v6/obem_v6.py [ось ...]` — оси в аргументах считаются замороженными на «выкл.»
(замороженная родительская ось замораживает и дочерние). Данных не читает.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from fastbt import canon_v5 as c5  # noqa: E402
from fastbt import canon_v6 as c6  # noqa: E402


def zamorozhennye(kub: dict, F) -> set:
    """F + все дочерние оси, у которых заморожены все родители."""
    zam, nov = set(F), c6.novye(kub)
    izm = True
    while izm:
        izm = False
        for n, v in nov.items():
            if n not in zam and v.get("roditel") and set(v["roditel"]) <= zam:
                zam.add(n)
                izm = True
    return zam


def tablica(F=()) -> list:
    """[(sid, осей, V5, V6 номинал, V6 без tf, V6 после заморозки F)]."""
    out = []
    for sid in c6.SIDS:
        k = c6.load_kub(sid)
        c6.proverit(k)
        out.append((sid, len(c6.osi(k, None)), c5.obem(k["v5"]), c6.obem(k), c6.obem(k, ["tf"]), c6.obem(k, zamorozhennye(k, F))))
    return out


def main() -> int:
    F = sys.argv[1:]
    print("sid | осей | V5 | V6 номинал | V6 без tf | V6 после заморозки " + str(F))
    for sid, n, v5, v6, bez, zam in tablica(F):
        print(f"{sid} | {n} | {v5:.3e} | {v6:.3e} | {bez:.3e} | {zam:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
