"""Моррис по кубу v7 (лист V7-8a): тот же механизм, что progon_morris_v6 (траектории morris_v6, правило morris_v6_zamorozka, счёт srv, flock, чекпоинт),
но канон — fastbt.canon_v7 (лист V7-1b; API как у canon_v6: load_kub, proverit, osi, canon_v7, NOVYE, tochki_v6→tochki_v7, KUB6→каталог куба v7 под тем же именем KUB6), оценщик — EvaluatorV7, период polnaya.
Подмена `canon_v6` → `canon_v7` в модулях progon_morris_v6 и progon_morris_v6_polnyj — только на время прохода (выход — назад), копии кода нет.
Выход: <out>/<Sk>/{morris_y.jsonl, morris_<Sk>.csv, zamorozka.yaml}. Данных сам не читает. Запуск:
python3 -m fastbt.progon_morris_v7 --kub v7 --period polnaya --out <каталог> --procs N [--r 1]."""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

from fastbt import progon_morris_v6 as pm, progon_morris_v6_polnyj as pol, progon_server as srv, progon_sobol as ps
from fastbt import progon_v7 as p7

MODULI = (pm, pol)


@contextlib.contextmanager
def kanon_v7(cn):
    """canon_v6 → cn в модулях Морриса v6 на время прохода; нет атрибута canon_v6 в модуле — он пропускается."""
    staryj = [(m, m.canon_v6) for m in MODULI if hasattr(m, "canon_v6")]
    for m, _ in staryj:
        m.canon_v6 = cn
    try:
        yield
    finally:
        for m, c in staryj:
            m.canon_v6 = c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Моррис v7 → заморозка (zamorozka.yaml)")
    ap.add_argument("--kub", required=True, choices=[p7.SETKA])
    ap.add_argument("--period", required=True, choices=[p7.PERIOD])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--procs", type=int, default=srv.procs_po_umolchaniyu())
    ap.add_argument("--strategii", default=",".join(p7.SEMEJSTVA))
    ap.add_argument("--r", type=int, help="траекторий (по умолчанию 5000 // (k + 1)); 1 — малая проверка")
    ap.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prodolzhit", action="store_true")
    a = ap.parse_args(argv)
    from fastbt import sobytiya_e1
    sobytiya_e1.trebovat_svod()
    cn = p7.canon7()
    rz = ps.Rezhim(p7.SETKA, a.period, "seed", False, "t0d")
    staryj_ocen, ps.OCENSHCHIK_V5 = ps.OCENSHCHIK_V5, ("fastbt.eval_v7", "EvaluatorV7")
    try:
        with p7.zona_ctx(a.period), kanon_v7(cn), p7.podmeny({}, p7.SHAG):
            for sid in a.strategii.split(","):
                i = pm.strategiya(sid, a, a.out, rz, a.r)
                print(f"{sid}: точек {i['tochek']}, статусы {i['statusy']}, исключений {i['oshibok']}, заморожено {len(i['zamorozheno'])}: {i['zamorozheno']}")
    finally:
        ps.OCENSHCHIK_V5 = staryj_ocen
    return 0


if __name__ == "__main__":
    sys.exit(main())
