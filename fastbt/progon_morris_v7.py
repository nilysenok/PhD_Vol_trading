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
from fastbt import canon_v7 as c7, progon_v7 as p7

MODULI = (pm, pol)


class Kanon7:
    """Вместо модуля canon_v6 в модулях Морриса v6: те же имена (load_kub, proverit, osi, canon_v6, novye, NOVYE, KUB6), но по семье:
    S1–S6 — canon_v7, S8–S12 — canon_v7b (progon_v7.semya). Оси — GOTOVY_V7 (как у сэмплера Соболя). Второго определения канона нет."""
    NOVYE = (c7.OSI7,)
    KUB6 = c7.KUB7
    novye = staticmethod(c7.novye)

    @staticmethod
    def load_kub(sid):
        return p7.semya(sid).load_kub(sid, p7.kub_put7(sid).parent)

    @staticmethod
    def proverit(kub):
        p7.semya(kub["sid"]).proverit(kub)

    @staticmethod
    def osi(kub):
        return p7.semya(kub["sid"]).osi(kub, c7.GOTOVY_V7)

    @staticmethod
    def canon_v6(sid, pt, kub):
        m = p7.semya(sid)
        return (m.canon_v7b if hasattr(m, "canon_v7b") else m.canon_v7)(sid, pt, kub)


def spisok_osej7(kub: dict, ox: list) -> list:
    """Оси для правила заморозки: новая — из раздела osi7 куба v7 (замораживается по правилу), остальные не замораживаются (как «оси v5» у v6)."""
    nov = set(c7.novye(kub))
    return [{"os": n, "gruppa": c7.OSI7 if n in nov else "v6", "novaya": n in nov} for n, _, _ in ox]


@contextlib.contextmanager
def kanon_v7(cn=Kanon7):
    """canon_v6 → cn (по умолчанию Kanon7) и spisok_osej → spisok_osej7 в модулях Морриса v6 на время прохода; выход — назад; нет атрибута — модуль пропускается."""
    staryj = [(m, m.canon_v6) for m in MODULI if hasattr(m, "canon_v6")]
    so, pm.spisok_osej = pm.spisok_osej, spisok_osej7
    for m, _ in staryj:
        m.canon_v6 = cn
    try:
        yield
    finally:
        pm.spisok_osej = so
        for m, c in staryj:
            m.canon_v6 = c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Моррис v7 → заморозка (zamorozka.yaml)")
    ap.add_argument("--kub", required=True, choices=[p7.SETKA])
    ap.add_argument("--period", required=True, choices=[p7.PERIOD, "train4"], help="train4 — только сухой прогон на Маке (данных 2022+ нет)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--procs", type=int, default=srv.procs_po_umolchaniyu())
    ap.add_argument("--strategii", default=",".join(p7.SEMEJSTVA))
    ap.add_argument("--r", type=int, help="траекторий (по умолчанию 5000 // (k + 1)); 1 — малая проверка")
    ap.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prodolzhit", action="store_true")
    a = ap.parse_args(argv)
    from fastbt import sobytiya_e1
    sobytiya_e1.trebovat_svod()
    p7.canon7()                                                  # отказ, если V7-1b не слит
    rz = ps.Rezhim(p7.SETKA, a.period, "seed", False, "t0d")
    staryj_ocen, ps.OCENSHCHIK_V5 = ps.OCENSHCHIK_V5, ("fastbt.eval_v7", "EvaluatorV7")
    try:
        with p7.zona_ctx(a.period), kanon_v7(), p7.podmeny({}, p7.SHAG):
            for sid in a.strategii.split(","):
                i = pm.strategiya(sid, a, a.out, rz, a.r)
                print(f"{sid}: точек {i['tochek']}, статусы {i['statusy']}, исключений {i['oshibok']}, заморожено {len(i['zamorozheno'])}: {i['zamorozheno']}")
                [print(f"  ошибка ×{k}: {m}") for m, k in i["prichiny"].items()]
    finally:
        ps.OCENSHCHIK_V5 = staryj_ocen
    return 0


if __name__ == "__main__":
    sys.exit(main())
