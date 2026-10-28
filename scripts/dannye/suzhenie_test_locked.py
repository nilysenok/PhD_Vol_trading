"""Сужение test_locked/ до 2025+ (Ж-112 п. 1), процедура Ш1–Ш4:
docs/otchety/2026-10-28_test_locked_suzhenie.md (ветка origin/arh-msp-fazy).

Режимы: --manifest-do (Ш1), --plan (Ш2), --rezat (Ш3), --kopirovat-a (класс «а», Ж-118), --manifest-posle (Ш4).
По умолчанию dry-run, без записи; запись — только с --primenit. Значения столбцов не читаются
и не печатаются. Пути — параметрами, относительно --koren.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import suzhenie_obshee as ob  # noqa: E402
import suzhenie_rezka as rz  # noqa: E402


def manifest_do(koren, vkhod, dirs, out, primenit):
    rows = [ob.meta(Path(koren) / r, r) for r in ob.find_files(koren, vkhod, skip=dirs)]
    for r in rows:
        ob.vyvod(**{k: r[k] for k in ("path", "size", "nrows", "dmin", "dmax")})
    if primenit:
        ob.pishi_tsv(Path(koren) / out, rows)
    return rows


def plan(koren, vkhod, dirs):
    for rel in ob.find_files(koren, vkhod, skip=dirs):
        m = ob.meta(Path(koren) / rel, rel)
        dates = ob.dates_of(Path(koren) / rel, m["date_col"]) if m["date_col"] and m["nrows"] else []
        klass, cut, h = ob.classify(rel, m, dates)
        ob.vyvod(path=rel, klass=klass, h=h, cut=cut)


ZAMKI = Path(__file__).resolve().parents[2] / ".claude/hooks/zamki.json"


def proveri_a(koren, rel, m, h, dst_rel, pravila):
    """Причина отказа копирования класса «а» (Ж-118) или None."""
    dst = Path(koren) / dst_rel
    if any(r.search(dst_rel) for r in pravila):
        return "путь копии под замком чтения"
    if m["dmax"] > ob.T1_HI or not ob.okno_ok([], m["dmax"], h):
        return "дата или цель последней строки за 31.12.2024"
    if dst.exists():
        return "копия уже есть, перезапись запрещена"
    return None


def kopirovat_a(koren, vkhod, dirs, primenit):
    """Ж-118: файлы класса «а» копируются (не переносятся) в Z2; число нарушений."""
    z2 = dirs[0]
    pravila = [re.compile(x) for x in json.loads(ZAMKI.read_text())["zakryto_chtenie"]]
    n_a = n_ok = fail = 0
    for rel in ob.find_files(koren, vkhod, skip=dirs):
        m = ob.meta(Path(koren) / rel, rel)
        if not (m["date_col"] and m["nrows"]):
            continue
        klass, _, h = ob.classify(rel, m, ob.dates_of(Path(koren) / rel, m["date_col"]))
        if klass != "а":
            continue
        n_a += 1
        dst_rel = (Path(z2) / rz.put_kopii_a(rel)).as_posix()
        why = proveri_a(koren, rel, m, h, dst_rel, pravila)
        if why:
            ob.vyvod(path=rel, klass=klass, status="FAIL", reason=why)
            fail += 1
            continue
        if primenit:
            dst = Path(koren) / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(koren) / rel, dst)
            if ob.sha_file(dst) != m["sha256"]:
                dst.unlink()
                ob.vyvod(path=rel, klass=klass, status="FAIL", reason="sha копии не совпал")
                fail += 1
                continue
            n_ok += 1
        ob.vyvod(path=rel, klass=klass, status="skopirovano" if primenit else "dry-run")
    print(f"ITOG_A fajlov={n_a} skopirovano={n_ok} fail={fail}")
    return fail


def razbor(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    for f in ("manifest-do", "plan", "rezat", "kopirovat-a", "manifest-posle"):
        g.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--koren", required=True, help="корень, от него считаются все пути")
    ap.add_argument("--vkhod", nargs="+", required=True, help="файлы/каталоги относительно --koren")
    ap.add_argument("--z2-dir", default="data/regen/test1_2022_2024")
    ap.add_argument("--z3-dir", default="data/regen/test_locked/holdout_2025")
    ap.add_argument("--ishodnye-dir", default="data/regen/test_locked/_ishodnye")
    ap.add_argument("--out", help="манифест на выходе (относительно --koren)")
    ap.add_argument("--do-tsv", help="манифест «до» (для --manifest-posle)")
    ap.add_argument("--primenit", action="store_true", help="писать на диск (иначе dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="по умолчанию; без записи")
    return ap.parse_args(argv)


def main(argv=None):
    a = razbor(argv)
    if a.primenit and a.dry_run:
        raise SystemExit("--primenit и --dry-run несовместимы")
    dirs = (a.z2_dir, a.z3_dir, a.ishodnye_dir)
    if a.manifest_do:
        manifest_do(a.koren, a.vkhod, dirs, a.out or "docs/resheniya/manifest_test_locked_do.tsv", a.primenit)
    elif a.plan:
        plan(a.koren, a.vkhod, dirs)
    elif a.rezat:
        return 1 if rz.rezat(a.koren, a.vkhod, dirs, a.primenit) else 0
    elif a.kopirovat_a:
        return 1 if kopirovat_a(a.koren, a.vkhod, dirs, a.primenit) else 0
    else:
        if not a.do_tsv:
            raise SystemExit("--manifest-posle требует --do-tsv")
        out = a.out or "docs/resheniya/manifest_test_locked_posle.tsv"
        return 1 if rz.posle(a.koren, a.vkhod, dirs, a.do_tsv, out, a.primenit) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
