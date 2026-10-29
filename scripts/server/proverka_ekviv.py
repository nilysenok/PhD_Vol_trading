#!/usr/bin/env python3
"""Проверка эквивалентности на сервере (Ж-131 п. 3, Ж-133 п. 4): пересчёт набора эталона Мака и сравнение (scripts/server/ekviv.py).

OK — пишется отметка (--otmetka: kod=<sha256 tar>, dannye=<sha256 манифеста>, etalony=<sha256 npz>), без неё zapusk_sobol.sh
не стартует. FAIL — отметки нет (старая удаляется до счёта), выход 1, первой строкой — первое расхождение. Допуск не расширяется.
    python3 proverka_ekviv.py --etalony etalony_<коммит>.npz --otmetka proverka/OK --kod-sha <..> --dannye-sha <..> [--procs N]
    python3 proverka_ekviv.py ... --moi <npz>     # сравнить с готовым файлом вместо пересчёта (тест, повторный разбор)
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ekviv  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--etalony", required=True)
    ap.add_argument("--otmetka", required=True)
    ap.add_argument("--kod-sha", required=True)
    ap.add_argument("--dannye-sha", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--moi", help="готовый npz вместо пересчёта")
    a = ap.parse_args(argv)
    otm = Path(a.otmetka)
    otm.parent.mkdir(parents=True, exist_ok=True)
    otm.unlink(missing_ok=True)
    eta, meta_eta = ekviv.zagruzit(a.etalony)
    tochki = ekviv.tochki_iz(eta)
    print(f"proverka: эталон {Path(a.etalony).name} ({meta_eta.get('machine')}, numba {meta_eta.get('numba')}, "
          f"numpy {meta_eta.get('numpy')}): якорей {sum(m.startswith('yakor') for m in eta)}, точек {len(tochki)}", flush=True)
    t0 = time.time()
    if a.moi:
        moi, meta_moi = ekviv.zagruzit(a.moi)
    else:
        moi, meta_moi = ekviv.poschitat(tochki, a.procs)
        meta_moi.update(ekviv.sreda_meta())
    ok, bad, dmax = ekviv.sravnit(eta, meta_eta, moi, meta_moi)
    nf = ekviv.net_fajla(moi)
    if nf:
        ok, bad = False, [f"{nf[0]}: оценщик не нашёл файл — данных не хватает ({len(nf)} записей)"] + bad
    verdikt = dict(ok=ok, rashozhdeniy=len(bad), pervye=bad[:20], max_delta_yakor=dmax, tochek=len(tochki),
                   etalon=meta_eta, zdes=meta_moi, sekund=round(time.time() - t0, 1), vremya=time.strftime("%F %T %Z"))
    (otm.parent / f"verdikt_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(
        json.dumps(verdikt, ensure_ascii=False, indent=1, default=repr))
    if not ok:
        print(f"proverka: расхождений {len(bad)} (первые 20 — {otm.parent}/verdikt_*.json); счёт не запускать, вывод — дирижёру")
        print(f"FAIL: {bad[0]}")
        return 1
    eh = hashlib.sha256(Path(a.etalony).read_bytes()).hexdigest()
    otm.write_text(f"kod={a.kod_sha}\ndannye={a.dannye_sha}\netalony={eh}\n")
    print(f"OK: якорь A_fix 6 Sk max |Δ| = {dmax:.1e} ≤ {ekviv.TOL_YAKOR:g}; {len(tochki)} точек побайтно; "
          f"{verdikt['sekund']:.0f} с; отметка {otm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
