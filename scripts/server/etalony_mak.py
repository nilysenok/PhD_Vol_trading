#!/usr/bin/env python3
"""Эталоны эквивалентности на Маке (Ж-131 п. 3): якорь A_fix × 6 Sk на train4 + N точек Соболя -> <out>.npz и <out>.npz.sha256.

Запускает `scripts/server/sobrat_perenos.sh` из распакованного tar кода (PYTHONPATH = каталог tar, FASTBT_MAIN = данные Мака,
сверенные с манифестом). Данные — только train4 (≤ 2019); ≥ 2022 и test_locked не читаются. Метрики не печатаются.
    python3 scripts/server/etalony_mak.py --out perenos/etalony_<коммит>.npz [--tochek 200] [--procs 4]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ekviv  # noqa: E402
import ekviv_io  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tochek", type=int, default=ekviv.N_TOCHEK)
    ap.add_argument("--seed", type=int, default=ekviv.SEED)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args(argv)
    t0 = time.time()
    tochki = ekviv.vybor(a.tochek, a.seed)
    print(f"etalony: якорь A_fix × 6 Sk на train4 и {len(tochki)} точек (seed {a.seed}), процессов {a.procs}", flush=True)
    rez, meta = ekviv.poschitat(tochki, a.procs)
    nf = ekviv.net_fajla(rez)
    if nf:
        print(f"FAIL etalony: в {len(nf)} записях оценщик не нашёл файл (данных не хватает), первая {nf[0]}: "
              f"{rez[nf[0]]['sig'][-300:]}")        # хвост записи: там status/oshibka с именем файла
        return 1
    meta.update(ekviv.sreda_meta(), seed=a.seed, tochek=len(tochki), sozdan=time.strftime("%F %T %Z"))
    ekviv_io.sohranit(a.out, rez, meta)
    h = hashlib.sha256(Path(a.out).read_bytes()).hexdigest()
    Path(a.out + ".sha256").write_text(f"{h}  {Path(a.out).name}\n")
    oshibok = sum('"status": "oshibka"' in r["sig"] for r in rez.values())
    print(f"etalony: {a.out} sha256 {h}; записей {len(rez)}, со status = oshibka {oshibok}; "
          f"{time.time() - t0:.0f} с; {json.dumps({k: meta[k] for k in ('python', 'machine', 'numba', 'numpy')})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
