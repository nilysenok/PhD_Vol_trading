#!/usr/bin/env python3
"""Эталоны эквивалентности на Маке (Ж-131 п. 3): якорь A_fix × 6 Sk на train4 + N точек Соболя -> <out>.npz и <out>.npz.sha256.
Плюс набор v6 (ekviv_v6.py: якорь v6 × 6 Sk + 200 точек куба v6, эталон и быстрый слой; записи `v6*__…`, v5-часть не меняется).

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
import ekviv_v6  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tochek", type=int, default=ekviv.N_TOCHEK)
    ap.add_argument("--seed", type=int, default=ekviv.SEED)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--tochek-v6", type=int, default=ekviv_v6.N_V6, help="точек Соболя куба v6 (поровну по Sk, только status = ok)")
    ap.add_argument("--bez-v6", action="store_true", help="только набор v5 (как до листа v6)")
    a = ap.parse_args(argv)
    t0 = time.time()
    tochki = ekviv.vybor(a.tochek, a.seed)
    print(f"etalony: якорь A_fix × 6 Sk на train4 и {len(tochki)} точек (seed {a.seed}), процессов {a.procs}", flush=True)
    rez, meta = ekviv.poschitat(tochki, a.procs)
    t5 = time.time() - t0
    if not a.bez_v6:
        rez6, m6, kod6 = ekviv_v6.poschitat(None, set(), a.procs, a.tochek_v6)
        ekviv_v6.dobavit(rez, meta, rez6, m6, kod6)
        print(f"etalony: v6 — записей {len(rez6)} (быстрый слой {sum(m.startswith('v6bystr') for m in rez6)}), "
              f"пропущено не-ok при наборе {sum(m6['v6_propusheno'].values())}; {time.time() - t0 - t5:.0f} с", flush=True)
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
