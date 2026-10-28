#!/usr/bin/env python3
"""Самопроверка детектора на адаптерах справочных моделей (26.10): внедрённая утечка обязана ловиться.
ref_garch: калибровка шага t по году оценки шага t+1 -> (а) падает. lstm/gru: признак leak_rv = цель строки -> (а) и (б) падают.
python3 tests/leak_selfcheck_ref.py --models ref_garch,lstm,gru [--h 22] [--log файл]; код выхода 0 — все утечки пойманы."""
import argparse, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('OMP_NUM_THREADS', '1')
from leak import registry as R
from leak.runner import check_future, check_permute

T19 = '2019-06-30'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='ref_garch,lstm,gru')
    ap.add_argument('--h', type=int, default=22)
    ap.add_argument('--log', default=None)
    a = ap.parse_args()
    fh = open(a.log, 'w') if a.log else None

    def log(s):
        print(s, flush=True)
        if fh:
            fh.write(s + '\n'); fh.flush()
    ok_all = True
    for n in a.models.split(','):
        ad = R.models(n)
        checks = [('future', lambda: check_future(ad, T19, a.h, False, inject=True, log=log))]
        if n != 'ref_garch':
            checks.append(('permute', lambda: check_permute(ad, a.h, False, inject=True, log=log)))
        for kind, fn in checks:
            bad = [v for v in fn() if not v.ok]
            ok_all &= bool(bad)
            log(f'{"OK  " if bad else "BAD "} самопроверка {n} ({kind}, h={a.h}, утечка внедрена): '
                + (f'упало {len(bad)} стадий; {bad[0].stage}: {bad[0].detail[:140]}' if bad else 'НЕ упал (детектор слеп)'))
    return 0 if ok_all else 2


if __name__ == '__main__':
    sys.exit(main())
