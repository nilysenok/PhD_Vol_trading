#!/usr/bin/env python3
"""Разбор падения (б) «перемешанная цель» у справочных NN (26.10): та же проверка детектора при других seed перестановки и по тикерам.
Если модель обгоняет константу лишь при части seed и за счёт одного тикера — это шум оценки константы на тяжёлых хвостах, а не утечка.
python3 tests/leak_permute_seeds_ref.py --model gru --h 1 --seeds 42 1 2 3 4 [--log файл]"""
import argparse, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from leak import registry as R, noise as N
from leak.compare import compare_permute, qlike
from leak.models_ref import tcol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gru'); ap.add_argument('--h', type=int, default=1)
    ap.add_argument('--seeds', type=int, nargs='*', default=[42, 1, 2, 3, 4]); ap.add_argument('--log', default=None)
    a = ap.parse_args()
    fh = open(a.log, 'w') if a.log else None

    def log(s):
        print(s, flush=True)
        if fh:
            fh.write(s + '\n'); fh.flush()
    ad = R.models(a.model)
    base = ad.load()
    col = f'rv_target_h{a.h}'
    for sd in a.seeds:
        w = N.permute_within_ticker(base, [tcol(a.h)], seed=sd)
        res = ad.run(w, a.h, ad.permute_steps(False))
        tf = ad.target_frame(w, a.h)
        v = compare_permute(res, tf, a.h)[0]
        per = []
        g = res[(res.kind == 'val') & (res.ty - 1 <= 2021)]
        for tk, gt in g.groupby('ticker'):
            qm, qc = [], []
            for ty, gg in gt.groupby('ty'):
                c = tf[(tf.year <= ty - 2) & (tf.ticker == tk)][col].mean()
                ok = gg.rv_actual.values > 1e-12
                qm.append(qlike(gg.rv_actual.values[ok], gg.rv_pred.values[ok])); qc.append(qlike(gg.rv_actual.values[ok], np.full(ok.sum(), c)))
            per.append(f'{tk} {np.mean(qm):.3f}/{np.mean(qc):.3f}')
        log(f'seed {sd}: {"PASS" if v.ok else "FAIL"} {v.detail.split(";")[0]} | по тикерам модель/константа: ' + ', '.join(per))
    return 0


if __name__ == '__main__':
    sys.exit(main())
