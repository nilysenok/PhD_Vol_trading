#!/usr/bin/env python3
"""Д17 (агент F): детектор утечек для прошедших кандидатов (по образцу tests/leak_detector.py).

(а) «Будущее не влияет»: во входной матрице кандидата (features_pit_v3 + RRV) все числовые колонки строк с датой > t
    (признаки, RV, RRV-признаки, цели) умножаются на лог-нормальный шум; итерированный HAR пересчитывается с нуля по
    зашумлённой матрице. Шаги, у которых окно обучения+валидации кончается не позже t − h (val-год < год t), обучаются
    заново тем же кодом (d17_f.task, режим final, seed 42); прогнозы val и теста на датах <= t обязаны совпасть побайтно.
    Проверяются шаг 1 и последний допустимый шаг для каждой t.
(б) «Сдвинутая цель»: цель переставлена внутри тикера (seed 42) на всей матрице; val-QLIKE шагов 2–3 не лучше
    константы (среднее переставленной цели обучающего окна).
Самопроверка: подложенный признак «RV следующей строки тикера» (вычисляется после зашумления) обязан поймать (а).
python3 tests/test_d17_f_leak.py --h 5 --design V0_h5 --variant rrv_gk+iterhar
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/analysis'))
import d17_f as F  # noqa: E402

L = F.L
DATES = ('2019-06-30', '2021-12-31')
KEYS = ('date', 'ticker', 'year')


def prepare(design, variant, leak=False):
    F.setup([design], [variant])
    if leak:
        A = F.G['A'][design]
        A['leak_next_rv'] = A.sort_values(['ticker', 'date']).groupby('ticker')['rv'].shift(-1).reindex(A.index)
        F.G['cols'][(design, variant)] = F.G['cols'][(design, variant)] + ['leak_next_rv']


def noisy(design, variant, t, leak):
    A = F.G['A'][design].copy()
    m = (pd.to_datetime(A['date']) > pd.Timestamp(t)).values
    rng = np.random.default_rng(7)
    num = [c for c in A.columns if c not in KEYS and pd.api.types.is_float_dtype(A[c])]
    for c in num:
        A.loc[m, c] = A.loc[m, c].values * rng.lognormal(0, 0.5, m.sum())
    if leak:  # подложенная утечка считается по зашумлённым данным
        A['leak_next_rv'] = A.sort_values(['ticker', 'date']).groupby('ticker')['rv'].shift(-1).reindex(A.index)
    return A


def run_steps(design, variant, h, steps):
    if 'iterhar' in variant.split('+'):
        F.G['ih'] = F.iterhar(F.G['A'][design], h)
    F._REF.clear()
    F.G['final'] = True
    out = {}
    for m in F.DESIGNS[design]['models']:
        for st in steps:
            out[(m, st)] = F.task((design, variant, m, st, (42,)))
    return out


def same_before(a, b, t):
    bad = 0
    for k in a:
        for x, y in zip(a[k], b[k]):
            mx, my = x[pd.to_datetime(x.date) <= t], y[pd.to_datetime(y.date) <= t]
            bad += int(len(mx) != len(my) or mx.rv_pred.to_numpy().tobytes() != my.rv_pred.to_numpy().tobytes())
    return bad


def check_a(design, variant, h, leak=False):
    prepare(design, variant, leak)
    base_A = F.G['A'][design]
    res = []
    for t in DATES:
        ty = pd.Timestamp(t).year
        ok_steps = [s for s in range(1, 10) if L.TEST_YEARS[s - 1] - 1 < ty]  # val-год < год t
        steps = sorted({1, max(ok_steps)})
        F.G['A'][design] = base_A
        ref = run_steps(design, variant, h, steps)
        F.G['A'][design] = noisy(design, variant, t, leak)
        got = run_steps(design, variant, h, steps)
        F.G['A'][design] = base_A
        res.append((t, steps, same_before(ref, got, pd.Timestamp(t))))
    return res


def check_b(design, variant, h):
    prepare(design, variant)
    A = F.G['A'][design]
    tc = L.tcol(h)
    rng = np.random.default_rng(42)
    A[tc] = A.groupby('ticker')[tc].transform(lambda s: pd.Series(rng.permutation(s.values), index=s.index))
    if 'iterhar' in variant.split('+'):
        F.G['ih'] = F.iterhar(A, h)
    F._REF.clear()
    F.G['final'] = False
    out = []
    for m in F.DESIGNS[design]['models']:
        for st in (2, 3):
            v = F.task((design, variant, m, st, (42,)))
            tp = L.split(A, h, st)[0]
            const = float(tp[tc].mean())
            y = v.rv_actual.values
            out.append((m, st, L.qlike(y, v.rv_pred.values), L.qlike(y, np.full(len(y), const))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, required=True)
    ap.add_argument('--design', required=True)
    ap.add_argument('--variant', required=True)
    a = ap.parse_args()
    fail = False
    print(f'детектор Д17 F: h={a.h} {a.design} {a.variant}', flush=True)
    for t, steps, bad in check_a(a.design, a.variant, a.h):
        print(f'(а) t={t}: шаги {steps}, расхождений прогнозов до t: {bad}', flush=True)
        fail |= bad > 0
    sc = check_a(a.design, a.variant, a.h, leak=True)
    caught = any(b > 0 for _, _, b in sc)
    print(f'самопроверка (признак RV(t+1)): поймана = {caught} {[b for *_, b in sc]}', flush=True)
    fail |= not caught
    for m, st, q, qc in check_b(a.design, a.variant, a.h):
        ok = q >= qc * 0.999
        print(f'(б) {m} шаг {st}: QLIKE {q:.4f}, константа {qc:.4f} -> {"OK" if ok else "ЛУЧШЕ КОНСТАНТЫ"}', flush=True)
        fail |= not ok
    print('ДЕТЕКТОР:', 'FAIL' if fail else 'PASS', flush=True)
    sys.exit(1 if fail else 0)


if __name__ == '__main__':
    main()
