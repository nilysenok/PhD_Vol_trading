#!/usr/bin/env python3
"""Детектор утечек (задание владельца 25.10): запускать на каждой версии конвейера до любого отчёта.
  (а) «Будущее не влияет»: входы после даты t заменены шумом; прогнозы моделей / позиции стратегий на датах <= t, сделанные
      шагами с окном обучения+валидации, заканчивающимся не позже t-h, обязаны совпасть с эталоном побайтно.
  (б) «Сдвинутая цель»: цель переставлена внутри тикера (seed 42); val-QLIKE разработочного набора не лучше константы.
Запуск: python3 tests/leak_detector.py --models p1,harj,gibrid --strategies v4 [--smoke] [--selfcheck]
Модели: p1 | harj | gibrid | gibrid_v7lag1 | pipeline. Стратегии: v4 (маска [0]) | v4_mask1 (старая [1]) | fastbt | none.
Код выхода: 0 — всё в порядке; 1 — упала хотя бы одна стадия (решение М8: падает только стадия, остальные проверяются);
2 — самопроверка детектора не сработала."""
import argparse, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
from leak import registry as R, selfcheck
from leak.runner import run_adapter

DATES = ['2019-06-30', '2021-12-31', '2023-06-30']  # середина 2019, 2021-12-31, середина 2023


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', default='none', help='адаптеры моделей через запятую')
    ap.add_argument('--strategies', default='none', help='адаптеры стратегий через запятую')
    ap.add_argument('--dates', default=','.join(DATES), help='три даты t через запятую')
    ap.add_argument('--h', default=None, help='горизонты моделей через запятую (smoke: 22; полный: 1,5,22)')
    ap.add_argument('--smoke', action='store_true', help='один шаг на дату, один горизонт, 1 поток')
    ap.add_argument('--no-permute', action='store_true', help='пропустить проверку (б)')
    ap.add_argument('--selfcheck', action='store_true', help='самопроверки детектора (утечка обязана ловиться)')
    ap.add_argument('--log', default=None, help='дублировать вывод в файл')
    a = ap.parse_args(argv)
    fh = open(a.log, 'w') if a.log else None

    def log(s):
        print(s, flush=True)
        if fh:
            fh.write(s + '\n'); fh.flush()

    hs = [int(x) for x in (a.h or ('22' if a.smoke else '1,5,22')).split(',')]
    dates = a.dates.split(',')
    verdicts, t0 = [], time.time()
    names = [('model', n) for n in a.models.split(',') if n != 'none'] + [('strat', n) for n in a.strategies.split(',') if n != 'none']
    for kind, n in names:
        log(f'== {kind}: {n}')
        ad = R.models(n) if kind == 'model' else R.strategies(n)
        vs = run_adapter(ad, dates, hs, a.smoke, not a.no_permute, log)
        for v in vs:
            log(v.line())
        verdicts += vs
    code = 0
    bad = [v for v in verdicts if not v.ok]
    if a.selfcheck:
        log('== самопроверки детектора')
        if not selfcheck.run(a.smoke, log):
            code = 2
    if bad:
        log('\nУПАЛИ СТАДИИ (стоп только этих стадий; остальные проверены):')
        for v in bad:
            log(f'  стадия={v.stage}  t={v.t}  проверка={v.check}  -> {v.detail}')
        code = code or 1
    log(f'\nИТОГО: {len(verdicts) - len(bad)} из {len(verdicts)} проверок пройдено, самопроверки: '
        f'{"не запускались" if not a.selfcheck else ("OK" if code != 2 else "СБОЙ")}; {time.time() - t0:.0f} с')
    return code


if __name__ == '__main__':
    sys.exit(main())
