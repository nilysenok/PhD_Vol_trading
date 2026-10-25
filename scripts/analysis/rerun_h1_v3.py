#!/usr/bin/env python3
"""Дизайн h=1 на features_pit_v3 при ТЕХ ЖЕ параметрах B v2b (владелец 25.10, 20:40): без нового поиска.
Точки — из data/regen/Bv2b_*: топ-5 (seed 42, ранги 1-5) шагов 1-6 из Bv2b_{m}_h1_top5_val.parquet, seed 43/44 точки ранга 1,
шаги 7-9 — точка шага 6 (Bv2b_selected_h1.csv: только model/step/point), seed 42/43/44. Исключённые признаки — Bv2b (копия
в {tag}_excluded_features.csv). Обучение — tune_Bv2.final (все строки окна, ES patience 30 по val-году, max_bin 255, nthread 1).
Выход в формате tune_Bv2: data/regen/{tag}_{m}_h1_{val,top5_val,seeds4344_val}.parquet, тест — data/regen/test_locked/{tag}/
(только прогнозы, метрики не считаются). qlike_val шагов 7-9 в {tag}_selected_h1.csv не пишется (подтверждающий набор закрыт).
--check: те же задачи на features_pit_v2 для двух точек и сверка с Bv2b (воспроизводимость конвейера), без записи.
Запуск: python3 scripts/analysis/rerun_h1_v3.py --tag Bv3 --max-bin 255 --features <общий data/regen>/features_pit_v3.parquet --workers 7"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'moex_volatility' / 'scripts'))
import tune_Bv2 as T  # noqa: E402  (читает --tag/--max-bin/--features из sys.argv)

import argparse  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SRC = 'Bv2b'
H = 1


def tasks_from_src(models=T.MODELS):
    R = T.OUT
    sel = pd.read_csv(R / f'{SRC}_selected_h{H}.csv', usecols=['model', 'step', 'point'])
    tk = []
    for m in models:
        t5 = pd.read_parquet(R / f'{SRC}_{m}_h{H}_top5_val.parquet', columns=['step', 'point', 'rank', 'seed']).drop_duplicates()
        for step in range(1, 7):
            g = t5[t5.step == step].sort_values('rank')
            assert g['rank'].tolist() == [1, 2, 3, 4, 5] and (g.seed == 42).all()
            p1 = int(sel[(sel.model == m) & (sel.step == step)].point.iloc[0])
            assert int(g.point.iloc[0]) == p1, (m, step)
            tk += [(m, H, step, int(r.point), 42, int(r.rank)) for r in g.itertuples()] + [(m, H, step, p1, sd, 1) for sd in (43, 44)]
        p6 = int(sel[(sel.model == m) & (sel.step == 6)].point.iloc[0])
        assert (sel[(sel.model == m) & (sel.step >= 7)].point == p6).all()
        tk += [(m, H, step, p6, sd, 1) for step in (7, 8, 9) for sd in (42, 43, 44)]
    return tk


def run(P, tk):
    fin = []
    for r, pv, pte in P.imap_unordered(T.final, tk, chunksize=1):
        s = T.SD('xgb', H, r['step'])
        fin.append((r, T.frame(s, 'val', r['step'], pv, point=r['point'], seed=r['seed'], rank=r['rank']),
                    T.frame(s, 'test', r['step'], pte, point=r['point'], seed=r['seed'], rank=r['rank'])))
        print(f"{time.strftime('%T')} готово {len(fin)}/{len(tk)}: {r['model']} шаг {r['step']} точка {r['point']} seed {r['seed']} ранг {r['rank']}", flush=True)
    return fin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--max-bin'); ap.add_argument('--features')
    ap.add_argument('--workers', type=int, default=7); ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    if a.check:  # воспроизводимость конвейера на v2: XGB шаг 1 и шаг 4, ранг 1, seed 42 -> Bv2b побайтно / до 1e-12
        assert a.tag == SRC and 'features_pit_v2' in str(T.FEAT)
        tk = [t for t in tasks_from_src(('xgb',)) if t[2] in (1, 4) and t[4] == 42 and t[5] == 1]
        with Pool(min(a.workers, len(tk))) as P:
            fin = run(P, tk)
        ref = pd.read_parquet(T.OUT / f'{SRC}_xgb_h{H}_val.parquet')
        for r, fv, _ in fin:
            g = ref[ref.step == r['step']].sort_values(['date', 'ticker']).rv_pred.values
            n = fv.sort_values(['date', 'ticker']).rv_pred.values
            print(f"СВЕРКА v2 шаг {r['step']}: строк {len(n)}, max|отн. разн.| = {np.max(np.abs(n / g - 1)):.2e}, побайтно {n.tobytes() == g.tobytes()}", flush=True)
        return
    assert a.tag != SRC
    exc = T.OUT / f'{a.tag}_excluded_features.csv'
    if not exc.exists():
        shutil.copy(T.OUT / f'{SRC}_excluded_features.csv', exc)
    assert T.EXC == exc
    tk = tasks_from_src()
    t0 = time.time()
    with Pool(a.workers) as P:
        fin = run(P, tk)
    T.LOCK.mkdir(parents=True, exist_ok=True)
    sels = pd.DataFrame([r for r, *_ in fin if r['seed'] == 42 and r['rank'] == 1]).sort_values(['model', 'step'])
    sels.loc[sels.step >= 7, 'qlike_val'] = np.nan  # подтверждающий набор не открывается
    sels.join(T.GRID.reset_index().set_index(['model', 'point']), on=['model', 'point']).to_csv(T.OUT / f'{a.tag}_selected_h{H}.csv', index=False)
    for m in T.MODELS:
        pick = lambda f, ix: pd.concat([x[ix] for x in sorted(fin, key=lambda x: (x[0]['step'], x[0]['rank'], x[0]['seed'])) if f(x[0]) and x[0]['model'] == m])
        for k, ix, d in (('val', 1, T.OUT), ('test', 2, T.LOCK)):
            pick(lambda r: r['seed'] == 42 and r['rank'] == 1, ix).to_parquet(d / f'{a.tag}_{m}_h{H}_{k}.parquet', index=False)
            pick(lambda r: r['seed'] == 42 and r['step'] <= 6, ix).to_parquet(d / f'{a.tag}_{m}_h{H}_top5_{k}.parquet', index=False)
            pick(lambda r: r['seed'] != 42, ix).to_parquet(d / f'{a.tag}_{m}_h{H}_seeds4344_{k}.parquet', index=False)
    print(f'h=1 {a.tag} готов {time.strftime("%F %T")}, {time.time() - t0:.0f}s; задач {len(tk)}; признаки {T.FEAT}', flush=True)


if __name__ == '__main__':
    main()
