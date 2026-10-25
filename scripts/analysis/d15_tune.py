#!/usr/bin/env python3
"""Д15 (решение владельца 25.10, 21:40): улучшенный ЧИСТЫЙ бустинг V0 (без V9 и без прогноза HAR-J в признаках) на h=5/22.
Обёртка над tune_L2_optuna.py (та же процедура: optuna TPE 30, seed 42, свой study на (модель, шаг), шаги 1-6; шаги 7-9 — точка шага 6;
seed 43/44 — выбранная точка) с двумя добавками:
  --space B   — пространство B v2 (SPACEB, как L2/D12);  --space d93 — узкое пространство Д9.3 (SPACE93: depth 2-3 / leaves 4-8,
                min_data_in_leaf 200-1000, reg_lambda 10-100 лог, feature_fraction 0,3-0,5, lr 0,01-0,05), потолок 800 деревьев;
  cfg mono_sign = -1 — обратный знак monotone_constraints на rv/rv_d/rv_w/rv_m/bv (как d13_run.prep).
Запрещённые ключи cfg (V9 / HAR-J в признаках): v9, harf, h1f, casc — скрипт падает.
L2_FEAT=data/regen/features_pit_v3.parquet python3 scripts/analysis/d15_tune.py --h 5 --space B --cfg '{}' --tag D15b_h5_base --workers 3"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
assert 'features_pit_v3' in os.environ.get('L2_FEAT', ''), 'Д15: L2_FEAT должен указывать на features_pit_v3.parquet'
import l2_lib as L
import tune_L2_optuna as T
from l2_tune import SPACE93

BANNED = ('v9', 'harf', 'h1f', 'casc', 'intra')


def resume(h, cfg, tag, workers, btag='Bv2b'):
    """Досчёт после обрыва на стадии дообучения: точки — из {tag}_trials.csv (подбор уже выполнен); выбранная точка шага
    (лучшее не снятое испытание) переобучается с seed 42/43/44 на шагах 1-6 и точкой шага 6 на шагах 7-9. Детерминированно
    (nthread=1, тот же seed) — seed 42 побитно равен прогнозу испытания. top5/allpts в этом режиме не пишутся."""
    import multiprocessing as mp, time
    import numpy as np, pandas as pd
    OUT, LOCK = L.REGEN, L.REGEN / 'test_locked' / tag
    LOCK.mkdir(parents=True, exist_ok=True)
    A, FC = L.load_base(None, tuple(x for x in cfg.get('blocks', '').split(',') if x), h)
    A2, cols, _, _ = L.build_features(A, FC, h, cfg)
    T.G.update(A=A2, cols=cols, cfg=cfg, h=h, btag=btag, n=30)
    TR = pd.read_csv(OUT / f'{tag}_trials.csv')
    sel = TR[~TR.pruned].loc[lambda d: d.groupby(['model', 'step']).qlike_val.idxmin()]
    prm = lambda r, m: {k: (int(getattr(r, k)) if sc in ('int', 'logint') else float(getattr(r, k))) for k, _, _, sc in T.SPACEB[m]}
    tasks = []
    for r in sel.itertuples():
        tasks += [(r.model, r.step, prm(r, r.model), sd, int(r.trial)) for sd in (42, 43, 44)]
        if r.step == 6:
            tasks += [(r.model, s, prm(r, r.model), sd, int(r.trial)) for s in (7, 8, 9) for sd in (42, 43, 44)]
    t0 = time.time()
    print(f'{tag}: досчёт по trials.csv, задач {len(tasks)}, воркеров {workers}', flush=True)
    with mp.get_context('fork').Pool(workers) as P:
        rf = P.map(T.refit, tasks, chunksize=1)
    S = sel.copy(); S['seed'] = 42
    add = pd.DataFrame([dict(model=r['model'], h=h, step=r['step'], best_iteration=r['best_iteration'], seed=42,
                             **sel[(sel.model == r['model']) & (sel.step == 6)].iloc[0][[k for k, *_ in T.SPACEB[r['model']]] + ['trial']].to_dict())
                        for r, _, _ in rf if r['step'] >= 7 and r['seed'] == 42])
    selall = pd.concat([S, add], ignore_index=True).sort_values(['model', 'step'])
    selall['qlike_val'] = np.where(selall.step <= 6, selall.qlike_val, np.nan)
    selall.drop(columns=['pruned', 'sec'], errors='ignore').to_csv(OUT / f'{tag}_selected_h{h}.csv', index=False, float_format='%.17g')
    for m in ('lgbm', 'xgb'):
        for k, ix, d in (('val', 1, OUT), ('test', 2, LOCK)):
            main = pd.concat([fr[ix] for fr in rf if fr[0]['model'] == m and fr[0]['seed'] == 42])
            seeds = pd.concat([fr[ix] for fr in rf if fr[0]['model'] == m and fr[0]['seed'] != 42])
            main.sort_values(['step', 'date', 'ticker']).to_parquet(d / f'{tag}_{m}_h{h}_{k}.parquet', index=False)
            seeds.to_parquet(d / f'{tag}_{m}_h{h}_seeds4344_{k}.parquet', index=False)
    V = {m: pd.read_parquet(OUT / f'{tag}_{m}_h{h}_val.parquet') for m in ('xgb', 'lgbm')}
    Sm = L.summarize({m: v[v.step <= 6] for m, v in V.items()})
    # сверка: seed 42 на шагах 1-6 = qlike_val выбранного испытания
    chk = max(abs(L.qlike(g.rv_actual.values, g.rv_pred.values) - sel[(sel.model == m) & (sel.step == st)].qlike_val.iloc[0])
              for m, v in V.items() for st, g in v[v.step <= 6].groupby('step'))
    print(f"{tag} готов {time.strftime('%F %T')}, {time.time() - t0:.0f} с (досчёт; сверка с trials {chk:.2e}): метрика (шаги 2-6) "
          f"{Sm['metric']:.4f} ({Sm['best_model']}), лучший из 4 {Sm['best4']:.4f} ({Sm['best4_mc']}); xgb ord {Sm['xgb_ord']:.4f}, lgbm ord {Sm['lgbm_ord']:.4f}", flush=True)


def main():
    argv = sys.argv[1:]
    space = 'B'
    res = '--resume' in argv
    if res:
        argv.remove('--resume')
    if '--space' in argv:
        i = argv.index('--space'); space = argv[i + 1]; del argv[i:i + 2]
    cfg = json.loads(argv[argv.index('--cfg') + 1]) if '--cfg' in argv else {}
    bad = [k for k in BANNED if cfg.get(k)]
    assert not bad, f'Д15: чистый бустинг, запрещены ключи {bad}'
    if space == 'd93':
        T.SPACEB = SPACE93
        cfg.setdefault('nmax', 800)
        argv[argv.index('--cfg') + 1] = json.dumps(cfg)
    elif space != 'B':
        raise SystemExit(f'неизвестное пространство {space}')
    orig = L.step_data

    def step_data(*a, **k):
        D = orig(*a, **k)
        c = a[5] if len(a) > 5 else k['cfg']
        if c.get('mono_sign') == -1:
            D['mono'] = [-m for m in D['mono']]
        if c.get('rel'):  # (б): у первой строки каждого тикера (17 строк 2014 г., только обучение) rv_m = NaN -> смещение = медиана ln RV_m части
            import numpy as np
            for k, o in D['off'].items():
                if np.isnan(o).any():
                    D['off'][k] = np.where(np.isnan(o), np.nanmedian(o), o)
        return D
    L.step_data = step_data
    print(f'Д15: пространство {space}', flush=True)
    if res:
        g = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d
        return resume(int(g('--h')), cfg, g('--tag'), int(g('--workers', 3)))
    sys.argv = [sys.argv[0]] + argv
    T.main()


if __name__ == '__main__':
    main()
