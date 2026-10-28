"""Переподбор гиперпараметров на одном h: optuna TPE, 30 испытаний, seed 42, пространство = SPACE tune_B.py (диапазоны B),
критерий — валидационный QLIKE шага (свой study на шаг x модель, как выбор B по шагу). Процедура поиска уровня 2 и финальной
комбинации; B (ЛГК 100) не меняется. Один study = одна задача воркера (испытания подряд, один Dataset на воркер — группировка
по (шаг, h)); отсечение: на итерации 100 val-QLIKE хуже лучшего завершённого испытания study на 20% — снять (TrialState.PRUNED).
Очередь: LightGBM раньше XGBoost, поздние шаги раньше. --protocol b_v2 — протокол B v2 (models/protocol_bv2.py)."""
import logging, time
from concurrent.futures import as_completed
from pathlib import Path
import numpy as np, pandas as pd
from . import cache as K, config as C, data as D, pipeline as PL, train as TR

log = logging.getLogger('models')
B_SEL = [Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/.claude/worktrees/agent-afcfa1385cb335411/data/regen/B_selected.csv'),
         C.MAIN / 'data/regen/B_selected.csv']
PRUNE_AT, PRUNE_X = 100, 1.2


def suggest(trial, model):
    p = {}
    for k, lo, hi, sc in C.space_B()[model]:
        if sc in ('int', 'logint'):
            p[k] = trial.suggest_int(k, int(lo), int(hi), log=sc == 'logint')
        else:
            p[k] = trial.suggest_float(k, float(lo), float(hi), log=sc == 'log')
    return p


def study(a):
    """Воркер: один study (модель, h, шаг) целиком. Детерминирован при seed 42 (испытания последовательно)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    rows, best = [], None
    for _ in range(a['n_trials']):
        tr = st.ask()
        p = suggest(tr, a['model'])
        prune = (PRUNE_AT, PRUNE_X * best) if (a['prune'] and best is not None) else None
        r = TR.run_task(dict(dk=a['dk'], fk=a['fk'], model=a['model'], h=a['h'], step=a['step'], params=p, train=a['train'], seed=42,
                             tag=f'tune_h{a["h"]}', prune=prune))
        if r.get('pruned'):
            tr.report(r['qlike_val'], PRUNE_AT)
            st.tell(tr, state=optuna.trial.TrialState.PRUNED)
        else:
            st.tell(tr, r['qlike_val'])
            best = r['qlike_val'] if best is None else min(best, r['qlike_val'])
        rows.append(dict(model=a['model'], h=a['h'], step=a['step'], trial=tr.number, **p, best_iteration=r['best_iteration'],
                         qlike_val=r['qlike_val'], pruned=bool(r.get('pruned')), key=r['key'], sec=r.get('sec_total', 0), hit=r['hit']))
    return rows


def run(h, models, n_trials, design_path, root, workers, confirm=False, smoke=False, protocol='optuna', prune=True):
    if protocol == 'b_v2':
        from . import protocol_bv2
        return protocol_bv2.run(h, workers, smoke)
    T, t0 = K.Timer(), time.time()
    des = C.load_design(design_path) if design_path else {'train': dict(C.DEFAULT_TRAIN)}
    train = des['train']
    st = C.steps(confirm, smoke)
    dk = D.stage_data(D.source(des, root), T)
    fk = D.stage_features(dk, [(h, s) for s in st], T)
    jobs = [dict(dk=dk, fk=fk[(h, s)], model=m, h=h, step=s, train=train, n_trials=n_trials, prune=prune) for m in models for s in st]
    jobs.sort(key=lambda j: (j['model'] != 'lgbm', -j['step']))  # длинные первыми
    rows = []
    with PL.pool(min(workers, len(jobs))) as P:
        for f in as_completed([P.submit(study, j) for j in jobs]):
            r = f.result()
            rows += r
            log.info('study %s h%s step%s: best q=%.4f, pruned %d/%d, %.0fs', r[0]['model'], h, r[0]['step'],
                     min(x['qlike_val'] for x in r if not x['pruned']), sum(x['pruned'] for x in r), len(r), sum(x['sec'] for x in r))
    tr_df = pd.DataFrame(rows).sort_values(['model', 'step', 'trial'])
    ok = tr_df[~tr_df.pruned]
    best = ok.loc[ok.groupby(['model', 'step']).qlike_val.idxmin()]
    out = C.CACHE / 'tune'
    out.mkdir(parents=True, exist_ok=True)
    tag = f'h{h}_{"_".join(models)}_{"smoke" if smoke else "all" if confirm else "dev"}'
    tr_df.to_csv(out / f'trials_{tag}.csv', index=False, float_format='%.17g')
    best.drop(columns=['trial', 'key', 'sec', 'hit', 'pruned']).to_csv(out / f'selected_{tag}.csv', index=False, float_format='%.17g')
    log.info('tune done %.1fs, %d trials, pruned %d', time.time() - t0, len(tr_df), int(tr_df.pruned.sum()))
    cand = [C.data_path(des['source'])] if des.get('source') else (B_SEL if not des.get('data') else [])
    bsel = next((pd.read_csv(p) for p in cand if p.exists()), None)  # B (ЛГК 100) — только справка: протокол ES другой
    res = []
    for m, g in best.groupby('model'):
        for nm, mask in (('разраб. 2016-2021', g.step <= C.DEV), ('подтв. 2022-2024', g.step > C.DEV)):
            if not mask.any():
                continue
            r = dict(h=h, модель=m, набор=nm, шагов=int(mask.sum()), qlike_optuna30=round(float(g[mask].qlike_val.mean()), 4),
                     снято=int(tr_df[(tr_df.model == m) & tr_df.step.isin(g[mask].step)].pruned.sum()))
            if bsel is not None and 'qlike_val' in bsel:
                b = bsel[(bsel.model == m) & (bsel.h == h) & bsel.step.isin(g[mask].step)]
                r['qlike_B'] = round(float(b.qlike_val.mean()), 4) if len(b) == mask.sum() else np.nan
            res.append(r)
    res = pd.DataFrame(res)
    res['сек'] = round(time.time() - t0, 1)
    return res
