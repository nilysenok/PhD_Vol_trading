"""Переподбор гиперпараметров на одном h: optuna TPE, 30 испытаний, seed 42, пространство = SPACE tune_B.py (диапазоны B),
критерий — валидационный QLIKE шага (свой study на каждый шаг x модель, как выбор B по шагу). Процедура поиска уровня 2 и финальной
комбинации; B (ЛГК 100) не меняется. Study идут параллельно (по одному испытанию на study за раз — детерминированно при seed 42),
каждое испытание — обычное обучение конвейера (кэш, прогнозы val/test сохраняются)."""
import logging, time
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path
import numpy as np, pandas as pd
from . import cache as K, config as C, data as D, pipeline as PL, train as TR

log = logging.getLogger('models')
B_SEL = [Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/.claude/worktrees/agent-afcfa1385cb335411/data/regen/B_selected.csv'),
         C.ROOT / 'data/regen/B_selected.csv']


def suggest(trial, model):
    p = {}
    for k, lo, hi, sc in C.space_B()[model]:
        if sc in ('int', 'logint'):
            p[k] = trial.suggest_int(k, int(lo), int(hi), log=sc == 'logint')
        else:
            p[k] = trial.suggest_float(k, float(lo), float(hi), log=sc == 'log')
    return p


def run(h, models, n_trials, design_path, root, workers, confirm=False, smoke=False):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    T, t0 = K.Timer(), time.time()
    des = C.load_design(design_path) if design_path else {'train': dict(C.DEFAULT_TRAIN)}
    train = des['train']
    st = C.steps(confirm, smoke)
    src = D.source(des, root)
    dk = D.stage_data(src, T)
    fk = D.stage_features(dk, [(h, s) for s in st], T)
    studies = {(m, s): optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42)) for m in models for s in st}
    rows, fut = [], {}

    def submit(P, m, s):
        tr = studies[(m, s)].ask()
        p = suggest(tr, m)
        t = dict(dk=dk, fk=fk[(h, s)], model=m, h=h, step=s, params=p, train=train, seed=42, tag=f'tune_h{h}')
        fut[P.submit(TR.run_task, t)] = (m, s, tr, p)

    with PL.pool(workers) as P:
        for (m, s) in sorted(studies, key=lambda x: -x[1]):
            submit(P, m, s)
        pending = set(fut)
        while pending:
            dn, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in dn:
                m, s, tr, p = fut.pop(f)
                r = f.result()
                studies[(m, s)].tell(tr, r['qlike_val'])
                rows.append(dict(model=m, h=h, step=s, trial=tr.number, **p, best_iteration=r['best_iteration'], qlike_val=r['qlike_val'],
                                 key=r['key'], sec=r.get('sec_total', 0), hit=r['hit']))
                log.info('tune %s h%s step%s trial %s q=%.4f bi=%s hit=%s', m, h, s, tr.number, r['qlike_val'], r['best_iteration'], r['hit'])
                if len(studies[(m, s)].trials) < n_trials:
                    submit(P, m, s)
                    pending |= {x for x in fut if x not in pending}
    tr_df = pd.DataFrame(rows).sort_values(['model', 'step', 'trial'])
    best = tr_df.loc[tr_df.groupby(['model', 'step']).qlike_val.idxmin()]
    out = C.CACHE / 'tune'
    out.mkdir(parents=True, exist_ok=True)
    tag = f'h{h}_{"_".join(models)}_{"smoke" if smoke else "all" if confirm else "dev"}'
    tr_df.to_csv(out / f'trials_{tag}.csv', index=False, float_format='%.17g')
    best.drop(columns=['trial', 'key', 'sec', 'hit']).to_csv(out / f'selected_{tag}.csv', index=False, float_format='%.17g')  # формат B_selected -> mkdesign
    log.info('tune done %.1fs, %d trials', time.time() - t0, len(tr_df))
    cand = [C.ROOT / des['source'] if not str(des['source']).startswith('/') else Path(des['source'])] if des.get('source') else (B_SEL if not des.get('data') else [])
    bsel = next((pd.read_csv(p) for p in cand if p.exists()), None)  # B (ЛГК 100) для сравнения — только справка, протокол ES другой
    res = []
    for m, g in best.groupby('model'):
        for nm, mask in (('разраб. 2016-2021', g.step <= C.DEV), ('подтв. 2022-2024', g.step > C.DEV)):
            if not mask.any():
                continue
            r = dict(h=h, модель=m, набор=nm, шагов=int(mask.sum()), qlike_optuna30=round(float(g[mask].qlike_val.mean()), 4))
            if bsel is not None:
                b = bsel[(bsel.model == m) & (bsel.h == h) & bsel.step.isin(g[mask].step)]
                r['qlike_B_LHS100'] = round(float(b.qlike_val.mean()), 4) if len(b) == mask.sum() else np.nan
            res.append(r)
    res = pd.DataFrame(res)
    res['сек'] = round(time.time() - t0, 1)
    return res
