#!/usr/bin/env python3
"""Уровень 2: оценка рычагов на точке B (параметры {tag}_selected_h{h}.csv по шагам; без подбора), XGB и LGBM на шагах 1-6 (только val).
Метрика — лучший бустинг с поправкой Б0.1 ord вне выборки на шагах 2-6 (gamma — без поправки), справочно — лучший из четырёх оценок s2.
python3 scripts/analysis/l2_run.py --h 22 --tag Bv2b --lever Д2 --configs '[{"name":"hl1","cfg":{"hl":1}}, ...]' [--workers 10]
Выход: data/regen/L2/h{h}/{name}_val.parquet, {name}_gain.csv; строка на конфигурацию — data/regen/L2/levers_h{h}.csv."""
import argparse, json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L

G = {}  # наследуется воркерами через fork


def h1_oof(A, FC, tag, step):
    """Д8.6: ln-прогноз h=1 (XGB, точка {tag} h=1 шага, число деревьев = best_iteration финальной модели) — OOF по годам на обучающем окне шага;
    val/test — модель на всём окне. Только данные окна шага (<= конец обучающего окна) -> признак строк val/test не видит их целей."""
    import xgboost as xgb
    P = L.selected_params(tag, 1)[('xgb', step)]
    tp, vi, td, _, _ = L.split(A, 1, step)
    keep = [c for c in FC if c not in (L.excluded('xgb', 1, tag) if step > 1 else set())]
    X = lambda d: L.clean(d[keep].values).astype(np.float32)
    y = np.log(tp['rv_target_h1'].values + 1e-10)
    prm = dict(tree_method='hist', max_bin=L.MAX_BIN, max_depth=int(P['max_depth']), eta=P['learning_rate'], subsample=P['subsample'],
               colsample_bytree=P['colsample_bytree'], min_child_weight=P['min_child_weight'], alpha=P['reg_alpha'], reg_lambda=P['reg_lambda'], nthread=1, seed=42, verbosity=0)
    n = int(P['best_iteration']) + 1
    out = []
    oof = np.zeros(len(tp))
    for yr in sorted(tp.year.unique()):
        m = (tp.year == yr).values
        b = xgb.train(prm, xgb.DMatrix(X(tp[~m]), y[~m]), n)
        oof[m] = b.predict(xgb.DMatrix(X(tp[m])))
    out.append(pd.DataFrame({'date': tp.date.values, 'ticker': tp.ticker.values, 'h1_ln': oof}))
    b = xgb.train(prm, xgb.DMatrix(X(tp), y), n)
    for d in (vi, td):
        out.append(pd.DataFrame({'date': d.date.values, 'ticker': d.ticker.values, 'h1_ln': b.predict(xgb.DMatrix(X(d)))}))
    # строки обучающего окна h=5/22 без цели h=1 (последний день) — пусто; дубликатов нет: окно/val/test не пересекаются
    return pd.concat(out, ignore_index=True)


def _h1_task(step):
    return step, h1_oof(G['A0'], G['FC'], G['h1tag'], step)


def task(a):
    name, model, step = a
    A, cols, cfg = G['built'][name]
    h, tag = G['h'], G['tag']
    p = G['P'][(model, step)]
    h1 = G.get('h1')
    ref = L.make_ref(model, A, cols, h, cfg, tag, h1=h1) if step > 1 else None
    D = L.step_data(A, cols, h, step, model, cfg, tag, h1=h1)
    pv, pt, bi, gain, _ = L.fit(model, p, D, cfg, ref=ref)
    if cfg.get('backfit') and cfg.get('v9'):  # Д11.2: одна итерация обратной подгонки — HAR-J на ln y − f(x), бустинг на новых остатках
        _, ptr, _, _, _ = L.fit(model, p, {**D, 'X': {**D['X'], 'te': D['X']['tr']}, 'off': {**D['off'], 'te': D['off']['tr']}}, cfg, ref=ref)
        f_tr = ptr - D['off']['tr']
        tpf = D['parts']['tr']; hc = L.harj_cols(A)
        wh = L.rq_weights(tpf) if cfg.get('rqw') else None
        hl2 = L.har_ln(tpf, [D['parts'][k] for k in ('tr', 'va', 'te')], h, hc, wh, bool(cfg.get('huber')), y=np.log(D['y']['tr'] + 1e-10) - f_tr)
        D = {**D, 'off': dict(zip(('tr', 'va', 'te'), hl2))}
        pv, pt, bi, gain, _ = L.fit(model, p, D, cfg, ref=ref)
    alpha = 1.0
    if cfg.get('alpha') and cfg.get('v9'):  # Д11.3: ŷ = HAR + α·f(x), α по val-году шага
        fv = pv - D['off']['va']
        alpha = min((0.5, 0.75, 1.0), key=lambda a_: L.qlike(D['y']['va'], np.exp(D['off']['va'] + a_ * fv)))
        pv = D['off']['va'] + alpha * fv
    vi = D['parts']['va']
    fr = pd.DataFrame({'date': vi.date.values, 'ticker': vi.ticker.values, 'year': D['vy'], 'step': step, 'model': model,
                       'rv_actual': D['y']['va'], 'rv_pred': np.clip(np.exp(pv), 1e-10, None)})
    nxt = None
    if cfg.get('avg2') and step < 6:  # Д8.4: та же модель шага s -> val-год шага s+1
        D2 = L.step_data(A, cols, h, step + 1, model, cfg, tag, h1=h1, ex_step=step)  # столбцы — как у модели шага s
        assert D2['names'] == D['names']
        _, pn, _, _, _ = L.fit(model, p, {**D, 'X': {**D['X'], 'te': D2['X']['va']}, 'off': {**D['off'], 'te': D2['off']['va']}}, cfg, ref=ref)
        v2 = D2['parts']['va']
        nxt = pd.DataFrame({'date': v2.date.values, 'ticker': v2.ticker.values, 'step': step + 1, 'model': model, 'rv_prev': np.clip(np.exp(pn), 1e-10, None)})
    return name, model, step, fr, dict(zip(D['names'], gain)), (bi, alpha), nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, required=True); ap.add_argument('--tag', default='Bv2b'); ap.add_argument('--lever', default='')
    ap.add_argument('--configs', required=True); ap.add_argument('--workers', type=int, default=10); ap.add_argument('--h1tag', default=None)
    ap.add_argument('--matrix', default=None, help='альтернативная матрица признаков')
    ap.add_argument('--blocks', default='', help='d7,d10 — блоки признаков (PR #27, #28)')
    ap.add_argument('--params', default=None, help='CSV выбранных точек вместо {tag}_selected_h{h}.csv (точка Д9.3)')
    a = ap.parse_args()
    cf = json.loads(Path(a.configs).read_text()) if a.configs.endswith('.json') else json.loads(a.configs)
    t0 = time.time()
    A, FC = L.load_base(a.matrix, tuple(x for x in a.blocks.split(',') if x), a.h)
    G.update(h=a.h, tag=a.tag, P=L.selected_params(a.params or a.tag, a.h), A0=A, FC=FC, h1tag=a.h1tag or a.tag, built={})
    for c in cf:
        A2, cols, cats, mono = L.build_features(A, FC, a.h, c['cfg'])
        G['built'][c['name']] = (A2, cols, c['cfg'])
        print(f"{c['name']}: признаков {len(cols)} (кат. {len(cats)}, монот. {len(mono)})", flush=True)
    ctx = mp.get_context('fork')
    with ctx.Pool(a.workers) as P:
        if any(c['cfg'].get('h1f') for c in cf):
            G['h1'] = dict(P.map(_h1_task, range(1, 7), chunksize=1))
    with ctx.Pool(a.workers) as P:
        tasks = [(c['name'], m, s) for c in cf for s in range(6, 0, -1) for m in ('lgbm', 'xgb')]
        res = P.map(task, tasks, chunksize=1)
    od = L.L2DIR / f'h{a.h}'; od.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cf:
        n, cfg = c['name'], c['cfg']
        R = [r for r in res if r[0] == n]
        V = pd.concat([r[3] for r in R], ignore_index=True)
        if cfg.get('avg2'):
            N = pd.concat([r[6] for r in R if r[6] is not None], ignore_index=True)
            V = V.merge(N, on=['date', 'ticker', 'step', 'model'], how='left')
            V['rv_pred'] = np.where(V.rv_prev.notna(), (V.rv_pred + V.rv_prev) / 2, V.rv_pred)
            V = V.drop(columns='rv_prev')
        V.to_parquet(od / f'{n}_val.parquet', index=False)
        pd.DataFrame([dict(model=r[1], step=r[2], feature=k, gain=v) for r in R for k, v in r[4].items()]).to_csv(od / f'{n}_gain.csv', index=False)
        S = L.summarize({m: V[V.model == m] for m in ('xgb', 'lgbm')}, gam=cfg.get('obj') == 'gamma')
        bi = {f'bi_{r[1]}_s{r[2]}': r[5][0] for r in R}
        bi.update({f'alpha_{r[1]}_s{r[2]}': r[5][1] for r in R})
        row = dict(time=time.strftime('%F %T'), matrix=(Path(a.matrix).name if a.matrix else '') + a.blocks, h=a.h, tag=a.tag if not a.params else f'{a.tag}+{Path(a.params).name}', lever=a.lever, name=n, cfg=json.dumps(cfg, ensure_ascii=False),
                   n_features=len(G['built'][n][1]), **{k: v for k, v in S.items() if k != 'per_step'},
                   per_step=json.dumps({m: [round(x, 5) for x in q] for m, q in S['per_step'].items()}), **bi)
        rows.append(row)
        print(f"{n}: метрика {S['metric']:.4f} ({S['best_model']}), лучший из 4: {S['best4']:.4f} ({S['best4_mc']}); xgb ord {S['xgb_ord']:.4f}, lgbm ord {S['lgbm_ord']:.4f}, "
              f"xgb raw {S['xgb_raw']:.4f}, lgbm raw {S['lgbm_raw']:.4f}", flush=True)
    out = L.L2DIR / f'levers_h{a.h}.csv'
    df = pd.DataFrame(rows)
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f'готово за {time.time() - t0:.0f} с', flush=True)


if __name__ == '__main__':
    main()
