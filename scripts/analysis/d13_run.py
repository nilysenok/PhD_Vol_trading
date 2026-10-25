#!/usr/bin/env python3
"""Д13 (второй круг, решение владельца 25.10, 20:55): рычаги поверх текущей базы каждого h на features_pit_v3, без подбора.
Базы: h=1 — точка Bv2b (cfg {}, XGB и LGBM; финальный V0 — XGB top5, здесь — одна точка ранга 1); h=5 — V9 + Д8.12 + Д7 (v3), точка L2_h5;
h=22 — V9 + LightGBM linear_tree λ=1, точка L2_h22. Метрика — средний OOS QLIKE шагов 2-6 лучшего бустинга с поправкой Б0.1 ord
(s2 по val-году шага t-1); справочно — лучший из 4 оценок s2 и обёртка (среднее XGB и LGBM по RV). Ворота и c1-c4 — gibrid_v17.py
(нога single, oos, шаги 2-6, дизайн single|V0_{лучший}|B-ord|L0).
Дополнительные ключи cfg (кроме l2_lib): intra (Д13.7), casc (Д13.1, признак h1_ln — d13_lib.h1_cascade), mono_sign (-1 — обратный знак),
mono_har (+1 только на har_ln), multi (Д13.2, одна модель на h ∈ {1,5,22} с признаком h; точка — --params).
--final NAME: шаги 1-9 (7-9 — точка шага 6), val -> data/regen/NAME_{m}_h{h}_val.parquet, test -> data/regen/test_locked/NAME/ (не открывать).
python3 scripts/analysis/d13_run.py --h 22 --lever Д13.3 --configs '[{"name":"cs","cfg":{"v9":true,"linear_lambda":1,"cs":true}}]' [--gate] [--ref 0.2136]"""
import argparse, json, multiprocessing as mp, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d13_lib as T
import l2_lib as L

G = {}
BASE = {1: ({}, '', 'Bv2b'), 5: ({'v9': True, 'harf': True}, 'd7', 'L2_h5'), 22: ({'v9': True, 'linear_lambda': 1}, '', 'L2_h22')}
DEVLOG = Path(__file__).resolve().parents[2] / 'docs/otchety/dev_log_d13.md'


def params_path(tag, h):
    return str(L.REGEN / f'{tag}_selected_h{h}.csv')


def _casc_task(j):
    st, Y, P, keep = j
    return T.h1_cascade_task(G['A1'], G['FC1'], st, Y, P, keep)


def cascade(steps, pool):
    if T.CASC_CACHE.exists():
        c = pd.read_parquet(T.CASC_CACHE)
        if set(steps) <= set(c.step.unique()):
            return T.h1_dict(c)
    A1, FC1 = T.load(1)
    G.update(A1=A1, FC1=FC1)
    res = pool.map(_casc_task, T.h1_cascade_jobs(A1, FC1, range(1, 10)), chunksize=1)  # сразу шаги 1-9 (для --final)
    c = pd.concat(res, ignore_index=True)
    T.D13DIR.mkdir(parents=True, exist_ok=True); c.to_parquet(T.CASC_CACHE, index=False)
    return T.h1_dict(c)


def prep(D, cfg):
    if cfg.get('mono_sign') == -1:
        D['mono'] = [-m for m in D['mono']]
    if cfg.get('mono_har'):
        D['mono'] = [1 if n == 'har_ln' else 0 for n in D['names']]
    return D


def task(a):
    name, model, step = a
    A, cols, cfg = G['built'][name]
    h, btag = G['h'], G['btag']
    p = G['P'][(model, min(step, 6))]
    h1 = G.get('h1') if cfg.get('casc') else None
    c2 = {**cfg, 'h1f': True} if cfg.get('casc') else cfg
    ref = L.make_ref(model, A, cols, h, c2, btag, h1=h1) if step > 1 else None
    D = prep(L.step_data(A, cols, h, step, model, c2, btag, h1=h1), cfg)
    pv, pt, bi, gain, _ = L.fit(model, p, D, c2, ref=ref, seed=42)
    out = []
    for k, pr, yr in (('va', pv, D['vy']), ('te', pt, D['ty'])):
        d = D['parts'][k]
        out.append(pd.DataFrame({'date': d.date.values, 'ticker': d.ticker.values, 'year': yr, 'step': step, 'model': model,
                                 'rv_actual': D['y'][k], 'rv_pred': np.clip(np.exp(pr), 1e-10, None)}))
    return name, model, step, out[0], out[1], bi, dict(zip(D['names'], gain))


def task_multi(a):
    """Д13.2: строки трёх h сложены, признак h; смещение V9 — ln HAR-J своего h; ES — на сложенном val-годе шага; бины — свои (без ref)."""
    name, model, step = a
    A, cols, cfg = G['built'][name]
    p = G['P'][(model, min(step, 6))]
    Ds = {h: L.step_data(A, cols, h, step, model, cfg, G['btag']) for h in (1, 5, 22)}
    names = [n for n in Ds[5]['names'] if all(n in Ds[h]['names'] for h in (1, 22))]
    cat = lambda f: np.concatenate([f(h) for h in (1, 5, 22)])
    X = {k: cat(lambda h: np.column_stack([Ds[h]['X'][k][:, [Ds[h]['names'].index(n) for n in names]], np.full(len(Ds[h]['X'][k]), h, np.float32)]))
         for k in ('tr', 'va', 'te')}
    D = dict(X=X, y={k: cat(lambda h: Ds[h]['y'][k]) for k in ('tr', 'va', 'te')}, ytr_ln=cat(lambda h: Ds[h]['ytr_ln']),
             off={k: cat(lambda h: Ds[h]['off'][k]) for k in ('tr', 'va', 'te')}, w=None, names=names + ['h'], cats=[], mono=[0] * (len(names) + 1))
    pv, pt, bi, gain, _ = L.fit(model, p, D, cfg, ref=None, seed=42)
    res = {}
    for k, pr in (('va', pv), ('te', pt)):
        i = 0
        for h in (1, 5, 22):
            d = Ds[h]['parts'][k]; n = len(d)
            res[(h, k)] = pd.DataFrame({'date': d.date.values, 'ticker': d.ticker.values, 'year': Ds[h]['vy'] if k == 'va' else Ds[h]['ty'], 'step': step,
                                        'model': model, 'rv_actual': Ds[h]['y'][k], 'rv_pred': np.clip(np.exp(pr[i:i + n]), 1e-10, None)})
            i += n
    return name, model, step, res, bi


def gate(h, b, V, regen_dir):
    gd = T.D13DIR / 'gate'; gd.mkdir(parents=True, exist_ok=True)
    for m in ('xgb', 'lgbm'):
        V[(V.model == m) & (V.step <= 6)].drop(columns=['model'], errors='ignore').to_parquet(gd / f'{b}_{m}_h{h}_val.parquet', index=False)
    out = T.D13DIR / f'gibrid_{b}'
    dl = T.D13DIR / 'gibrid_devlog.md'
    r = subprocess.run([sys.executable, '-W', 'ignore', str(Path(__file__).resolve().parent / 'gibrid_v17.py'), '--boost', b, '--h', str(h),
                        '--regen', str(gd), '--out', str(out), '--devlog', str(dl), '--features', str(L.FEAT)], capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-2000:], flush=True); return {}
    rk = pd.read_csv(out / f'ranking_{b}_h{h}.csv')
    txt = dl.read_text().splitlines()
    lines = [x for x in txt if x.startswith(f'h={h}: ворота') or x.startswith(f'h={h}: выбор по правилу')][-2:]
    res = {'gate_lines': ' || '.join(lines), 'rule_top': rk.iloc[0].design, 'rule_top_failed': '' if pd.isna(rk.iloc[0].failed) else rk.iloc[0].failed}
    for m, tag in (('xgb', 'XGB'), ('lgbm', 'LGB')):
        row = rk[rk.design == f'single|V0_{tag}|B-ord|L0']
        if len(row):
            rr = row.iloc[0]
            res[f'crit_{m}'] = ''.join(c[1] if bool(rr[c]) else '-' for c in ('c1', 'c2', 'c3', 'c4'))
            res[f'rank_{m}'] = int(row.index[0]) + 1
    return res


def summarize_row(V6):
    S = L.summarize({m: V6[V6.model == m] for m in ('xgb', 'lgbm')})
    W = V6.groupby(['date', 'ticker', 'step'], as_index=False).agg(rv_actual=('rv_actual', 'first'), rv_pred=('rv_pred', 'mean'))
    S['wrap_ord'] = L.evaluate(W)['ord'][0]
    return S


def log_line(h, lever, name, S, ref, extra=''):
    DEVLOG.parent.mkdir(parents=True, exist_ok=True)
    if not DEVLOG.exists():
        DEVLOG.write_text('# Журнал Д13 (второй круг, 25.10)\n\nМетрика — средний OOS QLIKE шагов 2-6 лучшего бустинга, Б0.1 ord; матрица features_pit_v3 (SHA-256 8c51bfa1…dfe0bc); '
                          'подтверждающий набор и тест не открывались.\n\n| время | h | рычаг | вариант | метрика (модель) | к базе | лучший из 4 | обёртка XGB+LGBM | ворота/критерии |\n|---|---|---|---|---|---|---|---|---|\n')
    d = f'{(S["metric"] / ref - 1) * 100:+.2f}%' if ref else '—'
    with open(DEVLOG, 'a') as f:
        f.write(f"| {time.strftime('%H:%M')} | {h} | {lever} | {name} | {S['metric']:.4f} ({S['best_model']}) | {d} | {S['best4']:.4f} | {S['wrap_ord']:.4f} | {extra} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, required=True); ap.add_argument('--lever', default=''); ap.add_argument('--configs', required=True)
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--btag', default='Bv2b'); ap.add_argument('--params', default=None)
    ap.add_argument('--gate', action='store_true'); ap.add_argument('--ref', type=float, default=None, help='метрика текущей базы h')
    ap.add_argument('--gate-min', type=float, default=0.01, help='ворота — только при улучшении к --ref не меньше этого')
    ap.add_argument('--final', default=None)
    a = ap.parse_args()
    cf = json.loads(Path(a.configs).read_text()) if a.configs.endswith('.json') else json.loads(a.configs)
    h = a.h
    base_cfg, blocks, ptag = BASE[h]
    t0 = time.time()
    multi = any(c['cfg'].get('multi') for c in cf)
    A, FC = T.load(h, '' if multi else blocks)
    G.update(h=h, btag=a.btag, P=L.selected_params(a.params or params_path(ptag, h), h), built={})
    for c in cf:
        A2, cols, cats, mono = L.build_features(A, FC, h, c['cfg'])
        if c['cfg'].get('intra'):
            A2, cols = T.add_intraday(A2, cols)
        G['built'][c['name']] = (A2, cols, c['cfg'])
        print(f"{c['name']}: v3{'+' + blocks if blocks and not multi else ''}, признаков {len(cols)} (кат. {len(cats)}, монот. {len(mono)}), строк {len(A2)}", flush=True)
    steps = list(range(9, 0, -1)) if a.final else list(range(6, 0, -1))
    ctx = mp.get_context('fork')
    if any(c['cfg'].get('casc') for c in cf):
        with ctx.Pool(a.workers) as P:
            G['h1'] = cascade(sorted(steps), P)
        print(f'каскад h=1 готов ({time.time() - t0:.0f} с)', flush=True)
    tasks = [(c['name'], m, s) for c in cf for s in steps for m in ('lgbm', 'xgb')]
    with ctx.Pool(a.workers) as P:
        res = P.map(task_multi if multi else task, tasks, chunksize=1)
    od = T.D13DIR / f'h{h}'; od.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cf:
        n, cfg = c['name'], c['cfg']
        R = [r for r in res if r[0] == n]
        per_h = {hh: (pd.concat([r[3][(hh, 'va')] for r in R], ignore_index=True), pd.concat([r[3][(hh, 'te')] for r in R], ignore_index=True))
                 for hh in (1, 5, 22)} if multi else {h: (pd.concat([r[3] for r in R], ignore_index=True), pd.concat([r[4] for r in R], ignore_index=True))}
        if not multi:
            pd.DataFrame([dict(model=r[1], step=r[2], feature=k, gain=v) for r in R for k, v in r[6].items()]).to_csv(od / f'{n}_gain.csv', index=False)
        for hh, (V, Tt) in per_h.items():
            V.to_parquet(T.D13DIR / f'h{hh}' / f'{n}_val.parquet' if (T.D13DIR / f'h{hh}').exists() else od / f'{n}_h{hh}_val.parquet', index=False)
            V6 = V[V.step <= 6]
            S = summarize_row(V6)
            ref = a.ref if hh == h else None
            row = dict(time=time.strftime('%F %T'), h=hh, lever=a.lever, name=n, cfg=json.dumps(cfg, ensure_ascii=False), n_features=len(G['built'][n][1]),
                       metric=S['metric'], best_model=S['best_model'], best4=S['best4'], best4_mc=S['best4_mc'], xgb_ord=S['xgb_ord'], lgbm_ord=S['lgbm_ord'],
                       wrap_ord=S['wrap_ord'], rel_ref=(S['metric'] / ref - 1) if ref else np.nan,
                       per_step=json.dumps({m: [round(x, 5) for x in q] for m, q in S['per_step'].items()}),
                       bi=json.dumps({f'{r[1]}_s{r[2]}': int(r[5] if not multi else r[4]) for r in R if r[2] <= 6}))
            print(f"h={hh} {n}: метрика {S['metric']:.4f} ({S['best_model']}), лучший из 4 {S['best4']:.4f} ({S['best4_mc']}); xgb ord {S['xgb_ord']:.4f}, "
                  f"lgbm ord {S['lgbm_ord']:.4f}, обёртка {S['wrap_ord']:.4f}" + (f"; к базе {row['rel_ref'] * 100:+.2f}%" if ref else ''), flush=True)
            extra = ''
            if a.gate and (ref is None or S['metric'] <= ref * (1 - a.gate_min) or multi):
                row.update(gate(hh, f'D13_h{hh}_{n}', V6, A))
                extra = f"c1-c4 xgb {row.get('crit_xgb')} lgbm {row.get('crit_lgbm')}; правило: {row.get('rule_top')} ({row.get('rule_top_failed') or 'все'})"
                print(f"  ворота: {row.get('gate_lines')}; {extra}", flush=True)
            elif a.gate:
                extra = 'ворота не считались (< 1%)'
            if not a.final:
                log_line(hh, a.lever, n, S, ref, extra)
            rows.append(row)
            if a.final:
                lock = L.REGEN / 'test_locked' / a.final; lock.mkdir(parents=True, exist_ok=True)
                for m in ('xgb', 'lgbm'):
                    V[V.model == m].drop(columns='model').sort_values(['step', 'date', 'ticker']).to_parquet(L.REGEN / f'{a.final}_{m}_h{hh}_val.parquet', index=False)
                    Tt[Tt.model == m].drop(columns='model').sort_values(['step', 'date', 'ticker']).to_parquet(lock / f'{a.final}_{m}_h{hh}_test.parquet', index=False)
                print(f'  выходы: data/regen/{a.final}_{{xgb,lgbm}}_h{hh}_val.parquet (шаги 1-9), test -> test_locked/{a.final}/ (не открывался)', flush=True)
    out = T.D13DIR / f'levers_h{h}.csv'
    df = pd.DataFrame(rows)
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f'готово за {time.time() - t0:.0f} с', flush=True)


if __name__ == '__main__':
    main()
