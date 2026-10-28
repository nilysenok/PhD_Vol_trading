#!/usr/bin/env python3
"""Второй круг Д12 (h=5, h=22): рычаги на точке optuna-30 уровня 2 (L2_h{h}_selected_h{h}.csv), без подбора.
Матрица — L2_FEAT (по умолчанию features_pit_v2; в зачёт — features_pit_v3, as-of), блок Д7 для h=5 — {FEAT.stem}_d7_h5.parquet.
Каждая конфигурация: XGB и LGBM на шагах 1-6 (ранняя остановка на val-годе шага, patience 30), метрика — средний OOS QLIKE шагов 2-6
лучшего бустинга с поправкой Б0.1 ord (s2 по val-году шага t-1); справочно — лучший из четырёх оценок s2.
Д12.2 (усадка ŷ = HAR + α·f) — постобработка: в val-кадре хранится ln HAR-J (смещение V9), α ∈ {0,25; 0,5; 0,75; 1} фиксированно на h,
плюс справочно α_t по val-году шага t-1 (вне выборки). Ворота и критерии c1-c4 — гибрид gibrid_v17.py (oos, шаги 2-6, эталон — та же s2),
дизайн рычага — single|V0_{лучший бустинг}|B-ord|L0.
--final: шаги 1-9 (7-9 — точка шага 6), val -> data/regen/{out}_{m}_h{h}_val.parquet, test -> data/regen/test_locked/{out}/ (не открывать);
val шагов 7-9 (подтверждающий набор) не оценивается и не печатается.
python3 scripts/analysis/d12_run.py --h 22 --lever Д12.3 --configs '[{"name":"d123","cfg":{"v9":true,"linear_lambda":1,"har_oof":true}}]' [--gate]"""
import argparse, json, multiprocessing as mp, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L

G = {}
ALPHAS = (0.25, 0.5, 0.75, 1.0)
D12DIR = L.REGEN / 'D12'


def task(a):
    name, model, step = a
    A, cols, cfg = G['built'][name]
    h, btag = G['h'], G['btag']
    p = G['P'][(model, min(step, 6))]
    ref = L.make_ref(model, A, cols, h, cfg, btag) if step > 1 else None
    D = L.step_data(A, cols, h, step, model, cfg, btag)
    pv, pt, bi, gain, _ = L.fit(model, p, D, cfg, ref=ref, seed=42)
    out = []
    for k, pr, yr in (('va', pv, D['vy']), ('te', pt, D['ty'])):
        d = D['parts'][k]
        out.append(pd.DataFrame({'date': d.date.values, 'ticker': d.ticker.values, 'year': yr, 'step': step, 'model': model,
                                 'rv_actual': D['y'][k], 'rv_pred': np.clip(np.exp(pr), 1e-10, None), 'har_ln': D['off'][k]}))
    return name, model, step, out[0], out[1], bi


def shrink(V, alpha):
    """ŷ = exp(ln HAR + α·(ln ŷ − ln HAR)); alpha — число или {step: α}."""
    a = V.step.map(alpha).values if isinstance(alpha, dict) else alpha
    lh = V.har_ln.values
    return V.assign(rv_pred=np.clip(np.exp(lh + a * (np.log(V.rv_pred.values) - lh)), 1e-10, None))


def alpha_wf(V):
    """α_t по val-году шага t-1 (QLIKE с поправкой exp(s2/2), s2 по остаткам того же года t-1 — всё до шага t); шаг 1-2 -> по шагу 1."""
    res = {}
    for st in sorted(V.step.unique()):
        g = V[(V.step == max(1, st - 1)) & (V.rv_actual > 1e-12)]
        def q(a):
            s = shrink(g, a); e = np.log(s.rv_actual.values + 1e-10) - np.log(s.rv_pred.values)
            return L.qlike(s.rv_actual.values, s.rv_pred.values * np.exp(np.var(e) / 2))
        res[int(st)] = min(ALPHAS, key=q)
    return res


def gate(h, name, V, feat):
    """Гибрид на val шагов 1-6 (нога single): ворота + c1-c4 дизайна рычага."""
    b = f'D12_{name}'
    gd = D12DIR / 'gate'; gd.mkdir(parents=True, exist_ok=True)
    for m in ('xgb', 'lgbm'):
        V[(V.model == m) & (V.step <= 6)].drop(columns=['model', 'har_ln'], errors='ignore').to_parquet(gd / f'{b}_{m}_h{h}_val.parquet', index=False)
    out = D12DIR / f'gibrid_{b}'
    dl = D12DIR / 'gibrid_devlog.md'
    r = subprocess.run([sys.executable, '-W', 'ignore', str(Path(__file__).resolve().parent / 'gibrid_v17.py'), '--boost', b, '--h', str(h),
                        '--regen', str(gd), '--out', str(out), '--devlog', str(dl), '--features', str(feat)], capture_output=True, text=True)
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
            res[f'failed_{m}'] = '' if pd.isna(rr.failed) else rr.failed
            res[f'rank_{m}'] = int(row.index[0]) + 1
            res[f'q_{m}'] = float(rr.mean_q)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, required=True); ap.add_argument('--lever', default=''); ap.add_argument('--configs', required=True)
    ap.add_argument('--workers', type=int, default=6); ap.add_argument('--btag', default='Bv2b')
    ap.add_argument('--params', default=None, help='CSV точек (по умолчанию data/regen/L2_h{h}_selected_h{h}.csv)')
    ap.add_argument('--blocks', default=None, help='по умолчанию d7 для h=5, пусто для h=22')
    ap.add_argument('--gate-features', default=None, help='матрица HAR-J для гибрида (Д12.1: features_pit_v3 — 17 тикеров)')
    ap.add_argument('--gate', action='store_true'); ap.add_argument('--alpha', action='store_true', help='Д12.2: таблица α')
    ap.add_argument('--final', default=None, help='имя выхода D12_h{h}_<...> — шаги 1-9, val и test (test_locked)')
    a = ap.parse_args()
    cf = json.loads(Path(a.configs).read_text()) if a.configs.endswith('.json') else json.loads(a.configs)
    h = a.h
    blocks = a.blocks if a.blocks is not None else ('d7' if h == 5 else '')
    t0 = time.time()
    A, FC = L.load_base(None, tuple(x for x in blocks.split(',') if x), h)
    G.update(h=h, btag=a.btag, P=L.selected_params(a.params or str(L.REGEN / f'L2_h{h}_selected_h{h}.csv'), h), built={})
    for c in cf:
        A2, cols, _, _ = L.build_features(A, FC, h, c['cfg'])
        G['built'][c['name']] = (A2, cols, c['cfg'])
        print(f"{c['name']}: матрица {L.FEAT.name}{'+' + blocks if blocks else ''}, признаков {len(cols)}, строк {len(A2)}", flush=True)
    steps = range(9, 0, -1) if a.final else range(6, 0, -1)
    with mp.get_context('fork').Pool(a.workers) as P:
        res = P.map(task, [(c['name'], m, s) for c in cf for s in steps for m in ('lgbm', 'xgb')], chunksize=1)
    od = D12DIR / f'h{h}'; od.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cf:
        n, cfg = c['name'], c['cfg']
        R = [r for r in res if r[0] == n]
        V = pd.concat([r[3] for r in R], ignore_index=True)
        V.to_parquet(od / f'{n}_val.parquet', index=False)
        V6 = V[V.step <= 6]
        S = L.summarize({m: V6[V6.model == m] for m in ('xgb', 'lgbm')})
        row = dict(time=time.strftime('%F %T'), matrix=L.FEAT.name + ('+' + blocks if blocks else ''), h=h, lever=a.lever, name=n,
                   cfg=json.dumps(cfg, ensure_ascii=False), metric=S['metric'], best_model=S['best_model'], best4=S['best4'], best4_mc=S['best4_mc'],
                   xgb_ord=S['xgb_ord'], lgbm_ord=S['lgbm_ord'], per_step=json.dumps({m: [round(x, 5) for x in q] for m, q in S['per_step'].items()}),
                   bi=json.dumps({f'{r[1]}_s{r[2]}': int(r[5]) for r in R if r[2] <= 6}))
        print(f"{n}: метрика {S['metric']:.4f} ({S['best_model']}), лучший из 4 {S['best4']:.4f} ({S['best4_mc']}); xgb ord {S['xgb_ord']:.4f}, "
              f"lgbm ord {S['lgbm_ord']:.4f}", flush=True)
        if a.gate:
            row.update(gate(h, f'h{h}_{n}', V6, a.gate_features or L.FEAT)); print(f"  ворота: {row.get('gate_lines')}; c1-c4 xgb {row.get('crit_xgb')} lgbm {row.get('crit_lgbm')}", flush=True)
        rows.append(row)
        if a.alpha and cfg.get('v9'):
            for al in ALPHAS:
                Sa = L.summarize({m: shrink(V6[V6.model == m], al) for m in ('xgb', 'lgbm')})
                ra = dict(row, name=f'{n}_a{al}', lever='Д12.2', cfg=json.dumps({**cfg, 'alpha': al}), metric=Sa['metric'], best_model=Sa['best_model'],
                          best4=Sa['best4'], best4_mc=Sa['best4_mc'], xgb_ord=Sa['xgb_ord'], lgbm_ord=Sa['lgbm_ord'],
                          per_step=json.dumps({m: [round(x, 5) for x in q] for m, q in Sa['per_step'].items()}))
                for k in [k for k in ra if k.startswith(('crit_', 'failed_', 'rank_', 'q_', 'gate_', 'rule_'))]:
                    ra.pop(k)
                print(f"  α={al}: метрика {Sa['metric']:.4f} ({Sa['best_model']}), лучший из 4 {Sa['best4']:.4f}", flush=True)
                rows.append(ra)
            wf = {m: alpha_wf(V6[V6.model == m]) for m in ('xgb', 'lgbm')}
            Sw = L.summarize({m: shrink(V6[V6.model == m], wf[m]) for m in ('xgb', 'lgbm')})
            rows.append(dict({k: v for k, v in row.items() if not k.startswith(('crit_', 'failed_', 'rank_', 'q_', 'gate_', 'rule_'))},
                             name=f'{n}_awf', lever='Д12.2', cfg=json.dumps({**cfg, 'alpha': 'wf', 'alpha_t': wf}), metric=Sw['metric'],
                             best_model=Sw['best_model'], best4=Sw['best4'], best4_mc=Sw['best4_mc'], xgb_ord=Sw['xgb_ord'], lgbm_ord=Sw['lgbm_ord']))
            print(f"  α_t вне выборки {wf}: метрика {Sw['metric']:.4f} ({Sw['best_model']})", flush=True)
        if a.final:
            al = cfg.get('alpha_post')
            Vf = shrink(V, al) if al is not None else V
            T = pd.concat([r[4] for r in R], ignore_index=True)
            Tf = shrink(T, al) if al is not None else T
            lock = L.REGEN / 'test_locked' / a.final; lock.mkdir(parents=True, exist_ok=True)
            for m in ('xgb', 'lgbm'):
                Vf[Vf.model == m].drop(columns='model').sort_values(['step', 'date', 'ticker']).to_parquet(L.REGEN / f'{a.final}_{m}_h{h}_val.parquet', index=False)
                Tf[Tf.model == m].drop(columns='model').sort_values(['step', 'date', 'ticker']).to_parquet(lock / f'{a.final}_{m}_h{h}_test.parquet', index=False)
            print(f'  выходы: data/regen/{a.final}_{{xgb,lgbm}}_h{h}_val.parquet (шаги 1-9), test -> test_locked/{a.final}/ (не открывался)', flush=True)
    out = D12DIR / f'levers_h{h}.csv'
    df = pd.DataFrame(rows)
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f'готово за {time.time() - t0:.0f} с', flush=True)


if __name__ == '__main__':
    main()
