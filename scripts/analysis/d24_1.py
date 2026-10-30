#!/usr/bin/env python3
"""Д24.1 (владелец 26.10, 02:15): монотонные ограничения LightGBM/XGBoost на финалистах — неубывание прогноза по RV_d, RV_w, RV_m, RVI,
рыночной RV и прогнозу HAR (где он признак). Общие файлы конвейера (l2_lib, tune_L2, tune_L2_optuna, gibrid_*) НЕ меняются: ограничения
ставит обёртка l2_lib.step_data в этом процессе (ключ cfg "mono24": true), как d15_tune/d17_1.

Признаки с +1 (если есть среди признаков модели; MONO24):
  rv, rv_d (RV дня t), rv_w, rv_m; close_RVI (RVI, уровень индекса на закрытии t); rv_IMOEX (RV индекса IMOEX — рыночная RV);
  har_ln (ln прогноза HAR-J/HAR-X, Д8.12); hx_ln_rvi2, hx_ln_rvm_imoex, hx_ln_rvm_mkt (регрессоры HAR-X в матрице Д16.5).
  Ночное умолчание: у остаточных учеников (V9: цель ln RV − ln HAR) знак +1 ставится на те же признаки остатка (как в постановке).

Режимы:
  tune   — переподбор по протоколу B v2 (tune_L2.py: ЛГК LGBM-100 / XGB-40(+100), отсечение 100→50→25→15, 50% строк, шаги 7-9 — точка шага 6)
           с cfg финалиста + mono24. Выход — формат gibrid_io ({tag}_{m}_h{h}{,_top5,_seeds4344}_val.parquet; тест — test_locked/{tag}/).
  fixed  — абляция без переподбора: точки финалиста ({src}_selected_h{h}.csv или Bv2b для h=1), seed 42/43/44 (+ top5 для h=1), шаги 1-9.
  gate   — порог третьего круга по 3 seed: OOS 2017-2020 (шаги 2-5, Б0.1 ord, s² шага t−1), лучший из XGB/LGBM на seed;
           принят, если среднее по seed лучше основы ≥ 1% и лучше на КАЖДОМ seed; 2021 (шаг 6) и 2020 — отдельно; c1-c3 на 2016-2021.
Общее для d24_2/d24_3: refit-движок (refit_jobs), year_q, crit, harj_val.
python3 scripts/analysis/d24_1.py tune  --h 1 --cfg '{"harf":true,"mono24":true}' --tag D24_1_h1_d812m --workers 2
python3 scripts/analysis/d24_1.py fixed --h 1 --cand D12_h1_A_d812 --tag D24_1_h1_d812mf --workers 2
python3 scripts/analysis/d24_1.py gate  --h 1 --tag D24_1_h1_d812m --cur D12_h1_A_d812"""
import argparse, json, multiprocessing as mp, os, sys, time
from pathlib import Path

R0 = Path(os.environ.get('FASTBT_MAIN', '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading')) / 'data' / 'regen'  # К-4
os.environ.setdefault('L2_FEAT', str(R0 / 'features_pit_v3.parquet'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import l2_lib as L

R = L.REGEN
K = ['date', 'ticker', 'step']
M = ('lgbm', 'xgb')
MONO24 = ('rv', 'rv_d', 'rv_w', 'rv_m', 'close_RVI', 'rv_IMOEX', 'har_ln', 'hx_ln_rvi2', 'hx_ln_rvm_imoex', 'hx_ln_rvm_mkt')
SPB = {'xgb': [('max_depth', 'int'), ('learning_rate', 'log'), ('subsample', 'lin'), ('colsample_bytree', 'lin'), ('min_child_weight', 'logint'),
               ('reg_alpha', 'log'), ('reg_lambda', 'log')],
       'lgbm': [('num_leaves', 'int'), ('learning_rate', 'log'), ('bagging_fraction', 'lin'), ('feature_fraction', 'lin'), ('min_data_in_leaf', 'logint'),
                ('reg_alpha', 'log'), ('reg_lambda', 'log')]}

# финалисты Д24.1: имя -> (h, cfg, матрица L2_FEAT, блоки, источник точек)
FIN = {
    'D12_h1_A_d812': (1, {'harf': True}, 'features_pit_v3.parquet', '', 'Bv2b'),
    'D12_h5': (5, {'v9': True, 'harf': True, 'huber': True}, 'features_pit_v3.parquet', 'd7', 'D12_h5'),
    'D16_5_h22': (22, {}, 'features_pit_v3_d165.parquet', '', 'D16_5_h22'),
    'D15_h22': (22, {}, 'features_pit_v3.parquet', '', 'D15_h22'),
    'D12_h1_V0': (1, {}, 'features_pit_v3.parquet', '', 'Bv2b'),
}

_SD0 = L.step_data


def step_data(A, cols, h, step, model, cfg, base_tag, **kw):
    """Обёртка l2_lib.step_data: mono24 — +1 на MONO24 (остальные 0); rel — NaN смещения -> медиана (как d15_tune)."""
    D = _SD0(A, cols, h, step, model, cfg, base_tag, **kw)
    if cfg.get('mono24'):
        D['mono'] = [1 if n in MONO24 else 0 for n in D['names']]
    elif cfg.get('mono_sign') == -1:
        D['mono'] = [-x for x in D['mono']]
    if cfg.get('rel'):
        for k, o in D['off'].items():
            if np.isnan(o).any():
                D['off'][k] = np.where(np.isnan(o), np.nanmedian(o), o)
    return D


L.step_data = step_data


# ---------------------------------------------------------------- оценка (как d16_510.py: правило третьего круга)
def s2(e, code):
    if code in ('rob', 'smrob'):
        return float((1.4826 * np.median(np.abs(e - np.median(e)))) ** 2)
    return float(np.var(e))


def year_q(V, code='ord'):
    """V: [step, rv_actual, rv_pred] шагов 1..6 -> {шаг: QLIKE с Б0.1}; шаг k>=2 — s² по val шага k−1 (sm/smrob — до трёх последних);
    шаг 1 — s² по своим остаткам (только для критериев 2016, одинаково для всех)."""
    v = V[(V.rv_actual > 1e-12) & (V.step <= 6)]
    Gs = {st: g for st, g in v.groupby('step')}
    E = {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in Gs.items()}
    out = {}
    for st, g in Gs.items():
        if code == 'raw':
            F = 1.0
        else:
            src = [st] if st == 1 else [j for j in range(max(1, st - 3) if code in ('sm', 'smrob') else st - 1, st) if j in E]
            F = np.exp(s2(np.concatenate([E[j] for j in src]), code) / 2)
        out[int(st)] = L.qlike(g.rv_actual.values, g.rv_pred.values * F)
    return out


def summ(q):
    return dict(oos1720=float(np.mean([q[s] for s in (2, 3, 4, 5)])), y2020=q[5], y2021=q[6], dev1721=float(np.mean([q[s] for s in range(2, 7)])))


def harj_val(h):
    p = R / f'D16_10_harj_h{h}_val.parquet'
    if p.exists():
        return pd.read_parquet(p)
    A, FC = L.load_base(R / 'features_pit_v3.parquet')
    hc = L.harj_cols(A)
    out = []
    for s in range(1, 7):
        tp, vi, _, _, _ = L.split(A, h, s)
        pv = L.har_ln(tp, [vi], h, hc)[0]
        out.append(pd.DataFrame({'date': vi['date'].values, 'ticker': vi['ticker'].values, 'step': s, 'rv_actual': vi[L.tcol(h)].values, 'rv_pred': np.exp(pv)}))
    d = pd.concat(out, ignore_index=True)
    d.to_parquet(p, index=False)
    return d


def crit(qd, qh, qs):
    """c1–c3 на шагах 1–6 (2016–2021) для одиночного дизайна: qd — дизайн, qh — HAR-J, qs — {модель: одиночная} с той же s²."""
    st = range(1, 7)
    hy = np.array([qd[s] for s in st]); har = np.array([qh[s] for s in st])
    sing = np.minimum.reduce([np.array([q[s] for s in st]) for q in qs.values()]) if qs else hy
    worst = (hy / har).max()
    worst_single = min((np.array([q[s] for s in st]) / har).max() for q in qs.values()) if qs else np.inf
    return dict(c1=bool((hy <= 1.02 * har).all()), c2=bool((hy <= 1.02 * sing).all()), c3=bool(worst < worst_single))


def seed_frame(T, m, h, seed, regen=R):
    """single-нога модели m с данным seed: seed 42 — {T}_{m}_h{h}_val; 43/44 — _seeds4344; 45/46 — D16_10 (если есть); 45-51 — D24_3."""
    if seed == 42:
        v = pd.read_parquet(regen / f'{T}_{m}_h{h}_val.parquet')
    elif seed in (43, 44):
        v = pd.read_parquet(regen / f'{T}_{m}_h{h}_seeds4344_val.parquet')
    else:
        p = R / f'D24_3_{T}_{m}_h{h}_seeds4551_val.parquet'
        v = pd.read_parquet(p)
    if 'seed' in v:
        v = v[v.seed == seed]
    if 'rank' in v:
        v = v[v['rank'] == 1] if (v['rank'] == 1).any() else v
    return v[v.step <= 6]


def gate(tag, cur, h, seeds=(42, 43, 44), code='ord'):
    def met(T):
        out = {}
        for sd in seeds:
            ev = {m: year_q(seed_frame(T, m, h, sd), code) for m in M}
            bm = min(ev, key=lambda m: summ(ev[m])['oos1720'])
            out[sd] = (summ(ev[bm]), bm, ev)
        return out
    a, b = met(tag), met(cur)
    ma, mb = np.mean([a[s][0]['oos1720'] for s in a]), np.mean([b[s][0]['oos1720'] for s in b])
    each = all(a[s][0]['oos1720'] < b[s][0]['oos1720'] for s in a)
    H = year_q(harj_val(h), code)
    ev42 = a[seeds[0]][2]; bm = a[seeds[0]][1]
    c = crit(ev42[bm], H, {m: ev42[m] for m in M if m != bm})
    r = dict(tag=tag, cur=cur, h=h, mean_tag=ma, mean_cur=mb, rel=ma / mb - 1, each_seed=each, accepted=bool(ma <= 0.99 * mb and each),
             y2021_tag=np.mean([a[s][0]['y2021'] for s in a]), y2021_cur=np.mean([b[s][0]['y2021'] for s in b]),
             y2020_tag=np.mean([a[s][0]['y2020'] for s in a]), y2020_cur=np.mean([b[s][0]['y2020'] for s in b]),
             **{f'tag_s{s}': a[s][0]['oos1720'] for s in a}, **{f'cur_s{s}': b[s][0]['oos1720'] for s in b},
             **{f'model_s{s}': a[s][1] for s in a}, **{f'tag_q{2015 + s}': ev42[bm][s] for s in range(1, 7)},
             **{f'harj_q{2015 + s}': H[s] for s in range(1, 7)}, **c)
    return r


# ---------------------------------------------------------------- refit-движок (точки известны)
G = {}


def set_matrix(h, matrix, blocks):
    """Матрица финалиста: l2_lib.FEAT / FEAT_CFG подменяются в процессе (load_base читает их при вызове)."""
    L.FEAT = R / (f'{Path(matrix).stem}_{blocks}_h{h}.parquet' if blocks else matrix)
    c = (R / matrix).with_suffix('.config.json')
    L.FEAT_CFG = c if c.exists() else R / 'features_pit_v3.config.json'


def load_cfg(h, cfg, matrix, blocks):
    set_matrix(h, matrix, blocks)
    A, FC = L.load_base()
    A2, cols, _, _ = L.build_features(A, FC, h, cfg)
    G.update(A=A2, cols=cols, cfg=cfg, h=h)
    return len(cols)


_REF = {}


def _unit(a):
    """(модель, шаг, [(params, seed, rank, point)], btag) -> [(m, s, seed, rank, point, bi, val, test)]; один step_data на единицу."""
    m, s, items, btag = a
    D = L.step_data(G['A'], G['cols'], G['h'], s, m, G['cfg'], btag)
    if s > 1 and m not in _REF:
        _REF[m] = L.make_ref(m, G['A'], G['cols'], G['h'], G['cfg'], btag)
    res = []
    for p, seed, rank, pnt in items:
        pv, pt, bi, _, _ = L.fit(m, p, D, G['cfg'], seed=seed, ref=_REF.get(m) if s > 1 else None)
        fr = []
        for part, pred, yr in (('va', pv, D['vy']), ('te', pt, D['ty'])):
            d = D['parts'][part]
            fr.append(pd.DataFrame({'date': d['date'].values, 'ticker': d['ticker'].values, 'year': yr, 'step': s, 'point': pnt, 'seed': seed,
                                    'rank': rank, 'rv_actual': d[L.tcol(G['h'])].values, 'rv_pred': np.clip(np.exp(pred), 1e-10, None)}))
        res.append((m, s, seed, rank, pnt, bi, fr[0], fr[1]))
    return res


def refit_jobs(tasks, workers, btag):
    """tasks: [(m, step, params, seed, rank, point)] -> список результатов _unit."""
    _REF.clear()
    units = {}
    for m, st, p, sd, rk, pnt in tasks:
        units.setdefault((m, st), []).append((p, sd, rk, pnt))
    U = sorted(units.items(), key=lambda kv: (-kv[0][1], kv[0][0]))
    with mp.get_context('fork').Pool(workers) as P:
        return [x for r in P.map(_unit, [(m, st, it, btag) for (m, st), it in U], chunksize=1) for x in r]


def cast(r, m):
    return {k: (int(r[k]) if sc in ('int', 'logint') else float(r[k])) for k, sc in SPB[m]}


def points(src, h, steps, top5=False):
    """-> {(m, step): [(params, rank, point)]}: точки финалиста. Bv2b (h=1) — сетка ЛГК и top5; иначе — {src}_selected_h{h}.csv (точка шага; 7-9 — шага 6)."""
    out = {}
    if src == 'Bv2b':
        import d12_h1 as H1
        grid = H1.grid()
        S = pd.read_csv(R / f'Bv2b_selected_h{h}.csv')
        for m in M:
            T = pd.read_parquet(R / f'Bv2b_{m}_h{h}_top5_val.parquet').groupby(['step', 'rank']).point.first()
            for s in steps:
                sp = int(S[(S.model == m) & (S.step == s)].point.iloc[0])
                out[(m, s)] = [(grid[(m, sp)], 1, sp)]
                if top5 and s <= 6:
                    out[(m, s)] += [(grid[(m, int(T[(s, r)]))], r, int(T[(s, r)])) for r in range(2, 6)]
    else:
        S = pd.read_csv(R / f'{src}_selected_h{h}.csv')
        for m in M:
            for s in steps:
                r = S[(S.model == m) & (S.step == s)].iloc[0]
                out[(m, s)] = [(cast(r, m), 1, int(r['trial']) if 'trial' in r else int(r['point']))]
    return out


def write_legs(res, tag, h, seeds_main=(42,), suffix_seeds='_seeds4344', top5=False):
    lock = R / 'test_locked' / tag
    lock.mkdir(parents=True, exist_ok=True)
    for m in M:
        rr = [x for x in res if x[0] == m]
        sel = lambda f: (pd.concat([x[6] for x in rr if f(x)], ignore_index=True), pd.concat([x[7] for x in rr if f(x)], ignore_index=True))
        files = {'': sel(lambda x: x[3] == 1 and x[2] in seeds_main)}
        if any(x[2] not in seeds_main for x in rr):
            files[suffix_seeds] = sel(lambda x: x[2] not in seeds_main and x[3] == 1)
        if top5:
            files['_top5'] = sel(lambda x: x[1] <= 6 and x[2] == 42)
        for sfx, (v, t) in files.items():
            v.sort_values(['step', 'seed', 'rank', 'date', 'ticker']).to_parquet(R / f'{tag}_{m}_h{h}{sfx}_val.parquet', index=False)
            t.sort_values(['step', 'seed', 'rank', 'date', 'ticker']).to_parquet(lock / f'{tag}_{m}_h{h}{sfx}_test.parquet', index=False)


# ---------------------------------------------------------------- команды
def cmd_fixed(a):
    h, cfg, matrix, blocks, src = FIN[a.cand]
    cfg = dict(cfg, mono24=True) if not a.nomono else dict(cfg)
    n = load_cfg(h, cfg, matrix, blocks)
    btag = 'Bv2b' if src == 'Bv2b' else (a.btag or 'Bv2b')
    top5 = src == 'Bv2b'
    P = points(src, h, range(1, 10), top5=top5)
    tasks = []
    for (m, s), its in P.items():
        for p, rk, pnt in its:
            tasks.append((m, s, p, 42, rk, pnt))
            if rk == 1:
                tasks += [(m, s, p, sd, 1, pnt) for sd in (43, 44)]
    t0 = time.time()
    print(f'{time.strftime("%T")} fixed {a.cand} -> {a.tag}: h={h} cfg={cfg} признаков {n}, задач {len(tasks)}, воркеров {a.workers}', flush=True)
    res = refit_jobs(tasks, a.workers, btag)
    write_legs(res, a.tag, h, top5=top5)
    pd.DataFrame([dict(model=x[0], step=x[1], seed=x[2], rank=x[3], point=x[4], best_iteration=x[5]) for x in res]).to_csv(R / f'{a.tag}_trees.csv', index=False)
    print(f'{time.strftime("%T")} {a.tag} готов за {time.time() - t0:.0f} с', flush=True)


def cmd_tune(a, rest):
    h, cfg, matrix, blocks, _ = FIN[a.cand]
    cfg = dict(cfg, mono24=True)
    set_matrix(h, matrix, blocks)
    import tune_L2 as T
    sys.argv = [sys.argv[0], '--h', str(h), '--cfg', json.dumps(cfg), '--space', 'B', '--workers', str(a.workers), '--tag', a.tag] + rest
    T.main()


def corrupt(A, t0, seed=7):
    """Всё с даты t0 включительно: числовые столбцы × шум exp(N(0,1)), цели перемешаны (детектор «будущее не влияет»)."""
    rng = np.random.default_rng(seed)
    A = A.copy()
    m = (pd.to_datetime(A['date']) >= pd.Timestamp(t0)).values
    num = [c for c in A.columns if c not in ('date', 'ticker', 'year') and pd.api.types.is_float_dtype(A[c])]
    for c in num:
        v = A.loc[m, c].values
        A.loc[m, c] = rng.permutation(v) if 'target' in c else v * np.exp(rng.normal(size=len(v)))
    return A


def leak_check(label, step, unit=None):
    """G загружен (A, cols, cfg, h, pts). Модель шага step (seed 42) обучена на годах до val-года включительно;
    шум с 1 января тестового года шага -> val-прогнозы обязаны совпасть побайтно; самопроверка: шум с 1 января val-года -> обязаны измениться."""
    unit = unit or _unit
    ty = L.TEST_YEARS[step - 1]
    A0 = G['A']
    rows = []
    for m in M:
        p, rk, pnt = G['pts'][(m, step)][0]
        _REF.clear()
        base = unit((m, step, [(p, 42, 1, pnt)], 'Bv2b'))[0][6]
        for t0, must in ((f'{ty}-01-01', 'равны'), (f'{ty - 1}-01-01', 'изменились')):
            G['A'] = corrupt(A0, t0)
            _REF.clear()
            v = unit((m, step, [(p, 42, 1, pnt)], 'Bv2b'))[0][6]
            G['A'] = A0
            d = float(np.abs(np.log(v.rv_pred.values) - np.log(base.rv_pred.values)).max()) if len(v) == len(base) else np.inf
            ok = bool(d == 0.0) if must == 'равны' else bool(d > 0)
            rows.append(dict(contour=label, model=m, step=step, noise_from=t0, expect=must, max_abs_dln=d, ok=ok))
            print(f'leak {label} {m} шаг {step}: шум с {t0} -> max|Δ ln| = {d:.3g} (ожидается: {must}) {"OK" if ok else "ПРОВАЛ"}', flush=True)
    _REF.clear()
    return rows


def save_leak(out):
    p = R / 'D24_leak.csv'
    df = pd.DataFrame(out).assign(time=time.strftime('%F %T'))
    if p.exists():
        df = pd.concat([pd.read_csv(p), df], ignore_index=True)
    df.to_csv(p, index=False)
    assert all(r['ok'] for r in out), 'детектор: провал'


def cmd_leak(a):
    out = []
    for cand in a.cands:
        h, cfg, matrix, blocks, src = FIN[cand]
        load_cfg(h, dict(cfg, mono24=True), matrix, blocks)
        G['pts'] = points(src, h, [a.step])
        out += leak_check(f'{cand}+mono24', a.step)
    save_leak(out)


def cmd_promote(a):
    """Прошедший порог: гибрид gibrid_v17.py (--dev-eval oos, шаги 7-9 скрыты, эталон — та же s²), yaml кандидата третьего круга
    (data/regen/final3/kandidaty/{boost}.yaml) и OOS-прогнозы для объединителя Д22 (data/regen/D24_oos/{boost}_h{h}_{val|test}.parquet:
    среднее RV seed 42/43/44 каждой модели, шаги 1-9; тест — test_locked/D24_oos/, без метрик)."""
    import subprocess, yaml
    h = a.h
    feat = R / a.features
    out = R / 'D24_gibrid' / a.boost
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-W', 'ignore', str(Path(__file__).resolve().parent / 'gibrid_v17.py'), '--boost', a.boost, '--h', str(h),
           '--features', str(feat), '--legs', *a.legs, '--out', str(out), '--devlog', str(R / 'D24_gibrid' / 'gibrid_devlog.md')]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent))
    (out / 'gibrid.log').write_text(r.stdout[-20000:] + '\n' + r.stderr[-5000:])
    print(f'gibrid_v17 {a.boost}: код {r.returncode}', flush=True)
    for k, d in (('val', R / 'D24_oos'), ('test', R / 'test_locked' / 'D24_oos')):
        d.mkdir(parents=True, exist_ok=True)
        src = R if k == 'val' else R / 'test_locked' / a.boost
        fr = []
        for m in M:
            v = pd.concat([pd.read_parquet(src / f'{a.boost}_{m}_h{h}_{k}.parquet'), pd.read_parquet(src / f'{a.boost}_{m}_h{h}_seeds4344_{k}.parquet')])
            v = v[v['rank'] == 1] if 'rank' in v else v
            fr.append(v.groupby(K, as_index=False).agg(year=('year', 'first'), rv_actual=('rv_actual', 'first'), rv_pred=('rv_pred', 'mean')).assign(model=m))
        pd.concat(fr).to_parquet(d / f'{a.boost}_h{h}_{k}.parquet', index=False)
    y = dict(boost=a.boost, h=h, semeystvo=a.semeystvo, rychag=a.rychag, opisanie=a.opisanie, regen='data/regen', legs=list(a.legs),
             odinochnaya=False, chistyy=bool(a.chistyy), test_dir=f'data/regen/test_locked/{a.boost}', osnova=a.osnova, n_variantov=a.n,
             features=f'data/regen/{a.features}', oos_d22=f'data/regen/D24_oos/{a.boost}_h{h}_val.parquet', kod='scripts/analysis/d24_1.py, d24_2.py')
    for m in M:
        v = pd.read_parquet(R / f'{a.boost}_{m}_h{h}_val.parquet', columns=['step'])
        t = pd.read_parquet(R / 'test_locked' / a.boost / f'{a.boost}_{m}_h{h}_test.parquet', columns=['step'])
        assert sorted(v.step.unique()) == list(range(1, 10)) and sorted(t.step.unique()) == list(range(1, 10)), (m, a.boost)
    od = R / 'final3' / 'kandidaty'
    yaml.safe_dump(y, open(od / f'{a.boost}.yaml', 'w'), allow_unicode=True, sort_keys=False)
    print(f'{od / (a.boost + ".yaml")} записан', flush=True)


def cmd_gate(a):
    r = gate(a.tag, a.cur, a.h)
    p = R / 'D24_1_gate.csv'
    df = pd.DataFrame([dict(time=time.strftime('%F %T'), **r)])
    if p.exists():
        df = pd.concat([pd.read_csv(p), df], ignore_index=True)
    df.to_csv(p, index=False, float_format='%.6g')
    print(json.dumps({k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else (bool(v) if isinstance(v, (bool, np.bool_)) else v)) for k, v in r.items()}, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('fixed'); p.add_argument('--cand', required=True); p.add_argument('--tag', required=True); p.add_argument('--workers', type=int, default=2)
    p.add_argument('--btag', default=None); p.add_argument('--nomono', action='store_true')
    p = sp.add_parser('tune'); p.add_argument('--cand', required=True); p.add_argument('--tag', required=True); p.add_argument('--workers', type=int, default=2)
    p = sp.add_parser('gate'); p.add_argument('--h', type=int, required=True); p.add_argument('--tag', required=True); p.add_argument('--cur', required=True)
    p = sp.add_parser('leak'); p.add_argument('--cands', nargs='+', required=True); p.add_argument('--step', type=int, default=3)
    p = sp.add_parser('promote'); p.add_argument('--boost', required=True); p.add_argument('--h', type=int, required=True)
    p.add_argument('--features', default='features_pit_v3.parquet'); p.add_argument('--legs', nargs='+', default=['single', 'top5', 'seeds3'])
    p.add_argument('--semeystvo', required=True); p.add_argument('--rychag', required=True); p.add_argument('--opisanie', required=True)
    p.add_argument('--osnova', required=True); p.add_argument('--chistyy', type=int, default=0); p.add_argument('--n', type=int, default=1)
    a, rest = ap.parse_known_args()
    if a.cmd == 'promote':
        return cmd_promote(a)
    if a.cmd == 'tune':
        return cmd_tune(a, rest)
    {'fixed': cmd_fixed, 'gate': cmd_gate, 'leak': cmd_leak}[a.cmd](a)


if __name__ == '__main__':
    main()
