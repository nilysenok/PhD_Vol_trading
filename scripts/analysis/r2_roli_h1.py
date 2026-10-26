#!/usr/bin/env python3
"""R2, агент Б1 (протокол v2 docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md, строка «Б1»): h=1, val шагов 1–6.
  f1    — F_1 = D22_h1 top5/V0_XGB/B-sm/L0 на seed 42–51 (42 — контроль с первым раундом, 43–51 — для 2σ). Нога top5 XGB с seed s:
          пять точек top5 шага (tune_L2 D22_h1), каждая с seed s, среднее RV; обучение — l2_lib.fit путём d22_h1 (patch), как в первом раунде;
          F_1 = нога × exp(s²_sm/2), s² по val шагов t−3..t−1 (gibrid_bias.correct, F_oos), шаг 1 — 1,0.
          Контроль: single и пять точек top5 seed 42, single seed 43/44 — с сохранёнными рядами (max |Δ ln| ≤ 1e-7);
          QLIKE F_1 seed 42 (шаги 2–5 и 6) = designs_rank первого раунда (|Δ| ≤ 1e-9). Иначе — стоп.
  leak  — детектор «будущее не влияет» на этом коде (шаг yaml leak_step): шум с 1 января тестового года — val не меняется; с val-года — меняется.
Тест не считается и не пишется (раздел «Тестовая часть»: тестовые прогнозы — после заморозки, только для победителей).
Выход: data/regen/R2/cache/B1/<ряд>_h1_s<seed>_val.parquet (date, ticker, year, step, rv_actual, rv_pred), INDEX_B1.csv; сырьё — B1/_raw/.
python3 scripts/analysis/r2_roli_h1.py f1 [--workers 5]"""
import os, sys, time, json, hashlib, argparse
from pathlib import Path

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ROOT = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading')
R = ROOT / 'data' / 'regen'
YAML = REPO / 'models' / 'designs' / 'r2' / 'r2_B1_roli_h1.yaml'
import yaml
Y = yaml.safe_load(open(YAML))
CMD = sys.argv[1] if len(sys.argv) > 1 else ''
if CMD in ('f1', 'leak'):
    os.environ['L2_FEAT'] = str(R / Y['F1']['matrix'])
elif CMD in ('roli', 'leak_roli'):
    os.environ['L2_FEAT'] = str(R / Y['roles_matrix'])
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd

H = int(Y['h'])
STEPS = tuple(Y['steps'])
CACHE = ROOT / Y['paths']['cache']
INDEX = ROOT / Y['paths']['index']
TOL = float(Y['control_tol'])
COLS = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
KEY = ['date', 'ticker', 'step']
STOP = CACHE / 'STOP_B1.txt'


def log(*a):
    print(time.strftime('%F %T'), *a, flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


CODE_SHA = sha_file(Path(__file__).resolve())
YAML_SHA = sha_file(YAML)


def put(df, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)


def fpath(ryad, seed):
    return CACHE / f'{ryad}_h{H}_s{"det" if seed is None else seed}_val.parquet'


def index_add(p, ryad, seed, df, note=''):
    st = sorted(int(x) for x in df.step.unique())
    row = dict(файл=str(p.relative_to(ROOT)), ряд=ryad, h=H, seed='det' if seed is None else seed, шаги=f'{st[0]}-{st[-1]}', строки=len(df),
               sha256=sha_file(p), sha_кода=CODE_SHA, yaml=f'{YAML.relative_to(REPO)}@{YAML_SHA[:16]}', время=time.strftime('%F %T'), примечание=note)
    I = pd.read_csv(INDEX) if INDEX.exists() else pd.DataFrame(columns=list(row))
    I = pd.concat([I[I['файл'] != row['файл']], pd.DataFrame([row])], ignore_index=True).sort_values('файл')
    tmp = INDEX.with_name(INDEX.name + '.tmp')
    I.to_csv(tmp, index=False)
    os.replace(tmp, INDEX)


def append_csv(p, rows):
    df = pd.DataFrame(rows)
    if p.exists():
        df = pd.concat([pd.read_csv(p), df], ignore_index=True)
    df.to_csv(p, index=False)


def q_dev(df):
    """QLIKE по шагам 2–6 (val 2017–2021, rv_actual > 1e-12). Шагов > 6 в кэше Б1 нет по построению."""
    import l2_lib as L
    d = df[(df.step >= 2) & (df.step <= 6) & (df.rv_actual > 1e-12)]
    return {int(s): float(L.qlike(g.rv_actual.values, g.rv_pred.values)) for s, g in d.groupby('step')}


# ---------------------------------------------------------------- F_1: нога D22_h1 XGB (single/top5) по seed
G = {}


def setup_f1():
    t0 = time.time()
    c = Y['F1']
    import d22_h1 as P22  # подмены l2_lib (har_ln -> ln HAR-X⁺ шага, признаки слоя 2, load_base + HXP/HJ)
    import l2_lib as L
    assert P22.CFG == c['cfg'] and P22.TAG == c['boost']
    P22.patch()
    A, FC = L.load_base()
    A2, cols, _, _ = L.build_features(A, FC, H, c['cfg'])
    assert Path(L.FEAT).name == c['matrix']
    GR = pd.read_csv(R / f'{c["boost"]}_grid.csv').set_index(['model', 'point'])
    S = pd.read_csv(R / f'{c["boost"]}_selected_h{H}.csv')
    m = c['model']
    T = pd.read_parquet(R / c['saved']['top5'], columns=['step', 'rank', 'point']).groupby(['step', 'rank']).point.first()
    P = {}
    for s in STEPS:
        assert int(T[(s, 1)]) == int(S[(S.model == m) & (S.step == s)].point.iloc[0])
        P[s] = [(GR.loc[(m, int(T[(s, r)]))].to_dict(), r, int(T[(s, r)])) for r in range(1, 6)]
    G.update(L=L, A=A2, cols=cols, cfg=c['cfg'], btag=c['btag'], m=m, P=P)
    log(f'F_1: {c["boost"]} {m}, матрица {Path(L.FEAT).name}, признаков {len(cols)}, строк {len(A2)}, настройка {time.time() - t0:.0f} с')


_DC, _RC = {}, {}


def fit_unit(s, seed, items):
    L, m = G['L'], G['m']
    if s not in _DC:
        _DC.clear()
        _DC[s] = L.step_data(G['A'], G['cols'], H, s, m, G['cfg'], G['btag'])
    D = _DC[s]
    if s > 1 and m not in _RC:
        _RC[m] = L.make_ref(m, G['A'], G['cols'], H, G['cfg'], G['btag'])
    ref = _RC.get(m) if s > 1 else None
    fr = []
    for p, rank, pid in items:
        t0 = time.time()
        pv, _pt, bi, _, _ = L.fit(m, p, D, G['cfg'], seed=seed, ref=ref)  # тестовый прогноз не сохраняется
        d = D['parts']['va']
        fr.append(pd.DataFrame({'date': d['date'].values, 'ticker': d['ticker'].values, 'year': D['vy'], 'step': s,
                                'rv_actual': d[L.tcol(H)].values, 'rv_pred': np.clip(np.exp(pv), 1e-10, None),
                                'rank': rank, 'pid': pid, 'seed': seed, 'trees': bi, 'sec': time.time() - t0}))
    return pd.concat(fr, ignore_index=True)


def raw_f1(seed, s):
    return CACHE / '_raw' / f'{Y["F1"]["boost"]}_{Y["F1"]["model"]}_s{seed}_st{s:02d}.parquet'


def job_f1(t):
    s, seed = t
    v = fit_unit(s, seed, G['P'][s])
    put(v, raw_f1(seed, s))
    return s, seed, float(v.groupby('rank').sec.first().sum())


def control_f1(seed, V):
    c = Y['F1']
    rows = []
    checks = [('single', c['saved']['single'], 1), ('top5', c['saved']['top5'], None)] if seed == 42 else \
             ([('s4344', c['saved']['s4344'], 1)] if seed in (43, 44) else [])
    for kind, f, rk in checks:
        S = pd.read_parquet(R / f)
        S = S[(S.step <= 6) & (S.seed == seed)]
        k = KEY + (['rank'] if rk is None else [])
        mine = V if rk is None else V[V['rank'] == rk]
        if rk is not None:
            S = S[S['rank'] == rk]
        mm = mine[k + ['rv_pred']].merge(S[k + ['rv_pred']], on=k, how='outer', suffixes=('', '_saved'), indicator=True)
        b = mm[mm._merge == 'both']
        d = float(np.abs(np.log(b.rv_pred) - np.log(b.rv_pred_saved)).max()) if len(b) else np.inf
        ok = bool(len(b) == len(mm) and d <= TOL)
        rows.append(dict(ryad=f'{c["boost"]}_{c["model"]}', seed=seed, kind=kind, saved=f, rows=len(mm), matched=len(b), max_abs_dln=d, ok=ok,
                         time=time.strftime('%F %T')))
        log(f'контроль s{seed} {kind}: строк {len(b)}/{len(mm)}, max|Δ ln| = {d:.2e} {"OK" if ok else "ПРОВАЛ"}')
    if rows:
        append_csv(CACHE / 'control_B1.csv', rows)
        if not all(r['ok'] for r in rows):
            STOP.write_text(f'{time.strftime("%F %T")} контроль не пройден: {rows}\n')
            raise SystemExit(f'СТОП: seed {seed} не повторяет сохранённый ряд первого раунда')


def close_f1(seed):
    c = Y['F1']
    V = pd.concat([pd.read_parquet(raw_f1(seed, s)) for s in STEPS], ignore_index=True)
    control_f1(seed, V)
    leg = f'{c["boost"]}_{c["model"]}'
    srt = lambda d: d.sort_values(['step', 'date', 'ticker']).reset_index(drop=True)
    single = srt(V[V['rank'] == 1][COLS])
    top5 = srt(V.groupby(KEY, as_index=False).agg(year=('year', 'first'), rv_actual=('rv_actual', 'first'), rv_pred=('rv_pred', 'mean'))[COLS])
    ok = top5[top5.rv_actual > 1e-12]
    E = {int(st): np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in ok.groupby('step')}
    assert sorted(E) == list(STEPS) and c['code'] == 'sm'
    F = {st: 1.0 if st == 1 else float(np.exp(np.var(np.concatenate([E[j] for j in range(max(1, st - 3), st)])) / 2)) for st in STEPS}
    f1 = top5.assign(rv_pred=top5.rv_pred.values * top5.step.map(F).values)
    if seed == 42:
        q = q_dev(f1)
        m1720, q21 = float(np.mean([q[s] for s in (2, 3, 4, 5)])), q[6]
        dd = max(abs(m1720 - c['control']['mean_1720']), abs(q21 - c['control']['q2021']))
        log(f'контроль F_1 s42: 2017–2020 {m1720:.12f} (первый раунд {c["control"]["mean_1720"]:.12f}), 2021 {q21:.12f} '
            f'({c["control"]["q2021"]:.12f}), max|Δ| {dd:.1e}')
        append_csv(CACHE / 'control_B1.csv', [dict(ryad='F1', seed=42, kind='qlike_designs_rank', saved=c['control']['design_file'], rows=len(f1),
                                                   matched=len(f1), max_abs_dln=dd, ok=bool(dd <= 1e-9), time=time.strftime('%F %T'))])
        if dd > 1e-9:
            STOP.write_text(f'{time.strftime("%F %T")} контроль F_1 не пройден: {dd}\n')
            raise SystemExit('СТОП: F_1 seed 42 не повторяет первый раунд')
    for ryad, df, note in ((f'{leg}_single', single, 'нога single D22_h1 XGB'), (f'{leg}_top5', top5, 'нога top5 D22_h1 XGB (5 точек, seed s)'),
                           ('F1', f1, 'F_1 = top5/V0_XGB/B-sm/L0')):
        p = fpath(ryad, seed)
        put(df, p)
        index_add(p, ryad, seed, df, note + ('; контроль с первым раундом' if seed == 42 else ''))
    pd.DataFrame([dict(step=s, F_sm=F[s]) for s in STEPS]).to_csv(CACHE / f'F1_h{H}_s{seed}_factors.csv', index=False)
    log(f'F_1 s{seed}: записаны single, top5, F1')


def cmd_f1(a):
    import multiprocessing as mp
    if STOP.exists():
        raise SystemExit(f'СТОП-файл {STOP}: {STOP.read_text()}')
    setup_f1()
    seeds = tuple(a.seeds or Y['F1']['seeds'])
    tasks = [(s, sd) for sd in seeds for s in sorted(STEPS, reverse=True) if not raw_f1(sd, s).exists()]
    closed = {sd for sd in seeds if fpath('F1', sd).exists()}
    for sd in seeds:
        if sd not in closed and all(raw_f1(sd, s).exists() for s in STEPS):
            close_f1(sd); closed.add(sd)
    log(f'F_1: задач {len(tasks)}, воркеров {a.workers}')
    t0, n = time.time(), 0
    if tasks:
        with mp.get_context('fork').Pool(a.workers) as P:
            for s, sd, sec in P.imap_unordered(job_f1, tasks, chunksize=1):
                n += 1
                el = time.time() - t0
                log(f'шаг {s} s{sd}: 5 точек за {sec:.0f} с; {n}/{len(tasks)}, прошло {el / 60:.1f} мин, осталось ~{el / n * (len(tasks) - n) / 60:.0f} мин')
                if sd not in closed and all(raw_f1(sd, x).exists() for x in STEPS):
                    close_f1(sd); closed.add(sd)
    log(f'F_1: готово, seed {sorted(closed)}, стена {(time.time() - t0) / 60:.1f} мин')


# ---------------------------------------------------------------- роли уровня 1: признак / остаток / офсет gamma (seed 42, точка шага)
RS = {}      # текущая (роль, нога) в процессе-воркере
_BASE = {}


def leg_file(leg):
    if leg == 'HARX_d17':
        return R / 'D17' / 'harx_base_h1.parquet'
    return ROOT / Y['legs_dir'] / f'{leg}_h{H}_lnall.parquet'


def base_of(leg):
    """{(шаг, part): DataFrame(date, ticker, base_ln)} — ln-прогноз ноги шага на строках окна; HARJ_l2 -> None (l2_lib.har_ln без подмены)."""
    if leg == 'HARJ_l2':
        return None
    if leg not in _BASE:
        b = pd.read_parquet(leg_file(leg), columns=['date', 'ticker', 'step', 'part', 'base_ln'])
        b = b[b.step.isin(STEPS)].copy()
        b['date'] = b['date'].astype(RS['date_dtype'])
        assert not b.duplicated(['date', 'ticker', 'step', 'part']).any(), leg
        _BASE[leg] = {(int(s), p): g[['date', 'ticker', 'base_ln']] for (s, p), g in b.groupby(['step', 'part'])}
    return _BASE[leg]


def install_role_wrapper(L):
    """Подмена l2_lib.step_data в процессе (так же действует в make_ref — бины окна шага 1 видят тот же признак):
    признак (harf) — столбец har_ln := ln ноги; остаток / gamma (v9) — смещение off := ln ноги; b01 — off += s²/2 по остаткам ln ноги на
    обучающем окне (как d17_1.step_data); gamma — метка ≥ 1e-8. Строки обучения и val без прогноза ноги — ошибка; тест шага не используется (0)."""
    sd0 = L.step_data

    def step_data(A, cols, h, step, model, cfg, base_tag, **kw):
        D = sd0(A, cols, h, step, model, cfg, base_tag, **kw)
        B = base_of(RS['leg'])
        if B is not None:
            for k, d in D['parts'].items():
                bb = B.get((step, k))
                v = d[['date', 'ticker']].merge(bb, on=['date', 'ticker'], how='left')['base_ln'].values if bb is not None else np.full(len(d), np.nan)
                if k == 'te':
                    v = np.where(np.isnan(v), 0.0, v)
                assert not np.isnan(v).any(), (RS['leg'], step, k)
                if cfg.get('harf'):
                    assert D['names'][-1] == 'har_ln'
                    D['X'][k][:, -1] = v.astype(np.float32)
                if cfg.get('v9'):
                    D['off'][k] = v
        D['s2'] = np.nan
        if cfg.get('b01'):
            s2 = float(np.var(np.log(np.clip(D['y']['tr'], 1e-10, None)) - D['off']['tr']))
            D['off'] = {k: v + s2 / 2 for k, v in D['off'].items()}
            D['s2'] = s2
        if cfg.get('obj') == 'gamma':
            D['y'] = {k: np.clip(v, 1e-8, None) for k, v in D['y'].items()}
        return D
    L.step_data = step_data


def role_points(spec, m):
    if spec['points'] == 'bv2b':
        import d12_h1 as D1
        grid = D1.grid()
        S = pd.read_csv(R / 'Bv2b_selected_h1.csv')
        return {s: grid[(m, int(S[(S.model == m) & (S.step == s)].point.iloc[0]))] for s in STEPS}
    kind, tag = spec['points'].split(':')
    assert kind == 'optuna'
    from tune_L2 import SPACEB
    S = pd.read_csv(R / f'{tag}_selected_h{H}.csv')
    out = {}
    for s in STEPS:
        r = S[(S.model == m) & (S.step == s)].iloc[0]
        out[s] = {k: (int(r[k]) if sc in ('int', 'logint') else float(r[k])) for k, _, _, sc in SPACEB[m]}
    return out


def setup_roli():
    t0 = time.time()
    import l2_lib as L
    assert Path(L.FEAT).name == Y['roles_matrix']
    install_role_wrapper(L)
    A, FC = L.load_base()
    RS.update(L=L, A=A, date_dtype=A['date'].dtype, cols={}, P={})
    for role, spec in Y['roles'].items():
        RS['cols'][role] = L.build_features(A, FC, H, spec['cfg'])[1]
        for m in spec['models']:
            RS['P'][(role, m)] = role_points(spec, m)
    RS['A'] = L.build_features(A, FC, H, {})[0]
    assert all(c in RS['A'] for cs in RS['cols'].values() for c in cs)
    log(f'роли: матрица {Path(L.FEAT).name}, строк {len(A)}, признаков ' + ', '.join(f'{r} {len(c)}' for r, c in RS['cols'].items())
        + f'; настройка {time.time() - t0:.0f} с')


_RREF = {}


def fit_role(leg, role, m, s):
    L = RS['L']
    spec = Y['roles'][role]
    cfg = spec['cfg']
    RS['leg'] = leg
    D = L.step_data(RS['A'], RS['cols'][role], H, s, m, cfg, 'Bv2b')
    ref = None
    if s > 1:
        k = (leg, role, m)
        if k not in _RREF:
            if len(_RREF) >= 6:
                _RREF.clear()
            _RREF[k] = L.make_ref(m, RS['A'], RS['cols'][role], H, cfg, 'Bv2b')
        ref = _RREF[k]
    t0 = time.time()
    pv, _pt, bi, _, _ = L.fit(m, RS['P'][(role, m)][s], D, cfg, seed=42, ref=ref)  # тестовый прогноз не сохраняется
    d = D['parts']['va']
    return pd.DataFrame({'date': d['date'].values, 'ticker': d['ticker'].values, 'year': D['vy'], 'step': s, 'rv_actual': d[L.tcol(H)].values,
                         'rv_pred': np.clip(np.exp(pv), 1e-10, None), 'trees': bi, 's2_b01': D['s2'], 'sec': time.time() - t0})


def raw_role(leg, role, m, s):
    return CACHE / '_raw' / 'roli' / f'{leg}__{role}_{m}_st{s:02d}.parquet'


def job_role(t):
    leg, role, m, s = t
    v = fit_role(leg, role, m, s)
    put(v, raw_role(leg, role, m, s))
    return leg, role, m, s, float(v.sec.iloc[0])


def ryad_role(leg, role, m):
    return f'{leg}__{role}_{m}'


def close_role(leg, role):
    spec = Y['roles'][role]
    fr = {}
    for m in spec['models']:
        V = pd.concat([pd.read_parquet(raw_role(leg, role, m, s)) for s in STEPS], ignore_index=True).sort_values(['step', 'date', 'ticker'])
        fr[m] = V[COLS].reset_index(drop=True)
        c = spec.get('control', {})
        if c.get('leg') == leg:
            S = pd.read_parquet(R / c['file'].format(m=m))
            S = S[S.step.isin(STEPS)]
            S = S[S.seed == 42] if 'seed' in S else S
            S = S[S['rank'] == 1] if 'rank' in S else S
            mm = fr[m][KEY + ['rv_pred']].merge(S[KEY + ['rv_pred']], on=KEY, how='outer', suffixes=('', '_saved'), indicator=True)
            b = mm[mm._merge == 'both']
            d = float(np.abs(np.log(b.rv_pred) - np.log(b.rv_pred_saved)).max()) if len(b) else np.inf
            ok = bool(len(b) == len(mm) and d <= TOL)
            append_csv(CACHE / 'control_B1.csv', [dict(ryad=ryad_role(leg, role, m), seed=42, kind='role', saved=c['file'].format(m=m), rows=len(mm),
                                                       matched=len(b), max_abs_dln=d, ok=ok, time=time.strftime('%F %T'))])
            log(f'контроль {ryad_role(leg, role, m)}: строк {len(b)}/{len(mm)}, max|Δ ln| = {d:.2e} {"OK" if ok else "ПРОВАЛ"}')
            if not ok:
                STOP.write_text(f'{time.strftime("%F %T")} контроль роли не пройден: {leg} {role} {m} {d}\n')
                raise SystemExit(f'СТОП: {ryad_role(leg, role, m)} не повторяет первый раунд')
    if Y.get('mean_variant') and len(fr) == 2:
        a, b = fr['xgb'], fr['lgbm']
        mm = a.merge(b[KEY + ['rv_pred']], on=KEY, suffixes=('_x', '_l'), validate='one_to_one')
        assert len(mm) == len(a) == len(b)
        fr['mean'] = mm.assign(rv_pred=(mm.rv_pred_x + mm.rv_pred_l) / 2)[COLS]
    for m, df in fr.items():
        p = fpath(ryad_role(leg, role, m), 42)
        put(df, p)
        index_add(p, ryad_role(leg, role, m), 42, df, f'роль {role}, нога {leg}, модель {m}, точки {spec["points"]}, seed 42; без поправки s²')
    log(f'роль {role} × нога {leg}: записано {", ".join(fr)}')


def legs_available():
    d = ROOT / Y['legs_dir']
    return sorted(p.name[:-len(f'_h{H}_lnall.parquet')] for p in d.glob(f'*_h{H}_lnall.parquet')) if d.exists() else []


def cmd_roli(a):
    import multiprocessing as mp
    if STOP.exists():
        raise SystemExit(f'СТОП-файл {STOP}: {STOP.read_text()}')
    legs = a.legs or (['HARJ_l2', 'HARX_d17'] + legs_available())
    roles = a.roles or list(Y['roles'])
    setup_roli()
    for leg in legs:
        if leg not in ('HARJ_l2',):
            assert leg_file(leg).exists(), leg_file(leg)
    units = [(leg, role) for leg in legs for role in roles]
    tasks = [(leg, role, m, s) for leg, role in units for m in Y['roles'][role]['models'] for s in sorted(STEPS, reverse=True)
             if not raw_role(leg, role, m, s).exists()]
    closed = {u for u in units if all(fpath(ryad_role(u[0], u[1], m), 42).exists() for m in Y['roles'][u[1]]['models'])}
    ready = lambda u: all(raw_role(u[0], u[1], m, s).exists() for m in Y['roles'][u[1]]['models'] for s in STEPS)
    for u in units:
        if u not in closed and ready(u):
            close_role(*u); closed.add(u)
    log(f'роли: ног {len(legs)}, ролей {len(roles)}, задач {len(tasks)}, воркеров {a.workers}')
    t0, n = time.time(), 0
    if tasks:
        with mp.get_context('fork').Pool(a.workers) as P:
            for leg, role, m, s, sec in P.imap_unordered(job_role, tasks, chunksize=1):
                n += 1
                el = time.time() - t0
                if n % 10 == 0 or n == len(tasks):
                    log(f'{n}/{len(tasks)} (последняя: {leg} {role} {m} шаг {s}, {sec:.0f} с), прошло {el / 60:.1f} мин, осталось ~{el / n * (len(tasks) - n) / 60:.0f} мин')
                u = (leg, role)
                if u not in closed and ready(u):
                    close_role(*u); closed.add(u)
    log(f'роли: готово, единиц {len(closed)}/{len(units)}, стена {(time.time() - t0) / 60:.1f} мин')


def cmd_leak_roli(a):
    """Детектор на коде ролей: шум в матрице и в ln-прогнозе ноги с даты t0; t0 = 1 января тестового года шага -> val равны, t0 = 1 января val-года -> изменились."""
    setup_roli()
    L = RS['L']
    s = int(Y['leak_role_step'])
    ty = L.TEST_YEARS[s - 1]
    leg = a.leg
    A0 = RS['A']
    rows = []
    for role in Y['roles']:
        m = Y['roles'][role]['models'][0]
        _BASE.clear(); _RREF.clear()
        base = fit_role(leg, role, m, s)
        for t0, must in ((f'{ty}-01-01', 'равны'), (f'{ty - 1}-01-01', 'изменились')):
            RS['A'] = corrupt(A0, t0)
            _BASE.clear(); _RREF.clear()
            B = base_of(leg)
            if B is not None:
                rng = np.random.default_rng(11)
                for k, g in B.items():
                    msk = (pd.to_datetime(g['date']) >= pd.Timestamp(t0)).values
                    g = g.copy()
                    g.loc[msk, 'base_ln'] = g.loc[msk, 'base_ln'].values + rng.normal(size=int(msk.sum()))
                    B[k] = g
            v = fit_role(leg, role, m, s)
            RS['A'] = A0
            _BASE.clear(); _RREF.clear()
            d = float(np.abs(np.log(v.rv_pred.values) - np.log(base.rv_pred.values)).max()) if len(v) == len(base) else np.inf
            ok = bool(d == 0.0) if must == 'равны' else bool(d > 0)
            rows.append(dict(ryad=ryad_role(leg, role, m), step=s, noise_from=t0, expect=must, max_abs_dln=d, ok=ok, time=time.strftime('%F %T')))
            log(f'leak {ryad_role(leg, role, m)} шаг {s}: шум с {t0} -> max|Δ ln| val = {d:.3g} (ожидается: {must}) {"OK" if ok else "ПРОВАЛ"}')
    append_csv(CACHE / 'leak_B1.csv', rows)
    if not all(r['ok'] for r in rows):
        STOP.write_text(f'{time.strftime("%F %T")} детектор ролей: провал {leg}\n')
        raise SystemExit('СТОП: детектор ролей не пройден')


# ---------------------------------------------------------------- детектор «будущее не влияет»
def corrupt(A, t0, seed=7):
    """Всё с даты t0 включительно: числовые столбцы × exp(N(0,1)), цели перемешаны (как d24_1.corrupt)."""
    rng = np.random.default_rng(seed)
    A = A.copy()
    msk = (pd.to_datetime(A['date']) >= pd.Timestamp(t0)).values
    for c in [c for c in A.columns if c not in ('date', 'ticker', 'year') and pd.api.types.is_float_dtype(A[c])]:
        v = A.loc[msk, c].values
        A.loc[msk, c] = rng.permutation(v) if 'target' in c else v * np.exp(rng.normal(size=len(v)))
    return A


def cmd_leak(a):
    setup_f1()
    s = int(Y['F1']['leak_step'])
    ty = G['L'].TEST_YEARS[s - 1]
    A0, item = G['A'], G['P'][s][:1]
    base = fit_unit(s, 42, item)
    rows = []
    for t0, must in ((f'{ty}-01-01', 'равны'), (f'{ty - 1}-01-01', 'изменились')):
        G['A'] = corrupt(A0, t0)
        _DC.clear(); _RC.clear()
        v = fit_unit(s, 42, item)
        G['A'] = A0
        _DC.clear(); _RC.clear()
        d = float(np.abs(np.log(v.rv_pred.values) - np.log(base.rv_pred.values)).max()) if len(v) == len(base) else np.inf
        ok = bool(d == 0.0) if must == 'равны' else bool(d > 0)
        rows.append(dict(ryad='D22_h1_xgb (F_1)', step=s, noise_from=t0, expect=must, max_abs_dln=d, ok=ok, time=time.strftime('%F %T')))
        log(f'leak шаг {s}: шум с {t0} -> max|Δ ln| val = {d:.3g} (ожидается: {must}) {"OK" if ok else "ПРОВАЛ"}')
    append_csv(CACHE / 'leak_B1.csv', rows)
    if not all(r['ok'] for r in rows):
        STOP.write_text(f'{time.strftime("%F %T")} детектор: провал\n')
        raise SystemExit('СТОП: детектор не пройден')


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('f1'); p.add_argument('--workers', type=int, default=int(Y['workers'])); p.add_argument('--seeds', type=int, nargs='*')
    sp.add_parser('leak')
    p = sp.add_parser('roli'); p.add_argument('--workers', type=int, default=int(Y['workers'])); p.add_argument('--legs', nargs='*')
    p.add_argument('--roles', nargs='*')
    p = sp.add_parser('leak_roli'); p.add_argument('--leg', default='HARX_d17')
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    {'f1': cmd_f1, 'leak': cmd_leak, 'roli': cmd_roli, 'leak_roli': cmd_leak_roli}[a.cmd](a)


if __name__ == '__main__':
    main()
