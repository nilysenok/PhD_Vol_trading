#!/usr/bin/env python3
"""R2 v2, агент Б2 (протокол docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md, строка «Б2»): роли бустинга для ног уровня 1 на h=5/22
и F_5 = D12_h5 wrap2 на seed 42–51. yaml — models/designs/r2/r2_B2_roli_h5_h22.yaml (таблица «роль × h -> точки», записана до счёта).

Обучение — код первого раунда без изменений: l2_lib.step_data / make_ref / fit (бины окна шага 1, ES на val-годе, patience 30),
обёртка d24_1 (как Д24.3), для офсета gamma — d17_1 (b01: init_score = ln ноги + s²/2, клип метки). Нога подставляется вместо
ln HAR-J подменой l2_lib.har_ln (как d16_v9.py): ln ноги на строках шага (tr — внутри выборки, va — OOS, te — может отсутствовать).
Только val шагов 1–6, seed 42 (F_5 — seed 42–51). Метрик нет: код не считает QLIKE; контроль — max |Δ ln rv_pred| на шагах 1–6.
Тест и val шагов 7–10 не пишутся (после заморозки — отдельно, только для новых моделей победителей).

python3 scripts/analysis/r2_roli_h5_h22.py f5 --workers 5                    # F_5: контроль переобучением + запись из сохранённых
python3 scripts/analysis/r2_roli_h5_h22.py kontrol --role ost --h 5          # движок роли на известных ногах (HAR-J, HAR-X)
python3 scripts/analysis/r2_roli_h5_h22.py leak --role ost --h 5             # детектор «будущее не влияет»
python3 scripts/analysis/r2_roli_h5_h22.py role --role ost --h 5 --leg HARXg7 --file data/regen/R2/cache/E/....parquet
python3 scripts/analysis/r2_roli_h5_h22.py index"""
import argparse, hashlib, io, multiprocessing as mp, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
YAML = ROOT / 'models' / 'designs' / 'r2' / 'r2_B2_roli_h5_h22.yaml'
REGEN = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')
CACHE = REGEN / 'R2' / 'cache' / 'B2'
INDEX = REGEN / 'R2' / 'cache' / 'INDEX_B2.csv'
os.environ.setdefault('L2_FEAT', str(REGEN / 'features_pit_v3.parquet'))
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import yaml

K = ['date', 'ticker', 'step']
COLS = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
DEV = 6
Y = yaml.safe_load(open(YAML))
G = {}
_REF = {}


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(Path(p).read_bytes())


def now():
    return time.strftime('%F %T')


def log(*a):
    print(now(), *a, flush=True)


# ---------------------------------------------------------------- настройка (патчи — в процессе)
def setup(spec, gamma=False):
    """spec: h, cfg, matrix, blocks, points, models. Подмена l2_lib.har_ln: G['legtab'] = None -> HAR-J (как в первом раунде)."""
    import d24_1 as X
    if gamma:
        import d17_1
        assert d17_1.BASE_KIND == 'harj', 'офсет: основа подставляется через har_ln (D17_BASE не задавать)'
    import l2_lib as L
    _har0 = L.har_ln

    def har_ln_leg(tp, parts, h, hc, w=None, huber=False, y=None):
        T = G.get('legtab')
        if T is None:
            return _har0(tp, parts, h, hc, w, huber, y)
        k = int(tp['year'].max()) + 1 - 2015  # шаг по последнему году обучения
        out = []
        for i, d in enumerate(parts):  # step_data: [tr, va, te]
            part = ('tr', 'va', 'te')[i]
            if 'wide' in T:  # формат Э: date, ticker, ln_s1..ln_s6 (ln модели шага k на всех строках)
                c = f'ln_s{k}'
                v = d[['date', 'ticker']].merge(T['wide'][['date', 'ticker', c]], on=['date', 'ticker'], how='left')[c].values \
                    if c in T['wide'] else np.full(len(d), np.nan)
            else:
                t = T.get((k, part))
                v = d[['date', 'ticker']].merge(t, on=['date', 'ticker'], how='left')['ln'].values if t is not None else np.full(len(d), np.nan)
            if part == 'te':
                v = np.where(np.isnan(v), 0.0, v)  # тест не пишется и не используется
            assert not np.isnan(v).any(), f'нога: нет ln на строках шага {k}, часть {part}'
            out.append(v)
        return out

    L.har_ln = har_ln_leg
    n = X.load_cfg(spec['h'], dict(spec['cfg']), spec['matrix'], spec['blocks'])
    G.update(L=L, X=X, A=X.G['A'], cols=X.G['cols'], cfg=dict(spec['cfg']), h=spec['h'], spec=spec)
    return n


def load_leg(path):
    """-> {(step, part): DataFrame[date, ticker, ln]} (формат D17/harx_base_h{h}.parquet: step, part, base_ln | ln)
    или {'wide': DataFrame[date, ticker, ln_s1..ln_s6]} (формат Э: R2/cache/E/lnall/<нога>_h{h}_lnall.parquet)."""
    d = pd.read_parquet(path)
    if 'ln_s1' in d:
        d['date'] = pd.to_datetime(d['date'])
        return {'wide': d[['date', 'ticker'] + [f'ln_s{k}' for k in range(1, DEV + 1) if f'ln_s{k}' in d]]}
    if 'ln' not in d:
        d = d.rename(columns={'base_ln': 'ln'})
    d = d[d.step <= DEV]
    return {(int(s), p): g[['date', 'ticker', 'ln']].reset_index(drop=True) for (s, p), g in d.groupby(['step', 'part'])}


def points(spec):
    """-> {(m, step): params}, шаги 1–6, точка ранга 1."""
    X, h, src = G['X'], spec['h'], spec['points']
    if src == 'Bv2b':
        P = X.points('Bv2b', h, range(1, DEV + 1))
        return {k: v[0][0] for k, v in P.items() if k[0] in spec['models']}
    S = pd.read_csv(REGEN / f'{src}_selected_h{h}.csv')
    if 'seed' in S:
        S = S[S.seed == 42]
    return {(m, s): X.cast(S[(S.model == m) & (S.step == s)].iloc[0], m) for m in spec['models'] for s in range(1, DEV + 1)}


# ---------------------------------------------------------------- единица обучения
def unit(a):
    m, s, items = a  # items: [(params, seed)]
    L, h, cfg = G['L'], G['h'], G['cfg']
    D = L.step_data(G['A'], G['cols'], h, s, m, cfg, 'Bv2b')
    if s > 1 and m not in _REF:
        _REF[m] = L.make_ref(m, G['A'], G['cols'], h, cfg, 'Bv2b')
    res = []
    for p, seed in items:
        pv, _, bi, _, _ = L.fit(m, p, D, cfg, seed=seed, ref=_REF.get(m) if s > 1 else None)
        d = D['parts']['va']
        res.append((m, s, seed, bi, pd.DataFrame({'date': d['date'].values, 'ticker': d['ticker'].values, 'year': D['vy'], 'step': s, 'model': m,
                                                  'seed': seed, 'rv_actual': d[L.tcol(h)].values, 'rv_pred': np.clip(np.exp(pv), 1e-10, None)})))
    return res


def fit_all(P, seeds, workers, chunk=2):
    U = []
    for (m, s), p in P.items():
        items = [(p, sd) for sd in seeds]
        U += [(m, s, items[i:i + chunk]) for i in range(0, len(items), chunk)]
    U.sort(key=lambda u: (-u[1], u[0]))
    _REF.clear()
    if workers <= 1:
        res = [x for u in U for x in unit(u)]
    else:
        with mp.get_context('fork').Pool(workers) as pool:
            res = [x for r in pool.imap_unordered(unit, U, chunksize=1) for x in r]
    _REF.clear()
    V = pd.concat([x[4] for x in res], ignore_index=True)
    tr = pd.DataFrame([dict(model=x[0], step=x[1], seed=x[2], best_iteration=x[3]) for x in res])
    return V, tr


def compare(mine, ref):
    """mine: [date, ticker, step, rv_pred]; ref: [date, ticker, step, rv_pred] (шаги 1–6) -> (строк моих, сохр., общих, max |Δ ln|)."""
    a, b = mine[mine.step <= DEV], ref[ref.step <= DEV]
    mm = a[K + ['rv_pred']].merge(b[K + ['rv_pred']], on=K, suffixes=('_m', '_s'))
    d = float(np.abs(np.log(mm.rv_pred_m) - np.log(mm.rv_pred_s)).max()) if len(mm) else np.inf
    return len(a), len(b), len(mm), d


def pick(d, seed=None, model=None):
    if seed is not None and 'seed' in d:
        d = d[d.seed == seed]
    if model is not None and 'model' in d:
        d = d[d.model == model]
    if 'rank' in d:
        d = d[d['rank'] == 1]
    return d[d.step <= DEV]


# ---------------------------------------------------------------- запись val (шаги 1–6)
def write(df, name, h, seed, idx):
    df = df[COLS].sort_values(['step', 'date', 'ticker']).reset_index(drop=True)
    assert df.step.between(1, DEV).all() and (df.year <= 2021).all()
    b = io.BytesIO(); df.to_parquet(b, index=False); b = b.getvalue()
    p = CACHE / f'{name}_h{h}_s{seed}_val.parquet'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b)
    idx.append(dict(fayl=str(p.relative_to(REGEN)), ryad=name, h=h, seed=seed, shagi='1-6', stroki=len(df), sha256=sha_bytes(b),
                    sha_koda=sha_file(__file__)[:16], yaml=f'{YAML.relative_to(ROOT)}@{sha_file(YAML)[:16]}', vremya=now()))


def level_mean(a, b):
    m = a.merge(b[K + ['rv_pred']], on=K, suffixes=('_a', '_b'), validate='one_to_one')
    return m.assign(rv_pred=(m.rv_pred_a + m.rv_pred_b) / 2).drop(columns=['rv_pred_a', 'rv_pred_b'])


def save_index(idx, tag):
    (CACHE / '_index').mkdir(parents=True, exist_ok=True)
    p = CACHE / '_index' / f'{tag}.csv'
    pd.DataFrame(idx).to_csv(p, index=False)


def save_control(rows, tag):
    (CACHE / '_control').mkdir(parents=True, exist_ok=True)
    C = pd.DataFrame(rows)
    C.to_csv(CACHE / '_control' / f'{tag}.csv', index=False)
    print(C.to_string(index=False), flush=True)
    return C


# ---------------------------------------------------------------- команды
def cmd_f5(a):
    t0 = time.time()
    sp = Y['f5']
    n = setup(sp)
    seeds = list(Y['seeds_f5'])
    P = points(sp)
    log(f'F_5: D12_h5, признаков {n}, обучений {len(P) * len(seeds)} (seed {seeds[0]}–{seeds[-1]}, шаги 1–6), воркеров {a.workers}')
    V, tr = fit_all(P, seeds, a.workers)
    CACHE.mkdir(parents=True, exist_ok=True)
    tr.to_csv(CACHE / '_trees_F5_D12_h5.csv', index=False)
    log(f'F_5: переобучение за {time.time() - t0:.0f} с')
    sv = sp['saved']
    S = {}
    for m in sp['models']:
        for sd in seeds:
            f = sv['s42'] if sd == 42 else (sv['s4344'] if sd in (43, 44) else sv['s4551'])
            S[(m, sd)] = pick(pd.read_parquet(REGEN / f.format(m=m)), seed=sd)
    rows = []
    for (m, sd), ref in S.items():
        nm, nr, nc, d = compare(pick(V, sd, m), ref)
        rows.append(dict(ryad=f'D12_h5_{m}', seed=sd, rows_mine=nm, rows_saved=nr, rows_common=nc, max_dln=d, ok=bool(nm == nr == nc and d <= Y['control_tol'])))
    w42 = level_mean(pick(V, 42, 'xgb'), pick(V, 42, 'lgbm'))
    nm, nr, nc, d = compare(w42, pick(pd.read_parquet(REGEN / sv['wrap2_s42'])))
    rows.append(dict(ryad='F5_D12_h5_wrap2 (переобучение) vs D12_h5_xgb_h5_wrap2_val', seed=42, rows_mine=nm, rows_saved=nr, rows_common=nc, max_dln=d,
                     ok=bool(nm == nr == nc and d <= Y['control_tol'])))
    C = save_control(rows, 'F5')
    if not C[C.seed == 42].ok.all():
        log('СТОП F_5: контроль seed 42 не пройден — ряды не записаны'); sys.exit(2)
    if not C.ok.all():
        log('внимание: расхождение на seed ≠ 42 — см. _control/F5.csv (ряды пишутся из сохранённых)')
    idx = []
    for sd in seeds:
        x, l = S[('xgb', sd)], S[('lgbm', sd)]
        write(x, 'F5_D12_h5_xgb', 5, sd, idx)
        write(l, 'F5_D12_h5_lgbm', 5, sd, idx)
        write(level_mean(x, l), 'F5_D12_h5_wrap2', 5, sd, idx)
    save_index(idx, 'F5')
    log(f'F_5: записано {len(idx)} файлов за {time.time() - t0:.0f} с')


def role_spec(role, h):
    return Y['roles'][role][h]


def cmd_kontrol(a):
    """Движок роли на известных ногах: HAR-J (без подмены) и HAR-X (D17/harx_base) против сохранённых рядов первого раунда, seed 42."""
    sp = role_spec(a.role, a.h)
    gam = a.role == 'gam'
    setup(sp, gamma=gam)
    h = a.h
    P = points(sp)
    checks = []
    if a.role == 'priz':
        checks.append(('V0 (cfg {})', None, {}, {m: f'Bv3_{m}_h{h}_val.parquet' for m in sp['models']}))
    if a.role == 'ost':
        ref = {5: 'D12_h5', 22: 'D12_h22'}[h]
        checks.append(('нога HAR-J', None, None, {m: f'{ref}_{m}_h{h}_val.parquet' for m in sp['models']}))
        if h == 5:
            checks.append(('нога HAR-X', f'D17/harx_base_h{h}.parquet', None, {m: ('D16_h5_D12x_val.parquet', m) for m in sp['models']}))
    if a.role == 'gam':
        ref = {5: 'D17_1_h5_G5Xopt', 22: 'D17_1_h22_G22Xopt'}[h]
        checks.append(('нога HAR-X', f'D17/harx_base_h{h}.parquet', None, {m: f'{ref}_{m}_h{h}_val.parquet' for m in sp['models']}))
    rows = []
    cfg0 = G['cfg']
    for label, legf, cfg, refs in checks:
        G['legtab'] = load_leg(REGEN / legf) if legf else None
        G['cfg'] = cfg0 if cfg is None else cfg
        V, _ = fit_all(P, [42], a.workers)
        for m, f in refs.items():
            fm = f if isinstance(f, str) else f[0]
            ref = pick(pd.read_parquet(REGEN / fm), seed=42, model=m)
            nm, nr, nc, d = compare(pick(V, 42, m), ref)
            rows.append(dict(role=a.role, h=h, check=label, model=m, ref=fm, rows_mine=nm, rows_saved=nr, rows_common=nc, max_dln=d,
                             ok=bool(nm == nr == nc and d <= Y['control_tol'])))
    G['cfg'], G['legtab'] = cfg0, None
    C = save_control(rows, f'kontrol_{a.role}_h{h}')
    if not C.ok.all():
        log(f'СТОП: контроль движка роли {a.role} h={h} не пройден'); sys.exit(2)


def cmd_leak(a):
    """Детектор: модель шага (seed 42) обучена на годах < val-года. Шум (входы × exp(N), цели перемешаны) с 1 января тестового года ->
    val-прогнозы равны побайтно; с 1 января val-года -> изменились (подложенная утечка поймана). Нога — HAR-X (D17/harx_base): её таблица
    не портится (детектор ног — у агента Э), проверяется путь подмены и бустинг роли."""
    sp = role_spec(a.role, a.h)
    setup(sp, gamma=a.role == 'gam')
    X, L = G['X'], G['L']
    st = Y['leak']['step']
    G['legtab'] = load_leg(REGEN / f'D17/harx_base_h{a.h}.parquet')
    P = points(sp)
    ty = L.TEST_YEARS[st - 1]
    A0 = G['A']
    rows = []
    for m in sp['models']:
        base = unit((m, st, [(P[(m, st)], 42)]))[0][4]
        for t0, must in ((f'{ty}-01-01', 'равны'), (f'{ty - 1}-01-01', 'изменились')):
            G['A'] = X.corrupt(A0, t0); _REF.clear()
            v = unit((m, st, [(P[(m, st)], 42)]))[0][4]
            G['A'] = A0; _REF.clear()
            d = float(np.abs(np.log(v.rv_pred.values) - np.log(base.rv_pred.values)).max()) if len(v) == len(base) else np.inf
            ok = (d == 0.0) if must == 'равны' else (d > 0)
            rows.append(dict(role=a.role, h=a.h, model=m, step=st, noise_from=t0, expect=must, max_abs_dln=d, ok=bool(ok), time=now()))
            log(f'leak {a.role} h={a.h} {m} шаг {st}: шум с {t0} -> max|Δ ln| = {d:.3g} (ожидается: {must}) {"OK" if ok else "ПРОВАЛ"}')
    p = CACHE / '_leak.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if p.exists():
        df = pd.concat([pd.read_csv(p), df], ignore_index=True)
    df.to_csv(p, index=False)
    assert all(r['ok'] for r in rows), 'детектор: провал'


def cmd_roles(a):
    """Все ноги Э на этом h (R2/cache/E/lnall/<нога>_h{h}_lnall.parquet) — одна загрузка матрицы на роль."""
    t0 = time.time()
    sp = role_spec(a.role, a.h)
    setup(sp, gamma=a.role == 'gam')
    src = REGEN / 'R2' / 'cache' / 'E' / 'lnall'
    legs = sorted(f.name[:-len(f'_h{a.h}_lnall.parquet')] for f in src.glob(f'*_h{a.h}_lnall.parquet'))
    if a.legs:
        legs = [x for x in legs if x in set(a.legs)]
    log(f'роль {a.role} h={a.h}: ног {len(legs)}: {" ".join(legs)}')
    for leg in legs:
        f = src / f'{leg}_h{a.h}_lnall.parquet'
        one(a.role, a.h, leg, f, sp, a.workers)
    log(f'роль {a.role} h={a.h}: готово за {time.time() - t0:.0f} с')


def one(role, h, leg, f, sp, workers):
    t0 = time.time()
    G['legtab'] = load_leg(f)
    P = points(sp)
    V, tr = fit_all(P, [42], workers)
    (CACHE / '_trees').mkdir(parents=True, exist_ok=True)
    tr.assign(role=role, leg=leg, h=h).to_csv(CACHE / '_trees' / f'{role}_{leg}_h{h}.csv', index=False)
    idx = []
    fr = {m: pick(V, 42, m) for m in sp['models']}
    for m, d in fr.items():
        write(d, f'{role}_{leg}_{m}', h, 42, idx)
    if len(fr) == 2:
        write(level_mean(fr['xgb'], fr['lgbm']), f'{role}_{leg}_mean', h, 42, idx)
    for r in idx:
        r['vhod'] = f'{Path(f).relative_to(REGEN)}@{sha_file(f)[:16]}'
    save_index(idx, f'{role}_{leg}_h{h}')
    log(f'  {role} {leg} h={h}: {len(idx)} файлов за {time.time() - t0:.0f} с')


def cmd_role(a):
    t0 = time.time()
    sp = role_spec(a.role, a.h)
    setup(sp, gamma=a.role == 'gam')
    f = Path(a.file) if Path(a.file).is_absolute() else REGEN.parent.parent / a.file
    one(a.role, a.h, a.leg, f, sp, a.workers)
    log(f'роль {a.role} нога {a.leg} h={a.h}: {time.time() - t0:.0f} с')


def cmd_index(a):
    fs = sorted((CACHE / '_index').glob('*.csv'))
    I = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).sort_values(['h', 'ryad', 'seed'])
    I.to_csv(INDEX, index=False)
    log(f'INDEX_B2: {len(I)} файлов, рядов {I.ryad.nunique()}')


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('f5'); p.add_argument('--workers', type=int, default=Y['workers'])
    for nm in ('kontrol', 'leak', 'role', 'roles'):
        p = sp.add_parser(nm)
        p.add_argument('--role', required=True, choices=list(Y['roles'])); p.add_argument('--h', type=int, required=True, choices=[5, 22])
        p.add_argument('--workers', type=int, default=Y['workers'])
        if nm == 'role':
            p.add_argument('--leg', required=True); p.add_argument('--file', required=True)
        if nm == 'roles':
            p.add_argument('--legs', nargs='*', default=None)
    sp.add_parser('index')
    a = ap.parse_args()
    {'f5': cmd_f5, 'kontrol': cmd_kontrol, 'leak': cmd_leak, 'role': cmd_role, 'roles': cmd_roles, 'index': cmd_index}[a.cmd](a)


if __name__ == '__main__':
    main()
