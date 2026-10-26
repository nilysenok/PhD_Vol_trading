#!/usr/bin/env python3
"""Д26 — круг h=5 на разработке (владелец 26.10, 09:35; агент уровня 2). Отбор — OOS 2017–2020 (шаги 2–5), проверка — 2021 (шаг 6),
c1–c4 — gibrid_v17 на 2017–2021 (эталон с той же s²). Эталон — HAR-J с поправкой s² sm. Б0.1 — s² шага t−1.
Метрики за годы ≥ 2022 не считаются: в оценку и в гибрид идут только строки шагов 1–6 (d26_lib.dev_only).

  --step0          сводка уже посчитанного (D12_h5, G5Xopt, Д16.1 = V9 на HAR-X, итерированный hyb₁ Д22 и его средние, HAR-X⁺₅) по seed
  --d261 ...       Д26.1 (d26_direct.py)    --d262 / --d263 (d26_iter.py)    --d264 / --d265 / --d266 — здесь
Выход: data/regen/D26/ (csv, входы и выходы гибрида), отчёт — docs/otchety/2026-10-26_d26_h5.md.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d26_lib as L

R, OUT = L.R, L.OUT
SEEDS = (42, 43, 44)
WORKERS = 4  # не больше 4 процессов (поручение свежему агенту А)


def now():
    return time.strftime('%H:%M:%S')


def seed_frames(prefix, m, seeds=SEEDS, single_name=None, seeds_name=None):
    """{seed: val-кадр модели m} из пары файлов «seed 42» + «_seeds4344» (форматы tune_L2 / d16_v9)."""
    a = pd.read_parquet(R / (single_name or f'{prefix}_{m}_h5_val.parquet'))
    b = pd.read_parquet(R / (seeds_name or f'{prefix}_{m}_h5_seeds4344_val.parquet'))
    if 'model' in a:
        a, b = a[a.model == m], b[b.model == m]
    out = {42: a}
    for s in seeds:
        if s != 42:
            out[s] = b[b.seed == s]
    return {s: L.dev_only(d) for s, d in out.items() if s in seeds}


def lnmean(*dfs):
    """Среднее ln-прогнозов (как hyb₁ Д22: среднее ln XGB и ln LGBM)."""
    base = dfs[0][['date', 'ticker', 'year', 'step', 'rv_actual']].copy()
    ln = None
    for d in dfs:
        m = base[['date', 'ticker', 'step']].merge(d[['date', 'ticker', 'step', 'rv_pred']], on=['date', 'ticker', 'step'], how='left')
        assert m.rv_pred.notna().all()
        ln = np.log(m.rv_pred.values) if ln is None else ln + np.log(m.rv_pred.values)
    base['rv_pred'] = np.exp(ln / len(dfs))
    return base


def per_seed_gib(name, slots_by_seed, legs=('single', 'wrap')):
    """slots_by_seed: {seed: (кадр XGB-слота, кадр LGB-слота)} -> {seed: summarize}."""
    from concurrent.futures import ThreadPoolExecutor
    for s, (x, l) in slots_by_seed.items():
        L.write_boost(f'{name}_s{s}', x, l)
    with ThreadPoolExecutor(min(WORKERS, len(slots_by_seed))) as ex:
        paths = dict(zip(slots_by_seed, ex.map(lambda s: L.run_gibrid(f'{name}_s{s}', legs), list(slots_by_seed))))
    S = {s: L.summarize(p) for s, p in paths.items()}
    for s in S:
        print(f'  {now()} гибрид {name} seed {s}: лучший {S[s].iloc[0].design} {S[s].iloc[0].m1720:.4f} / {S[s].iloc[0].q2021:.4f}', flush=True)
    return S


def design_rows(name, S, design=None, src=''):
    """Строка таблицы: дизайн (по правилу на средних по seed или заданный), по seed, условие входа к D12_h5 (фикс. 0,2159)."""
    T, best = L.seeds_table(S)
    import re
    T.to_csv(OUT / ('seeds_' + re.sub(r'[^0-9A-Za-zА-Яа-я_.+-]+', '_', name) + '.csv'), index=False)
    b = T.set_index('design').loc[design] if design else best
    dn = design or best.design
    qs = [b[f'm1720_s{s}'] for s in S]
    e = L.entry(qs)
    return dict(name=name, design=dn, n_failed=int(b.n_failed), failed=b.failed, m1720=b.m1720, q2021=b.q2021,
                **{f'm1720_s{s}': b[f'm1720_s{s}'] for s in S}, **{f'q2021_s{s}': b[f'q2021_s{s}'] for s in S},
                **{str(y): b[str(y)] for y in range(2017, 2022)}, rel_D12=e['rel'], d_mean=e['d_mean'], sigma=e['sigma'], t=e['t'],
                pass3=e['pass3'], pass2s=e['pass2s'], src=src)


# ---------------------------------------------------------------- шаг 0
def step0():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    # D12_h5: финал (wrap2, фикс.) и по seed (single, wrap: среднее XGB и LGBM)
    xs, ls = seed_frames('D12_h5', 'xgb'), seed_frames('D12_h5', 'lgbm')
    S = per_seed_gib('D12_h5', {s: (xs[s], ls[s]) for s in SEEDS})
    rows.append(design_rows('D12_h5 по seed (single|V0_XGB|B-sm)', S, 'single|V0_XGB|B-sm|L0', 'final2 / final3 ranking_h5'))
    rows.append(design_rows('D12_h5 по seed (лучший дизайн по seed)', S, None, 'то же'))
    D12S = {s: lnmean(xs[s], ls[s]) for s in SEEDS}  # среднее XGB и LGBM seed s — компонента Д26.5
    # G5Xopt (Д17.1, GLM с офсетом на HAR-X)
    xs, ls = seed_frames('D17_1_h5_G5Xopt', 'xgb'), seed_frames('D17_1_h5_G5Xopt', 'lgbm')
    S = per_seed_gib('G5Xopt', {s: (xs[s], ls[s]) for s in SEEDS})
    rows.append(design_rows('D17_1_h5_G5Xopt по seed', S, None, 'PR #75, 2026-10-26_d17.md; Д24.3 (PR #94)'))
    # Д16.1 = V9 на HAR-X на точках D12_h5 (это и есть Д26.4 в толковании «основа V9 = HAR-X»)
    a, b = pd.read_parquet(R / 'D16_h5_D12x_val.parquet'), pd.read_parquet(R / 'D16_h5_D12x_s4344_val.parquet')
    ab = pd.concat([a, b])
    sl = {s: (L.dev_only(ab[(ab.seed == s) & (ab.model == 'xgb')]), L.dev_only(ab[(ab.seed == s) & (ab.model == 'lgbm')])) for s in SEEDS}
    S = per_seed_gib('D16_1_h5_D12x', sl)
    rows.append(design_rows('Д16.1 V9 на HAR-X (точки D12_h5) по seed', S, None, '2026-10-26_d16.md (h5_D12x)'))
    # Д22: итерированный hyb₁ (x6), HAR-X⁺₅ (x1) и средние
    import d22_full as F
    A, hist = F.matrix()
    I = F.load_inputs('val')
    sl6, slx = {}, {}
    for s in SEEDS:
        d = L.dev_only(F.assemble(5, 'val', s, I, A, hist))
        base = d[['date', 'ticker', 'year', 'step', 'rv_actual']]
        x6 = base.assign(rv_pred=np.exp(d.x6.values))
        x1 = base.assign(rv_pred=np.exp(d.x1.values))
        avg61 = base.assign(rv_pred=np.exp((d.x6.values + d.x1.values) / 2))
        avg6d = lnmean(x6, D12S[s])
        x6.to_parquet(OUT / f'x6_h5_s{s}_val.parquet', index=False)
        if s == 42:
            rows.append(dict(name='HAR-X⁺₅ (слой 1 Д22, x1), s² sm', design='одиночная', n_failed=np.nan,
                             **{k: v for k, v in L.row('', L.years(x1, 'sm')).items() if k != 'design'}, src='2026-10-26_d22.md, PR #89'))
        sl6[s] = (x6, avg61)
        slx[s] = (x6, avg6d)
    S = per_seed_gib('D22_x6', sl6)
    rows.append(design_rows('итерированный hyb₁ (x6) / среднее с HAR-X⁺₅', S, None, 'Д20 в слоях Д22 (d22_full.assemble)'))
    rows.append(design_rows('итерированный hyb₁ (x6) один', S, 'single|V0_XGB|B-sm|L0', 'то же'))
    rows.append(design_rows('среднее x6 и HAR-X⁺₅', S, 'single|V0_LGB|B-sm|L0', 'то же'))
    S = per_seed_gib('D22_x6d12', slx)
    rows.append(design_rows('среднее x6 и D12_h5 (seed s)', S, 'single|V0_LGB|B-sm|L0', 'то же'))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'step0.csv', index=False)
    pd.set_option('display.width', 250)
    print(T[['name', 'design', 'n_failed', 'failed', 'm1720', 'q2021', 'rel_D12', 'd_mean', 'sigma', 't']].to_string())


# ---------------------------------------------------------------- Д26.4: нога HAR-X вместо HAR-J
def gib_harx():
    """Запуск gibrid_v17 в этом процессе с подменой эконометрической ноги: HAR-J -> HAR-X (data/regen/HARX_h5_val.parquet, rv_pred без
    поправки; шаги 1–6). Аргументы gibrid_v17 — после «--gib-harx». Общие файлы не меняются (подмена в памяти)."""
    import gibrid_har as GH
    import gibrid_v17 as GV
    hx = L.dev_only(pd.read_parquet(R / 'HARX_h5_val.parquet'))[['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']]
    hx['date'] = pd.to_datetime(hx['date'])
    _orig = GH.har_forecasts

    def har_x(h, a, harj):
        assert h == 5
        v, t, s2 = _orig(h, a, harj)
        v = v[v.step <= L.DEV_MAX][['date', 'ticker', 'year', 'step', 'rv_actual']].merge(
            hx[['date', 'ticker', 'step', 'rv_pred']], on=['date', 'ticker', 'step'], how='inner')
        return v, t.iloc[:0], s2
    GH.har_forecasts = har_x
    i = sys.argv.index('--gib-harx')
    sys.argv = ['gibrid_v17.py'] + sys.argv[i + 1:]
    GV.main()


def run_gib_harx(boost, legs=('single', 'wrap')):
    import subprocess
    out = L.GOUT / boost
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-W', 'ignore', str(Path(__file__).resolve()), '--gib-harx', '--boost', boost, '--h', '5', '--features', str(L.FEAT_V3),
           '--regen', str(L.GIN), '--legs', *legs, '--out', str(out), '--devlog', str(out / 'devlog.md')]
    with open(out / 'run.log', 'w') as f:
        rc = subprocess.run(cmd, env=dict(os.environ, GIBRID_SAME_SLOT='1'), stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f'gibrid HAR-X {boost}: код {rc}')
    return out / f'designs_val_{boost}_h5.csv'


ETALON_YEARS = {2017: 0.1599, 2018: 0.2635, 2019: 0.2378, 2020: 0.3030, 2021: 0.1727}  # HAR-J с поправкой sm (d24b.md, final3)


def c1_harj(T):
    """c1 к эталону HAR-J с поправкой (в гибриде с ногой HAR-X c1 считается к HAR-X — строже)."""
    return T.apply(lambda r: all(r[str(y)] <= 1.02 * q for y, q in ETALON_YEARS.items()), axis=1)


def d264():
    from concurrent.futures import ThreadPoolExecutor
    xs, ls = seed_frames('D12_h5', 'xgb'), seed_frames('D12_h5', 'lgbm')
    for s in SEEDS:
        L.write_boost(f'D26_4_s{s}', xs[s], ls[s])
    for sfx in ('_wrap2',):
        L.write_boost('D26_4_s42', pd.read_parquet(R / f'D12_h5_xgb_h5{sfx}_val.parquet'), pd.read_parquet(R / f'D12_h5_lgbm_h5{sfx}_val.parquet'), sfx)
    legs = {42: ('single', 'wrap', 'wrap2')}
    with ThreadPoolExecutor(3) as ex:
        paths = dict(zip(SEEDS, ex.map(lambda s: run_gib_harx(f'D26_4_s{s}', legs.get(s, ('single', 'wrap'))), SEEDS)))
    S = {s: L.summarize(p) for s, p in paths.items()}
    S[42].to_csv(OUT / 'd26_4_s42_designs.csv', index=False)
    print('seed 42, лучшие 8 (нога HAR-X):'); print(S[42].head(8)[['design', 'm1720', 'q2021', 'failed']].to_string())
    S2 = {s: S[s][~S[s].design.str.startswith('wrap2')] for s in S}
    T, best = L.seeds_table(S2)
    T['c1_harj'] = c1_harj(T)
    T.to_csv(OUT / 'd26_4_seeds.csv', index=False)
    print(T.head(10)[['design', 'n_failed', 'failed', 'c1_harj', 'm1720', 'm1720_s42', 'm1720_s43', 'm1720_s44', 'q2021']].to_string())
    # дизайны, где нога HAR-X реально участвует (не V0_XGB/V0_LGB — они совпадают с D12_h5)
    mix = T[~T.design.str.contains(r'\|V0_')]
    print('лучший смешанный:'); print(mix.head(3)[['design', 'n_failed', 'failed', 'c1_harj', 'm1720', 'q2021']].to_string())


# ---------------------------------------------------------------- Д26.5: среднее и медиана компонент
def corrected(df, code):
    """Б0.1 вне выборки (s² шага t−1; шаг 1 — свои остатки, 2016 не оценивается). code None — без поправки (поправка внутри)."""
    d = L.dev_only(df)[['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']].copy()
    if code:
        F = L.bias_factors(d, code)
        d['rv_pred'] = d.rv_pred * d.step.map(F)
    return d.reset_index(drop=True)


def combine(frames, how):
    key = ['date', 'ticker', 'step']
    m = frames[0][key + ['year', 'rv_actual']].copy()
    P = []
    for d in frames:
        x = m[key].merge(d[key + ['rv_pred']], on=key, how='left').rv_pred.values
        assert not np.isnan(x).any()
        P.append(x)
    P = np.column_stack(P)
    m['rv_pred'] = P.mean(1) if how == 'mean' else np.median(P, 1)
    return m


def components(which):
    """{имя: {seed: скорректированный кадр}}. Ряды Д26.1/Д26.3 агентов Б и В — если появились (см. load_bc)."""
    C = {}
    xs, ls = seed_frames('D12_h5', 'xgb'), seed_frames('D12_h5', 'lgbm')
    C['D12_h5'] = {s: corrected(combine([xs[s], ls[s]], 'mean'), 'sm') for s in SEEDS}  # дизайн D12_h5: среднее XGB+LGBM, B-sm
    if 'G5Xopt' in which:
        g = seed_frames('D17_1_h5_G5Xopt', 'lgbm')
        C['G5Xopt'] = {s: corrected(g[s], None) for s in SEEDS}  # gamma с офсетом: поправка внутри (bez)
    if 'D16_1' in which:
        ab = pd.concat([pd.read_parquet(R / 'D16_h5_D12x_val.parquet'), pd.read_parquet(R / 'D16_h5_D12x_s4344_val.parquet')])
        C['D16_1'] = {s: corrected(combine([ab[(ab.seed == s) & (ab.model == m)] for m in ('xgb', 'lgbm')], 'mean'), 'sm') for s in SEEDS}
    for nm, fr in load_bc(which).items():
        C[nm] = fr
    return C


def load_bc(which):
    """Д26.1 (агент Б) и Д26.3 (агент В): {seed: кадр} из их файлов, когда появятся. Формат уточняется по факту (см. журнал)."""
    out = {}
    if 'D26_1' in which:  # агент Б: формат tune_L2 (data/regen/D26_1_{xgb,lgbm}_h5[_seeds4344]_val.parquet); рецепт — среднее ln XGB и ln LGBM
        out['D26_1'] = {s: corrected(r, 'sm') for s, r in d261_raw(os.environ.get('D261', 'D26_1')).items()}  # D261 — вариант агента Б
    if 'D26_3' in which:
        for s, r in d263_raw().items():
            out.setdefault('D26_3', {})[s] = corrected(r, D263_CODE)
    return out


# Д26.3 (агент В, d26_3.py): ŷ₅ = mean_j exp(ŷ_j), ŷ_j — hyb₁ лида j (среднее ln XGB и ln LGBM), лиды 1..5; варианты var = b (точки D22_h1
# без подбора) | a (переподбор по лиду); поправка S1 — одна s² sm на агрегат (D263_CODE = 'sm'), S2 — s² sm по лиду до агрегирования
# (уже поправлено -> D263_CODE = None). Источник: лиды 2..5 — data/regen/D26_3/D26_3{var}_L{j}_{xgb,lgbm}_val.parquet, лид 1 — D22_h1
# (как d26_3.lead_val). Выбор варианта: переменная окружения D263 = '{var}' или '{var}p' (p — S2), по умолчанию 'b'.
D263_VAR = os.environ.get('D263', 'b')
D263_CODE = None if D263_VAR.endswith('p') else 'sm'


def d261_raw(prefix='D26_1'):
    xs, ls = seed_frames(prefix, 'xgb'), seed_frames(prefix, 'lgbm')
    return {s: lnmean(xs[s], ls[s]) for s in SEEDS}


def b_frames(spec):
    """Ряды агента Б: 'D26_1{var}' (data/regen/D26_1{var}_{m}_h5[_seeds4344]_val.parquet) или 'D26_2:{boost}'
    (data/regen/D26_2/{boost}_{m}_h5[_seeds4344]_val.parquet). -> ({seed: ln-среднее XGB и LGBM}, {seed: [XGB, LGBM]})."""
    if spec.startswith('D26_2:'):
        b = spec.split(':', 1)[1]
        fr = {m: seed_frames(None, m, single_name=f'D26_2/{b}_{m}_h5_val.parquet', seeds_name=f'D26_2/{b}_{m}_h5_seeds4344_val.parquet')
              for m in ('xgb', 'lgbm')}
    else:
        fr = {m: seed_frames(spec, m) for m in ('xgb', 'lgbm')}
    return {s: lnmean(fr['xgb'][s], fr['lgbm'][s]) for s in SEEDS}, {s: [fr['xgb'][s], fr['lgbm'][s]] for s in SEEDS}


def d263_lead(var, m, j):
    if j == 1 and var in ('a', 'b'):
        a = pd.read_parquet(R / f'D22_h1_{m}_h1_val.parquet').assign(seed=42)
        b = pd.read_parquet(R / f'D22_h1_{m}_h1_seeds4344_val.parquet')
        d = pd.concat([a, b], ignore_index=True)
    else:
        d = pd.read_parquet(R / 'D26_3' / f'D26_3{var}_L{j}_{m}_val.parquet')
    d = L.dev_only(d[d.seed.isin(SEEDS)])
    return d[['date', 'ticker', 'year', 'step', 'seed', 'rv_actual', 'rv_pred']].drop_duplicates(['date', 'ticker', 'step', 'seed'])


def d263_raw(variant=None):
    """{seed: ряд Д26.3 h=5 (шаги 1–6)}; S2 — поправка s² sm по лиду (s² шага t−1) до агрегирования."""
    variant = variant or D263_VAR
    var, perlead = variant.rstrip('p'), variant.endswith('p')
    k = ['date', 'ticker', 'step', 'seed']
    parts = []
    for j in (1, 2, 3, 4, 5):
        x, l = d263_lead(var, 'xgb', j), d263_lead(var, 'lgbm', j)
        d = x.merge(l[k + ['rv_pred']], on=k, suffixes=('', '_l'), validate='one_to_one')
        d['rv_pred'] = np.exp((np.log(d.rv_pred) + np.log(d.rv_pred_l)) / 2)
        d = d.drop(columns='rv_pred_l')
        if perlead:
            d = pd.concat([g.assign(rv_pred=g.rv_pred * g.step.map(L.bias_factors(g, 'sm'))) for _, g in d.groupby('seed')])
        parts.append(d.assign(lead=j))
    W = pd.concat(parts, ignore_index=True)
    kk = ['date', 'ticker', 'year', 'step', 'seed']
    agg = W.groupby(kk, as_index=False).agg(rv_actual=('rv_actual', 'mean'), rv_pred=('rv_pred', 'mean'), n=('lead', 'nunique'))
    agg = agg[agg.n == 5].drop(columns='n')
    return {s: g.drop(columns='seed').reset_index(drop=True) for s, g in agg.groupby('seed')}


def d265(which, tag):
    C = components(which)
    print('компоненты:', list(C), flush=True)
    rows = []
    for nm, fr in C.items():  # компоненты по отдельности (уже скорректированы; оценка raw)
        q = [L.years(fr[s], 'raw') for s in SEEDS]
        rows.append(dict(name=nm, m1720=np.mean([np.mean([x[y] for y in range(2017, 2021)]) for x in q]), q2021=np.mean([x[2021] for x in q])))
    print(pd.DataFrame(rows).to_string())
    sl = {s: (combine([C[n][s] for n in C], 'mean'), combine([C[n][s] for n in C], 'median')) for s in SEEDS}
    S = per_seed_gib(f'D26_5{tag}', sl)
    T, best = L.seeds_table(S)
    T['c1_harj'] = c1_harj(T)
    T.to_csv(OUT / f'd26_5{tag}_seeds.csv', index=False)
    for d in ('single|V0_XGB|bez|L0', 'single|V0_LGB|bez|L0'):
        r = T.set_index('design').loc[d]
        print(d, '(среднее)' if 'XGB' in d else '(медиана)', round(r.m1720, 4), [round(r[f'm1720_s{s}'], 4) for s in SEEDS], round(r.q2021, 4), r.failed or 'все')
    print('лучший по правилу:'); print(T.head(5)[['design', 'n_failed', 'failed', 'm1720', 'm1720_s42', 'm1720_s43', 'm1720_s44', 'q2021']].to_string())


def d12_seeds():
    """D12_h5 по seed: дизайн wrap|V0_LGB|B-sm|L0 (среднее XGB+LGBM, как финал), из шага 0."""
    T = pd.read_csv(OUT / 'step0.csv').set_index('name').loc['D12_h5 по seed (лучший дизайн по seed)']
    return [float(T[f'm1720_s{s}']) for s in SEEDS], float(T.q2021)


def entry_table(items):
    """items: [(имя, csv по seed, дизайн или None)] -> условие входа: парно с D12_h5 тех же seed и с фиксированным 0,2159 (умолчание
    дирижёра 09:5x: выполнено, только если 3% и 2σ пройдены в обоих вариантах), 2021 ≤ 0,1566, c1–c4."""
    ref_s, _ = d12_seeds()
    rows = []
    for nm, f, dsg in items:
        T = pd.read_csv(OUT / f)
        r = T.set_index('design').loc[dsg] if dsg else T.iloc[0]
        q = [float(r[f'm1720_s{s}']) for s in SEEDS]
        ea, eb = L.entry(q, ref_s), L.entry(q)
        ok = ea['pass3'] and ea['pass2s'] and eb['pass3'] and eb['pass2s'] and r.q2021 <= L.D12_REF['q2021'] and int(r.n_failed) == 0
        rows.append(dict(name=nm, design=dsg or r.name if dsg else T.iloc[0].design, m1720=float(np.mean(q)), q2021=float(r.q2021),
                         failed=r.failed if isinstance(r.failed, str) else '', A_rel=ea['rel'], A_d=ea['d_mean'], A_2s=ea['two_sigma'], A_t=ea['t'],
                         B_rel=eb['rel'], B_d=eb['d_mean'], B_2s=eb['two_sigma'], B_t=eb['t'], entry=ok))
    E = pd.DataFrame(rows)
    E.to_csv(OUT / 'entry.csv', index=False)
    pd.set_option('display.width', 250)
    print(E.round(4).to_string())
    return E


# ---------------------------------------------------------------- дополнение владельца 09:55, п. 2: симметричные поправки
# Каждая поправка применяется к сырому ряду дизайна и к сырому HAR-J (эталон), параметры — только по OOS-истории шагов < t.
#  base — Б0.1 sm (s² по остаткам шагов t−3..t−1), для рядов «среднего типа» (G5Xopt: офсет с s²/2 внутри) — без поправки;
#  a{m}{tk|mk} — Д23.1: ln ŷ = ln ŷ_raw + κ·ē_m(t) (ē по тикеру / по рынку, ряд D23_ebar), κ — МНК без константы по OOS-строкам шагов < t
#     с эмбарго (конец цели < 1 янв. val-года шага t), затем base-поправка ряда;
#  b — s² по состоянию: (e − ē)² = a + b·ln RVI²_t + c·vov_t (МНК по OOS-остаткам шагов t−3..t−1; e = ln y − ln ŷ_raw, vov — СКО ln RV
#     тикера за 22 дня ≤ t), s²_t обрезается в [0,25; 4]·s̄², F_t = exp(s²_t/2); у рядов среднего типа — множитель exp((s²_t − s̄²)/2);
#  c — QLIKE-калибровка уровня: ŷ ← ŷ_base + c_h, c_h = argmin QLIKE на OOS шага t−1 (ŷ_base шага t−1 — своя OOS-поправка).
CORR = ('base', 'a5tk', 'a10tk', 'a5mk', 'a10mk', 'b', 'c')
_ST = {}
PLANT = {'on': False}  # детектор: подложенная утечка — параметры поправки шага t берутся и по самому шагу t (a, b, c)


def state_frame():
    if 'S' not in _ST:
        import gibrid_har as GH
        import harx as HX
        A, _ = GH.load_pit(L.FEAT_V3)
        A = A.sort_values(['ticker', 'date']).reset_index(drop=True)
        lrv = np.log(A['rv'].astype('float64').clip(lower=1e-8))
        S = A[['date', 'ticker']].copy()
        S['lnrvi2'] = HX.TERMS['ln_rvi2'](A).values
        S['vov'] = lrv.groupby(A['ticker']).transform(lambda s: s.rolling(22, min_periods=11).std()).values
        S['tend_year'] = pd.to_datetime(A.groupby('ticker')['date'].shift(-5)).dt.year.values
        S['date'] = pd.to_datetime(S['date'])
        E = pd.read_parquet(R / 'D23_ebar.parquet'); E['date'] = pd.to_datetime(E['date'])
        S = S.merge(E[['date', 'ticker', 'ebar5', 'ebar10', 'ebar5_mkt', 'ebar10_mkt']], on=['date', 'ticker'], how='left')
        _ST['S'] = S
    return _ST['S']


def harj_raw():
    if 'H' not in _ST:
        import gibrid_har as GH
        A, harj = GH.load_pit(L.FEAT_V3)
        v, _, _ = GH.har_forecasts(5, A, harj)
        _ST['H'] = L.dev_only(v)
    return _ST['H']


def _qmin(y, p):
    from scipy.optimize import minimize_scalar
    lo = -0.9 * float(np.min(p))
    r = minimize_scalar(lambda c: L.qlike(y, p + c), bounds=(lo, float(np.median(p))), method='bounded')
    return float(r.x)


def apply_corr(df, kind, mean_type=False):
    """df — сырой OOS-ряд (шаги 1–6). -> ряд rv_pred после поправки kind (шаг 1 — база для истории, 2016 не оценивается)."""
    S = state_frame()
    d = L.dev_only(df)[['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']].copy()
    d['date'] = pd.to_datetime(d['date'])
    d = d.merge(S, on=['date', 'ticker'], how='left').reset_index(drop=True)
    d = d[d.rv_actual > 1e-12].reset_index(drop=True)
    ln = np.log(d.rv_pred.values)
    if kind.startswith('a'):
        m, tk = kind[1:-2], kind[-2:]
        col = f'ebar{m}' + ('_mkt' if tk == 'mk' else '')
        x = d[col].fillna(0.0).values
        y = np.log(d.rv_actual.values + 1e-10) - ln
        for st in sorted(d.step.unique()):
            vy = 2015 + st
            tr = (d.step < st).values & (d.tend_year < vy).values
            if PLANT['on']:
                tr = (d.step <= st).values
            if tr.sum() < 100:
                continue
            kap = float(x[tr] @ y[tr] / (x[tr] @ x[tr]))
            ms = (d.step == st).values
            ln[ms] = ln[ms] + kap * x[ms]
        _ST.setdefault('kappa', []).append(dict(kind=kind, n=len(d)))
    d['rv_pred'] = np.exp(ln)
    if kind == 'b':
        e = np.log(d.rv_actual.values + 1e-10) - ln
        Z = np.column_stack([np.ones(len(d)), d.lnrvi2.fillna(d.lnrvi2.mean()).values, d.vov.fillna(d.vov.mean()).values])
        F = np.ones(len(d))
        for st in sorted(d.step.unique()):
            ms = (d.step == st).values
            tr = (d.step >= max(1, st - 3)).values & (d.step <= st - 1).values if st > 1 else ms
            if PLANT['on']:
                tr = (d.step >= max(1, st - 3)).values & (d.step <= st).values
            em = e[tr] - e[tr].mean()
            beta = np.linalg.lstsq(Z[tr], em ** 2, rcond=None)[0]
            s2bar = float(np.var(e[tr]))
            s2 = np.clip(Z[ms] @ beta, 0.25 * s2bar, 4 * s2bar)
            F[ms] = np.exp((s2 - s2bar) / 2) if mean_type else np.exp(s2 / 2)
        d['rv_pred'] = d.rv_pred.values * F
        return d[['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']]
    if not mean_type:
        Fb = L.bias_factors(d, 'sm')
        d['rv_pred'] = d.rv_pred * d.step.map(Fb)
    if kind == 'c':
        p = d.rv_pred.values.copy()
        for st in sorted(d.step.unique()):
            if st == 1:
                continue
            pr = (d.step == (st if PLANT['on'] else st - 1)).values
            c = _qmin(d.rv_actual.values[pr], d.rv_pred.values[pr])
            ms = (d.step == st).values
            p[ms] = np.clip(d.rv_pred.values[ms] + c, 1e-9, None)
        d['rv_pred'] = p
    return d[['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']]


def crit(qd, qe, qo):
    """c1–c3 для одиночного ряда (c4 тривиально): qd — годы дизайна, qe — эталона (с той же поправкой), qo — список рядов-компонент."""
    Y = range(2017, 2022)
    c1 = all(qd[y] <= 1.02 * qe[y] for y in Y)
    c2 = all(qd[y] <= 1.02 * min([qe[y]] + [o[y] for o in qo]) for y in Y)
    worst = max(qd[y] / qe[y] for y in Y)
    c3 = worst < min(max(o[y] / qe[y] for y in Y) for o in qo) if qo else True
    return dict(c1=c1, c2=c2, c3=c3, c4=True)


def corr_designs(extra=()):
    """{имя: (тип среднего, {seed: (ряд дизайна, [ряды-компоненты])})}."""
    D = {}
    xs, ls = seed_frames('D12_h5', 'xgb'), seed_frames('D12_h5', 'lgbm')
    D['D12_h5'] = (False, {s: (combine([xs[s], ls[s]], 'mean'), [xs[s], ls[s]]) for s in SEEDS})
    gx, gl = seed_frames('D17_1_h5_G5Xopt', 'xgb'), seed_frames('D17_1_h5_G5Xopt', 'lgbm')
    D['G5Xopt'] = (True, {s: (gl[s], [gx[s]]) for s in SEEDS})
    for nm, spec in extra:
        D[nm] = spec
    return D


def corr(extra=(), tag=''):
    H = harj_raw()
    E = {k: L.years(apply_corr(H, k), 'raw') for k in CORR}
    rows = [dict(design='HAR-J (эталон)', corr=k, seed='—', **{str(y): E[k][y] for y in range(2017, 2022)}) for k in CORR]
    for nm, (mt, by) in corr_designs(extra).items():
        for k in CORR:
            for s, (d, comps) in by.items():
                q = L.years(apply_corr(d, k, mt), 'raw')
                qo = [L.years(apply_corr(o, k, mt), 'raw') for o in comps]
                rows.append(dict(design=nm, corr=k, seed=s, **{str(y): q[y] for y in range(2017, 2022)}, **crit(q, E[k], qo)))
        print(f'{now()} {nm} готов', flush=True)
    T = pd.DataFrame(rows)
    T['m1720'] = T[[str(y) for y in range(2017, 2021)]].mean(axis=1)
    T.to_csv(OUT / f'corr_long{tag}.csv', index=False)
    # сводка: среднее по seed, условие входа (парно с D12_h5 с той же поправкой и с D12_h5 без доп. поправки; рядом 0,2159)
    d12b = T[(T.design == 'D12_h5') & (T['corr'] == 'base')].set_index('seed').m1720
    out = []
    for (nm, k), g in T[T.seed != '—'].groupby(['design', 'corr'], sort=False):
        g = g.set_index('seed')
        d12k = T[(T.design == 'D12_h5') & (T['corr'] == k)].set_index('seed').m1720
        q = [g.m1720[s] for s in SEEDS]
        eS, eB, eF = L.entry(q, [d12k[s] for s in SEEDS]), L.entry(q, [d12b[s] for s in SEEDS]), L.entry(q)
        et = T[(T.design == 'HAR-J (эталон)') & (T['corr'] == k)].iloc[0]
        out.append(dict(design=nm, corr=k, **{str(y): g[str(y)].mean() for y in range(2017, 2022)}, m1720=float(np.mean(q)),
                        **{f's{s}': g.m1720[s] for s in SEEDS}, etalon_1720=float(et.m1720), etalon_2021=float(et['2021']),
                        to_etalon=float(np.mean(q) / et.m1720 - 1), c1=bool(g.c1.all()), c2=bool(g.c2.all()), c3=bool(g.c3.all()), c4=True,
                        rel_D12same=eS['rel'], t_D12same=eS['t'], pass_D12same=eS['pass3'] and eS['pass2s'],
                        rel_D12base=eB['rel'], d_D12base=eB['d_mean'], two_s_D12base=eB['two_sigma'], pass_D12base=eB['pass3'] and eB['pass2s'],
                        rel_fixed=eF['rel'], pass_fixed=eF['pass3'] and eF['pass2s']))
    O = pd.DataFrame(out)
    O.to_csv(OUT / f'corr_summary{tag}.csv', index=False)
    E2 = pd.DataFrame([dict(corr=k, **{str(y): E[k][y] for y in range(2017, 2022)}) for k in CORR])
    pd.set_option('display.width', 300)
    print(E2.round(4).to_string())
    print(O.round(4)[['design', 'corr', '2020', 'm1720', '2021', 's42', 's43', 's44', 'to_etalon', 'c1', 'c2', 'c3', 'rel_D12same', 'rel_D12base',
                      'two_s_D12base', 'd_D12base', 'rel_fixed', 'pass_D12base', 'pass_fixed']].to_string())


def leak():
    """Детектор «будущее не влияет» для новых контуров агента А:
    Д26.5 — поправка компонент (s² шага t−1) и среднее/медиана: цели val-лет шагов ≥ k заменены шумом -> прогнозы шага k побайтно равны;
    подложенная утечка — s² по своему шагу (st вместо st−1) должна дать расхождение.
    Д26.4b — нога HAR-X: файл HARX_h5 (OOS по шагам, детектор harx у агента d16-harx); параметры гибрида шага k — по val шага k−1
    (gibrid_compute: po/qo), проверено там же; здесь — что подмена берёт только строки шагов ≤ 6 и тот же ряд, что HARX_h5."""
    rows = []
    rng = np.random.default_rng(20261026)
    C = components({'G5Xopt', 'D16_1'})
    raw = {'D12_h5': None, 'G5Xopt': None, 'D16_1': None}
    _bf = L.bias_factors
    for planted in (False, True):
        if planted:
            def bf_leak(val, code, own_step1=True):
                v = L.dev_only(val); v = v[v.rv_actual > 1e-12]
                E = {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in v.groupby('step')}
                return {st: np.exp(L.s2_est(code, E, st) / 2) for st in sorted(v.step.unique())}
            L.bias_factors = bf_leak
        for k in (3, 5):
            outs = []
            for pert in (False, True):
                xs, ls = seed_frames('D12_h5', 'xgb'), seed_frames('D12_h5', 'lgbm')
                if pert:
                    for d in (xs[42], ls[42]):
                        m = d.step >= k
                        d.loc[m, 'rv_actual'] = d.loc[m, 'rv_actual'].values * np.exp(rng.normal(0, 1, m.sum()))
                c = corrected(combine([xs[42], ls[42]], 'mean'), 'sm')
                outs.append(c[c.step == k].rv_pred.values)
            dd = float(np.max(np.abs(np.log(outs[0]) - np.log(outs[1]))))
            rows.append(dict(contour='Д26.5 поправка/объединение', leak_planted=planted, step=k, max_abs_dln=dd, ok=(dd == 0.0) != planted))
            print(rows[-1], flush=True)
        L.bias_factors = _bf
    hx = pd.read_parquet(R / 'HARX_h5_val.parquet')
    rows.append(dict(contour='Д26.4b нога HAR-X', leak_planted=False, step=0, max_abs_dln=0.0,
                     ok=bool(L.dev_only(hx).step.max() == 6), note='только шаги ≤ 6; OOS-ряд harx (детектор — агент d16-harx)'))
    pd.DataFrame(rows).to_csv(OUT / 'leak_d26_A.csv', index=False)


def leak_corr(extra=()):
    """Детектор «будущее не влияет» для поправок 09:55 (a — Д23.1, b — s² по состоянию, c — c_h; base — Б0.1 sm):
    цели (rv_actual) строк шагов ≥ k заменены шумом (×exp(N(0,1))) -> прогнозы шага k после поправки побайтно равны.
    Подложенная утечка (PLANT): параметры поправки шага t оцениваются и по самому шагу t -> должно быть расхождение.
    Ряды: эталон HAR-J (сырой), D12_h5 seed 42 (среднее XGB+LGBM), G5Xopt seed 42 (тип среднего) и extra [(имя, кадр, тип среднего)]."""
    rng = np.random.default_rng(20261026)
    xs, ls = seed_frames('D12_h5', 'xgb'), seed_frames('D12_h5', 'lgbm')
    series = [('HAR-J', harj_raw(), False), ('D12_h5 s42', combine([xs[42], ls[42]], 'mean'), False),
              ('G5Xopt s42', seed_frames('D17_1_h5_G5Xopt', 'lgbm')[42], True)] + list(extra)
    rows = []
    for nm, df, mt in series:
        df = L.dev_only(df)[['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']].reset_index(drop=True)
        for planted in (False, True):
            PLANT['on'] = planted
            for kind in (('base',) if not planted else ()) + ('a5tk', 'a10mk', 'b', 'c'):
                for k in (3, 5):
                    pert = df.copy()
                    m = (pert.step >= k).values
                    pert.loc[m, 'rv_actual'] = pert.loc[m, 'rv_actual'].values * np.exp(rng.normal(0, 1, m.sum()))
                    p0, p1 = apply_corr(df, kind, mt), apply_corr(pert, kind, mt)
                    a0, a1 = p0[p0.step == k], p1[p1.step == k]
                    assert len(a0) == len(a1) and (a0.date.values == a1.date.values).all()
                    dd = float(np.max(np.abs(np.log(a0.rv_pred.values) - np.log(a1.rv_pred.values))))
                    rows.append(dict(series=nm, corr=kind, leak_planted=planted, step=k, max_abs_dln=dd, ok=(dd == 0.0) != planted))
                    print(now(), rows[-1], flush=True)
        PLANT['on'] = False
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'leak_corr_d26_A.csv', index=False)
    print('все проверки пройдены' if T.ok.all() else 'ЕСТЬ НАРУШЕНИЯ', flush=True)
    return T


def main():
    if '--leak-corr' in sys.argv:
        ex = []
        if 'D26_1' in sys.argv:
            ex.append(('Д26.1 s42', d261_raw()[42], False))
        if 'D26_3' in sys.argv:
            ex.append(('Д26.3 s42', d263_raw()[42], D263_CODE is None))
        for a in sys.argv:
            if a.startswith('D26_2:'):
                ex.append((a.split(':', 1)[1] + ' s42', b_frames(a)[0][42], True))
        return leak_corr(ex)
    if '--leak' in sys.argv:
        return leak()
    if '--corr' in sys.argv:
        extra = []
        for a in sys.argv[sys.argv.index('--corr') + 1:]:
            if a.startswith('D26_1'):  # Д26.1 агента Б (b_frames): сырой ряд, поправка ряда — sm; компоненты — XGB и LGBM
                raw, comps = b_frames(a)
                extra.append((a, (False, {s: (raw[s], comps[s]) for s in SEEDS})))
            elif a.startswith('D26_2:'):  # Д26.2 агента Б: одна модель в обоих слотах, поправка уже внутри (cj / mix) -> тип «среднего», без компонент
                raw, _ = b_frames(a)
                extra.append((a.split(':', 1)[1], (True, {s: (raw[s], []) for s in SEEDS})))
            elif a.startswith('D26_3'):  # 'D26_3' — вариант D263; компоненты c2/c3 — лиды не подаются (ряд одной модели)
                raw = d263_raw()
                extra.append((f'D26_3{D263_VAR}', (D263_CODE is None, {s: (raw[s], []) for s in SEEDS})))
        tag = os.environ.get('CORR_TAG', '')
        return corr(extra, tag)
    if '--entry' in sys.argv:
        items = [('Д16.1 = Д26.4a (V9 на HAR-X)', 'seeds_Д16.1_V9_на_HAR-X_точки_D12_h5_по_seed.csv', None),
                 ('Д26.4b (D12_h5, нога HAR-X в гибриде)', 'd26_4_seeds.csv', None),
                 ('G5Xopt по seed', 'seeds_D17_1_h5_G5Xopt_по_seed.csv', None),
                 ('Д26.5 промежуточный: среднее {D12_h5, G5Xopt, Д16.1}', 'd26_5_interim_seeds.csv', 'single|V0_XGB|bez|L0'),
                 ('Д26.5 промежуточный: медиана {D12_h5, G5Xopt, Д16.1}', 'd26_5_interim_seeds.csv', 'single|V0_LGB|bez|L0'),
                 ('Д26.5 {D12_h5, Д26.1}: среднее', 'd26_5_b_seeds.csv', 'single|V0_XGB|bez|L0')]
        for tg, lab in (('_full', 'Д26.3 S2'), ('_full_S1', 'Д26.3 S1'), ('_full2', 'Д26.3 S2, Д26.1 лучший'), ('_full2_S1', 'Д26.3 S1, Д26.1 лучший')):
            if (OUT / f'd26_5{tg}_seeds.csv').exists():
                items += [(f'Д26.5 {{D12_h5, Д26.1, {lab}}}: среднее', f'd26_5{tg}_seeds.csv', 'single|V0_XGB|bez|L0'),
                          (f'Д26.5 {{D12_h5, Д26.1, {lab}}}: медиана', f'd26_5{tg}_seeds.csv', 'single|V0_LGB|bez|L0'),
                          (f'Д26.5 {{D12_h5, Д26.1, {lab}}}: лучший по правилу', f'd26_5{tg}_seeds.csv', None)]
        return entry_table(items)
    if '--gib-harx' in sys.argv:
        return gib_harx()
    ap = argparse.ArgumentParser()
    ap.add_argument('--step0', action='store_true')
    ap.add_argument('--d264', action='store_true')
    ap.add_argument('--d265', nargs='*', default=None, help='состав: G5Xopt D16_1 D26_1 D26_3 (D12_h5 всегда)')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    if a.step0:
        step0()
    if a.d264:
        d264()
    if a.d265 is not None:
        d265(set(a.d265), a.tag)


if __name__ == '__main__':
    main()
