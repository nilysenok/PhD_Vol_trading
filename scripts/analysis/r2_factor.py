#!/usr/bin/env python3
"""R2, агент O: фаза B — скрининг дробным факторным планом 2^(14−8) разрешения IV (протокол v3, разд. 3; yaml r2_O_loyo.yaml, factor).

plan     — сверка плана models/designs/r2/factor_plan_v1.csv: SHA-256; восстановление по стандартному порядку A–F и генераторам (п. 3.2);
           баланс 32/32, попарная ортогональность, Σ x_i·x_j·x_k = 0 (главные эффекты не смешаны с парами), слова длины 3 нет (разрешение IV);
           затем 64 дизайна на h по схеме v1 -> models/designs/r2/factor_v1/h{h}.yaml (валидация r2_loyo.validate, сложность).
           Фиктивный фактор на своём h в дизайн не пишется (ключ остаётся default): прогноз от него не зависит, хэш кэша совпадает.
effects  — по таблице шагов прогонов (R2/O/factor/q_h{h}.csv: unit = run, variant — точка сетки ноги; столбцы как у r2_loyo):
           q_r(Y) — вложенный выбор по c3 на T\\{Y} (п. 2.1), z = ln q; e_j(Y) = среднее z при +1 − среднее при −1; ē_j, s_j (ddof = 1),
           интервал ē ± 2,776·s/√5 (t₄); Lenth PSE и эффекты фиктивных — справочно. Порог п. 3.4: оставить, если ē ≤ −0,005 и верхняя
           граница < 0; фиктивный — выключен всегда. Выход: R2/O/factor/effekty_h{h}.csv, R2/O/gate_B_h{h}.yaml.

python3 scripts/analysis/r2_factor.py --selftest
python3 scripts/analysis/r2_factor.py plan
python3 scripts/analysis/r2_factor.py effects --h 1 5 22"""
import argparse, copy, itertools, sys
from pathlib import Path
import numpy as np, pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r2_loyo as O  # noqa: E402  вложенный выбор, критерии, валидация схемы, гейты

FC = O.CFG['factor']
PLAN = O.REPO / FC['plan']
LET = 'ABCDEFGHJKLMNO'
BASIS = 'ABCDEF'
# фактор: (путь ключа схемы, значение при −1, значение при +1); для признаков — добавляемые группы
FAKTORY = {
    'A': ('data.recency_halflife', 'inf', '4y'),
    'B': ('data.history_start', 2014, 'ISS_start'),
    'C': ('boost.resid_source', 'in_sample', 'cross_fit'),
    'D': ('boost.monotone', 'off', 'on'),
    'E': ('boost.clip_sigma', 'off', 3),
    'F': ('calib.s2_state', 'const', 'RVI_volvol'),
    'G': ('calib.kappa', 0.5, 'fit'),
    'H': ('calib.c_qlike', 'off', 'roll_2y'),
    'J': ('features', None, ['G2', 'G3']),
    'K': ('features', None, ['G5', 'G6']),
    'L': ('features', None, ['G8', 'G9']),
    'M': ('mh.persist', 'off', 'qlike'),
    'N': ('mh.reconcile', 'off', 'MinT'),
    'O': ('ens.agg', 'mean', 'median'),
}
FIKTIVNYE = {1: ('M', 'N'), 5: ('D',), 22: ('D',)}  # п. 3.1: monotone — только для role = feature; persist/reconcile на h=1 не применимы
ENET = {'lambda': [0.01, 0.1, 1], 'alpha': [0, 0.5, 1]}
BAZA = {  # п. 3.1, базовый дизайн скрининга; остальные ключи — default схемы
    1: dict(leg=dict(type='HARX', regressors='S4', enet=ENET), boost=dict(role='feature', lib='both_avg', grid='current'),
            features=['G0', 'G1'], ens=dict(seeds=3), _tochki='D22_h1'),
    5: dict(leg=dict(type='HARX', regressors='S3', enet=ENET), boost=dict(role='residual', lib='both_avg', grid='current'),
            features=['G0', 'G1'], ens=dict(seeds=3), _tochki='D12_h5'),
    22: dict(leg=dict(type='HARJ'), boost=dict(role='residual', lib='both_avg', grid='current'),
             features=['G0', 'G1'], ens=dict(seeds=3), _tochki='D12_h22'),
}


# ---------------------------------------------------------------- план
def plan_postroit():
    """Таблица 64 × 14 по стандартному порядку A–F (A меняется быстрее всех) и генераторам."""
    rows = []
    for r in range(64):
        x = {f: (1 if (r >> b) & 1 else -1) for b, f in enumerate(BASIS)}
        for f, w in FC['generatory'].items():
            x[f] = int(np.prod([x[c] for c in w]))
        rows.append([r + 1] + [x[f] for f in LET])
    return pd.DataFrame(rows, columns=['run'] + list(LET))


def plan_proverit(path=PLAN):
    """SHA, совпадение с построенным, баланс, ортогональность, Σ x_i x_j x_k = 0, нет слов длины 3, распределение длин слов."""
    s = O.sha(path)
    assert s == FC['plan_sha256'], f'SHA-256 плана {s} != {FC["plan_sha256"]}'
    P = pd.read_csv(path)
    X = P.drop(columns='run')
    X.columns = [c[0] for c in X.columns]
    assert list(X.columns) == list(LET)
    B = plan_postroit()
    assert (B[list(LET)].values == X.values).all() and (P.run.values == np.arange(1, 65)).all(), 'план не восстанавливается по генераторам'
    M = X.values
    assert (M.sum(axis=0) == 0).all(), 'баланс 32/32'
    assert (M.T @ M == 64 * np.eye(14)).all(), 'ортогональность'
    for i, j, k in itertools.combinations(range(14), 3):
        assert (M[:, i] * M[:, j] * M[:, k]).sum() == 0, f'{LET[i]}{LET[j]}{LET[k]}: главный эффект смешан с парой'
    words = [set(f + w) for f, w in FC['generatory'].items()]
    dl = {}
    for n in range(1, 9):
        for c in itertools.combinations(words, n):
            w = set()
            for x in c:
                w ^= x
            dl[len(w)] = dl.get(len(w), 0) + 1
    assert 3 not in dl and dl.get(4) == 31, dl
    return dict(sha256=s, dliny_slov=dict(sorted(dl.items())), razreshenie='IV')


def dizain(h, x, run):
    """Дизайн прогона: база h + значения факторов (+1 — значение «+», −1 — default «−»); фиктивные факторы h не пишутся."""
    d = copy.deepcopy({k: v for k, v in BAZA[h].items() if not k.startswith('_')})
    d = {'design': f'B_h{h}_r{run:02d} (точки {BAZA[h]["_tochki"]})', 'h': h, **d}
    for f in LET:
        if f in FIKTIVNYE[h] or x[f] < 0:
            continue
        key, _, plus = FAKTORY[f]
        if key == 'features':
            d['features'] = d['features'] + [g for g in plus if g not in d['features']]
            continue
        a, b = key.split('.')
        d.setdefault(a, {})[b] = plus
    O.validate(d)
    return d


def plan_dizainy():
    info = plan_proverit()
    P = pd.read_csv(PLAN)
    P.columns = ['run'] + list(LET)
    out = O.REPO / FC['dizainy']
    out.mkdir(parents=True, exist_ok=True)
    for h in (1, 5, 22):
        ds = [dizain(h, r._asdict(), r.run) for r in P.itertuples()]
        meta = [dict(run=int(r.run), slozhnost=O.slozhnost(d), **{f: int(getattr(r, f)) for f in LET}) for d, r in zip(ds, P.itertuples())]
        f = out / f'h{h}.yaml'
        f.write_text(yaml.safe_dump(dict(plan=str(FC['plan']), plan_sha256=info['sha256'], fiktivnye=list(FIKTIVNYE[h]),
                                         dizainy=ds, meta=meta), allow_unicode=True, sort_keys=False))
        print(f'h={h}: 64 дизайна -> {f}; сложность {min(m["slozhnost"] for m in meta)}–{max(m["slozhnost"] for m in meta)}', flush=True)
    print('план:', info, flush=True)
    return info


# ---------------------------------------------------------------- эффекты (п. 3.3) и порог (п. 3.4)
def q_po_skladkam(tab):
    """q_r(Y) прогона r: вложенный выбор по c3 на T\\{Y} (п. 2.1; если c3 нет ни у одного варианта — лучший по среднему, с пометкой)."""
    rows = []
    for u, gu in tab.groupby('unit', sort=False):
        G = {v: g.sort_values('step') for v, g in gu.groupby('variant', sort=False)}
        vs = list(G)
        for Yst in O.T:
            tr = tuple(s for s in O.T if s != Yst)
            order = sorted(vs, key=lambda v: (G[v][G[v].step.isin(tr)].q_hyb.mean(), vs.index(v)))
            v, _, _ = O.pick(G, order, tr, O.MODES['v3']['sel'], c1_E=True, c3_E=True)
            vv = v if v is not None else order[0]
            rows.append(dict(unit=u, Y=O.Y(Yst), variant=vv, c3=v is not None, q=float(G[vv].set_index('step').q_hyb[Yst])))
    return pd.DataFrame(rows)


def effekty(qf, plan, h):
    """qf — unit (run, int или 'r01'), Y, q. -> таблица эффектов по факторам с решением п. 3.4."""
    qf = qf.assign(run=qf.unit.map(lambda u: int(str(u).lstrip('r').split('_')[-1]) if not isinstance(u, (int, np.integer)) else int(u)))
    z = qf.pivot(index='run', columns='Y', values='q').sort_index().apply(np.log)
    assert len(z) == 64 and z.notna().all().all(), 'нужны все 64 прогона × 5 складок'
    X = plan.set_index('run').loc[z.index]
    E = pd.DataFrame({f: (z[X[f].values == 1].mean() - z[X[f].values == -1].mean()) for f in LET}).T  # строки — факторы, столбцы — Y
    eb, s = E.mean(axis=1), E.std(axis=1, ddof=1)
    half = FC['t_975_4'] * s / np.sqrt(len(E.columns))
    pse_e = np.abs(eb.values)
    s0 = 1.5 * np.median(pse_e)
    pse = 1.5 * np.median(pse_e[pse_e < 2.5 * s0]) if (pse_e < 2.5 * s0).any() else np.nan
    fikt = set(FIKTIVNYE[h])
    T = pd.DataFrame(dict(faktor=list(LET), klyuch=[FAKTORY[f][0] for f in LET], e_bar=eb.values, s=s.values, ci_low=(eb - half).values,
                          ci_high=(eb + half).values, fiktivnyy=[f in fikt for f in LET], **{f'e_{y}': E[y].values for y in E.columns}))
    T['vklyuchen'] = (T.e_bar <= FC['porog']) & (T.ci_high < 0) & ~T.fiktivnyy
    T['lenth_pse'] = pse
    T['shum_fiktivnyh'] = float(np.abs(T[T.fiktivnyy].e_bar).max()) if T.fiktivnyy.any() else np.nan
    return T


def effects(hs):
    out = O.R / FC['out']
    out.mkdir(parents=True, exist_ok=True)
    info = plan_proverit()
    plan = pd.read_csv(PLAN)
    plan.columns = ['run'] + list(LET)
    for h in hs:
        fq = O.R / FC['q_vhod'].format(h=h)
        tab = pd.read_csv(fq)
        qf = q_po_skladkam(tab)
        T = effekty(qf, plan, h)
        T.to_csv(out / f'effekty_h{h}.csv', index=False, float_format='%.6g')
        qf.to_csv(out / f'q_skladki_h{h}.csv', index=False, float_format='%.6g')
        g = O.write_gate('B', h, dict(plan_sha256=info['sha256'], vklyucheny=list(T[T.vklyuchen].faktor),
                                       faktory=T[['faktor', 'klyuch', 'e_bar', 'ci_low', 'ci_high', 'fiktivnyy', 'vklyuchen']].to_dict('records'),
                                       porog=dict(e_bar_max=FC['porog'], ci_high_lt=0), lenth_pse=float(T.lenth_pse.iloc[0]),
                                       c3_net_v_skladke=int((~qf.c3).sum()), N_B=64,
                                       lozhno_vklyuchennyh_ozhid=O.CFG['lozhnye']['n_faktorov'] * O.CFG['lozhnye']['skrining_alpha']),
                         [fq, PLAN, O.REPO / FC['dizainy'] / f'h{h}.yaml'])
        print(f'h={h}: включены {list(T[T.vklyuchen].faktor)} -> {g}', flush=True)


# ---------------------------------------------------------------- самопроверка
def selftest():
    info = plan_proverit()
    plan = pd.read_csv(PLAN)
    plan.columns = ['run'] + list(LET)
    # подложенная ошибка плана ловится
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    bad = pd.read_csv(PLAN)
    bad.iloc[5, 3] *= -1
    bad.to_csv(tmp / 'p.csv', index=False)
    try:
        plan_proverit(tmp / 'p.csv')
        raise AssertionError('испорченный план не пойман')
    except AssertionError as e:
        assert 'SHA' in str(e), e
    # дизайны: база без факторов — сложность базы; фиктивные не пишутся; прогон 64 — все «+»
    d1 = dizain(1, {f: -1 for f in LET}, 1)
    assert O.slozhnost({k: v for k, v in d1.items()}) == 5 and 'mh' not in d1
    d64 = dizain(1, {f: 1 for f in LET}, 64)
    assert 'mh' not in d64 and d64['features'] == ['G0', 'G1', 'G2', 'G3', 'G5', 'G6', 'G8', 'G9'] and d64['ens']['agg'] == 'median'
    d5 = dizain(5, {f: 1 for f in LET}, 64)
    assert 'monotone' not in d5['boost'] and d5['mh'] == {'persist': 'qlike', 'reconcile': 'MinT'}
    # эффекты на синтетике: z = база + Σ β_j x_j + шум; A −2% и K −1,2% — включены; C +1% — нет; J −0,3% (< 0,5%) — нет;
    # M на h=1 — фиктивный (даже с «эффектом») — нет; шумовой фактор с большой дисперсией по складкам — нет (интервал через 0)
    rng = np.random.default_rng(3)
    beta = dict(A=-0.02, C=0.01, J=-0.003, K=-0.012, M=-0.03)
    rows = []
    for r in plan.itertuples():
        x = r._asdict()
        for y in range(2017, 2022):
            zz = np.log(0.25) + sum(b * x[f] for f, b in beta.items()) / 2 + rng.normal(0, 0.002)  # e_j = β_j
            zz += 0.03 * x['B'] * (1 if y % 2 else -1) / 2  # B: эффект знакопеременный по складкам -> интервал через 0
            rows.append(dict(unit=f'r{r.run:02d}', Y=y, q=float(np.exp(zz))))
    T = effekty(pd.DataFrame(rows), plan, 1).set_index('faktor')
    assert np.isclose(T.loc['A', 'e_bar'], -0.02, atol=2e-3) and T.loc['A', 'vklyuchen'] and T.loc['K', 'vklyuchen']
    assert not T.loc['C', 'vklyuchen'] and not T.loc['J', 'vklyuchen'] and not T.loc['B', 'vklyuchen']
    assert T.loc['M', 'fiktivnyy'] and not T.loc['M', 'vklyuchen'] and T.loc['B', 'ci_low'] < 0 < T.loc['B', 'ci_high']
    assert set(T[T.vklyuchen].index) == {'A', 'K'}
    # вложенный выбор по c3 внутри складки: отложенный год не влияет на выбор варианта (утечки нет)
    tab = []
    for r in range(1, 65):
        for v in range(3):
            for st in O.T:
                tab.append(dict(unit=f'r{r:02d}', variant=f'v{v}', step=st, q_hyb=0.25 + 0.002 * v + 0.001 * rng.normal(), q_har=0.30,
                                q_har_c1=0.30, q_xgb_single=0.31, q_lgb_single=0.31, scheme='cell', v1_ml='', w_har=0., w_xgb=0., w_lgb=0.))
    tab = pd.DataFrame(tab)
    q1 = q_po_skladkam(tab)
    t2 = tab.copy()
    t2.loc[t2.step == 4, 'q_hyb'] *= 3
    q2 = q_po_skladkam(t2)
    assert (q1[q1.Y == 2019].variant.values == q2[q2.Y == 2019].variant.values).all()
    assert effekty(q1, plan, 5).shape[0] == 14
    print('selftest r2_factor: OK (план: SHA, восстановление по генераторам, баланс, ортогональность, IV, A4 =', info['dliny_slov'].get(4),
          '; дизайны по схеме v1 и фиктивные; эффекты, t4-интервал, порог 3.4; вложенный выбор без утечки)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', nargs='?', choices=['plan', 'effects'])
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--h', type=int, nargs='*', default=[1, 5, 22])
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == 'plan':
        return plan_dizainy()
    if a.cmd == 'effects':
        return effects(a.h)
    ap.print_help()


if __name__ == '__main__':
    main()
