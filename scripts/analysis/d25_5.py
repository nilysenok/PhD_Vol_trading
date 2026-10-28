#!/usr/bin/env python3
"""Д25.5 — сглаживание по датам начала: ŷ_t ← λ·ŷ_t + (1−λ)·ŷ_{t−1} (уровень RV, по тикеру; ŷ_{t−1} — несглаженный OOS-прогноз предыдущей
торговой даты, известный в t), λ ∈ {0,5; 0,7; 1}. λ шага k выбирается по QLIKE сглаженного ряда на val-годе шага k−1 и применяется к val-году
шага k (тест шага k — λ по val-году шага k, без метрик). Шаг 1 (у схем OOS нет) — ряд шага 1 в выборке (известен к концу 2016 г.).
Применяется к финалистам всех h (CONFIRM_OPENED.json) + финал h=22 (HAR-J с поправкой) и к прошедшим вариантам Д25.3/25.4.
c1–c4 — gibrid_designs.flags на таблице шагов дизайна (те же одиночные модели и веса, q_hyb — сглаженный прогноз), шаги 2–6.
python3 scripts/analysis/d25_5.py run | leak"""
import os, sys, argparse
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d25_4 as L4
import gibrid_designs as D
from d25_4 import OUT, LOCK, KEY, CRIT, RANK, HOLD, qlike

LAMS = (0.5, 0.7, 1.0)
os.environ['GIBRID_SAME_SLOT'] = '1'  # как final3 (Д24.6: одиночная модель в обоих слотах)


def smooth(v, lam_of_step=None, lam=None, col='rv_pred'):
    """v: KEY + year + rv_pred (+ rv_ins для шага 1). Возвращает массив сглаженного прогноза (порядок строк v)."""
    p = v[col].values  # v отсортирован по (ticker, date) — prep()
    prev = v.groupby('ticker')[col].shift(1).values
    prev = np.where(np.isfinite(prev), prev, p)
    L = np.full(len(v), lam) if lam is not None else v.step.map(lam_of_step).values.astype(float)
    return L * p + (1 - L) * prev


def choose(v):
    """λ шага k по val-году шага k−1 (сглаженный ряд при каждом λ). -> {шаг: λ_val}, {шаг: λ_test} (тест шага k — по val шага k)."""
    q = {}
    for lam in LAMS:
        s = smooth(v, lam=lam)
        for st, g in v.assign(_s=s).groupby('step'):
            g = g[(g.rv_actual > 1e-12) & np.isfinite(g._s)]
            # шаги 7–9: QLIKE — только внутренний выбор λ для val шагов 8–9 и теста (как выбор схем в gibrid); не выводится и не сохраняется
            q[(st, lam)] = qlike(g.rv_actual.values, g._s.values) if len(g) else np.nan
    steps = sorted(v.step.unique())
    lv = {st: (1.0 if st == 1 else min(LAMS, key=lambda l: (q[(st - 1, l)], -l))) for st in steps}
    lt = {st: min(LAMS, key=lambda l: (q[(st, l)], -l)) for st in steps}
    return lv, lt


def prep(v):
    """Ряд для сглаживания: OOS-прогноз; на шаге 1 (нет OOS) — прогноз в выборке шага 1."""
    v = v.sort_values(['ticker', 'date']).reset_index(drop=True)
    if 'rv_ins' in v:
        v['rv_pred'] = np.where(np.isfinite(v.rv_pred), v.rv_pred, v.rv_ins)
    return v


def apply(v):
    v = prep(v)
    lv, lt = choose(v)
    v['rv_sm'] = smooth(v, lam_of_step=lv)
    return v, lv, lt


def oos_with_ins(cand, h, design, test=False):
    """design_oos + столбец rv_ins (в выборке шага, нужен только на шаге 1)."""
    import gibrid_schemes as S
    v, t, ch = L4.design_oos(cand, h, design, test=test)
    # в выборке: select возвращает rv_pred (параметры шага) — берём повторно для шага 1
    c = L4.cand_cfg(cand, h)
    var, sel, lab, lam, *bl = design.split('|')
    if not sel.startswith('V0_'):
        S.MODE = bl[0] if bl else 'rv'
        from gibrid_compute import compute
        from gibrid_io import load_boost
        hv, ht, ext = L4.har_inputs(h, c['features'], False)
        bx, bl_ = load_boost(c['gboost'], h, c['regen'], var)
        res = D.shrink(compute(hv, ht, h, bx, bl_, 'bez' if lab == 'bez' else lab[2:], h, [], ext), float(lam[1:]))
        _, ov, _ = D.select(res, sel, h, 'oos')
        S.MODE = 'rv'
        v = v.merge(ov[KEY + ['rv_pred']].rename(columns={'rv_pred': 'rv_ins'}), on=KEY, how='left')
    return v, t, ch


def flags_of(ch, qs):
    c = ch[ch.step.isin(CRIT)].copy()
    c['q_hyb'] = c.step.map(qs).values
    fl = D.flags(c)
    return {k: fl[k] for k in ('c1', 'c2', 'c3', 'c4')}


def run(a):
    rows, lams = [], []
    todo = [(h, c, d) for h, lst in L4.FINALISTS.items() for c, d in lst]
    if a.d253:  # прошедшие варианты Д25.3 (А): наклон HAR-J (V0_XGB) и наклон бустинга (V0_LGB), seed 42/43/44
        todo = [(22, f'D25_3s{s}', d) for s in (42, 43, 44) for d in ('single|V0_XGB|B-sm|L0', 'single|V0_LGB|B-sm|L0')]
    for h, cand, des in todo:
        if True:
            v, _, ch = oos_with_ins(cand, h, des)
            v, lv, lt = apply(v)
            q0, q1 = L4.year_table(v.assign(rv_pred=np.where(v.step == 1, np.nan, v.rv_pred))), L4.year_table(v, 'rv_sm')
            # q0 — исходный дизайн (шаги 2–6, OOS); сверка с final3 — по m1720
            for nm, q in (('исходный (λ=1)', q0), ('сглаживание Д25.5', q1)):
                rows.append(dict(h=h, cand=cand, design=des, variant=nm, final=(L4.FINAL.get(h) == cand), **L4.summ(q),
                                 **{f'y{2015 + s}': q[s] for s in CRIT}, **flags_of(ch, q)))
            lams.append(dict(h=h, cand=cand, design=des, **{f'lam_s{s}': lv[s] for s in CRIT}))
            tag = cand + ('' if not a.d253 else '_' + des.split('|')[1])
            v[KEY + ['year', 'rv_actual', 'rv_pred', 'rv_sm']].to_parquet(OUT / f'd25_5_{tag}_h{h}_val.parquet', index=False)
            print(rows[-2]['cand'], h, round(rows[-2]['m1720'], 4), '->', round(rows[-1]['m1720'], 4),
                  {int(s): lv[s] for s in CRIT}, flush=True)  # только шаги 2–6: λ шагов 7–9 выбраны по подтверждающему — не выводятся
    T = pd.DataFrame(rows)
    base = T[T.variant.str.startswith('исх')].set_index(['h', 'cand', 'design'])
    T['d_vs_base'] = [r.m1720 / base.loc[(r.h, r.cand, r.design), 'm1720'] - 1 for r in T.itertuples()]
    T['d2021_vs_base'] = [r.q2021 / base.loc[(r.h, r.cand, r.design), 'q2021'] - 1 for r in T.itertuples()]
    T['pass2021'] = T.q2021 <= T.h.map(L4.BEST2021)
    sfx = '_d253' if a.d253 else ''
    T.to_csv(OUT / f'd25_5_summary{sfx}.csv', index=False)
    pd.DataFrame(lams).to_csv(OUT / f'd25_5_lambdas{sfx}.csv', index=False)
    print(T.drop(columns=['design']).to_string())


def leak(a):
    """Детектор: портим rv_actual и прогнозы с даты среза — сглаженный прогноз и λ до среза не меняются; подложенная утечка (λ по тому же
    шагу) обязана сработать."""
    v = pd.read_parquet(OUT / 'd25_5_D23_1_h22e_h22_val.parquet')[KEY + ['year', 'rv_actual', 'rv_pred']]
    cut = pd.Timestamp('2019-01-01')
    vb = v.copy()
    f = vb.date >= cut
    rng = np.random.default_rng(1)
    vb.loc[f, 'rv_actual'] *= rng.uniform(0.2, 5, f.sum())
    vb.loc[f, 'rv_pred'] *= rng.uniform(0.5, 2, f.sum())
    A, _, _ = apply(v.copy())
    B, _, _ = apply(vb.copy())
    m = A.merge(B[KEY + ['rv_sm']], on=KEY, suffixes=('', '_b'))
    pre = m.date < cut
    d = float(np.max(np.abs(np.log(m.rv_sm[pre]) - np.log(m.rv_sm_b[pre]))))
    ok = d < 1e-12
    print(f'сглаживание: max |Δ ln ŷ| до среза = {d:.2e}', 'OK' if ok else 'УТЕЧКА')
    # подложенная утечка: λ шага k по val-году самого шага k (а для 2018 — смотрит в порченный 2019 через сдвиг вперёд)
    def leaky(x):
        x = prep(x)
        _, lt = choose(x)
        x['rv_sm'] = smooth(x.assign(rv_pred=x.groupby('ticker').rv_pred.shift(-1).fillna(x.rv_pred)), lam_of_step=lt)
        return x
    A2, B2 = leaky(v.copy()), leaky(vb.copy())
    m2 = A2.merge(B2[KEY + ['rv_sm']], on=KEY, suffixes=('', '_b'))
    pre = m2.date < cut
    d2 = float(np.max(np.abs(np.log(m2.rv_sm[pre]) - np.log(m2.rv_sm_b[pre]))))
    print(f'подложенная утечка: max |Δ| = {d2:.2e}', 'сработал (OK)' if d2 > 1e-12 else 'НЕ СРАБОТАЛ')
    print('ДЕТЕКТОР Д25.5:', 'OK' if ok and d2 > 1e-12 else 'ОШИБКА')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'leak'])
    ap.add_argument('--d253', action='store_true', help='прошедшие варианты Д25.3 вместо финалистов')
    a = ap.parse_args()
    {'run': run, 'leak': leak}[a.cmd](a)
