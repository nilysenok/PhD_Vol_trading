#!/usr/bin/env python3
"""Финальная сборка второго круга, шаг 2: ОДИНАКОВАЯ обёртка для всех кандидатов, включая V0 (решение владельца 25.10, 20:40/20:45).
Без нового обучения — из уже сохранённых прогнозов кандидата {T}_{xgb,lgbm}_h{h}{,_top5,_seeds4344}_{val,test}.parquet.
Компоненты (как wrap_v9.py; набор — общий для всех кандидатов, поэтому топ-5: топ-10 есть не у всех поисков):
  1a — среднее топ-5 точек поиска шага (на шагах 7-9 топ-5 нет — точка шага 6, как нога top5 гибрида);
  1b — среднее 3 seed (42/43/44) лучшей точки;
  2  — среднее LightGBM и XGBoost.
База — лучшая из двух моделей (single, seed 42) по OOS QLIKE шагов 2-6 с Б0.1 ord (l2_lib.evaluate). Компонент берётся, если OOS не хуже базы;
взятые проверяются вместе, при ухудшении убираются по одному, начиная с худшего. Порядок: среднее членов внутри модели (1a ∪ 1b) -> 2.
Выход — нога wrap2 для gibrid_v17.py: слот базовой модели = обёртка, другой слот = другая модель с теми же внутримодельными компонентами.
Подтверждающий набор (шаги 7-9) пишется в файл ноги (нужен гибриду), но здесь НЕ оценивается; тест — только в test_locked/{T}/, не оценивается.
python3 scripts/analysis/final2_wrap.py --boost D12_h5 --h 5"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L

R = L.REGEN
K = ['date', 'ticker', 'step']
M = ('lgbm', 'xgb')


def rd(T, m, h, kind, k):
    sfx = {'single': '', 'top5': '_top5', 'seeds': '_seeds4344'}[kind]
    p = (R if k == 'val' else R / 'test_locked' / T) / f'{T}_{m}_h{h}{sfx}_{k}.parquet'
    return pd.read_parquet(p)


def build(T, h, comps, k):
    """-> {model: DataFrame[date, ticker, year, step, rv_actual, rv_pred]} с внутримодельными компонентами; '2' — отдельно."""
    out = {}
    for m in M:
        s = rd(T, m, h, 'single', k)
        s = s[s.seed == 42] if 'seed' in s else s
        base = s[K + ['year', 'rv_actual']].copy()
        mem = []
        if '1a' in comps:
            t5 = rd(T, m, h, 'top5', k)
            has = set(t5.step.unique())
            mem.append(t5[K + ['rv_pred']])
            mem.append(s[~s.step.isin(has)][K + ['rv_pred']])  # шаги без топ-5 (7-9): точка шага 6
        else:
            mem.append(s[K + ['rv_pred']])
        if '1b' in comps:
            sd = rd(T, m, h, 'seeds', k)
            mem.append(sd[K + ['rv_pred']])
            if '1a' in comps:  # лучшая точка seed 42 уже в топ-5 (ранг 1); на шагах без топ-5 — в single
                pass
        allm = pd.concat(mem, ignore_index=True)
        f = allm.groupby(K, as_index=False).rv_pred.mean()
        out[m] = base.merge(f, on=K, how='left', validate='one_to_one')
        assert out[m].rv_pred.notna().all(), (T, m, k)
    return out


def combine(P, comps, prim):
    if '2' not in comps:
        return P[prim]
    a, b = P['lgbm'], P['xgb'][K + ['rv_pred']]
    m = a.merge(b, on=K, suffixes=('_l', '_x'), validate='one_to_one')
    return m.assign(rv_pred=(m.rv_pred_l + m.rv_pred_x) / 2).drop(columns=['rv_pred_l', 'rv_pred_x'])


def ev(df):
    e = L.evaluate(df[df.step <= 6])
    return e['ord'][0], min(e[c][0] for c in L.CODES), e['ord'][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boost', required=True); ap.add_argument('--h', type=int, required=True)
    ap.add_argument('--out', default=str(R / 'final2'))
    a = ap.parse_args()
    T, h = a.boost, a.h
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    P0 = build(T, h, (), 'val')
    e0 = {m: ev(P0[m]) for m in M}
    prim = min(M, key=lambda m: e0[m][0])
    q0 = e0[prim][0]
    rows = [dict(boost=T, h=h, comp='база (single, seed 42)', model=prim, q=q0, best4=e0[prim][1], per_step=' / '.join(f'{x:.4f}' for x in e0[prim][2]), taken='база'),
            dict(boost=T, h=h, comp='справка: другая модель', model=[m for m in M if m != prim][0], q=e0[[m for m in M if m != prim][0]][0],
                 best4=e0[[m for m in M if m != prim][0]][1], per_step='', taken='—')]
    ind = {}
    for c in ('1a', '1b', '2'):
        q, b4, ps = ev(combine(build(T, h, (c,), 'val'), (c,), prim))
        ind[c] = q
        rows.append(dict(boost=T, h=h, comp=c, model='lgbm+xgb' if c == '2' else prim, q=q, best4=b4, per_step=' / '.join(f'{x:.4f}' for x in ps),
                         taken='да' if q <= q0 else 'нет'))
    keep = [c for c in ind if ind[c] <= q0]
    qa = q0
    while keep:
        q, b4, ps = ev(combine(build(T, h, tuple(keep), 'val'), tuple(keep), prim))
        rows.append(dict(boost=T, h=h, comp='вместе: ' + '+'.join(keep), model='lgbm+xgb' if '2' in keep else prim, q=q, best4=b4,
                         per_step=' / '.join(f'{x:.4f}' for x in ps), taken='да' if q <= q0 else 'нет'))
        qa = q
        if q <= q0:
            break
        keep.remove(max(keep, key=lambda c: ind[c]))
    if not keep:
        qa = q0
    final = tuple(keep)
    tab = pd.DataFrame(rows)
    tab['vs_base'] = tab.q / q0 - 1
    tab['final'] = '+'.join(final) or 'none'
    tab.to_csv(od / f'wrap2_{T}_h{h}.csv', index=False, float_format='%.6g')
    other = [m for m in M if m != prim][0]
    for k in ('val', 'test'):
        P = build(T, h, tuple(c for c in final if c != '2'), k)
        W = combine(P, final, prim)
        slots = {prim: W, other: P[other]}
        d = R if k == 'val' else R / 'test_locked' / T
        for m in M:
            slots[m].assign(wrap='+'.join(final) or 'none').sort_values(K).to_parquet(d / f'{T}_{m}_h{h}_wrap2_{k}.parquet', index=False)
    print(f'{T} h={h}: база {prim} {q0:.5f}; ' + ', '.join(f'{c} {ind[c]:.5f} ({ind[c] / q0 - 1:+.2%})' for c in ind)
          + f'; итог {"+".join(final) or "none"} {qa:.5f} ({qa / q0 - 1:+.2%})', flush=True)


if __name__ == '__main__':
    main()
