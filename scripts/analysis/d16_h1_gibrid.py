#!/usr/bin/env python3
"""Кандидат третьего круга D16_h1_d812x (Д16.1, h=1: Д8.12 с ln HAR-X): файлы в формате gibrid_v17 / final3 + yaml.
val (data/regen):  D16_h1_d812x_{xgb,lgbm}_h1_val.parquet (single: точка ранга 1, seed 42, шаги 1-9),
                   ..._h1_top5_val.parquet (ранги 1-5, шаги 1-6), ..._h1_seeds4344_val.parquet (seed 43/44, шаги 1-9);
тест (test_locked/D16_h1_d812x/): те же имена с _test (без метрик);
шаг 10: data/regen/D16/step10/D16_h1_d812x_{m}_h1_s10_val.parquet (val 2025, seed 42/43/44), test_locked/D16_h1_d812x/step10/..._s10_test.parquet;
yaml: data/regen/final3/kandidaty/D16_h1_d812x.yaml."""
from pathlib import Path
import pandas as pd

R = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')
B = 'D16_h1_d812x'
TD = R / 'test_locked' / B
COLS = ['date', 'ticker', 'year', 'step', 'seed', 'rank', 'rv_actual', 'rv_pred']


def rd(sfx, kind):
    tag = f'{B}{sfx}'
    p = R / f'{tag}_val.parquet' if kind == 'val' else R / 'test_locked' / tag / f'{tag}_test.parquet'
    return pd.read_parquet(p)


def put(df, name, kind, d=None):
    df = df[COLS].sort_values(['step', 'date', 'ticker', 'seed', 'rank']).reset_index(drop=True)
    d = d or (R if kind == 'val' else TD)
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / f'{name}_{kind}.parquet', index=False)
    return len(df)


def main():
    src = {'xgb': {'base': ['', '_s79'], 's4344': ['_s4344', '_s4344b'], 's10': ['_s10', '_s10b']},
           'lgbm': {'base': ['_L'], 's4344': ['_Ls4344'], 's10': ['_Ls10']}}
    rep = []
    for m, s in src.items():
        for kind in ('val', 'test'):
            base = pd.concat([rd(x, kind) for x in s['base']], ignore_index=True)
            base = base[base.model == m]
            single = base[(base['rank'] == 1) & (base.seed == 42) & (base.step <= 9)]
            top5 = base[base.step <= 6]
            seeds = pd.concat([rd(x, kind) for x in s['s4344']], ignore_index=True)
            seeds = seeds[(seeds.model == m) & (seeds.step <= 9)]
            assert sorted(single.step.unique()) == list(range(1, 10)) and top5['rank'].nunique() == 5
            assert sorted(seeds.step.unique()) == list(range(1, 10)) and sorted(seeds.seed.unique()) == [43, 44]
            n1 = put(single, f'{B}_{m}_h1', kind); n5 = put(top5, f'{B}_{m}_h1_top5', kind); n3 = put(seeds, f'{B}_{m}_h1_seeds4344', kind)
            s10 = pd.concat([rd(x, kind) for x in s['s10']], ignore_index=True)
            s10 = s10[(s10.model == m) & (s10.step == 10)]
            assert sorted(s10.seed.unique()) == [42, 43, 44]
            n10 = put(s10, f'{B}_{m}_h1_s10', kind, R / 'D16' / 'step10' if kind == 'val' else TD / 'step10')
            rep.append((m, kind, n1, n5, n3, n10))
    y = f"""boost: {B}
h: 1
semeystvo: гибрид Д8.12 на HAR-X
rychag: Д16.1
opisanie: XGBoost/LightGBM на ln RV + ln HAR-X как признак (Д8.12 с основой HAR-X вместо HAR-J; HAR-X = HAR-J + RVI, RV IMOEX, RV рынка, LHAR, PC1, HARQ, RSV±, эластичная сеть); точки ЛГК B v2b без нового подбора
regen: data/regen
legs: [single, top5, seeds3]
odinochnaya: false
chistyy: false
test_dir: data/regen/test_locked/{B}
step10_dir: data/regen/D16/step10
step10_lock: data/regen/test_locked/{B}/step10
n_variantov: 11
"""
    (R / 'final3' / 'kandidaty').mkdir(parents=True, exist_ok=True)
    (R / 'final3' / 'kandidaty' / f'{B}.yaml').write_text(y)
    for r in rep:
        print(r)
    print('yaml ->', R / 'final3' / 'kandidaty' / f'{B}.yaml')


if __name__ == '__main__':
    main()
