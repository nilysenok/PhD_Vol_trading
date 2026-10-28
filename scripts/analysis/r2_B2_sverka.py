#!/usr/bin/env python3
"""R2 Б2: сверка рядов ролей на ногах Э версии первого раунда (_r1) с сохранёнными рядами первого раунда, seed 42, val шагов 1–6.
Ожидается совпадение ≤ 1e-7 там, где нога Э совпадает с основой первого раунда:
  ost HARX_inner_r1 h=5  -> D16_h5_D12x (Д16.1: V9 на HAR-X, точки D12_h5);
  gam HARX_inner_r1 h=5  -> D17_1_h5_G5Xopt (LGBM);  h=22 -> D17_1_h22_G22Xopt (LGBM);
  ost HARJ_r1 h=22       -> D12_h22 (HAR-J МНК, как l2_lib.har_ln).
Справочно (совпадения не ждём): ost HARJ_r1 h=5 -> D12_h5 — основа D12_h5 — HAR-J по Хьюберу (cfg huber), у Э — МНК.
Метрик нет. Выход: data/regen/R2/cache/B2/_control/sverka_r1.csv.
python3 scripts/analysis/r2_B2_sverka.py"""
from pathlib import Path
import numpy as np, pandas as pd

R = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')
B2 = R / 'R2' / 'cache' / 'B2'
K = ['date', 'ticker', 'step']
TOL = 1e-7
CHECKS = [  # (ряд Б2, h, сохранённый файл, модель в сохранённом, ждём совпадения)
    ('ost_HARX_inner_r1_xgb', 5, 'D16_h5_D12x_val.parquet', 'xgb', True),
    ('ost_HARX_inner_r1_lgbm', 5, 'D16_h5_D12x_val.parquet', 'lgbm', True),
    ('gam_HARX_inner_r1_lgbm', 5, 'D17_1_h5_G5Xopt_lgbm_h5_val.parquet', None, True),
    ('gam_HARX_inner_r1_lgbm', 22, 'D17_1_h22_G22Xopt_lgbm_h22_val.parquet', None, True),
    ('ost_HARJ_r1_xgb', 22, 'D12_h22_xgb_h22_val.parquet', None, True),
    ('ost_HARJ_r1_lgbm', 22, 'D12_h22_lgbm_h22_val.parquet', None, True),
    ('ost_HARJ_r1_xgb', 5, 'D12_h5_xgb_h5_val.parquet', None, False),
    ('ost_HARJ_r1_lgbm', 5, 'D12_h5_lgbm_h5_val.parquet', None, False),
]


def main():
    rows = []
    for name, h, ref, m, must in CHECKS:
        a = pd.read_parquet(B2 / f'{name}_h{h}_s42_val.parquet')
        b = pd.read_parquet(R / ref)
        b = b[(b.seed == 42) & (b.step <= 6)] if 'seed' in b else b[b.step <= 6]
        if m is not None:
            b = b[b.model == m]
        if 'rank' in b:
            b = b[b['rank'] == 1]
        mm = a[K + ['rv_pred']].merge(b[K + ['rv_pred']], on=K, suffixes=('_a', '_b'))
        d = float(np.abs(np.log(mm.rv_pred_a) - np.log(mm.rv_pred_b)).max())
        ok = (len(mm) == len(a) == len(b) and d <= TOL) if must else None
        rows.append(dict(ryad=name, h=h, ref=ref, rows=len(a), rows_ref=len(b), rows_common=len(mm), max_dln=d, zhdem=must, ok=ok))
    T = pd.DataFrame(rows)
    (B2 / '_control').mkdir(parents=True, exist_ok=True)
    T.to_csv(B2 / '_control' / 'sverka_r1.csv', index=False)
    print(T.to_string(index=False))
    assert T[T.zhdem].ok.all(), 'сверка r1: расхождение'


if __name__ == '__main__':
    main()
