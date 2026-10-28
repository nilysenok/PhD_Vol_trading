#!/usr/bin/env python3
"""σ_pred(h=1) final3 на весь период стратегий 2017–02.2026 тем же кодом сборки, что прогнозы главы 3.

Код — дословно фаза build из scripts/analysis/final3.py (строки 638–644, общий файл не правится):
σ = √rv_pred из test_locked/final3/final_h1_test.parquet (финал D22_h1 top5|V0_XGB|B-sm|L0, шаги 1–10),
float32, сортировка (date, ticker). Выход — data/regen/sigma_pred_h1_final3_full.parquet (+ _steps.csv,
_sha256.json); проверка: на 2017–2021 побитно равен sigma_pred_h1_final3.parquet (разработка не меняется),
и сверка на всех годах. Метрики не считаются (поручение владельца 26.10, 09:50, п. 1; решение 10:05, п. 6–8).
python3 scripts/analysis/sigma_pred_h1_final3_full.py [--check-only]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')
SRC = R / 'test_locked' / 'final3' / 'final_h1_test.parquet'
REF = R / 'sigma_pred_h1_final3.parquet'
OUT = R / 'sigma_pred_h1_final3_full.parquet'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bitwise(a, b, lo, hi):
    ma = a.date.dt.year.between(lo, hi).values
    mb = b.date.dt.year.between(lo, hi).values
    x, y = a[ma].reset_index(drop=True), b[mb].reset_index(drop=True)
    if len(x) != len(y) or not x[['date', 'ticker']].equals(y[['date', 'ticker']]):
        return False, len(x), len(y)
    return bool(np.array_equal(x.sigma_pred_h1.values.view('u4'), y.sigma_pred_h1.values.view('u4'))), len(x), len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-only', action='store_true')
    a = ap.parse_args()
    if not a.check_only:
        # --- final3.py, фаза build, σ_pred(h=1) ---
        T = pd.read_parquet(SRC, columns=['date', 'ticker', 'step', 'rv_pred'])
        sp = T.assign(sigma_pred_h1=np.sqrt(T.rv_pred.values).astype('float32')).sort_values(['date', 'ticker'])
        sp[['date', 'ticker', 'sigma_pred_h1']].to_parquet(OUT, index=False)
        sp.groupby('step').date.agg(['min', 'max', 'size']).to_csv(OUT.with_name('sigma_pred_h1_final3_full_steps.csv'))
        # ---
    f, ref = pd.read_parquet(OUT), pd.read_parquet(REF)
    chk = {f'{lo}-{hi}': bitwise(f, ref, lo, hi) for lo, hi in ((2017, 2021), (2022, 2026), (2017, 2026))}
    info = dict(path=str(OUT), sha256=sha(OUT), rows=len(f), first=str(f.date.min().date()), last=str(f.date.max().date()),
                source=str(SRC), source_sha256=sha(SRC), ref=str(REF), ref_sha256=sha(REF),
                bitwise_vs_final3={k: dict(equal=v[0], rows_full=v[1], rows_final3=v[2]) for k, v in chk.items()},
                rows_by_year={int(k): int(v) for k, v in f.groupby(f.date.dt.year).size().items()},
                nan=int(f.sigma_pred_h1.isna().sum()))
    if not a.check_only:
        OUT.with_name('sigma_pred_h1_final3_full_sha256.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return 0 if chk['2017-2021'][0] else 1


if __name__ == '__main__':
    sys.exit(main())
