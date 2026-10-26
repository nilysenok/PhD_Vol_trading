"""Проверка п. 1–3: σ_pred h=1 final3_full, HAR-J h=1, h=5 — побитно против тестовых файлов главы 3."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')
T = R / 'test_locked' / 'final3'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compare(name, file, col, src, srccol):
    f = pd.read_parquet(file)
    s = pd.read_parquet(src)
    print(f'\n== {name}: {file.name} (SHA {sha(file)}) vs √{srccol} {src.name} (SHA {sha(src)})')
    print('  столбцы файла:', list(f.columns), 'dtype', f[col].dtype, 'строк', len(f))
    print('  столбцы источника:', list(s.columns)[:30])
    ref = s.assign(x=np.sqrt(s[srccol].values).astype('float32')).sort_values(['date', 'ticker'])
    ref = ref[['date', 'ticker', 'x']].reset_index(drop=True)
    g = f.sort_values(['date', 'ticker']).reset_index(drop=True)
    print('  строк: файл', len(g), 'источник', len(ref), 'NaN файл', int(g[col].isna().sum()),
          'NaN источник', int(ref.x.isna().sum()))
    print('  период файла', g.date.min().date(), '—', g.date.max().date(),
          '; источника', ref.date.min().date(), '—', ref.date.max().date())
    m = g.merge(ref, on=['date', 'ticker'], how='outer', indicator=True)
    print('  outer merge:', m['_merge'].value_counts().to_dict())
    b = m[m._merge == 'both']
    a1 = b[col].values.astype('float32').view('u4')
    a2 = b.x.values.astype('float32').view('u4')
    eq = a1 == a2
    nan_both = np.isnan(b[col].values) & np.isnan(b.x.values)
    eq = eq | nan_both
    print('  побитно равных:', int(eq.sum()), 'из', len(b), '; файл dtype исходный', f[col].dtype)
    by = pd.DataFrame({'y': b.date.dt.year.values, 'eq': eq}).groupby('y').eq.agg(['sum', 'size'])
    print('  по годам (равных/всего):', {int(k): f'{int(v["sum"])}/{int(v["size"])}' for k, v in by.iterrows()})
    if (~eq).any():
        bad = b[~eq]
        print('  примеры расхождений:\n', bad.head(10).to_string())
        print('  max|Δ|', float(np.nanmax(np.abs(bad[col].values - bad.x.values))))
    only = m[m._merge != 'both']
    if len(only):
        print('  строки только в одном:\n', only[['date', 'ticker', '_merge']].head(20).to_string())
    return bool(eq.all()) and len(only) == 0


ok = {}
ok['h1'] = compare('σ_pred h=1 final3_full', R / 'sigma_pred_h1_final3_full.parquet', 'sigma_pred_h1',
                   T / 'final_h1_test.parquet', 'rv_pred')
ok['harj'] = compare('σ HAR-J h=1', R / 'sigma_pred_h1_harj_final2.parquet', 'sigma_pred_h1',
                     T / 'final_h1_test.parquet', 'har_corr')
ok['h5'] = compare('σ_pred h=5', R / 'sigma_pred_h5_final2.parquet', 'sigma_pred_h5',
                   T / 'final_h5_test.parquet', 'rv_pred')
# Файлы на диске побайтно = результату кода сборки (повтор кода сборки в память, без записи)
f = pd.read_parquet(R / 'sigma_pred_h1_final3_full.parquet')
s = pd.read_parquet(T / 'final_h1_test.parquet', columns=['date', 'ticker', 'step', 'rv_pred'])
sp = s.assign(sigma_pred_h1=np.sqrt(s.rv_pred.values).astype('float32')).sort_values(['date', 'ticker'])
x = sp[['date', 'ticker', 'sigma_pred_h1']].reset_index(drop=True)
print('\nfinal3_full.equals(код сборки в памяти):', f.reset_index(drop=True).equals(x),
      '; dtypes', dict(f.dtypes.astype(str)))
# final2 h1 (старый гибрид) — для справки: отличается ли от final3_full
f2 = pd.read_parquet(R / 'sigma_pred_h1_final2.parquet')
mm = f.merge(f2, on=['date', 'ticker'], suffixes=('', '_f2'))
print('final2 h1 vs final3_full: общих строк', len(mm), 'равных',
      int((mm.sigma_pred_h1.values == mm.sigma_pred_h1_f2.values).sum()))
print('\nИТОГ', ok)
sys.exit(0 if all(ok.values()) else 1)
