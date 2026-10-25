"""Ворота: прогнозы конвейера = сохранённые прогнозы существующего кода (B_* tune_B.py, hyb_* gibrid_v17.py) до 1e-9.
Сравнение тестовых прогнозов здесь — тождество файлов, не метрики (метрики теста — только --scope test)."""
from pathlib import Path
import numpy as np, pandas as pd
from . import cache as K, hybrid as HY

KEYS = ['date', 'ticker', 'step']


def _cmp(name, mine, ref, steps):
    ref = ref[ref.step.isin(steps)]
    m = mine.merge(ref[KEYS + ['rv_pred', 'rv_actual']], on=KEYS, suffixes=('', '_ref'), how='outer', indicator=True)
    both = m[m._merge == 'both']
    d = np.abs(both.rv_pred - both.rv_pred_ref)
    return dict(check=name, n_mine=len(mine), n_ref=len(ref), n_matched=len(both), max_abs_diff=float(d.max()) if len(d) else np.nan,
                max_rel_diff=float((d / both.rv_pred_ref.abs()).max()) if len(d) else np.nan,
                ok=bool(len(both) == len(mine) == len(ref) and len(d) and d.max() <= 1e-9))


def _pred(scope, kind, keys):
    return pd.concat([pd.read_parquet(K.pred_path(scope, kind, k)) for k in keys], ignore_index=True)


def boost(man, ref_dir, prefix='B_'):
    rows = []
    for mh, keys in man['boost'].items():
        m, h = mh.split('_')
        for s in ('val', 'test'):
            nm = f'{prefix}{m}_h{h}_{s}.parquet'
            f = next((x for x in (Path(ref_dir) / nm, Path(ref_dir) / 'test_locked' / nm) if x.exists()), None)
            if f is not None:
                rows.append(_cmp(f'{f.name}', _pred(s, 'boost', keys), pd.read_parquet(f), man['steps']))
    return rows


def hybrid(man, ref_dir, boost_name):
    rows, ref_dir = [], Path(ref_dir)
    fc = ref_dir / f'final_choice_{boost_name}.csv'
    fin_ref = pd.read_csv(fc).set_index('h').design.to_dict() if fc.exists() else {}
    dv = ref_dir / 'designs_val.csv'
    for h, hk in man['hybrid'].items():
        h = int(h)
        fin = K.meta('hybrid', hk)['final']
        pairs = [(HY.CODE_DESIGN, f'hyb_{boost_name}_h{h}')]
        if fin_ref.get(h) == fin:
            pairs.append((fin, f'hyb_{boost_name}_final_h{h}'))
        elif fin_ref:
            rows.append(dict(check=f'final h{h}', ok=False, n_mine=0, n_ref=0, n_matched=0, max_abs_diff=np.nan, max_rel_diff=np.nan,
                             note=f'правило: конвейер {fin}, gibrid {fin_ref.get(h)}'))
        for des, base in pairs:
            for s in ('val', 'test'):
                f = ref_dir / f'{base}_{s}.parquet'
                if f.exists():
                    rows.append(_cmp(f.name, pd.read_parquet(K.pred_path(s, 'hyb', f'{hk}_{HY.fname(des)}')), pd.read_parquet(f), man['steps']))
        if dv.exists():  # q_hyb по всем дизайнам single на разработочном наборе
            R = pd.read_csv(K.stage_dir('hybrid', hk) / 'designs.csv')
            ref = pd.read_csv(dv)
            ref = ref[(ref.h == h) & ref.design.str.startswith('single|') & ref.step.isin(man['steps']) & ref.q_hyb.notna()]
            m = R.merge(ref[['design', 'step', 'q_hyb', 'scheme']], on=['design', 'step'], suffixes=('', '_ref'))
            d = np.abs(m.q_hyb - m.q_hyb_ref)
            rows.append(dict(check=f'designs_val h{h} q_hyb+scheme', n_mine=len(R), n_ref=len(ref), n_matched=len(m), max_abs_diff=float(d.max()),
                             max_rel_diff=float((d / m.q_hyb_ref).max()), ok=bool(len(m) == len(ref) and d.max() <= 1e-9 and (m.scheme == m.scheme_ref).all())))
    return rows
