#!/usr/bin/env python3
"""Д17 (агент F): выжившие кандидаты h=1 — полный прогон шагов 1–9 в формате кандидата третьего круга.

Обёртка над d12_h1.py без его правки: те же точки (top5 ЛГК B v2b на шагах 1–6, точка шага 6 на 7–9, seed 43/44),
то же обучение l2_lib.fit, та же запись val/test (test_locked/D12_h1/<boost> + ссылка test_locked/<boost>).
Добавка — признаки RRV из features_pit_v3_rrv.parquet (строки и старые колонки = v3 побайтно):
  A_d812_rrvgk  {"harf": true} + rrv_gk_d/w/m      (основа D12_h1_A_d812)
  V0_rrvgk      {}             + rrv_gk_d/w/m      (основа D12_h1_V0)
  A_d812_rrvpk / V0_rrvpk — то же с Parkinson.
Метрики печатаются только по шагам 2–6 (d12_h1.evaluate_dev); val 7–9 и тест не оцениваются.
python3 scripts/analysis/d17_f_final.py --cands A_d812_rrvgk [--workers 3]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L  # noqa: E402
import d12_h1  # noqa: E402

RRV = {'rrvgk': ['rrv_gk_d', 'rrv_gk_w', 'rrv_gk_m'], 'rrvpk': ['rrv_pk_d', 'rrv_pk_w', 'rrv_pk_m']}
for base, cfg in (('A_d812', {'harf': True}), ('V0', {})):
    for k in RRV:
        d12_h1.CANDS[f'{base}_{k}'] = dict(cfg, d17_rrv=k)  # ключ d17_rrv l2_lib не читает

_bf = L.build_features


def build_features(A, FC, h, cfg):
    A2, cols, cats, mono = _bf(A, FC, h, cfg)
    return A2, cols + [c for c in RRV.get(cfg.get('d17_rrv'), []) if c not in cols], cats, mono


L.build_features = build_features

if __name__ == '__main__':
    if '--matrix' not in sys.argv:
        sys.argv += ['--matrix', str(L.REGEN / 'features_pit_v3_rrv.parquet')]
    if '--prefix' not in sys.argv:
        sys.argv += ['--prefix', 'D17F_h1']
    (L.REGEN / 'D17F_h1_logs').mkdir(exist_ok=True)
    d12_h1.main()
