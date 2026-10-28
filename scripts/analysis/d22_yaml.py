#!/usr/bin/env python3
"""Д22: yaml кандидатов третьего круга (формат final3.py) — только для прошедших порог (h=1: слой 2, D22_h1).
python3 scripts/analysis/d22_yaml.py [--h 1 5 22]"""
import argparse
from pathlib import Path

K = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/final3/kandidaty')
Y = {
    1: """boost: D22_h1
h: 1
semeystvo: гибрид Д8.12 на HAR-X⁺ (Д22, слой 2)
rychag: Д22
opisanie: "XGBoost/LightGBM на ln RV + ln HAR-X⁺₁ как признак (Д8.12; HAR-X⁺ = HAR-J + RVI, RV IMOEX, RV рынка, LHAR, PC1, HARQ, RSV±, RRV PK/GK d/w/m, календарь окна, ночной гэп; эластичная сеть, эмбарго); признаки v3 + календарь окна h=1 + RRV + гэп; винзоризация цели 1/99 по тикеру; переподбор по протоколу B v2 (ЛГК LGBM-100 / XGB-40->100 с отсечением, шаги 1-6 подбор, 7-9 точка шага 6, max_bin 255)"
regen: data/regen
legs: [single, top5, seeds3]
odinochnaya: false
chistyy: false
test_dir: data/regen/test_locked/D22_h1
step10_dir: data/regen/D22/step10
step10_lock: data/regen/test_locked/D22_h1/step10
n_variantov: 1
""",
}
TPL = """boost: D22_h{h}
h: {h}
semeystvo: "полный гибрид v3 (Д22)"
rychag: Д22
opisanie: "ridge в логах на окне t-1 по OOS x1..x8 (HAR-X⁺_h, разложение по подгоризонтам, r̂₁±, r̂₁·q, m̂, d̂, итерированный hyb₁{x7}, ē Д23); защита Д14, винзоризация 0,5/99,5; одна модель в обоих слотах"
regen: data/regen
legs: [single, seeds3]
odinochnaya: false
chistyy: false
test_dir: data/regen/test_locked/D22_h{h}
step10_dir: data/regen/D22/step10
step10_lock: data/regen/test_locked/D22_h{h}/step10
n_variantov: 4
"""

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--h', type=int, nargs='*', default=[1])
    a = ap.parse_args()
    for h in a.h:
        txt = Y.get(h) or TPL.format(h=h, x7=', Д16.5' if h == 22 else '')
        (K / f'D22_h{h}.yaml').write_text(txt)
        print('yaml ->', K / f'D22_h{h}.yaml')
