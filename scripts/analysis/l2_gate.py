#!/usr/bin/env python3
"""Уровень 2: ворота бустинга на прогнозах варианта рычага (val шагов 1-6) — сборка гибрида gibrid_v17.py (oos, шаги 2-6, тест закрыт).
python3 scripts/analysis/l2_gate.py --h 22 --name a_v9 [--boost L2v_h22_a_v9]
Прогнозы {name}_val.parquet раскладываются в data/regen/L2/gate/{boost}_{xgb|lgbm}_h{h}_val.parquet (нога single; top5/seeds3 нет)."""
import argparse, subprocess, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--h', type=int, required=True); ap.add_argument('--name', required=True); ap.add_argument('--boost', default=None)
    a = ap.parse_args()
    b = a.boost or f'L2v_h{a.h}_{a.name}'
    gd = L.L2DIR / 'gate'; gd.mkdir(parents=True, exist_ok=True)
    V = pd.read_parquet(L.L2DIR / f'h{a.h}' / f'{a.name}_val.parquet')
    for m in ('xgb', 'lgbm'):
        V[V.model == m].drop(columns='model').to_parquet(gd / f'{b}_{m}_h{a.h}_val.parquet', index=False)
    out = L.L2DIR / f'gibrid_{b}'
    r = subprocess.run([sys.executable, '-W', 'ignore', str(Path(__file__).resolve().parent / 'gibrid_v17.py'), '--boost', b, '--h', str(a.h),
                        '--regen', str(gd), '--out', str(out), '--devlog', str(L.L2DIR / 'gibrid_devlog.md')], capture_output=True, text=True)
    print(r.stdout[-600:], r.stderr[-1500:] if r.returncode else '')
    txt = (L.L2DIR / 'gibrid_devlog.md').read_text().splitlines()
    for ln in [x for x in txt if x.startswith(f'h={a.h}: ворота') or x.startswith(f'h={a.h}: выбор по правилу')][-2:]:
        print(ln)


if __name__ == '__main__':
    main()
