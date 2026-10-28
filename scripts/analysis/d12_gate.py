#!/usr/bin/env python3
"""Д12: ворота и критерии c1-c4 гибрида (oos, шаги 2-6, ноги single/top5/seeds3, эталон — та же s2) на выходах optuna-30
data/regen/{boost}_{xgb|lgbm}_h{h}_*.parquet; HAR-J — features_pit_v3. Тест не читается (gibrid_io: test закрыт).
python3 scripts/analysis/d12_gate.py --boost D12_h22 --h 22"""
import argparse, subprocess, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--boost', required=True); ap.add_argument('--h', type=int, required=True)
    ap.add_argument('--features', default=str(L.REGEN / 'features_pit_v3.parquet'))
    a = ap.parse_args()
    out, dl = L.REGEN / 'D12' / f'gibrid_{a.boost}_opt', L.REGEN / 'D12' / 'gibrid_devlog.md'
    r = subprocess.run([sys.executable, '-W', 'ignore', str(Path(__file__).resolve().parent / 'gibrid_v17.py'), '--boost', a.boost, '--h', str(a.h),
                        '--regen', str(L.REGEN), '--features', a.features, '--out', str(out), '--devlog', str(dl)], capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-2000:]); return
    for ln in [x for x in dl.read_text().splitlines() if x.startswith((f'h={a.h}: ворота', f'h={a.h}: выбор по правилу'))][-2:]:
        print(ln)
    rk = pd.read_csv(out / f'ranking_{a.boost}_h{a.h}.csv')
    print(rk.head(5).to_string())
    print(rk[rk.design.str.contains(r'\|V0_(?:XGB|LGB)\|')].head(8).to_string())


if __name__ == '__main__':
    main()
