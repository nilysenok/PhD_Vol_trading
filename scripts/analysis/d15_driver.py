#!/usr/bin/env python3
"""Д15: кумулятивный проход вариантов на одном h (каждый — optuna-30 seed 42 через d15_tune.py, затем порог по 3 seed через d15_gate.py).
Порядок (постановка владельца 25.10, 21:40): (а) linear_tree λ ∈ {0,1; 1} → (б) цель ln(RV_avg/RV_m) → (в) узкое пространство Д9.3 →
(г) монотонность +1 / −1 на rv/rv_d/rv_w/rv_m/bv. Внутри рычага с двумя значениями берётся лучшее из прошедших порог.
Итог — копия файлов принятой комбинации под тегом D15_h{h} (val в data/regen, test — data/regen/test_locked/D15_h{h}/, без чтения).
python3 scripts/analysis/d15_driver.py --h 5 --workers 3   (база D15v_h5_base должна быть посчитана)"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault('L2_FEAT', '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/features_pit_v3.parquet')
import pandas as pd
import d15_gate as G
import l2_lib as L

LEVERS = [('a', [('lin0.1', {'linear_lambda': 0.1}, None), ('lin1', {'linear_lambda': 1}, None)]),
          ('b', [('rel', {'rel': True}, None)]),
          ('c', [('d93', {}, 'd93')]),
          ('d', [('mono+1', {'mono': True}, None), ('mono-1', {'mono': True, 'mono_sign': -1}, None)]),
          ('e', [('d7', {'blocks': 'd7'}, None)])]  # (д) только h=5: блок Д7 (внешние 5-дневные агрегаты, PC1, RVI, календарь; без HAR-J) — допущен постановкой


def run(h, tag, cfg, space, workers):
    log = G.OUT / f'{tag}.log'
    if log.exists() and 'готов' in log.read_text() and (L.REGEN / f'{tag}_xgb_h{h}_seeds4344_val.parquet').exists():
        return log.read_text().strip().splitlines()[-1] + ' [уже посчитан]'
    cmd = [sys.executable, str(HERE / 'd15_tune.py'), '--h', str(h), '--space', space, '--cfg', json.dumps(cfg), '--tag', tag, '--workers', str(workers)]
    if cfg.get('blocks'):
        cmd += ['--blocks', cfg['blocks']]
    if (L.REGEN / f'{tag}_trials.csv').exists():  # обрыв на дообучении — досчёт по trials.csv
        cmd.append('--resume')
    with open(log, 'a') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, env=os.environ)
    return log.read_text().strip().splitlines()[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, required=True); ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--only', default='', help='подмножество рычагов, напр. "a,b"')
    a = ap.parse_args()
    h = a.h
    cur, cur_cfg, cur_space = f'D15v_h{h}_base', {}, 'B'
    state = G.OUT / f'state_h{h}.json'
    if state.exists():
        s = json.loads(state.read_text()); cur, cur_cfg, cur_space = s['cur'], s['cfg'], s['space']
    for lev, alts in LEVERS:
        if a.only and lev not in a.only.split(','):
            continue
        res = []
        for name, mod, sp in alts:
            cfg = {**cur_cfg, **mod}
            space = sp or cur_space
            tag = f'D15v_h{h}_{lev}_{name.replace("+", "p").replace("-", "m")}'
            t0 = time.time()
            last = run(h, tag, cfg, space, a.workers)
            print(f'{time.strftime("%H:%M")} {tag}: {last} ({time.time() - t0:.0f} с)', flush=True)
            subprocess.run([sys.executable, str(HERE / 'd15_gate.py'), '--h', str(h), '--tag', tag, '--cur', cur, '--label', f'({lev}) {name}; cfg={cfg}; space={space}'], check=True)
            r = pd.read_csv(G.OUT / f'levers_h{h}.csv').iloc[-1]
            res.append((r['mean'], bool(r['accepted']), tag, cfg, space))
        ok = [x for x in res if x[1]]
        if ok:
            _, _, cur, cur_cfg, cur_space = min(ok)
            print(f'рычаг ({lev}) принят: {cur} cfg={cur_cfg} space={cur_space}', flush=True)
        else:
            print(f'рычаг ({lev}) отброшен; текущая {cur}', flush=True)
        state.write_text(json.dumps(dict(cur=cur, cfg=cur_cfg, space=cur_space)))
    fin = f'D15_h{h}'
    lock = L.REGEN / 'test_locked'
    (lock / fin).mkdir(parents=True, exist_ok=True)
    n = 0
    for d, sub in ((L.REGEN, None), (lock / cur, lock / fin)):
        for p in sorted(d.glob(f'{cur}_*')):
            shutil.copy2(p, (sub or L.REGEN) / p.name.replace(cur, fin, 1)); n += 1
    (G.OUT / f'final_h{h}.json').write_text(json.dumps(dict(final=fin, source=cur, cfg=cur_cfg, space=cur_space, files=n)))
    print(f'итог h={h}: {fin} = {cur} cfg={cur_cfg} space={cur_space}; скопировано файлов {n}', flush=True)


if __name__ == '__main__':
    main()
