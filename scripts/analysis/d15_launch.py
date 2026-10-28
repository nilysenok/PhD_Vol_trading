#!/usr/bin/env python3
"""Д15: запуск d15_chain.sh в отдельной сессии (не снимается вместе с группой процессов вызывающей оболочки).
python3 scripts/analysis/d15_launch.py <h> [workers] -> pid; лог data/regen/D15/driver_h<h>.log"""
import subprocess, sys
from pathlib import Path

h, w = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else '3')
log = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/D15') / f'driver_h{h}.log'
p = subprocess.Popen(['bash', str(Path(__file__).resolve().parent / 'd15_chain.sh'), h, w] + sys.argv[3:4],stdout=open(log, 'a'), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
print(p.pid)
