"""Внутренний раннер fastbt (выполняется ПОД cwd = каталог с пакетом fastbt, отдельным процессом):
python3 fastbt_run.py --root DIR --year Y --ticker SBER --sid S3 --out FILE.parquet [--mask-old]
Данные берёт из FASTBT_DATA_ROOT=DIR (копия нового ряда ohlcv_daily_1850.parquet, vpred_aligned.parquet) — как `fastbt.dev --data-root`.
Конфигурация: по умолчанию маска [0] (Config()), --mask-old = Config.gate() (маска [1], поведение c22a4b3)."""
import argparse, dataclasses, os, sys

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--year', type=int, required=True)
ap.add_argument('--ticker', default='SBER')
ap.add_argument('--sid', default='S3')
ap.add_argument('--out', required=True)
ap.add_argument('--mask-old', action='store_true')
a = ap.parse_args()
os.environ['FASTBT_DATA_ROOT'] = a.root
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
sys.path.insert(0, os.getcwd())
from fastbt.config import Config
from fastbt.pipeline import run_unit

cfg = Config.gate() if a.mask_old else Config()
cfg = dataclasses.replace(cfg, years_a=(a.year,), years_bcd=((a.year,) if a.year >= 2021 else ()), val_start=2020)
df, _ = run_unit(a.ticker, a.sid, cfg)
df.to_parquet(a.out, index=False)
