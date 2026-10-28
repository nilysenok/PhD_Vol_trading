"""Адаптер стратегий: СТАРЫЙ код c22a4b3 (strategies_walkforward_v4.py + strategies_walkforward.py, s5_rerun.py, s5s6_rerun.py),
один тикер, одна стратегия, дневной ТФ. mask=1 — исходный код (`_, train_end = _year_bounds(...)` = индекс 1 янв Y+1, окно
включает год теста); mask=0 — исправленная маска: во всех строках `_, X = _year_bounds(d, Y)` берётся `[0]`."""
import importlib.util, os, re, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from . import noise as N
from .world import REPO, MAIN_COPY
from .models_har import Prepared

COMMIT = 'c22a4b3'
FILES = ('strategies_walkforward', 'strategies_walkforward_v4', 's5_rerun', 's5s6_rerun')
_PAT = re.compile(r'^(\s*)_, (\w+) = _year_bounds\((.*)\)\s*$', re.M)
_CACHE = {}


def _scripts_dir():
    if 'dir' not in _CACHE:
        d = Path(tempfile.mkdtemp(prefix='leak_v4_', dir=os.environ.get('LEAK_TMP') or None))
        for f in FILES:
            (d / f'{f}.py').write_bytes(subprocess.check_output(
                ['git', 'show', f'{COMMIT}:moex_volatility/scripts/{f}.py'], cwd=REPO))
        _CACHE['dir'] = d
    return _CACHE['dir']


def v4_module(mask):
    if mask not in _CACHE:
        d = _scripts_dir()
        src = (d / 'strategies_walkforward_v4.py').read_text()
        if mask == 0:
            src, n = _PAT.subn(r'\1\2 = _year_bounds(\3)[0]', src)
            assert n >= 8, f'маска [0]: заменено {n} строк, ожидалось >= 8'
        f = d / f'v4_mask{mask}.py'
        f.write_text(src)
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
        spec = importlib.util.spec_from_file_location(f'v4_mask{mask}', f)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _CACHE[mask] = m
    return _CACHE[mask]


def data_dir():
    for r in (REPO, MAIN_COPY):
        if (r / 'moex_volatility/results/final/strategies/data/ohlcv_daily_full.parquet').exists():
            return r / 'moex_volatility/results/final/strategies/data'
    raise FileNotFoundError('moex_volatility/results/final/strategies/data')


class V4(Prepared):
    """Выходы: позиции (stage = подход A/B/C/D, kind = test, ty = год теста, rv_pred = позиция)."""

    def __init__(self, mask=0, ticker='SBER', sid='S3'):
        self.mask, self.ticker, self.sid = mask, ticker, sid
        self.name = f'v4_mask{mask}'

    kind = 'strategy'

    def load(self):
        D = data_dir()
        d = pd.read_parquet(D / 'ohlcv_daily_full.parquet')
        d['date'] = pd.to_datetime(d['date'])
        vp = pd.read_parquet(D / 'vpred_aligned.parquet')
        vp['date'] = pd.to_datetime(vp['date'])
        return dict(bars=d[d.ticker == self.ticker].sort_values('date').reset_index(drop=True), vp=vp)

    def cal(self, base, wd):
        return N.calendar(wd['bars'])

    def steps(self, t, smoke):
        return [pd.Timestamp(t).year]

    def make(self, base, h, inject=False):
        return base

    def noise(self, base, t, h, inject=False):
        return dict(bars=N.noise_bars(base['bars'], t), vp=N.noise_sigma(base['vp'], t))

    def run(self, wd, h, steps, tag=''):
        m, out = v4_module(self.mask), []
        for Y in steps:
            m.STRATEGY_IDS, m.TIMEFRAMES = [self.sid], ['daily']
            m.A_TEST_YEARS, m.BCD_TEST_YEARS = [Y], ([Y] if Y >= 2021 else [])  # val B/C/D жёстко с 2020 г.
            r = m.process_ticker_v4(self.ticker, wd['bars'], wd['bars'].iloc[:0], wd['vp'])
            rows = pd.DataFrame(r['positions'], columns=['date', 's', 'tf', 'ticker', 'stage', 'year', 'rv_pred', 'ret'])
            rows['kind'], rows['ty'], rows['rv_actual'] = 'test', Y, np.nan
            out.append(rows[['stage', 'kind', 'ty', 'date', 'ticker', 'rv_pred', 'rv_actual', 'year']])
        return pd.concat(out, ignore_index=True)
