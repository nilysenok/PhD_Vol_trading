"""Адаптер fastbt (ветка origin/fastbt): раннер tests/leak/fastbt_run.py в отдельном процессе поверх зашумлённого --data-root.
Контракт smoke: `python3 -m fastbt.dev --smoke --data-root DIR` (1 тикер x 1 стратегия) проверяется в load().
mask_old=True — режим `--leak-mask-old` (маска [1]); False — исправленная маска [0] (по умолчанию)."""
import io, os, shutil, subprocess, sys, tarfile, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from .registry import NotConnected
from .world import REPO
from .strat_v4 import V4, data_dir

RUNNER = Path(__file__).resolve().parent / 'fastbt_run.py'
_SRC = {}


def fastbt_src():
    """Каталог, в котором лежит пакет fastbt: репозиторий (после слияния) или распаковка origin/fastbt."""
    if 'p' not in _SRC:
        if os.environ.get('FASTBT_SRC'):
            _SRC['p'] = Path(os.environ['FASTBT_SRC'])
        elif (REPO / 'fastbt/dev.py').exists():
            _SRC['p'] = REPO
        else:
            try:
                tar = subprocess.check_output(['git', 'archive', 'origin/fastbt', 'fastbt'], cwd=REPO, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                raise NotConnected('fastbt не найден: нет origin/fastbt и FASTBT_SRC')
            d = Path(tempfile.mkdtemp(prefix='leak_fastbt_', dir=os.environ.get('LEAK_TMP') or None))
            tarfile.open(fileobj=io.BytesIO(tar)).extractall(d)
            _SRC['p'] = d
    return _SRC['p']


class Fastbt(V4):
    kind = 'strategy'

    def __init__(self, mask_old=False, ticker='SBER', sid='S3'):
        super().__init__(1 if mask_old else 0, ticker, sid)
        self.mask_old = mask_old
        self.name = 'fastbt' + ('_mask_old' if mask_old else '')

    def load(self):
        src = fastbt_src()
        base = super().load()
        tmp = Path(tempfile.mkdtemp(prefix='leak_fb_smoke_'))
        try:
            self._write(base, tmp)
            r = subprocess.run([sys.executable, '-m', 'fastbt.dev', '--smoke', '--data-root', str(tmp)], cwd=src,
                               capture_output=True, text=True, env={**os.environ, 'NUMBA_NUM_THREADS': '1'})
            if r.returncode:
                raise RuntimeError(f'`python3 -m fastbt.dev --smoke` упал: {r.stderr[-600:]}')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return base

    @staticmethod
    def _write(wd, d):
        bars = wd['bars'].copy()
        bars.to_parquet(d / 'ohlcv_daily_full.parquet', index=False)
        wd['vp'].to_parquet(d / 'vpred_aligned.parquet', index=False)

    def run(self, wd, h, steps, tag=''):
        src, out = fastbt_src(), []
        for Y in steps:
            tmp = Path(tempfile.mkdtemp(prefix='leak_fb_'))
            try:
                self._write(wd, tmp)
                cmd = [sys.executable, str(RUNNER), '--root', str(tmp), '--year', str(Y), '--ticker', self.ticker,
                       '--sid', self.sid, '--out', str(tmp / 'pos.parquet')] + (['--mask-old'] if self.mask_old else [])
                r = subprocess.run(cmd, cwd=src, capture_output=True, text=True)
                if r.returncode:
                    raise RuntimeError(f'fastbt упал ({tag}): {r.stderr[-800:]}')
                p = pd.read_parquet(tmp / 'pos.parquet')
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            d = pd.DataFrame({'stage': p['approach'], 'kind': 'test', 'ty': Y, 'date': p['date'], 'ticker': p['ticker'],
                              'rv_pred': p['position'].astype(float), 'rv_actual': np.nan, 'year': p['test_year']})
            out.append(d)
        return pd.concat(out, ignore_index=True)
