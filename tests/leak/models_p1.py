"""Адаптер P1: настоящий moex_volatility/scripts/regen_boosting.py (вариант P1, LightGBM gbdt + ранняя остановка).
Скрипт копируется во временное дерево рядом с зашумлёнными data/prepared; правки текста копии (с проверкой, что шаблон
найден): шаги = переменная LEAK_STEPS (smoke: один шаг), n_jobs=1 (мак загружен, детерминизм). Сам скрипт не меняется."""
import os, subprocess, sys, tempfile, shutil, json
from pathlib import Path
import pandas as pd
from .world import REPO, MAIN_COPY, to_long
from .models_har import Prepared

SCRIPT = 'moex_volatility/scripts/regen_boosting.py'
STEPS_LINE = "test_years = [y for y in range(2017, 2026) if y in all_data.year.values]"


def _src(rel):
    for r in (REPO, MAIN_COPY):
        if (r / rel).exists():
            return r / rel
    raise FileNotFoundError(rel)


def patched_script():
    s = _src(SCRIPT).read_text()
    assert STEPS_LINE in s and s.count("'n_jobs': -1") == 2, 'regen_boosting.py изменился: обновите шаблоны адаптера P1'
    return s.replace(STEPS_LINE, "test_years = [int(y) for y in os.environ['LEAK_STEPS'].split(',')]").replace("'n_jobs': -1", "'n_jobs': 1")


class P1(Prepared):
    name = 'p1'
    variant = 'P1'

    def steps(self, t, smoke):
        return [pd.Timestamp(t).year]  # P1 — один шаг (тестовый год = год t)

    def run(self, w, h, steps, tag=''):
        tmp = Path(tempfile.mkdtemp(prefix='leak_p1_', dir=os.environ.get('LEAK_TMP') or None))
        try:
            mv = tmp / 'moex_volatility'
            (mv / 'scripts').mkdir(parents=True)
            (mv / 'data/prepared').mkdir(parents=True)
            (mv / 'scripts/regen_boosting.py').write_text(patched_script())
            os.symlink(_src('moex_volatility/scripts/walk_forward_full.py'), mv / 'scripts/walk_forward_full.py')
            os.symlink(_src('moex_volatility/models'), mv / 'models')
            for k, d in w.parts.items():
                d.to_parquet(mv / f'data/prepared/{k}.parquet', index=False)
            json.dump({**w.cfg, 'feature_cols': w.features()}, open(mv / 'data/prepared/config.json', 'w'))
            env = {**os.environ, 'LEAK_STEPS': ','.join(map(str, steps)), 'OMP_NUM_THREADS': '1'}
            r = subprocess.run([sys.executable, str(mv / 'scripts/regen_boosting.py'), '--variant', self.variant, '--h', str(h)],
                               cwd=tmp, env=env, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f'regen_boosting упал ({tag}): {r.stderr[-800:]}')
            o = tmp / 'data/regen'
            return pd.concat([to_long('P1', k, pd.read_parquet(o / f'{self.variant}_h{h}_{k}.parquet')) for k in ('val', 'test')],
                             ignore_index=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
