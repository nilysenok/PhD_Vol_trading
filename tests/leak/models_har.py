"""Адаптеры HAR-J (scripts/analysis/gibrid_har.py) и гибрида V1-V7 (scripts/analysis/gibrid_v17.py) — в процессе."""
import sys
import numpy as np, pandas as pd
from . import noise as N
from .world import REPO, MAIN_COPY, to_long, info_end, load_world

AN = REPO / 'scripts/analysis'
if str(AN) not in sys.path:
    sys.path.insert(0, str(AN))


def _gh():
    import gibrid_har as GH
    return GH


def _harj(w, a):
    jump = [c for c in a.columns if 'jump' in c.lower() and 'target' not in c.lower() and 'idx_' not in c.lower()]
    hs = w.cfg['har_features'] + [c for c in jump if c not in w.cfg['har_features']]
    return [f for f in hs if f in a.columns] + list(w.extra)


def har_frames(w, h):
    a = w.frame()
    hv, ht, _ = _gh().har_forecasts(h, a, _harj(w, a))
    return hv, ht


class Prepared:
    """Общие операции над World (части data/prepared) для адаптеров P1 и HAR-J."""
    info_end = staticmethod(info_end)

    kind = 'model'

    def load(self):
        return load_world()

    def cal(self, base, w):
        return base.cal()

    def permute_steps(self, smoke):
        return [2019, 2021] if smoke else list(range(2017, 2023))  # val-годы 2018, 2020 / разработочный набор 2016-2021

    def steps(self, t, smoke):
        y = pd.Timestamp(t).year
        return [y] if smoke else list(range(2017, y + 1))

    def make(self, base, h, inject=False):
        return base.injected(h) if inject else base

    def noise(self, base, t, h, inject=False):
        w = base.noised(t)  # утечка внедряется ПОСЛЕ шума: признак = цель зашумлённого мира
        return w.injected(h) if inject else w

    def permute(self, base, h, inject=False):
        w = base.permuted()
        return w.injected(h) if inject else w

    def target_frame(self, w, h):
        return w.frame()


class HarJ(Prepared):
    name = 'harj'

    def run(self, w, h, steps, tag=''):
        hv, ht = har_frames(w, h)
        r = pd.concat([to_long('HAR-J', 'val', hv), to_long('HAR-J', 'test', ht)], ignore_index=True)
        return r[r['ty'].isin(steps)]


class Gibrid:
    """Гибрид на готовых компонентах: HAR-J (пересчёт) + xgb и P1(lgb) из data/regen. Шум применяется к входам гибрида:
    rv_pred компонент после t, rv_actual после pos(t)-h. v7_lag: 'h' (без заглядывания) или 1 (исходная реализация)."""
    info_end = staticmethod(info_end)

    kind = 'model'
    load, cal, permute_steps = Prepared.load, Prepared.cal, Prepared.permute_steps

    def __init__(self, v7_lag='h'):
        self.v7_lag = v7_lag
        self.name = 'gibrid' + ('' if v7_lag == 'h' else f'_v7lag{v7_lag}')

    def steps(self, t, smoke):
        y = pd.Timestamp(t).year
        return [y] if smoke else list(range(2017, y + 1))

    def _world(self, base, h):
        import gibrid_io as IO
        hv, ht = har_frames(base, h)
        regen = REPO / 'data/regen'
        bx, bl = IO.load_boost('P1', h, regen if regen.exists() else MAIN_COPY / 'data/regen', 'single')
        return dict(hv=hv, ht=ht, bx=bx, bl=bl, cal=base.cal(), A=base.frame())

    @staticmethod
    def _map(wd, fn):
        f = lambda d: fn(d)
        return dict(hv=f(wd['hv']), ht=f(wd['ht']), bx={k: f(v) for k, v in wd['bx'].items()},
                    bl={k: f(v) for k, v in wd['bl'].items()}, cal=wd['cal'], A=wd['A'])

    def make(self, base, h, inject=False):
        return self._world(base, h)

    def noise(self, base, t, h, inject=False):
        wd = self._world(base, h)
        return self._map(wd, lambda d: N.noise_table(d, t, {'rv_actual': h}, wd['cal']))

    def target_frame(self, wd, h):
        return wd['A']

    def permute(self, base, h, inject=False):
        wd = self._world(base, h)
        tgt = f'rv_target_h{h}'
        A = N.permute_within_ticker(wd['A'], [tgt])
        pm = A[['date', 'ticker', tgt]]

        def fn(d):
            m = d.drop(columns='rv_actual').merge(pm, on=['date', 'ticker'], how='left')
            return m.rename(columns={tgt: 'rv_actual'})[list(d.columns)]
        return {**self._map(wd, fn), 'A': A}

    def run(self, wd, h, steps, tag=''):
        import gibrid_compute as GC, gibrid_harcj as HC  # compute перенесён из gibrid_v17 (Д8); ext - HAR-CJ и признаки гейта
        lag = h if self.v7_lag == 'h' else int(self.v7_lag)
        A = HC.add_cj(wd['A'])
        res = GC.compute(wd['hv'], wd['ht'], h, wd['bx'], wd['bl'], 'bez', lag, [], dict(hcj=HC.forecasts(h, A), state=HC.state(A)))
        out = []
        for r in res:
            for k in r['pv']:
                for kind, frame, pr in (('val', r['vs'], r['pv'][k]), ('test', r['ts'], r['pt'][k])):
                    d = frame[['date', 'ticker', 'year', 'rv_actual']].copy()
                    d['rv_pred'] = np.asarray(pr)
                    out.append(to_long(k, kind, d))
        r = pd.concat(out, ignore_index=True)
        return r[r['ty'].isin(steps)]
