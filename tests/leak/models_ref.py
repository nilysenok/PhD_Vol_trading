"""Адаптеры справочных моделей ночного приоритета 2 (26.10) на features_pit_v3: lstm, gru (scripts/analysis/ref_nn_sv.py) и
ref_garch (GARCH и GJR с калибровкой, scripts/analysis/ref_garch_cal.py). Код моделей вызывается как есть, в процессе.
Мир — таблица features_pit_v3 (date, ticker, close, 21 признак, цели). Шум — noise.noise_table: признаки строк после t, цели —
строк календаря > pos(t) - h. Шаг k: vy = 2015 + k, ty = 2016 + k; info_end(ty) = 31.12 года ty-1 (= конец vy).
NN: бюджет детектора уменьшен (LEAK_NN_CFG: 3 испытания, 4 эпохи, patience 2), подмножество тикеров LEAK_NN_TICKERS, три seed;
путь кода (эмбарго, стандартизация, исследование шага t-1 -> переобучение шага t, Б0.1 по шагу t-1) — тот же.
Внедрённая утечка (самопроверка): NN — признак leak_rv = цель строки; GARCH — калибровка шага t по году оценки шага t+1."""
import sys
import numpy as np, pandas as pd
from . import noise as N
from .world import REPO

AN = REPO / 'scripts/analysis'
if str(AN) not in sys.path:
    sys.path.insert(0, str(AN))
FEAT = '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/features_pit_v3.parquet'
LEAK_NN_TICKERS = ['SBER', 'LKOH', 'IRAO']
LEAK_NN_CFG = dict(ntr=3, epochs=4, pat=2, batch=32)
tcol = lambda h: 'rv_target_h1' if h == 1 else f'rv_target_mean_h{h}'
TG = {tcol(h): h for h in (1, 5, 22)}


def info_end(ty):
    return pd.Timestamp(f'{int(ty) - 1}-12-31')


class RefBase:
    kind = 'model'
    info_end = staticmethod(info_end)
    cols = ['close']

    def load(self):
        import ref_nn_sv as R
        d = pd.read_parquet(FEAT, columns=['date', 'ticker'] + sorted(set(self.cols + R.FC)) + list(TG))
        d['date'] = pd.to_datetime(d.date)
        d['year'] = d.date.dt.year
        return d.sort_values(['ticker', 'date']).reset_index(drop=True)

    def cal(self, base, w):
        return N.calendar(base)

    def steps(self, t, smoke):
        return list(range(1, pd.Timestamp(t).year - 2016 + 1))  # шаги с ty <= год t (цепочке нужны все предыдущие)

    def permute_steps(self, smoke):
        return list(range(1, 7))  # val-годы 2016-2021 (разработочный набор + шаг 1)

    @staticmethod
    def _inject(w, h):
        return w.assign(leak_rv=w[tcol(h)].values)

    def make(self, base, h, inject=False):
        return self._inject(base, h) if inject else base

    def noise(self, base, t, h, inject=False):
        w = N.noise_table(base, t, TG, N.calendar(base))
        return self._inject(w, h) if inject else w  # утечка внедряется после шума

    def permute(self, base, h, inject=False):
        w = N.permute_within_ticker(base, [tcol(h)])
        return self._inject(w, h) if inject else w

    def target_frame(self, w, h):
        return w.assign(**{f'rv_target_h{h}': w[tcol(h)].values})  # compare_permute читает rv_target_h{h}

    @staticmethod
    def _long(stage, E):
        d = E[['date', 'ticker', 'rv_actual', 'rv_pred']].copy()
        d['date'] = pd.to_datetime(d['date'])
        d['ty'] = (E['step'] + 2016).values
        d['stage'], d['kind'] = stage, E['part'].values
        return d


class RefNN(RefBase):
    def __init__(self, model):
        self.model, self.name = model, model.lower()

    def run(self, w, h, steps, tag=''):
        import ref_nn_sv as R
        cfg0, fc0 = dict(R.CFG), list(R.FC)
        R.CFG.update(LEAK_NN_CFG)
        if 'leak_rv' in w.columns:
            R.FC = fc0 + ['leak_rv']
        try:
            frs = {tk: g.sort_values('date').reset_index(drop=True) for tk, g in w[w.ticker.isin(LEAK_NN_TICKERS)].groupby('ticker')}
            E = R.chain(self.model, frs, h, steps)
        finally:
            R.CFG.clear(); R.CFG.update(cfg0); R.FC = fc0
        return self._long(self.model, E)


class RefGarch(RefBase):
    name = 'ref_garch'

    def run(self, w, h, steps, tag=''):
        import ref_garch_cal as G
        leak = 'leak_rv' in w.columns
        out, _, _ = G.run(w[['date', 'ticker', 'close'] + list(TG)], range(1, max(steps) + 1 + leak), leak=leak)
        r = pd.concat([self._long(m, out[(m, h)][out[(m, h)].step.isin(steps)]) for m in G.MODELS], ignore_index=True)
        return r.dropna(subset=['rv_pred'])  # val шага 1 без калибровки (нет шага 0) — вне сравнения
