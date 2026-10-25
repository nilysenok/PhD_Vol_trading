"""«Мир» данных для модельных адаптеров: части data/prepared (train/val/test/walkforward_*) + config.json.
Преобразования (шум, перестановка цели, внедрённая утечка) применяются к склейке частей и режутся обратно."""
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
import numpy as np, pandas as pd
from . import noise as N

REPO = Path(__file__).resolve().parents[2]
MAIN_COPY = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading')  # данные вне копии: только чтение
HS = (1, 5, 22)


def prepared_dir():
    for r in (REPO, MAIN_COPY):
        if (r / 'moex_volatility/data/prepared/config.json').exists():
            return r / 'moex_volatility/data/prepared'
    raise FileNotFoundError('moex_volatility/data/prepared')


@dataclass
class World:
    parts: dict
    cfg: dict
    extra: list = field(default_factory=list)  # внедрённые признаки (самопроверка)

    def big(self):
        return pd.concat(self.parts.values(), ignore_index=True)

    def frame(self):
        """Как gibrid_har.load_all: склейка, сортировка по date, столбец year."""
        a = self.big().sort_values('date').reset_index(drop=True)
        a['year'] = pd.to_datetime(a['date']).dt.year
        return a

    def cal(self):
        return N.calendar(self.big())

    def apply(self, fn, extra=()):
        big = fn(self.big())
        parts, i = {}, 0
        for k, d in self.parts.items():
            parts[k] = big.iloc[i:i + len(d)].reset_index(drop=True)
            i += len(d)
        return replace(self, parts=parts, extra=list(self.extra) + list(extra))

    def noised(self, t):
        cal = self.cal()
        tg = {f'rv_target_h{k}': k for k in HS}
        return self.apply(lambda b: N.noise_table(b, t, tg, cal))

    def permuted(self, seed=42):
        return self.apply(lambda b: N.permute_within_ticker(b, [f'rv_target_h{k}' for k in HS], seed))

    def injected(self, h):
        """Самопроверка: признак = будущая RV (цель той же строки), считается внутри мира (после шума)."""
        return self.apply(lambda b: b.assign(leak_rv=b[f'rv_target_h{h}'].values), extra=['leak_rv'])

    def features(self):
        return list(self.cfg['feature_cols']) + list(self.extra)


def load_world():
    P = prepared_dir()
    parts = {n: pd.read_parquet(P / f'{n}.parquet') for n in ('train', 'val', 'test')}
    for y in range(2020, 2027):
        fp = P / f'walkforward_{y}.parquet'
        if fp.exists():
            d = pd.read_parquet(fp)
            if len(d) > 100:
                parts[f'walkforward_{y}'] = d
    return World(parts, json.load(open(P / 'config.json')))


def info_end(ty):
    """Последняя дата данных, нужная шагу walk-forward с тестовым годом ty (обучение + валидация = годы < ty)."""
    return pd.Timestamp(f'{ty - 1}-12-31')


def to_long(stage, kind, df):
    d = df[['date', 'ticker', 'year', 'rv_actual', 'rv_pred']].copy()
    d['ty'] = d['year'] + (1 if kind == 'val' else 0)
    d['stage'], d['kind'] = stage, kind
    return d
