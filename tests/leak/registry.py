"""Реестр адаптеров. Модели: p1, harj, gibrid (V7 lag=h), gibrid_v7lag1 (исходная реализация V7), pipeline (позже).
Стратегии: v4 (исправленная маска [0]), v4_mask1 (старая маска [1]), fastbt (маска [0]), fastbt_mask_old (--leak-mask-old).
Адаптер = объект с методами: load(), cal(base, w), make/noise/permute(base, ..., inject), run(w, h, steps), steps(t, smoke),
info_end(ty); для моделей ещё target_frame(w, h) и permute_steps(smoke)."""


class NotConnected(Exception):
    pass


class Pipeline:
    """Будущий конвейер моделей: `python3 -m models run --design <yaml> --smoke` (ветка origin/models-pipeline).
    Контракт: прогоны val/test по шагам в длинной таблице (stage, kind, ty, date, ticker, rv_actual, rv_pred)."""
    name = 'pipeline'

    def __init__(self, design=None):
        self.design = design

    def load(self):
        raise NotConnected('конвейер `python3 -m models run --design <yaml> --smoke` ещё не появился (ветка models-pipeline)')


def models(name):
    from .models_p1 import P1
    from .models_har import HarJ, Gibrid
    return {'p1': P1, 'harj': HarJ, 'gibrid': lambda: Gibrid('h'), 'gibrid_v7lag1': lambda: Gibrid(1),
            'pipeline': Pipeline}[name]()


def strategies(name):
    from .strat_v4 import V4
    from .strat_fastbt import Fastbt
    return {'v4': lambda: V4(0), 'v4_mask1': lambda: V4(1), 'fastbt': lambda: Fastbt(False),
            'fastbt_mask_old': lambda: Fastbt(True)}[name]()
