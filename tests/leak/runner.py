"""Прогон проверок (а) «будущее не влияет» и (б) «сдвинутая цель» для одного адаптера."""
import time
from .compare import Verdict, compare_future, compare_permute
from .registry import NotConnected

_RUNS = {}  # кэш прогонов в процессе: самопроверки переиспользуют результаты основного запуска


def _cached(key, fn):
    if key not in _RUNS:
        _RUNS[key] = fn()
    return _RUNS[key]


def _base(ad):
    return _cached((ad.name, 'base'), ad.load)


def check_future(ad, t, h, smoke, inject=False, log=print):
    """(а): эталон = прогон на исходных данных; сравнение — с прогоном, где данные после t заменены шумом."""
    base, steps = _base(ad), ad.steps(t, smoke)
    tag = f'{ad.name} t={t} h={h}{" +leak" if inject else ""}'
    t0 = time.time()
    w0 = _cached((ad.name, 'w0', h, inject), lambda: ad.make(base, h, inject))
    r0 = _cached((ad.name, 'ref', h, inject, tuple(steps)), lambda: ad.run(w0, h, steps, tag))
    r1 = ad.run(ad.noise(base, t, h, inject), h, steps, tag + ' noised')
    cal = ad.cal(base, w0)
    vs = compare_future(r0, r1, t, h, cal, ad.info_end)
    log(f'  ({tag}: {time.time() - t0:.0f} с)')
    return vs


def check_permute(ad, h, smoke, inject=False, log=print):
    """(б): цель переставлена внутри тикера (seed 42); val-QLIKE (val-годы <= 2021) не лучше константы."""
    base, steps = _base(ad), ad.permute_steps(smoke)
    t0 = time.time()
    wp = ad.permute(base, h, inject)
    res = ad.run(wp, h, steps, f'{ad.name} permuted')
    log(f'  ({ad.name} permute h={h}{" +leak" if inject else ""}: {time.time() - t0:.0f} с)')
    return compare_permute(res, ad.target_frame(wp, h), h)


def run_adapter(ad, dates, hs, smoke, permute=True, log=print):
    """Все проверки адаптера; падение одной стадии не останавливает остальные (решение М8)."""
    out = []
    try:
        for h in (hs if ad.kind == 'model' else [0]):
            for t in dates:
                out += [Verdict(f'{ad.name}:{v.stage}', v.check, f'{v.t[:10]} h={h}', v.ok, v.detail)
                        for v in check_future(ad, t, h, smoke, log=log)]
            if permute and ad.kind == 'model':
                out += [Verdict(f'{ad.name}:{v.stage}', v.check, f'-, h={h}', v.ok, v.detail) for v in check_permute(ad, h, smoke, log=log)]
    except NotConnected as e:
        out.append(Verdict(ad.name, 'future', '-', True, f'SKIP: {e}'))
    return out
