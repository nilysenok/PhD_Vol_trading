"""Самопроверки детектора: на заведомой утечке он обязан падать, на исправленных вариантах — проходить.
Случаи: (адаптер, дата t, h, внедрённая утечка, ожидание, какие стадии обязаны упасть/пройти)."""
import pandas as pd
from . import registry as R
from .runner import check_future, check_permute

T19, T21, T23 = '2019-06-30', '2021-12-31', '2023-06-30'


def _fails(vs, only=None):
    return [v for v in vs if not v.ok and (only is None or only in v.stage)]


def cases(smoke):
    """(название, функция -> (ok_самопроверки, пояснение))."""
    hm = 22

    def leak_feature(name, mk, kind):
        def f():
            ad = mk()
            vs = check_future(ad, T19, hm, smoke, inject=True)
            if kind == 'future':
                bad = _fails(vs)
                return bool(bad), f'падение: {bad[0].stage}: {bad[0].detail[:110]}' if bad else 'НЕ упал (детектор слеп)'
            vp = check_permute(ad, hm, smoke, inject=True)
            bad = _fails(vp)
            return bool(bad), f'падение: {bad[0].detail[:110]}' if bad else 'НЕ упал (детектор слеп)'
        return (f'{name}: признак = будущая RV -> {"(а)" if kind == "future" else "(б)"} падает', f)

    def strat(mask, t, expect_fail):
        def f():
            vs = check_future(R.strategies(mask), t, 0, smoke)
            bad = _fails(vs)
            if expect_fail:
                return bool(bad), (f'падение: {bad[0].stage}: {bad[0].detail[:110]}' if bad else 'НЕ упал (детектор слеп)')
            return not bad, 'прошёл' if not bad else f'упал: {bad[0].stage}: {bad[0].detail[:110]}'
        return (f'{mask}, t={t}: {"падает" if expect_fail else "проходит"}', f)

    def v7(lag_name, expect_fail, h):
        def f():
            vs = check_future(R.models(lag_name), T19, h, smoke)
            v7f, others = _fails(vs, 'V7'), [v for v in _fails(vs) if 'V7' not in v.stage]
            if expect_fail:
                return bool(v7f) and not others, (f'V7 упал: {v7f[0].detail[:110]}' if v7f else 'V7 НЕ упал') + (f'; упали и другие: {others[0].stage}' if others else '')
            return not (v7f or others), 'прошёл' if not (v7f or others) else f'упал: {(v7f or others)[0].stage}'
        return (f'{lag_name}, h={h}, t={T19}: V7 {"падает" if expect_fail else "проходит"}', f)

    return [leak_feature('harj', lambda: R.models('harj'), 'future'),
            leak_feature('p1', lambda: R.models('p1'), 'future'),
            leak_feature('harj', lambda: R.models('harj'), 'permute'),
            strat('v4_mask1', T19, True), strat('v4_mask1', T23, True),
            strat('v4', T19, False), strat('v4', T23, False),
            strat('fastbt_mask_old', T19, True), strat('fastbt_mask_old', T23, True),
            strat('fastbt', T19, False), strat('fastbt', T23, False),
            v7('gibrid_v7lag1', True, 5), v7('gibrid', False, 5), v7('gibrid_v7lag1', True, 22), v7('gibrid', False, 22)]


def run(smoke, log=print):
    res = []
    for name, f in cases(smoke):
        ok, why = f()
        log(f'{"OK  " if ok else "BAD "} самопроверка: {name} — {why}')
        res.append(ok)
    return all(res)
