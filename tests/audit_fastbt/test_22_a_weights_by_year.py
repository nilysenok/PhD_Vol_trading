"""22. Веса A по годам (топ-10, голосование, фильтр оборота) — только данные до 1 января Y (FA2b).

Дополняет tests/test_a_weights_future.py (шум `_noisy` оттуда же) и не дублирует его: тот тест
читает данные до 2023 года, здесь — только разработка ≤ 2021 (SBER, годы 2019 и 2020), и проверка
строже на один бар: шум начинается с бара ys+1 (второго бара года Y). Close первого бара года Y
участвует в обучении законно: доходность последнего бара Y−1 = log(close[ys]/close[ys−1]), а первая
позиция года Y решается на закрытии ys. Обязаны совпасть: топ-10 года Y, ансамбль на барах ≤ ys,
выбор фильтра оборота (удержание, N) и отфильтрованная позиция на барах ≤ ys.
"""
import importlib.util

import numpy as np
import pytest

from conftest import ROOT, SIGMA_H1, read_dev
from fastbt import config as fcfg
from fastbt import dev
from fastbt.data import OHLCV, year_bounds

_spec = importlib.util.spec_from_file_location("a_wf", ROOT / "tests" / "test_a_weights_future.py")
_awf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_awf)

TICKER = "SBER"
SIDS = ("S1", "S3", "S5")           # контртренд, тренд, диапазон (V41)
YEARS = (2019, 2020)
_RES = {}


def _compute():
    """Считается внутри теста (после мутации), один раз на процесс."""
    if _RES:
        return _RES
    from fastbt import data as fd
    from fastbt import design as dsg
    from fastbt import turnover
    from fastbt.approach_a import ApproachA
    from fastbt.unit import make_unit
    mp = pytest.MonkeyPatch()
    try:
        mp.setenv("FASTBT_NO_CACHE", "1")          # шумовые ряды не пишутся в общий кэш сигналов
        ohlcv, sig = read_dev(OHLCV), read_dev(SIGMA_H1)
        mp.setattr(fd, "daily_df", lambda: ohlcv)
        mp.setattr(fd, "vpred_df", lambda: sig)
        fd.load_ticker.cache_clear()
        td = fd.load_ticker(TICKER, True)
        assert td.dates.max() <= np.datetime64("2021-12-31")
        d, _ = dev.load_design(dev.ROOT / "fastbt" / "designs" / "final" / "main.yaml")
        cfg = dev.make_config(d, "dev")
        for sid in SIDS:
            dsg.apply(cfg.design)
            u0 = make_unit(TICKER, sid, td)
            a0 = ApproachA(u0, cfg)
            for Y in YEARS:
                ys = year_bounds(td.dates, Y)[0]
                td1 = _awf._noisy(td, ys + 1, 7)
                assert np.array_equal(td1.close[:ys + 1], td.close[:ys + 1])
                u1 = make_unit(TICKER, sid, td1)
                a1 = ApproachA(u1, cfg)
                row = []
                for a, u in ((a0, u0), (a1, u1)):
                    top = a.year(Y)
                    ens = a.ensemble(top)
                    tf = turnover.apply({Y: (top, ens)}, u, cfg, "A")[Y]
                    row.append((top, ens, tf[0]["tf"], tf[1]))
                _RES[(sid, Y)] = (ys, row)
    finally:
        mp.undo()
        from fastbt import data as fd2
        from fastbt import design as dsg2
        fd2.load_ticker.cache_clear()
        dsg2.apply(())
    return _RES


@pytest.mark.parametrize("sid", SIDS)
@pytest.mark.parametrize("Y", YEARS)
def test_a_weights_only_before_Y(sid, Y):
    ys, ((top0, ens0, tf0, f0), (top1, ens1, tf1, f1)) = _compute()[(sid, Y)]
    assert top0 and len(top0) == 10
    assert top1 == top0                                    # топ-10 (сигнал × RM × исполнение)
    assert np.array_equal(ens1[:ys + 1], ens0[:ys + 1])    # голосование на барах ≤ ys
    assert tf1 == tf0                                      # фильтр оборота: (удержание, N)
    assert np.array_equal(f1[:ys + 1], f0[:ys + 1])


def test_train_window_ends_at_jan1():
    """Окно обучения A и фильтра оборота: [WARMUP, 1 янв Y) при маске [0]."""
    d, _ = dev.load_design(dev.ROOT / "fastbt" / "designs" / "final" / "main.yaml")
    cfg = dev.make_config(d, "dev")
    assert cfg.bound() == 0 and not cfg.mask_old
    assert cfg.cal_end_year(2020) == 2020


MUTATIONS = {
    # отбор и фильтр оборота года Y обучаются по 1 янв Y+1 (включён год Y — старая утечка [1])
    "train_includes_Y": lambda mp: mp.setattr(fcfg.Config, "bound", lambda self: 1),
}
