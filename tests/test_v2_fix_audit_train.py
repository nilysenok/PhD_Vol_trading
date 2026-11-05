"""Замечания проверяющего Ж-200 (05.11.2026), п. 1–3: перехват фактических матриц обучения и PIT-выбор по LOYO.

Мутации (подложены в копии): M1 «финальная модель без последних 252 пар» (fit.py: X[idx] -> X[tr]); окно до d+1;
w = None в block_fit; cut_row(F, Y+1); pairs_upto(T_cut+1).
"""
import numpy as np
import pandas as pd
import pytest

from models_v2 import run_fix
from models_v2.s4 import fit, rows, runner, select

HP = fit.HP(3, 10.0, 1.0, 0.1, 1.0)
HP_FAST = fit.HP(2, 10.0, 1.0, 0.3, 1.0, 1.0, 40)


class Dz:
    def __init__(self, X, y):
        self.X, self.y, self.off = X, y, np.zeros(len(y))


class Fx:
    def __init__(self, n, start="2013-01-01"):
        self.seg, self.dates, self.rv = np.zeros(n, int), pd.bdate_range(start, periods=n).values, np.ones(n)


@pytest.fixture
def spy(monkeypatch):
    """Перехват xgb.train: матрицы (метки = номера строк t, веса) обучающих, останов-блока и финальной моделей."""
    calls, real = [], fit.xgb.train

    def train(params, dtrain, num_boost_round, evals=(), **kw):
        calls.append({"label": dtrain.get_label().copy(), "w": dtrain.get_weight().copy(),
                      "es": [(e.get_label().copy(), e.get_weight().copy()) for e, _ in evals]})
        return real(params, dtrain, num_boost_round, evals=evals, **kw)

    monkeypatch.setattr(fit.xgb, "train", train)
    return calls


def labelled(n=900, seed=0):
    """y[t] = t: метки перехваченных матриц сами называют строки t, вошедшие в обучение."""
    X = np.random.default_rng(seed).normal(size=(n, 4))
    return X, np.arange(n, dtype=float)


# ---- п. 1: фактические матрицы финальной модели ----

@pytest.mark.parametrize("T", [800, 650])
def test_final_model_matrix_has_pair_t_plus_1_eq_d_and_none_beyond(spy, T):
    X, y = labelled()
    runner.block_fit(Fx(900), Dz(X, y), HP, T, [T], False, None, "work")
    assert len(spy) == 2, "ожидались предварительная и финальная модели"
    final = spy[1]["label"].astype(int)
    assert final.max() + 1 == T, "в финальной матрице нет пары с t+1 = d"
    assert (final + 1 <= T).all(), "в финальной матрице пара из будущего"
    assert len(final) == T and np.array_equal(np.sort(final), np.arange(T))   # все пары окна, без выброшенных 252
    prelim, (es, _) = spy[0]["label"].astype(int), spy[0]["es"][0]
    assert len(es) == 252 and es.max() == final.max() and prelim.max() < es.min()   # блок остановки — хвост окна, вне предварительного обучения


def test_final_matrix_in_oracle_and_perm_variants_also_through_d(spy):
    X, y = labelled()
    for v in ("oracle", "perm"):
        spy.clear()
        runner.block_fit(Fx(900), Dz(X, y), HP, 800, [800], False, None, v)
        final = spy[1]["label"].astype(int)
        assert final.max() + 1 == 800 and (final + 1 <= 800).all()


def test_mutation_final_without_last_252_is_caught_by_matrix_check(spy, monkeypatch):
    """M1 на уровне перехвата: подмена идёт через ту же train, а проверка видит дефицит пар."""
    X, y = labelled()
    idx = fit.pair_index(np.zeros(900, int), X, y, (0, 1, 2, 3), 800)
    fit.fit_i1(X, y, idx[:-252], HP)           # то, что сделала бы мутация внутри fit_i1: финал на урезанных парах
    assert spy[1]["label"].astype(int).max() + 1 != 800


# ---- п. 2: веса доходят до обучения и до блока остановки ----

def test_weights_reach_prelim_es_and_final_matrices(spy):
    X, y = labelled()
    T = 800
    runner.block_fit(Fx(900), Dz(X, y), HP, T, [T], True, None, "work")
    w = fit.weights(np.arange(900), T)
    pre, fin = spy[0], spy[1]
    t_pre, t_fin, t_es = pre["label"].astype(int), fin["label"].astype(int), spy[0]["es"][0][0].astype(int)
    close = lambda a, b: a.size == b.size and np.allclose(a, b, rtol=1e-6, atol=0)   # DMatrix хранит веса во float32  # noqa: E731
    assert pre["w"].size == len(t_pre) and close(pre["w"], w[t_pre])
    assert close(spy[0]["es"][0][1], w[t_es]), "веса не дошли до блока остановки"
    assert close(fin["w"], w[t_fin]) and fin["w"].max() > 0.99    # самая свежая пара весит 0,5^(1/504)


def test_no_weights_when_flag_off(spy):
    X, y = labelled()
    runner.block_fit(Fx(900), Dz(X, y), HP, 800, [800], False, None, "work")
    assert spy[0]["w"].size == 0 and spy[1]["w"].size == 0 and spy[0]["es"][0][1].size == 0


def test_weights_reach_loyo_fold_fit(spy):
    F = Fx(1300)
    X, y = labelled(1300)
    cut = int(np.searchsorted(F.dates, np.datetime64("2017-01-01"))) - 1
    runner.fold_score(F, Dz(X, y), HP_FAST, cut, 2016, True)
    w = fit.weights(np.arange(1300), cut)
    t_fin = spy[1]["label"].astype(int)
    assert spy[1]["w"].size == len(t_fin) and np.allclose(spy[1]["w"], w[t_fin], rtol=1e-6, atol=0)
    assert spy[0]["es"][0][1].size > 0


# ---- п. 3: выбор года Y не видит целей >= 01.01.Y ----

def serial_jobs(F, DZ, jobs, kind, procs=4):
    runner._init(F, DZ)
    return [runner._fold_job(j) for j in jobs]


def loyo_world(corrupt_from_date=None, seeds=(3,)):
    """Синтетика: цель = 0,5·x0 + шум. Строки с датой цели >= corrupt_from_date подменяются другим режимом (будущее меняет выбор)."""
    F = Fx(1300)
    out = {}
    for s in seeds:
        rng = np.random.default_rng(s)
        X = rng.normal(size=(1300, 4))
        y = 0.5 * X[:, 0] + 0.1 * rng.normal(size=1300)
        out[s] = [X, y]
    rv = np.exp(0.3 * np.random.default_rng(9).normal(size=1300))
    if corrupt_from_date is not None:
        c = int(np.searchsorted(F.dates, np.datetime64(corrupt_from_date))) - 1   # последняя строка < даты
        rng = np.random.default_rng(99)
        for X, y in out.values():
            y[c:] = rng.normal(6.0, 3.0, len(y) - c)          # y[t] — цель строки t+1: строки t >= c смотрят в дату >= corrupt_from_date
            X[c + 1:] = rng.normal(size=(len(y) - c - 1, 4)) * 5
        rv[c + 1:] = 40.0
    F.rv = rv
    return F, {s: Dz(X, y) for s, (X, y) in out.items()}


@pytest.fixture
def serial(monkeypatch):
    monkeypatch.setattr(runner, "map_jobs", serial_jobs)


def same_scores(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return a["score"].to_numpy().tobytes() == b["score"].to_numpy().tobytes()


def test_run_loyo_ignores_targets_from_year_Y(serial):
    cands = {2017: {"a": (3, HP_FAST, False), "b": (3, fit.HP(3, 10.0, 1.0, 0.1, 1.0, 1.0, 40), True)}}
    base = select.run_loyo(*loyo_world(), cands, (2017,), 1)[0]
    fut = select.run_loyo(*loyo_world("2017-01-01"), cands, (2017,), 1)[0]
    assert np.isfinite(base["score"]).all() and (base["n_folds"] == 2).all()
    assert same_scores(base, fut), "выбор года 2017 увидел цели >= 01.01.2017"
    past = select.run_loyo(*loyo_world("2016-01-01"), cands, (2017,), 1)[0]
    assert not same_scores(base, past)                        # контроль: порча прошлого меняет оценки, тест не пустой


def test_run_loyo_known_answer_on_cut_row():
    F = Fx(1300)
    c = select.cut_row(F, 2017)
    assert F.dates[c] < np.datetime64("2017-01-01") <= F.dates[c + 1]


def captured_tables(monkeypatch):
    tabs = []
    monkeypatch.setattr(run_fix.prov4, "write_csv", lambda df, path, extra="", imoex="": tabs.append(df))
    monkeypatch.setattr(run_fix, "YEARS", (2017,))
    return tabs


def test_choose_ignores_targets_from_year_Y(serial, monkeypatch):
    tabs, hps = captured_tables(monkeypatch), (HP_FAST, fit.HP(3, 30.0, 2.0, 0.1, 1.0, 1.0, 40))
    ch0 = run_fix.choose(*loyo_world(), 3, hps, "t", "тест")
    ch1 = run_fix.choose(*loyo_world("2017-01-01"), 3, hps, "t", "тест")
    assert same_scores(tabs[0], tabs[1]) and ch0 == ch1
    run_fix.choose(*loyo_world("2016-01-01"), 3, hps, "t", "тест")
    assert not same_scores(tabs[0], tabs[2])


def test_choose_i6_ignores_targets_from_year_Y(serial, monkeypatch):
    tabs = captured_tables(monkeypatch)
    keys = ["%s_%s" % (f, k) for f in ("level", "m", "delta", "resid") for k in ("base", "mkx")]

    def ctx(date):
        F, dz = loyo_world(date, seeds=range(len(keys)))
        return rows.Ctx(F, np.arange(0), dict(zip(keys, dz.values())), HP_FAST, {2017: HP_FAST})

    p0 = run_fix.choose_i6(ctx(None), "mkx")
    p1 = run_fix.choose_i6(ctx("2017-01-01"), "mkx")
    assert same_scores(tabs[0], tabs[1]) and p0 == p1 and len(tabs[0]) == 16
    run_fix.choose_i6(ctx("2016-01-01"), "mkx")
    assert not same_scores(tabs[0], tabs[2])
