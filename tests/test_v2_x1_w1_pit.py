"""W1.5: PIT состояний — будущие даты не меняют группы прошлых дат (S1, S3, S4, S5, S7, S8); мутации: «пороги по всей выборке»; S8 по парам «закрыто + 1». Сеть не нужна."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s13 import v6_0_q as Q
from models_v2.s16 import x1_w1_join as J, x1_w1_st as S


def whole_sample(x) -> np.ndarray:
    """Мутант: пороги терцилей по ВСЕЙ выборке (как если бы будущее было известно)."""
    x = np.asarray(x, float)
    q1, q2 = np.nanquantile(x, [1 / 3, 2 / 3])
    return np.where(np.isfinite(x), (x > q1).astype(int) + (x > q2).astype(int), -1)


def test_tercile_only_past_known_answer_and_future_does_not_change_past() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=700)
    lab = S.tercile(x)
    assert (lab[:S.MIN_PAST] == -1).all() and (lab[S.MIN_PAST:] >= 0).all()                     # пока прошлых < 252 — нет порога
    q1, q2 = np.quantile(x[:400], [1 / 3, 2 / 3])
    assert lab[400] == int(x[400] > q1) + int(x[400] > q2)                                       # порог дня 400 — по x[:400]
    y = x.copy()
    y[500:] += 5.0                                                                               # будущее изменено
    assert (S.tercile(y)[:500] == lab[:500]).all()


def test_mutation_thresholds_on_whole_sample_is_caught(monkeypatch) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=700)
    y = x.copy()
    y[500:] += 5.0
    assert (S.tercile(y)[:500] == S.tercile(x)[:500]).all()
    monkeypatch.setattr(S, "tercile", whole_sample)
    assert (S.tercile(y)[:500] != S.tercile(x)[:500]).any()


def daily(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rv = np.exp(rng.normal(0, 0.4, n))
    return pd.DataFrame(dict(rv=rv, j=rv * rng.uniform(0, 0.3, n), imoex_abs=np.abs(rng.normal(0, 0.01, n))), index=pd.bdate_range("2018-01-03", periods=n))


def test_s3_s4_s7_future_days_do_not_change_past_groups_and_mutation_is_caught(monkeypatch) -> None:
    a, b = daily(), daily()
    b.iloc[600:, 0] = b.iloc[600:, 0] ** 3                                                       # будущие дни изменены: рассеяние ln RV ×3 (шок S3), доля скачка ×10 (S4), |r| IMOEX ×3 (S7)
    b.iloc[600:, 1] = b.iloc[600:, 1] * 10.0
    b.iloc[600:, 2] = b.iloc[600:, 2] * 3.0
    la, lb = S.day_labels(a), S.day_labels(b)
    for k in ("S3", "S4", "S7"):
        assert (la[k].iloc[:600] == lb[k].iloc[:600]).all(), k
    monkeypatch.setattr(S, "tercile", whole_sample)
    ma, mb = S.day_labels(a), S.day_labels(b)
    assert all((ma[k].iloc[:600] != mb[k].iloc[:600]).any() for k in ("S3", "S4", "S7"))


def synth_al(n: int = 700, seed: int = 0) -> dict:
    d = pd.bdate_range("2018-01-03", periods=n)
    tp = pd.DatetimeIndex(list(d[1:]) + [d[-1] + pd.offsets.BDay(1)])
    rng = np.random.default_rng(seed)
    rv = np.exp(rng.normal(0, 0.3, n))
    return {k: pd.DataFrame(dict(d=d, t_plus=tp, rv=rv, F=rv * np.exp(rng.normal(0.1, 0.2, n)))) for k in ("K1", "K2", "X1")}


def test_s5_s8_future_rows_do_not_change_past_groups() -> None:
    a, b = synth_al(), synth_al()
    for k in ("K1", "K2", "X1"):
        b[k].loc[500:, "F"] *= 4.0
        b[k].loc[500:, "rv"] *= 2.0
    la, lb = S.pair_labels(a, "K2"), S.pair_labels(b, "K2")
    for k in la:
        assert (la[k][:500] == lb[k][:500]).all(), k


def test_s8_uses_only_closed_pairs_and_mutation_closed_plus_one_is_caught(monkeypatch) -> None:
    al = synth_al()
    d, tp = al["K1"]["d"].to_numpy(), al["K1"]["t_plus"].to_numpy()
    rv = al["K1"]["rv"].to_numpy(float)
    dif = Q.ql(rv, al["X1"]["F"].to_numpy(float)) - Q.ql(rv, al["K2"]["F"].to_numpy(float))
    f = S.s8_feature(dif, tp, d)
    i = 300
    assert np.isnan(f[S.LAST_PAIRS - 1]) and f[S.LAST_PAIRS] == pytest.approx(dif[:S.LAST_PAIRS].mean(), abs=1e-15)
    assert f[i] == pytest.approx(dif[i - S.LAST_PAIRS:i].mean(), abs=1e-15)                    # строки j < i: t⁺_j ≤ d_i; собственная пара дня d (t⁺ > d) не входит
    dif2 = dif.copy()
    dif2[i] += 1.0                                                                               # собственная (ещё не закрытая) пара дня d_i
    assert S.s8_feature(dif2, tp, d)[i] == f[i]
    monkeypatch.setattr(S, "closed_end", lambda tp_, d_: np.searchsorted(tp_, d_, side="right") + 1)   # мутант: «закрыто + 1»
    assert S.s8_feature(dif2, tp, d)[i] != f[i]


@pytest.fixture(scope="module")
def real():
    return J.load_joined()[0]


def test_real_s1_prefix_equals_full_and_calendar_pairs_are_closed_by_d(real) -> None:
    k1 = real["K1"]
    x = k1["F"].to_numpy(float)
    full = S.tercile(x)
    assert (S.tercile(x[:1500]) == full[:1500]).all()                                            # группа прошлой даты не зависит от последующих лет
    d, tp = pd.DatetimeIndex(k1["d"]).to_numpy(), pd.DatetimeIndex(k1["t_plus"]).to_numpy()
    k = S.closed_end(tp, d)
    assert (k == np.arange(len(d))).all()                                                        # к d_i закрыты ровно пары 0…i−1 (t⁺_j = d_{j+1} ≤ d_i); i-я пара не закрыта
