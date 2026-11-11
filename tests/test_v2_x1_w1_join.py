"""W1.1: склейка архива и продолжения — нет дублей, стык подряд по календарю, RV одинакова у всех рядов. Мутации: дубль даты на стыке; подмена RV одной даты. Сеть не нужна; реальные файлы только читаются."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_w1_join as J


def frame(start: str, n: int, seed: int = 0) -> pd.DataFrame:
    d = pd.bdate_range(start, periods=n)
    tp = pd.DatetimeIndex(list(d[1:]) + [d[-1] + pd.offsets.BDay(1)])
    rng = np.random.default_rng(seed)
    rv = np.exp(rng.normal(0, 0.3, n))
    return pd.DataFrame(dict(d=d, t_plus=tp, rv=rv, F=rv * np.exp(rng.normal(0.1, 0.2, n))))


def parts() -> tuple:
    """Архив до t⁺ = 2026-02-27 и продолжение сразу за ним (по рабочим дням)."""
    a = frame("2026-01-01", 40)
    a = a[a["t_plus"] <= J.ARCH_END].reset_index(drop=True)
    n = frame(str(a["t_plus"].iloc[-1].date()), 10, seed=1)
    return a, n


def test_stitch_known_answer_no_duplicates_and_contiguous() -> None:
    a, n = parts()
    x = J.stitch(a, n)
    assert len(x) == len(a) + len(n) and not x["d"].duplicated().any() and not x["t_plus"].duplicated().any()
    assert (x["part"] == "archive").sum() == len(a) and x["d"].is_monotonic_increasing
    assert x.loc[x["part"] == "new", "d"].iloc[0] == a["t_plus"].iloc[-1]          # первая новая d = последний архивный t⁺


def test_mutation_duplicate_date_at_junction_is_caught() -> None:
    a, n = parts()
    J.stitch(a, n)                                                                  # честная склейка проходит
    dup = pd.concat([n.iloc[[0]].assign(d=a["d"].iloc[-1]), n], ignore_index=True)  # новая строка с датой d последней архивной
    with pytest.raises(SystemExit):
        J.stitch(a, dup)
    dup_tp = n.copy()
    dup_tp.loc[0, "t_plus"] = n["t_plus"].iloc[1]                                   # повтор t⁺
    with pytest.raises(SystemExit):
        J.stitch(a, dup_tp)


def test_rv_cross_check_passes_and_mutation_substituted_rv_one_date_is_caught() -> None:
    a, n = parts()
    k1 = J.stitch(a, n)
    x1 = k1.copy()
    J.rv_cross_check({"K1": k1, "X1": x1})
    bad = x1.copy()
    bad.loc[len(bad) - 3, "rv"] *= 1.0001                                           # на продолжении
    with pytest.raises(SystemExit) as e:
        J.rv_cross_check({"K1": k1, "X1": bad})
    assert str(k1["d"].iloc[len(k1) - 3].date()) in str(e.value)                    # перечень дат в сообщении
    bad2 = x1.copy()
    bad2.loc[2, "rv"] *= 1.0001                                                     # на архиве
    with pytest.raises(SystemExit):
        J.rv_cross_check({"K1": k1, "X1": bad2})


@pytest.fixture(scope="module")
def real():
    return J.load_joined()


def test_real_series_join_is_clean(real) -> None:
    fr, jn, _ = real
    assert (jn["dup_d"] == 0).all() and (jn["dup_tp"] == 0).all() and jn["stitch_ok"].all()
    assert (jn["calendar_bad"] == 0).all() and (jn["rv_vs_k1_bad"] == 0).all() and (jn["rv_vs_daily_bad"] == 0).all()
    new = jn[jn["n_new"] > 0]
    assert set(new["series"]) == {"K1", "K2", "K3", "X1", "ZB", "ZB0", "cfg8"} and (new["n_new"] == 180).all()
    assert set(new["last_arch_tp"]) == {"2026-02-27"} and set(new["first_new_tp"]) == {"2026-03-02"}
    assert set(new["tp_last"]) == {"2026-11-10"}
    assert (jn["archive_vs_fwd_maxdiff"].dropna() == 0).all()                       # архивные строки = fwd-файлы побитно
    assert set(jn.loc[jn["rv_ref_missing"] != "", "rv_ref_missing"]) == {"2026-11-04"}   # единственная дата, которой нет в запечатанной RV (FWD_CHECKS: «только у бегуна 1»)


def test_real_mutation_rv_substituted_on_real_data_is_caught(real) -> None:
    fr = {k: v.copy() for k, v in real[0].items()}
    J.rv_cross_check(fr)
    i = int(np.flatnonzero(fr["X1"]["part"] == "new")[7])
    fr["X1"].loc[i, "rv"] *= 1.001
    with pytest.raises(SystemExit) as e:
        J.rv_cross_check(fr)
    assert str(fr["X1"]["d"].iloc[i].date()) in str(e.value)


def test_real_release_time_d_to_next_canonical_day(real) -> None:
    """t⁺ каждой строки — следующая каноническая строка календаря после d (прогноз дня d не использует более поздних данных)."""
    nxt = J.calendar_next()
    for k, x in real[0].items():
        assert (pd.DatetimeIndex(nxt.reindex(x["d"]).to_numpy()) == pd.DatetimeIndex(x["t_plus"])).all(), k
