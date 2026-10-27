"""Синтетика для scripts/analysis/sverka_A_predzashchita.py: переключатели меняют результат, ошибка ловится."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import sverka_A_predzashchita as S  # noqa: E402

N = 55


def panel(drop_pair=False, scale=1.0):
    idx = pd.bdate_range("2019-01-01", periods=N)
    rows = []
    for t in ("X", "Y"):
        for s in ("S1", "S2"):
            if drop_pair and (t, s) == ("Y", "S2"):
                continue
            pos = np.where(np.arange(N) % 10 < 5, 1.0, 0.0) * (1 if t == "X" else -1) * scale
            gr = np.random.default_rng(7 + len(t + s)).normal(0, .01, N) * pos
            rows.append(pd.DataFrame(dict(date=idx, ticker=t, strategy=s, position=pos, gross=gr)))
    return pd.concat(rows), pd.Series(.1, index=idx)


def test_selftest_runs():
    S.selftest()


def test_komissiya_znak():
    pos, rate = panel()
    m0 = S.portfolio(pos, rate, 0.0)["excess"].mean()
    assert S.portfolio(pos, rate, .01)["excess"].mean() < m0
    assert S.portfolio(pos, rate, -.01)["excess"].mean() > m0      # перепутанный знак виден


def test_agregirovanie_polnaya_panel_ravna_i_neravna_pri_dyre():
    pos, rate = panel()
    assert np.allclose(S.portfolio(pos, rate, .01, "pairs")["r"], S.portfolio(pos, rate, .01, "tickw")["r"])
    hole, rate = panel(drop_pair=True)
    assert not np.allclose(S.portfolio(hole, rate, 0, "pairs")["r"], S.portfolio(hole, rate, 0, "tickw")["r"])


def test_stavka_dni_i_plecho():
    pos, rate = panel()
    assert not np.allclose(S.portfolio(pos, rate, 0, fwd=True)["excess"], S.portfolio(pos, rate, 0, fwd=False)["excess"])
    big, rate = panel(scale=3.0)
    assert not np.allclose(S.portfolio(big, rate, 0, lev_cal=True)["total"], S.portfolio(big, rate, 0, lev_cal=False)["total"])
    flat = S.portfolio(pos, rate, 0, lev_cal=True)                  # экспозиция ≤ 1: плечо вклада не даёт
    assert np.allclose(flat["total"], S.portfolio(pos, rate, 0, lev_cal=False)["total"])


def test_vyhod_n5_platit_na_kontse():
    pos, rate = panel()
    a, b = S.portfolio(pos, rate, .01, exit_fee=True), S.portfolio(pos, rate, .01, exit_fee=False)
    assert a["r"].iloc[-1] < b["r"].iloc[-1] and np.allclose(a["r"].iloc[:-1], b["r"].iloc[:-1])


# ---- лестница (sverka_A_lestnica.py): синтетика из четырёх шагов ----
import types  # noqa: E402

import pytest  # noqa: E402

import sverka_A_lestnica as L  # noqa: E402

EFFECT = {"a": 0.5, "b": -0.2, "c": 0.1, "d": 0.3}                  # вклад ключа в Шарп
STEPS4 = [(k, k, {k: True}, "") for k in "abcd"]


def ev_synth(st):
    return (1.0 + sum(EFFECT[k] for k in "abcd" if st.get(k)),)


def test_lestnica_summa_i_ostatok():
    target = 1.0 + sum(EFFECT.values())
    lad = L.ladder(ev_synth, {}, STEPS4, target)
    assert [s[0] for s in lad["steps"]] == list("abcd")
    assert [round(s[5], 9) for s in lad["steps"]] == [0.5, -0.2, 0.1, 0.3]       # знак Δ = после − до
    assert abs(sum(s[5] for s in lad["steps"]) + lad["rest"] - (target - lad["first"][0])) < 1e-12
    assert abs(lad["rest"]) < 1e-12
    L.check_rest(lad, target)


def test_lestnica_perestavlennye_shagi_i_znak_lovyatsya():
    target = 1.0 + sum(EFFECT.values())
    swapped = [STEPS4[1], STEPS4[0]] + STEPS4[2:]
    lad = L.ladder(ev_synth, {}, swapped, target)
    assert [s[0] for s in lad["steps"]][:2] == ["b", "a"]
    assert [round(s[5], 9) for s in lad["steps"]] != [0.5, -0.2, 0.1, 0.3]       # перестановка видна в Δ
    L.check_rest(lad, target)                                                    # сумма от порядка не зависит
    flipped = L.ladder(lambda st: (2 - ev_synth(st)[0],), {}, STEPS4, target)    # перепутанный знак эффекта
    with pytest.raises(AssertionError):
        L.check_rest(flipped, target)


def test_lestnica_ostatok_ne_nol_padaet():
    target = 1.0 + sum(EFFECT.values()) + 0.01
    lad = L.ladder(ev_synth, {}, STEPS4, target)
    assert abs(lad["rest"] - 0.01) < 1e-12
    with pytest.raises(AssertionError):
        L.check_rest(lad, target)


# ---- мутации в памяти: каждая подложенная ошибка должна ловиться ----
SRC = Path(S.__file__).read_text(encoding="utf-8")


def mutant(old, new):
    assert SRC.count(old) == 1, old
    m = types.ModuleType("sverka_mut")
    m.__file__ = S.__file__
    sys.modules["sverka_mut"] = m
    exec(compile(SRC.replace(old, new), "sverka_mut", "exec"), m.__dict__)
    return m


def suite(M):
    """Проверки четырёх мест; AssertionError, если модуль M ошибается."""
    fri = pd.DatetimeIndex(["2019-01-04", "2019-01-07", "2019-01-08"])       # пт, пн, вт
    assert list(M.cal_days(fri, True)) == [3, 1, 1], "Н2: дни вперёд [d, d_next)"
    assert list(M.cal_days(fri, False)) == [1, 3, 1]
    x = [1.0, 2.0, 4.0, 3.0]
    assert abs(M.sharpe(x) - np.mean(x) * np.sqrt(252) / np.std(x, ddof=1)) < 1e-12, "ddof Шарпа"
    pos, rate = panel(scale=3.0)                                              # плечо 3 на «включённых» днях
    p = M.portfolio(pos, rate, 0.0)
    cal = M.cal_days(p.index, True)
    exp = p["r"] - (p["exposure"] - 1) * (rate.values + .02) / 365 * cal
    hi = p["exposure"] > 1
    assert hi.any() and np.allclose(p["total"][hi], exp[hi]), "знак платы за плечо"
    pos, _ = panel()
    sparse = pd.Series({pos["date"].min(): .05, pos["date"].min() + pd.Timedelta(days=20): .30})
    q = M.portfolio(pos, sparse, 0.0)
    got = ((q["total"] - q["excess"]) / M.cal_days(q.index, True) * 365).values
    want = np.where(q.index < sparse.index[1], .05, .30)
    assert np.allclose(got, want), "ffill ставки"


def test_suite_prohodit_na_originale():
    suite(S)


@pytest.mark.parametrize("old,new", [
    ("cal[:-1] = dd\n    else:\n        cal[1:] = dd", "cal[1:] = dd\n    else:\n        cal[:-1] = dd"),   # направление дней Н2
    ("- np.maximum(0, ex - 1) * (kr + .02)", "+ np.maximum(0, ex - 1) * (kr + .02)"),                    # знак платы за плечо
    ("ddof=1", "ddof=0"),                                                                                 # ddof Шарпа
    (".ffill().bfill()", ".bfill()"),                                                                     # ffill ставки
], ids=["N2_napravlenie", "plecho_znak", "ddof", "ffill"])
def test_mutaciya_lovitsya(old, new):
    with pytest.raises(AssertionError):
        suite(mutant(old, new))
