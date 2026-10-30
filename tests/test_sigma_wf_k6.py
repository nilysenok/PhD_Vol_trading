"""К-6: склейка шагов 1-8 (прежний ряд) и 9-10 (К-5) для r2_sborka_test и гейты G1-G3 (scripts/dannye/wf_sigma_910_sborka.py).
Синтетика, данные проекта и сеть не нужны. Мутации: шаги 9-10 из прежнего ряда; val шага 9 из прежнего ряда в s² шага 10; дыра в склейке;
конец 2026-02-04 (лишняя строка после конца); граница склейки сдвинута на 6.
python3 -m pytest tests/test_sigma_wf_k6.py -q"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_s = importlib.util.spec_from_file_location("wk6", Path(__file__).resolve().parents[1] / "scripts" / "dannye" / "wf_sigma_910_sborka.py")
W = importlib.util.module_from_spec(_s)
_s.loader.exec_module(W)
TK, ND = ("AAA", "BBB"), 40


def _ryad(seed, steps_test, steps_val, f0, extra=None):
    """final_d0_h1_test: test шага k - год 2016+k, val шага k - год 2015+k; F_sm = f0 + 0.01 k (test), 1.0 (val)."""
    rng, rows = np.random.default_rng(seed), []
    for part, steps, dy in (("test", steps_test, 2016), ("val", steps_val, 2015)):
        for k in steps:
            for d in pd.bdate_range(f"{dy + k}-01-03", periods=ND):
                for t in TK:
                    ya = float(rng.uniform(1e-4, 9e-4))
                    lr = ya * float(np.exp(rng.normal(0, 0.3)))
                    F = f0 + 0.01 * k if part == "test" else 1.0
                    rows.append(dict(date=d, ticker=t, year=dy + k, step=k, part=part, rv_actual=ya, rv_pred=lr * F, leg_raw=lr, F_sm=F, design="D"))
    d = pd.DataFrame(rows)
    if extra is not None:
        d = pd.concat([d, extra], ignore_index=True)
    return d


@pytest.fixture
def kit():
    old = _ryad(1, range(1, 9), (7, 8, 9), 1.2)
    drop = (old.part == "test") & (old.step == 8) & (old.date == old[(old.part == "test") & (old.step == 8)].date.max())
    old = old[~drop].reset_index(drop=True)  # хвост шага 8, которого нет в прежнем ряду
    new = _ryad(2, range(1, 11), (7, 8, 9, 10), 1.3)
    return old, new


def test_sklejka_znacheniya(kit):
    old, new = kit
    s = W.sklejka(old, new)
    k = ["date", "ticker", "step", "part"]
    a = old[old.step <= 8].merge(s, on=k, suffixes=("", "_s"))
    assert len(a) == (old.step <= 8).sum() and np.array_equal(a.leg_raw, a.leg_raw_s) and np.array_equal(a.F_sm, a.F_sm_s)
    b = new[new.step >= 9].merge(s, on=k, suffixes=("", "_s"))
    assert len(b) == (new.step >= 9).sum() and np.array_equal(b.leg_raw, b.leg_raw_s)
    v = s[s.part == "val"]
    for kk in (9, 10):  # F_sm шагов 9-10: правило R1 по склеенным val (7,8 прежние; 9,10 К-5)
        e = np.concatenate([np.log(g.rv_actual.values + 1e-10) - np.log(g.leg_raw.values) for _, g in v[v.step.between(kk - 2, kk)].groupby("step")])
        f = s[(s.part == "test") & (s.step == kk)].F_sm.unique()
        assert len(f) == 1 and f[0] == pytest.approx(np.exp(np.var(e) / 2), rel=1e-12)
    assert (v[v.step == 7].leg_raw.values == old[(old.part == "val") & (old.step == 7)].leg_raw.values).all()


def test_hvost_shaga_8_iz_k5_s_prezhnim_F(kit):
    old, new = kit
    s = W.sklejka(old, new)
    t8 = s[(s.part == "test") & (s.step == 8)]
    assert len(t8) == len(old[(old.part == "test") & (old.step == 8)]) + len(TK)
    assert t8.F_sm.nunique() == 1 and t8.F_sm.iloc[0] == old[(old.part == "test") & (old.step == 8)].F_sm.iloc[0]
    assert np.allclose(t8.rv_pred, t8.leg_raw * t8.F_sm, rtol=1e-12)


def test_g1_chestnyj_i_mutatsiya_granitsa_6(kit):
    old, new = kit
    itog = W.sklejka(old, new).query("part == 'test'")
    staryj = old[old.part == "test"]
    assert W.g1(itog, staryj)["ok"]
    with pytest.raises(AssertionError):
        W.sklejka(old, new, posl=6)  # граница склейки сдвинута: шаги 7-8 взяты бы из нового прогона
    nt = new[new.part == "test"]
    bad = itog.drop(columns=["leg_raw"]).merge(nt[["date", "ticker", "step", "leg_raw"]], on=["date", "ticker", "step"])  # leg_raw шагов 1-8 из К-5
    assert not W.g1(bad, staryj)["ok"]


def test_mutatsiya_9_10_iz_prezhnego_ryada(kit):
    old, new = kit
    with pytest.raises(AssertionError):
        W.sklejka(old, new, posl=10)  # всё из прежнего: шагов 9-10 К-5 нет
    with pytest.raises(AssertionError):
        W.sklejka(old, new, posl=9)


def test_g3_dyra_i_konec(kit):
    old, new = kit
    itog = W.sklejka(old, new).query("part == 'test'")
    konec = itog.date.max()
    assert W.g3(itog, new, konec)["ok"]
    dyra = itog.drop(itog[itog.step == 9].index[5])  # одна строка потеряна
    r = W.g3(dyra, new, konec)
    assert not r["ok"] and r["propusk"] == 1
    lishnyaya = pd.concat([itog, itog.iloc[[-1]].assign(date=konec + pd.Timedelta(days=1))], ignore_index=True)  # 2026-02-04
    r = W.g3(lishnyaya, new, konec)
    assert not r["ok"] and r["lishnih"] == 1
    assert W.KONEC == pd.Timestamp("2026-02-03")


def test_g2_nezavisimyj_s2_shaga_10_i_mutatsiya_val9_staryj(kit):
    old, new = kit
    s = W.sklejka(old, new)
    f = W.f_s2_embargo(s[s.part == "val"], s, 1, 10)
    itog = s[(s.part == "test") & (s.step == 10)].assign(F_sm=f)
    assert W.g2(itog, s, 1)["ok"]
    v = s[s.part == "val"]
    mut = pd.concat([v[v.step != 9], old[(old.part == "val") & (old.step == 9)]], ignore_index=True)  # val шага 9 из прежнего ряда
    f_mut = W.f_s2_embargo(mut, s, 1, 10)
    assert f_mut != pytest.approx(f, rel=1e-9)
    assert not W.g2(itog.assign(F_sm=f_mut), s, 1)["ok"]
