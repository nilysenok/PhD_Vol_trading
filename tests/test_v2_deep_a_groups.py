"""A10: группы по величине RV, события скачков."""
import numpy as np
import pandas as pd

from models_v2.s5 import common, groups, pit


def test_fact_gruppy_dolei():
    x = np.random.default_rng(0).random(1000)
    b = groups.fact_bucket(x)
    assert [int((b == g).sum()) for g in range(5)] == [200, 200, 400, 150, 50]


def test_decluster_i_sobytie_skachka():
    assert list(pit.decluster([1, 1, 0, 0, 0, 0, 1, 0, 1])) == [0, 6]
    rng = np.random.default_rng(3)
    n = 700
    rv = np.exp(rng.normal(size=n))
    j = 0.05 * rv
    j[600] = 0.9 * rv[600]
    dl = pd.DataFrame({"rv": rv, "j": j}, index=pd.date_range("2017-01-01", periods=n))
    f = groups.event_flags(dl, "jump")
    assert f[600] and f[:252].sum() == 0


def test_mut_porogi_s_tekushchim_dnem_skryvayut_sobytie():
    x = np.arange(1.0, 701.0)   # каждый день — новый максимум
    ok = pit.past_bucket(x, [1.0])
    bad = pit.past_bucket(x, [1.0], include_current=True)
    assert (ok[252:] == 1).all() and (bad[252:] == 0).all()


def test_real_tablicy_summa_n_i_pometka():
    for f, note in (("RV_GROUPS_FACT.csv", "объяснительно"), ("RV_GROUPS_PIT.csv", "PIT")):
        t = common.read_comment_csv(common.REP / f)
        assert t[t.model == "b2"]["n"].sum() == common.N_GRID and t.note.str.contains(note).all()
        assert list(t.group.unique()) == list(groups.LABELS)
    ev = common.read_comment_csv(common.REP / "JUMP_EVENTS.csv")
    assert set(ev.offset) == set(range(-5, 6)) and (ev.n_events > 50).all()
