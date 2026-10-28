"""К-21: относительная сила к сектору (контракт §2Б.3). Синтетика с известным ответом + мутации."""
import numpy as np
import pandas as pd
import pytest

from fastbt import e1_rel_sektor as rs

D = pd.bdate_range("2021-01-04", periods=6)
R = np.array([np.nan, 0.02, -0.02, 0.03, 0.0, -0.015])      # доходности тикера
RS = np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0])            # доходности сектора, кроме дня 4
RS[4] = 0.05
SG = [0.01, 0.01, 0.1, 0.01, 0.01, 0.01]
PIT0 = pd.Timestamp("2017-03-20")


def _ser(r):
    return pd.Series(np.exp(np.nancumsum(np.nan_to_num(r))), index=D) * 100


def ctx(pit=None, idx_r=RS):
    pit = pit or {"AAA": [(PIT0, "IDX")], "LSRG": [(PIT0, None), (pd.Timestamp("2021-06-21"), "MOEXRE")]}
    close = {"AAA": _ser(R), "LSRG": _ser(R)}
    sig = {t: pd.Series(SG, index=D) for t in close}
    return rs.Ctx({"IDX": _ser(idx_r)}, close, sig, pit)


def al(mode, w=1, k=1.0, t="AAA", c=None):
    return rs.allow({"rel_napravlenie": mode, "rel_k": k, "rel_okno": w}, t, D, c or ctx())


def test_rel_known_answer():
    z, _ = ctx().rel("AAA", 1)
    # σ̂_{d−1}: σ̂_0…σ̂_4 = .01, .01, .1, .01, .01; excess дней 1…5: .02, −.02, .03, −.05, −.015
    assert np.isnan(z.iloc[0])
    assert z.iloc[1:].tolist() == pytest.approx([2.0, -2.0, 0.3, -5.0, -1.5])


def test_window_scaling_sqrt_w():
    z, _ = ctx().rel("AAA", 2)
    assert z.iloc[3] == pytest.approx((-0.02 + 0.03) / (0.1 * np.sqrt(2)))


def test_directions_po_sile_and_protiv_are_mirrors():
    lo, sr = al("po_sile")
    plo, psr = al("protiv")
    assert lo.tolist() == [False, True, False, False, False, False]
    assert sr.tolist() == [False, False, True, False, True, True]
    assert (plo == sr).all() and (psr == lo).all()


def test_net_is_p1():
    lo, sr = al("net", c=None)
    assert lo.all() and sr.all()


def test_no_index_ticker_is_allowed():
    c = ctx({"AAA": [(PIT0, "MOEXTN")]})                         # MOEXTN нет в ctx.idx
    lo, sr = al("po_sile", c=c)
    assert lo.all() and sr.all()


def test_lsrg_before_2021_06_21_no_filter():
    lo, sr = al("po_sile", t="LSRG")
    assert lo.all() and sr.all()                                 # «нет» → индекса нет → фильтр не действует


def test_real_csv_pit():
    pit = rs.sektor_pit()
    assert pit["SBER"][-1][1] == "MOEXFN"
    assert [n for d0, n in pit["LSRG"] if d0 < pd.Timestamp("2021-06-21")] == [None] * 17
    assert rs.bez_indeksa(pit, {"MOEXFN"}, ["SBER", "LSRG", "AFLT"], "2021-01-04") == ["LSRG", "AFLT"]


def test_mutation_close_d_plus_1_caught(monkeypatch):
    base = al("po_sile")
    monkeypatch.setattr(rs, "CLOSE_LAG", -1)
    mut = al("po_sile")
    assert any((a != b).any() for a, b in zip(base, mut))


def test_mutation_lsrg_foreign_index_caught():
    ok = al("po_sile", t="LSRG")
    foreign = al("po_sile", t="LSRG", c=ctx({"LSRG": [(PIT0, "IDX")]}))
    assert ok[0].all() and not foreign[0].all()


def test_mutation_sigma_d_caught(monkeypatch):
    base = al("po_sile")
    monkeypatch.setattr(rs, "SIGMA_LAG", 0)
    assert any((a != b).any() for a, b in zip(base, al("po_sile")))


def test_mutation_side_swap_caught():
    lo, sr = al("po_sile")
    assert not (lo == sr).all() and (lo != al("protiv")[0]).any()


def test_bad_mode_and_empty():
    with pytest.raises(ValueError):
        al("zzz")
    lo, sr = rs.allow({"rel_napravlenie": "po_sile"}, "AAA", pd.DatetimeIndex([]), ctx())
    assert len(lo) == 0 and len(sr) == 0
