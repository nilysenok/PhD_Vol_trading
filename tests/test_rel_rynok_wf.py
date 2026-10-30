"""rel_rynok на зоне wf: бенчмарк «рынок» (IMOEX с 1900) не составной сектор — проверка устаревшей ребалансировки (Д4) не к нему.
Причина (до правки): даты ряда > 2021-12-31 и > 1900-01-01 + 120 дней → ValueError «сектор … 1900-01-01»."""
import numpy as np
import pandas as pd
import pytest

from fastbt import e1_rel_sektor as rel
from fastbt import kontekst_v6 as kv

DATES = pd.bdate_range("2014-01-01", "2026-02-27")
N = len(DATES)


def _ser(x):
    return pd.Series(np.asarray(x, float), index=DATES)


def _ctx(pit=None):
    r = np.zeros(N)
    r[100] = 0.10
    close = _ser(100 * np.exp(np.cumsum(r)))
    sg = _ser(np.full(N, 0.01))
    flat = _ser(np.full(N, 50.0))
    c, s = {"T": close}, {"T": sg}
    return kv.rynok_ctx(flat, c, s) if pit is None else rel.Ctx({"IDX": flat}, c, s, pit)


def test_rynok_na_wf_ne_padaet_i_znachenie_prezhnee():
    z, names = _ctx().rel("T", 1)
    assert z.iloc[100] == pytest.approx(10.0) and z.iloc[-1] == 0.0
    assert (names == kv.IMOEX).all()


def test_rynok_train4_prezhnee():
    ctx = _ctx()
    ctx.close = {"T": ctx.close["T"][:"2021-12-31"]}
    z, _ = ctx.rel("T", 1)
    assert z.iloc[100] == pytest.approx(10.0)


def test_nastoyashchij_sektor_ustarel_na_wf_otkaz():
    with pytest.raises(ValueError, match="Д4"):
        _ctx({"T": [(pd.Timestamp("2017-03-20"), "IDX")]}).rel("T", 1)
