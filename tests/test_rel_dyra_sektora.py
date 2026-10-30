"""Дыра в баре индекса сектора (2022-03-24/25: у IMOEX и тикера бары есть, у MOEXMM нет): день — по IMOEX с флагом,
оценка не падает с «в td_rel нет дней сетки». Без дыры — побайтно прежнее; ведущие дни до первого дня сектора не заполняются."""
import numpy as np
import pandas as pd
import pytest

from fastbt import rel_ryad as R
from fastbt.eval_v6_osi import rel_td_ind

DAYS = pd.bdate_range("2021-03-01", periods=10)


def _bars(seed, p0, skip=(), first=0):
    rng = np.random.default_rng(seed)
    ts = []
    for i, d in enumerate(DAYS):
        if i in skip or i < first:
            continue
        ts += list(pd.date_range(d + pd.Timedelta(hours=9, minutes=50), d + pd.Timedelta(hours=18, minutes=40), freq="10min"))
    c = p0 * np.exp(np.cumsum(rng.normal(0, 0.002, len(ts))))
    o = np.r_[p0, c[:-1]]
    return pd.DataFrame({"timestamp": ts, "open": o, "high": np.maximum(o, c) * 1.0005, "low": np.minimum(o, c) * 0.9995,
                         "close": c, "volume": 100})


ROWS = [(pd.Timestamp("2021-01-01"), "A")]


def test_dyra_sektora_den_po_imoex_s_flagom_i_bez_otkaza():
    T, M, A = _bars(1, 100.0), _bars(4, 3000.0), _bars(2, 50.0, skip=(5,))
    got, fl = R.td_rel_sektor(T, M, {"A": A}, ROWS, "T")
    assert len(got) == 10
    f5 = fl[fl["date"] == DAYS[5]].iloc[0]
    assert f5["flag"] == R.F_NET_BAROV and f5["benchmark"] == "IMOEX"
    assert (fl["flag"].drop(5) == "").all() and (fl["benchmark"].drop(5) == "A").all()
    td = type("TD", (), {"dates": DAYS.to_numpy(), "ticker": "T", "sigma": None})()
    rel_td_ind(got, td)       # раньше: ValueError «в td_rel нет 1 дней сетки td»


def test_bez_dyry_pobaytno_prezhnee():
    T, M, A = _bars(1, 100.0), _bars(4, 3000.0), _bars(2, 50.0)
    got, fl = R.td_rel_sektor(T, M, {"A": A}, ROWS, "T")
    pd.testing.assert_frame_equal(got, R.td_rel(T, A, "T"), check_exact=True)
    assert (fl["flag"] == "").all()


def test_veduschie_dni_do_pervogo_dnya_sektora_ne_zapolnyayutsya():
    T, M, A = _bars(1, 100.0), _bars(4, 3000.0), _bars(2, 50.0, first=3)
    got, _ = R.td_rel_sektor(T, M, {"A": A}, ROWS, "T")
    assert len(got) == 7 and pd.to_datetime(got["date"]).min() == DAYS[3]


def test_dyra_v_kontekste_dryg_tikera_ne_nulevaya():
    T, M, A = _bars(1, 100.0), _bars(4, 3000.0), _bars(2, 50.0, skip=(5,))
    got, _ = R.td_rel_sektor(T, M, {"A": A}, ROWS, "T")
    assert got["close"].notna().all() and (got["close"] > 0).all()
    with pytest.raises(ValueError):                    # настоящая дыра у тикера по-прежнему отказ
        td = type("TD", (), {"dates": DAYS.to_numpy(), "ticker": "T"})()
        rel_td_ind(got.drop(index=2), td)
