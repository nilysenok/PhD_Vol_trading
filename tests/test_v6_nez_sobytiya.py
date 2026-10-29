"""Независимые тесты V6-7 (К17–К19, окна событий, point-in-time). Тестировщик, не автор."""
import numpy as np
import pandas as pd
import pytest

from fastbt import sobytiya_e1 as S

D = pd.bdate_range("2019-03-01", periods=40)


def _ref_cbr(dates, events, do, posle):
    """Эталон циклом: день k в блоке, если есть событие с якорем a (торговый день перед
    заседанием), a-do < k <= a+posle, и дата известности <= dates[k]."""
    d = pd.DatetimeIndex(dates)
    out = np.zeros(len(d), bool)
    for m, kf in events:
        j = next((i for i, x in enumerate(d) if x >= m), None)
        if j is None or j == 0:
            continue
        a = j - 1
        for k in range(len(d)):
            if a - do < k <= a + posle and kf <= d[k]:
                out[k] = True
    return out


def _df(events):
    return pd.DataFrame({"date": [e[0] for e in events], "known_from": [e[1] for e in events]})


EVENTS = [(D[10], D[0]), (D[20], D[17]), (D[35], D[2])]


@pytest.mark.parametrize("do,posle", [(1, 0), (2, 0), (3, 1), (5, 2), (1, 3), (8, 0)])
def test_cbr_protiv_ciklicheskogo_etalona(do, posle):
    got = S.cbr_block(D, _df(EVENTS), do=do, posle=posle)
    assert got.tolist() == _ref_cbr(D, EVENTS, do, posle).tolist()


def test_vykl_pobaytno_v5_defaults():
    df = _df(EVENTS)
    assert S.cbr_block(D, df).tobytes() == S.cbr_block(D, df, do=1, posle=0).tobytes()
    assert np.flatnonzero(S.cbr_block(D, df)).tolist() == [9, 19, 34]
    # sobytiya(): do/posle по умолчанию не меняют вывод
    a = S.sobytiya(D, ["X"], df)
    b = S.sobytiya(D, ["X"], df, do=1, posle=0)
    assert a["cbr"]["X"].tobytes() == b["cbr"]["X"].tobytes()
    assert not a["net"]["X"].any()


def test_zasedanie_v_vyhodnoy_yakor_pyatnitsa():
    fri = pd.Timestamp("2019-03-08")
    assert fri in D
    sat = pd.Timestamp("2019-03-09")
    got = S.cbr_block(D, _df([(sat, D[0])]))
    assert np.flatnonzero(got).tolist() == [D.get_loc(fri)]


def test_pit_mutatsiya_35_znanie_iz_buduschego():
    """Событие D[20] объявлено в D[17]; do=6 -> дни 15,16 блокировать нельзя."""
    got = S.cbr_block(D, _df([(D[20], D[17])]), do=6)
    assert np.flatnonzero(got).tolist() == [17, 18, 19]


def test_prichinnost_budushchie_sobytiya_ne_meniayut_proshloe():
    K = 14
    base = S.cbr_block(D, _df(EVENTS), do=6, posle=2)
    # добавляем события, ставшие известными после K (с широким окном назад)
    extra = EVENTS + [(D[K + 4], D[K + 1]), (D[K + 8], D[K + 3])]
    more = S.cbr_block(D, _df(extra), do=6, posle=2)
    assert base[:K + 1].tolist() == more[:K + 1].tolist()
    assert base.tolist() != more.tolist()


def test_prichinnost_isporchennye_known_from_posle_K():
    K = 25
    ev = [(D[30], D[22]), (D[36], D[27])]
    ev2 = [(D[30], D[22]), (D[36], D[39])]  # known_from сдвинут в будущее
    a = S.cbr_block(D, _df(ev), do=7, posle=1)
    b = S.cbr_block(D, _df(ev2), do=7, posle=1)
    assert a[:K + 1].tolist() == b[:K + 1].tolist()


def test_okno_obrezano_po_krayam():
    ev = [(D[1], D[0]), (D[39], D[0])]
    got = S.cbr_block(D, _df(ev), do=5, posle=5)
    assert np.flatnonzero(got[:8]).tolist() == [0, 1, 2, 3, 4, 5]   # якорь 0: окно 0..5
    assert np.flatnonzero(got[30:]).tolist() == list(range(4, 10))  # якорь 38: 34..39 (обрезано справа)
    assert len(got) == len(D)


def test_zasedanie_pervyj_den_ili_posle_konca_ignoriruetsya():
    assert not S.cbr_block(D, _df([(D[0], D[0])]), do=3).any()          # j=0
    assert not S.cbr_block(D, _df([(D[-1] + pd.Timedelta(days=10), D[0])]), do=3).any()  # j>=len


def test_pustye_vkhody():
    assert len(S.cbr_block(pd.DatetimeIndex([]), _df([]))) == 0
    one = pd.DatetimeIndex([D[0]])
    assert S.cbr_block(one, _df(EVENTS), do=3).tolist() == [False]
    assert not S.cbr_block(D, _df([]), do=3, posle=3).any()


@pytest.mark.parametrize("do,posle", [(0, 0), (-1, 0), (1, -1)])
def test_validatsiya(do, posle):
    with pytest.raises(ValueError):
        S.cbr_block(D, _df(EVENTS), do=do, posle=posle)
    with pytest.raises(ValueError):
        S.div_block(D, pd.DataFrame(columns=["ticker", "last_cum_date", "announce_date"]), "A", do, posle)


def test_granica_2022():
    d = pd.bdate_range("2021-12-27", periods=8)
    with pytest.raises(PermissionError):
        S.cbr_block(d, _df([]))


def _ref_div(dates, rows, ticker, do, posle):
    d = pd.DatetimeIndex(dates)
    out = np.zeros(len(d), bool)
    for t, lc, an in rows:
        if t != ticker:
            continue
        cand = [i for i, x in enumerate(d) if x <= lc]
        if not cand:
            continue
        a = cand[-1]
        if an > d[a]:
            continue
        for k in range(len(d)):
            if a - do < k <= a + posle and an <= d[k]:
                out[k] = True
    return out


def test_div_protiv_etalona_i_vykl():
    rows = [("A", D[10], D[0]), ("A", D[22], D[19]), ("B", D[15], D[1]),
            ("A", pd.Timestamp("2019-03-30"), D[3])]  # суббота -> предыдущий торговый
    df = pd.DataFrame(rows, columns=["ticker", "last_cum_date", "announce_date"])
    for do, posle in [(1, 0), (3, 2), (6, 1)]:
        assert S.div_block(D, df, "A", do, posle).tolist() == _ref_div(D, rows, "A", do, posle).tolist()
    base = S.div_block(D, df, "A")   # выкл.: ровно last_cum_date; суббота 30.03 -> пятница 29.03
    assert np.flatnonzero(base).tolist() == sorted([10, 22, D.get_loc(pd.Timestamp("2019-03-29"))])
    assert not S.div_block(D, df, "Z", do=4, posle=4).any()


def test_div_pit_obyavleno_pozzhe_ne_blokiruet():
    rows = [("A", D[10], D[12])]
    df = pd.DataFrame(rows, columns=["ticker", "last_cum_date", "announce_date"])
    assert not S.div_block(D, df, "A", do=5, posle=5).any()


def test_div_prichinnost():
    K = 12
    rows = [("A", D[10], D[0]), ("A", D[20], D[15])]
    more = rows + [("A", D[26], D[K + 2])]
    df1 = pd.DataFrame(rows, columns=["ticker", "last_cum_date", "announce_date"])
    df2 = pd.DataFrame(more, columns=["ticker", "last_cum_date", "announce_date"])
    a = S.div_block(D, df1, "A", do=10, posle=2)
    b = S.div_block(D, df2, "A", do=10, posle=2)
    assert a[:K + 1].tolist() == b[:K + 1].tolist()


def test_sobytiya_div_tolko_pri_verified():
    df = pd.DataFrame([("A", D[10], D[0])], columns=["ticker", "last_cum_date", "announce_date"])
    assert set(S.sobytiya(D, ["A"], _df(EVENTS), df)) == {"net", "cbr"}
    full = S.sobytiya(D, ["A"], _df(EVENTS), df, div_verified=True, do=2, posle=1)
    assert full["cbr_div"]["A"].tolist() == (full["div"]["A"] | full["cbr"]["A"]).tolist()
    with pytest.raises(ValueError):
        S.sobytiya(D, ["A"], _df(EVENTS), None, div_verified=True)
