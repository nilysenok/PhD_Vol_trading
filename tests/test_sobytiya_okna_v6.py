"""V6-7: окна событий К17–К19. Синтетика с известным ответом; «выкл.» (1, 0) побайтно = v5."""
import numpy as np
import pandas as pd
import pytest

from fastbt import sobytiya_e1 as S

D = pd.bdate_range("2019-01-01", periods=30)   # D[10] — день события / last_cum_date
CBR = pd.DataFrame({"date": [D[10]], "known_from": [D[0]]})
DIV = pd.DataFrame({"ticker": ["A"], "last_cum_date": [D[10]], "announce_date": [D[0]]})


def idx(m):
    return list(np.flatnonzero(m))


def v5_cbr(dates, cbr):          # эталон: прежняя реализация, блок на один день
    d = pd.DatetimeIndex(dates)
    out = np.zeros(len(d), bool)
    for m, kf in zip(pd.to_datetime(cbr["date"]), pd.to_datetime(cbr["known_from"])):
        j = int(d.searchsorted(m, side="left"))
        if 0 < j < len(d) and kf <= d[j - 1]:
            out[j - 1] = True
    return out


def test_vykl_pobaytno_v5():
    assert S.cbr_block(D, CBR).tobytes() == v5_cbr(D, CBR).tobytes()
    assert idx(S.cbr_block(D, CBR)) == [9]
    assert idx(S.div_block(D, DIV, "A")) == [10]


@pytest.mark.parametrize("do", [1, 2, 3, 5])
def test_do_dni_pered_sobytiem(do):
    assert idx(S.cbr_block(D, CBR, do=do)) == list(range(10 - do, 10))


@pytest.mark.parametrize("posle", [0, 1, 2])
def test_posle_dni_sobytiya_i_sledom(posle):
    assert idx(S.cbr_block(D, CBR, do=1, posle=posle)) == list(range(9, 10 + posle))


def test_div_okno():
    assert idx(S.div_block(D, DIV, "A", do=3, posle=2)) == [8, 9, 10, 11, 12]


def test_pit_zaglyadyvanie_vpered_mutaciya():
    """Известно только с D[8]: дни 5–7 блокировать нельзя (знание из будущего, мутация № 35)."""
    c = pd.DataFrame({"date": [D[10]], "known_from": [D[8]]})
    assert idx(S.cbr_block(D, c, do=5)) == [8, 9]
    assert idx(S.cbr_block(D, c, do=5)) != list(range(5, 10))   # окно без проверки дало бы это
    dv = DIV.assign(announce_date=D[9])
    assert idx(S.div_block(D, dv, "A", do=4, posle=1)) == [9, 10, 11]
    late = DIV.assign(announce_date=D[11])
    assert idx(S.div_block(D, late, "A", do=3)) == []   # объявлено после дня — блока нет


def test_kraya_i_validaciya():
    c = pd.DataFrame({"date": [D[1]], "known_from": [D[0]]})
    assert idx(S.cbr_block(D, c, do=5)) == [0]                   # окно обрезано слева
    c2 = pd.DataFrame({"date": [D[29]], "known_from": [D[0]]})
    assert idx(S.cbr_block(D, c2, do=1, posle=2)) == [28, 29]    # и справа
    with pytest.raises(ValueError):
        S.cbr_block(D, CBR, do=0)
    with pytest.raises(ValueError):
        S.cbr_block(D, CBR, posle=-1)


def test_sobytiya_probros_i_net():
    o = S.sobytiya(D, ["A"], CBR, DIV, div_verified=True, do=2, posle=1)
    assert idx(o["cbr"]["A"]) == [8, 9, 10]
    assert idx(o["div"]["A"]) == [9, 10, 11]
    assert idx(o["cbr_div"]["A"]) == [8, 9, 10, 11]
    assert not o["net"]["A"].any()


def test_granica_2022():
    with pytest.raises(PermissionError):
        S.cbr_block(pd.bdate_range("2021-12-20", periods=12), CBR, do=3)
