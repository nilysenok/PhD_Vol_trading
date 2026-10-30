"""Тестировщик #869: мутации `fastbt.msp_v5_dohod` и расхождение с автоматом `tf_os_v5.pozicii` (деньги).

Дефект Д1 (xfail strict): R/T = «держим сторону последнего бара вчера весь день» ≠ внутридневной автомат движка
(вход со 2-го бара, сигнальный выход/разворот внутри дня). Стопов/тейков в `tf_os_v5` нет — есть сигнальные выходы.
"""
import numpy as np
import pandas as pd
import pytest

from fastbt import msp_v5, tf_os_v5 as TF
from fastbt import msp_v5_dohod as M
from tests.test_msp_v5_dohod import _b10, RATE

TOCHKA = dict(okno=10, podtv="none")


def _dvizhok_vs_sobrat(seed):
    b = _b10(60, seed)
    td = TF.ticker_data(TF.bary(b, "1h"), "X")
    ss, den = TF.signaly("S3", td, [TOCHKA], "1h"), TF.den_id(td)
    pos = TF.pozicii(ss.sig[0], ss.exit_l[0], ss.exit_s[0], den)
    g, c, t = TF.dohod(pos, td.open, td.close, den, RATE)
    ix = lambda x: pd.Series(x, index=den).groupby(level=0).sum().set_axis(pd.DatetimeIndex(np.unique(den)))  # noqa
    eng, eng_t = ix(g - c), ix(t)
    st, r, tt = M.tiker("S3", "1h", TOCHKA, b, "X")
    mine = (st * r - RATE * st.abs() * tt).shift(1)
    return eng, mine, eng_t, (st.abs() * tt).shift(1)


@pytest.mark.xfail(strict=True, reason="Д1: R/T МСП — не доход движка tf_os_v5 (автомат pozicii не используется)")
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_d1_dohod_msp_raven_dvizhku_s_vnutridnevnym_avtomatom(seed):
    eng, mine, eng_t, mine_t = _dvizhok_vs_sobrat(seed)
    assert np.allclose(mine.dropna().values, eng.reindex(mine.dropna().index).values, atol=1e-12)
    assert eng_t.sum() == pytest.approx(mine_t.sum())


def test_d1_raschozhdenie_izmereno():
    """Фиксирует размер расхождения (синтетика, 60 дней, 1h, S3): оборот движка в разы выше."""
    eng, mine, eng_t, mine_t = _dvizhok_vs_sobrat(1)
    assert eng_t.sum() > 3 * mine_t.sum() and not np.isclose(eng.sum(), mine.sum(), atol=1e-3)


def test_mut_sdvig_nulevoj_lovitsya_na_sobrat():
    """R без сдвига (строка d = доходность дня d) меняется от возмущения дня d; штатная — нет."""
    def r_d1(sdvig, mult):
        b = _b10(30, 4)
        d1 = b["date"] == sorted(b["date"].unique())[5]
        b = b.assign(close=np.where(d1, b["close"] * mult, b["close"]))
        f = TF.bary(b, "1h"); td = TF.ticker_data(f, "X"); den = TF.den_id(td)
        _, _, r, t = M.ryad_dnya(np.zeros(len(den), np.int8), td.open, td.close, den)
        return sdvig(r, t)[0].iloc[5]
    assert r_d1(M.sdvig_d_plus_1, 1.0) == pytest.approx(r_d1(M.sdvig_d_plus_1, 1.1), nan_ok=True)
    assert r_d1(lambda r, t: (r, t), 1.0) != pytest.approx(r_d1(lambda r, t: (r, t), 1.1))


def test_mut_T_tolko_vhod_lovitsya():
    """Оборот единицы за полный день = 2 (вход + закрытие); мутация «только вход» = 1 ловится."""
    o, c, d = np.array([1., 1.1, 1.2]), np.array([1.05, 1.15, 1.25]), np.full(3, 17532, np.int64) * 86400 * 10**9
    _, _, r, t = M.ryad_dnya(np.zeros(3, np.int8), o, c, d)
    assert t.iloc[0] == 2.0 and t.iloc[0] != 1.0


def test_mut_storona_s_pervogo_bara_lovitsya():
    o, c, den = (np.arange(1., 7.), np.arange(1., 7.) + .1,
                 np.repeat(pd.bdate_range("2018-01-09", periods=2).values.astype("datetime64[ns]"), 3).astype("int64"))
    sig = np.array([1, 0, 0, -1, 0, 0], np.int8)
    _, st, _, _ = M.ryad_dnya(sig, o, c, den)
    assert list(st) == [0, 0] and list(st) != list(sig[::3])
    sig = np.array([1, 1, -1, -1, -1, 1], np.int8)
    assert list(M.ryad_dnya(sig, o, c, den)[1]) == [-1, 1] != list(sig[::3])


def test_d4_net_bara_d_plus_1_pri_pozicii_otkaz_knigi():
    """Последний день окна: R=NaN; при storona ≠ 0 книга msp_v5 отказывает (не нули)."""
    res = M.sobrat(["S3"], ["A"], ("2018-01-01", "2019-02-01"), "1h", {"S3": TOCHKA}, {"A": _b10(300, 1)})
    R, T = res["R_brutto"], res["T"]
    assert np.isnan(R.iloc[-1, 0]) and np.isnan(T.iloc[-1, 0])
    one = pd.DataFrame(1.0, index=R.index, columns=R.columns)
    sila = one * np.random.default_rng(0).uniform(0.1, 1, R.shape)      # ранг силы нужен окно 252 д.
    sig = {"S3": {"storona": one, "sila": sila, "g": one}}              # позиция ≠ 0 в последней строке
    with pytest.raises(ValueError, match="NaN"):
        msp_v5.progon(sig, one * 0.02, {"A": "x"}, {"R_brutto": R, "T": T}, pd.Series(0.0, index=R.index), 0.0, k=1)
