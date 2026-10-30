"""Тестировщик #762: то, что test_supertrend_nan.py не ловил. Дыры ПОСЛЕ старта (старт не пересчитывается и не засевается будущим),
старт требует все три H/L/C, и сквозная проверка: сигналы S4 на относительном ряду (sektor) реально появляются.
Мутации (проверены в отдельной копии): е) старт после ПОСЛЕДНЕГО нечислового бара; ё) старт по одному close (H/L не смотрим)."""
import numpy as np
import pytest

from fastbt.signals import s34
from test_supertrend_nan import MAIN, NAN54, _est, s_nan

P, M = 14, 3.0


def _s_dyroj(a, b):
    h, l, c = s_nan()
    h[a:b] = l[a:b] = c[a:b] = np.nan
    return h, l, c


@pytest.mark.parametrize("a,b", [(300, 303), (68, 69), (60, 62), (580, 581)])
def test_dyra_posle_starta_bez_perezapuska(a, b):
    """Старт — по первому конечному бару; дыра дальше ничего не пересчитывает: ST = ядро на срезе с первого бара, как у конечного начала."""
    h, l, c = _s_dyroj(a, b)
    st = s34.supertrend(h, l, c, P, M)
    assert st[NAN54:].tobytes() == s34._supertrend0(h[NAN54:], l[NAN54:], c[NAN54:], P, M).tobytes()
    assert np.isnan(st[:NAN54]).all()


def test_dyra_prichinnost_do_dyry_ne_menyaetsya():
    """ST до дыры совпадает с ST ряда, обрезанного перед дырой: дыра и будущее не влияют на прошлое."""
    h, l, c = _s_dyroj(300, 303)
    st = s34.supertrend(h, l, c, P, M)
    cut = s34.supertrend(h[:300], l[:300], c[:300], P, M)
    assert st[:300].tobytes() == cut.tobytes()
    assert np.isfinite(st[NAN54 + P:300]).all()                        # мутация е): старт после последней дыры → здесь NaN


@pytest.mark.parametrize("kto", [0, 1, 2])
def test_start_trebuet_vse_tri_ryada(kto):
    """Бар 54 с NaN только в одном из H/L/C — не старт: ST стартует с бара 55 (мутация ё: смотрит только close)."""
    h, l, c = s_nan()
    [h, l, c][kto][NAN54] = np.nan
    st = s34.supertrend(h, l, c, P, M)
    assert np.isnan(st[:NAN54 + 1 + P]).all() and np.isfinite(st[NAN54 + 1 + P:]).all()


@pytest.mark.skipif(not _est(), reason="нет FASTBT_MAIN / дневного ряда 18:50")
@pytest.mark.xfail(strict=True, reason="БЛОКЕР #762: на относительном ряду NaN-префикс отравляет compute_base (adx14 ≡ 0) → фильтр ADX>20 "
                   "отсекает все сигналы S4; одной правки supertrend мало. Снять xfail, когда сигналы появятся.")
def test_real_s4_sektor_signaly_cherez_semyu():
    from fastbt import etap1, istochniki_v6
    from fastbt.eval_v6_osi import rel_td_ind
    from fastbt.signals.interface import get_family
    inp = etap1.build_inputs("S4", period="train4")
    tik = list(inp.sk.tickers)[:3]
    ist = istochniki_v6.zagruzit("train4", tik)
    fam, n = get_family("S4", setka="v5"), 0
    for t in tik:
        td, _ = inp.sk._td[t]
        ts, is_ = rel_td_ind(ist.rel(t, "sektor"), td, True)
        n += int((fam(ts, is_, [{"period": 5, "mnozh": 0.5}]).sig[0] != 0).sum())
    assert n > 0, MAIN
