"""Тестировщик #859: сквозные проверки комиссии оборота множителей и мутации (известный ответ руками). Синтетика; без сети."""
import numpy as np
import pytest

from fastbt import kniga_msp as km
from fastbt import msp_v5
from tests.test_kniga_msp_komissiya_mnozh import ARGS, C, _ist, _kom
from tests.test_msp_v5 import _run

ORIG = km.komissiya_mnozh


def test_vhod_pri_postoyannyh_mnozhitelyah_nol():
    assert _kom([0.8, 0], [0, 0], [0.8, 0], [0, 0]) == 0.0         # c = 1


def test_vhod_i_smena_mnozhitelya_oblagaetsya_chast():
    """Вход raw 0,5 при c = 2 (w = 1,0); вчера было 0,3 при c = 1 (w = 0,3): Δw = 0,7, Δraw·c = 0,2·2 = 0,4 -> 0,3 облагается."""
    assert _kom([1.0, 0], [0.3, 0], [0.5, 0], [0.3, 0]) == pytest.approx(C * (0.7 - 0.4))


def test_vyhod_ne_oblagaetsya_pri_c_vcherashnem():
    assert _kom([0, 0], [0.6, 0], [0, 0], [0.3, 0]) == 0.0


def test_vyhod_c_segodnyashniy_byl_by_oshibkoy():
    """Выход в тот же день, когда вес Sk другой: вчерашний c = 2 (w_prev 0,6 при raw_prev 0,3); платит только Δ сверх."""
    assert _kom([0, 0.1], [0.6, 0], [0, 0.05], [0.3, 0]) == 0.0   # выход (c=2) и вход другого тикера целиком в парах


def test_max_ne_zanizhaet_netto_protivopolozhnye_sk():
    """Один тикер в двух Sk: +0,5 (c = 1) и −0,5 (c = 1) входят вместе -> нетто 0, комиссия книги 0 (обе ноги в excess пар)."""
    t = np.array([0, 0])
    assert _kom([0.5, -0.5], [0, 0], [0.5, -0.5], [0, 0], tik=t) == 0.0
    # а вот смена c у одной ноги при тех же raw: Δw = 0,25·(−1)... оборот нетто не теряется
    assert _kom([0.5, -0.25], [0.5, -0.5], [0.5, -0.5], [0.5, -0.5], tik=t) == pytest.approx(C * 0.25)


def test_max_ne_skryvaet_oborot_lishnij_vhod():
    """|Δw| < |ΣΔraw·c| невозможно при c = w/raw по каждому столбцу и одном знаке; при разных c — дарим 0, не отрицательное."""
    assert _kom([0.2, 0], [0, 0], [0.5, 0], [0, 0]) == 0.0


def test_hedzh_vklyuchenie_vyklyuchenie_smena_beta():
    z = [0, 0]
    assert _kom(z, z, z, z, h=-0.3, hp=0.0) == pytest.approx(C * 0.3)
    assert _kom(z, z, z, z, h=0.0, hp=-0.3) == pytest.approx(C * 0.3)
    assert _kom(z, z, z, z, h=-0.5, hp=-0.2) == pytest.approx(C * 0.3)
    assert _kom(z, z, z, z, h=0.2, hp=-0.1) == pytest.approx(C * 0.3)          # смена знака β


def test_skvoznoy_vyklyuchatel_bez_signalov():
    """День без новых сигналов: срабатывает выключатель (f = 0,5) -> комиссия = ставка·Σ|Δu|."""
    ist = _ist()
    a = km.kniga_dnya(20, *ARGS, ist, k=6, rho=0.001)
    ist["ex_kniga"][:21] = -0.01                                    # просадка > 5 %
    b = km.kniga_dnya(21, *ARGS, ist, k=6, rho=0.001)
    assert a.vykl == 1.0 and b.vykl == 0.5
    assert b.komissiya == pytest.approx(C * np.abs(b.u - a.u).sum()) and b.komissiya > 0


def test_skvoznoy_msp_v5_raznost_ravna_komissii():
    """ex_kniga[i] = u@R + h·imoex − komissiya[i] на каждом дне; Σ комиссий > 0; есть день без смены сырых позиций с комиссией."""
    kom = []
    km_orig = km.komissiya_mnozh
    km.komissiya_mnozh = lambda *a, **k: kom.append(km_orig(*a, **k)) or kom[-1]
    try:
        r = _run(0.0, None)
    finally:
        km.komissiya_mnozh = km_orig
    from tests.test_msp_v5 import _vhod
    _, d, im = _vhod()
    R = np.nan_to_num(np.tile(d.to_numpy(float), (1, 2)))
    brutto = (r.u.to_numpy() * R).sum(1) + r.hedzh.to_numpy() * np.nan_to_num(im.to_numpy(float))
    kom = np.array(kom)
    np.testing.assert_allclose(brutto - r.ex_kniga.to_numpy(), kom, atol=1e-15)
    assert kom.sum() > 0 and (kom[5:] > 0).any()
    assert msp_v5.kniga_msp.KOMISSIYA == C


# ---- мутанты (подменяют функцию; тесты выше обязаны упасть) ----
def _m_abs_w(w, wp, raw, rawp, tik, h, hp, stavka=C):
    return stavka * (np.abs(np.asarray(w)).sum() + abs(h - hp))


def _m_bez_hedzha(w, wp, raw, rawp, tik, h, hp, stavka=C):
    return ORIG(w, wp, raw, rawp, tik, 0.0, 0.0)


def _m_dvoynoe(w, wp, raw, rawp, tik, h, hp, stavka=C):
    n = int(np.max(tik)) + 1
    return stavka * (np.abs(np.bincount(tik, np.asarray(w, float) - np.asarray(wp, float), n)).sum() + abs(h - hp))


def _m_c_segodnya(w, wp, raw, rawp, tik, h, hp, stavka=C):
    w, wp, raw, rawp = (np.asarray(x, float) for x in (w, wp, raw, rawp))
    c = np.where(raw != 0, w / np.where(raw != 0, raw, 1), 1.0)           # при выходе c = 1 вместо вчерашнего
    n = int(np.max(tik)) + 1
    dw, dr = np.bincount(tik, w - wp, n), np.bincount(tik, (raw - rawp) * c, n)
    return stavka * (np.maximum(np.abs(dw) - np.abs(dr), 0.0).sum() + abs(h - hp))


UNIT = [test_vhod_pri_postoyannyh_mnozhitelyah_nol, test_vhod_i_smena_mnozhitelya_oblagaetsya_chast,
        test_vyhod_c_segodnyashniy_byl_by_oshibkoy, test_hedzh_vklyuchenie_vyklyuchenie_smena_beta,
        test_max_ne_zanizhaet_netto_protivopolozhnye_sk, test_skvoznoy_vyklyuchatel_bez_signalov]


@pytest.mark.parametrize("mut", [_m_abs_w, _m_bez_hedzha, _m_dvoynoe, _m_c_segodnya])
def test_mutant_lovitsya(monkeypatch, mut):
    monkeypatch.setattr(km, "komissiya_mnozh", mut)
    padenij = 0
    for t in UNIT:
        try:
            t()
        except AssertionError:
            padenij += 1
    assert padenij >= 1, mut.__name__
