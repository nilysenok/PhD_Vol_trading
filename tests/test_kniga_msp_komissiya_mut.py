"""Тестировщик #859 (контракт Ж-149 от 31.10): R_brutto + T, ex = u@R − ставка·Σ|u|·T − ставка·|Δh|. Синтетика; без сети.
Мутации: подмена `komissiya_oborota` (хедж) и сверка с формулами-мутантами (|u| без T; знак u; комиссия внутри R)."""
import numpy as np
import pandas as pd
import pytest

from fastbt import kniga_msp as km
from fastbt import msp_v5
from tests.test_kniga_msp_komissiya_mnozh import C
from tests.test_msp_v5 import _run

T = 200                                                      # > OKNO_BETA + прогрев: хедж живой
IDX = pd.bdate_range("2016-01-04", periods=T)


def _f(v):
    return pd.DataFrame(v, index=IDX, columns=["A", "B"])


def _sborshchik(seed=11):
    rng = np.random.default_rng(seed)
    return 0.01 * rng.standard_normal((T, 2)), rng.uniform(0.5, 2.0, (T, 2)), rng   # R знаковый, T > 0 разный по тикерам


def _progon(rb, tt, rng):
    return _vyzov(rb, tt)


def _sig(storona):
    rng = np.random.default_rng(4)
    return {"S1": {"storona": _f(np.array([list(storona)] * T)), "sila": _f(rng.uniform(0, 1, (T, 2))),
                   "g": _f(np.ones((T, 2)))}}


def _vyzov(rb, tt, im=None, storona=(1.0, -1.0)):
    return msp_v5.progon(_sig(storona), _f(np.full((T, 2), 0.02)), {"A": "X", "B": "Y"},
                         {"R_brutto": _f(rb), "T": _f(tt)}, pd.Series(0.0, index=IDX) if im is None else im,
                         0.0, k=2, rho=0.001)


def _ekspress(r, rb, tt, im=None):
    """Эталон руками: u@R − C·Σ|u|·T + h·imoex − C·|Δh|."""
    u, h = r.u.to_numpy(), r.hedzh.to_numpy()
    imv = np.zeros(T) if im is None else im.to_numpy()
    dh = np.abs(np.diff(h, prepend=0.0))
    return (u * rb).sum(1) - C * (np.abs(u) * tt).sum(1) + h * imv - C * dh


def test_hedzh_komissiya_0_0004_na_modul_delta():
    """Хедж: 0,0004·|Δh| — известный ответ, оба знака смены β."""
    assert km.komissiya_oborota([-0.5], [-0.2]) == pytest.approx(0.0004 * 0.3)
    assert km.komissiya_oborota([0.2], [-0.1]) == pytest.approx(0.0004 * 0.3)
    assert km.komissiya_oborota([0.0], [0.0]) == 0.0


def test_skvoznoy_ex_ravno_ekspress_s_hedzhem():
    """Сквозной путь через msp_v5 с живым хеджем (β меняется): ex_kniga совпадает с формулой руками на каждом дне."""
    rng = np.random.default_rng(6)
    rb, tt, _ = _sborshchik()
    im = pd.Series(0.01 * rng.standard_normal(T), index=IDX)
    r = _vyzov(rb + 0.3 * im.to_numpy()[:, None], tt, im)
    assert np.abs(np.diff(r.hedzh.to_numpy())).max() > 0      # хедж действительно менялся: проверка не пустая
    np.testing.assert_allclose(r.ex_kniga.to_numpy(), _ekspress(r, rb + 0.3 * im.to_numpy()[:, None], tt, im), atol=1e-15)


def test_long_i_short_odnogo_razmera_platyat_odinakovo():
    """R = 0, T = 1: лонг и шорт платят одинаково C·|u|; комиссия внутри R дала бы сумму со знаком (нетто 0)."""
    tt = np.ones((T, 2))
    r = _vyzov(np.zeros((T, 2)), tt)
    u = r.u.to_numpy()
    dni = np.where((u[:, 0] > 0) & (u[:, 1] < 0))[0]
    assert len(dni) > 20
    for i in dni:
        assert r.ex_kniga.to_numpy()[i] == pytest.approx(-C * np.abs(u[i]).sum(), abs=1e-15) and r.ex_kniga.to_numpy()[i] < 0
    assert (r.ex_kniga.to_numpy() <= 0).all()                       # ни одного дня с возвратом комиссии


def test_den_bez_signalov_smena_mnozhitelya_spisyvaet_C_u_novyj_T():
    """Тест владельца: сигналы те же, T постоянно; в дни, где |u| менялся из-за множителя, списание = C·Σ|u_new|·T."""
    rb, tt, _ = _sborshchik()
    r = _progon(np.zeros((T, 2)), tt, None)
    u = r.u.to_numpy()
    m = np.abs(u).sum(1)
    dni = np.where((np.abs(np.diff(m)) > 1e-9) & (m[1:] > 0))[0] + 1
    assert len(dni) > 0
    for i in dni:
        assert r.ex_kniga.to_numpy()[i] == pytest.approx(-C * (np.abs(u[i]) * tt[i]).sum(), abs=1e-15)
        assert r.ex_kniga.to_numpy()[i] < 0


def test_T_otricatelnyj_otkaz():
    rb, tt, _ = _sborshchik()
    tt[10, 0] = -1.0
    with pytest.raises(ValueError, match="T"):
        _vyzov(rb, tt)


def _den_s_poziciej():
    rb, tt, _ = _sborshchik()
    u = _vyzov(rb, tt).u.to_numpy()
    i = int(np.flatnonzero(u[:, 0] != 0)[10])
    return rb, tt, i


def test_T_nan_otkaz_a_ne_nol():
    """Д4: NaN в обороте при позиции ≠ 0 не заменяется нулём молча (иначе комиссия исчезает без следа)."""
    rb, tt, i = _den_s_poziciej()
    tt[i, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        _vyzov(rb, tt)


def test_R_nan_pri_pozicii_otkaz():
    rb, tt, i = _den_s_poziciej()
    rb[i, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        _vyzov(rb, tt)


def test_nan_bez_poziciji_dopustim():
    """NaN в день без позиции (прогрев) — законно, не отказ."""
    rb, tt, i = _den_s_poziciej()
    rb[0, :] = np.nan
    tt[0, :] = np.nan
    assert np.isfinite(_vyzov(rb, tt).ex_kniga.to_numpy()).all()


def test_mutant_nan_to_num_vozvrashchen_lovitsya(monkeypatch):
    """Мутант: _bez_nan снова молча зануляет NaN -> тесты отказа обязаны упасть."""
    monkeypatch.setattr(msp_v5, "_bez_nan", lambda r, t, p, i, d, k: (np.nan_to_num(r), np.nan_to_num(t)))
    for t in (test_T_nan_otkaz_a_ne_nol, test_R_nan_pri_pozicii_otkaz):
        with pytest.raises(BaseException):
            t()


def test_staryj_klyuch_dohod_otkaz():
    with pytest.raises(ValueError):
        _run(0.0, None, dohod=None) if False else msp_v5.progon(
            _sig((1.0, -1.0)), _f(np.full((T, 2), 0.02)), {"A": "X", "B": "Y"},
            {"dohod": _f(np.zeros((T, 2)))}, pd.Series(0.0, index=IDX), 0.0, k=2)


def test_test_msp_v5_ne_oslablen():
    """Правка _run в test_msp_v5: T=0 -> прежние результаты; вне хеджа комиссии у пустой книги нет."""
    r = _run(0.0, None)
    assert np.isfinite(r.ex_kniga.to_numpy()).all()
    u, h = r.u.to_numpy(), r.hedzh.to_numpy()
    assert (np.abs(u).sum(1) > 0).any() and (np.abs(h) > 0).any()


# ---- мутанты ----
def _m_bez_hedzha(w, wp, stavka=C):
    return 0.0


def test_mutant_hedzh_bez_komissii_lovitsya(monkeypatch):
    rng = np.random.default_rng(6)
    rb, tt, _ = _sborshchik()
    im = pd.Series(0.01 * rng.standard_normal(T), index=IDX)
    R = rb + 0.3 * im.to_numpy()[:, None]
    ok = _vyzov(R, tt, im)
    ref = _ekspress(ok, R, tt, im)
    monkeypatch.setattr(km, "komissiya_oborota", _m_bez_hedzha)
    mut = _vyzov(R, tt, im)
    assert not np.allclose(mut.ex_kniga.to_numpy(), _ekspress(mut, R, tt, im), atol=1e-12)   # формула руками его ловит
    assert np.allclose(ok.ex_kniga.to_numpy(), ref, atol=1e-15)


@pytest.mark.parametrize("nazvanie", ["modul_u_bez_T", "znak_u", "komissiya_vnutri_R"])
def test_mutant_formuly_otlichaetsya_ot_realnoj(nazvanie):
    """Реальный ex_kniga на данных, где лонг ≠ шорт по T, не совпадает ни с одной из формул-мутантов."""
    rb, tt, _ = _sborshchik()
    r = _vyzov(rb, tt)
    u, ex = r.u.to_numpy(), r.ex_kniga.to_numpy()
    brutto = (u * rb).sum(1)
    mut = {"modul_u_bez_T": brutto - C * np.abs(u).sum(1),
           "znak_u": brutto - C * (u * tt).sum(1),
           "komissiya_vnutri_R": (u * (rb - C * tt)).sum(1)}[nazvanie]
    assert not np.allclose(ex, mut, atol=1e-9)
    assert np.allclose(ex, brutto - C * (np.abs(u) * tt).sum(1), atol=1e-15)
