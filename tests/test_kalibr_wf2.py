"""К-wf2 (Ж-136, §11 черновика предрегистрации v2): период `wf` в калибраторе v5 и контролях. Синтетика, данных нет.
Окно года t — [01.01.2015; 31.12(t−1)]; для 2026 это 44 квартала 2015Q1…2025Q4, без 2026-Q1.
Мутация «окно с 1 января года t» (granica включает первый бар года t) обязана ловиться.
python3 -m pytest tests/test_kalibr_wf2.py -q
"""
import numpy as np
import pytest

from fastbt import kalibr_v5 as K
from fastbt import kontroli_v5 as kv

_D = np.arange("2014-08-26", "2026-02-28", dtype="datetime64[D]")
DATY = _D[np.is_busday(_D)]        # 1 января 2026 (чт) — рабочий день: мутация его захватит


def _kvartaly(okno) -> list:
    d = DATY[(DATY >= np.datetime64(okno[0])) & (DATY <= np.datetime64(okno[1]))]
    return sorted({(int(x.astype("datetime64[Y]").astype(int)) + 1970, (int(str(x)[5:7]) - 1) // 3 + 1) for x in d})


def test_pul_vse_eshche_otkazyvaet_daty_2020_plus():
    """Охрана `Pul.__post_init__` (≥ 2020) не тронута: пул wf на реальных датах до снятия замка невозможен."""
    with pytest.raises(ValueError):
        kv.sintez_pul(3, DATY, m=3)


def _pul():
    """Пул на календаре wf: охрана ≥ 2020 обходится только в тесте — пул строится на ранних датах, потом daty подменяются."""
    rann = np.arange("1990-01-01", "2010-01-01", dtype="datetime64[D]")
    pul = kv.sintez_pul(3, rann[np.is_busday(rann)][:DATY.size], m=3)
    pul.daty = DATY
    return pul


def test_periody_wf_gody_i_nachalo():
    assert K.PERIODY["wf"] == ("2015-01-01", tuple(range(2017, 2027)))
    assert K.PERIODY["train4"] == ("2016-01-01", (2017, 2018, 2019))     # прежние не тронуты
    assert K.PERIODY["v2"] == ("2015-01-01", (2017, 2018, 2019))


def test_kontroli_parametry_ot_perioda():
    assert kv.parametry("wf") == (tuple(range(2017, 2027)), "2015-01-01")
    assert (kv.GODY, kv.NACHALO) == ((2017, 2018, 2019), "2016-01-01")   # умолчание — train4, как раньше
    with pytest.raises(KeyError):
        kv.parametry("net_takogo")


def test_okno_2026_44_kvartala_bez_2026q1():
    pul = _pul()
    nachalo, _ = K.PERIODY["wf"]
    r = K.god(pul, 2026, nachalo)
    q = _kvartaly(r["okno"])
    assert len(q) == 44 and q[0] == (2015, 1) and q[-1] == (2025, 4) and (2026, 1) not in q
    assert r["okno"] == ("2015-01-01", "2025-12-31")


def test_okno_2017_8_kvartalov():
    r = K.god(_pul(), 2017, K.PERIODY["wf"][0])
    assert len(_kvartaly(r["okno"])) == 8


def test_mutatsiya_okno_s_1_yanvarya_goda_t_lovitsya(monkeypatch):
    """Подложенная ошибка: граница включает бар 01.01.t (side='right') — окно 2026 видит 2026-Q1."""
    monkeypatch.setattr(K, "granica", lambda daty, t: int(np.searchsorted(daty, np.datetime64(f"{t}-01-01"), side="right")))
    r = K.god(_pul(), 2026, K.PERIODY["wf"][0])
    assert (2026, 1) in _kvartaly(r["okno"])           # мутация действует, и тест выше на ней упал бы
    with pytest.raises(AssertionError):
        assert len(_kvartaly(r["okno"])) == 44 and r["okno"][1] == "2025-12-31"


def test_kalibr_wf_10_godov_poslednij_2026():
    pul = _pul()
    v = kv.kalibr(pul, *kv.parametry("wf"))
    god = kv._god(pul)[v.dni]
    assert sorted(set(god.tolist())) == list(range(2017, 2027))
    assert str(pul.daty[v.dni[-1]]) == "2026-02-27"
