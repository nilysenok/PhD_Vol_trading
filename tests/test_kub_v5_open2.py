"""К-wf5 (Ж-137 п. 6–7): якорь t_vyhod кубов v5 и v6 — open2 (выход по сигналу дня x по open x+1, Д2). Синтетика и yaml, данных нет."""
import pytest

from fastbt import canon_v5 as C5
from fastbt import canon_v6 as C6
from fastbt import eval_v5_osi as OSI
from fastbt.exec_e_vremya import vyhod_plan

SIDS = C5.SIDS


def osnovnaya_vyhod_po_signalu_d_plus_1(sid: str) -> bool:
    """Основная строка: якорь куба доходит до разбора, и выход по сигналу дня d исполняется не в день d (сдвиг ≥ 1)."""
    ya = C5.load_kub(sid)["yakorya_progon"]
    sp = {n: v["setka"][len(v["setka"]) // 2] for n, v in C5.load_kub(sid)["signal"].items()}
    tv = OSI.razbor(sid, {**sp, **ya, "stop_nach": "atr:2.0"}).nov["t_vyhod"]
    return vyhod_plan(tv, "signal")[0] >= 1


@pytest.mark.parametrize("sid", SIDS)
def test_yakor_v5_open2_i_dohodit_do_ispolneniya(sid):
    assert C5.load_kub(sid)["yakorya_progon"]["t_vyhod"] == "open2"
    assert osnovnaya_vyhod_po_signalu_d_plus_1(sid)


@pytest.mark.parametrize("sid", SIDS)
def test_mutaciya_vyhod_po_close_d_v_osnovnoy_stroke_lovitsya(sid, monkeypatch):
    """Мутация: якорь t_vyhod = close (выход по сигналу d по close d). Тест основной строки обязан упасть."""
    OSI._yakorya_progon.cache_clear()
    kub = C5.load_kub(sid)
    kub["yakorya_progon"]["t_vyhod"] = "close"
    monkeypatch.setattr(OSI, "load_kub", lambda s: kub)
    monkeypatch.setattr(C5, "load_kub", lambda s: kub)
    try:
        assert not osnovnaya_vyhod_po_signalu_d_plus_1(sid)
    finally:
        OSI._yakorya_progon.cache_clear()


@pytest.mark.parametrize("sid", SIDS)
def test_v6_t_vyhod_vykl_open2_sovpadaet_s_yakorem_v5(sid):
    ox = C6.novye(C6.load_kub(sid))["t_vyhod"]
    assert [str(x) for x in ox["setka"]] == ["open2", "close", "1500", "1700"]
    assert ox["vykl"] == C5.load_kub(sid)["yakorya_progon"]["t_vyhod"]
