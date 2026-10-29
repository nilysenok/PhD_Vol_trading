"""Spr-fix (Ж-129 п. 4, Ж-137 п. 6): защита «оси zh047 только на якоре» пропускает t_vyhod = close ТОЛЬКО роли spravka. Синтетика, данных нет."""
import inspect

import pytest

from fastbt import canon_v5 as C5
from fastbt import eval_v5, eval_v6
from fastbt import eval_v5_osi as OSI

SIDS = C5.SIDS


def tochka(sid: str, **ya_zamena) -> dict:
    kub = C5.load_kub(sid)
    sp = {n: v["setka"][len(v["setka"]) // 2] for n, v in kub["signal"].items()}
    return {**sp, **kub["yakorya_progon"], **ya_zamena, "stop_nach": "atr:2.0"}


@pytest.mark.parametrize("sid", SIDS)
def test_spravka_close_schitaetsya(sid):
    assert OSI.razbor(sid, tochka(sid, t_vyhod="close"), rol="spravka").nov["t_vyhod"] == "close"


@pytest.mark.parametrize("sid", SIDS)
def test_osnovnoy_prohod_s_close_otkaz(sid):
    for rol in (None, "sobol", "kontrol"):
        with pytest.raises(ValueError, match="не на якоре"):
            OSI.razbor(sid, tochka(sid, t_vyhod="close"), rol=rol)


@pytest.mark.parametrize("sid", SIDS)
def test_spravka_drugaya_os_vne_yakorya_otkaz(sid):
    ya = C5.load_kub(sid)["yakorya_progon"]
    for k in ya:
        if k != "t_vyhod":
            with pytest.raises(ValueError, match="не на якоре"):
                OSI.razbor(sid, tochka(sid, **{k: "chuzhoe"}), rol="spravka")
    with pytest.raises(ValueError, match="не на якоре"):
        OSI.razbor(sid, tochka(sid, t_vyhod="1500"), rol="spravka")      # у spravka разрешён только close


@pytest.mark.parametrize("sid", SIDS)
def test_spravka_na_yakore_dayot_yakor(sid):
    assert OSI.razbor(sid, tochka(sid), rol="spravka").nov["t_vyhod"] == "open2"


def test_mutaciya_zashchita_snyata_dlya_vseh_rolei_lovitsya(monkeypatch):
    """Мутация: razbor пропускает close при любой роли. Тест основного прохода обязан упасть (close не отвергнут)."""
    nast = OSI.razbor
    monkeypatch.setattr(OSI, "razbor", lambda sid, pt, signal=None, rol=None: nast(sid, pt, signal, "spravka"))
    with pytest.raises(pytest.fail.Exception):
        test_osnovnoy_prohod_s_close_otkaz(SIDS[0])


@pytest.mark.parametrize("cls", [eval_v5.EvaluatorV5, eval_v6.EvaluatorV6])
def test_rol_peredayotsya_otsenshchiku_yavno(cls):
    assert inspect.signature(cls.__init__).parameters["rol"].default is None


def test_v6_razbor_poluchaet_rol():
    assert "self.rol)" in inspect.getsource(eval_v6.EvaluatorV6.evaluate_v2)
    assert "self.rol)" in inspect.getsource(eval_v5.EvaluatorV5.evaluate_v2)
