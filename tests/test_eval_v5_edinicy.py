"""Л9: единицы, знак и запрет zh047 в разборе точки куба v5 (fastbt/eval_v5_osi.py). Синтетика, данных не читает.

Проверка по известному ответу (proverka.md п. 5): значение оси → поле Params / единица / знак посчитаны вручную.
Мутации, которые тесты обязаны ловить (комментарий проверяющего к PR #611):
в1 ed_t тейка atr↔pct; в2 стоп pct считается как atr; в3 k_e ×2 у контр-входа и минус у тренда; г снята проверка zh047.
"""
import pytest

from fastbt import canon_v5
from fastbt.eval_v5_osi import PT4, razbor, yakor

KONTR = ("S1", "S2", "S5", "S6")
TREND = ("S3", "S4")


def _tochka(sid, **osi):
    """Точка v5: сигнальные оси — середина сетки, остальные — якорь куба, поверх — osi."""
    k = canon_v5.load_kub(sid)
    sp = {n: v["setka"][len(v["setka"]) // 2] for n, v in k["signal"].items()}
    return {**sp, **yakor(sid), **osi}


@pytest.mark.parametrize("sid", KONTR)
@pytest.mark.parametrize("tejk,ed", [("pct:2.5", "pct"), ("atr:2.5", "atr")])
def test_tejk_edinica(sid, tejk, ed):
    ext = razbor(sid, _tochka(sid, tejk=tejk)).ext
    assert ext["ed_t"] == ed and ext["a"] == 2.5


@pytest.mark.parametrize("sid", KONTR)
@pytest.mark.parametrize("stop,ed", [("pct:3.0", "pct"), ("atr:3.0", "atr"), ("chand:3.0", "atr")])
def test_stop_edinica(sid, stop, ed):
    r = razbor(sid, _tochka(sid, stop_nach=stop))
    assert r.ext["ed_s"] == ed and r.pt4["k_s"] == 3.0


@pytest.mark.parametrize("sid", KONTR + TREND)
def test_stop_sigma_bez_ed_s(sid):
    r = razbor(sid, _tochka(sid, stop_nach="sigma:1.5"))
    assert "ed_s" not in r.ext and r.pt4["k_s"] == 1.5


@pytest.mark.parametrize("sid", KONTR)
def test_stop_i_tejk_nezavisimy(sid):
    ext = razbor(sid, _tochka(sid, stop_nach="pct:3.0", tejk="atr:1.5")).ext
    assert (ext["ed_s"], ext["ed_t"], ext["a"]) == ("pct", "atr", 1.5)


@pytest.mark.parametrize("sid", KONTR)
def test_k_e_kontr_limit_odin_k_odnomu(sid):
    r = razbor(sid, _tochka(sid, k_e=0.75))
    assert r.pt4["vhod"] == 0.75 and "k_e" not in r.ext          # не ×2, и не в ext
    assert razbor(sid, _tochka(sid, k_e=0)).pt4["vhod"] == 0.0


@pytest.mark.parametrize("sid", TREND)
@pytest.mark.parametrize("k_e", [-0.5, -0.25, 0.25])
def test_k_e_trend_znak_sohranyaetsya(sid, k_e):
    r = razbor(sid, _tochka(sid, k_e=k_e))
    assert r.ext["k_e"] == k_e                                    # знак как в точке: минус остаётся минусом
    assert r.nov["vhod_tip"] == "stop_zayavka" and r.pt4["vhod"] == PT4["vhod"]


@pytest.mark.parametrize("sid", TREND)
def test_k_e_trend_nol_ne_pishetsya(sid):
    assert "k_e" not in razbor(sid, _tochka(sid, k_e=0)).ext


@pytest.mark.parametrize("sid", canon_v5.SIDS)
def test_zh047_ne_na_yakore_valueerror(sid):
    ya = canon_v5.load_kub(sid)["yakorya_progon"]
    for os, yak in ya.items():
        drugoe = "ne_yakor_" + str(yak)
        with pytest.raises(ValueError, match="zh047"):
            razbor(sid, {**_tochka(sid), os: drugoe})


@pytest.mark.parametrize("sid", canon_v5.SIDS)
def test_zh047_na_yakore_dopustimo_i_ne_vliyaet(sid):
    ya = canon_v5.load_kub(sid)["yakorya_progon"]
    base = razbor(sid, _tochka(sid))
    assert razbor(sid, {**_tochka(sid), **ya}) == base


def test_neizvestnaya_os_valueerror():
    with pytest.raises(ValueError, match="не знает"):
        razbor("S1", {**_tochka("S1"), "zzz": 1})
