"""T1 (Ж-036, #308): развёртка сетки Э1-б `fastbt/sweep_setka.py` и разбор плоских параметров `sweep_post.split`.
Ключи условных блоков S5/S6 не сталкиваются; «вне v1» не попадает в сетку. Синтетическая сетка, без данных."""
import pytest

from fastbt import sweep_post as sp
from fastbt import sweep_setka as st

SIGN = {"filter_set"}
SETKA = {"bloki": {
    "signal": {"parametry": [{"imya": "filter_set", "ishodnoe": 0, "setka": [0, 1]}]},
    "rm_atr": {"uslovie": {"rm_tip": "atr"}, "parametry": [
        {"imya": "max_hold", "ishodnoe": 20, "setka": [10, 20, 30]},
        {"imya": "cooldown_bars", "ishodnoe": 0, "setka": [0, 2]}]},
    "rm_pivot": {"uslovie": {"rm_tip": "pivot"}, "parametry": [
        {"imya": "max_hold", "ishodnoe": 8, "setka": [4, 8, 12]},
        {"imya": "cooldown_bars", "ishodnoe": 0, "setka": [0, 3]}]},
    "filtry": {"parametry": [
        {"imya": "adx_min", "ishodnoe": 20, "setka": [10, 20, 30], "pravka_znach": [10]},
        {"imya": "obyom_nizhe_sma5", "ishodnoe": "on", "setka": ["on", "off"], "dostup": "pravka"}]},
    "rm_zadel": {"parametry": [{"imya": "k", "ishodnoe": 1, "setka": [1, 2], "dostup": "pravka"}]},   # √RV с Э1-в не «вне v1»
    "rm_vybor": {"parametry": [{"imya": "rm_tip", "ishodnoe": "atr", "setka": ["atr", "pivot", "sqrt_rv"], "pravka_znach": ["sqrt_rv"]}]},
}}


def ok_keys_unique(expand=st.expand) -> bool:
    e = expand(SETKA)
    return (e["grid"].get("rm_atr.max_hold") == [10, 20, 30] and e["grid"].get("rm_pivot.max_hold") == [4, 8, 12]
            and e["base"].get("rm_atr.max_hold") == 20 and e["base"].get("rm_pivot.max_hold") == 8
            and "max_hold" not in e["grid"] and "cooldown_bars" not in e["grid"] and "filter_set" in e["grid"]
            and len(e["base"]) == len(e["grid"]))


def ok_split_no_collision(split=sp.split) -> bool:
    p = {"rm_tip": "pivot", "rm_atr.max_hold": 5, "rm_pivot.max_hold": 12, "rm_atr.cooldown_bars": 7}
    a = split("S5", p, SIGN)
    b = split("S5", {**p, "rm_tip": "atr"}, SIGN)
    return (a[0] == "pivot" and a[1]["max_hold"] == 12 and a[1]["cooldown_bars"] == 0
            and b[0] == "atr" and b[1]["max_hold"] == 5 and b[1]["cooldown_bars"] == 7)


def test_klyuchi_uslovnykh_blokov_ne_stalkivayutsya():
    assert ok_keys_unique() and ok_split_no_collision()


def test_oat_base_podstavlyaet_uslovie_bloka():
    e = st.expand(SETKA)
    assert st.oat_base(e, "rm_pivot.max_hold")["rm_tip"] == "pivot" and st.oat_base(e, "rm_atr.max_hold")["rm_tip"] == "atr"
    assert st.oat_base(e, "adx_min") == e["base"]
    p = st.oat_base(e, "rm_pivot.max_hold")
    assert sp.split("S5", {**p, "rm_pivot.max_hold": 12}, SIGN)[1]["max_hold"] == 12     # условие + значение доходят до RM


def test_vne_v1_ne_popadaet_v_setku():
    e = st.expand(SETKA)
    assert e["grid"]["adx_min"] == [20, 30]                                           # pravka_znach убрано
    assert "obyom_nizhe_sma5" not in e["grid"] and "k" not in e["grid"]               # dostup=pravka, блок √RV
    assert e["grid"]["rm_tip"] == ["atr", "pivot"]                                    # sqrt_rv убрано
    assert len(e["vne_v1"]) >= 4 and any("adx_min" in s for s in e["vne_v1"])


def test_split_vne_v1_i_neizvestnyy_klyuch():
    assert sp.split("S5", {"rm_tip": "sqrt_rv"}, SIGN)[0] == "sqrt_rv"                  # Э1-в: считается fastbt/rm_sqrt_rv.py
    with pytest.raises(sp.VneV1):
        sp.split("S3", {"trail_type": "off"}, SIGN)
    with pytest.raises(KeyError):
        sp.split("S5", {"rm_pivot.net_takogo": 1}, SIGN)


def test_neaktivnyy_blok_ignoriruetsya():
    r = sp.split("S5", {"rm_tip": "atr", "rm_pivot.max_hold": 99}, SIGN)
    assert r[1]["max_hold"] == sp.RM_DEF["atr"]["max_hold"]


def test_mut_setka_klyuchi_bez_bloka(monkeypatch):
    """Мутация: имя без префикса блока -> max_hold двух блоков сливается."""
    assert ok_keys_unique()

    def flat(setka):
        e = st.expand(setka)
        return {**e, "base": {k.rpartition(".")[2]: v for k, v in e["base"].items()},
                "grid": {k.rpartition(".")[2]: v for k, v in e["grid"].items()}}
    assert not ok_keys_unique(flat)


def test_mut_split_beret_imya_bez_bloka():
    """Мутация: prefix блока отброшен перед разбором -> побеждает последний max_hold, неактивный блок протекает."""
    assert ok_split_no_collision()

    def flat(sid, p, names):
        return sp.split(sid, {k.rpartition(".")[2] if k.startswith("rm_") and "." in k else k: v for k, v in p.items()}, names)
    assert not ok_split_no_collision(flat)
