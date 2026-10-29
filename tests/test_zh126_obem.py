"""Объём куба zh126 (Ж-127 п. 1): синтетика с известным ответом, зависимые оси,
общий yaml без точек/результатов. Сеть и данные не читаются."""
import copy
import importlib.util
import shutil
from pathlib import Path

import pytest
import yaml

KOREN = Path(__file__).resolve().parents[1]
OBSHCHEE = KOREN / "fastbt" / "designs" / "zh126" / "zh126_obshchee.yaml"
_spec = importlib.util.spec_from_file_location("obem_kuba", KOREN / "scripts" / "setki" / "obem_kuba.py")
ok = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ok)

SIGNAL = {"a": {"znacheniya": [1, 2, 3]}, "b": {"ot": 1, "do": 5, "shag": 2}, "t": {"znacheniya": ["x", "y"]}}
ZAV_C = {"os": "c", "roditel": "t", "znacheniya_po_roditelyu": {"x": [1, 2, 3], "y": [1]}}
ZAV_D = {"os": "d", "roditel": "t", "znacheniya_po_roditelyu": {"x": [1, 2], "y": [1, 2, 3]}}

SINT_OBSH = {
    "tip_strategii": {"S1": "kontr", "S3": "trend"},
    "rm": {"osi": {"m": {"znacheniya": [0, 5]}, "tip": {"znacheniya": ["p", "a"]}},
           "zavisimye_osi": [{"os": "v", "roditel": "tip", "znacheniya_po_roditelyu": {"p": [1, 2], "a": [1]}}]},
    "filtry": {"f1": {"znacheniya_po_tipu": {"kontr": [0, 1, 2], "trend": [0, 1]}}, "f2": {"znacheniya": [0, 1]}},
    "filtry_po_tipu": {"kontr": ["f1", "f2"], "trend": ["f1"]},
}


@pytest.fixture(scope="module")
def obsh():
    with open(OBSHCHEE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _zapisat(path, data):
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


def test_sintet_os_bez_zavisimyh():
    # 3 × 3 × 2 (t без детей — обычная ось)
    assert ok.obem_bloka(SIGNAL, []) == 18


def test_zavisimaya_os_uchityvaetsya():
    # 3 × 3 × (3 + 1); мутация «зависимость не учтена» дала бы 18 или 3·3·2·3 = 54
    assert ok.obem_bloka(SIGNAL, [ZAV_C]) == 36


def test_zavisimaya_os_v_osyah_ne_udvaivaetsya():
    osi = dict(SIGNAL, c={"znacheniya": [1, 2, 3]})
    assert ok.obem_bloka(osi, [ZAV_C]) == 36


def test_zavisimye_dve_u_odnogo_roditelya():
    # 3 × 3 × (3·2 + 1·3)
    assert ok.obem_bloka(SIGNAL, [ZAV_C, ZAV_D]) == 81


def test_zavisimye_slovarem():
    zav = {"c": {"roditel": "t", "znacheniya_po_roditelyu": ZAV_C["znacheniya_po_roditelyu"]}}
    assert ok.obem_bloka(SIGNAL, zav) == 36


def test_ot_do_shag_ne_nacelo_oshibka():
    with pytest.raises(ValueError):
        ok.chislo_znacheniy({"ot": 1, "do": 4, "shag": 2})


def test_sintet_katalog_itog(tmp_path, capsys):
    _zapisat(tmp_path / "x_obshchee.yaml", SINT_OBSH)
    _zapisat(tmp_path / "x_S1.yaml", {"signal": SIGNAL, "zavisimye_osi": [ZAV_C], "obem_signal": 36})
    _zapisat(tmp_path / "x_S3.yaml", {"signal": {"w": {"ot": 10, "do": 30, "shag": 10}}})
    assert ok.main(["--katalog", str(tmp_path)]) == 0
    vyvod = capsys.readouterr().out
    # РМ = 2 × (2 + 1) = 6; S1: 36 × 6 × 6 = 1296; S3: 3 × 6 × 2 = 36
    assert "РМ: 6 точек" in vyvod
    assert "= 1,296" in vyvod and "= 36" in vyvod
    assert "ИТОГО: 1,332 точек" in vyvod
    assert ok.chasy(1_800_000) == "10.0 ч на 1 процесс, 1.0 ч на 10 процессов"


def test_sintet_rashozhdenie_obem_signal(tmp_path, capsys):
    _zapisat(tmp_path / "x_obshchee.yaml", SINT_OBSH)
    _zapisat(tmp_path / "x_S1.yaml", {"signal": SIGNAL, "zavisimye_osi": [ZAV_C], "obem_signal": 18})
    assert ok.main(["--v5", str(tmp_path)]) == 1
    assert "РАСХОЖДЕНИЕ" in capsys.readouterr().out


def test_bez_s_faylov_rabotaet(tmp_path, capsys):
    shutil.copy(OBSHCHEE, tmp_path / "zh126_obshchee.yaml")
    assert ok.main(["--katalog", str(tmp_path)]) == 0
    vyvod = capsys.readouterr().out
    assert "S-файлов нет" in vyvod and "фильтры trend" in vyvod


def _gruppa(obsh, imya):
    rm = obsh["rm"]
    z = next(z for z in rm["zavisimye_osi"] if z["os"] == imya)
    return ok.obem_bloka({z["roditel"]: rm["osi"][z["roditel"]]}, [z])


def test_stop_25(obsh):
    assert _gruppa(obsh, "stop_znach") == 25


def test_teyk_33(obsh):
    assert _gruppa(obsh, "teyk_znach") == 33


def test_stop_teyk_mutaciya_sokrashcheniya_lovitsya(obsh):
    o = copy.deepcopy(obsh)
    z = next(z for z in o["rm"]["zavisimye_osi"] if z["os"] == "teyk_znach")
    z["znacheniya_po_roditelyu"]["pct"].remove(0.003)  # без 33-го из часового
    assert _gruppa(o, "teyk_znach") == 32 != _gruppa(obsh, "teyk_znach")


def test_rm_i_filtry_obem(obsh):
    assert ok.obem_rm(obsh) == 25 * 33 * 9 * 5 * 15
    assert ok.obem_filtrov(obsh, "kontr") == 7 * 6 * 6 * 6 * 7 * 4 * 4 * 5 * 4
    assert ok.obem_filtrov(obsh, "trend") == 5 * 6 * 6 * 6 * 7 * 4 * 4


def test_zapreshchennye_klyuchi_net(obsh):
    assert ok.zapreshchennye_klyuchi(obsh) == []


@pytest.mark.parametrize("klyuch", sorted(ok.ZAPRET))
def test_zapreshchennye_klyuchi_mutaciya_lovitsya(obsh, klyuch):
    o = copy.deepcopy(obsh)
    o["filtry"]["atr_lo"][klyuch] = 1
    o["rm"]["zavisimye_osi"][0][klyuch.upper()] = [{"x": 1}]
    ozhid = [f"filtry.atr_lo.{klyuch}", f"rm.zavisimye_osi[0].{klyuch.upper()}"]
    assert sorted(ok.zapreshchennye_klyuchi(o)) == ozhid


def test_zapreshchennye_klyuchi_kod_vyhoda(tmp_path):
    o = copy.deepcopy(SINT_OBSH)
    o["rm"]["best"] = {"m": 5}
    _zapisat(tmp_path / "x_obshchee.yaml", o)
    assert ok.main(["--katalog", str(tmp_path)]) == 1


def test_istochnik_u_kazhdoy_osi(obsh):
    bloki = list(obsh["rm"]["osi"].values()) + obsh["rm"]["zavisimye_osi"] + list(obsh["filtry"].values())
    assert bloki and all(b.get("istochnik_stroki") for b in bloki)


def test_prichinnoe_opredelenie_procentiley(obsh):
    for imya in ("atr_lo", "squeeze_pct"):
        assert obsh["filtry"][imya]["opredelenie"].startswith("причинное: окно до d включительно")


def test_filtry_po_tipu_sushchestvuyut(obsh):
    for tip in ("kontr", "trend"):
        assert set(obsh["filtry_po_tipu"][tip]) <= set(obsh["filtry"])
    assert set(obsh["tip_strategii"].values()) == {"kontr", "trend"}
