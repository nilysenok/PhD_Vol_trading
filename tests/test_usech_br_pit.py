"""Усечение br_contracts.json (Д-c): запись входит по frsttrade <= границы (PIT спецификации), а не «все даты <= границы».
Мутация «все даты» (контракт с lsttrade после границы выброшен) -> активный контракт на 27.02.2026 пропадает -> тесты падают."""
import importlib.util
import json
import os
from pathlib import Path

import pandas as pd
import pytest

SRV = Path(__file__).resolve().parents[1] / "scripts/server"
_s = importlib.util.spec_from_file_location("json_daty", SRV / "json_daty.py")
JD = importlib.util.module_from_spec(_s)
_s.loader.exec_module(JD)

from fastbt import vneshnie_1850 as V  # noqa: E402

DO = "2026-02-28"


def _z(month, secid, fr, ls):
    return {"month": month, "secid": secid, "shortname": secid, "frsttrade": fr, "lsttrade": ls}


SINT = [
    _z("2026-01", "BRF6", "2024-12-25", "2026-01-05"),
    _z("2026-02", "BRG6", "2025-01-28", "2026-02-02"),
    _z("2026-03", "BRH6", "2025-02-25", "2026-03-02"),     # lsttrade после границы, спецификация известна
    _z("2026-04", "BRJ6", "2025-03-26", "2026-04-01"),
    _z("2026-09", "BRU6", "2026-03-10", "2026-09-01"),     # frsttrade после границы: неизвестен
]


def _tablica(zapisi) -> pd.DataFrame:
    c = pd.DataFrame(zapisi)[["secid", "lsttrade"]]
    c["lsttrade"] = pd.to_datetime(c["lsttrade"])
    c = c.sort_values("lsttrade").reset_index(drop=True)
    c["roll"] = [V.roll_date(x) for x in c["lsttrade"]]
    return c


def _usech(tmp: Path, zapisi, granica=DO):
    src, dst = tmp / "src.json", tmp / "o" / "dst.json"
    src.write_text(json.dumps(zapisi))
    st = JD.usech(src, dst, granica)
    return st, json.loads(dst.read_text())


def _aktivnyj(zapisi, den="2026-02-27"):
    return V.active_secid([den], _tablica(zapisi)).iloc[0]


def test_sintetika_kontrakt_posle_granicy_po_lsttrade_ostayotsya(tmp_path):
    st, kopiya = _usech(tmp_path, SINT)
    assert [z["secid"] for z in kopiya] == ["BRF6", "BRG6", "BRH6", "BRJ6"]
    assert st[:2] == (4, 1)
    assert _aktivnyj(kopiya) is not None and _aktivnyj(kopiya) == _aktivnyj(SINT[:4])


def test_mutatsiya_vse_daty_lovitsya(tmp_path, monkeypatch):
    """Старое правило (все даты записи <= границы) подложено: тест активного контракта обязан упасть."""
    monkeypatch.setattr(JD, "daty_zapisi", lambda z, pit=False: JD._daty(z))
    _, kopiya = _usech(tmp_path, SINT)
    assert _aktivnyj(kopiya) is None                      # мутация ломает именно то, что ловит тест выше
    with pytest.raises(AssertionError):
        assert [z["secid"] for z in kopiya] == ["BRF6", "BRG6", "BRH6", "BRJ6"]


def test_max_data_po_frsttrade_a_ne_po_lsttrade(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(SINT[:4]))
    assert JD.max_data(p, pit=True) == pd.Timestamp("2025-03-26")          # lsttrade 2026-04-01 манифест wf не отвергает
    assert JD.max_data(p) == pd.Timestamp("2026-04-01")                     # пакет ≤ 2021: прежнее правило
    p.write_text(json.dumps(SINT))
    assert JD.max_data(p, pit=True) == pd.Timestamp("2026-03-10")          # frsttrade после границы -> манифест отвергнет


def test_wf_kopiya_prohodit_manifest_a_original_2021_net(tmp_path):
    MP = _mp()
    st, _ = _usech(tmp_path, SINT)
    dst = str(tmp_path / "o" / "dst.json")
    assert MP.proverit(dst, "", pd.Timestamp(DO)) == ("ok", "2025-03-26")
    assert MP.proverit(str(tmp_path / "src.json"), "", pd.Timestamp(DO))[0] == "net"      # BRU6 с frsttrade 2026-03-10
    assert MP.proverit(dst, "")[0] == "net"                                                 # в пакете ≤ 2021 даты 2026 не принимаются


def test_granica_2021_pryezhnee_pravilo_bez_dat_2022(tmp_path):
    z = [_z("2021-12", "BRZ1", "2021-03-01", "2021-12-15"), _z("2022-01", "BRF2", "2021-04-01", "2022-01-14")]
    st, kopiya = _usech(tmp_path, z, "2021-12-31")
    assert [x["secid"] for x in kopiya] == ["BRZ1"] and st == (1, 1, "2021-12-15")


def _mp():
    spec = importlib.util.spec_from_file_location("manifest_perenos", SRV / "manifest_perenos.py")
    m = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(SRV))
    spec.loader.exec_module(m)
    return m


def test_zapis_bez_frsttrade_po_vsem_datam(tmp_path):
    _, kopiya = _usech(tmp_path, [{"month": "2026-05", "x": 1}, {"month": "2026-01", "x": 2}])
    assert kopiya == [{"month": "2026-01", "x": 2}]


def test_realnyj_br_contracts_aktivnyj_na_27_02_2026(tmp_path):
    main = os.environ.get("FASTBT_MAIN")
    p = Path(main) / "data/iss_ext/br_contracts.json" if main else None
    if p is None or not p.exists():
        pytest.skip("нет FASTBT_MAIN/data/iss_ext/br_contracts.json")
    _, kopiya = _usech(tmp_path, json.loads(p.read_text()))
    assert _aktivnyj(kopiya) is not None
    assert not any(z["frsttrade"] > DO for z in kopiya)
