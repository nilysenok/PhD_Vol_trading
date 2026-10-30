"""Тестировщик #818/#819 (пакет wf): мутации и проверки, которых нет в тестах авторов. Синтетика; реальный br_contracts.json — только на чтение
через FASTBT_MAIN, иначе skip. Мутации: PIT-правило включено для границы 2021; запись без frsttrade; период по умолчанию wf; v6 игнорирует период;
perenos_do 2026-02-27. Shell proverka_yakor.sh — в test_wf_paket_mut_819_sh.py."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRV = ROOT / "scripts/server"
sys.path.insert(0, str(SRV))


def _modul(imya, put):
    s = importlib.util.spec_from_file_location(imya, put)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


JD = _modul("json_daty_t", SRV / "json_daty.py")
ET = _modul("test_etalony_wf_obr", ROOT / "tests/test_etalony_wf.py")       # Fake-оценщик и сборка эталона авторов #819
ekviv, ekviv_io, ps, e6 = ET.ekviv, ET.ekviv_io, ET.ps, ET.e6
etalony_mak, proverka_ekviv = ET.etalony_mak, ET.proverka_ekviv
from fastbt import vneshnie_1850 as V  # noqa: E402

DO = "2026-02-28"


def _z(month, secid, fr, ls):
    return {"month": month, "secid": secid, "shortname": secid, "frsttrade": fr, "lsttrade": ls}


def _usech(tmp, zapisi, granica=None):
    src, dst = tmp / "s.json", tmp / "o" / "d.json"
    src.write_text(json.dumps(zapisi))
    st = JD.usech(src, dst, granica)
    return st, json.loads(dst.read_text())


# ---------------------------------------------------------------- #818

def test_pit_tolko_dlya_granicy_posle_2021(tmp_path):
    z = [_z("2022-01", "BRF2", "2021-04-01", "2022-01-14")]       # frsttrade 2021, lsttrade 2022
    assert _usech(tmp_path, z)[1] == []                           # граница по умолчанию 2021-12-31: даты 2022 не попадают
    assert _usech(tmp_path, z, "2021-12-31")[1] == []
    assert len(_usech(tmp_path, z, DO)[1]) == 1                   # wf: по frsttrade
    assert JD.max_data(tmp_path / "s.json") == pd.Timestamp("2022-01-14")          # pit=False: все даты
    assert JD.max_data(tmp_path / "s.json", pit=True) == pd.Timestamp("2021-04-01")


def test_mutatsiya_pit_dlya_2021_lovitsya(tmp_path, monkeypatch):
    """Подложено: pit включён всегда (граница 2021 сравнивается с «меньше 2020») -> в пакет ≤ 2021 просачивается контракт 2022."""
    z = [_z("2022-01", "BRF2", "2021-04-01", "2022-01-14")]
    monkeypatch.setattr(JD, "GRANICA", pd.Timestamp("2020-01-01"))
    assert len(_usech(tmp_path, z, "2021-12-31")[1]) == 1         # мутация видна: запись 2022 осталась
    monkeypatch.undo()
    assert _usech(tmp_path, z, "2021-12-31")[1] == []


def test_wf_frsttrade_posle_granicy_vyhodit_dazhe_esli_lsttrade_rano(tmp_path):
    z = [_z("2026-01", "BRX", "2026-03-10", "2026-01-05")]        # нелепая, но проверяет, что границей служит именно frsttrade
    assert _usech(tmp_path, z, DO)[1] == []


def test_wf_zapis_bez_frsttrade_po_vsem_datam(tmp_path):
    """Запись без frsttrade (или frsttrade не дата) — все даты записи; мутация «pit по месяцу/lsttrade» не пройдёт."""
    z = [{"month": "2026-05", "lsttrade": "2026-05-04"}, {"month": "2026-01", "lsttrade": "2026-01-05"},
         {"month": "2026-01", "frsttrade": None, "lsttrade": "2026-01-05"}, {"month": "2026-01", "frsttrade": "abc", "lsttrade": "2026-03-05"}]
    kopiya = _usech(tmp_path, z, DO)[1]
    assert kopiya == [z[1], z[2]]


def test_lyuboj_den_yanvar_fevral_2026_imeet_aktivnyj_kontrakt():
    """(б) Реальный br_contracts.json, усечённый на 2026-02-28: активный контракт не None на каждый будний день 2026-01-01…2026-02-27."""
    main = os.environ.get("FASTBT_MAIN")
    p = Path(main) / "data/iss_ext/br_contracts.json" if main else None
    if p is None or not p.exists():
        pytest.skip("нет FASTBT_MAIN/data/iss_ext/br_contracts.json")
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        _, kopiya = _usech(Path(t), json.loads(p.read_text()), DO)
    c = pd.DataFrame(kopiya)[["secid", "lsttrade"]]
    c["lsttrade"] = pd.to_datetime(c["lsttrade"])
    c = c.sort_values("lsttrade").reset_index(drop=True)
    c["roll"] = [V.roll_date(x) for x in c["lsttrade"]]
    dni = pd.bdate_range("2026-01-01", "2026-02-27")
    a = V.active_secid(dni, c)
    assert len(dni) == 42 and not a.isna().any()


def test_oglashenie_dekabr_2021_bez_kontrakta_v_pakete_do_2021(tmp_path):
    """(г) Фиксирует оговорку к счёту train4: в пакете ≤ 2021 контракт BRF2 (lsttrade 2022-01-03) выброшен «все даты ≤ границы» ->
    с ролла BRZ1 (2021-11-24) активного контракта нет. Если правило изменят — тест надо пересмотреть сознательно."""
    z = [_z("2021-11", "BRX1", "2020-10-26", "2021-11-01"), _z("2021-12", "BRZ1", "2020-11-25", "2021-12-01"),
         _z("2022-01", "BRF2", "2020-12-30", "2022-01-03")]
    kopiya = _usech(tmp_path, z, "2021-12-31")[1]
    c = pd.DataFrame(kopiya)[["secid", "lsttrade"]]
    c["lsttrade"] = pd.to_datetime(c["lsttrade"])
    c["roll"] = [V.roll_date(x) for x in c["lsttrade"]]
    a = V.active_secid(pd.bdate_range("2021-11-22", "2021-12-31"), c)
    assert a.loc["2021-11-23"] == "BRZ1" and a.loc["2021-11-24":].isna().all()


# ---------------------------------------------------------------- #819

def test_period_po_do_27_fevralya_otkaz():
    for plokho in ("2026-02-27", "2026-2-28", "2026-02-28T", "wf"):
        with pytest.raises(ValueError):
            ekviv.period_po_do(plokho)
    assert ekviv.period_po_do(" 2026-02-28 ") == "wf"


def test_etalony_i_proverka_otkazyvayut_pri_27_fevralya(monkeypatch, tmp_path):
    ET._sreda(monkeypatch, "wf")
    with pytest.raises(ValueError):
        etalony_mak.main(["--out", str(tmp_path / "e.npz"), "--do", "2026-02-27", *ET.ARGS])
    f = ET._sobrat(tmp_path, DO)
    ps._W.clear()
    otm = tmp_path / "pr/OK"
    with pytest.raises(ValueError):
        proverka_ekviv.main(["--etalony", f, "--kod-sha", "k", "--dannye-sha", "d", "--procs", "1", "--otmetka", str(otm), "--do", "2026-02-27"])
    assert not otm.exists()


def test_period_po_umolchaniyu_train4_a_ne_wf(monkeypatch, tmp_path):
    ET._sreda(monkeypatch, "train4")
    monkeypatch.delenv("PERENOS_DO", raising=False)
    f = str(tmp_path / "e.npz")
    assert etalony_mak.main(["--out", f, *ET.ARGS]) == 0
    assert "period" not in ekviv_io.zagruzit(f)[1] and set(ET.Fake.vidennye) == {"train4"}


def test_mutatsiya_umolchanie_wf_lovitsya(monkeypatch, tmp_path):
    ET._sreda(monkeypatch, "train4")
    monkeypatch.delenv("PERENOS_DO", raising=False)
    monkeypatch.setitem(ekviv.PERIOD_PO_DO, "", "wf")             # подложено: пустой perenos_do = wf
    with pytest.raises((PermissionError, ValueError)):
        etalony_mak.main(["--out", str(tmp_path / "e.npz"), *ET.ARGS])


def test_v6_tochki_wf_schitayutsya_v_zone_wf_a_ne_dev(monkeypatch, tmp_path):
    ET._sreda(monkeypatch, "wf")
    rez6, m6, _ = e6.poschitat(None, set(), 1, 12, "wf")
    assert len(rez6) >= 18 and set(ET.Fake.vidennye) == {"wf"}
    assert all('"period": "wf"' in r["sig"] for m, r in rez6.items() if m.startswith("v6tochka__"))
    ps._W.clear()


def test_mutatsiya_v6_ignoriruet_period_lovitsya(monkeypatch):
    """Подложено: rezhim() v6 всегда берёт период dev (train4) -> оценщик wf отказывает (PermissionError), а не молча считает dev."""
    ET._sreda(monkeypatch, "wf")
    orig = e6.rezhim
    monkeypatch.setattr(e6, "rezhim", lambda sid, period=None: orig(sid, None))
    with pytest.raises((PermissionError, ValueError)):          # зона данных / проверка дат оценщика: отказ, не молчаливый счёт dev
        e6.poschitat(None, set(), 1, 12, "wf")


def test_wf_tochki_v5_otkaz():
    with pytest.raises(ValueError):
        ekviv.poschitat([("S1", 1)], 1, "wf")


def test_etalon_train4_protiv_do_wf_fail_kod_1(monkeypatch, tmp_path):
    ET._sreda(monkeypatch, "train4")
    f = ET._sobrat(tmp_path, "")
    ps._W.clear()
    otm = tmp_path / "pr/OK"
    a = ["--etalony", f, "--kod-sha", "k", "--dannye-sha", "d", "--procs", "1", "--otmetka", str(otm)]
    assert proverka_ekviv.main([*a, "--do", DO]) == 1 and not otm.exists()

