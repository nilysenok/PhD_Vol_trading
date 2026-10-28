"""Реестр отбора A на В1: годы-колонки, полнота записи, старый реестр нетронут. Мутации — в самих тестах."""
import csv
import hashlib
import re
from pathlib import Path

import pytest

from fastbt import reestr_otbora as old
from fastbt import reestr_otbora_v1 as R

ROOT = Path(__file__).resolve().parents[1]
STARYJ = ROOT / "docs/otbor/reestr_otbora_A.csv"
MANIFEST = ROOT / "docs/otchety/2026-10-28_arhiv_reestr_2018_2019.md"


def stroka(i, period="train4", s="S1"):
    r = {c: "" for c in R.kolonki(period)}
    r.update(strategiya=s, klyuch=f"z_entry={i}", rezhim="oat", istochnik="v1_it1", status="kandidat")
    return r


def partiya(n, period="train4"):
    return [stroka(i, period) for i in range(n)]


def test_kolonki_v1_imeyut_2016_2019_i_itog():
    k = R.kolonki("train4")
    assert [f"sharpe_{y}" for y in (2016, 2017, 2018, 2019)] == [c for c in k if c.startswith("sharpe_20")]
    assert "sharpe_train4" in k and "sharpe_2016" in R.kolonki("train4")
    assert R.kolonki("train") != k and "sharpe_2016" not in R.kolonki("train")


def test_fajl_v1_shapka_sovpadaet_s_kolonkami():
    with open(ROOT / "docs/otbor/reestr_otbora_A_v1.csv", encoding="utf-8") as f:
        rows = [l for l in f if not l.startswith("#")]
    assert next(csv.reader(rows[:1])) == R.kolonki("train4") and len(rows) == 1


def test_zapis_chislo_strok_ravno_chislu_ocenennyh(tmp_path):
    p = tmp_path / "r.csv"
    assert R.dopisat(partiya(5), "train4", 5, p) == 5
    assert len(R.chitat(str(p))) == 5


def test_mutaciya_kandidat_propushchen_lovitsya(tmp_path):
    p = tmp_path / "r.csv"
    ocenennye = 5
    s_propuskom = partiya(5)[:-1]  # мутация: один кандидат не записан
    with pytest.raises(ValueError, match="пропущен"):
        R.dopisat(s_propuskom, "train4", ocenennye, p)
    assert not p.exists()


def test_mutaciya_god_2016_ne_v_kolonkah_lovitsya(tmp_path, monkeypatch):
    p = tmp_path / "r.csv"
    rows = partiya(2)
    monkeypatch.setitem(R.PERIODY, "train4", (2017, 2018, 2019))  # мутация: 2016 выпал из колонок
    with pytest.raises(ValueError, match="2016"):
        R.dopisat(rows, "train4", 2, p)
    assert not p.exists()


def test_stroka_bez_2016_ne_prinimaetsya(tmp_path):
    r = partiya(1)
    del r[0]["sharpe_2016"]
    with pytest.raises(ValueError):
        R.dopisat(r, "train4", 1, tmp_path / "r.csv")


def test_tolko_dopisyvaet_povtor_nichego_ne_dobavlyaet(tmp_path):
    p = tmp_path / "r.csv"
    R.dopisat(partiya(3), "train4", 3, p)
    pered = p.read_bytes()
    assert R.dopisat(partiya(3), "train4", 3, p) == 0 and p.read_bytes() == pered
    assert R.dopisat(partiya(4), "train4", 4, p) == 1 and p.read_bytes().startswith(pered)


def test_shapka_drugogo_perioda_otkaz(tmp_path):
    p = tmp_path / "r.csv"
    R.dopisat(partiya(1, "train"), "train", 1, p)
    with pytest.raises(ValueError, match="шапка"):
        R.dopisat(partiya(1), "train4", 1, p)


def test_staryj_reestr_sha256_v_manifeste():
    m = re.search(r"`([0-9a-f]{64})`", MANIFEST.read_text(encoding="utf-8")).group(1)
    assert hashlib.sha256(STARYJ.read_bytes()).hexdigest() == m


def test_train_zapis_v_staryj_reestr_pobajtno_prezhnyaya(tmp_path):
    assert R.VYHOD["train"] == old.VYHOD
    rows = old.chitat(str(STARYJ))
    out = tmp_path / "copy.csv"
    old.zapisat(rows, str(out))
    # хвост файла дописан etap1_ansambl_run через pandas (\n вместо \r\n csv-писателя): различие только в концах строк
    assert out.read_bytes().replace(b"\r\n", b"\n") == STARYJ.read_bytes().replace(b"\r\n", b"\n")
    assert STARYJ.read_bytes().startswith(out.read_bytes()[:6911985])
