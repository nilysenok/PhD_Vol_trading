"""Реестр отбора A на В1: годы-колонки, полнота записи, старый реестр нетронут. Мутации — в самих тестах."""
import csv
import hashlib
import pickle
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
    # 29.10 (Р-6): реестр только дополняется (otbor_statistika п. 4) — после e8d00284/f149d752 в нём строки-указатели
    # на файлы проходов (status=ukazatel); тест требовал len == 1 (пустой реестр) и закреплял состояние до них.
    kol = R.kolonki("train4")
    assert next(csv.reader(rows[:1])) == kol
    assert all(r[kol.index("status")] == "ukazatel" for r in csv.reader(rows[1:]))


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


def sintet_ledger(tmp_path, n=6, sk="S1", it=3):
    """Синтетический ledger В1: {(pkey, 'train4'): (params, res)} + SHA256SUMS.txt."""
    cache = {((("z_entry", i),), "train4"): ({"z_entry": i}, {
        "by_year": {y: 0.1 * i + (y - 2016) for y in (2016, 2017, 2018, 2019)},
        "sdelok_y": {y: 10 + i for y in (2016, 2017, 2018, 2019)}, "vyrozhdena": i == 0}) for i in range(n)}
    p = tmp_path / sk / f"ledger_it{it}.pkl"
    p.parent.mkdir(parents=True)
    p.write_bytes(pickle.dumps(cache, protocol=4))
    (tmp_path / "SHA256SUMS.txt").write_text(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {sk}/ledger_it{it}.pkl  N={n}\n")
    return cache


def test_iz_ledger_chislo_strok_ravno_ledger(tmp_path):
    sintet_ledger(tmp_path, 6)
    out = tmp_path / "r.csv"
    assert R.iz_ledger("train4", 3, "S1", tmp_path, out) == 6
    rows = R.chitat(str(out))
    assert len(rows) == 6 and rows[2]["sharpe_2016"] == repr(0.2) and rows[0]["vyrozhdena"] == "1"
    assert rows[1]["sharpe_train4"] == repr(0.1) and rows[1]["sdelok_2019"] == "11.0"


def test_iz_ledger_povtor_ne_dublirit(tmp_path):
    sintet_ledger(tmp_path)
    out = tmp_path / "r.csv"
    R.iz_ledger("train4", 3, "S1", tmp_path, out)
    pered = out.read_bytes()
    assert R.iz_ledger("train4", 3, "S1", tmp_path, out) == 0 and out.read_bytes() == pered


def test_povtor_it3_it4_ne_zadvaivaet_kandidatov(tmp_path):
    sintet_ledger(tmp_path, 6, it=3)
    sintet_ledger_kopiya = tmp_path / "S1" / "ledger_it4.pkl"
    sintet_ledger_kopiya.write_bytes((tmp_path / "S1" / "ledger_it3.pkl").read_bytes())
    with open(tmp_path / "SHA256SUMS.txt", "a") as f:
        f.write(f"{hashlib.sha256(sintet_ledger_kopiya.read_bytes()).hexdigest()}  S1/ledger_it4.pkl  N=6\n")
    out = tmp_path / "r.csv"
    assert R.iz_ledger("train4", 3, "S1", tmp_path, out) == 6
    assert R.iz_ledger("train4", 4, "S1", tmp_path, out) == 6   # повтор записан под другим источником
    rows = R.chitat(str(out))
    assert {r["istochnik"] for r in rows} == {"zh051_v1_it3", "zh051_v1_it4"} and len(rows) == 12
    assert R.chislo_konfiguracij(rows) == 6                      # мутация len(rows) дала бы 12


def test_mutaciya_propusk_stroki_ledger_lovitsya(tmp_path, monkeypatch):
    sintet_ledger(tmp_path)
    orig = R.dopisat
    # мутация: последняя строка ledger теряется при переносе — ловится числом строк
    monkeypatch.setattr(R, "dopisat", lambda rows, p, n, path: orig(rows[:-1], p, n, path))
    with pytest.raises(ValueError, match="пропущен"):
        R.iz_ledger("train4", 3, "S1", tmp_path, tmp_path / "r.csv")
    assert not (tmp_path / "r.csv").exists()


def test_iz_ledger_sha_ne_sovpal_otkaz(tmp_path):
    sintet_ledger(tmp_path)
    (tmp_path / "S1" / "ledger_it3.pkl").write_bytes(b"x")
    with pytest.raises(ValueError, match="sha256"):
        R.iz_ledger("train4", 3, "S1", tmp_path, tmp_path / "r.csv")


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
