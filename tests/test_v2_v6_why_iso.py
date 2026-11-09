"""V6 лист Z3 (Ж-230), G65: изоляция — нет обучателей в модулях v6_why_*; `--check` ничего не пишет (мутация «пишет» ловится); на копиях выходов `--check` проходит."""
import shutil

from models_v2.s13 import run_v6_why as R, v6_why_check as CK, v6_why_data as DT, v6_why_stage as ST


def test_net_obuchatelej_v_modulyah():
    assert CK.no_trainers() == []


def test_mutatsiya_import_obuchatelya_lovitsya():
    for src in ("import lightgbm", "import xgboost as xgb", "from xgboost import XGBRegressor", "from lightgbm import LGBMRegressor", "from models_v2.s13 import v6_1_fit", "from models_v2.s13.v6_1_fit import fit", "import sklearn.ensemble"):
        assert CK.imports_trainer(src, "x.py"), src
    assert CK.imports_trainer("import numpy as np\nfrom models_v2.s13 import v6_0_q as Q", "x.py") == []


def copies(tmp_path, monkeypatch):
    w = tmp_path / "why"
    shutil.copytree(DT.WHY, w)
    doc = tmp_path / DT.DOC.name
    shutil.copy(DT.DOC, doc)
    monkeypatch.setattr(DT, "WHY", w)
    monkeypatch.setattr(DT, "CP_FILE", w / "CHECKPOINT_WHY.json")
    monkeypatch.setattr(DT, "DOC", doc)
    return w


def test_check_na_kopiyah_prohodit_i_nichego_ne_pishet(tmp_path, monkeypatch, capsys):
    copies(tmp_path, monkeypatch)
    before = ST.own_digest()
    assert R.main(True) == 0
    assert ST.own_digest() == before
    assert "ИТОГ: OK" in capsys.readouterr().out


def test_mutatsiya_check_pishet_fajl_lovitsya(tmp_path, monkeypatch, capsys):
    w = copies(tmp_path, monkeypatch)
    orig = CK.all_bad

    def writing(r):
        (w / "WHY_YEARS.csv").write_text((w / "WHY_YEARS.csv").read_text() + "\n")
        return orig(r)
    monkeypatch.setattr(CK, "all_bad", writing)
    assert R.main(True) == 1
    assert "изменил собственные файлы" in capsys.readouterr().out


def test_mutatsiya_tablica_na_diske_izmenena_lovitsya(tmp_path, monkeypatch):
    w = copies(tmp_path, monkeypatch)
    p = w / "WHY_GSIZE.csv"
    lines = p.read_text().splitlines()
    lines[3] = lines[3].replace(",", ",9", 1) if "," in lines[3] else lines[3]
    p.write_text("\n".join(lines) + "\n")
    assert R.main(True) == 1
