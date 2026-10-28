"""Ж-118: класс «а» копируется в Z2 (не переносится); синтетика во временном корне.

Мутации: перенос вместо копии (исходник исчез) и путь копии с test_locked -> тесты падают.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "dannye"))
import suzhenie_obshee as ob  # noqa: E402
import suzhenie_rezka as rz  # noqa: E402
import suzhenie_test_locked as st  # noqa: E402

Z2, Z3, ISH = "data/regen/test1_2022_2024", "data/regen/hold25", "data/regen/ish"
A = "data/regen/test_locked/final3/old_test.parquet"
V = "data/regen/test_locked/final3/mixed_test.parquet"
KOPIYA = f"{Z2}/regen/final3/old__a.parquet"
D = ["--z2-dir", Z2, "--z3-dir", Z3, "--ishodnye-dir", ISH, "--vkhod", "data"]


def zapisat(koren, rel, start, end):
    p = Path(koren) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    idx = pd.bdate_range(start, end)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame({"date": idx, "val": 1.0}), preserve_index=False), p)


@pytest.fixture
def koren(tmp_path):
    zapisat(tmp_path, A, "2022-03-01", "2024-06-28")
    zapisat(tmp_path, V, "2024-06-01", "2025-02-28")
    return tmp_path


def zapusk(koren, *extra):
    return st.main(["--kopirovat-a", "--koren", str(koren), *D, *extra])


def test_kopiya_po_pravilnomu_puti(koren, capsys):
    sha = ob.sha_file(koren / A)
    assert zapusk(koren, "--primenit") == 0
    assert "ITOG_A fajlov=1 skopirovano=1 fail=0" in capsys.readouterr().out
    assert ob.sha_file(koren / KOPIYA) == sha
    assert (koren / A).exists() and ob.sha_file(koren / A) == sha
    assert not (koren / Z2 / "regen/final3/mixed__a.parquet").exists()  # 2025 — не класс «а»


def test_dry_run_nichego_ne_pishet(koren, capsys):
    assert zapusk(koren) == 0
    assert "ITOG_A fajlov=1 skopirovano=0 fail=0" in capsys.readouterr().out
    assert not (koren / KOPIYA).exists()


def test_povtor_otkazyvaet(koren, capsys):
    assert zapusk(koren, "--primenit") == 0
    sled = (koren / KOPIYA).stat().st_mtime_ns
    assert zapusk(koren, "--primenit") == 1
    assert "fail=1" in capsys.readouterr().out
    assert (koren / KOPIYA).stat().st_mtime_ns == sled


def test_mutatsiya_perenos_vmesto_kopii_lovitsya(koren, monkeypatch):
    monkeypatch.setattr(st.shutil, "copy2", lambda s, d: shutil.move(str(s), str(d)))
    zapusk(koren, "--primenit")
    with pytest.raises(AssertionError):
        assert (koren / A).exists(), "исходник исчез: перенос вместо копии"


def test_mutatsiya_put_s_test_locked_lovitsya(koren, monkeypatch, capsys):
    monkeypatch.setattr(rz, "put_kopii_a", lambda rel: Path("test_locked") / Path(rel).name)
    assert zapusk(koren, "--primenit") == 1
    assert "под замком" in capsys.readouterr().out
    assert not (koren / Z2 / "test_locked").exists()


def test_manifest_posle_uchityvaet_kopiyu(koren, capsys):
    do = "docs/do.tsv"
    assert st.main(["--manifest-do", "--primenit", "--out", do, "--koren", str(koren), *D]) == 0
    assert zapusk(koren, "--primenit") == 0
    capsys.readouterr()
    args = ["--manifest-posle", "--do-tsv", str(koren / do), "--koren", str(koren), *D]
    assert st.main(args) == 0
    (koren / KOPIYA).write_bytes(b"x")  # копия не совпала по sha -> FAIL
    assert st.main(args) == 1
