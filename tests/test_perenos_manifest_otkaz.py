"""Перенос на сервер (Ж-133): манифест с датой > 2021-12-31, test_locked и подменённым файлом отвергается целиком."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

_p = Path(__file__).resolve().parents[1] / "scripts/server/manifest_proverka.py"
_s = importlib.util.spec_from_file_location("manifest_proverka", _p)
mp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(mp)


def _mf(tmp_path, put="data/a.csv", data=b"date\n2021-12-30\n", md="2021-12-30"):
    f = tmp_path / "src" / "data/a.csv"   # файл всегда внутри tmp; путь в манифесте может быть запрещённым
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(data)
    m = tmp_path / "m.tsv"
    m.write_text("\t".join(mp.SHAPKA) + f"\n{put}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\t{md}\tx\n")
    return m, tmp_path / "src"


def _run(m, src, tmp_path):
    return mp.main(["--manifest", str(m), "--koren", str(tmp_path / "out"), "--istochnik", str(src), "--strogo"])


def test_chistyj_prohodit(tmp_path):
    m, src = _mf(tmp_path)
    assert _run(m, src, tmp_path) == 0
    assert (tmp_path / "out/data/a.csv").exists()


def test_fajl_s_datoj_2022_otvergnut_do_kopirovaniya(tmp_path, capsys):
    m, src = _mf(tmp_path, data=b"date\n2022-01-03\n", md="2022-01-03")
    assert _run(m, src, tmp_path) == 1
    assert "FAIL manifest" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("put", ["test_locked/a.csv", "data/regen/test1_2022_2024/a.csv", "../a.csv", "/etc/a"])
def test_zapreshchennyj_put(tmp_path, put):
    m, src = _mf(tmp_path, put=put)
    assert _run(m, src, tmp_path) == 1


def test_podmenennyj_fajl_otvergnut(tmp_path):
    m, src = _mf(tmp_path)
    (src / "data/a.csv").write_bytes(b"date\n2021-12-31\n")  # тот же размер, другой sha256
    assert _run(m, src, tmp_path) == 1


def test_bez_dat_prinimaetsya_tolko_bukvalno(tmp_path):
    m, src = _mf(tmp_path, md="без дат")
    assert _run(m, src, tmp_path) == 0
    m, src = _mf(tmp_path / "x", md="2022 без дат")
    assert _run(m, src, tmp_path / "x") == 1


def test_sha256_otmetka_ryadom_ne_lishnyaya_a_chuzhaya_lishnyaya(tmp_path):
    m, src = _mf(tmp_path)
    assert _run(m, src, tmp_path) == 0
    (tmp_path / "out/data/a.csv.sha256").write_text("x")       # отметка daily_1850.ensure рядом с файлом манифеста
    assert _run(m, src, tmp_path) == 0
    (tmp_path / "out/data/b.csv.sha256").write_text("x")       # отметка файла, которого в манифесте нет
    assert _run(m, src, tmp_path) == 1


def test_lishnij_fajl_v_koren(tmp_path):
    m, src = _mf(tmp_path)
    assert _run(m, src, tmp_path) == 0
    (tmp_path / "out/data/lishnij.csv").write_text("x")
    assert _run(m, src, tmp_path) == 1
