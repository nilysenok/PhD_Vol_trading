"""Манифест переноса (Ж-131 п. 2): max дата по содержимому; test_locked не читается; 2022+ отвергается."""
import builtins
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_P = Path(__file__).resolve().parent.parent / "scripts" / "server" / "manifest_perenos.py"
_s = importlib.util.spec_from_file_location("manifest_perenos", _P)
M = importlib.util.module_from_spec(_s)
_s.loader.exec_module(M)


def _pq(p: Path, daty, col="date"):
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({col: pd.to_datetime(daty), "x": range(len(daty))}).to_parquet(p)


def _run(root: Path, *zapisi):
    return M.sobrat(str(root), [(z[0], "t", z[1] if len(z) > 1 else "") for z in zapisi])


def test_prinyat_do_2021(tmp_path):
    _pq(tmp_path / "a.parquet", ["2019-01-03", "2021-12-31"])
    r = _run(tmp_path, ("a.parquet",))
    assert [x[3] for x in r["prinyato"]] == ["2021-12-31"] and not r["otvergnuto"]
    assert len(r["prinyato"][0][1]) == 64


def test_parquet_2022_otvergaetsya(tmp_path):
    _pq(tmp_path / "a.parquet", ["2021-12-30", "2022-01-03"])
    r = _run(tmp_path, ("a.parquet",))
    assert not r["prinyato"] and "2022-01-03" in r["otvergnuto"][0][1]


def test_csv_2022_otvergaetsya_a_2021_net(tmp_path):
    (tmp_path / "b.csv").write_text("TRADEDATE,CLOSE\n2021-12-30,1\n2022-01-03,2\n")
    (tmp_path / "c.csv").write_text("TRADEDATE,CLOSE\n2021-12-30,1\n2021-12-31,2\n")
    r = _run(tmp_path, ("*.csv",))
    assert [x[0] for x in r["prinyato"]] == ["c.csv"] and [x[0] for x in r["otvergnuto"]] == ["b.csv"]


def test_data_po_soderzhimomu_a_ne_po_imeni(tmp_path):
    _pq(tmp_path / "ryad_2019.parquet", ["2023-05-02"])
    r = _run(tmp_path, ("*.parquet",))
    assert not r["prinyato"] and r["otvergnuto"][0][0] == "ryad_2019.parquet"
    _pq(tmp_path / "nazvanie_bez_goda.parquet", ["2023-05-02"])
    assert len(_run(tmp_path, ("nazvanie_bez_goda.parquet",))["otvergnuto"]) == 1


def test_imya_s_2022_ne_chitaetsya(tmp_path, monkeypatch):
    _pq(tmp_path / "x_2022.parquet", ["2019-01-03"])
    def boom(*a, **k):
        raise AssertionError("прочитан")
    monkeypatch.setattr(M.pq, "ParquetFile", boom)
    r = _run(tmp_path, ("*.parquet",))
    assert "202x" in r["otvergnuto"][0][1]


def test_test_locked_ne_chitaetsya_vovse(tmp_path, monkeypatch):
    _pq(tmp_path / "test_locked" / "z.parquet", ["2019-01-03"])
    (tmp_path / "data" / "regen" / "test1_2022_2024").mkdir(parents=True)
    (tmp_path / "data" / "regen" / "test1_2022_2024" / "q.csv").write_text("date\n2019-01-01\n")
    real_open, bad = builtins.open, []

    def guard_open(f, *a, **k):
        if "test_locked" in str(f) or "test1_2022_2024" in str(f):
            bad.append(str(f))
            raise AssertionError("открыт запрещённый файл")
        return real_open(f, *a, **k)

    def guard_pq(f, *a, **k):
        bad.append(str(f))
        raise AssertionError("parquet запрещённого файла")

    monkeypatch.setattr(builtins, "open", guard_open)
    monkeypatch.setattr(M.pq, "ParquetFile", guard_pq)
    monkeypatch.setattr(M.pd, "read_csv", guard_pq)
    r = _run(tmp_path, ("**/*.parquet",), ("**/*.csv",))
    assert not bad and not r["prinyato"] and not r["otvergnuto"]
    assert {x[1] for x in r["isklyucheno"]} == {"test_locked", "data/regen/test1_2022_2024/"}


def test_bez_dat_tolko_s_obosnovaniem(tmp_path):
    (tmp_path / "n.csv").write_text("a,b\n1,2\n")
    assert _run(tmp_path, ("n.csv",))["otvergnuto"]
    r = _run(tmp_path, ("n.csv", "справочник без дат"))
    assert r["prinyato"][0][3] == "без дат"


def test_chislovaya_data_yyyymmdd(tmp_path):
    (tmp_path / "i.csv").write_text("date,v\n20211230,1\n20220103,2\n")
    assert _run(tmp_path, ("i.csv",))["otvergnuto"]


@pytest.mark.parametrize("d", ["2021-12-31 18:50:00"])
def test_granica_vklyuchitelna(tmp_path, d):
    (tmp_path / "g.csv").write_text(f"begin,v\n{d},1\n")
    assert _run(tmp_path, ("g.csv",))["prinyato"]
