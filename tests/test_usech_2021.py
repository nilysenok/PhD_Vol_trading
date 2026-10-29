"""Усечение данных до 2021-12-31 (Ж-131 п. 2; Д3): 2022 не попадает в копию, фильтр на чтении, point-in-time."""
import importlib.util
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

_P = Path(__file__).resolve().parent.parent / "scripts" / "server" / "usech_2021.py"
_s = importlib.util.spec_from_file_location("usech_2021", _P)
U = importlib.util.module_from_spec(_s)
_s.loader.exec_module(U)


def _csv(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_csv_2022_ne_popadaet_granica(tmp_path):
    src = _csv(tmp_path / "a.csv", "date,v\n2021-12-30,1\n2021-12-31,2\n2022-01-01,3\n2023-05-05,4\n")
    st = U.usech_csv(src, tmp_path / "o" / "a.csv", ("date",))
    txt = (tmp_path / "o" / "a.csv").read_text()
    assert "2022" not in txt and "2023" not in txt and "2021-12-31,2" in txt
    assert (st.stroki, st.otbrosheno, st.max_data, st.ostanov) == (2, 2, "2021-12-31", False)
    st = U.usech_csv(src, tmp_path / "o2.csv", ("date",), ostanov=True)       # опция для отсортированных файлов
    assert (st.stroki, st.otbrosheno, st.ostanov) == (2, 1, True)


def test_csv_neotsortirovan_dochityvaetsya(tmp_path):
    src = _csv(tmp_path / "a.csv", "date,v\n2021-01-01,1\n2022-01-01,2\n2020-06-01,3\n2021-06-01,4\n")
    st = U.usech_csv(src, tmp_path / "o.csv", ("date",))
    assert st.stroki == 3 and not st.ostanov            # строка после «2022» не потеряна, остановки нет


def test_pit_data_obyavleniya_i_sobytiya(tmp_path):
    src = _csv(tmp_path / "c.csv",
               "date,event,retrieved_at,known_from\n"
               "2021-12-30,ok,2026-01-01,2021-11-01\n"
               "2021-12-30,obyavleno_pozzhe,2026-01-01,2022-01-05\n"       # событие <= 2021, объявлено в 2022 -> нет
               "2022-01-10,budushchee,2026-01-01,2021-11-01\n"             # объявлено <= 2021, событие 2022 -> нет
               "2021-10-01,bez_obyavleniya,2026-01-01,\n")                  # дата объявления неизвестна -> нет
    st = U.usech_csv(src, tmp_path / "o.csv", U.KAL, ("retrieved_at",))
    d = pd.read_csv(tmp_path / "o.csv", dtype=str, keep_default_na=False)
    assert list(d["event"]) == ["ok"] and st.otbrosheno == 3
    assert d["retrieved_at"].tolist() == [""]           # дата скачивания (2026) в копию не едет


def test_csv_komentarii_i_neparsimaya_data(tmp_path):
    src = _csv(tmp_path / "k.csv", "d,v\n2020-01-01,1\n# источник 2026-10-27\n")
    U.usech_csv(src, tmp_path / "o.csv", ("d",))
    assert "2026" not in (tmp_path / "o.csv").read_text()
    bad = _csv(tmp_path / "b.csv", "d,v\nne-data,1\n")
    with pytest.raises(ValueError, match="не разобрана"):
        U.usech_csv(bad, tmp_path / "o2.csv", ("d",))


def _pq(p: Path, ts):
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": pd.to_datetime(ts), "close": range(len(ts))}).to_parquet(p)
    return p


def test_parquet_vnutridnevnoe_31_dekabrya_vhodit_2022_net(tmp_path):
    src = _pq(tmp_path / "a.parquet", ["2021-12-30 10:00", "2021-12-31 23:50", "2022-01-01 00:00", "2024-03-01 10:00"])
    st = U.usech_pq(src, tmp_path / "o" / "a.parquet", "timestamp")
    t = pq.read_table(tmp_path / "o" / "a.parquet").to_pandas()
    assert t["timestamp"].max() < pd.Timestamp("2022-01-01") and len(t) == 2 and st.max_data == "2021-12-31"


def test_parquet_filtr_na_chtenii_polnyj_fajl_ne_gruzitsya(tmp_path, monkeypatch):
    src = _pq(tmp_path / "a.parquet", [f"2021-12-{d:02d} 10:00" for d in range(20, 32)] + [f"2022-01-{d:02d} 10:00" for d in range(1, 29)])
    def zapret(*a, **k):
        raise AssertionError("полное чтение файла")
    monkeypatch.setattr(U.pq, "read_table", zapret)
    monkeypatch.setattr(U.pq.ParquetFile, "read", zapret)
    monkeypatch.setattr(U.pq.ParquetFile, "read_row_groups", zapret)
    st = U.usech_pq(src, tmp_path / "o.parquet", "timestamp", paket=4, ostanov=True)
    assert st.ostanov and st.paketov == 4 and st.stroki == 12   # 40 строк, прочитано 4 пакета (16), остальные 24 не читались
    st = U.usech_pq(src, tmp_path / "o2.parquet", "timestamp", paket=4)          # по умолчанию: пакетами до конца, без read()
    assert not st.ostanov and st.paketov == 10 and st.stroki == 12


def test_parquet_neotsortirovan_ne_teryaet_2021_posle_2022(tmp_path):
    src = _pq(tmp_path / "a.parquet", ["2021-01-01", "2022-01-01", "2020-06-01", "2021-06-01"])
    st = U.usech_pq(src, tmp_path / "o.parquet", "timestamp", paket=2)
    assert st.stroki == 3 and not st.ostanov


def test_parquet_vse_2022_kopiya_ne_sozdayotsya(tmp_path):
    src = _pq(tmp_path / "a.parquet", ["2022-01-03", "2022-01-04"])
    U.usech_pq(src, tmp_path / "o.parquet", "timestamp")
    assert not (tmp_path / "o.parquet").exists()


def test_manifest_istochnik_i_perezapusk(tmp_path):
    _pq(tmp_path / "moex_discovery/data/raw/candles_10m/X.parquet", ["2021-12-29 10:00", "2022-02-01 10:00"])
    _csv(tmp_path / "data/corporate_actions.csv", "ticker,date,factor\nA,2020-01-01,2\nA,2023-01-01,3\n")
    tsv = tmp_path / "m.tsv"
    tsv.write_text("put\tsha256\tbajty\tmax_data\tchitatel\nold.csv\tabc\t5\t2020-01-01\tx\n", encoding="utf-8")
    (tmp_path / "m_otvergnuto.tsv").write_text(
        "put\tprichina\tchitatel\ndata/corporate_actions.csv\tmax_data 2023-01-01 > 2021-12-31\tr\n", encoding="utf-8")
    vyhod = tmp_path / "data" / "server_2021"
    for _ in range(2):                                   # повторный прогон не дублирует строки
        zap = U.usech(tmp_path, vyhod)
        itog = U.obnovit_manifest(tmp_path, zap, vyhod, tsv)
    rows = [ln.split("\t") for ln in tsv.read_text().splitlines()]
    assert len(rows) == 4 and rows[0][-2:] == ["istochnik", "sha256_istochnika"] and "2022" not in itog.split("max дата")[1][:6]
    kop = {r[0]: r for r in rows[1:]}
    r = kop["data/server_2021/data/corporate_actions.csv"]
    assert r[5] == "data/corporate_actions.csv" and r[3] <= "2021-12-31" and len(r[6]) == 64
    assert (tmp_path / "m_otvergnuto.tsv").read_text().count("заменён усечённой копией") == 1
