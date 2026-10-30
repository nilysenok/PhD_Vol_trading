"""Список переноса wf (лист wfП-2, Д-a/Д-d): CNYRUB currency_10m, маркер дневки, составы 2022–2026 входят в манифест wf;
записи > 2026-02-28 не входят. Данные — синтетика. Мутация «строки нет» (из списка убрана строка) -> тест падает."""
import importlib.util
import datetime as dt
from pathlib import Path

import pandas as pd

SRV = Path(__file__).resolve().parents[1] / "scripts/server"


def _load(name):
    sys_path = str(SRV)
    import sys
    sys.path.insert(0, sys_path)
    s = importlib.util.spec_from_file_location(name, SRV / f"{name}.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


MP, U = _load("manifest_perenos"), _load("usech_2021")
SPISOK = SRV / "perenos_spisok.tsv"
DO = pd.Timestamp("2026-02-28")
CNY = "moex_discovery/data/external/moex_iss/currency_10m/CNYRUB_TOM.parquet"
MARKER = "data/fastbt/daily/ohlcv_daily_1850.parquet.sha256"
IMOEX = "docs/issledovaniya/2026-10-30_sostav_imoex_2022_2026.csv"
SEKT = "docs/issledovaniya/2026-10-30_sostav_sektorov_2022_2026.csv"


def _koren(tmp: Path, cny_do="2026-02-27", sostav_do="2026-01-19") -> Path:
    (tmp / CNY).parent.mkdir(parents=True)
    pd.DataFrame({"begin": pd.to_datetime(["2024-06-13 10:00", cny_do + " 18:50"]), "close": [1.0, 2.0]}).to_parquet(tmp / CNY)
    (tmp / MARKER).parent.mkdir(parents=True)
    (tmp / MARKER).write_text("0" * 64 + "\n")
    (tmp / IMOEX).parent.mkdir(parents=True)
    (tmp / IMOEX).write_text(f"date_eff,known_from,n,tickers\n2022-03-21,2022-03-04,1,SBER\n{sostav_do},{sostav_do},1,SBER\n")
    (tmp / SEKT).write_text(f"тикер,дата_ребалансировки,индекс\nAFLT,2022-06-20,MOEXTN\nSBER,{sostav_do},MOEXFN\n")
    return tmp


def _prinyato(tmp: Path, spisok=None) -> set[str]:
    zap = MP.razobrat(spisok or SPISOK)
    return {x[0] for x in MP.sobrat(str(tmp), zap, DO)["prinyato"]}


def test_kazhdaya_stroka_spiska_s_tabulyatsiej():
    """Шаблон и читатель разделены табуляцией (слипшаяся строка = шаблон не находит файлов, молча)."""
    for s in SPISOK.read_text(encoding="utf-8").splitlines():
        if s.strip() and not s.startswith("#"):
            assert "\t" in s and not s.split("\t")[0].endswith(" "), s[:60]
    assert "data/iss_ext/bars_10m/BR*.parquet" in {x[0] for x in MP.razobrat(SPISOK)}


def test_g1_cny_i_marker_v_manifeste_wf(tmp_path):
    p = _prinyato(_koren(tmp_path))
    assert CNY in p and MARKER in p


def test_g1_sostavy_2022_2026_kopiyami_v_manifeste_wf(tmp_path, monkeypatch):
    k = _koren(tmp_path)
    monkeypatch.setattr(U, "CUT", dt.date(2026, 2, 28))
    vyhod = k / "data/server_wf"
    U.usech(k, vyhod)
    kop = {x[0] for x in MP.sobrat(str(vyhod), [("docs/issledovaniya/sostav_*_2022_2026.csv", "t", "")], DO)["prinyato"]}
    assert kop == {"docs/issledovaniya/sostav_imoex_2022_2026.csv", "docs/issledovaniya/sostav_sektorov_2022_2026.csv"}


def test_mutatsiya_stroki_net_lovitsya(tmp_path):
    k = _koren(tmp_path)
    bez = tmp_path / "bez.tsv"
    bez.write_text("".join(s for s in SPISOK.read_text(encoding="utf-8").splitlines(True) if "CNYRUB_TOM" not in s and ".sha256" not in s))
    p = _prinyato(k, bez)
    assert CNY not in p and MARKER not in p                       # на этом тест G1 выше упал бы


def test_g4_zapis_posle_2026_02_28_ne_popadaet(tmp_path, monkeypatch):
    k = _koren(tmp_path, cny_do="2026-03-02", sostav_do="2026-03-02")
    assert CNY not in _prinyato(k)                                # файл с датой > границы отвергнут целиком
    monkeypatch.setattr(U, "CUT", dt.date(2026, 2, 28))
    vyhod = k / "data/server_wf"
    U.usech(k, vyhod)
    for f in vyhod.rglob("sostav_*_2022_2026.csv"):
        assert "2026-03-02" not in f.read_text(encoding="utf-8")
    cny = vyhod / CNY
    assert not cny.exists() or pd.read_parquet(cny)["begin"].max() < pd.Timestamp("2026-03-01")
