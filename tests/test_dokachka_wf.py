"""Докачка хвоста 02.2026 (scripts/dannye/dokachka_wf.py): синтетический ответ ISS, в сеть не ходит, реальные данные не читаются.
Мутация № 36-класса: «не обрезает после 2026-02-28» — ловится test_mutaciya_bez_obrezki_lovitsya."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_sp = importlib.util.spec_from_file_location("dokachka_wf", Path(__file__).resolve().parents[1] / "scripts/dannye/dokachka_wf.py")
D = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(D)

COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _bars(ts):
    return pd.DataFrame({"timestamp": pd.to_datetime(ts), "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10}, columns=COLS)


def _fake_iss(rows_total):
    """Поддельный GET: отдаёт страницами по 500 строк из rows_total (список строк [begin, open, ...])."""
    def get(url):
        start = int(url.split("start=")[1].split("&")[0])
        return {"candles": {"columns": ["begin", "open", "high", "low", "close", "volume"], "data": rows_total[start:start + 500]}}
    return get


def test_stranicy_sobiraet_vse_stranicy_i_pustoj_otvet():
    rows = [[f"2026-02-0{1 + i // 300} 10:{i % 60:02d}:00", 1, 2, 0, 1, 5] for i in range(1234)]
    df = D.stranicy("x/", "T", 10, "2026-02-01", "2026-02-28", get=_fake_iss(rows))
    assert len(df) == 1234
    assert len(D.stranicy("x/", "T", 10, "2026-02-01", "2026-02-28", get=_fake_iss([]))) == 0


def test_dopisat_staroe_neizmenno_i_hvost_dobavlen():
    old = _bars(["2026-02-02 10:00", "2026-02-03 23:40"])
    new = _bars(["2026-02-03 23:40", "2026-02-04 10:00", "2026-02-27 10:00"])
    res, bad = D.dopisat(old, new, "timestamp")
    assert bad == 0
    assert res.iloc[:2].equals(old)
    assert list(res["timestamp"].iloc[2:]) == list(pd.to_datetime(["2026-02-04 10:00", "2026-02-27 10:00"]))


def test_oborvannyj_poslednij_den_zamenyaetsya():
    old = _bars(["2026-02-03 10:00", "2026-02-03 10:10"])
    new = _bars(["2026-02-03 10:00", "2026-02-03 10:10", "2026-02-04 10:00"])
    new.loc[1, "close"] = 9.9          # ISS отдал другое значение бара последнего дня (обрыв прошлой выгрузки)
    res, zam = D.dopisat(old, new, "timestamp")
    assert zam == 2 and res["close"].tolist() == [1.5, 9.9, 1.5]


def test_polnyj_poslednij_den_ne_trogaetsya():
    old = _bars(["2026-02-03 10:00", "2026-02-03 10:10"])
    new = _bars(["2026-02-03 10:00", "2026-02-03 10:10", "2026-02-04 10:00"])
    res, zam = D.dopisat(old, new, "timestamp")
    assert zam == 0 and res.iloc[:2].equals(old) and len(res) == 3


def test_propavshaya_stroka_poslednego_dnya_zamenyaetsya_a_pustoj_otvet_net():
    old = _bars(["2026-02-03 10:00", "2026-02-03 10:10"])
    assert D.dopisat(old, _bars(["2026-02-03 10:00", "2026-02-04 10:00"]), "timestamp")[1] == 2
    res, zam = D.dopisat(old, _bars([]), "timestamp")   # ISS ничего не вернул — старое остаётся
    assert zam == 0 and res.equals(old)


def test_ranshie_dni_nikogda_ne_zamenyayutsya():
    old = _bars(["2026-02-02 10:00", "2026-02-03 10:00"])
    new = _bars(["2026-02-02 10:00", "2026-02-03 10:00", "2026-02-04 10:00"])
    new.loc[0, "close"] = 7.7          # расходится предпоследний день — не трогаем
    res, zam = D.dopisat(old, new, "timestamp")
    assert zam == 0 and res.iloc[:2].equals(old)
    new.loc[1, "close"] = 8.8          # расходится последний — заменяем только его
    res, zam = D.dopisat(old, new, "timestamp")
    assert zam == 1 and res["close"].tolist() == [1.5, 8.8, 1.5]


def test_mutaciya_zamenyaet_i_predposlednij_den_lovitsya(monkeypatch):
    """Мутант: день для замены берётся на сутки раньше. test_ranshie_dni_... обязан на нём упасть."""
    old = _bars(["2026-02-02 10:00", "2026-02-03 10:00"])
    new = _bars(["2026-02-02 10:00", "2026-02-03 10:00", "2026-02-04 10:00"])
    new.loc[0, "close"] = 7.7
    src = Path(D.__file__).read_text(encoding="utf-8").replace("day = k_old.max().normalize()", "day = k_old.max().normalize() - pd.Timedelta(days=1)")
    ns = {"__name__": "mut"}
    exec(compile(src, "mut", "exec"), ns)
    res, zam = ns["dopisat"](old, new, "timestamp")
    assert zam != 0 and not res.iloc[:2].equals(old)   # мутант заменил бы предпоследний день; test_ranshie_dni_... это запрещает


def test_posle_28_fevralya_nichego_ne_pishetsya():
    old = _bars(["2026-02-03 10:00"])
    new = _bars(["2026-02-27 10:00", "2026-02-28 23:50", "2026-03-01 00:00", "2026-03-02 10:00"])
    res, _ = D.dopisat(old, new, "timestamp")
    assert res["timestamp"].max() == pd.Timestamp("2026-02-28 23:50")
    assert len(res) == 3


def test_mutaciya_bez_obrezki_lovitsya(monkeypatch):
    monkeypatch.setattr(D, "obrezat", lambda df, col, konec=D.KONEC: df)
    old = _bars(["2026-02-03 10:00"])
    new = _bars(["2026-02-27 10:00", "2026-03-02 10:00"])
    res, _ = D.dopisat(old, new, "timestamp")
    with pytest.raises(AssertionError):
        assert res["timestamp"].max() <= pd.Timestamp("2026-02-28 23:59:59")


def test_tipy_hvosta_kak_u_starogo():
    old = _bars(["2026-02-03 10:00"])
    new = pd.DataFrame({"timestamp": pd.to_datetime(["2026-02-04 10:00"]), "open": [1], "high": [2], "low": [0], "close": [1], "volume": [5.0]})
    res, _ = D.dopisat(old, new, "timestamp")
    assert dict(res.dtypes) == dict(old.dtypes)


def test_null_v_objeme_ne_zapolnyaetsya_nulem():
    """ISS отдал volume=null (CNYRUB_TOM с 2025-12): пропуск остаётся пропуском (Д4), старая часть не меняется."""
    old = _bars(["2026-02-03 10:00"])
    new = pd.DataFrame({"timestamp": pd.to_datetime(["2026-02-04 10:00"]), "open": [1.0], "high": [2.0], "low": [0.0],
                        "close": [1.0], "volume": [None]})
    res, _ = D.dopisat(old, new, "timestamp")
    assert pd.isna(res["volume"].iloc[1]) and res["volume"].iloc[0] == 10
    assert res["volume"].dtype == "Int64"


def test_zapis_csv_gz_ravna_csv(tmp_path):
    """Сжатый вариант раскрывается в те же байты, что и обычный CSV; sha256 — от несжатых байт."""
    import gzip
    import hashlib
    df = pd.DataFrame({"begin": ["2026-02-02 09:59:00"], "open": [1.5], "high": [2.0], "low": [1.0], "close": [1.75],
                       "volume": [10], "value": [17.5]})
    sha1 = D.zapis_csv(df, tmp_path / "a.csv", False)
    sha2 = D.zapis_csv(df, tmp_path / "b.csv", True)
    raw = (tmp_path / "a.csv").read_bytes()
    assert gzip.decompress((tmp_path / "b.csv.gz").read_bytes()) == raw
    assert sha1 == sha2 == hashlib.sha256(raw).hexdigest()
    assert raw.splitlines()[0] == b"begin,open,high,low,close,volume,value"


def test_obrezat_pustoj_vhod():
    assert len(D.obrezat(pd.DataFrame({"begin": []}), "begin")) == 0
