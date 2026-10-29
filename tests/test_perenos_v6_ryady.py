"""Ряды оценщика v6 в манифесте переноса (Ж-131 п. 2): Brent, USDRUB, vpred_aligned, br_contracts.json.
Файл с датой 2022 отвергается, усечённая копия не содержит 2022; подложенное правило/файл ловится."""
import fnmatch
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

_S = Path(__file__).resolve().parent.parent / "scripts" / "server"


def _mod(name):
    sp = importlib.util.spec_from_file_location(name, _S / f"{name}.py")
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


MP, U, JD = _mod("manifest_perenos"), _mod("usech_2021"), _mod("json_daty")
BR, USD, VP, JS = ("data/iss_ext/bars_10m", "moex_discovery/data/external/moex_iss/currency_10m/USD000UTSTOM.parquet",
                   "moex_volatility/results/final/strategies/data/vpred_aligned.parquet", "data/iss_ext/br_contracts.json")
# что открывает istochniki_v6.zagruzit + EvaluatorV6 (трасса Морриса S1–S6, train4 и v2, 29.10): шаблон → пример пути
TRASSA = {"data/iss_ext/bars_10m/BR*.parquet": f"{BR}/BRZ1.parquet", "data/iss_ext/br_contracts.json": JS,
          "moex_discovery/data/external/moex_iss/currency_10m/USD000UTSTOM.parquet": USD,
          "moex_volatility/results/final/strategies/data/vpred_aligned.parquet": VP,
          "moex_discovery/data/final_v6/candles_10m/external/*.parquet": "moex_discovery/data/final_v6/candles_10m/external/IMOEX.parquet"}


def _pq(p: Path, kol: str, daty, strok=False):
    p.parent.mkdir(parents=True, exist_ok=True)
    d = pd.to_datetime(pd.Series(daty))
    pd.DataFrame({kol: d.dt.strftime("%Y-%m-%d %H:%M:%S") if strok else d, "close": 1.0}).to_parquet(p)


def _koren(r: Path, s2022=True) -> Path:
    _pq(r / BR / "BRZ1.parquet", "begin", ["2021-12-01 10:00", "2021-12-01 10:10"])        # весь ≤ 2021
    _pq(r / BR / "BRF2.parquet", "begin", ["2021-12-20 10:00", "2022-01-10 10:00"] if s2022 else ["2021-12-20 10:00"])
    _pq(r / USD, "begin", ["2021-12-30 18:40:00", "2022-01-03 10:00:00"], strok=True)
    _pq(r / VP, "date", ["2021-12-30", "2022-01-03"])
    (r / JS).parent.mkdir(parents=True, exist_ok=True)
    (r / JS).write_text(json.dumps([
        {"month": "2021-12", "secid": "BRZ1", "frsttrade": "2021-02-01", "lsttrade": "2021-12-15"},
        {"month": "2022-01", "secid": "BRF2", "frsttrade": "2021-03-01", "lsttrade": "2022-01-14"}]))
    return r


def _zapisi():
    return MP.razobrat(_S / "perenos_spisok.tsv")


def test_trassa_pokryta_pravilami_i_spiskom():
    shabl = [p[0] for p in U.PRAVILA]
    spis = {z[0] for z in _zapisi()}
    for sh, primer in TRASSA.items():
        assert any(fnmatch.fnmatch(primer, s) for s in shabl), f"{primer}: нет правила усечения"
        assert sh in spis, f"{sh}: нет в perenos_spisok.tsv"


def test_originaly_s_2022_otvergayutsya_kopii_prinimayutsya(tmp_path):
    k = _koren(tmp_path / "m")
    r = MP.sobrat(str(k), _zapisi())
    ok, net = {x[0] for x in r["prinyato"]}, {x[0] for x in r["otvergnuto"]}
    assert f"{BR}/BRZ1.parquet" in ok and {f"{BR}/BRF2.parquet", USD, VP, JS} <= net      # 2022 в содержимом → отказ
    v = tmp_path / "v" / "data" / "server_2021"
    U.usech(k, v)
    r2 = MP.sobrat(str(v.parent.parent), [("data/server_2021/**/*", "t", "")])
    assert r2["otvergnuto"] == [] and len(r2["prinyato"]) >= 5
    assert max(x[3] for x in r2["prinyato"]) <= "2021-12-30"
    assert json.loads((v / JS).read_text())[0]["secid"] == "BRZ1" and len(json.loads((v / JS).read_text())) == 1


def test_original_do_2021_prinyat_kopii_net_dublya(tmp_path):
    """Файл целиком ≤ 2021 (BRZ1) принят оригиналом: копия того же пути дала бы «дубль» в manifest_proverka (упало на сборке пакета)."""
    k = _koren(tmp_path / "m")
    tsv = tmp_path / "m.tsv"
    r = MP.sobrat(str(k), _zapisi())
    MP.zapisat(r, tsv)
    v = k / "data" / "server_2021"
    U.obnovit_manifest(k, U.usech(k, v), v, tsv)
    puti = [ln.split("\t")[0] for ln in tsv.read_text().splitlines()[1:]]
    assert f"{BR}/BRZ1.parquet" in puti and f"data/server_2021/{BR}/BRZ1.parquet" not in puti
    assert f"data/server_2021/{BR}/BRF2.parquet" in puti and len(puti) == len(set(puti))
    assert len({p.removeprefix("data/server_2021/") for p in puti}) == len(puti)         # ни один путь не дважды


def test_json_zapis_s_datoj_2022_ne_popadaet(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"a": "2021-12-31", "b": "2021-12"}, {"a": "2021-11-01", "b": "2022-01-14"}, {"x": "текст"}]))
    assert JD.max_data(p) == pd.Timestamp("2022-01-14")
    assert JD.usech(p, tmp_path / "o.json") == (2, 1, "2021-12-31")        # запись с одной датой 2022 целиком отброшена
    assert JD.max_data(tmp_path / "o.json") == pd.Timestamp("2021-12-31")
    p.write_text(json.dumps([{"x": "текст"}]))
    assert JD.max_data(p) == "NODATES"


def test_json_ne_spisok_otvergaetsya(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"a": "2021-01-01"}')
    st, tx = MP.proverit(str(p), "")
    assert st == "net" and "не прочитан" in tx


def test_mutaciya_bez_pravila_kopii_net(tmp_path, monkeypatch):
    """Подложенная ошибка: правила для рядов v6 убраны → файла в усечённом пакете нет (тест на покрытие выше упал бы)."""
    k = _koren(tmp_path / "m")
    monkeypatch.setattr(U, "PRAVILA", [p for p in U.PRAVILA if "iss_ext" not in p[0] and "vpred" not in p[0]])
    v = tmp_path / "v" / "data" / "server_2021"
    U.usech(k, v)
    assert not (v / f"{BR}/BRF2.parquet").exists() and not (v / VP).exists()


def test_mutaciya_polovina_dat_v_zapisi_ne_pomogaet(tmp_path, monkeypatch):
    """Подложенная ошибка: граница усечения JSON сдвинута за 2022 → дата 2022 в копии, манифест её отвергает."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"frsttrade": "2021-03-01", "lsttrade": "2022-01-14"}]))
    monkeypatch.setattr(JD, "GRANICA", pd.Timestamp("2099-01-01"))
    JD.usech(p, tmp_path / "o.json")
    st, tx = MP.proverit(str(tmp_path / "o.json"), "")
    assert st == "net" and "2022-01-14" in tx


@pytest.mark.parametrize("imya", ["BRH2_2022.parquet", "BRF8_2024.parquet"])
def test_imya_s_godom_2022_ne_chitaetsya(tmp_path, imya):
    _pq(tmp_path / imya, "begin", ["2021-01-01 10:00"])
    st, tx = MP.proverit(str(tmp_path / imya), "")
    assert st == "net" and "202x" in tx
