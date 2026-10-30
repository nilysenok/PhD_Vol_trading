"""Пакет переноса wf (лист W1, Ж-136): манифест данных до 2026-02-28, отметка снятого замка, усечение календарей по --do.
Синтетика во временном каталоге + проверка закоммиченного docs/server/manifest_dannyh_wf.tsv (содержимое данных не читается).
Мутации: манифест wf собирается без проверки отметки (тест отказа падает); усечение не по --do, а по 2021 (тест wf падает);
календарь с known_from после границы попал в копию (тест PIT падает)."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SRV = REPO / "scripts/server"
DO = "2026-02-28"


def _load(name):
    import sys
    sys.path.insert(0, str(SRV))
    s = importlib.util.spec_from_file_location(name + "_wf", SRV / f"{name}.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


MP, MPR, U = _load("manifest_perenos"), _load("manifest_proverka"), _load("usech_2021")


def _tsv(imya: str) -> list[list[str]]:
    return [ln.split("\t") for ln in (REPO / "docs/server" / imya).read_text(encoding="utf-8").splitlines()[1:]]


NET_MANIFESTA = not (REPO / "docs/server/manifest_dannyh_wf.tsv").exists()     # манифест wf пересобирает дирижёр после докачки данных
pytestmark_m = pytest.mark.skipif(NET_MANIFESTA, reason="docs/server/manifest_dannyh_wf.tsv ещё не закоммичен (пересборка после докачки)")


@pytestmark_m
def test_manifest_wf_ne_dalshe_2026_02_28_i_bez_zapreta():
    m = _tsv("manifest_dannyh_wf.tsv")
    assert len(m) > 300
    daty = [r[3] for r in m if r[3] != "без дат"]
    assert max(daty) <= DO and max(daty) >= "2026-02-01"                      # wf покрыт до конца периода, не дальше
    assert all("test_locked" not in r[0] and "test1_2022_2024" not in r[0] for r in m)
    assert MPR.prochitat(REPO / "docs/server/manifest_dannyh_wf.tsv", DO)       # проверяльщик пакета принимает


def test_manifest_wf_s_umolchaniem_2021_otvergaetsya(tmp_path):
    h = "0" * 64
    m = tmp_path / "m.tsv"
    m.write_text("\t".join(MPR.SHAPKA) + f"\ndata/a.csv\t{h}\t1\t2026-02-27\tx\n")
    with pytest.raises(MPR.Otkaz, match="> 2021-12-31"):
        MPR.prochitat(m)
    assert MPR.prochitat(m, DO)[0][3] == "2026-02-27"


def _otkaz(fn) -> bool:
    try:
        fn()
    except PermissionError:
        return True
    return False


def test_bez_otmetki_otkaz_s_otmetkoj_prinimaetsya(tmp_path):
    net, est = tmp_path / "net.json", tmp_path / "est.json"
    est.write_text(json.dumps({"preregistraciya_v2": "docs/preregistraciya_v2.md", "sha256": "a" * 64, "data_snyatiya": "2026-10-30"}))
    assert _otkaz(lambda: MP.verhnyaya_data(DO, net))
    assert MP.verhnyaya_data(DO, est) == pd.Timestamp(DO)
    assert MP.verhnyaya_data("2021-12-31", net) == pd.Timestamp("2021-12-31")      # умолчание отметки не требует


def test_mutaciya_manifest_wf_bez_proverki_otmetki_lovitsya(tmp_path, monkeypatch):
    net = tmp_path / "net.json"
    monkeypatch.setattr(MP, "verhnyaya_data", lambda do, otmetka=net: pd.Timestamp(do))      # мутант: проверка отметки снята
    assert not _otkaz(lambda: MP.verhnyaya_data(DO, net))                                    # мутант не отказывает...
    arg = ["--koren", str(tmp_path), "--spisok", str(tmp_path / "s.tsv"), "--vyhod", str(tmp_path / "o/m.tsv"), "--do", DO, "--otmetka", str(net)]
    (tmp_path / "s.tsv").write_text("*.csv\tt\n")
    (tmp_path / "a.csv").write_text("TRADEDATE,CLOSE\n2026-02-27,1\n")
    assert MP.main(arg) == 0                                                                 # ...и манифест собран без отметки: это и ловит
    monkeypatch.undo()
    arg[arg.index("--vyhod") + 1] = str(tmp_path / "o2/m.tsv")
    assert MP.main(arg) == 1 and not (tmp_path / "o2").exists()                              # настоящий код: отказ до записи


def _kal(p: Path):
    p.write_text("date,known_from,retrieved_at\n2026-02-13,2026-02-01,2026-10-30\n2026-03-05,2026-02-20,2026-10-30\n"
                 "2026-02-20,2026-03-10,2026-10-30\n2026-12-18,2026-12-01,2026-10-30\n")


def test_usech_do_wf_pit_obe_daty_menshe_granitsy(tmp_path, monkeypatch):
    monkeypatch.setattr(U, "CUT", pd.Timestamp(DO).date())
    src, dst = tmp_path / "cbr.csv", tmp_path / "o/cbr.csv"
    _kal(src)
    st = U.usech_csv(src, dst, U.KAL, ("retrieved_at",))
    rows = dst.read_text().splitlines()[1:]
    assert rows == ["2026-02-13,2026-02-01,"] and st.otbrosheno == 3                    # событие и объявление — обе <= границы


def test_mutaciya_usech_po_2021_vmesto_do_lovitsya(tmp_path, monkeypatch):
    src, dst = tmp_path / "cbr.csv", tmp_path / "o/cbr.csv"
    _kal(src)
    monkeypatch.setattr(U, "CUT", pd.Timestamp("2021-12-31").date())                    # мутант: граница не сдвинута на --do
    U.usech_csv(src, dst, U.KAL, ("retrieved_at",))
    assert dst.read_text().splitlines()[1:] == []                                       # wf-календарь пуст — тест выше это не пропустит


def test_usech_pq_i_json_po_do(tmp_path, monkeypatch):
    monkeypatch.setattr(U, "CUT", pd.Timestamp(DO).date())
    t = pd.DataFrame({"begin": pd.to_datetime(["2026-02-27 18:50", "2026-02-28 10:00", "2026-03-01 10:00"]), "v": [1, 2, 3]})
    t.to_parquet(tmp_path / "a.parquet")
    st = U.usech_pq(tmp_path / "a.parquet", tmp_path / "o.parquet", "begin")
    assert st.stroki == 2 and st.max_data.startswith("2026-02-28")                      # внутридневное время последнего дня входит
    (tmp_path / "c.json").write_text(json.dumps([{"lsttrade": "2026-02-02"}, {"lsttrade": "2026-05-04"}]))
    assert U.usech_json(tmp_path / "c.json", tmp_path / "oc.json").stroki == 1


def test_usech_main_bez_otmetki_otkaz(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(U, "CUT", U.CUT)
    arg = ["--koren", str(tmp_path), "--do", DO, "--otmetka", str(tmp_path / "net.json"), "--tsv", str(tmp_path / "m.tsv")]
    assert U.main(arg) == 1 and "нет отметки" in capsys.readouterr().out
    assert not (tmp_path / "data").exists()                                             # копии не созданы


def test_otmetka_v_repozitorii_nepusta():
    p = REPO / "docs/resheniya/wf_zamok_snyat.json"
    assert p.is_file() and p.stat().st_size > 0
