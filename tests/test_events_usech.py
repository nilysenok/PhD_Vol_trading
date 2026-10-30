"""data/events.csv в пакете <= 2021: усечённая копия, а не прямой файл (после слияния данных wf в файле строки 2022-2026)."""
import hashlib
import importlib.util
from pathlib import Path

_R = Path(__file__).resolve().parent.parent
_s = importlib.util.spec_from_file_location("usech_2021", _R / "scripts" / "server" / "usech_2021.py")
U = importlib.util.module_from_spec(_s)
_s.loader.exec_module(U)

SHA_PREZHNIJ = "3e20e0f9ea5b49052f4c0dd9d5a99d8b83dfae299dba7a87b8bbe68c5cdcf189"
DATY = ("data_obyavleniya", "data_sobytiya")


def _pravilo():
    return next(p for p in U.PRAVILA if p[0] == "data/events.csv")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_pravilo_obe_daty():
    assert _pravilo()[1:3] == ("csv", DATY)


def test_kopiya_pobajtno_ravna_prezhnemu_fajlu(tmp_path):
    U.usech_csv(_R / "data" / "events.csv", tmp_path / "e.csv", _pravilo()[2], _pravilo()[3])
    assert (tmp_path / "e.csv").stat().st_size == 52202 and _sha(tmp_path / "e.csv") == SHA_PREZHNIJ


def test_obyavleno_v_2022_otbrosheno(tmp_path):
    src = tmp_path / "e.csv"
    src.write_text("ticker,tip,data_obyavleniya,data_sobytiya,istochnik\n"
                   "A,ok,2021-11-01,2021-12-01,u\n"
                   "B,pozzhe,2022-01-10,2021-12-30,u\n"          # объявлено в 2022 - будущее
                   "C,budushchee,2021-12-01,2022-02-01,u\n", encoding="utf-8")
    st = U.usech_csv(src, tmp_path / "o.csv", _pravilo()[2], _pravilo()[3])
    assert (tmp_path / "o.csv").read_text().splitlines()[1:] == ["A,ok,2021-11-01,2021-12-01,u"] and st.otbrosheno == 2


def test_manifest_do_2021_bez_pryamogo_events():
    for ln in (_R / "docs/server/manifest_dannyh_2021.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        c = ln.split("\t")
        assert c[0] != "data/events.csv"
        if c[5] == "data/events.csv":
            assert c[0] == "data/server_2021/data/events.csv" and c[1] == SHA_PREZHNIJ


def test_mutatsiya_usechenie_po_odnoj_date_lovitsya(tmp_path):
    src = tmp_path / "e.csv"
    src.write_text("ticker,tip,data_obyavleniya,data_sobytiya,istochnik\n"
                   "B,pozzhe,2022-01-10,2021-12-30,u\n"
                   "C,budushchee,2021-12-01,2022-02-01,u\n", encoding="utf-8")
    for odna in (DATY[:1], DATY[1:]):                              # мутация: усечение по одной дате пропускает строку
        U.usech_csv(src, tmp_path / "m.csv", odna, ())
        assert len((tmp_path / "m.csv").read_text().splitlines()) == 2
    U.usech_csv(src, tmp_path / "o.csv", _pravilo()[2], ())
    assert len((tmp_path / "o.csv").read_text().splitlines()) == 1
