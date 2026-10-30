"""Тестировщик #788: пробелы сверки kod_sha/dannye_sha (мутации а–г автора-проверки пойманы существующими тестами)."""
import copy
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1] / "scripts/server"
sys.path.insert(0, str(SERVER))
import ekviv_sravn as es  # noqa: E402
from test_sverka_bez_vhody import META, _sr, _zapis  # noqa: E402


def test_etalon_s_sha_server_bez_sha_fail():
    """Эталон хранит sha, а сервер не передал свой (None) -> расхождение, не молчаливый пропуск."""
    m = copy.deepcopy(META)
    m["kod_sha"] = m["dannye_sha"] = None
    r = _sr({"yakor__S1": _zapis()}, m)
    assert r["kod"] == 1 and any("sha" in s for s in r["diskretnoe"])


def test_staryj_etalon_bez_sha_ne_sveryaetsya():
    """Эталон без sha (собран без флагов) -> sha не сверяются (документированное поведение, SHA_VHODOV)."""
    eta_meta = {**META, "kod_sha": None, "dannye_sha": None}
    r = es.sravnit({"yakor__S1": _zapis()}, eta_meta, {"yakor__S1": _zapis()}, META, 1e-12)
    assert r["kod"] == 0


def test_sobrat_perenos_peredayot_sha_v_etalony():
    src = (SERVER / "sobrat_perenos.sh").read_text()
    stroka = src[src.index("etalony_mak.py"):].split("\n\n")[0]
    assert "--kod-sha" in stroka and "--dannye-sha" in stroka
