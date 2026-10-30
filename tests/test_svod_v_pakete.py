"""Свод Д8 в пакете переноса: код и данные лежат в разных каталогах (как собирает `sobrat_perenos.sh`), свод едет усечённой
копией манифеста и кладётся накладкой в корень КОДА (`docs/issledovaniya/`, как dividendy/msp). Без свода v6 отказывает на старте.
Мутации: свод вне манифеста; SVOD_CSV уехал от места накладки (корень кода без свода) — отказ вместо div_verified=True;
`trebovat_svod` ничего не делает — тест отказа падает."""
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MAN = REPO / "docs/server/manifest_dannyh_2021.tsv"
SVOD_REL = "docs/issledovaniya/div_sverka_svod.csv"
sys.path.insert(0, str(REPO / "scripts/server"))
import usech_2021  # noqa: E402


def _stroka():
    rows = [ln.split("\t") for ln in MAN.read_text(encoding="utf-8").splitlines()[1:] if ln.strip()]
    return [r for r in rows if len(r) == 7 and r[5] == SVOD_REL]


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_manifest_soderzhit_svod_i_sha_sovpadaet(tmp_path):
    r = _stroka()
    assert len(r) == 1, "свода Д8 нет в манифесте (или дубль)"
    put, sha, _, mx, _, ist, sha_ist = r[0]
    assert put == "data/server_2021/docs/issledovaniya/div_sverka_svod.csv" and mx <= "2021-12-31"
    cp = tmp_path / "svod.csv"
    st = usech_2021.usech_csv(REPO / SVOD_REL, cp, ("record_date", "announce_pit"))
    assert (st.stroki, st.otbrosheno) == (173, 74) and st.max_data <= "2021-12-31"   # свод 247 строк: 74 с датой 2022+ усечены
    assert _sha(cp) == sha == sha_ist, "усечённая копия ≠ строке манифеста"
    assert _sha(REPO / SVOD_REL) != sha, "полный свод шире усечённой копии"
    полный = (REPO / SVOD_REL).read_text(encoding="utf-8").splitlines()
    assert all(ln in полный for ln in cp.read_text(encoding="utf-8").splitlines())


def test_svod_v_nakladke_a_ne_v_data_syrye():
    nak = re.search(r"NAKLADKA='([^']+)'", (REPO / "scripts/server/sobrat_perenos.sh").read_text(encoding="utf-8")).group(1)
    assert re.match(nak, _stroka()[0][5]), "istochnik свода не попадает под накладку в корень кода"
    from fastbt import sobytiya_e1
    assert sobytiya_e1.SVOD_CSV == REPO / _stroka()[0][5], "SVOD_CSV ≠ istochnik манифеста"


def _paket(tmp: Path, s_nakladkoj: bool) -> Path:
    """kod/ — копия fastbt в одном каталоге, data/ — копия манифеста в другом; накладка — как в razvernut.sh."""
    kod, data = tmp / "kod", tmp / "data"             # на сервере файл манифеста лежит в data/ по istochnik (manifest_proverka)
    ist = _stroka()[0][5]
    shutil.copytree(REPO / "fastbt", kod / "fastbt", ignore=shutil.ignore_patterns("tests", "__pycache__", "data"))
    (data / ist).parent.mkdir(parents=True)
    shutil.copy(REPO / SVOD_REL, data / ist)
    if s_nakladkoj:
        (kod / ist).parent.mkdir(parents=True)
        shutil.copy(data / ist, kod / ist)
    return kod


def _zapusk(kod: Path, *argv: str):
    env = {k: v for k, v in os.environ.items() if k != "FASTBT_MAIN"} | {"PYTHONPATH": str(kod), "NUMBA_NUM_THREADS": "1"}
    return subprocess.run([sys.executable, *argv], cwd=kod, env=env, capture_output=True, text=True, timeout=300)


PROVERKA = ("from fastbt import istochniki_v6 as i, sobytiya_e1 as s; import pandas as pd\n"
            "f = i.Fajly(pd.Timestamp('2022-01-01'), main='.')\n"
            "print('SVOD', s.SVOD_CSV.parent.parent.parent.name == 'kod', f.div_otfiltrovan(), len(s.load_div_svod(s.SVOD_CSV)) if f.div_otfiltrovan() else 0)\n")


def test_s_nakladkoj_div_verified_istina(tmp_path):
    kod = _paket(tmp_path, True)
    (tmp_path / "p.py").write_text(PROVERKA, encoding="utf-8")
    r = _zapusk(kod, str(tmp_path / "p.py"))
    assert r.returncode == 0 and "SVOD True True 213" in r.stdout, r.stdout + r.stderr[-600:]


def test_bez_nakladki_svod_ne_vidno_i_div_verified_lozh(tmp_path):
    """Мутация «SVOD_CSV от корня кода, а свод только в data/»: корень кода без свода — div_verified=False."""
    kod = _paket(tmp_path, False)
    (tmp_path / "p.py").write_text(PROVERKA, encoding="utf-8")
    r = _zapusk(kod, str(tmp_path / "p.py"))
    assert r.returncode == 0 and "SVOD True False" in r.stdout, r.stdout + r.stderr[-600:]


@pytest.mark.parametrize("modul,argv", [("fastbt.progon_morris_v6", ["--kub", "v6", "--period", "train4"]),
                                        ("fastbt.progon_server", ["--kub", "v6", "--period", "train4"])])
def test_v6_bez_svoda_otkaz_na_starte(tmp_path, modul, argv):
    kod = _paket(tmp_path, False)
    r = _zapusk(kod, "-m", modul, *argv, "--out", str(tmp_path / "out"))
    assert r.returncode not in (0, 2) and "нет свода Д8" in (r.stdout + r.stderr), r.stdout[-300:] + r.stderr[-600:]
    assert not (tmp_path / "out").exists(), "отказ должен быть до создания каталога счёта"


def test_trebovat_svod_otkaz_i_prohod(tmp_path):
    from fastbt import sobytiya_e1
    with pytest.raises(SystemExit, match="нет свода Д8"):
        sobytiya_e1.trebovat_svod(tmp_path / "net.csv")
    sobytiya_e1.trebovat_svod(REPO / SVOD_REL)
