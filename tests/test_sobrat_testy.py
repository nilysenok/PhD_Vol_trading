"""Т-срв (Ж-160 п. 2): сборка кода для pytest на сервере. Сети нет; полный pytest не запускается (сухой режим)."""
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOBRAT = REPO / "scripts" / "server" / "sobrat_testy.sh"
TESTY = REPO / "scripts" / "server" / "perenos" / "testy.sh"
ISKL = REPO / "scripts" / "server" / "tar_iskluchit_2022plus.txt"
SUHOJ = ("tests/test_manifest_perenos.py", "tests/test_perenos_pamyat.py")


def _iskl():
    return {s for s in ISKL.read_text(encoding="utf-8").splitlines() if s and not s.startswith("#")}


def _sobrat(script, out, ref="HEAD"):
    env = dict(os.environ, TESTY_OUT=str(out), TESTY_REPO=str(REPO))
    return subprocess.run(["bash", str(script), ref], env=env, capture_output=True, text=True, timeout=300)


def _imena(tar):
    with tarfile.open(tar) as t:
        return [m.name.split("/", 1)[1] for m in t.getmembers() if m.isfile() and "/" in m.name]


def _plohie(tar):
    iskl = _iskl()
    return [n for n in _imena(tar) if n in iskl or "test_locked" in n or "test1_2022_2024" in n]


def test_g1_tar_malenkij_est_testy_net_2022plus(tmp_path):
    r = _sobrat(SOBRAT, tmp_path / "o")
    assert r.returncode == 0, r.stdout + r.stderr
    tars = list((tmp_path / "o").glob("testy_*.tar.gz"))
    assert len(tars) == 1
    assert tars[0].stat().st_size < 50 * 2**20
    imena = _imena(tars[0])
    assert any(n.startswith("tests/") for n in imena) and any(n.startswith("fastbt/") for n in imena)
    assert _plohie(tars[0]) == []
    sums = (tmp_path / "o" / "SHA256SUMS_testy").read_text()
    assert tars[0].name in sums and re.search(r"^[0-9a-f]{64}  ", sums)
    assert "rsync" in r.stdout and "$SRV" in r.stdout and "testy.sh" in r.stdout
    assert (tmp_path / "o" / "iskluchenno.txt").exists()


def test_g2_mutaciya_filtr_2022_snyat_lovitsya(tmp_path):
    # мутант: строка с меткой FILTR2022 заменена на «ничего не фильтровать»
    tekst = SOBRAT.read_text(encoding="utf-8")
    assert tekst.count("# FILTR2022") == 1
    mut = tmp_path / "sobrat_mut.sh"
    mut.write_text("\n".join('cp "$TMP/vse.txt" "$TMP/belyj.txt"' if "# FILTR2022" in s else s
                             for s in tekst.splitlines()) + "\n", encoding="utf-8")
    r = _sobrat(mut, tmp_path / "m")
    assert r.returncode != 0 and "FAIL" in r.stdout and "tar_iskluchit_2022plus" in r.stdout, r.stdout + r.stderr
    assert not list((tmp_path / "m").glob("SHA256SUMS_testy"))             # пакет не выдан
    assert _plohie(next((tmp_path / "m").glob("testy_*.tar.gz"))) != []   # и tar мутанта действительно «грязный»


def _paket(tmp_path):
    r = _sobrat(SOBRAT, tmp_path / "o")
    assert r.returncode == 0, r.stdout + r.stderr
    p = tmp_path / "perenos"
    shutil.copytree(tmp_path / "o", p)
    shutil.copy(TESTY, p / "testy.sh")
    tar = next(p.glob("testy_*.tar.gz"))
    return p, tar, tar.name[len("testy_"):-len(".tar.gz")]


def _testy(p, sha, tmp_path):
    env = dict(os.environ, TESTY_SUHOJ="1", TESTY_SUHOJ_FAJLY=" ".join(SUHOJ), TMPDIR=str(tmp_path))
    return subprocess.run(["bash", str(p / "testy.sh"), sha], env=env, capture_output=True, text=True,
                          timeout=300, cwd=p)


def test_g3_testy_sh_suhoj_rezhim(tmp_path):
    p, _, sha = _paket(tmp_path)
    r = _testy(p, sha, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    itog = next(tmp_path.glob("testy_suhoj.*/itog.txt")).read_text(encoding="utf-8")
    for k in ("sha=", "passed=", "failed=", "skipped=", "vremya_s=", "pytest:", "upavshie:"):
        assert k in itog
    assert f"sha={sha}" in itog
    assert not (p / "kod").exists() and not (p / "data").exists() and not (p / "testy").exists()
    r2 = _testy(p, "0" * 12, tmp_path)
    assert r2.returncode != 0 and "FAIL" in r2.stdout              # нет tar -> отказ


def test_g3_sums_ne_sovpadaet_otkaz(tmp_path):
    p, tar, sha = _paket(tmp_path)
    tar.write_bytes(tar.read_bytes() + b"x")
    r = _testy(p, sha, tmp_path)
    assert r.returncode != 0 and "FAIL" in r.stdout
