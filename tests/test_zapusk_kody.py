"""Сервер 30.10: zapusk_sobol.sh писал «ошибки точек» при коде 1 progon_server, хотя это отказ продолжать чужой ledger.
Коды progon_server: 1 — SystemExit (отказ / аргументы), 2 — есть строки status = oshibka. Мутация: сообщение кода 1 как у кода 2 — падает."""
import subprocess
from pathlib import Path

SH = Path(__file__).resolve().parents[1] / "scripts/server/perenos"
OTKAZ = "ledger/train4_s1/S1/checkpoint.json: --prodolzhit — отказ, не совпали ['kod', 'vhody']\n"


def _fail(tmp: Path, rc: int, vyvod: str) -> str:
    (tmp / "obshchee.sh").write_text((SH / "obshchee.sh").read_text())
    (tmp / "VERSIYA").write_text("commit_short=abc1234\ntar_sha256=ff\nnumba_threads=1\nnumba_layer=workqueue\n")
    (tmp / "vyv.txt").write_text(vyvod)
    r = subprocess.run(["bash", "-c", f". ./obshchee.sh; fail_progon s1 {rc} vyv.txt"], cwd=tmp, capture_output=True, text=True)
    assert r.returncode == 1
    return r.stdout


def test_kod_1_otkaz_prodolzheniya_ne_oshibki_tochek(tmp_path):
    t = _fail(tmp_path, 1, OTKAZ)
    assert "отказался продолжать чужой ledger" in t and "ошибок точек нет" in t
    assert "['kod', 'vhody']" in t and "сейчас код abc1234" in t and "перенесите ledger в сторону" in t
    assert "status = oshibka" not in t


def test_kod_1_bez_otkaza_prodolzheniya_argumenty(tmp_path):
    t = _fail(tmp_path, 1, "одна стратегия указана дважды\n")
    assert "отказ или неверные аргументы" in t and "status = oshibka" not in t


def test_kod_2_oshibki_tochek_i_nepredvidennyj_kod(tmp_path):
    assert "ошибки точек (строки status = oshibka" in _fail(tmp_path, 2, "")
    assert "код 137 — неожиданный код" in _fail(tmp_path, 137, "")


def test_zapusk_zovyot_razbor_kodov():
    s = (SH / "zapusk_sobol.sh").read_text()
    assert 'fail_progon "$pr" "$rc"' in s and "ошибки точек — строки" not in s     # мутация: старая единая фраза
