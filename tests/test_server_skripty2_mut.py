"""Тестировщик #792: мутации server-skripty-2, не пойманные тестами автора (б, в, г) и мутация 1 (sha не передаются)."""
import os
import re
import subprocess
from pathlib import Path

from test_pamyat_progrev import SH, _proc, _uss

SERVER = SH.parent
MB = 1024


def _strochka_etalony(src: str) -> str:
    return src[src.index("etalony_mak.py"):].split("\n\n")[0]


def _sha_ok(src: str) -> bool:
    s = _strochka_etalony(src)
    return '--kod-sha "$TAR_SHA"' in s and '--dannye-sha "$(sha_file "$OUT/manifest_dannyh.tsv")"' in s


def test_sobrat_perenos_sha_tochnye_fajly():
    """kod-sha — sha tar, dannye-sha — sha manifest_dannyh.tsv из OUT (не tar, не исходного MANIFEST)."""
    assert _sha_ok((SERVER / "sobrat_perenos.sh").read_text())


def test_mutaciya_sha_ne_peredayutsya_lovitsya():
    src = (SERVER / "sobrat_perenos.sh").read_text()
    mut = re.sub(r'  --kod-sha .*--dannye-sha "\$\(sha_file "\$OUT/manifest_dannyh.tsv"\)" \\\n', "", src)
    assert mut != src and "--kod-sha" not in _strochka_etalony(mut) and not _sha_ok(mut)


def test_mutaciya_dannye_sha_ne_togo_fajla_lovitsya():
    src = (SERVER / "sobrat_perenos.sh").read_text()
    for zam in ('"$(sha_file "$MANIFEST")"', '"$TAR_SHA"'):
        mut = src.replace('--dannye-sha "$(sha_file "$OUT/manifest_dannyh.tsv")"', f"--dannye-sha {zam}")
        assert mut != src and not _sha_ok(mut)
    mut = src.replace('--kod-sha "$TAR_SHA"', '--kod-sha "$(sha_file "$OUT/manifest_dannyh.tsv")"')
    assert mut != src and not _sha_ok(mut)


def test_rabochij_s_podstrokoj_progrev_v_argumentah_propuskaetsya_poka_tak_zadumano(tmp_path):
    """(б) Фиксирует границу: рабочий с «progrev_» в cmdline выпадает из замера. В реальных аргументах
    (--out ledger/<период>_<ключ>, --kub, --period train4|v2|wf) подстроки нет; каталог cache/progrev_<период> — только у прогрева.
    Тест — документ: если рабочий получит такой аргумент, замер занизится (мутация осознанная)."""
    pr = _proc(tmp_path, 1893, 227, 2)
    (pr / "200" / "cmdline").write_bytes(b"python\0-m\0fastbt.progon_server\0--out\0ledger/progrev_x")
    mx, sm = _uss(tmp_path, SH / "pamyat.sh", pr, 2)
    assert sm == 227 * MB and mx == 227 * MB          # рабочий 200 пропущен, остался один (201)


def _fail_mut(tmp: Path, rc: int, vyvod: str, mut) -> str:
    (tmp / "obshchee.sh").write_text(mut((SH / "obshchee.sh").read_text()))
    (tmp / "VERSIYA").write_text("commit_short=abc1234\ntar_sha256=ff\nnumba_threads=1\nnumba_layer=workqueue\n")
    (tmp / "vyv.txt").write_text(vyvod)
    r = subprocess.run(["bash", "-c", f". ./obshchee.sh; fail_progon s1 {rc} vyv.txt"], cwd=tmp, capture_output=True, text=True,
                       env={**os.environ})
    return r.stdout


def _perepytat(src: str) -> str:
    """(в) коды 1 и 2 перепутаны: ветка 2) <-> 1)."""
    a = src.replace("    2) fail \"проход $1: progon_server вернул код 2", "    9) fail \"проход $1: progon_server вернул код 2")
    return a.replace("    1) if grep", "    2) if grep").replace("    9) fail", "    1) fail")


def test_mutaciya_kody_1_i_2_pereputany_lovitsya(tmp_path):
    otkaz = "x: --prodolzhit — отказ, не совпали ['kod']\n"
    mut = _perepytat((SH / "obshchee.sh").read_text())
    assert mut != (SH / "obshchee.sh").read_text()
    t1 = _fail_mut(tmp_path, 1, otkaz, _perepytat)       # код 1 в мутанте говорит про ошибки точек
    assert "ошибки точек (строки status = oshibka" in t1 and "отказался продолжать" not in t1
    t2 = _fail_mut(tmp_path, 2, "", _perepytat)          # код 2 в мутанте — «отказ/аргументы»
    assert "ошибки точек (строки status = oshibka" not in t2
    # на исходнике — наоборот
    assert "отказался продолжать чужой ledger" in _fail_mut(tmp_path, 1, otkaz, lambda s: s)
    assert "ошибки точек (строки status = oshibka" in _fail_mut(tmp_path, 2, "", lambda s: s)
