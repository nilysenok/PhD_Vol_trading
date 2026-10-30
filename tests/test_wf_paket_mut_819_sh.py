"""Тестировщик #819: perenos/proverka_yakor.sh на синтетическом пакете (эталон без записей, только meta с периодом). Пересчёта нет:
несовпадение периода эталона и perenos_do (VERSIYA) останавливает сверку до счёта, код возврата 1, отметки proverka/OK нет.
Мутации: из скрипта убран `--do`; perenos_do потерян в VERSIYA (wf-эталон молча сверился бы как train4)."""
import hashlib
import os
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

SRV = Path(__file__).resolve().parents[1] / "scripts/server"
sys.path.insert(0, str(SRV))
import ekviv_io  # noqa: E402

DO = "2026-02-28"


def _paket(tmp, do, period_etalona):
    """Минимальный perenos/: оболочка (obshchee.sh, proverka_yakor.sh) + код (скрипты ekviv*) + пустой эталон нужного периода."""
    P = tmp / "perenos"
    P.mkdir()
    for n in ("obshchee.sh", "proverka_yakor.sh"):
        (P / n).write_text((SRV / "perenos" / n).read_text())
    kod = tmp / "kod_src/scripts/server"
    kod.mkdir(parents=True)
    for n in ("ekviv.py", "ekviv_io.py", "ekviv_sravn.py", "ekviv_v6.py", "proverka_ekviv.py"):
        (kod / n).write_text((SRV / n).read_text())
    tar = P / "kod_abc.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(tmp / "kod_src", arcname="kod")
    sha = hashlib.sha256(tar.read_bytes()).hexdigest()
    kd = P / "kod" / sha
    (kd / "scripts/server").mkdir(parents=True)
    sums = []
    for n in sorted(os.listdir(kod)):
        (kd / "scripts/server" / n).write_text((kod / n).read_text())
        sums.append(f"{hashlib.sha256((kd / 'scripts/server' / n).read_bytes()).hexdigest()}  ./scripts/server/{n}")
    (P / "kod" / f"{sha}.SHA256SUMS").write_text("\n".join(sums) + "\n")
    for f in (kd / "scripts/server").iterdir():
        f.chmod(stat.S_IREAD)
    (P / "VERSIYA").write_text(f"commit_short=abc\ntar_sha256={sha}\nnumba_threads=1\nnumba_layer=workqueue\nperenos_do={do}\n")
    (P / "manifest_dannyh.tsv").write_text("put\tsha256\n")
    for n in ("manifest_proverka.py", "versii.py"):
        (P / n).write_text("import sys\nsys.exit(0)\n")
    (P / "requirements_server.txt").write_text("")
    (P / ".venv/bin").mkdir(parents=True)
    py = P / ".venv/bin/python3"
    py.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    py.chmod(0o755)
    meta = {"machine": "t", "numba": "0", "numpy": "0"} | ({"period": period_etalona} if period_etalona != "train4" else {})
    ekviv_io.sohranit(str(P / "etalony_abc.npz"), {}, meta)
    return P


def _sh(P, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("PERENOS_")}
    e["PERENOS_SYSTEM_SITE"] = "1"                 # numpy тестового интерпретатора может лежать в пользовательском site
    return subprocess.run(["bash", "proverka_yakor.sh"], cwd=P, env={**e, **env}, capture_output=True, text=True)


def test_sh_etalon_train4_pri_perenos_do_wf_fail_kod_1(tmp_path):
    P = _paket(tmp_path, DO, "train4")
    r = _sh(P)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "FAIL дискретное" in r.stdout and "даёт wf" in r.stdout and not (P / "proverka/OK").exists()


def test_sh_mutatsiya_do_ne_peredayotsya_lovitsya(tmp_path):
    """Подложено: из proverka_yakor.sh убран `--do`; тогда сообщение о несовпадении периода исчезает (его и ждёт тест выше)."""
    P = _paket(tmp_path, DO, "train4")
    s = (P / "proverka_yakor.sh").read_text()
    assert '--do "${PERENOS_DO:-}"' in s
    (P / "proverka_yakor.sh").write_text(s.replace('--do "${PERENOS_DO:-}"', ""))
    assert "даёт wf" not in _sh(P).stdout


def test_sh_mutatsiya_perenos_do_poteryan_lovitsya(tmp_path):
    """Эталон wf, а perenos_do в VERSIYA пуст (поле потеряно при сборке): сверка не должна пройти молча как train4."""
    P = _paket(tmp_path, "", "wf")
    r = _sh(P)
    assert r.returncode == 1 and "даёт train4" in r.stdout and not (P / "proverka/OK").exists()
