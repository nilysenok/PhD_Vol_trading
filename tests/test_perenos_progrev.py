"""Перенос: прогрев кэшей одним процессом до пула (zapusk_sobol.sh) и доля ядра в svodka_ledger (итог 29.10)."""
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SH = Path(__file__).resolve().parents[1] / "scripts/server/perenos"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/server"))
import svodka_ledger as sv  # noqa: E402

STUB = """#!/usr/bin/env bash
if [ "$*" = "-m fastbt.progon_server --help" ]; then echo "--k3"; exit 0; fi
echo "$*" >> "$STUB_LOG"
[ -n "${STUB_FAIL:-}" ] && case "$*" in *--n1\\ 1\\ *|*--r\\ 1*) exit 7 ;; esac
exit 0
"""


def _pakyet(tmp: Path) -> Path:
    for f in ("obshchee.sh", "pamyat.sh", "zapusk_sobol.sh"):
        (tmp / f).write_text((SH / f).read_text())
    (tmp / "VERSIYA").write_text("commit_short=abc\ntar_sha256=ff\nnumba_threads=1\nnumba_layer=workqueue\nkod_slit=0\n")
    (tmp / "kod/ff/fastbt").mkdir(parents=True)
    (tmp / "kod/ff/fastbt/progon_server.py").write_text("")
    (tmp / ".venv/bin").mkdir(parents=True)
    py = tmp / ".venv/bin/python3"
    py.write_text(STUB)
    py.chmod(py.stat().st_mode | stat.S_IXUSR)
    return tmp


def _ryad(tmp: Path, **env):
    log = tmp / "calls.log"
    e = {**os.environ, "STUB_LOG": str(log), "PERENOS_PROHODY": "s1", "PERENOS_YADRA": "132", **env}
    r = subprocess.run(["bash", "zapusk_sobol.sh", "__ryad", "train4"], cwd=tmp, env=e, capture_output=True, text=True)
    return r, (log.read_text().splitlines() if log.exists() else [])


def test_progrev_idet_do_pula_odin_process_odna_tochka(tmp_path):
    r, calls = _ryad(_pakyet(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(calls) == 3
    prog, pul, yadro = calls
    assert "svodka_ledger.py --yadro" in yadro and "ledger/train4_s1" in yadro      # после прохода — строка «ядро»
    assert "--n1 1 " in prog + " " and "--procs 1" in prog and "progrev_train4" in prog
    assert "--procs 120" in pul and "--n1 200000" in pul and "--fast" in pul
    assert "прогрев кэшей: " in r.stdout and " с" in r.stdout


def test_progrev_upal_pul_ne_startuet(tmp_path):
    r, calls = _ryad(_pakyet(tmp_path), STUB_FAIL="1")
    assert r.returncode != 0 and "прогрев упал" in r.stdout
    assert len(calls) == 1 and "--procs 1" in calls[0]


def test_perenos_fast_0_bez_fast(tmp_path):
    _, calls = _ryad(_pakyet(tmp_path), PERENOS_FAST="0")
    assert not any("--fast" in c for c in calls)


def test_v6_progrev_morris_r1_do_morrisa(tmp_path):
    _, calls = _ryad(_pakyet(tmp_path), PERENOS_KUB="v6")
    assert "progon_morris_v6" in calls[0] and "--r 1" in calls[0] and "--procs 1" in calls[0]
    assert "progon_morris_v6" in calls[1] and "--procs 120" in calls[1]


def test_dolya_yadra_schitaet_fast_i_pustye_lenty():
    assert sv.dolya_yadra([{"sloy": "fast"}, {"sloy": "slow"}, {"sloy": "fast"}, {}]) == "50.0"
    assert sv.dolya_yadra([]) == "—"
    assert sv.dolya_yadra([{"sloy": "slow"}]) == "0.0"           # мутация: считать долю slow → 100.0


@pytest.mark.skipif(not (SH / "zapusk_sobol.sh").exists(), reason="нет скрипта")
def test_progrev_pered_pulom_v_tekste_skripta():
    t = (SH / "zapusk_sobol.sh").read_text()
    assert t.index('progrev "$per"') < t.index("-m fastbt.progon_morris_v6 --kub v6 --period \"$per\" --out \"$mo\"")
    assert t.index('progrev "$per"') < t.index('-m fastbt.progon_server --kub "$kub"')
