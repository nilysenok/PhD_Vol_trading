"""Перенос на сервер, лист P-1: процессы двух цепочек (сумма ≤ ядра − 4) и сужение zabrat.sh (ledger 2.0 остаётся на сервере)."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SH = Path(__file__).resolve().parents[1] / "scripts/server/perenos"


def _bash(tmp: Path, code: str, **env) -> subprocess.CompletedProcess:
    (tmp / "obshchee.sh").write_text((SH / "obshchee.sh").read_text())
    (tmp / "VERSIYA").write_text("commit_short=abc\ntar_sha256=ff\nnumba_threads=1\nnumba_layer=workqueue\n")
    e = {**os.environ, **env}
    return subprocess.run(["bash", "-c", f". ./obshchee.sh; {code}"], cwd=tmp, env=e, capture_output=True, text=True)


def test_po_umolchaniyu_120_processov_cepochki_delyat_po_zaprosu(tmp_path):      # Ж-137 п. 2 (было (132 − 4) / 2 = 64)
    assert _bash(tmp_path, "processov", PERENOS_YADRA="132").stdout.strip() == "120"
    assert _bash(tmp_path, "processov", PERENOS_YADRA="100").stdout.strip() == "96"      # потолок ядра − 4
    assert _bash(tmp_path, "processov", PERENOS_YADRA="132", PERENOS_CEPOCHEK="2").stdout.strip() == "64"
    assert _bash(tmp_path, "processov", PERENOS_YADRA="132", PERENOS_PROCS_DEF="50").stdout.strip() == "50"


def _zhivaya(tmp: Path, per: str, n: int):
    (tmp / "logs").mkdir(exist_ok=True)
    (tmp / "logs" / f"sobol_{per}.pid").write_text(str(os.getpid()))      # живой pid — этого теста
    (tmp / "logs" / f"sobol_{per}.procs").write_text(str(n))


def test_summa_processov_zhivyh_cepochek_ne_bolshe_yadra_minus_4(tmp_path):
    _zhivaya(tmp_path, "v2", 64)
    ok = _bash(tmp_path, "proverit_summu train4 64; echo ok", PERENOS_YADRA="132")
    assert ok.stdout.strip().endswith("ok") and ok.returncode == 0           # 64 + 64 = 128 ≤ 128
    bad = _bash(tmp_path, "proverit_summu train4 65; echo ok", PERENOS_YADRA="132")
    assert bad.returncode != 0 and "FAIL" in bad.stdout and "ok" not in bad.stdout.split("FAIL")[0]
    # полная загрузка («ядра − 4» каждой) при живой другой цепочке — отказ
    full = _bash(tmp_path, 'proverit_summu train4 "$(PERENOS_CEPOCHEK=1 processov)"', PERENOS_YADRA="132")
    assert full.returncode != 0


def test_mertvaya_cepochka_ne_schitaetsya(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/sobol_v2.pid").write_text("999999")
    (tmp_path / "logs/sobol_v2.procs").write_text("128")
    assert _bash(tmp_path, "chuzhie_processy train4", PERENOS_YADRA="132").stdout.strip() == "0"


@pytest.mark.skipif(shutil.which("rsync") is None, reason="нет rsync")
def test_zabrat_beryot_ledger_21_i_ne_beryot_20(tmp_path):
    src, dst = tmp_path / "srv", tmp_path / "mak"
    for chain, v in (("train4_s1", "2.1"), ("v2_s1", "2.0")):
        d = src / "ledger" / chain / "S1"
        (d / "p00" / "port").mkdir(parents=True)
        (d / "manifest.json").write_text('{\n "v": "%s"\n}\n' % v)
        (d / "checkpoint.json").write_text("{}")
        (d / "p00" / "port" / "00000.npy").write_text("x")
    for k in ("tables", "logs"):
        (src / k).mkdir()
        (src / k / "f.txt").write_text("x")
    r = subprocess.run(["bash", str(SH / "zabrat.sh"), str(src), str(dst)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (dst / "ledger/train4_s1/S1/p00/port/00000.npy").exists()             # шарды 2.1 — на Маке
    assert not (dst / "ledger/v2_s1/S1/p00").exists()                            # шарды 2.0 — нет
    assert (dst / "ledger/v2_s1/S1/manifest.json").exists() and (dst / "ledger/v2_s1/S1/checkpoint.json").exists()
    assert (dst / "tables/f.txt").exists() and (dst / "logs/f.txt").exists()
    dry = subprocess.run(["bash", str(SH / "zabrat.sh"), str(src), str(tmp_path / "dry")], capture_output=True, text=True,
                         env={**os.environ, "PERENOS_DRYRUN": "1"})
    assert dry.returncode == 0 and not list((tmp_path / "dry").rglob("*.npy"))
