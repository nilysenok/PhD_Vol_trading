"""Перенос на сервер, лист P-2 (Ж-136 п. 5): период wf в zapusk_sobol.sh — процессы 80 по умолчанию, отметка владельца.
Мутации: wf считает как train4/v2 ((132−4)/2 = 64 или 128 — тест процессов падает); отметка проверяется только на существование
пустого файла / не проверяется (тест отметки падает); `PERENOS_PROCS_WF` игнорируется."""
import os
import subprocess
from pathlib import Path

SH = Path(__file__).resolve().parents[1] / "scripts/server/perenos"


def _bash(tmp: Path, code: str, **env) -> subprocess.CompletedProcess:
    (tmp / "obshchee.sh").write_text((SH / "obshchee.sh").read_text())
    (tmp / "VERSIYA").write_text("commit_short=abc\ntar_sha256=ff\nnumba_threads=1\nnumba_layer=workqueue\n")
    e = {k: v for k, v in os.environ.items() if not k.startswith("PERENOS_")}
    return subprocess.run(["bash", "-c", f". ./obshchee.sh; {code}"], cwd=tmp, env={**e, **env}, capture_output=True, text=True)


def test_wf_po_umolchaniyu_80_processov_a_ne_polovina_yader(tmp_path):
    assert _bash(tmp_path, "processov wf", PERENOS_YADRA="132").stdout.strip() == "80"
    assert _bash(tmp_path, "processov train4", PERENOS_YADRA="132").stdout.strip() == "120"     # прочие периоды — 120 (Ж-137)
    assert _bash(tmp_path, "processov", PERENOS_YADRA="132").stdout.strip() == "120"


def test_wf_peremennaya_i_potolok_yadra_minus_4(tmp_path):
    assert _bash(tmp_path, "processov wf", PERENOS_YADRA="132", PERENOS_PROCS_WF="40").stdout.strip() == "40"
    assert _bash(tmp_path, "processov wf", PERENOS_YADRA="64").stdout.strip() == "60"            # 80 > 64 − 4
    assert _bash(tmp_path, "processov wf", PERENOS_YADRA="132", PERENOS_PROCS="3").stdout.strip() == "3"   # явное — главнее


def test_wf_summa_s_zhivoj_cepochkoj_ne_bolshe_yadra_minus_4(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/sobol_train4.pid").write_text(str(os.getpid()))
    (tmp_path / "logs/sobol_train4.procs").write_text("64")
    bad = _bash(tmp_path, 'proverit_summu wf "$(processov wf)"; echo ok', PERENOS_YADRA="132")
    assert bad.returncode != 0 and "FAIL" in bad.stdout                    # 64 + 80 > 128


def test_wf_bez_otmetki_vladeltsa_otkaz(tmp_path):
    r = _bash(tmp_path, "otmetka_wf; echo ok")
    assert r.returncode != 0 and "FAIL: wf" in r.stdout and "ok" not in r.stdout.split("FAIL")[0]
    (tmp_path / "wf_zamok_snyat.json").write_text("")                      # пустая отметка — не отметка
    assert _bash(tmp_path, "otmetka_wf; echo ok").returncode != 0
    (tmp_path / "wf_zamok_snyat.json").write_text('{"sha256": "x"}')
    r = _bash(tmp_path, "otmetka_wf; echo ok")
    assert r.returncode == 0 and r.stdout.strip().endswith("ok")


def test_zapusk_sobol_znaet_period_wf_i_zovet_otmetku():
    s = (SH / "zapusk_sobol.sh").read_text()
    assert "train4|v2|wf" in s and "otmetka_wf" in s and 'processov "$per"' in s
