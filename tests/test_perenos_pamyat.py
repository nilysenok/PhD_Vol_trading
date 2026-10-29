"""Перенос на сервер, Ж-137 п. 1–3: N1 = 200 000, проверка памяти при старте (подставные ps/free), DOPUSK 1e-12.
Мутации: «без проверки памяти» (pamyat_reshenie всегда n; сброс zapusk_sobol без pamyat_sledit) — тесты снижения и вызова падают;
N1 262 144 в zapusk_sobol — тест N1 падает; DOPUSK по умолчанию 0 — тест допуска падает."""
import os
import stat
import subprocess
from pathlib import Path

SH = Path(__file__).resolve().parents[1] / "scripts/server/perenos"


def _bash(tmp: Path, code: str, **env) -> subprocess.CompletedProcess:
    for f in ("obshchee.sh", "pamyat.sh"):
        (tmp / f).write_text((SH / f).read_text())
    (tmp / "VERSIYA").write_text("commit_short=abc\ntar_sha256=ff\nnumba_threads=1\nnumba_layer=workqueue\n")
    return subprocess.run(["bash", "-c", f". ./obshchee.sh; . ./pamyat.sh; {code}"], cwd=tmp, env={**os.environ, **env},
                          capture_output=True, text=True)


def _stub(tmp: Path, name: str, body: str) -> str:
    d = tmp / "bin"
    d.mkdir(exist_ok=True)
    f = d / name
    f.write_text("#!/bin/sh\n" + body)
    f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return f"{d}:{os.environ['PATH']}"


def test_reshenie_snizhaet_kogda_ne_pomeshchaetsya(tmp_path):
    # 120 × 2 ГБ × 1,15 = 276 ГБ > 200 ГБ -> int(200 / (2 × 1,15)) = 86
    kb = 1024 * 1024
    assert _bash(tmp_path, f"pamyat_reshenie 120 {2 * kb} {200 * kb}").stdout.strip() == "86"
    assert _bash(tmp_path, f"pamyat_reshenie 120 {1 * kb} {200 * kb}").stdout.strip() == "120"     # помещается — без изменений
    assert _bash(tmp_path, f"pamyat_reshenie 120 {300 * kb} {200 * kb}").stdout.strip() == "1"      # не меньше 1


def test_nablyudatel_snizhaet_i_pishet_prichinu_v_log(tmp_path):
    kb = 1024 * 1024
    path = _stub(tmp_path, "ps", f'case "$*" in *ppid=*) echo "11 10 {2 * kb}"; echo "12 10 {2 * kb}";;'
                                 f' *) echo "10 1 1000";; esac\n')
    path = _stub(tmp_path, "free", f'echo "Mem: 262144000 0 0 0 0 {180 * kb}"\n').replace(f"{tmp_path}/bin:", f"{tmp_path}/bin:", 1)
    (tmp_path / "logs").mkdir()
    (tmp_path / "ledger/train4_s1/S1").mkdir(parents=True)
    (tmp_path / "ledger/train4_s1/S1/checkpoint.json").write_text("{}")
    (tmp_path / "zapusk_sobol.sh").write_text("#!/bin/sh\necho restart PERENOS_PROCS=$PERENOS_PROCS\n")
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        # ps подставной: считает потомками корня pid 10; pid 10 — условный корень (sleep жив, чтобы проверка kill -0 прошла)
        code = f'pamyat_rss(){{ echo "{2 * kb} {4 * kb}"; }}; pamyat_ostanovit(){{ :; }}; pamyat_sledit train4 {sleeper.pid} 120'
        r = _bash(tmp_path, code, PATH=path, PERENOS_PAMYAT_ZHDAT="1")
        assert r.returncode == 0, r.stderr
        log = (tmp_path / "logs/pamyat_train4.log").read_text()
        assert "СНИЖЕНИЕ процессов 120 → " in log and "RSS макс 2048 МБ" in log
        assert "продолжение с" in log
        n = int(log.split("→ ")[1].split(":")[0])
        assert 1 <= n < 120 and (tmp_path / "logs/sobol_train4.procs").read_text().strip() == str(n)
    finally:
        sleeper.kill()


def test_rss_potomkov_korня_iz_ps(tmp_path):
    path = _stub(tmp_path, "ps", 'printf "10 1 5\\n11 10 100\\n12 11 300\\n13 99 900\\n"\n')
    assert _bash(tmp_path, "pamyat_rss 10", PATH=path).stdout.split() == ["300", "400"]      # чужой pid 13 не считается


def test_zapusk_vyzyvaet_proverku_pamyati_i_n1_200000():
    s = (SH / "zapusk_sobol.sh").read_text()
    assert "pamyat_sledit" in s and '. "$P/pamyat.sh"' in s          # мутация «без проверки памяти» убирает вызов
    assert "PERENOS_N1:-200000" in s and "262144" not in s


def test_sborshchik_beryot_kuby_v6_v_tar():
    """Находка сквозной проверки 29.10: белый список не содержал designs/v6 — Моррис v6 на сервере не нашёл бы кубов."""
    s = (SH.parent / "sobrat_perenos.sh").read_text()
    assert "(v5|v6|zh126)/" in s
    ls = subprocess.run(["git", "ls-files", "fastbt/designs/v6"], capture_output=True, text=True, cwd=SH.parents[2]).stdout
    assert "fastbt/designs/v6/kub_S1.yaml" in ls


def test_dopusk_po_umolchaniyu_1e12():
    src = (Path(__file__).resolve().parents[1] / "scripts/server/proverka_ekviv.py").read_text()
    assert 'os.environ.get("DOPUSK", "1e-12")' in src
    assert "DOPUSK:-1e-12" in (SH / "proverka_yakor.sh").read_text()
