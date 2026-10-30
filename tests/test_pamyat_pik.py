"""pamyat.sh, разовый пик (30.10): рабочие 1,0–1,9 ГБ, в конце прохода один замер дал 3840 МБ — цепочка 107 → 57 срезана зря.
Правило: срез при превышении в двух замерах подряд или сразу при MemAvailable < 5 % MemTotal; родитель progon_server в максимум не входит.
Мутации: «срез по одному замеру» ([ "$podryad" -lt 2 ] → 1) — test_pik_odin_zamer падает; «родитель × n» (progon_server не исключён) —
test_roditel_ne_v_maksimume падает; «угрозы нет» (ugroza всегда 0) — test_nehvatka_pamyati_srez_srazu падает. /proc — каталог PERENOS_PROC."""
from __future__ import annotations

import os
import stat
import subprocess
import threading
from pathlib import Path

SH = Path(__file__).resolve().parents[1] / "scripts/server/perenos"
GB = 1024 * 1024
NORMA, PIK = 1992294, 3984588            # КБ: 1,9 ГБ и 3840 МБ
SRC = (SH / "pamyat.sh").read_text()


def _bash(tmp: Path, code: str, pamyat: str | None = None, **env) -> subprocess.CompletedProcess:
    (tmp / "obshchee.sh").write_text((SH / "obshchee.sh").read_text())
    (tmp / "pamyat.sh").write_text(pamyat if pamyat is not None else SRC)
    (tmp / "VERSIYA").write_text("commit_short=abc\ntar_sha256=ff\nnumba_threads=1\nnumba_layer=workqueue\n")
    return subprocess.run(["bash", "-c", f". ./obshchee.sh; . ./pamyat.sh; {code}"], cwd=tmp, env={**os.environ, **env},
                          capture_output=True, text=True, timeout=60)


def _zamery(tmp: Path, seriya: list[int], av_gb: int, pamyat: str | None = None, zhit: int = 6) -> str:
    """i-й замер — USS макс seriya[i] (последний повторяется); ΣUSS = n × USS; n = 107, всего 251 ГБ, доступно av_gb."""
    (tmp / "logs").mkdir()
    (tmp / "zapusk_sobol.sh").write_text("#!/bin/sh\necho restart PERENOS_PROCS=$PERENOS_PROCS\n")
    cnt = tmp / "cnt"
    cnt.write_text("0")
    arr = " ".join(map(str, seriya))
    body = (f'i=$(( $(cat {cnt}) + 1 )); echo $i > {cnt}; a=({arr}); u=${{a[$(( i - 1 < {len(seriya)} ? i - 1 : {len(seriya) - 1} ))]}}; '
            f'echo "$u $(( u * 107 ))"')
    sleeper = subprocess.Popen(["sleep", str(zhit)])
    threading.Thread(target=sleeper.wait, daemon=True).start()
    try:
        code = (f'pamyat_uss(){{ {body}; }}; pamyat_svobodno(){{ echo "{av_gb * GB} {251 * GB}"; }}; pamyat_ostanovit(){{ :; }}; '
                f'pamyat_sledit train4 {sleeper.pid} 107')
        r = _bash(tmp, code, pamyat=pamyat, PERENOS_PAMYAT_INTERVAL="1")
        assert r.returncode == 0, r.stderr
    finally:
        sleeper.kill()
    return (tmp / "logs/pamyat_train4.log").read_text()


def test_pik_odin_zamer(tmp_path):
    log = _zamery(tmp_path, [NORMA, PIK, NORMA], av_gb=226)
    assert "СНИЖЕНИЕ" not in log and "ждём второго" in log        # G1: один пик 3,8 ГБ при остальных 1,9 — среза нет
    mut = SRC.replace('[ "$podryad" -lt 2 ]', '[ "$podryad" -lt 1 ]')
    assert mut != SRC
    d = tmp_path / "m"
    d.mkdir()
    assert "СНИЖЕНИЕ процессов 107 → 57" in _zamery(d, [NORMA, PIK, NORMA], av_gb=226, pamyat=mut)    # срез по одному замеру: тест выше падает


def test_dva_zamera_podryad_srez(tmp_path):
    log = _zamery(tmp_path, [NORMA, PIK, PIK], av_gb=226)
    assert "СНИЖЕНИЕ процессов 107 → 57" in log and "ждём второго" in log      # G2: второй замер подряд — срез


def test_nehvatka_pamyati_srez_srazu(tmp_path):
    log = _zamery(tmp_path, [PIK], av_gb=5, zhit=1)               # 5 ГБ из 251 (< 5 %): угроза OOM, замер один
    assert "СНИЖЕНИЕ" in log and "ждём второго" not in log        # G3
    mut = SRC.replace("ugroza=0; [", "ugroza=0; false && [")
    assert mut != SRC
    d = tmp_path / "m"
    d.mkdir()
    assert "СНИЖЕНИЕ" not in _zamery(d, [PIK], av_gb=5, pamyat=mut, zhit=1)


def test_roditel_ne_v_maksimume(tmp_path):
    proc, bind = tmp_path / "proc", tmp_path / "bin"
    bind.mkdir()
    for pid, uss, cmd in ((11, PIK, "python -m fastbt.progon_server"), (12, NORMA, "python -m fastbt.progon_server"),
                          (13, NORMA, "python -m fastbt.progon_server")):
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "smaps_rollup").write_text(f"Private_Clean:   {uss} kB\nPrivate_Dirty:   0 kB\n")
        (proc / str(pid) / "cmdline").write_bytes(cmd.replace(" ", "\0").encode())
    ps = bind / "ps"
    ps.write_text("#!/bin/sh\nprintf '10 1\\n11 10\\n12 11\\n13 11\\n'\n")
    ps.chmod(ps.stat().st_mode | stat.S_IEXEC)
    env = dict(PATH=f"{bind}:{os.environ['PATH']}", PERENOS_PROC=str(proc))
    assert _bash(tmp_path, "pamyat_uss 10", **env).stdout.split() == [str(NORMA), str(PIK + 2 * NORMA)]    # G: родитель в сумме, не в максимуме
    mut = SRC.replace("grep -q 'progon_server' && continue", "grep -q 'NEVER' && continue")
    assert mut != SRC
    assert _bash(tmp_path, "pamyat_uss 10", pamyat=mut, **env).stdout.split()[0] == str(PIK)             # родитель × n: тест выше падает
