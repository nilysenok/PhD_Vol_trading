"""pamyat.sh (30.10): память процесса — USS из smaps_rollup (не RSS форка), доступная — MemAvailable с верхом MemTotal, замер периодический.
Мутации: «RSS вместо USS» (Private_* → Rss) — test_uss падает; «сумма RSS без потолка MemTotal» — test_ne_zavyshaet падает;
«замер один раз» (sleep → return) — test_periodicheskij_zamer_lovit_rost падает. /proc подменяется каталогом PERENOS_PROC, ps — заглушкой."""
from __future__ import annotations

import os
import stat
import subprocess
import threading
from pathlib import Path

SH = Path(__file__).resolve().parents[1] / "scripts/server/perenos"
GB = 1024 * 1024        # КБ в ГБ


def _bash(tmp: Path, code: str, pamyat: str | None = None, **env) -> subprocess.CompletedProcess:
    (tmp / "obshchee.sh").write_text((SH / "obshchee.sh").read_text())
    (tmp / "pamyat.sh").write_text(pamyat if pamyat is not None else (SH / "pamyat.sh").read_text())
    (tmp / "VERSIYA").write_text("commit_short=abc\ntar_sha256=ff\nnumba_threads=1\nnumba_layer=workqueue\n")
    return subprocess.run(["bash", "-c", f". ./obshchee.sh; . ./pamyat.sh; {code}"], cwd=tmp, env={**os.environ, **env},
                          capture_output=True, text=True, timeout=60)


def _ps(tmp: Path, text: str) -> str:
    d = tmp / "bin"
    d.mkdir(exist_ok=True)
    f = d / "ps"
    f.write_text(f"#!/bin/sh\nprintf '{text}'\n")
    f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return f"{d}:{os.environ['PATH']}"


def _smaps(proc: Path, pid: int, rss: int, clean: int, dirty: int) -> None:
    (proc / str(pid)).mkdir(parents=True)
    (proc / str(pid) / "smaps_rollup").write_text(
        f"Rss:             {rss} kB\nPss:             {rss // 2} kB\nPrivate_Clean:   {clean} kB\nPrivate_Dirty:   {dirty} kB\nShared_Clean:    7 kB\n")


def _fake_proc(tmp: Path) -> Path:
    proc = tmp / "proc"
    _smaps(proc, 11, rss=2 * GB, clean=100, dirty=200)         # форк: RSS 2 ГБ, своих страниц 300 КБ
    _smaps(proc, 12, rss=3 * GB, clean=50, dirty=450)          # внук: 500 КБ своих
    _smaps(proc, 13, rss=9 * GB, clean=9 * GB, dirty=0)        # чужой процесс (не потомок корня 10)
    return proc


PS_TXT = "10 1\\n11 10\\n12 11\\n13 99\\n"


def test_uss_potomkov_a_ne_rss(tmp_path):
    proc, path = _fake_proc(tmp_path), _ps(tmp_path, PS_TXT)
    r = _bash(tmp_path, "pamyat_uss 10", PATH=path, PERENOS_PROC=str(proc))
    assert r.stdout.split() == ["500", "800"], r.stderr          # Private_Clean + Private_Dirty потомков 11 и 12; чужой 13 не считается
    mut = (SH / "pamyat.sh").read_text().replace("Private_(Clean|Dirty):", "Rss:")
    assert mut != (SH / "pamyat.sh").read_text()
    out = _bash(tmp_path, "pamyat_uss 10", pamyat=mut, PATH=path, PERENOS_PROC=str(proc)).stdout.split()
    assert out == [str(3 * GB), str(5 * GB)]                      # так считал RSS: тест выше на этом падает


def test_svobodno_iz_meminfo_i_net_proc_na_make(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "meminfo").write_text("MemTotal:       263192576 kB\nMemFree:  1 kB\nMemAvailable:   20971520 kB\n")
    assert _bash(tmp_path, "pamyat_svobodno", PERENOS_PROC=str(proc)).stdout.split() == ["20971520", "263192576"]
    assert _bash(tmp_path, "pamyat_svobodno", PERENOS_PROC=str(tmp_path / "net")).stdout.strip() == ""      # Мак: не измерено


def _sledit(tmp: Path, uss_body: str, svob: str, n: int = 120, pamyat: str | None = None, zhit: int = 4) -> tuple:
    (tmp / "logs").mkdir()
    (tmp / "zapusk_sobol.sh").write_text("#!/bin/sh\necho restart PERENOS_PROCS=$PERENOS_PROCS\n")
    sleeper = subprocess.Popen(["sleep", str(zhit)])
    threading.Thread(target=sleeper.wait, daemon=True).start()      # убрать зомби: `kill -0` по зомби успешен
    try:
        code = (f'pamyat_uss(){{ {uss_body}; }}; pamyat_svobodno(){{ echo "{svob}"; }}; pamyat_ostanovit(){{ :; }}; '
                f'pamyat_sledit train4 {sleeper.pid} {n}')
        r = _bash(tmp, code, pamyat=pamyat, PERENOS_PAMYAT_INTERVAL="1")
        assert r.returncode == 0, r.stderr
    finally:
        sleeper.kill()
    return (tmp / "logs/pamyat_train4.log").read_text(), r


def test_ne_zavyshaet_dostupnoe_120_po_2_4_gb(tmp_path):
    uss = 2516582                                                 # 2,4 ГБ своих страниц на процесс (рост рабочих)
    log, _ = _sledit(tmp_path, f'echo "{uss} {120 * uss}"', f"{10 * GB} {251 * GB}")
    assert "СНИЖЕНИЕ процессов 120 → " in log and "USS макс 2457 МБ" in log
    n = int(log.split("→ ")[1].split(":")[0])
    assert n == int(251 * GB / (uss * 1.15)) == 90               # верх — MemTotal 251 ГБ: av + ΣUSS (298 ГБ) физической памяти не бывает
    assert "продолжение с 90 процессами" in log
    mut = (SH / "pamyat.sh").read_text().replace('[ "$tot" -gt "$tm" ] && tot=$tm', ":")
    assert mut != (SH / "pamyat.sh").read_text()
    d = tmp_path / "m"
    d.mkdir()
    log2, _ = _sledit(d, f'echo "{uss} {120 * uss}"', f"{10 * GB} {251 * GB}", pamyat=mut)
    assert int(log2.split("→ ")[1].split(":")[0]) > 90            # без потолка завышено (103): тест выше падает


def test_pomeshchaetsya_ne_snizhaet(tmp_path):
    log, _ = _sledit(tmp_path, f'echo "{1 * GB} {120 * GB}"', f"{200 * GB} {251 * GB}")
    assert "СНИЖЕНИЕ" not in log and "помещается" in log and "цепочка закончилась" in log


def test_periodicheskij_zamer_lovit_rost(tmp_path):
    cnt = tmp_path / "cnt"
    cnt.write_text("0")
    # i-й замер: USS растёт на 0,6 ГБ; ΣUSS = 120 × USS; доступно 100 ГБ, всего 251 ГБ; первый замер помещается, затем нет
    body = (f'i=$(( $(cat {cnt}) + 1 )); echo $i > {cnt}; u=$(( i * 600000 )); echo "$u $(( u * 120 ))"')
    log, _ = _sledit(tmp_path, body, f"{100 * GB} {251 * GB}")
    assert log.index("помещается") < log.index("СНИЖЕНИЕ")        # рост поймал более поздний замер, не первый
    assert int(cnt.read_text()) >= 2
    mut = (SH / "pamyat.sh").read_text().replace('    sleep "$gran"\n  done', "    return 0\n  done")
    assert mut != (SH / "pamyat.sh").read_text()
    d = tmp_path / "m"
    d.mkdir()
    cnt.write_text("0")
    log2, _ = _sledit(d, body, f"{100 * GB} {251 * GB}", pamyat=mut)
    assert "СНИЖЕНИЕ" not in log2                                  # один замер роста не видит: тест выше падает


def test_net_proc_ne_izmereno(tmp_path):
    (tmp_path / "logs").mkdir()
    sleeper = subprocess.Popen(["sleep", "5"])
    try:
        r = _bash(tmp_path, f"pamyat_sledit train4 {sleeper.pid} 120", PERENOS_PROC=str(tmp_path / "net"), PERENOS_PAMYAT_INTERVAL="1")
    finally:
        sleeper.kill()
    assert r.returncode == 0, r.stderr
    assert "память не измерена" in (tmp_path / "logs/pamyat_train4.log").read_text()
