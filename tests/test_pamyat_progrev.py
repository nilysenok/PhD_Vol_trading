"""Сервер 30.10: наблюдатель памяти замерил USS процесса ПРОГРЕВА (1 893 МБ) и ложно снизил процессы 120 → 117.
Рабочий пула — 227 МБ. Прогрев (cmdline содержит progrev_) вне замера. Мутация «прогрев в замере» — тест падает."""
import os
import stat
import subprocess
from pathlib import Path

SH = Path(__file__).resolve().parents[1] / "scripts/server/perenos"
MB = 1024


def _proc(tmp: Path, progrev_mb: int, rabochie_mb: int, n: int) -> Path:
    """Подставной /proc: корень 100, прогрев 101 (cmdline ...progrev_train4), n рабочих 200+."""
    pr = tmp / "proc"
    pids = {101: ("python -m fastbt.progon_server --out cache/progrev_train4", progrev_mb)}
    pids.update({200 + i: ("python -m fastbt.progon_server --out ledger/train4_s1", rabochie_mb) for i in range(n)})
    for pid, (cmd, mb) in pids.items():
        d = pr / str(pid)
        d.mkdir(parents=True)
        (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode())
        (d / "smaps_rollup").write_text(f"Private_Clean: 0 kB\nPrivate_Dirty: {mb * MB} kB\nRss: 1 kB\n")
    return pr


def _uss(tmp: Path, pamyat: Path, proc: Path, n: int) -> list[int]:
    d = tmp / "bin"
    d.mkdir(exist_ok=True)
    ps = d / "ps"
    ps.write_text("#!/bin/sh\n" + "echo '101 100'\n" + "".join(f"echo '{200 + i} 100'\n" for i in range(n)))
    ps.chmod(ps.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{d}:{os.environ['PATH']}", "PERENOS_PROC": str(proc)}
    r = subprocess.run(["bash", "-c", f". {pamyat}; pamyat_uss 100"], env=env, capture_output=True, text=True)
    return [int(x) for x in r.stdout.split()]


def _reshenie(n: int, mx: int) -> int:
    tot = 251 * 1024 * MB                                # доступно цепочке (КБ): 251 ГБ
    r = subprocess.run(["bash", "-c", f". {SH / 'pamyat.sh'}; pamyat_reshenie {n} {mx} {tot}"], capture_output=True, text=True)
    return int(r.stdout)


def test_progrev_ne_v_zamere_snizheniya_net(tmp_path):
    n = 120
    mx, sm = _uss(tmp_path, SH / "pamyat.sh", _proc(tmp_path, 1893, 227, n), n)
    assert mx == 227 * MB and sm == n * 227 * MB          # прогрев не вошёл ни в максимум, ни в сумму
    assert _reshenie(n, mx) == n                          # 120 × 227 МБ × 1,15 помещается — снижения нет


def test_mutatsiya_progrev_v_zamere_lovitsya(tmp_path):
    n = 120
    mut = tmp_path / "pamyat_mut.sh"
    src = (SH / "pamyat.sh").read_text()
    stroka = next(s for s in src.splitlines() if "progrev_'" in s and "continue" in s)
    mut.write_text(src.replace(stroka, ""))               # мутация: проверка прогрева убрана
    mx, _ = _uss(tmp_path, mut, _proc(tmp_path, 1893, 227, n), n)
    assert mx == 1893 * MB                                # прогрев в замере -> ложный максимум 1 893 МБ
    assert _reshenie(n, mx) < n                           # ложное снижение — его и ловит основной тест
