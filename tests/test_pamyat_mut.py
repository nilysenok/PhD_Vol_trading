"""Проверка #790 (память Соболя): пробелы после правки автора. Д2 (#752: чекпоинт прошлого прохода принят за новый) в новой логике
pamyat.sh ждать чекпоинт перестал; его место — «прошлый проход не считается в новом»: после перезапуска наблюдатель идёт за НОВЫМ pid
(не за убитым старым) и максимум USS прошлого прохода не переносится. Мутации: «pid=$np» убран — тест падает; «top=0» убран — падает;
файлы logs/sobol_<период>.procs/.pid не пишутся — падает (это защита удалённого теста test_nablyudatel_snizhaet...).
Кэш: очистка на каждой точке (а не раз в K) на длинной серии; реальная v5-точка (FASTBT_V6_REAL=1) — результат побайтно прежний
с очисткой и без неё, пересчёта сигналов внутри точки не больше, чем у свежего оценщика."""
from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest

SH = Path(__file__).resolve().parents[1] / "scripts/server/perenos"
GB = 1024 * 1024
USS_OLD, USS_NEW = 2516582, 1048576          # КБ: прошлый проход (2,4 ГБ) и новый (1 ГБ)


def _perezapusk(tmp: Path, pamyat: str | None = None) -> tuple[str, Path]:
    """Старый проход (pid sleeper) → СНИЖЕНИЕ → перезапуск (заглушка живёт 4 с). Возвращает (лог, файл вызовов pamyat_uss по pid)."""
    for f in ("obshchee.sh", "pamyat.sh"):
        (tmp / f).write_text(pamyat if (f == "pamyat.sh" and pamyat is not None) else (SH / f).read_text())
    (tmp / "VERSIYA").write_text("commit_short=abc\ntar_sha256=ff\nnumba_threads=1\nnumba_layer=workqueue\n")
    (tmp / "logs").mkdir()
    (tmp / "zapusk_sobol.sh").write_text("#!/bin/sh\necho restart PERENOS_PROCS=$PERENOS_PROCS\nsleep 4\n")
    calls = tmp / "calls"
    old = subprocess.Popen(["sleep", "20"])
    threading.Thread(target=old.wait, daemon=True).start()
    body = (f'echo "$1" >> {calls}; if [ "$1" = {old.pid} ]; then echo "{USS_OLD} $((120 * {USS_OLD}))"; '
            f'else echo "{USS_NEW} $((90 * {USS_NEW}))"; fi')
    code = (f'pamyat_uss(){{ {body}; }}; pamyat_svobodno(){{ echo "{150 * GB} {251 * GB}"; }}; pamyat_ostanovit(){{ kill "$1"; }}; '
            f'pamyat_sledit train4 {old.pid} 120')
    try:
        r = subprocess.run(["bash", "-c", f". ./obshchee.sh; . ./pamyat.sh; {code}"], cwd=tmp, capture_output=True, text=True, timeout=60,
                           env={**os.environ, "PERENOS_PAMYAT_INTERVAL": "1"})
        assert r.returncode == 0, r.stderr
    finally:
        old.kill()
    (tmp / "old_pid").write_text(str(old.pid))
    return (tmp / "logs/pamyat_train4.log").read_text(), calls


def _mut(old: str, new: str) -> str:
    src = (SH / "pamyat.sh").read_text()
    assert old in src
    return src.replace(old, new)


def _usloviya(d: Path, log: str, calls: Path) -> list[bool]:
    """Условия правильного перезапуска (каждое защищает своё): замеры по новому pid; pid- и procs-файлы; сниженное число; максимум нового."""
    old = (d / "old_pid").read_text()
    pids = calls.read_text().split()
    pf, qf = d / "logs/sobol_train4.pid", d / "logs/sobol_train4.procs"
    new = pf.read_text().strip() if pf.exists() else ""
    return [new not in ("", old) and pids[0] == old and pids.count(new) >= 2,
            qf.exists() and qf.read_text().strip() == "90",
            log.count("СНИЖЕНИЕ") == 1 and "продолжение с 90 процессами" in log,
            log.strip().splitlines()[-1].endswith(f"USS макс {USS_NEW // 1024} МБ")]


def test_perezapusk_nablyudatel_idyot_za_novym_prohodom(tmp_path):
    log, calls = _perezapusk(tmp_path)
    assert _usloviya(tmp_path, log, calls) == [True] * 4
    assert "PERENOS_PROCS=90" in (tmp_path / "logs/sobol_train4_pamyat.log").read_text()      # перезапуск получил сниженное число


@pytest.mark.parametrize("old,new,idx,opis", [
    ("pid=$np; n=$nn; top=0; lastlog=0", "n=$nn; top=0; lastlog=0", 0, "pid=$np убран: наблюдатель следит за убитым проходом"),
    ("pid=$np; n=$nn; top=0; lastlog=0", "pid=$np; n=$nn; lastlog=0", 3, "top=0 убран: максимум прошлого прохода (класс Д2 #752)"),
    ('echo "$nn" > "$P/logs/sobol_$per.procs"', ":", 1, "procs-файл не пишется"),
    ('echo "$np" > "$P/logs/sobol_$per.pid"', ":", 0, "pid-файл не обновляется"),
])
def test_mutatsii_perezapuska_lovyatsya(tmp_path, old, new, idx, opis):
    log, calls = _perezapusk(tmp_path, pamyat=_mut(old, new))
    assert _usloviya(tmp_path, log, calls)[idx] is False, opis


# ---------------------------------------------------------------- кэш оценщика
def test_kesh_dlinnaya_seriya_kazhdaya_tochka():
    from fastbt.etap1_eval_v2 import EvaluatorV2, NeSchitana

    class Z(EvaluatorV2):
        def __init__(self):
            self._sigs, self._pos, self.vhod, self.comm, self.fdr = {}, {}, [], 0.0004, "net"

        def evaluate(self, pt, period):
            self.vhod.append((len(self._sigs), len(self._pos)))
            self._sigs[pt["a"]], self._pos[pt["a"]] = {}, np.zeros(10)
            raise NeSchitana("z")

    ev = Z()
    for i in range(400):                                      # очистка «раз в K» с K < 400 даёт ненулевой вход в кэш
        ev.evaluate_v2({"a": i}, "train4")
    assert max(max(x) for x in ev.vhod) == 0


real = pytest.mark.skipif(os.environ.get("FASTBT_V6_REAL") != "1" or "FASTBT_MAIN" not in os.environ,
                          reason="реальные train4: FASTBT_V6_REAL=1 и FASTBT_MAIN")


class _BezOchistki(dict):
    def clear(self):                                          # прежнее поведение: кэш не сбрасывается
        pass


@real
def test_v5_tochka_s_neskolkimi_oknami_pobajtno_prezhnyaya():
    from fastbt import etap1, eval_v5
    from fastbt.eval_v5_osi import yakor
    inp, bp = etap1.build_inputs("S1", period="train4"), etap1.base_point("S1")
    inp.sk.tickers = list(inp.sk.tickers)[:3]
    sp = dict(ma_tip="sma", ma_window=20, dev_tip="std", z_entry=1.5, z_exit=0.5)
    pts = [{**yakor("S1"), **sp, "z_entry": z} for z in (1.5, 2.0, 1.5)]           # третья точка = первая: кэш был бы переиспользован
    def schet(ev):
        n = [0]
        f = ev._signals5
        ev._signals5 = lambda s: (n.__setitem__(0, n[0] + 1), f(s))[1]
        return n

    def snimok(r):
        return json.dumps({k: v for k, v in r.items() if k != "t_sek"}, sort_keys=True, default=str)

    e_new, e_old = eval_v5.EvaluatorV5("S1", inp, bp), eval_v5.EvaluatorV5("S1", inp, bp)
    e_old._sigs, e_old._pos = _BezOchistki(), _BezOchistki()
    n_new, n_old, n_fresh = schet(e_new), schet(e_old), 0
    for pt in pts:
        fresh = eval_v5.EvaluatorV5("S1", inp, bp)                 # эталон: пустой кэш на каждой точке
        n_f = schet(fresh)
        rn, xn, tn = e_new.evaluate_v2(pt, "train4")
        ro, xo, to = e_old.evaluate_v2(pt, "train4")
        rf, xf, tf = fresh.evaluate_v2(pt, "train4")
        assert rn["status"] == ro["status"] == rf["status"] == "ok"
        assert xn.tobytes() == xo.tobytes() == xf.tobytes() and tn.tobytes() == to.tobytes() == tf.tobytes()
        assert snimok(rn) == snimok(ro) == snimok(rf)
        n_fresh += n_f[0]
    assert n_new[0] == n_fresh > n_old[0] > 0                    # внутри точки кэш работает как у свежего; между точками его нет
