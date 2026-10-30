"""Тестировщик #826: переменная-страж ensure (pid), finally, порог 2022, два процесса. Синтетика, без сети."""
from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from fastbt import daily_1850 as D

REPO = Path(__file__).resolve().parents[1]
NAME = "ohlcv_daily_1850.parquet"
GUARD = f"_FASTBT_ENSURE_{NAME}"


def _ryad(tag: float, d: str) -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["X"], "date": pd.to_datetime([d]), "open": [tag], "high": [tag + 1],
                         "low": [tag], "close": [tag], "volume": [5]})


def test_isklyuchenie_v_build_snimaet_peremennuyu(tmp_path, monkeypatch):
    def bad(main=None):
        assert os.environ.get(GUARD) == str(os.getpid())
        raise ValueError("boom")
    monkeypatch.setattr(D, "build", bad)
    with pytest.raises(ValueError):
        D.ensure(tmp_path / NAME, tmp_path)
    assert GUARD not in os.environ


def test_stoit_tolko_vo_vremya_build(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(D, "build", lambda main=None: (seen.append(os.environ.get(GUARD)), _ryad(1.0, "2021-02-26"))[1])
    D.ensure(tmp_path / NAME, tmp_path)
    assert seen == [str(os.getpid())] and GUARD not in os.environ


def test_vlozhennyj_vyzov_togo_zhe_processa_propuskaetsya(tmp_path, monkeypatch):
    n = []
    p = tmp_path / NAME

    def build(main=None):
        D.ensure(p, tmp_path)                       # вложенный: не должен зависнуть
        n.append(1)
        return _ryad(1.0, "2021-02-26")
    monkeypatch.setattr(D, "build", build)
    D.ensure(p, tmp_path)
    assert n == [1]


_SCRIPT = """
import sys, time, pandas as pd
from pathlib import Path
from fastbt import daily_1850 as D
p, log, pause = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
def build(main=None):
    with open(log, "a") as f: f.write("b\\n")
    time.sleep(pause)
    return pd.DataFrame({"ticker": ["X"], "date": pd.to_datetime(["2021-02-26"]), "open": [1.0], "high": [2.0],
                         "low": [1.0], "close": [1.0], "volume": [5]})
D.build = build
D.ensure(p, p.parent)
"""


def _run(tmp_path, env_extra, n=1, pause=0.0):
    sc = tmp_path / "s.py"
    sc.write_text(_SCRIPT)
    env = {**os.environ, **env_extra, "PYTHONPATH": str(REPO)}
    log = tmp_path / "log.txt"
    ps = [subprocess.Popen([sys.executable, str(sc), str(tmp_path / NAME), str(log), str(pause)], cwd=REPO, env=env)
          for _ in range(n)]
    try:
        for q in ps:
            assert q.wait(timeout=60) == 0
    finally:
        for q in ps:
            if q.poll() is None:
                q.kill()
    return log.read_text().split() if log.exists() else []


def test_subprocess_nasleduet_peremennuyu_no_ne_propuskaet(tmp_path):
    assert _run(tmp_path, {GUARD: str(os.getpid())}) == ["b"]      # pid родителя ≠ pid ребёнка: ряд собран


def test_dva_processa_odin_stroit(tmp_path):
    assert _run(tmp_path, {}, n=2, pause=1.5) == ["b"]
    assert (tmp_path / NAME).exists() and D.is_ready(tmp_path / NAME)


def _child(p, q):
    import fastbt.daily_1850 as DD
    DD.build = lambda main=None: (q.put("b"), _ryad(1.0, "2021-02-26"))[1]
    DD.ensure(Path(p), Path(p).parent)


def test_multiprocessing_fork_rebenok_ne_propuskaet(tmp_path, monkeypatch):
    method = "fork"          # spawn — тот же свежий интерпретатор, что и subprocess-тест выше
    monkeypatch.setenv(GUARD, str(os.getpid()))
    ctx = mp.get_context(method)
    q = ctx.Queue()
    pr = ctx.Process(target=_child, args=(str(tmp_path / NAME), q))
    pr.start()
    pr.join(60)
    if pr.is_alive():
        pr.kill()
        pytest.fail("зависание")
    assert pr.exitcode == 0 and q.get(timeout=5) == "b"


def test_wf_rovno_2022_01_03_bez_markera_otkaz(tmp_path, monkeypatch):
    p = tmp_path / NAME
    _ryad(2.0, "2022-01-03").to_parquet(p, index=False)
    monkeypatch.setattr(D, "build", lambda main=None: pytest.fail("пересобрал"))
    with pytest.raises(RuntimeError, match="маркера нет"):
        D.ensure(p, tmp_path)


def test_dev_2021_12_31_bez_markera_pereberaetsya(tmp_path, monkeypatch):
    p = tmp_path / NAME
    _ryad(2.0, "2021-12-31").to_parquet(p, index=False)
    monkeypatch.setattr(D, "build", lambda main=None: _ryad(9.0, "2021-12-31"))
    D.ensure(p, tmp_path)
    assert pd.read_parquet(p)["open"].iloc[0] == 9.0
