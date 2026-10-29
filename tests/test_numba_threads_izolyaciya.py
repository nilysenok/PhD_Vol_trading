"""Б3 (29.10): NUMBA_NUM_THREADS не остаётся изменённым после вызова (mutations.md № 43).

Причина: numba после запуска потоков отказывает на любой новой компиляции, если
NUMBA_NUM_THREADS в окружении изменился (RuntimeError: Cannot set NUMBA_NUM_THREADS).
s9._job, s9.build_cache и pipeline.run_all ставили "1" в процессе и не возвращали.
python3 -m pytest tests/test_numba_threads_izolyaciya.py -q
"""
import multiprocessing
import subprocess
import sys
import textwrap

import pytest

from fastbt import dev, pipeline, s9

VAR = "NUMBA_NUM_THREADS"


class _Stop(Exception):
    pass


def _boom(*a, **k):
    raise _Stop


@pytest.fixture(params=["4", None], ids=["bylo-4", "ne-zadano"])
def prezhnee(request, monkeypatch):
    if request.param is None:
        monkeypatch.delenv(VAR, raising=False)
    else:
        monkeypatch.setenv(VAR, request.param)
    return request.param


def _proveryaem(prezhnee):
    import os
    assert os.environ.get(VAR) == prezhnee


def test_s9_job_vozvrashchaet_peremennuyu(prezhnee, monkeypatch):
    monkeypatch.setattr(dev, "load_design", _boom)     # падаем сразу после установки "1"
    with pytest.raises(_Stop):
        s9._job(("NO_SUCH_TICKER", {}))
    _proveryaem(prezhnee)


def test_s9_build_cache_vozvrashchaet_peremennuyu(prezhnee, monkeypatch):
    monkeypatch.setattr(s9, "PERIOD", "train4")
    monkeypatch.setattr(multiprocessing, "get_context", _boom)
    with pytest.raises(_Stop):
        s9.build_cache(workers=2)
    _proveryaem(prezhnee)


def test_pipeline_run_all_vozvrashchaet_peremennuyu(prezhnee, monkeypatch):
    monkeypatch.setattr(multiprocessing, "get_context", _boom)
    with pytest.raises(_Stop):
        pipeline.run_all(None, tickers=["SBER"], workers=2, log=lambda *_: None)
    _proveryaem(prezhnee)


def test_numba_ne_ronyaet_kompilyatsiyu_posle_s9_job():
    """Сквозное воспроизведение: запущенные потоки + _job -> следующая компиляция не падает."""
    code = textwrap.dedent("""
        import os, numpy as np
        os.environ["NUMBA_NUM_THREADS"] = "4"
        from numba import njit, prange
        from fastbt import dev, s9
        def boom(*a, **k):
            raise RuntimeError("stop")
        dev.load_design = boom
        @njit(parallel=True)
        def f(a):
            s = 0.0
            for i in prange(a.size):
                s += a[i]
            return s
        f(np.ones(10))
        try:
            s9._job(("NO_SUCH_TICKER", {}))
        except Exception:
            pass
        @njit(parallel=True)
        def g(a):
            s = 0.0
            for i in prange(a.size):
                s += a[i]
            return s
        print(g(np.ones(10)))
    """)
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    assert r.stdout.strip() == "10.0"
