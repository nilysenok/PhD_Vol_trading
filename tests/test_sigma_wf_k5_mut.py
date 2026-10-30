"""К-5, тестировщик PR #805: мутации, не пойманные tests/test_sigma_wf_k5.py (seed, контроль R1 в обоих режимах, G1', потолок процессов).
Модуль vyr_etalony_finaly подменён заглушкой (синтетика, ни данных, ни сети).
python3 -m pytest tests/test_sigma_wf_k5_mut.py -q"""
import importlib.util
import subprocess
import types
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("wk5m", Path(__file__).resolve().parents[1] / "scripts" / "dannye" / "wf_sigma_910_progon.py")
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)


def _stub(tmp_path, calls):
    def stop(*a, **k):
        raise SystemExit("контроль R1 не пройден")

    def setup(d0, *a, **k):
        calls.append(("setup", d0))
        return "setup-ok"
    return types.SimpleNamespace(SEEDS=None, WORK=tmp_path / "w", control=stop, control_q=stop, f5_setup=setup,
                                 cmd_f1=lambda ns: calls.append(("cmd_f1", ns.workers)), cmd_f5=lambda ns: calls.append(("cmd_f5", ns.workers)))


@pytest.mark.parametrize("n", [1, 5, 10])
def test_seed_peredayotsya_v_vyr(tmp_path, monkeypatch, n):
    calls, V = [], None
    V = _stub(tmp_path, calls)
    monkeypatch.setattr(W, "_vyr", lambda out: V)
    W.f_v_obvyazke(tmp_path, "f1", False, n)
    assert V.SEEDS == tuple(range(42, 42 + n)) and V.WORK.exists() and ("cmd_f1", 3) in calls


def test_novye_matricy_kontrol_myagkij_oba(tmp_path, monkeypatch):
    calls = []
    V = _stub(tmp_path, calls)
    monkeypatch.setattr(W, "_vyr", lambda out: V)
    W.f_v_obvyazke(tmp_path, "f5", False, 2)
    assert V.control("x") is None and V.control_q("x") is None  # оба обёрнуты


def test_staraya_kontrol_ne_myagkij_i_d0_ostanovlen(tmp_path, monkeypatch):
    """G1': контроль R1 на старой матрице обязан остановить при расхождении (не глотается); d0=True — стоп; d0=False проходит."""
    calls = []
    V = _stub(tmp_path, calls)
    monkeypatch.setattr(W, "_vyr", lambda out: V)
    W.f_v_obvyazke(tmp_path, "f5", True, 2)
    with pytest.raises(SystemExit):
        V.control("x")
    with pytest.raises(SystemExit):
        V.control_q("x")
    with pytest.raises(SystemExit, match="d0"):
        V.f5_setup(True)
    assert V.f5_setup(False) == "setup-ok"


def test_zapusk_potolok_processov_i_koren(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(subprocess, "run", lambda cmd, env=None: seen.update(cmd=cmd, env=env) or types.SimpleNamespace(returncode=0))
    assert W.zapusk(tmp_path, "f1", workers=10) == 0
    assert seen["cmd"][-1] == "3" and seen["env"]["FASTBT_MAIN"] == str(tmp_path) and seen["env"]["NUMBA_NUM_THREADS"] == "4"
