"""Детерминизм sha источников v7 (V7-8b, ошибка слитого #900): `Ist7.sha` хэшировал `tobytes()` массива dtype=object (адреса объектов,
а не значения) — ключ точки S8–S12 менялся между процессами. Тесты: два процесса с разным PYTHONHASHSEED дают один sha; равные по значению,
но разные по адресам массивы — один sha; подмена значения оси при том же ключе — промах кэша (Д5); мутация «tobytes object» падает."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastbt import eval_v7 as E
from fastbt.signals.v7_kross import Panel

ROOT = Path(__file__).resolve().parents[1]
SKRIPT = """
import numpy as np, pandas as pd
from fastbt import eval_v7 as E
from fastbt.signals.v7_kross import Panel
d = pd.bdate_range("2018-01-09", periods=6)
c = pd.DataFrame(np.arange(12.0).reshape(6, 2) + 10, index=d, columns=["AAA", "BBB"])
metki = np.array([["MOEX" + "FN", "MOEX" + "OG"]] * 6, dtype=object)
pn = Panel(daty=d.values, tikery=["AAA", "BBB"], c=c, imoex=pd.Series(np.arange(6.0), index=d), metki=metki,
           sigma=np.full((6, 2), 0.01), ok=np.ones((6, 2), bool))
b = pd.DataFrame({"timestamp": d, "open": 1.0, "close": 2.0, "t": ["x" + str(i) for i in range(6)]})
print(E.Ist7(b10={"AAA": b}, panel=pn).sha, E.Ist7(b10={"AAA": b}).sha, E.Ist7(panel=pn).sha)
"""


def _sha_v_processe(seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-c", SKRIPT], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_sha_odinakov_v_raznyh_processah():
    """G0/G1: S8 (панель), S10 (10-минутки) и обе вместе — три sha, одинаковые при PYTHONHASHSEED 1 / 2 / 3 (адреса и хэш-соль разные)."""
    a, b, c = _sha_v_processe("1"), _sha_v_processe("2"), _sha_v_processe("3")
    assert a == b == c and len(a.split()) == 3 and all(len(x) == 64 for x in a.split())


D = pd.bdate_range("2018-01-09", periods=6)


def _panel(metki) -> Panel:
    c = pd.DataFrame(np.arange(12.0).reshape(6, 2) + 10, index=D, columns=["AAA", "BBB"])
    return Panel(daty=D.values, tikery=["AAA", "BBB"], c=c, imoex=pd.Series(np.arange(6.0), index=D), metki=metki,
                 sigma=np.full((6, 2), 0.01), ok=np.ones((6, 2), bool))


def _metki(a: str = "FN", b: str = "OG") -> np.ndarray:
    """Новые объекты-строки на каждый вызов (адреса разные, значения равны)."""
    m = np.empty((6, 2), dtype=object)
    for i in range(6):
        m[i, 0], m[i, 1] = "MOEX" + "".join(a), "MOEX" + "".join(b)
    return m


def _sha_tobytes_object(ist: E.Ist7) -> str:
    """Мутант: прежний код — tobytes() массива object (адреса)."""
    import hashlib
    h = hashlib.sha256()
    for a in (ist.panel.metki, ist.panel.sigma, ist.panel.ok):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def test_sha_ravnye_znacheniya_raznye_adresa_odin_sha():
    a, b = E.Ist7(panel=_panel(_metki())), E.Ist7(panel=_panel(_metki()))
    assert a.panel.metki[0, 0] is not b.panel.metki[0, 0] and a.panel.metki[0, 0] == b.panel.metki[0, 0]
    assert a.sha == b.sha


def test_mutaciya_tobytes_object_lovitsya():
    """G2: мутант (tobytes object) даёт разный хэш у равных по значению массивов — тест выше на нём падает."""
    a, b = E.Ist7(panel=_panel(_metki())), E.Ist7(panel=_panel(_metki()))
    assert _sha_tobytes_object(a) != _sha_tobytes_object(b)
    assert a.sha == b.sha


@pytest.mark.parametrize("sid", ["S8", "S9", "S10", "S11", "S12"])
def test_podmena_znacheniya_osi_menyaet_klyuch(sid):
    """G3 (Д5, №7): подмена значения сектора, σ, допуска или 10-минутки при том же sid меняет ключ точки `_k7`."""
    b = pd.DataFrame({"timestamp": D, "close": np.arange(6.0)})

    def klyuch(ist) -> tuple:
        ev = object.__new__(E.EvaluatorV7)
        ev.sid, ev.ist7 = sid, ist
        return ev._k7()
    base = klyuch(E.Ist7(b10={"AAA": b}, panel=_panel(_metki())))
    assert base == klyuch(E.Ist7(b10={"AAA": b.copy()}, panel=_panel(_metki())))
    assert klyuch(E.Ist7(b10={"AAA": b}, panel=_panel(_metki(b="TN")))) != base
    b2 = b.copy()
    b2.loc[3, "close"] += 1.0
    assert klyuch(E.Ist7(b10={"AAA": b2}, panel=_panel(_metki()))) != base
    p = _panel(_metki())
    p.sigma[2, 1] = 0.02
    assert klyuch(E.Ist7(b10={"AAA": b}, panel=p)) != base
    p = _panel(_metki())
    p.ok[4, 0] = False
    assert klyuch(E.Ist7(b10={"AAA": b}, panel=p)) != base


@pytest.mark.parametrize("sid", ["S1", "S2", "S3", "S4", "S5", "S6"])
def test_s1_s6_klyuch_bez_suffiksa_v7(sid):
    """S1–S6 в v7 — путь v6 побайтно: суффикса v7 в ключе нет (sha источников v7 не вмешивается)."""
    ev = object.__new__(E.EvaluatorV7)
    ev.sid, ev.ist7 = sid, E.Ist7(panel=_panel(_metki()))
    assert ev._k7() == ()
