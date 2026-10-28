"""Шаг 3′ (D, B): мутации входов через боевой путь `s9_shag_db.points()` (не копии затворов в тесте).
Сдвиг σ / RVI на +1, RVI_d вместо RVI_{d−1} (и D6 с RVI_d в наборе точек) — каждая роняет проверку «будущего».
python3 -m pytest tests/test_s9_shag_db_mut.py -q
"""
import inspect
import sys
import textwrap
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "scripts" / "analysis")]

import s9_shag_common as C  # noqa: E402
import s9_shag_db as DB  # noqa: E402

from fastbt import s9  # noqa: E402

N, K0 = 700, 400


def _inputs(seed=5):
    rng = np.random.RandomState(seed)
    return np.exp(rng.normal(-4, 0.4, (2, N))), 18 + np.abs(rng.normal(0, 5, (2, N)))


def _eng(S, RV, ignore_lag=False):
    rvi = (lambda lag: RV) if ignore_lag else (lambda lag: np.c_[np.full((2, lag), np.nan), RV[:, :N - lag]])
    return types.SimpleNamespace(sigma=lambda s: S, rvi=rvi, n_t=np.array([N, N - 50]),
                                 calib_closed=lambda G: 0.3)


def _gates(S, RV, podhod="D", ignore_lag=False, points=None):
    """{семья: стек затворов} — из боевой DB.points (D: без D6; B: сетка cap × band из yaml)."""
    pts = (points or DB.points)(podhod, _eng(S, RV, ignore_lag), "x", s9)
    out = {}
    for f, _, G, *_ in pts:
        out.setdefault(f, []).append(G)
    return {f: np.stack(v) for f, v in out.items()}


def _peek(fn, S, RV, which):
    """Семьи, у которых затвор до дня k зависит от будущего (σ: [:k]; RVI_k до дня k включительно не виден)."""
    a = fn(S, RV)
    S2, RV2 = S.copy(), RV.copy()
    (S2 if which == "sigma" else RV2)[:, K0] *= 1.9
    b = fn(S2, RV2)
    cut = K0 if which == "sigma" else K0 + 1
    return {f for f in a if not np.array_equal(a[f][..., :cut], b[f][..., :cut], equal_nan=True)}


def _points_with_d6():
    """Мутация боевого points: D6 (RVI_d) не выкидывается из кандидатов — str.replace с одной заменой."""
    src = textwrap.dedent(inspect.getsource(DB.points))
    assert src.count('fams.pop("D6")') == 1
    ns = dict(vars(DB))
    exec(src.replace('fams.pop("D6")', "pass"), ns)
    return ns["points"]


def test_d_gates_no_future_and_shift_mutations_caught(monkeypatch):
    monkeypatch.setattr(s9, "V2", True)
    S, RV = _inputs()
    g = _gates(S, RV)
    assert {f: len(v) for f, v in g.items()} == dict(D1=9, D2=18, D4=27, D5=3, D6lag1=3, D7=6)   # D6 — не кандидат
    assert sum(len(v) for v in g.values()) == C.plan("D")["na_ryad"]
    assert _peek(_gates, S, RV, "sigma") == set() and _peek(_gates, S, RV, "rvi") == set()
    leaky = lambda s, r: _gates(np.roll(s, -1, axis=1), r)                       # noqa: E731  σ_{d+1}
    assert _peek(leaky, S, RV, "sigma") >= {"D1", "D2", "D4", "D5", "D7"}
    same_day = lambda s, r: _gates(s, r, ignore_lag=True)                        # noqa: E731  RVI_d
    assert "D6lag1" in _peek(same_day, S, RV, "rvi")
    leaky_r = lambda s, r: _gates(s, np.roll(r, -1, axis=1))                     # noqa: E731  RVI +1
    assert "D6lag1" in _peek(leaky_r, S, RV, "rvi")


def test_points_with_d6_in_candidates_is_caught(monkeypatch):
    monkeypatch.setattr(s9, "V2", True)
    S, RV = _inputs()
    bad = lambda s, r: _gates(s, r, points=_points_with_d6())                    # noqa: E731
    assert "D6" in bad(S, RV) and "D6" not in _gates(S, RV)
    assert "D6" in _peek(bad, S, RV, "rvi")                                       # RVI_d: сегодняшний RVI виден


def test_b_weights_no_future_and_shift_caught():
    S, RV = _inputs()
    b = lambda s, r: _gates(s, r, "B")                                            # noqa: E731
    assert set(b(S, RV)) == {"B"} and len(b(S, RV)["B"]) == C.plan("B")["na_ryad"]
    assert _peek(b, S, RV, "sigma") == set()
    assert _peek(lambda s, r: b(np.roll(s, -1, axis=1), r), S, RV, "sigma") == {"B"}
