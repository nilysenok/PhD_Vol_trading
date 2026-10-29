"""П6 (Ж-127): голосование ансамбля по позициям автомата v2 (`fastbt/kalibr_v5_golos.py`). Синтетика, без сети.
Известный ответ: v4-эквивалентный вход (дни из одного бара, без стопов и тейков) → позиция автомата = позиция слоя v4 = sig,
значит голос по `Result.pos` совпадает с `execlayer.vote` побайтно. Мутация: голос по позиции дня d+1 (mutations.md, сдвиг входа).
python3 -m pytest tests/test_kalibr_v5_golos_v2.py -q"""
import itertools

import numpy as np
import pytest

from fastbt import execlayer, grids, kalibr_v5 as K, kalibr_v5_golos as G
from fastbt.data import year_bounds
from fastbt.exec_e_types import Day, Params
from fastbt.exec_e_v4sloj import run_v4

DATY = np.array([d for d in np.arange("2016-01-01", "2020-01-01", dtype="datetime64[D]") if np.is_busday(d)])
N = DATY.size
S_T, E_T = year_bounds(DATY, 2018)
FILTRY = [dict(), dict(consec=2), dict(min_hold=3), dict(cooldown=2), dict(consec=2, min_hold=2, cooldown=1)]


def _days(n):
    return [Day(np.array([100.0]), np.array([101.0]), np.array([99.0]), np.array([100.0]), False) for _ in range(n)]


def _sigs(seed, k=10):
    return np.random.default_rng(seed).choice([-1.0, 0.0, 1.0], size=(k, N), p=[0.25, 0.4, 0.35])


def _poz_v2(sig, **flt):
    return run_v4("T", _days(N), sig, np.full(N, 0.01), Params(), **flt).pos


def _v4_pos(sig, **flt):
    return execlayer.exec_layer(sig.astype(float), np.zeros(N), flt.get("consec", 1), flt.get("min_hold", 0), flt.get("cooldown", 0), 0.0, N)


def _vote_ref(rows):
    out = np.zeros(N)
    execlayer.vote(np.ascontiguousarray(np.stack(rows)), len(rows), S_T, E_T, out, grids.VOTE_FRAC)
    return out[S_T:E_T]


def _golos(sigs, **flt):
    rows = {f"m{i}": _poz_v2(s, **flt) for i, s in enumerate(sigs)}
    return G.golos_v2(lambda kl, k: rows[kl], [dict(kl=kl) for kl in rows], 0, N, S_T, E_T)


@pytest.mark.parametrize("flt,seed", list(itertools.product(FILTRY, range(3))))
def test_golos_v2_ravno_vote_v4_pobajtno(flt, seed):
    sigs = _sigs(seed)
    ref = _vote_ref([_v4_pos(s, **flt) for s in sigs])
    assert (np.abs(ref) > 0).any()
    assert _golos(sigs, **flt).tobytes() == ref.tobytes()


def test_golos_dolya_vote_frac():
    """Ровно половина членов в позиции — голоса нет (порог строгий: > VOTE_FRAC·k); на одного больше — есть."""
    s = np.zeros((10, N))
    s[:5, S_T:E_T] = 1.0
    assert not _golos(s)[:50].any()
    s[5, S_T:E_T] = 1.0
    assert (_golos(s)[:50] == 1.0).all()


def test_pustoj_ansambl_none_i_nevernaya_dlina():
    assert G.golos_v2(lambda kl, k: None, [], 0, N, S_T, E_T) is None
    with pytest.raises(ValueError):
        G.golos_v2(lambda kl, k: np.zeros(N - 1), [dict(kl="a")], 0, N, S_T, E_T)


def _pul():
    osi = {"a": list(range(10)), "b": list(range(5))}
    rng = np.random.default_rng(1)
    tochki = [dict(zip(osi, v)) for v in itertools.product(*osi.values())]
    ex = 0.01 * rng.standard_normal((len(tochki), 1, N)) + 0.0004 * np.array([[[p["a"] / 10 + 1]] for p in tochki])
    return K.Pul("S1", DATY, osi, tochki, ex, rng.choice([-1.0, 0.0, 1.0], size=ex.shape))


def test_god_s_poz_fn_ravno_god_v4_pobajtno():
    p = _pul()
    ref = K.god(p, 2018, "2016-01-01")["tikery"][0]
    poz_fn = lambda kl, k: _poz_v2(p.poz[p.idx[kl], k])         # noqa: E731
    got = K.god(p, 2018, "2016-01-01", poz_fn)["tikery"][0]
    assert got["chleny"] == ref["chleny"] and len(ref["chleny"]) == grids.ENS_K
    assert (np.abs(ref["poz"]) > 0).any() and got["poz"].tobytes() == ref["poz"].tobytes()


def test_mutaciya_golos_po_pozicii_dnya_d_plus_1_lovitsya(monkeypatch):
    sigs = _sigs(0)
    ref = _vote_ref([_v4_pos(s) for s in sigs])
    assert _golos(sigs).tobytes() == ref.tobytes()

    def vote_d1(stack, k, s, e, out, frac=0.5):
        execlayer.vote(np.ascontiguousarray(np.roll(stack, -1, axis=1)), k, s, e, out, frac)
    monkeypatch.setattr(G, "vote", vote_d1)
    assert _golos(sigs).tobytes() != ref.tobytes()
