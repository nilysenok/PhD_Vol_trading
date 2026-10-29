"""К-wf6, лист 1: ядро CSCV по кварталам wf (fastbt/pbo_wf.py). Синтетика, без данных и сети."""
import numpy as np
import pytest

from fastbt import pbo_wf as W

ND = 63


def sums(x):
    """x[P, 45, ND] → (S, Q2, nb) по кварталам."""
    return x.sum(axis=2), (x ** 2).sum(axis=2), np.full(x.shape[1], float(x.shape[2]))


def shum(P=300, seed=1, edge=None):
    x = np.random.default_rng(seed).normal(0.0, 0.01, (P, W.NQ, ND))
    if edge is not None:
        x[0] += edge                        # точка 0 — стабильное преимущество во всех кварталах
    return x


def test_bloki_45_kvartalov_10_blokov_celye_kvartaly():
    g = W.granicy()
    assert len(g) == 10 and g[0][0] == 0 and g[-1][1] == 45
    assert all(a2 == b1 for (_, a2), (b1, _) in zip(g, g[1:]))
    S, Q2, nb = sums(shum(P=3))
    Sb, Qb, nbb = W.bloki(S, Q2, nb)
    assert Sb.shape == (3, 10) and nbb.sum() == 45 * ND
    assert np.allclose(Sb.sum(axis=1), S.sum(axis=1))


@pytest.mark.parametrize("nq", [44, 46, 40])
def test_bloki_ne_45_kvartalov_otkaz(nq):
    x = np.zeros((3, nq, ND))
    with pytest.raises(ValueError):
        W.bloki(*sums(x))


def test_bloki_shag_granicy_mutatsiya_ne_celye_kvartaly_lovitsya():
    """Мутация: блоки по 4,5 квартала (45/10 дробное) — сумма блока не равна сумме целых кварталов."""
    S, Q2, nb = sums(shum(P=2))
    Sb, _, _ = W.bloki(S, Q2, nb)
    for (a, b), col in zip(W.granicy(), Sb.T):
        assert np.allclose(col, S[:, a:b].sum(axis=1))
    assert not np.allclose(Sb[:, 0], S[:, :4].sum(axis=1))


def test_razbieniya_252_i_dopolnyayut():
    rs = list(W.razbieniya())
    assert len(rs) == 252
    for tr, te in rs:
        assert len(tr) == len(te) == 5 and set(tr) | set(te) == set(range(10)) and not set(tr) & set(te)


def test_sharpe_iz_summ_raven_pryamomu():
    x = shum(P=4)[:, :, :]
    S, Q2, nb = sums(x)
    flat = x.reshape(4, -1)
    ref = flat.mean(axis=1) / flat.std(axis=1, ddof=1) * np.sqrt(252)
    assert np.allclose(W.sharpe_iz_summ(S.sum(1), Q2.sum(1), nb.sum() * 1.0 * 1), ref, atol=1e-9)


def test_vybor_vidit_tolko_in_sample():
    """Мутация «выбор видит всё»: Шарп, переданный выбору, считается только по блокам in-sample."""
    x = shum(P=20)
    S, Q2, nb = sums(x)
    Sb, Qb, nbb = W.bloki(S, Q2, nb)
    seen = []

    def spy(sh_is, tr):
        ref = W.sharpe_iz_summ(Sb[:, list(tr)].sum(1), Qb[:, list(tr)].sum(1), nbb[list(tr)].sum())
        seen.append(np.allclose(sh_is, ref))
        return [0]
    W.cscv_wf(Sb, Qb, nbb, spy)
    assert len(seen) == 252 and all(seen)


def test_shum_pbo_okolo_polovinu():
    r = W.pbo_wf(*sums(shum()), W.vybor_argmax)
    assert r["n"] == 252 and abs(r["pbo"] - 0.5) <= 0.12


def test_signal_pbo_okolo_nulya():
    r = W.pbo_wf(*sums(shum(edge=0.002)), W.vybor_argmax)
    assert r["pbo"] <= 0.05


def test_gran_odna_tochka_nan():
    assert np.isnan(W.pbo_wf(*sums(shum(P=1)), W.vybor_argmax)["pbo"])


def test_gran_pustaya_lenta_i_nulevoy_obyom_otkaz():
    with pytest.raises(ValueError):
        W.bloki(np.zeros((3, 0)), np.zeros((3, 0)), np.zeros(0))
    S, Q2, nb = sums(shum(P=3))
    nb[:5] = 0                                         # блок 0 без баров
    with pytest.raises(ValueError):
        W.bloki(S, Q2, nb)


def test_gran_nan_v_summah_otkaz():
    S, Q2, nb = sums(shum(P=3))
    S[1, 7] = np.nan
    with pytest.raises(ValueError):
        W.bloki(S, Q2, nb)


def test_gran_vybor_none_propusk():
    r = W.pbo_wf(*sums(shum(P=5)), lambda sh, tr: None)
    assert r["n"] == 0 and r["n_propusk"] == 252 and np.isnan(r["pbo"])
