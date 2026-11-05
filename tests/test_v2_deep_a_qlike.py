"""A4: QLIKE модуля = формула владельца ln f + y/f с точностью до функции y (Д7)."""
import numpy as np

from models_v2 import metrics
from models_v2.s5 import common


def owner(y, f):
    return np.log(f) + y / f


def test_raznosti_poter_ravny_formule_vladelca():
    rng = np.random.default_rng(1)
    y = np.exp(rng.normal(size=500))
    f1, f2 = np.exp(rng.normal(size=500)), np.exp(rng.normal(size=500))
    a = metrics.qlike(y, f1) - metrics.qlike(y, f2)
    b = owner(y, f1) - owner(y, f2)
    assert np.abs(a - b).max() <= 1e-12


def test_mut_sigma_vmesto_sigma2_lovitsya():
    rng = np.random.default_rng(2)
    y = np.exp(rng.normal(size=500))
    f1, f2 = np.exp(rng.normal(size=500)), np.exp(rng.normal(size=500))
    a = metrics.qlike(y, np.sqrt(f1)) - metrics.qlike(y, np.sqrt(f2))
    b = owner(y, f1) - owner(y, f2)
    assert np.abs(a - b).max() > 1e-3


def test_mut_y_na_f_naoborot_lovitsya():
    rng = np.random.default_rng(3)
    y = np.exp(rng.normal(size=500))
    f1, f2 = np.exp(rng.normal(size=500)), np.exp(rng.normal(size=500))
    wrong = (np.log(f1) + f1 / y) - (np.log(f2) + f2 / y)
    assert np.abs(wrong - (owner(y, f1) - owner(y, f2))).max() > 1e-3


def test_real_kontroli_srednyaya_raznost_po_dvum_formulam():
    c = common.controls()
    a = (metrics.qlike(c.rv, c.s2_b2) - metrics.qlike(c.rv, c.s2_harj)).mean()
    b = (owner(c.rv, c.s2_b2) - owner(c.rv, c.s2_harj)).mean()
    assert abs(a - b) <= 1e-12
