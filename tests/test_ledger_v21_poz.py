"""Л7b: ledger 2.1 пишет |pos| (F_X/F_E) и сделки (F_R); okno(comm > 0) платит выход на конце окна (мутация № 6, #259).
Синтетика с известным ответом; сеть не нужна."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import fastbt.ledger_v2 as L
from fastbt import stats
from fastbt.etap1_eval_v2 import EvaluatorV2

DATY = np.arange(np.datetime64("2016-01-01"), np.datetime64("2020-01-01"))
DATY = DATY[np.is_busday(DATY)]
T = DATY.size
KOM = 0.002


def konec(t):
    return int((DATY < np.datetime64(f"{t}-01-01")).sum())


def ryad(seed, shape=()):
    """(net после комиссии, |pos|, lr, pos): позиция открыта на конце каждого окна — выход обязан быть оплачен."""
    g = np.random.default_rng(seed)
    pos = g.choice([-1.0, 0.0, 1.0], size=shape + (T,))
    for t in (2017, 2018, 2019):
        pos[..., konec(t) - 1] = 1.0
    lr = g.normal(0, 0.01, shape + (T,))
    net = pos * lr - np.abs(np.diff(pos, axis=-1, prepend=0.0)) * KOM
    return net, np.abs(pos), lr, pos


def zapisat(tmp, poz_flag, n=3):
    w = L.Pisatel21(tmp, "S1", 0, {"T": T, "kod": "K", "vhody": "V"}, shard=2, daty=DATY, poz=poz_flag)
    dan = []
    for i in range(n):
        p, ap, lr, pos = ryad(i)
        tk, atk, lrt, post = ryad(100 + i, (17,))
        sd = np.abs(np.diff(pos, prepend=0.0)) / 2
        std = np.abs(np.diff(post, axis=-1, prepend=0.0)) / 2
        row = {"v": L.V21, "klyuch_sha": f"s{i}", "status": "ok", "kod": "K", "vhody": "V",
               "by_year": {"2016": 1.0}, "by_quarter": {"2016Q1": 1.0}}
        w.add(row, p, tk, (ap, atk, sd, std)) if poz_flag else w.add(row, p, tk)
        dan.append((p, ap, lr, pos, tk, atk, lrt, post, sd))
    w.close()
    return dan


def test_okno_comm_ravno_pryamomu_raschetu(tmp_path):
    dan = zapisat(tmp_path, True)
    ch = L.Chitatel(tmp_path, "S1")
    assert ch.manifest["poz_konec"] is True
    for i, (_, _, lr, pos, _, _, lrt, post, _) in enumerate(dan):
        r = [x for x in ch.rows() if x["klyuch_sha"] == f"s{i}"]
        for t in (2017, 2018, 2019):
            pr = stats.sharpe_window(pos, lr, 0, konec(t), KOM)
            assert abs(ch.okno(t, r, comm=KOM)[0][0] - pr) <= 1e-12
            ptk = [stats.sharpe_window(post[k], lrt[k], 0, konec(t), KOM) for k in range(17)]
            assert np.max(np.abs(ch.okno(t, r, "tik", comm=KOM)[0] - ptk)) <= 1e-12


def test_mutaciya_6_vyhod_ne_oplachen_lovitsya(tmp_path):
    dan = zapisat(tmp_path / "a", True)
    ch = L.Chitatel(tmp_path / "a", "S1")
    nb = np.asarray(ch.manifest["nbars"], float)
    for i, d in enumerate(dan):
        pr = stats.sharpe_window(d[3], d[2], 0, konec(2018), KOM)
        s = np.stack([L.summy(DATY, d[0])])[0]                      # мутация: суммы без poz (F_X = 0)
        mut = stats.window(s, nb, L.segment_do_goda(2018), KOM)[0]
        assert abs(mut - pr) > 1e-6
        ok = stats.window(L.summy(DATY, d[0], d[1]), nb, L.segment_do_goda(2018), KOM)[0]
        assert abs(ok - pr) <= 1e-12


def test_f_r_i_f_e(tmp_path):
    dan = zapisat(tmp_path, True)
    ch = L.Chitatel(tmp_path, "S1")
    s = ch.summy()
    seg = L.segment_kvartala(DATY)
    for i, d in enumerate(dan):
        for k in range(16):
            m = seg == k
            assert s[i, k, stats.F_R] == d[8][m].sum() and s[i, k, stats.F_E] == (d[1][m] > 0).sum()


def test_obyom_zapisi_na_tochku(tmp_path):
    zapisat(tmp_path, True)
    na_tochku = sum(f.stat().st_size for f in (tmp_path / "S1").rglob("*.npy")) / 3
    print(f"запись на точку: {na_tochku / 1024:.1f} КБ")
    assert na_tochku <= 30_000


def test_bez_poz_okno_comm_otkaz_i_add_trebuet_poz(tmp_path):
    zapisat(tmp_path / "a", False)
    with pytest.raises(ValueError):
        L.Chitatel(tmp_path / "a", "S1").okno(2019, comm=KOM)
    w = L.Pisatel21(tmp_path / "b", "S1", 0, {"T": T, "kod": "K", "vhody": "V"}, daty=DATY, poz=True)
    row = {"v": L.V21, "klyuch_sha": "x", "status": "ok", "kod": "K", "vhody": "V"}
    with pytest.raises(ValueError):                                 # poz_konec есть, а позиций не дали
        w.add(row, np.zeros(T), np.zeros((17, T)))
    with pytest.raises(ValueError):                                 # смесь каталогов poz / без poz — отказ
        L.Pisatel21(tmp_path / "a", "S1", 1, {"T": T, "kod": "K", "vhody": "V"}, daty=DATY, poz=True)


def test_evaluator_poz_iz_expo():
    dates = pd.DatetimeIndex(["2015-12-30", "2016-01-04", "2016-01-05", "2016-01-06"])
    inv = np.array([0, 1, 2, 3, 2, 3])                              # тикер A: 4 даты, B: 2 даты
    rs = [SimpleNamespace(g=np.zeros(4), expo=np.array([1.0, 0.5, 0.0, 1.0]), cnt=np.array([2.0, 0.0, 2.0, 0.0])),
          SimpleNamespace(g=np.zeros(2), expo=np.array([0.5, 1.0]), cnt=np.array([2.0, 2.0]))]
    cnt = np.array([1.0, 1.0, 2.0, 2.0])
    me = SimpleNamespace(_last=(pd.DataFrame(index=dates), inv, SimpleNamespace(dates=dates, cnt=cnt), None, rs))
    port, tik, sd, tn = EvaluatorV2._poz(me, "train4")
    assert np.allclose(port, [0.5, 0.25, 1.0]) and np.allclose(tik[0], [0.5, 0.0, 1.0]) and np.allclose(tik[1], [0, 0.5, 1.0])
    assert np.allclose(sd, [0.0, 2.0, 1.0]) and np.allclose(tn[0], [0.0, 1.0, 0.0])
    assert port.shape == (3,) and tik.shape == (2, 3)
