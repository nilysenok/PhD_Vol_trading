"""К-2: досчёт членов калибратора v5 (`fastbt/kalibr_v5_doschet.py`) на синтетике, без сети и без данных.
Известный ответ: фальшивый оценщик со знаковыми позициями S[17, T] — poz_znak обязан вернуть sign(S), голос — `execlayer.vote` на них.
Мутации (mutations.md № 24/26 по смыслу — запись в ledger и N; сдвиг входа): сосед не пишется в ledger; знак без short; голос по d+1.
python3 -m pytest tests/test_kalibr_v5_doschet.py -q"""
import json
import types
import zlib
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

from fastbt import execlayer, grids, kalibr_v5 as K, kalibr_v5_doschet as D, kalibr_v5_golos as G, ledger_v2 as lm, progon_sobol as PS
from fastbt.data import year_bounds

DATY = np.array([d for d in np.arange("2016-01-01", "2020-01-01", dtype="datetime64[D]") if np.is_busday(d)])
T, KOD, VHODY = DATY.size, "kod1", "vh1"
Sreda = PS.Sreda
Row = namedtuple("Row", "g expo short")


class Evr:
    """Фальшивый оценщик: ряд и знаковые позиции — функции точки (crc32); интерфейс evaluate_v2 / _last / poz как у EvaluatorV2."""
    fdr = None

    def __init__(self):
        self.n = 0

    def evaluate_v2(self, pt, period, vozm=None):
        self.n += 1
        g = np.random.default_rng(zlib.crc32(json.dumps({k: str(v) for k, v in pt.items()}, sort_keys=True).encode()))
        mu = 0.002 * (5 - abs(float(pt["a"]) - 2.5) - abs(float(pt["b"]) - 2.5))
        ex = mu + 0.01 * g.standard_normal((17, T))
        self.S = g.choice([-1.0, 0.0, 1.0], size=(17, T))
        expo, short = 0.5 * np.abs(self.S), 0.5 * (self.S < 0)
        df = pd.DataFrame(index=pd.DatetimeIndex(DATY))
        self._last = df, np.tile(np.arange(T), 17), None, None, [Row(np.zeros(T), expo[i], short[i]) for i in range(17)]
        self.poz = (np.abs(self.S).mean(0), np.abs(self.S), np.zeros(T), np.zeros((17, T)))
        row = {"status": "ok", "prichina": None, "by_quarter": {}, "mediana_kv": 0.0}
        return row, ex.mean(0), ex


def sreda(evr=None, daty=DATY):
    return Sreda(evr or Evr(), KOD, VHODY, daty, [f"T{i}" for i in range(17)])


RZ = PS.R0
GRID = {"a": [float(i) for i in range(6)], "b": [float(i) for i in range(6)]}
DEKOD = {a: {lm.tok(v): v for v in vals} for a, vals in GRID.items()}
DIRA = None


def baza(root, v=lm.V, dyry=((2.0, 0.0),)):
    """Ledger с сеткой 6×6 без точек dyry; manifest с kod/vhody оценщика."""
    sr = sreda()
    mf = {"period": "train4", "kod": KOD, "vhody": VHODY, "kod_slit": False, "T": T}
    w = lm.Pisatel21(root, "S1", 0, mf, daty=DATY, poz=True) if v == lm.V21 else lm.Pisatel(root, "S1", 0, mf)
    for a in GRID["a"]:
        for b in GRID["b"]:
            if (a, b) in dyry:
                continue
            r, ex_p, ex_t = PS.schitat_tochku(sr, "S1", 0, {"a": a, "b": b}, "test", False, False, v, RZ)
            w.add(r, ex_p, ex_t, sr.evr.poz) if v == lm.V21 else w.add(r, ex_p, ex_t)
    w.close()
    np.save(root / "S1" / "daty.npy", DATY)
    return root


def pul(root, dc=None):
    p = K.pul_iz_ledger(root, "S1", GRID)
    p.dosschitat = None if dc is None else dc.dosschitat
    return p


def stroki(root):
    return sum(len(f.read_text().splitlines()) for f in (root / "S1").glob("p*/rows.jsonl"))


def sosed_dolzhen_zapisatsya(tmp_path, dc_cls=D.Doschet):
    root = baza(tmp_path)
    n0, str0 = len(lm.Chitatel(root, "S1").rows()), stroki(root)
    dc = dc_cls(sreda(), "S1", root, RZ, DEKOD)
    p = pul(root, dc)
    kl = K.klyuch("S1", {"a": "2.0", "b": "0.0"})
    r = p.ryad(kl, {"a": "2.0", "b": "0.0"}, 3)
    dc.dosschitat({"a": "2.0", "b": "0.0"})                      # повтор той же точки: строка не дублируется
    dc.zakryt()
    assert r is not None and p.n_proby() == n0 + 1                # сосед в N
    assert len(lm.Chitatel(root, "S1").rows()) == n0 + 1 and stroki(root) == str0 + 1   # в ledger — одна строка


def test_sosedi_pishutsya_v_ledger_i_N_odin_raz_v20(tmp_path):
    sosed_dolzhen_zapisatsya(tmp_path)


def test_sosedi_ryad_v20_v_tochnosti_float32(tmp_path):
    root = baza(tmp_path)
    dc = D.Doschet(sreda(), "S1", root, RZ, DEKOD)
    ex, poz = dc.dosschitat({"a": "2.0", "b": "0.0"})
    assert poz is None and ex.shape == (17, T) and (ex == ex.astype(np.float32).astype(float)).all()


def test_sosedi_ledger_v21_pishet_summy_i_N(tmp_path):
    root = baza(tmp_path, lm.V21)
    ch0 = lm.Chitatel(root, "S1")
    n0, dc = len(ch0.rows()), D.Doschet(sreda(), "S1", root, RZ, DEKOD)
    ex, _ = dc.dosschitat({"a": "2.0", "b": "0.0"})
    dc.zakryt()
    ch = lm.Chitatel(root, "S1")
    new = [r for r in ch.rows() if r["istochnik"] == D.ISTOCHNIK]
    assert len(new) == 1 and len(ch.rows()) == n0 + 1
    assert np.allclose(ch.summy(new, "tik")[0], lm.summy(DATY, ex, dc.sr.evr.poz[1], dc.sr.evr.poz[3]), atol=1e-12)        # суммы 2.1 = суммы ряда


def test_sosedi_v_kalibrovat_vhodyat_v_N(tmp_path):
    root = baza(tmp_path)
    n0, dc = len(lm.Chitatel(root, "S1").rows()), D.Doschet(sreda(), "S1", root, RZ, DEKOD)
    p = pul(root, dc)
    res = K.kalibrovat(p, "train4")
    dc.zakryt()
    assert p.n_proby() == n0 + len(p.novye) and len(p.novye) >= 1       # соседи без точки (2.0, 0.0) досчитаны
    assert len(lm.Chitatel(root, "S1").rows()) == p.n_proby()           # и N = строки ledger
    assert res[2018]["n_proby"] == p.n_proby()


def test_mutaciya_sosed_ne_pishetsya_v_ledger_lovitsya(tmp_path):
    class Bez(D.Doschet):
        def dosschitat(self, pt):
            ex, _ = super().dosschitat(pt)
            self.w.buf.clear()
            return ex, None
    with pytest.raises(AssertionError):
        sosed_dolzhen_zapisatsya(tmp_path, Bez)


def test_znak_poz_ravno_sign_S():
    evr, sr = Evr(), None
    evr.evaluate_v2({"a": 1.0, "b": 1.0}, "train4")
    assert (D.poz_znak(evr, "train4") == np.sign(evr.S)).all() and (D.poz_znak(evr, "train4") == -1).any()


def test_mutaciya_znak_bez_short_lovitsya():
    evr = Evr()
    evr.evaluate_v2({"a": 1.0, "b": 1.0}, "train4")
    bez = types.SimpleNamespace(_last=(*evr._last[:4], [Row(r.g, r.expo, None) for r in evr._last[4]]))
    assert not (D.poz_znak(bez, "train4") == np.sign(evr.S)).all()


def test_znak_den_vne_goda_i_net_stroki_dayut_nol():
    evr = Evr()
    evr.evaluate_v2({"a": 1.0, "b": 1.0}, "train4")
    assert D.poz_znak(evr, "tune2018").shape == (17, int((DATY.astype("datetime64[Y]").astype(int) + 1970 == 2018).sum()))


def test_golos_ravno_vote_na_binarnyh_pozitsiyah(tmp_path):
    root = baza(tmp_path)
    dc = D.Doschet(sreda(), "S1", root, RZ, DEKOD)
    kls = [K.klyuch("S1", {"a": f"{a}.0", "b": "1.0"}) for a in range(4)]
    s_t, e_t = year_bounds(DATY, 2018)
    got = G.golos_v2(dc.poz_fn, [dict(kl=k) for k in kls], 5, T, s_t, e_t)
    st = np.stack([dc.chlen(k)[1][5] for k in kls])
    ref = np.zeros(T)
    execlayer.vote(np.ascontiguousarray(st), 4, s_t, e_t, ref, grids.VOTE_FRAC)
    assert np.abs(got - ref[s_t:e_t]).max() <= 1e-12 and set(np.unique(st)) <= {-1.0, 0.0, 1.0}
    assert stroki(root) == len(lm.Chitatel(root, "S1").rows())          # позиции членов в ledger ничего не пишут
    sdv = np.zeros(T)
    execlayer.vote(np.ascontiguousarray(np.roll(st, -1, axis=1)), 4, s_t, e_t, sdv, grids.VOTE_FRAC)
    assert np.abs(got - sdv[s_t:e_t]).max() > 0                         # мутация «голос по d+1» отличима


def test_ryady_chlenov_srezy_goda(tmp_path):
    root = baza(tmp_path)
    dc = D.Doschet(sreda(), "S1", root, RZ, DEKOD)
    res = K.kalibrovat(pul(root, dc), "train4", dc.poz_fn)
    ry = D.ryady_chlenov(res, dc, DATY)
    s, e = year_bounds(DATY, 2018)
    k, tk = next((k, v) for k, v in res[2018]["tikery"].items() if v)
    assert ry[2018][k].shape == (len(tk["chleny"]), e - s)
    assert np.array_equal(ry[2018][k][0], dc.chlen(tk["chleny"][0])[0][k, s:e])


def test_otkaz_kod_ne_sovpadaet_s_manifest(tmp_path):
    root = baza(tmp_path)
    with pytest.raises(ValueError, match="kod/vhody"):
        D.Doschet(Sreda(Evr(), "drugoj", VHODY, DATY, []), "S1", root, RZ, DEKOD)


def test_otkaz_daty_2020(tmp_path):
    root = baza(tmp_path)
    dd = np.append(DATY, np.datetime64("2020-01-02"))
    with pytest.raises(ValueError, match="2020"):
        D.Doschet(Sreda(Evr(), KOD, VHODY, dd, []), "S1", root, RZ, DEKOD)


def test_raskodirovat_i_dekod_iz_kuba():
    kub = {"signal": {"a": {"setka": [1, 2.5, "x"]}}, "yakorya_progon": {"q": 0.75}}
    d = D.dekod_iz_kuba(kub)
    pt = D.raskodirovat({"a": "1.0", "q": "0.75", "k_x": "net", "z": "3.5", "t": 2}, d)
    assert pt == {"a": 1, "q": 0.75, "k_x": "net", "z": 3.5, "t": 2} and isinstance(pt["a"], int)
