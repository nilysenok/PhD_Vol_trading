"""V7-3: ось k_okup в оценщике v7 (fastbt/eval_v7_okup.py), синтетика. Данные не читает.
G1 «net = путь v6» (точка, ключ кэша); G2 известный ответ (терциль 1 проходит, 2 и 3 — нет; сделок не больше); G3 M-d1 (закрытая в день входа
не входит, закрытая в d−1 входит); G4 ключ кэша содержит k; порядок «тень без маски → основной прогон с маской». Реальные точки — внизу (тяжёлый, FASTBT_V6_REAL=1)."""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fastbt import eval_v7_okup as O
from fastbt.etap1_data import gate_trades
from fastbt.eval_v6 import EvaluatorV6
from fastbt.eval_v7 import EvaluatorV7
from fastbt.exec_e_types import Trade

N = 400
DATES = pd.bdate_range("2018-01-02", periods=N).values
SILA = np.array([1.0 + (i % 3) for i in range(N)])            # терциль силы дня i = 1 + i % 3 (при окне ≥ 30)
KOM = 0.0004                                                   # комиссия стороны; на сделку (вход + выход) 0,08 %


def _tr(j, gross_pct, vyhod=None, side=1, **kw):
    """Сделка по сигналу дня j, вход j+1, выход j+2 (или vyhod); gross_pct — % по цене."""
    x = j + 2 if vyhod is None else vyhod
    return Trade("T", j, j + 1, 0, 100.0, exit_day=x, exit_bar=0, exit_px=100.0 * (1 + side * gross_pct / 100), reason="take", side=side,
                 fee_in=KOM, fee_out=KOM, **kw)


class Baza:
    """Роль EvaluatorV6: _allow пропускает всё, _rows отдаёт готовые строки и помнит, была ли маска."""
    def __init__(self, trades):
        self.trades, self.vyzovy = trades, []
        self.inp = SimpleNamespace(meta={"T": {"dates": DATES, "s0": 0}})
        self.sk, self.r5 = SimpleNamespace(tickers=["T"]), SimpleNamespace(sp={})

    def _view(self, t, sp, q):
        return (SimpleNamespace(strength=np.array([SILA])),)

    def _allow(self, pt, t, dates, sig, f):
        return np.ones(len(sig), bool)

    def _rows(self, pt, period, anchor):
        self.vyzovy.append(self._okup_zh is None)
        return [(0, SimpleNamespace(trades=self.trades), 0, N - 1)]


class Ev(O.OkupMixin, Baza):
    pass


def _ev(trades, k=3):
    e = Ev(trades)
    e._okup_k = k
    return e


def _sig(starts):
    s = np.zeros(N)
    s[list(starts)] = 1.0
    return s


def _obuchenie(g2=0.2):
    """Терциль 1 (d_sig % 3 == 0): валовой 2,0 %; терциль 2: g2; терциль 3: −1,0 %. По ~75 сделок в окне каждого."""
    g = {0: 2.0, 1: g2, 2: -1.0}
    return [_tr(j, g[j % 3]) for j in range(60, 300)]


STARTY = (330, 334, 338)                                       # терцили 1, 2, 3 (1 + i % 3 для i = 330, 334, 338: 330 % 3 = 0, 334 % 3 = 1, 338 % 3 = 2)


def _allow(e, starts=STARTY):
    sig = _sig(starts)
    a = e._allow({}, "T", DATES, sig, "net")
    return sig, a


# ---------------- G2: известный ответ ----------------
def test_g2_izvestnyj_otvet_terciL_1_prohodit_2_i_3_net():
    e = _ev(_obuchenie(g2=0.2))
    e._okup_zh = O.zhurnal(e, e._rows({}, "p", False))
    sig, a = _allow(e)
    assert [bool(a[i]) for i in STARTY] == [True, False, False]          # 0,2 < 3 × 0,08 = 0,24; −1 < 0,24; 2,0 ≥ 0,24
    assert gate_trades(sig, a).astype(bool).sum() <= sig.astype(bool).sum()
    assert (gate_trades(sig, a) != 0).sum() == 1 and (gate_trades(sig, ~np.zeros(N, bool)) != 0).sum() == 3
    assert a.sum() == N - 2                                              # маска трогает только дни старта


def test_g2_k2_porog_nizhe_terciL_2_prohodit():
    e = _ev(_obuchenie(g2=0.2), k=2)                                     # 0,2 ≥ 2 × 0,08 = 0,16
    e._okup_zh = O.zhurnal(e, e._rows({}, "p", False))
    assert [bool(_allow(e)[1][i]) for i in STARTY] == [True, True, False]


# ---------------- журнал сделок ----------------
def test_zhurnal_izvestnyj_otvet_shorty_otkrytye_chasti():
    e = _ev([])
    short = _tr(90, 10.0, side=-1)                                       # шорт 100 → 90: +10 %
    chast = _tr(93, 5.0, parent=0)
    otkr = Trade("T", 96, 97, 0, 100.0)                                  # без выхода
    z = O.zhurnal(e, [(0, SimpleNamespace(trades=[short, chast, otkr]), 0, N - 1)])
    assert len(z) == 1
    r = z.iloc[0]
    assert np.isclose(r.gross, 10.0) and np.isclose(r.kom, 0.08) and r.short == 0.0 and r.tercil == 1.0 + 90 % 3
    assert pd.Timestamp(r.vyhod) == pd.Timestamp(DATES[92]) and r.uderzhanie == (DATES[92] - DATES[91]).astype("timedelta64[D]").astype(int)


def test_zhurnal_pustoj():
    z = O.zhurnal(_ev([]), [(0, SimpleNamespace(trades=[]), 0, N - 1)])
    assert len(z) == 0 and {"vyhod", "tercil", "uderzhanie", "gross", "kom", "short"} <= set(z.columns)


def test_pustoj_zhurnal_vhod_razreshen():
    e = _ev([])
    e._okup_zh = O.zhurnal(e, e._rows({}, "p", False))
    assert _allow(e)[1].all()                                            # < 20 сделок — вход разрешён с флагом


# ---------------- G3: M-d1 ----------------
def _d1_trades(vyhod20):
    """19 сделок терциля 1 (убыточных), закрытых до дня входа d = dates[211]; 20-я закрывается в день vyhod20."""
    return [_tr(j, -1.0) for j in range(60, 117, 3)] + [_tr(117, -1.0, vyhod=vyhod20)]


def test_g3_zakrytaya_v_den_vhoda_ne_uchityvaetsya_a_v_d_minus_1_da():
    sig = _sig([210])                                                    # день решения 210 (терциль 1), вход d = dates[211]
    for vyhod, ozhid in ((211, True), (210, False)):
        e = _ev(_d1_trades(vyhod))
        e._okup_zh = O.zhurnal(e, e._rows({}, "p", False))
        assert bool(e._allow({}, "T", DATES, sig, "net")[210]) is ozhid        # 211: n = 19 → разрешён с флагом; 210: n = 20, убыток → нет


def test_g3_mutaciya_d_plyus_1_lovitsya(monkeypatch):
    monkeypatch.setattr(O, "den_vhoda", lambda dates, i: dates[i + 2])        # мутация: день входа на сутки позже → закрытая в d попадает в окно
    e = _ev(_d1_trades(211))
    e._okup_zh = O.zhurnal(e, e._rows({}, "p", False))
    assert not e._allow({}, "T", DATES, _sig([210]), "net")[210]               # на каноне здесь True (тест выше)


def test_tercil_tolko_po_dnyam_do_resheniya():
    assert O.tercil(SILA, 330) == 1.0 and np.isnan(O.tercil(SILA, 10))          # терциль 1; < 30 наблюдений — пусто
    iz = SILA.copy()
    iz[331:] = 1000.0                                                            # будущее не должно менять терциль дня 330
    assert O.tercil(iz, 330) == O.tercil(SILA, 330)


# ---------------- порядок: тень без маски → основной прогон с маской ----------------
def test_ten_bez_maski_potom_osnovnoj_s_maskoj():
    e = _ev(_obuchenie())
    O.OkupMixin._rows(e, {}, "p", False)
    assert e.vyzovy == [True, False] and e._okup_zh is None                    # первый вызов без журнала, второй — с журналом; состояние снято


def test_vykl_odin_prohod():
    e = _ev(_obuchenie(), k=None)
    O.OkupMixin._rows(e, {}, "p", False)
    assert e.vyzovy == [True]


# ---------------- G1/G4: точка и ключ ----------------
def _v7():
    e = object.__new__(EvaluatorV7)
    e.sid, e.proxy, e.ist7, e._fk = "S1", "S1", SimpleNamespace(sha="x"), ()
    return e


def test_g1_net_ravno_otsutstviyu_osi(monkeypatch):
    vid = []
    monkeypatch.setattr(EvaluatorV6, "evaluate_v2", lambda self, pt, period, vozm=None: vid.append((dict(pt), self._fkey)) or "ok")
    e = _v7()
    for pt in ({"a": 1}, {"a": 1, "k_okup": "net"}):
        assert e.evaluate_v2(pt, "train4") == "ok"
    assert vid[0] == vid[1] == ({"a": 1}, ()) and e._okup_k is None
    e.evaluate_v2({"a": 1, "k_okup": 3}, "train4")
    assert vid[2][0] == {"a": 1} and e._okup_k == 3                           # ось не уходит в v6; k хранится в оценщике


def test_osi_k_nedopustimyj_otkaz():
    for k in (5, 0, "da", True):
        with pytest.raises(ValueError, match="k_okup"):
            O.k_iz_tochki({"k_okup": k})


def test_g4_klyuch_kesha_soderzhit_k(monkeypatch):
    keys = []
    for k in (None, 2, 3):
        e = _v7()
        e._okup_k = k
        keys.append(e._fkey)
    assert len({repr(x) for x in keys}) == 3 and keys[0] == ()


def test_g4_mutaciya_k_ne_v_klyuche_lovitsya(monkeypatch):
    monkeypatch.setattr(O.OkupMixin, "_okup_key", lambda self: ())
    a, b = _v7(), _v7()
    a._okup_k, b._okup_k = 2, 3
    assert a._fkey == b._fkey                                                  # на каноне ключи различны (тест выше падал бы)


# ---------------- реальные точки (G1, тяжёлый) ----------------
real = pytest.mark.skipif(os.environ.get("FASTBT_V6_REAL") != "1" or not __import__("fastbt.daily_1850", fromlist=["x"]).out_path().exists(),
                          reason="тяжёлый реальный тест (FASTBT_V6_REAL=1, дневной ряд 18:50); данные ≥ 2025 не читаются")


@real
@pytest.mark.parametrize("sid", ["S1", "S3", "S5"])
def test_g1_realnaya_tochka_net_pobajtno_kak_bez_osi(sid):
    from fastbt import etap1
    inp, base = etap1.build_inputs(sid, period="train4"), etap1.base_point(sid)
    e = EvaluatorV7(sid, inp, base)
    r0, x0, t0 = e.evaluate_v2(dict(base), "train4")
    r1, x1, t1 = e.evaluate_v2({**base, "k_okup": "net"}, "train4")
    for r in (r0, r1):
        r.pop("t_sek", None)
    assert repr(r0) == repr(r1) and x0.tobytes() == x1.tobytes() and t0.tobytes() == t1.tobytes()
    r3, _, _ = e.evaluate_v2({**base, "k_okup": 3}, "train4")
    assert sum(r3["sdelok_y"].values()) <= sum(r0["sdelok_y"].values())          # маска только запрещает входы
