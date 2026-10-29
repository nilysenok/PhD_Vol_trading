"""Т-683: подключение rang_starta в EvaluatorV5._rows (тренд S3/S4, стоп-заявка) на синтетике, без данных, без сети.
Ловит: ранг на d_sig (rg вместо rw), ранг дня входа, перенос через конец сигнала. Вес сделки = w[d_sig], d_sig — любой вечер
постановки пары s…s+k; сравнение строгое (равенство значений), а не «ненулевых не меньше».
python3 -m pytest tests/test_rang_trenda_podkl.py -q"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from fastbt import eval_v5
from fastbt.canon_v5 import NET

NAN = float("nan")
#             день:  0    1    2    3    4    5    6    7    8    9    10   11
SIG = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, 0.0])
RG = np.array([NAN, 0.8, NAN, NAN, NAN, 0.9, 0.3, NAN, NAN, 0.5, NAN, NAN])   # ранг — только в днях старта; день 5 — без сигнала
N = len(SIG)


def M_LIN(r):
    return (0.5 + r) / 1.5                                                     # m_sily = lin, M_MAX = 1,5


@dataclass
class _P:
    a_rang: object = None
    b0_c: object = None


def _vesa(monkeypatch, vhod_tip="stop_zayavka"):
    """Прогон настоящего _rows с подставленными источниками; возвращает (w, sig), которые ушли в run_rows."""
    ev = object.__new__(eval_v5.EvaluatorV5)
    r5 = SimpleNamespace(rang=True, rang_tip="obshchiy", m_sily="lin", razmer="ravnyj", ext={}, sp={}, pt4={},
                         nov={"vhod_tip": vhod_tip, "tejk_signal": NET})
    ev.r5, ev.sid, ev._days5 = r5, "S3", {}
    ev.sk = SimpleNamespace(tickers=["T"])
    ev.inp = SimpleNamespace(bars={"T": [None] * N}, close_n6={"T": np.ones(N)}, div={"T": np.zeros(N)})
    info = {"T": (0, N - 1, 0, None, SIG.copy(), np.ones(N), np.ones(N, bool))}
    ev._start = lambda period: info
    ev._ranki = lambda inf: {"T": (RG.copy(), np.ones(N, bool))}
    ev._params = lambda *a, **k: _P()
    ev._urovni = lambda *a, **k: {}
    ev._ed = lambda *a, **k: None
    ev.fdr_kw = lambda: {}
    got = {}

    def fake_run_rows(t, days, sig, sg, p, w=None, **kw):
        got["w"], got["sig"] = np.asarray(w, float), np.asarray(sig, float)
        return "rows"

    monkeypatch.setattr(eval_v5, "run_rows", fake_run_rows)
    monkeypatch.setattr(eval_v5, "_one_bar", lambda c: None)
    assert ev._rows({}, "train", False)[0][1] == "rows"
    return got["w"], got["sig"]


def test_ves_sdelki_trenda_ravna_rangu_starta_na_vseh_vecherah(monkeypatch):
    w, sig = _vesa(monkeypatch)
    assert np.array_equal(sig, SIG)                                          # сигнал не тронут (кандидат с рангом)
    assert w[1:4].tolist() == pytest.approx([M_LIN(0.8)] * 3)               # вход в день > s → вес по r, не 0 и не ранг d_sig
    assert w[6:9].tolist() == pytest.approx([M_LIN(0.3)] * 3)               # новый старт → r'
    assert w[9:11].tolist() == pytest.approx([M_LIN(0.5)] * 2)              # смена знака → новый старт


def test_ves_ne_perenositsya_cherez_konec_signala(monkeypatch):
    w, _ = _vesa(monkeypatch)
    assert w[0] == 0 and w[4] == 0 and w[5] == 0 and w[11] == 0            # день 5: ранг 0,9 есть, сигнала нет → вес 0
    assert w[6] != pytest.approx(M_LIN(0.8))                                # r' не равен прежнему r


def test_kontrtrend_ves_tolko_v_den_starta(monkeypatch):
    w, _ = _vesa(monkeypatch, vhod_tip="podtverzhdenie")
    assert w[1] == pytest.approx(M_LIN(0.8)) and w[6] == pytest.approx(M_LIN(0.3))   # день старта — ранг дня старта
    assert w[2] == 0 and w[7] == 0                                          # d_sig = день старта, переноса нет
