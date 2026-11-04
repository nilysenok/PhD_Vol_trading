"""Мутации к PR #846 (vozm.sig_fn у v5/v6): порядок «sig_fn → allow», ключ строки контроля, утечка в боевой ledger.
Реальные данные train4 (≤ 2019), 3 тикера; нет ряда 18:50 — skip. python3 -m pytest tests/test_eval_vozm_mut.py -q
"""
from types import SimpleNamespace

import numpy as np
import pytest

from fastbt import eval_v5, ledger_v2
from fastbt import progon_kontrol as pk
from test_eval_vozm_sig_fn import SP, _ev, _pt, real  # noqa: F401

ONES = {"sig_fn": lambda t, s, d: np.ones(len(s))}


_START = eval_v5.EvaluatorV5._start


def _gated(ev, vozm, monkeypatch):
    """Сигнал после _start (с allow) по тикерам и маски allow; allow принудительно закрывает чётные дни (на train4 события их не закрывают)."""
    orig = ev._allow

    def allow(pt, t, dates, sig, f):
        a = np.array(orig(pt, t, dates, sig, f), bool)
        a[::2] = False
        return a
    monkeypatch.setattr(ev, "_allow", allow)
    ev.evaluate_v2(_pt(), "train4", vozm)
    ev._ctx = dict(ev._ctx, vozm=dict(vozm))
    info = ev._start("train4")
    return [(v[4], v[6]) for v in info.values()]


def _start_posle_allow(self, period):
    """Подложенная ошибка: sig_fn после allow/gate_trades (а не до) — запрещённые allow-дни воскресают."""
    r, out = self.r5, {}
    fn = self._ctx["vozm"].get("sig_fn")
    saved = self._ctx["vozm"]
    self._ctx = dict(self._ctx, vozm={})
    info = _START(self, period)
    self._ctx = dict(self._ctx, vozm=saved)
    for t, v in info.items():
        sig = np.asarray(fn(t, v[4], v[3]), float) if fn else v[4]
        out[t] = (*v[:4], sig, *v[5:])
    return out


@real
@pytest.mark.parametrize("cls", ["v5", "v6"])
def test_sig_fn_do_allow_zapreshchennye_dni_ostayutsya_nulyami(cls, monkeypatch):
    par = _gated(_ev(cls), ONES, monkeypatch)
    zapret = sum(int((~al).sum()) for _, al in par)
    assert zapret > 0, "на train4 нет дней, закрытых allow — тест ничего не проверяет"
    for sig, al in par:
        assert not np.any(sig[~al] != 0)


@real
def test_mutaciya_sig_fn_posle_allow_lovitsya(monkeypatch):
    ev = _ev("v5")
    monkeypatch.setattr(eval_v5.EvaluatorV5, "_start", _start_posle_allow)
    par = _gated(ev, ONES, monkeypatch)
    assert any(np.any(sig[~al] != 0) for sig, al in par)           # на мутации инвариант выше нарушен


@real
def test_mutaciya_sig_fn_tolko_v5_lovitsya(monkeypatch):
    """v6 наследует _start v5: если v6 не передаёт vozm дальше (старый EvaluatorV2.evaluate_v2 без vozm) — sig_fn молча теряется."""
    ev = _ev("v6")
    zero = {"sig_fn": lambda t, s, d: np.zeros(len(s))}
    base = sum(ev.evaluate_v2(_pt(), "train4", zero)[0]["sdelok_y"].values())
    assert base == 0
    orig = eval_v5.EvaluatorV2.evaluate_v2
    monkeypatch.setattr(eval_v5.EvaluatorV2, "evaluate_v2", lambda s, pt, p, vozm=None: orig(s, pt, p))
    assert sum(ev.evaluate_v2(_pt(), "train4", zero)[0]["sdelok_y"].values()) > 0


def _row(z):
    ev = SimpleNamespace(sid="S1", fdr=None)
    row = {"period": "train4", "komissiya": 0.0004, "kod": "k", "vhody": "v"}
    return pk._row(ev, row, z, "k", "v", False)


def _kl_sha(vozm_sig):
    kontr = "k2:lag1"
    return _row(pk.Zadacha("kontrol", kontr, None, _pt(), vozm_sig))["klyuch_sha"]


def test_klyuch_kontrolya_otlichaetsya_ot_bazy_i_mezhdu_soboy():
    pt = _pt()
    baza = _row(pk.Zadacha("baza", None, None, pt, {}))["klyuch_sha"]
    k2 = _row(pk.Zadacha("kontrol", "k2:lag1", None, pt, {"sig_fn": None}))["klyuch_sha"]
    k2b = _row(pk.Zadacha("kontrol", "k2:lag2", None, pt, {"sig_fn": None}))["klyuch_sha"]
    pl = _row(pk.Zadacha("kontrol", "placebo:0", None, pt, {"sig_fn": None}))["klyuch_sha"]
    assert len({baza, k2, k2b, pl}) == 4


def test_mutaciya_klyuch_bez_kontrol_lovitsya(monkeypatch):
    orig = ledger_v2.klyuch
    monkeypatch.setattr(pk.ledger_v2, "klyuch", lambda *a: {k: v for k, v in orig(*a).items() if k not in ("kontrol", "vozm")})

    def row(z):
        r = _row(z)
        kl = pk.ledger_v2.klyuch("S1", z.pt, "train4", 0.0004, None)     # без полей контроля
        return ledger_v2.klyuch_sha(kl, "k", "v")
    pt = _pt()
    a = row(pk.Zadacha("kontrol", "k2:lag1", None, pt, {}))
    b = row(pk.Zadacha("baza", None, None, pt, {}))
    assert a == b                                                   # мутация реально склеивает ключи (значит тест выше её ловит)


@real
def test_boevoy_pisatel_ne_prinimaet_stroku_s_funkciey(tmp_path):
    """Утечка в ledger: Pisatel.add обязан отказать сразу, а не падать позже на json.dumps при close (с мусором на диске)."""
    ev = _ev("v5")
    r, x, tik = ev.evaluate_v2(_pt(), "train4", {"sig_fn": lambda t, s, d: s})
    r = {**r, "klyuch_sha": "a" * 64}
    p = ledger_v2.Pisatel(tmp_path / "ledger", "S1", 1, dict(T=len(x), kod="x", vhody="y"))
    with pytest.raises((TypeError, ValueError)):
        p.add(r, x, tik)
        p.close()
    assert not any((p.pd / "rows.jsonl").read_text().strip().splitlines()), "строка с vozm попала в rows.jsonl"
