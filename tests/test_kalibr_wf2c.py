"""К-wf2c: период wf сохраняется во всех местах создания `Pul` (pul_iz_ledger, podpul, podpul_ex, sintez_pul) и охрана остаётся.
Синтетика, отметка замка — только во временном каталоге. Мутация «период не передан» (Pul теряет period -> train4)
ловится test_mutatsiya_period_ne_peredan_lovitsya. python3 -m pytest tests/test_kalibr_wf2c.py -q"""
import itertools
from pathlib import Path

import numpy as np
import pytest

from fastbt import etap1_eval as EV, kalibr_v5 as K, kontroli_v5 as KV, ledger_v2 as lm, period_v2 as PV

DATY = np.array([d for d in np.arange("2014-06-02", "2026-03-01", dtype="datetime64[D]") if np.is_busday(d)])
OSI = {"a": list(range(3)), "b": list(range(3))}
OSI_T = {a: [lm.tok(v) for v in vs] for a, vs in OSI.items()}


def _ledger(root: Path, period="wf"):
    rng = np.random.default_rng(1)
    man = {"T": DATY.size, "kod": "k", "vhody": "v", "period": period, "komissiya": 0.0004}
    w = lm.Pisatel(root, "S1", 0, man)
    for i, v in enumerate(itertools.product(*OSI.values())):
        x = 0.01 * rng.standard_normal(DATY.size)
        kl = lm.klyuch("S1", dict(zip(OSI, v)), period, 0.0004, None)
        row = {"status": "ok", "sobol_i": i, "v": lm.V, "klyuch": kl, "klyuch_sha": lm.klyuch_sha(kl, "k", "v"),
               "kod": "k", "vhody": "v", "by_year": {}, "by_quarter": {}}
        w.add(row, x, np.tile(x, (lm.NTIK, 1)))
    w.close()
    np.save(root / "S1" / "daty.npy", DATY)


def _otmetka(tmp_path, monkeypatch):
    otm = tmp_path / "wf_zamok_snyat.json"
    otm.write_text('{"preregistraciya_v2": "docs/preregistraciya_v2.md", "sha256": "' + "a" * 64
                   + '", "data_snyatiya": "2026-10-30"}')
    monkeypatch.setattr(PV, "OTMETKA", otm)       # только временный каталог, в репозитории отметки нет


def _proverit(tmp_path):
    _ledger(tmp_path)
    pul = K.pul_iz_ledger(tmp_path, "S1", OSI_T)
    mask = np.ones(len(pul.tochki), bool)
    mask[0] = False
    for p in (pul, KV.podpul(pul, mask), KV.podpul_ex(pul, pul.ex.copy())):
        assert p.period == "wf"


def test_wf_pul_sohranyaet_period(tmp_path, monkeypatch):
    _otmetka(tmp_path, monkeypatch)
    _proverit(tmp_path)


def test_pul_iz_ledger_bez_polya_period_train4(tmp_path):
    _ledger(tmp_path, "train4")
    ld = K.pul_iz_ledger  # train4 без отметки: даты ≥ 2020 запрещены -> берём короткие даты
    with pytest.raises(ValueError):
        ld(tmp_path, "S1", OSI_T)


def test_ohrana_wf_ostayotsya_bez_otmetki(tmp_path, monkeypatch):
    monkeypatch.setattr(PV, "OTMETKA", tmp_path / "net.json")
    _ledger(tmp_path)
    with pytest.raises(PV.ZamokVladeltsa):
        K.pul_iz_ledger(tmp_path, "S1", OSI_T)


def test_podpul_ne_obhodit_ohranu(tmp_path, monkeypatch):
    _otmetka(tmp_path, monkeypatch)
    _ledger(tmp_path)
    pul = K.pul_iz_ledger(tmp_path, "S1", OSI_T)
    monkeypatch.setattr(PV, "OTMETKA", tmp_path / "net.json")      # отметку сняли -> копия wf-пула отказывает
    with pytest.raises(PV.ZamokVladeltsa):
        KV.podpul_ex(pul, pul.ex.copy())


def test_sintez_pul_period():
    d = DATY[DATY < np.datetime64("2019-12-31")]
    assert KV.sintez_pul(1, d).period == "train4" and KV.sintez_pul(1, d, period="v2").period == "v2"


def test_honest_wf():
    assert EV.HONEST["wf"] == "net" and set(EV.PERIODS) == set(EV.HONEST)


def test_mutatsiya_period_ne_peredan_lovitsya(tmp_path, monkeypatch):
    _otmetka(tmp_path, monkeypatch)
    orig = K.Pul

    def bez_perioda(*a, **kw):
        kw.pop("period", None)
        return orig(*a, **kw)
    monkeypatch.setattr(K, "Pul", bez_perioda)
    with pytest.raises((AssertionError, ValueError)):
        _proverit(tmp_path)
