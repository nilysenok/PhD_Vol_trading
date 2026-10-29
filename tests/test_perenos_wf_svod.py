"""Перенос на сервер, лист P-2 (Ж-136): `svodka_ledger.py` и `fastbt.svod_server_v5` на синтетическом каталоге wf (2014-06…2026-02) —
без отказа. Данных и счёта нет: ряды случайные, известные по построению. Мутация: сводка отказывает на period = «wf» (тест падает)."""
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

from fastbt import kalibr_v5 as K, ledger_v2 as lm, period_v2 as PV, svod_server_v5 as SV

_p = Path(__file__).resolve().parents[1] / "scripts/server/svodka_ledger.py"
_s = importlib.util.spec_from_file_location("svodka_ledger", _p)
SL = importlib.util.module_from_spec(_s)
_s.loader.exec_module(SL)

DATY = np.array([d for d in np.arange("2014-06-02", "2026-03-01", dtype="datetime64[D]") if np.is_busday(d)])
OSI = {"a": list(range(4)), "b": list(range(3))}
OSI_S = {"S1": {a: [lm.tok(v) for v in vs] for a, vs in OSI.items()}}


def zapisat_wf(root: Path, seed=1, klyuch="seed", k3=False, sid="S1"):
    rng = np.random.default_rng(seed)
    man = {"T": DATY.size, "kod": "k", "vhody": "v", "period": "wf", "k3": k3, "sobol_klyuch": klyuch, "komissiya": 0.0004}
    w = lm.Pisatel(root, sid, 0, man)
    pts = [dict(zip(OSI, v)) for v in itertools.product(*OSI.values())]
    for i, pt in enumerate(pts):
        x = 0.0004 + 0.01 * rng.standard_normal(DATY.size)
        kl = lm.klyuch(sid, pt, "wf", 0.0004, None)
        row = {"status": "ok", "sobol_i": i, "v": lm.V, "klyuch": kl, "klyuch_sha": lm.klyuch_sha(kl, "k", "v"),
               "kod": "k", "vhody": "v", "by_year": {}, "by_quarter": {}}
        w.add(row, x, np.tile(x, (lm.NTIK, 1)))
    w.close()
    np.save(Path(root) / sid / "daty.npy", DATY)


def test_svodka_ledger_na_kataloge_wf_bez_otkaza(tmp_path, capsys):
    zapisat_wf(tmp_path / "wf_s1", seed=1)
    zapisat_wf(tmp_path / "wf_s2", seed=2, klyuch="seed2")
    zapisat_wf(tmp_path / "wf_k3", seed=3, k3=True)
    rows = SL.svodka_vse([tmp_path])
    assert [r["prohod"] for r in rows[:-1]] == ["k3", "wf s1", "wf s2"]
    assert all(r["sverka"] == "ok" for r in rows) and rows[-1]["tochek"] == 3 * 12 and rows[-1]["oshibok"] == 0
    assert SL.main([str(tmp_path)]) == 0 and "wf s1" in capsys.readouterr().out


def _tri_prohoda(tmp_path):
    zapisat_wf(tmp_path / "wf_s1", seed=1)
    zapisat_wf(tmp_path / "wf_s2", seed=2, klyuch="seed2")
    zapisat_wf(tmp_path / "wf_k3", seed=3, k3=True)


def _otmetka(tmp_path, monkeypatch):
    otm = tmp_path / "wf_zamok_snyat.json"           # только временный каталог; в репозитории отметки нет
    otm.write_text("{}")
    monkeypatch.setattr(PV, "OTMETKA", otm)


def _ohrana_kak_wf(orig):
    def f(self):                                      # как если бы пул знал свой период (manifest.period = wf)
        self.period = "wf"
        orig(self)
    return f


def test_svod_server_v5_na_kataloge_wf_bez_otkaza(tmp_path, monkeypatch):
    """При отметке замка и пуле с period = wf (К-wf2b, #645) три прохода wf сводятся, все таблицы записаны."""
    _otmetka(tmp_path, monkeypatch)
    monkeypatch.setattr(K.Pul, "_ohrana", _ohrana_kak_wf(K.Pul._ohrana))
    _tri_prohoda(tmp_path)
    assert {p["kl"][1] for p in SV.prohody(tmp_path)} == {"wf"}
    SV.svod(tmp_path, tmp_path / "out", OSI_S, 10)
    names = {f.name for f in (tmp_path / "out").iterdir()}
    assert {"kalibr_top10.csv", "kontroli_k1_k2_k3.csv"} <= names


@pytest.mark.xfail(strict=True, raises=ValueError, reason="находка P-2: K.pul_iz_ledger и kontroli_v5 строят Pul без period "
                   "(manifest.period = wf не передаётся) -> охрана #645 отказывает; снять xfail после правки fastbt/kalibr_v5.py")
def test_svod_server_v5_wf_pul_iz_ledger_ne_peredayot_period(tmp_path, monkeypatch):
    _otmetka(tmp_path, monkeypatch)
    _tri_prohoda(tmp_path)
    SV.svod(tmp_path, tmp_path / "out", OSI_S, 10)
