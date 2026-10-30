"""Тестировщик PR #795 (wf-int-a1): мутации, не пойманные tests/test_wf_int_a1.py.

(г) `zagruzit` не передаёт `period` в `Fajly` — wf молча читается как train4 (нет CNY в sha, нет границы wf).
(д) `s9.use_v2` не передаёт `PERIOD` в `sigma_src` — wf сверяется с SHA/источниками ch4_dev.
Синтетика, сеть и реальные данные не нужны. Глобалы s9 подменяются через monkeypatch (изоляция, мутация № 43/52).
"""
import hashlib

import numpy as np
import pandas as pd
import pytest

from fastbt import istochniki_v6 as iv, s9, vneshnie_1850 as v18

MIN = [9 * 60 + 50 + 10 * k for k in range(54)]
DAYS = pd.bdate_range("2024-05-13", "2024-06-28")


def _bars(seed: int, s0: float) -> pd.DataFrame:
    rng, rows, p = np.random.default_rng(seed), [], s0
    for d in DAYS:
        for m in MIN:
            q = p * float(np.exp(rng.normal(0, 0.002)))
            rows.append((d + pd.Timedelta(minutes=m), p, max(p, q) * 1.0005, min(p, q) * 0.9995, q, 1000.0))
            p = q
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


class Fake(iv.Fajly):
    def __init__(self, end=None, main=None, period=None):
        self.end, self.main, self.period = pd.Timestamp("2026-03-01"), None, period

    def akcii(self, t):
        return _bars(5, 90.0)

    def indeks(self, name):
        return {"usdrub": _bars(1, 90.0), "cnyrub": _bars(2, 12.0)}.get(name.lower())

    def zakrytie(self, t):
        return v18.close_1850(self.akcii(t), t)

    def brent(self):
        return {}, pd.DataFrame({"secid": [], "lsttrade": [], "roll": []})

    def kalendar(self):
        return pd.DataFrame({"date": [], "known_from": []}), pd.DataFrame()

    def div_otfiltrovan(self):
        return False


def test_zagruzit_peredayot_period_v_fajly(monkeypatch):
    """Мутация «period не передан из zagruzit в Fajly»: sha wf совпал бы с sha train4 (нет cnyrub)."""
    monkeypatch.setattr(iv, "Fajly", Fake)
    monkeypatch.setattr(iv, "konec", lambda period: pd.Timestamp("2026-03-01"))
    w, t = iv.zagruzit("wf", ["AAA"]), iv.zagruzit("train4", ["AAA"])
    assert w.sha == iv.sha_ryadov(Fake(period="wf"), ["AAA"])
    assert t.sha == iv.sha_ryadov(Fake(period="train4"), ["AAA"])
    assert w.sha != t.sha


def test_use_v2_wf_beryot_istochniki_i_sha_wf(tmp_path, monkeypatch):
    """Мутация «use_v2 вызывает sigma_src() без PERIOD»: wf сверялся бы с ch4_dev (шесть источников, SHA256.txt)."""
    wf = tmp_path / "wf"
    wf.mkdir()
    sums = []
    for lab in ("hyb_h1", "hyb_h5"):
        name = s9.SOURCES_V2[lab][0].name
        (wf / name).write_bytes(lab.encode())
        sums.append(f"{hashlib.sha256(lab.encode()).hexdigest()}  {name}")
    (wf / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    monkeypatch.setattr(s9, "CH4_WF", wf)
    for g in ("V2", "TABLES", "SUFFIX", "COMM"):
        monkeypatch.setattr(s9, g, getattr(s9, g))
    monkeypatch.setattr(s9, "SOURCES", {})
    monkeypatch.setattr(s9, "PERIOD", "wf")
    sha = s9.use_v2()
    assert sorted(sha) == ["hyb_h1", "hyb_h5"] and sorted(s9.SOURCES) == ["hyb_h1", "hyb_h5"]
    assert all(p.parent == wf for p, _, _ in s9.SOURCES.values())


def test_use_v2_wf_sha_ne_sovpal_otkaz(tmp_path, monkeypatch):
    """Подмена файла σ̂ wf при прежнем SHA256SUMS — SystemExit (проверка идёт именно по файлам wf)."""
    wf = tmp_path / "wf"
    wf.mkdir()
    sums = []
    for lab in ("hyb_h1", "hyb_h5"):
        name = s9.SOURCES_V2[lab][0].name
        (wf / name).write_bytes(b"podmena")
        sums.append(f"{hashlib.sha256(lab.encode()).hexdigest()}  {name}")
    (wf / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    monkeypatch.setattr(s9, "CH4_WF", wf)
    monkeypatch.setattr(s9, "PERIOD", "wf")
    with pytest.raises(SystemExit):
        s9.use_v2()
