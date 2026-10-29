"""Л10: драйвер Соболя на кубе v5 (Ж-127, Ж-129 п. 8, Ж-130) на синтетическом оценщике (данных нет). Мутации — test_mut_*."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import fastbt.canon_v5 as cv5
import fastbt.ledger_v2 as lm
import fastbt.progon_sobol as ps

T = 8
# Эталон пути designs/progon, снятый на базе до правки (kanon-v22-pereschet 63b2b369): точки S1…S6 [0, 300) и ledger 2.1 S1, S4 [0, 200).
ETALON_TOCHKI = "834f75d2d7d3cd0884d79b1d1d118a803fc67b92050290f5265cf26f22936277"
ETALON_LEDGER = "5cd56b6a08838aa12e87ca2fdcc1569d7bf9aed3010a1822d33213664fe98628"


class Fake:
    """evaluate_v2 по контракту EvaluatorV2: ряды зависят только от точки и периода; fdr — атрибут, как у EvaluatorV2."""
    fdr = None

    def evaluate_v2(self, pt, period):
        kl = lm.klyuch("S1", pt, period, 0.0004, None)
        rng = np.random.default_rng(int(lm.klyuch_sha(kl, "a", "b")[:8], 16))
        tik = rng.normal(0, 0.01, (17, T)).astype(np.float32).astype(np.float64)
        return {"by_year": {str(y): 0.5 for y in range(2016, 2020)}, "T": T, "mediana_kv": 0.1}, np.nanmean(tik, 0), tik


class FakeT0d(Fake):
    fdr = "t0d"


def sozdat(ocen=Fake, start="2016-01-04"):
    daty = np.arange(np.datetime64(start), np.datetime64(start) + T)
    return lambda sid: ps.Sreda(ocen(), "kod-fake", "vhody-fake", daty, [f"T{i:02d}" for i in range(17)])


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(ps, "fabrika", sozdat())
    monkeypatch.setattr(ps, "_W", {})


@pytest.fixture
def fake5(monkeypatch):
    monkeypatch.setattr(ps, "fabrika", sozdat(FakeT0d))
    monkeypatch.setattr(ps, "_W", {})


def stroki(root, sid):
    out = []
    for p in sorted((Path(root) / sid).glob("p[0-9]*/rows.jsonl")):
        out += [json.loads(x) for x in p.read_text().splitlines()]
    return out


def sha_kataloga(root: Path) -> str:
    """Все файлы каталога побайтно; из строк — без t_sek, из manifest — без даты создания."""
    h = hashlib.sha256()
    for f in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        b = f.read_bytes()
        if f.name == "rows.jsonl":
            b = "\n".join(json.dumps({k: v for k, v in json.loads(x).items() if k != "t_sek"}, sort_keys=True)
                          for x in b.decode().splitlines()).encode()
        elif f.name == "manifest.json":
            b = json.dumps({k: v for k, v in json.loads(b).items() if k != "sozdan"}, sort_keys=True).encode()
        h.update(str(f.relative_to(root)).encode() + b"\0" + b)
    return h.hexdigest()


def test_put_progon_tochki_pobajtno():
    pts = {s: ps.tochki(ps.load_kub(s), 0, 300) for s in ps.SIDS}
    assert hashlib.sha256(json.dumps(pts, sort_keys=True).encode()).hexdigest() == ETALON_TOCHKI


def test_put_progon_ledger_pobajtno(fake, tmp_path):
    ps.schet(["S1", "S4"], 0, 200, "sobol1", tmp_path, 1, False, False, lm.V21)    # 1 процесс: раскладка по p{NN} детерминирована
    assert sha_kataloga(tmp_path) == ETALON_LEDGER
