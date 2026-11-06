"""Ж-212 (Д5): ключ кэша har_pit.npz = SHA-256 входов + кода. Подмена входа или кода при том же размере ряда обязана давать промах; мутация «ключ только по длине» ловится."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from models_v2.s8 import hs_ctx, hs_feat


def fake_f(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(dates=np.arange(n).astype("datetime64[D]").astype("datetime64[ns]"), seg=np.zeros(n, int),
                           rv=np.exp(rng.normal(size=n)), X=rng.normal(size=(n, 5)))


def setup(monkeypatch, tmp_path, key_fn=None):
    calls = []
    monkeypatch.setattr(hs_ctx, "HAR_PIT", tmp_path / "har_pit.npz")
    monkeypatch.setattr(hs_ctx.P, "OUT", tmp_path)
    monkeypatch.setattr(hs_feat, "har_pit", lambda F: (calls.append(1), (np.full(len(F.rv), float(len(calls))), np.zeros((len(F.rv), 5))))[1])
    if key_fn is not None:
        monkeypatch.setattr(hs_ctx, "har_key", key_fn)
    return calls


def substitution_misses(monkeypatch, tmp_path, key_fn=None) -> bool:
    calls = setup(monkeypatch, tmp_path, key_fn)
    F = fake_f()
    hs_ctx.har_cached(F)
    hs_ctx.har_cached(F)
    assert len(calls) == 1                      # тот же вход — попадание
    F2 = fake_f()
    F2.rv = F2.rv.copy()
    F2.rv[5] *= 1.5                             # та же длина, другой вход
    lb, _ = hs_ctx.har_cached(F2)
    return len(calls) == 2 and lb[0] == 2.0


def test_substituted_input_gives_miss(monkeypatch, tmp_path):
    assert substitution_misses(monkeypatch, tmp_path)


def test_mutation_key_by_length_is_caught(monkeypatch, tmp_path):
    assert not substitution_misses(monkeypatch, tmp_path, key_fn=lambda F: str(len(F.rv)))


def test_each_input_changes_key():
    k0 = hs_ctx.har_key(fake_f())
    for name in ("dates", "seg", "rv", "X"):
        F = fake_f()
        a = np.array(getattr(F, name))
        a.flat[3] = a.flat[3] + (1 if name in ("dates", "seg") else 0.5)
        setattr(F, name, a)
        assert hs_ctx.har_key(F) != k0, name
    assert hs_ctx.har_key(fake_f()) == k0


def test_substituted_code_changes_key(monkeypatch, tmp_path):
    f = tmp_path / "m.py"
    f.write_text("x = 1\n")
    monkeypatch.setattr(hs_ctx, "CODE", (SimpleNamespace(__file__=str(f)),))
    k0 = hs_ctx.har_key(fake_f())
    f.write_text("x = 2\n")
    assert hs_ctx.har_key(fake_f()) != k0


def test_code_modules_are_the_builders():
    names = {m.__name__ for m in hs_ctx.CODE}
    assert {"models_v2.s8.hs_feat", "models_v2.s8.hs_pit", "models_v2.har", "models_v2.s2.features"} <= names


def test_old_cache_without_key_is_miss(monkeypatch, tmp_path):
    calls = setup(monkeypatch, tmp_path)
    F = fake_f()
    np.savez(tmp_path / "har_pit.npz", lb=np.zeros(len(F.rv)), beta=np.zeros((len(F.rv), 5)))
    hs_ctx.har_cached(F)
    assert len(calls) == 1
