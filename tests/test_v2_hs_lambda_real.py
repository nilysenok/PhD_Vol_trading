"""Ж-212 (G9): λ = 0 ≡ HAR на боевом пути. `hs_cstage.stage_cs` строит ряд Cs; при выборе λ = 0 он обязан быть побитно равен HAR из components.parquet. Мутации: λ=0 заменён малым ненулём; c0-смещение; offset по другому ряду."""
from __future__ import annotations

import numpy as np
import pytest

from models_v2.s6.e import comp
from models_v2.s8 import hs_cstage, hs_eval, hs_rules, hs_store


@pytest.fixture(scope="module")
def df():
    return comp.load()


def run_cs(monkeypatch, df, il, lambdas=None):
    put = {}
    n = len(df)
    rng = np.random.default_rng(5)
    rs = {("XGB", "Ca"): {"g": rng.normal(0, 0.3, n)}, ("XGB", "Cc"): {"g": rng.normal(0, 0.3, n)}}
    monkeypatch.setattr(hs_store, "put", lambda d, nm, s, positive=True: put.__setitem__((d, nm), np.asarray(s, float)))
    monkeypatch.setattr(hs_cstage.L, "register", lambda *a, **k: None)
    # первый вызов выбирает g из Ca–Ce (колонка 0), второй — λ (индекс il)
    seq = iter([np.zeros(n, int), np.full(n, il, int)])
    monkeypatch.setattr(hs_rules, "causal_pick", lambda *a, **k: next(seq))
    if lambdas is not None:
        monkeypatch.setattr(hs_cstage, "LAMBDAS", lambdas)
    r = hs_cstage.stage_cs({"df": df}, "XGB", rs)
    return r, put


def test_lambda_zero_real_path_equals_components_har_bitwise(monkeypatch, df):
    r, put = run_cs(monkeypatch, df, 0)
    har = df["HAR"].to_numpy(float)
    assert np.array_equal(put[("C", "C_xgb_Cs")], har)
    assert (r["g"] == 0).all()
    # боевая оценка строки: σ̂² = HAR·exp(g) при g = 0
    assert np.array_equal(har * np.exp(r["g"]), har)


def test_lambda_nonzero_differs(monkeypatch, df):
    _, put = run_cs(monkeypatch, df, 2)
    assert not np.array_equal(put[("C", "C_xgb_Cs")], df["HAR"].to_numpy(float))


def test_mutation_lambda_zero_replaced_by_small_is_caught(monkeypatch, df):
    _, put = run_cs(monkeypatch, df, 0, lambdas=(1e-9, 0.25, 0.5, 0.75, 1.0))
    assert not np.array_equal(put[("C", "C_xgb_Cs")], df["HAR"].to_numpy(float))


def test_mutation_c0_shift_and_full_sample_offset_are_caught(df):
    har = df["HAR"].to_numpy(float)
    g = np.zeros(len(har))
    assert np.array_equal(har * np.exp(g), har)
    assert not np.array_equal(har * np.exp(g + 1e-12), har)                      # смещение c0
    off = har * np.exp(np.log(df["rv"].to_numpy(float) / har).mean())            # offset по полной выборке
    assert not np.array_equal(off, har)


def test_eval_row_uses_har_times_exp_g(monkeypatch, df):
    """Боевая eval_row: s2 для строки = HAR·exp(g); при g = 0 в hs_eval.row уходит побитно HAR."""
    got = {}

    class Row(dict):
        def __missing__(self, k):
            return 0.0

    def fake_row(d, name, s2, ctrl, extra=None):
        got.setdefault("s2", np.asarray(s2, float))
        return Row()

    monkeypatch.setattr(hs_eval, "row", fake_row)
    monkeypatch.setattr(hs_cstage, "best_b", lambda lib: (("b", np.ones(len(df))), ("f", np.ones(len(df)))))
    r = {"name": "C_xgb_Cs", "lib": "XGB", "cfg": "Cs", "secs": 0.0, "g": np.zeros(len(df))}
    hs_cstage.eval_row({"df": df}, r)
    assert np.array_equal(got["s2"], df["HAR"].to_numpy(float))
