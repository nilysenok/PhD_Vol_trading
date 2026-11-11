"""S-C-DATA, реальные данные: состав пула, тождество на реальных сессиях, PIT усечением, записанные выходы, `--check` только читает."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import c_data, c_decomp as DC, c_run

CUT_A, CUT_B = "2018-06-29", "2018-12-31"


@pytest.fixture(scope="module")
def inp_b():
    return c_data.build_inputs(CUT_B)


def test_pool_is_16_without_sber_and_imoex():
    assert len(c_data.POOL) == 16 and "SBER" not in c_data.POOL and "IMOEX" not in c_data.POOL


def test_bars_sum_to_target_rv_and_identity_on_real_sessions(inp_b):
    assert np.abs(np.add.reduceat(inp_b.r ** 2, inp_b.off[:-1]) * 1e4 - inp_b.rv).max() < 1e-10      # r баров = целевая RV
    df = DC.run(inp_b, range(len(inp_b.days) - 60, len(inp_b.days)))
    assert len(df) == 120 and (df["ident_err"].abs() / df["rv"] < 1e-13).all() and (df["ident_err_pit"].abs() / df["rv"] < 1e-13).all()
    assert (df["cross"].abs() / df["rv"] < 1e-12).all() and (df["n_cov"] > 0).all()


def test_mutation_dropped_real_bar_breaks_identity(inp_b):
    s = len(inp_b.days) - 5
    k = inp_b.off[s] + 7
    m = c_data.Inputs(inp_b.days, inp_b.rv, inp_b.nbars, np.concatenate([inp_b.off[:s + 1], inp_b.off[s + 1:] - 1]), np.delete(inp_b.day, k),
                      np.delete(inp_b.slot, k), np.delete(inp_b.r, k), np.delete(inp_b.R, k, 0), np.delete(inp_b.ri, k), inp_b.stats)
    assert abs(DC.session_rows(m, s)[0]["ident_err"]) > 1e-6


def test_pit_truncation_of_real_future_does_not_change_rows(inp_b):
    a = c_data.build_inputs(CUT_A)                       # будущее после 2018-06-29 физически отсутствует
    n = len(a.days)
    for s in (n - 1, n - 40):
        assert DC.session_rows(a, s) == DC.session_rows(inp_b, s)


def test_real_sessions_of_next_day_not_in_session(inp_b):
    d = inp_b.days.to_numpy()[inp_b.day]
    assert (np.diff(d.astype("int64")) >= 0).all() and (inp_b.slot >= 0).all() and (inp_b.slot <= 52).all()
    assert (np.diff(inp_b.off) == inp_b.nbars).all()      # число баров сессии = n_bars build_daily (бары другой даты не попали)


def test_written_outputs_identity_all_sessions():
    f = c_run.OUT / "C_sessions.parquet"
    if not f.exists():
        pytest.skip("выходы не записаны: python3 -m models_v2.s18.c_run --write")
    df = pd.read_parquet(f)
    assert df["d"].nunique() > 3000 and set(df["k_factors"]) == {1, 2}
    assert (df["ident_err"].abs() / df["rv"] < 1e-13).all() and (df["ident_err_pit"].abs() / df["rv"] < 1e-13).all()
    assert (df["rv_common"] >= 0).all() and (df["rv_specific"] >= 0).all() and (df["rv_uncovered"] >= 0).all()


def test_check_reads_only_and_mutation_writing_check_is_caught(monkeypatch):
    if not (c_run.OUT / "C_sessions.parquet").exists():
        pytest.skip("выходы не записаны")
    before = c_run.dir_hash()
    assert c_run.main(["--check"]) == 0 and c_run.dir_hash() == before
    orig = c_run.check

    def writing_check(out):
        orig(out)
        (c_run.OUT / "tmp_mutant.txt").write_text("x")

    monkeypatch.setattr(c_run, "check", writing_check)
    try:
        with pytest.raises(AssertionError):
            c_run.main(["--check"])
    finally:
        (c_run.OUT / "tmp_mutant.txt").unlink(missing_ok=True)
    assert c_run.dir_hash() == before
