"""S-T (Ж-256): таблицы на реальных сохранённых прогнозах = прежним S_A / S_C / S_R_TABLE и _YEARS до точности печати (12 значащих цифр); --check только читает."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import t_chk as C, t_table as TT


@pytest.fixture(scope="module")
def b():
    return TT.build()


def test_problems_empty_on_real(b):
    assert C.problems(b["t"], b["inp"]) == []


def test_missing_rows_and_counts(b):
    tb = b["t"]["table"].set_index("series")
    assert set(tb.index[tb["status"] == "не рассчитывался"]) == {"S-RA0", "S-RA1", "S-RCD"}
    ok = tb[tb["status"] == "рассчитан"]
    assert (ok["n"] == 2198).all() and len(ok) == 22
    assert tb.at["S-RCD", "role"] == "структурный S-R"
    assert np.isnan(tb.at["S-RA0", "qlike"])


def test_known_qlike_and_delta_values(b):
    tb = b["t"]["table"].set_index("series")
    assert abs(tb.at["K1", "qlike"] - 0.243684857367) < 1e-11
    assert abs(tb.at["S-RCK", "d_W5-G_pct"] - (-0.208776141455)) < 1e-9
    assert tb.at["S-RCK", "d_X1_pct"] < 0 < tb.at["S-CK", "d_X1_pct"]
    assert abs(tb.at["K2", "d_K3_pct"] - (-0.322220312411)) < 1e-9 and abs(tb.at["K3", "d_K2_pct"] - 0.323261928006) < 1e-9


def test_check_reads_only_and_mutant_writer_is_caught(b, tmp_path):
    rep = tmp_path / "rep"
    rep.mkdir()
    for n, v in b["tx"].items():
        (rep / n).write_text(v, encoding="utf-8")
    h0 = TT.tree_hash(rep)
    assert TT.check(rep) == [] and TT.tree_hash(rep) == h0
    (rep / TT.OUTS[0]).write_text("мутация: --check записал файл", encoding="utf-8")
    assert TT.tree_hash(rep) != h0
    assert any("таблицы" in e for e in TT.check(rep))


def test_real_outputs_equal_recomputation(b):
    for n, v in b["tx"].items():
        assert (TT.REP / n).read_text(encoding="utf-8") == v, n


def test_old_tables_present_and_series_covered(b):
    for f in ("S_A_TABLE.csv", "S_C_TABLE.csv", "S_R_TABLE.csv"):
        o = pd.read_csv(TT.REP / f, comment="#")
        ser = set(o[o["role"].isin(["main", "reference"])]["series"]) - {"ZB"}
        assert ser <= set(b["t"]["table"]["series"]), f
