"""Ядро fastbt: сетки = c22a4b3, дымовой набор ворот (3 тикера × 2 стратегии × 2 года),
эквивалентность конфигураций на одном тикере, замок контрольного периода.
Запуск: python3 -m pytest fastbt/tests/test_core.py -q
"""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from fastbt import grids
from fastbt.config import Config


@pytest.fixture(scope="module")
def L():
    from fastbt.legacy import load_legacy
    return load_legacy()


def test_grids_match_legacy(L):
    for sid in ("S1", "S2"):
        assert grids.rm_grid(sid) == L.sw.expand_grid(L.v4.RM_GRIDS_V4[("Contrarian", "daily")])
    assert grids.rm_grid("S3") == L.v4.TREND_RM_GRID_V4
    s5 = [dict(r, _rm_type="pivot") for r in L.sw.expand_grid(L.s5r.S5_RM_PIVOT_DAILY)]
    s5 += [dict(r, _rm_type="atr") for r in L.sw.expand_grid(L.s5r.S5_RM_ATR_DAILY)]
    assert grids.rm_grid("S5") == s5
    s6 = [dict(r, _rm_type="atr") for r in L.sw.expand_grid(L.s6r.S6_RM_ATR_DAILY)]
    s6 += [dict(r, _rm_type="vwap") for r in L.sw.expand_grid(L.s6r.S6_RM_VWAP_DAILY)]
    assert grids.rm_grid("S6") == s6
    assert grids.exec_grid() == L.sw.expand_grid(L.v4.EXEC_GRID_DAILY)
    for cat in ("Contrarian", "Trend", "Range"):
        assert grids.b_grid(cat) == L.sw.expand_grid(L.sw.B_GRIDS_V3[cat])
    assert grids.C_THRESHOLDS_V41 == L.v41.C_THRESHOLDS_V41
    assert grids.C_RANGE_BANDS_V41 == L.v41.C_RANGE_BANDS_V41
    assert grids.C_THRESHOLDS == L.sw.C_THRESHOLDS
    assert grids.D_GATE_THRESHOLDS == L.v4.D_GATE_THRESHOLDS


V1_SKIP = ("Ъ ~23:20: ранг C всегда без текущего дня (аудит FA, Н1) — позиции C ворот v1 "
           "(c22a4b3) не воспроизводятся; печати v1 — docs/otchety/arhiv_do_audita/")


@pytest.mark.skip(reason=V1_SKIP)
def test_smoke_gate():
    """Дымовой набор: позиции = daily_positions_pre2026 для 2022–2023, все подходы."""
    from fastbt.gate import compare
    from fastbt.pipeline import run_all
    cfg = dataclasses.replace(Config.gate(), years_a=(2022, 2023), years_bcd=(2022, 2023))
    pos, _ = run_all(cfg, ["SBER", "IRAO", "MOEX"], ["S1", "S3"], log=lambda m: None)
    res = compare(pos, tickers=["SBER", "IRAO", "MOEX"], strategies=["S1_MeanRev", "S3_Donchian"])
    res = res[res["test_year"].isin([2022, 2023])]
    assert len(res) == 2 * 4 * 2
    assert (res["eq_share"] == 1.0).all(), res
    assert (res["only_new"] == 0).all() and (res["only_saved"] == 0).all(), res


@pytest.mark.skip(reason=V1_SKIP)
def test_equiv_one_ticker():
    from fastbt.equiv import run
    df = run(["SBER"], 25, seed=1)
    assert df["pos_eq"].all()
    assert df["ret_diff"].max() <= 1e-9
    assert df["sharpe_rel"].max() <= 1e-9


def test_stats_trades_equal_positions():
    from fastbt.batch import trades_one
    from fastbt.stats import NF, pos_from_trades, seg_stats_pos, seg_stats_trades
    from fastbt.unit import make_unit
    u = make_unit("MTLR", "S4")
    rng = np.random.default_rng(3)
    for _ in range(30):
        i, r = rng.integers(len(u.sigs)), rng.integers(len(u.rm))
        te, tx, td = trades_one(u, i, u.rmP[r], u.trend)
        a, b = np.zeros((len(u.nbars), NF)), np.zeros((len(u.nbars), NF))
        seg_stats_trades(te, tx, td, len(te), u.lr, u.segb, u.seg_of, u.n, 0.004, a)
        seg_stats_pos(pos_from_trades(te, tx, td, len(te), u.n), u.lr, u.segb, u.seg_of,
                      200, u.n, 0.004, b)
        assert np.array_equal(a, b)


def test_test_scope_locked(tmp_path, monkeypatch):
    from fastbt import dev
    monkeypatch.setattr(dev, "FINAL", tmp_path / "FINAL_DESIGN_strategies.md")
    with pytest.raises(dev.TestLocked):
        dev.check_scope("test")
    dev.check_scope("dev")
    (tmp_path / "FINAL_DESIGN_strategies.md").write_text("ok")
    dev.check_scope("test")
    with pytest.raises(SystemExit):
        monkeypatch.setattr(dev, "FINAL", tmp_path / "нет.md")
        dev.main(["--smoke", "--scope", "test"])


def test_msp_print_series():
    """Печатные ряды MSP_A/D_inst: EW по парам без комиссии (trading_ret до 1e-15)."""
    from fastbt import msp
    from fastbt.data import POS_PRE2026
    d = pd.read_parquet(POS_PRE2026, filters=[("tf", "=", "daily"), ("approach", "=", "A")])
    d["date"] = pd.to_datetime(d["date"])
    p = msp.portfolio(d[d["test_year"].between(2022, 2025)], msp.rate_restored(), 0.0)
    c = pd.read_csv(msp.MSP_CSV / "MSP_A_inst.csv", parse_dates=["date"]).set_index("date")
    assert (c["trading_ret"] - p.loc[c.index, "r"]).abs().max() < 1e-15
    assert abs(msp.sharpe(p.loc[c.index, "excess"]) - 1.3153) < 5e-5
