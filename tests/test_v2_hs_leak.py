"""Ж-212 (G14): главный тест обрезки на новых строках. (1) Таблицы направлений: trunc_max_diff == 0 на каждой строке (3 даты × ряд, переобучение с испорченным миром после 18:50 d).
(2) Контрольный прогон на реальном мире: B (XGB, Bf) и C (XGB, Cc) на одной дате — исходный и испорченный прогнозы совпадают; подложенное «закрыто + 1» в блоках ловится."""
from __future__ import annotations

import pandas as pd
import pytest

from models_v2.s8 import hs_ctx, hs_leak, hs_paths as P

CSVS = ("HAR_FEATURES_B.csv", "RESIDUAL_QLIKE_C.csv", "COMPONENT_D.csv", "CONDITIONAL_E.csv", "MOE_F.csv")


@pytest.mark.parametrize("csv", CSVS)
def test_every_new_row_has_zero_truncation_diff(csv):
    p = P.REP / csv
    if not p.exists():
        pytest.skip("направление не запускалось: %s" % csv)
    t = pd.read_csv(p, comment="#")
    if "trunc_max_diff" not in t.columns:
        pytest.skip("направление не запускалось (строк с прогнозами нет): %s" % csv)
    assert len(t) > 0 and (t["trunc_max_diff"] == 0).all(), t[["model", "trunc_max_diff"]]


@pytest.fixture(scope="module")
def ctx():
    if not (P.OUT / "har_pit.npz").exists():
        pytest.skip("нет данных мира")
    return hs_ctx.build()


def test_real_world_b_xgb_bf_and_c_xgb_cc_one_date_is_bitwise_unchanged(ctx):
    from models_v2.s8 import hs_cstage as C
    d = 2000
    assert hs_leak.trunc_diff(ctx, "XGB", ("Bf",), d) == 0.0
    assert hs_leak.trunc_c(ctx, C.x_builder(ctx, "S7"), "XGB", 0.0, (), d) == 0.0


def test_mutation_closed_plus_one_leaks_in_real_blocks(ctx):
    from models_v2.s8 import hs_feat
    F, d = ctx["w"]["F"], 2000
    Fc, mlc = hs_leak.corrupt(F, ctx["ml"], d)
    lbc, _ = hs_feat.har_pit(Fc)
    ok = hs_feat.block_e(Fc, lbc)[0][:d + 1]
    bad = hs_feat.block_e(Fc, lbc, extra=1)[0][:d + 1]
    assert (hs_feat.block_e(F, ctx["lb"])[0][:d + 1] == ok)[~pd.isna(ok)].all() and not (bad == hs_feat.block_e(F, ctx["lb"], extra=1)[0][:d + 1])[~pd.isna(bad)].all()
