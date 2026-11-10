"""Л4: cfg8, ПОЛНЫЙ ход по архивным датам (5 seeds, ≈ 80 с) — реальный архив; цен новых файлов не читает.
Гейты: все 2018 строк = `3/cfg8.parquet` V7 и `data/fwd/models/cfg8.parquet` побитно; число выбранных λ по seeds = CHECKPOINT V7 (`3.cfg8.lam_selected`);
даты переоценок = первый день месяца индекса; λ блока на обрезках (первая и последняя из 38) = λ полного хода по seeds. Остальное — `test_fwd_l4_cfg8.py`."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from models_v2.s17 import fwd_cfg8 as F8, fwd_src as S, fwd_store as FS  # noqa: I001  (fwd_cfg8 первым: torch до lightgbm)
from models_v2.s13 import v6_1_wf as W6
from models_v2.s14 import v7_1m_feat as FE, v7_1m_wf as WF, v7_3_cfg as C8, v7_2_run as R2

R = S.ROOT
ARCH = R / "data/regen/harj_hybrid_v7/3/cfg8.parquet"
HAVE = ARCH.exists() and F8.K23.exists() and F8.OUT.exists() and S.Paths().new_sber.exists() and (R / "data/regen/harj_hybrid_v7/stageA/tcn_w5_2026_s4.npz").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="нет архива V7, k23.parquet Л2, cfg8.parquet Л4 или begin новых файлов")


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    return S.Src(S.Paths(lock=tmp_path_factory.mktemp("l4f") / "no_lock.json"))


@pytest.fixture(scope="module")
def full(src):
    inp = F8.inputs_at(S.ARCHIVE_END, src)
    zm = inp["zmap"]
    df, per = C8.run_cfg(inp["tab"], FE.build_Z(inp["tab"]).to_numpy(float), lambda s: zm[s])
    return df, per, inp["tab"]


def test_polnyj_hod_arhivnye_stroki_pobitno(full):
    df = full[0]
    pd.testing.assert_frame_equal(df, pd.read_parquet(ARCH), check_exact=True)
    pd.testing.assert_frame_equal(df, pd.read_parquet(F8.OUT), check_exact=True)
    assert len(df) == 2018 and df["d"].iloc[-1] == pd.Timestamp("2026-02-26")


def test_lambda_po_seeds_i_daty_pereotsenok(full):
    _, per, tab = full
    cp = json.loads((R / "docs/otchety/harj_hybrid_v7/CHECKPOINT.json").read_text(encoding="utf-8"))
    assert R2.lam_counts(per) == cp["3"]["cfg8"]["lam_selected"]
    first = [tab.index[tab.index.to_period("M") == m][0] for m in tab.index.to_period("M").unique() if m.start_time >= pd.Timestamp(WF.FIRST_REFIT)]
    assert [tab.index[p] for p in W6.refit_positions(tab.index, WF.FIRST_REFIT)] == first       # правило: первый торговый день месяца
    for p in per:
        assert [f["d0"] for f in p["fits"] if f["fitted"]][-14:] == first[-14:] and first[-1] == pd.Timestamp("2026-02-02")


@pytest.mark.parametrize("cut", ["2025-12-30", "2026-02-26"])
def test_lambda_bloka_na_obrezke_ravna_lambda_polnogo_hoda(cut, src, full):
    store = FS.build_store(pd.Timestamp(cut), src)
    inp = F8.inputs_at(pd.Timestamp(cut), src, F8.mask_k23(pd.read_parquet(F8.K23), cut), store, FS.build_xw5(store))
    df = F8.build_cfg8(pd.Timestamp(cut), src, inp)
    d0 = df["d"].iloc[0]
    want = np.array([[f["lam"] for f in p["fits"] if f["d0"] == d0][0] for p in full[1]], float)
    assert np.array_equal(np.array(df.attrs["lam"]), want) and want.shape == (5, 2) and d0.month == pd.Timestamp(cut).month
