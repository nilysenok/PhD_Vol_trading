"""S-A-IMPL-1 (тестировщик): S(HAR, b=1) = признаки K1 на реальном ряду SBER; H_A(0) = K1; c = 0 ⇒ RV_b = RV побитно.
Известный ответ — архивный `daily.build_daily` (rv, rv_w, rv_m) и сохранённые коэффициенты K1 (`data/fwd/models/{,new/}k1.parquet`, только чтение)."""
import numpy as np
import pandas as pd
import pytest
import torch

from models_v2.s18 import a_feat as A
from s18_a_guard import ROOT, guard_real_files  # noqa: F401  (autouse)


@pytest.fixture(scope="module")
def F():
    return A.build_sber()


def ln_s_har(F):
    rows = np.arange(len(F.dates))
    with torch.no_grad():
        return A.ln_S(F, A.har_theta(), A.lag_idx(F, rows)).numpy()


def test_real_series_is_float64_and_full(F):
    assert F.r2.dtype == np.float64 and len(F.dates) == 3124 and F.r2.shape[1] == 53
    assert (F.n >= 30).all()


def test_s_har_equals_k1_windows_rel_1e12_and_nan_mask(F):
    ls = ln_s_har(F)
    for k, ref in enumerate((F.rv, F.rv_w, F.rv_m)):
        ok = np.isfinite(ref)
        # маска NaN: окно набрано ⇔ в сегменте не меньше RECT[k] строк
        assert (ok == (F.pos + 1 >= A.RECT[k])).all()
        S = np.exp(ls[ok, k])
        assert np.abs(S / ref[ok] - 1).max() <= 1e-12
        assert np.abs(ls[ok, k] - np.log(ref[ok])).max() <= 1e-12


def test_truncated_rows_are_finite_and_inside_segment(F):
    """Строки в начале сегмента: S конечен (усечение), и это среднее только своего сегмента."""
    ls = ln_s_har(F)
    i = int(np.flatnonzero((F.pos == 2))[1])       # третья строка второго сегмента
    assert F.seg[i - 2] == F.seg[i] != F.seg[i - 3]
    want = np.log(F.rv[i - 2:i + 1].mean())
    assert abs(ls[i, 1] - want) <= 1e-12
    assert abs(ls[i, 2] - want) <= 1e-12


def test_c_zero_rv_b_equals_rv_bitwise_and_torch_rel(F):
    E = np.ones((3, A.NS))
    got = A.rv_b_exact(F, E)
    for k in range(3):
        assert np.array_equal(got[:, k], F.rv)      # побитно
    assert np.abs(A.rv_b(F, torch.as_tensor(E)).numpy() / F.rv[:, None] - 1).max() <= 1e-12
    assert np.abs(A.slot_exp(A.har_theta()["c"]).numpy() - 1).max() == 0.0


def k1_table():
    p = ROOT / "data/fwd/models"
    k1 = pd.concat([pd.read_parquet(p / "k1.parquet"), pd.read_parquet(p / "new/k1.parquet")], ignore_index=True)
    return k1[~k1["d"].duplicated(keep="last")].reset_index(drop=True)


def test_h_a_zero_equals_k1_sigma2_on_all_saved_rows(F):
    """H_A(δ=0, a=HAR, b=1) = exp(b·[ln S, ln1p J] + s²/2) по сохранённым коэффициентам K1 ≤ 1e-12 отн."""
    k1 = k1_table()
    idx = F.dates.get_indexer(pd.DatetimeIndex(k1["d"]))
    assert (idx >= 0).all() and len(k1) >= 2198
    ls = ln_s_har(F)[idx]
    B = k1[["b_d", "b_w", "b_m"]].to_numpy()
    f = k1["b0"].to_numpy() + (B * ls).sum(1) + k1["b_J"].to_numpy() * np.log1p(F.j[idx])
    rel = np.abs(np.exp(f + k1["s2"].to_numpy() / 2) / k1["sigma2"].to_numpy() - 1)
    assert np.nanmax(rel) <= 1e-12 and np.isfinite(rel).all()


def test_h_a_zero_also_holds_for_a_nonzero_delta_difference_formula(F):
    """Разложение разд. 3.1: при a = HAR, b = 1 слагаемые β·ln S и β·ln S(HAR,1) сокращаются точно (≠ 0 только при отклонении)."""
    ls = ln_s_har(F)
    th = A.har_theta()
    th["omega"] = th["omega"] + 0.3
    rows = np.arange(len(F.dates))
    with torch.no_grad():
        moved = A.ln_S(F, th, A.lag_idx(F, rows)).numpy()
    assert np.abs(moved[F.pos >= 21, 2] - ls[F.pos >= 21, 2]).max() > 1e-3   # ω ≠ 0 меняет S, нулевой ω — нет
