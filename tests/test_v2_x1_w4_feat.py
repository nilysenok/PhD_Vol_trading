"""W4-B1, признаки S и Σ на продолжении (W4_SPEC 2.4, гейты B.G1–B.G2): архив побитно = файлам V6 (S — побитно, Σ — 1e-12); PIT: строка дня d по cut = d равна полному прогону.
Мутации — в `test_v2_x1_w4_feat_mut.py`. Реальные файлы только читаются (autouse-хэш); подмен модулей здесь нет."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models_v2.s13 import v6_1_feat as FE
from models_v2.s16 import x1_w4_feat as F
from models_v2.s17.fwd_src import Src

ROOT = Path(F.__file__).resolve().parents[2]
REAL = [F.FEAT / "S_ext.parquet", F.FEAT / "SIGMA_ext.parquet", F.VOLCSV, ROOT / "docs/otchety/harj_x1/ALL_TRIALS.csv"]
PIT_DATES = ["2026-03-02", "2026-10-01"]


def real_state() -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in REAL}


@pytest.fixture(autouse=True)
def real_files_untouched():
    before = real_state()
    yield
    assert real_state() == before


@pytest.fixture(scope="module")
def ctx() -> dict:
    src, h = Src(), F.load_h()
    full = F.rows_at(F.NEW_LAST, src, h)
    return {"src": src, "h": h, "ext": F.read_ext(), "dates": F.new_dates(), "stored": F.stored(), "full": full}


def test_archive_rows_bitwise_s_and_sigma_within_1e12(ctx) -> None:
    (S0, G0), (Se, Ge) = ctx["stored"], ctx["ext"]
    np.testing.assert_array_equal(Se.loc[S0.index].to_numpy(float), S0.to_numpy(float))   # S побитно (NaN == NaN)
    assert list(Se.columns) == list(S0.columns) and len(Se) == len(S0) + len(ctx["dates"])
    a = Ge.loc[G0.index][G0.columns].to_numpy(float)
    assert np.array_equal(np.isnan(a), np.isnan(G0.to_numpy(float)))
    assert np.nanmax(np.abs(a - G0.to_numpy(float))) <= 1e-12


def test_archive_recompute_by_cut_equals_files_and_gate_ok(ctx) -> None:
    res = F.check_archive(ctx["src"])
    assert res["ok"] and res["S_recompute_bitwise"] and res["S_ext_bitwise"]
    assert res["G_recompute_maxdiff"] <= 1e-12 and res["n_arch"] == len(ctx["stored"][0])


def test_new_days_equal_x1_dates_and_calendar_is_continuous(ctx) -> None:
    Se = ctx["ext"][0]
    new = Se.index[Se.index > F.ARCH_LAST]
    assert new.equals(ctx["dates"]) and new.min() > F.ARCH_LAST and new.max() == F.NEW_LAST and not Se.isna().any().any()


@pytest.mark.parametrize("d", PIT_DATES)
def test_pit_row_by_cut_d_equals_full_run_bitwise(ctx, d) -> None:
    d = pd.Timestamp(d)
    S, G = F.rows_at(d, ctx["src"], ctx["h"])
    Sf, Gf = ctx["full"]
    assert S.index[-1] == d and Sf.index[-1] == F.NEW_LAST
    assert F.bitwise(S.loc[[d]], Sf.loc[[d]]) and F.bitwise(G.loc[[d]], Gf.loc[[d]])
    Se, Ge = ctx["ext"]
    assert F.bitwise(S.loc[[d]], Se.loc[[d]]) and F.bitwise(G.loc[[d]][Ge.columns], Ge.loc[[d]])
    assert F.bitwise(S, Sf.loc[:d]) and F.bitwise(G, Gf.loc[:d])                          # и все строки до d: будущие бары прошлого не меняют


def test_pit_gate_honest_passes_on_sample(ctx) -> None:
    res = F.check_pit(ctx["src"], dates=PIT_DATES)
    assert res["ok"] and res["n_bad"] == 0 and res["d_equals_x1"] and res["max_abs_diff"] == 0.0


def test_s7_window_is_exactly_20_sessions_back() -> None:
    """Известный ответ: объём 1 на 25 днях и 21 на дне 26 → окно 20 включая d; до 20-й сессии NaN."""
    v = pd.Series([1.0] * 25 + [21.0], index=pd.bdate_range("2019-01-01", periods=26))
    r = FE.vol_ratio(v)
    assert r.iloc[:19].isna().all() and abs(r.iloc[19] - 1.0) < 1e-15
    assert abs(r.iloc[-1] - 21.0 / ((19 * 1.0 + 21.0) / 20)) < 1e-12


def attr_assignments(src: str, mods: set) -> list:
    out = []
    for n in ast.walk(ast.parse(src)):
        for t in (n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, (ast.AugAssign, ast.AnnAssign)) else []):
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id in mods:
                out.append(ast.unparse(n))
    return out


def test_no_module_attribute_assignment_in_w4_feat_tests_and_scanner_sees_one() -> None:
    for p in sorted(ROOT.glob("tests/test_v2_x1_w4_feat*.py")):
        src = p.read_text(encoding="utf-8")
        mods = {n.asname or n.name for t in ast.walk(ast.parse(src)) if isinstance(t, (ast.ImportFrom, ast.Import)) for n in t.names}
        assert attr_assignments(src, mods) == [], p.name
    assert attr_assignments("from models_v2.s13 import v6_1_feat as FE\nFE.vol_ratio = None\n", {"FE"})
