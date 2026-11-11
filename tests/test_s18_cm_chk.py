"""S-C-IMPL-1/3 (тестировщик): `--check` / `--check-pit` только читают (хэш всех выходов до и после, класс № 123), 𝓜 = 1 побитно на реальных 2198 датах при нулевых параметрах."""
import hashlib

import numpy as np
import pytest
import torch

from models_v2.s18 import cm_feat, cm_feat_chk, cm_layers as L, cm_model as M, cm_walk_chk
from s18_cm_guard import CM_DIR, GUARDED, ROOT, guard_real_files  # noqa: F401  (autouse)

WATCH = [CM_DIR, ROOT / "data/regen/harj_struct/C", ROOT / "data/regen/harj_x1/W5"]


def digest():
    h = hashlib.sha256()
    for p in GUARDED + [f for d in WATCH for f in sorted(d.rglob("*")) if f.is_file()]:
        h.update(str(p).encode() + p.read_bytes())
    return h.hexdigest()


@pytest.mark.parametrize("fn", [cm_feat_chk.check, cm_feat_chk.check_pit, cm_walk_chk.check, cm_walk_chk.check_pit])
def test_check_returns_clean_and_changes_no_output(fn):
    if not cm_feat.FEAT_PATH.exists():
        pytest.skip("feat.parquet не записан")
    before = digest()
    assert fn() == []
    assert digest() == before


def test_mutation_check_that_writes_is_caught_by_digest(monkeypatch):
    before = digest()
    orig = cm_feat_chk.check

    def writing():
        out = orig()
        (CM_DIR / "_mutant_tmp.txt").write_text("x")
        return out
    try:
        monkeypatch.setattr(cm_feat_chk, "check", writing)
        cm_feat_chk.check()
        assert digest() != before
    finally:
        (CM_DIR / "_mutant_tmp.txt").unlink(missing_ok=True)
    assert digest() == before


def test_real_2198_dates_zero_parameters_M_exactly_one_for_all_trained_variants():
    if not cm_feat.FEAT_PATH.exists():
        pytest.skip("feat.parquet не записан")
    x, _ = L.inputs()
    feat = cm_feat.load_feat()
    assert len(x) == 2198
    z = torch.zeros(M.NP, dtype=torch.float64)
    for var in M.TRAINED:
        V = M.build_data(x, feat, lag=var == "CKL").view(np.arange(len(x)))
        m = M.forward(z, V, var)["M"].numpy()
        assert (m == 1.0).all() and ((x["C"].to_numpy(float) * m) == x["C"].to_numpy(float)).all(), var
