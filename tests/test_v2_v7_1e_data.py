"""V7-1a-E, G13–G14: модули v7_1e_* не открывают данные SBER при импорте и вне `pack_from_npz` (AST + перезагрузка с запретом чтения);
реальные X_w1/X_w5/store_raw — только формы и маски, без обучения."""
import ast
import builtins
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from models_v2.s14 import v7_0_seq as SQ
from models_v2.s14 import v7_1e_net as NET
from models_v2.s14 import v7_1e_pre as P
from models_v2.s14 import v7_1e_rep as R

ROOT = Path(__file__).resolve().parents[1]
MODS = ["v7_1e_net", "v7_1e_pre", "v7_1e_rep", "v7_1e_e0", "v7_1e_syn"]
READERS = {"load", "read_parquet", "read_csv", "open", "loadtxt", "read_json", "read_feather"}
OUT = ROOT / "data" / "regen" / "harj_hybrid_v7"


def reader_calls(path: Path):
    """-> [(имя вызова, функция-владелец или None)] для всех вызовов чтения файлов в модуле."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    res = []

    def walk(node, owner):
        for ch in ast.iter_child_nodes(node):
            o = ch.name if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) else owner
            if isinstance(ch, ast.Call):
                f = ch.func
                nm = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if nm in READERS:
                    res.append((nm, owner))
            walk(ch, o)
    walk(tree, None)
    return res


def test_reads_only_inside_pack_from_npz():
    allowed = {("v7_1e_pre", "pack_from_npz")}
    for m in MODS:
        for nm, owner in reader_calls(ROOT / "models_v2" / "s14" / (m + ".py")):
            assert (m, owner) in allowed, (m, nm, owner)
    assert any(owner == "pack_from_npz" for _, owner in reader_calls(ROOT / "models_v2" / "s14" / "v7_1e_pre.py"))


def test_reader_detector_catches_module_level_read(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text("import numpy as np\nZ = np.load('x.npz')\n", encoding="utf-8")
    assert reader_calls(p) == [("load", None)]


@pytest.mark.parametrize("mod", MODS)
def test_import_opens_no_data(mod, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("чтение данных при импорте")
    for obj, nm in ((np, "load"), (pd, "read_parquet"), (pd, "read_csv")):
        monkeypatch.setattr(obj, nm, boom)
    monkeypatch.setattr(builtins, "open", boom)
    importlib.reload(importlib.import_module("models_v2.s14." + mod))


need_data = pytest.mark.skipif(not (OUT / "X_w5.npz").exists(), reason="нет X_w*.npz листа 0")


@need_data
@pytest.mark.parametrize("W", [1, 5])
def test_real_npz_shapes_masks_and_untrained_forward(W):
    pk = P.pack_from_npz(W, OUT)
    n = len(pk["days"])
    assert pk["X"].shape == (n, W, SQ.SLOTS, SQ.K) and pk["mask"].shape == (n, W, SQ.SLOTS) and pk["r2"].shape == (n, SQ.SLOTS)
    st = np.load(OUT / "store_raw.npz")
    assert np.array_equal(pk["mask"][:, -1], st["present"]) and np.array_equal(np.isfinite(pk["r2"]), st["present"])
    assert (pk["X"][~pk["mask"]] == 0).all() and np.isfinite(pk["X"]).all() and pk["norm_ok"].sum() == n - (SQ.MIN_SESS - 1)
    if W == 5:
        assert not pk["mask"][0, :-1].any()                    # первый день: дней до начала истории нет — маска 0
    rows = P.train_rows(pk, 2018)
    assert pd.DatetimeIndex(pk["days"])[rows].max() < pd.Timestamp(2018, 1, 1) and len(rows) > 500
    y, ok = P.targets(pk, rows, P.eps_pos(pk, rows))
    assert ok.sum() > 0.9 * len(rows) * 52 and np.isfinite(y[ok]).all() and P.eps_pos(pk, rows) > 0
    idx = np.array([400, 1500, 2600])
    for kind in ("tcn", "cnn"):                                # случайные веса: только форма и конечность, без обучения
        torch.manual_seed(0)
        z = R.encode(NET.build(kind).eval(), pk["X"][idx], pk["mask"][idx])
        assert z.shape == (3, 16) and np.isfinite(z).all()
