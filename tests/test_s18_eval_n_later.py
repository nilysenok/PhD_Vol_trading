"""Ж-259: N оценки берётся из СВОЕГО ключа регистрации, а не максимум по `S.*.registered` (позднейшая регистрация не меняет закрытый этап)."""
from __future__ import annotations

import json
import shutil

import pytest

from models_v2.s18 import rd_eval as RE
from models_v2.s18 import rd_eval_cfg as RC
from models_v2.s18 import z_eval as ZE
from models_v2.s18 import z_eval_cfg as ZC

LATER = {"n": 900, "n_wide": 3500, "registered_at": "12.11.2026 01:00:00"}


def _ckpt(tmp_path, own_key=None, own=None):
    p = tmp_path / "CHECKPOINT.json"
    cp = {"S.X.registered": dict(LATER)}
    if own_key:
        cp[own_key] = own
    p.write_text(json.dumps(cp), encoding="utf-8")
    return p


def _max_all_keys(ckpt):
    cp = json.loads(ckpt.read_text(encoding="utf-8"))
    ks = [v for k, v in cp.items() if k.startswith("S.") and k.endswith(".registered") and isinstance(v, dict)]
    return max(v["n"] for v in ks), max(v["n_wide"] for v in ks)


def test_z_registered_n_own_key_not_later(tmp_path):
    p = _ckpt(tmp_path, "S.Z.registered", {"n": 855, "n_wide": 3415})
    assert ZC.registered_n(p) == (855, 3415)


def test_rd_registered_n_own_key_not_later(tmp_path):
    p = _ckpt(tmp_path, "S.RCD.registered", {"n": 858, "n_wide": 3421})
    assert RC.registered_n(p) == (858, 3421)


def test_no_own_key_refuses(tmp_path):
    with pytest.raises(SystemExit):
        ZC.registered_n(_ckpt(tmp_path))
    with pytest.raises(SystemExit):
        RC.registered_n(_ckpt(tmp_path))


def test_mutation_max_over_all_keys_is_caught(tmp_path, monkeypatch):
    pz = _ckpt(tmp_path, "S.Z.registered", {"n": 855, "n_wide": 3415})
    monkeypatch.setattr(ZC, "registered_n", lambda ckpt=None: _max_all_keys(pz))
    assert ZC.registered_n(pz) == (900, 3500) != (855, 3415)
    monkeypatch.undo()
    assert ZC.registered_n(pz) == (855, 3415)
    pr = _ckpt(tmp_path, "S.RCD.registered", {"n": 858, "n_wide": 3421})
    monkeypatch.setattr(RC, "registered_n", lambda ckpt=None: _max_all_keys(pr))
    assert RC.registered_n(pr) != (858, 3421)


def _with_later_key(tmp_path, name):
    src = ZC.CKPT
    dst = tmp_path / name
    shutil.copyfile(src, dst)
    cp = json.loads(dst.read_text(encoding="utf-8"))
    cp["S.X.registered"] = dict(LATER)
    dst.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return dst


def test_z_check_passes_with_later_registration_battle_tables_read_only(tmp_path):
    before = ZC.CKPT.read_bytes()
    p = _with_later_key(tmp_path, "z.json")
    assert ZE.check(ZC.REP, p) == []
    assert ZC.CKPT.read_bytes() == before


def test_rd_check_passes_with_later_registration_battle_tables_read_only(tmp_path):
    before = RC.CKPT.read_bytes()
    p = _with_later_key(tmp_path, "rd.json")
    assert RE.check(RC.REP, p) == []
    assert RC.CKPT.read_bytes() == before


def test_mutation_max_over_all_keys_breaks_check(tmp_path, monkeypatch):
    p = _with_later_key(tmp_path, "m.json")
    monkeypatch.setattr(ZC, "registered_n", lambda ckpt=None: _max_all_keys(p))
    assert ZE.check(ZC.REP, p) != []
