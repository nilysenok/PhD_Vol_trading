"""Л7.1: писатель FREEZE_v2.json (`scripts/fwd/freeze_v2.py`) на синтетике в tmp. Научные ключи = v1, `diff_v1` только технические; ссылки на v1 и первый замок.
Мутации: подмена научного ключа, иной sha входа, существующий v2, первый замок не «сбой», файл models_v2 с другим sha, лживый diff_v1, нетехнический ключ."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from test_fwd_l7_world import FIX, STAMP, make_povtor_world, write_lock1  # noqa: I001 (первым: sys.path на scripts/fwd)
import freeze_v2 as FV  # noqa: E402
import otkrytie_povtor as PT  # noqa: E402


def b(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_chestnyj_v2_nauchnye_klyuchi_ravny_v1_diff_tolko_tekhnicheskie(tmp_path):
    w = make_povtor_world(tmp_path)
    v1_before, lock_before = b(w.ctx.prev_freeze), b(w.lock1)
    out = tmp_path / "out/FREEZE_v2.json"
    out.parent.mkdir()
    v2 = FV.write(path=out, v1_path=w.ctx.prev_freeze, lock1_path=w.lock1, fix_commit=FIX, root=w.root, fresh_fn=lambda c: w.fresh)
    assert json.loads(out.read_text(encoding="utf-8")) == {**v2, "written_at": v2["written_at"]} and v2["v"] == 2
    assert all(v2[k] == w.v1[k] for k in PT.SCI)
    keys = [x["key"] for x in v2["diff_v1"]]
    assert {x["kind"] for x in v2["diff_v1"]} == {"технический"} and set(keys) <= set(PT.TECH) and "calendar" not in keys
    assert {"v", "code_sha256", "git", "outputs", "owner_yes", "prev_freeze", "prev_lock", "prev_commit", "fix_commit", "written_at"} <= set(keys)
    assert v2["prev_commit"] == "a" * 40 and v2["fix_commit"] == FIX and v2["outputs"]["lock"] == PT.LOCK2 and v2["outputs"]["models"] == w.v1["outputs"]["models"]
    assert v2["owner_yes"]["zh"] == "Ж-249" and v2["owner_yes"]["quote"] == PT.OWNER_QUOTE
    assert v2["prev_freeze"]["sha256"] == v1_before and v2["prev_lock"]["sha256"] == lock_before and v2["prev_lock"]["status"] == "сбой"
    assert b(w.ctx.prev_freeze) == v1_before and b(w.lock1) == lock_before and not list(out.parent.glob("tmp_*"))  # v1 и первый замок не тронуты
    assert PT.check_diff(w.v1, json.loads(out.read_text(encoding="utf-8"))) == []


def test_sushchestvuyushchij_v2_otkaz_i_fajl_ne_menyaetsya(tmp_path):
    w = make_povtor_world(tmp_path)
    out = tmp_path / "FREEZE_v2.json"
    out.write_text('{"v": 2}', encoding="utf-8")
    before = b(out)
    with pytest.raises(FV.Refusal, match="не перезаписывается"):
        FV.write(path=out, v1_path=w.ctx.prev_freeze, lock1_path=w.lock1, fix_commit=FIX, root=w.root, fresh_fn=lambda c: w.fresh)
    assert b(out) == before
    with pytest.raises(FV.Refusal, match="fix_commit"):
        FV.write(path=tmp_path / "x.json", v1_path=w.ctx.prev_freeze, lock1_path=w.lock1, fix_commit="", root=w.root, fresh_fn=lambda c: w.fresh)


@pytest.mark.parametrize("group,key,val", [("inputs_new_sha256", "data/fwd/sber_m10/MANIFEST.json", "f" * 64), ("inputs_archive_sha256", "data/regen/a.bin", "0" * 64),
                                           ("inputs_archive_sha256", "data/regen/extra.bin", "1" * 64), ("derived_archive_sha256", "data/fwd/models/k1.parquet", "2" * 64)])
def test_inoj_sha_vkhoda_otkaz(tmp_path, group, key, val):
    w = make_povtor_world(tmp_path)
    fresh = copy.deepcopy(w.fresh)
    fresh[group][key] = val  # входы пересчитаны и не совпали с v1
    out = tmp_path / "FREEZE_v2.json"
    with pytest.raises(FV.Refusal, match=group):
        FV.write(path=out, v1_path=w.ctx.prev_freeze, lock1_path=w.lock1, fix_commit=FIX, root=w.root, fresh_fn=lambda c: fresh)
    assert not out.exists()


def test_vkhod_propal_otkaz(tmp_path):
    w = make_povtor_world(tmp_path)
    fresh = copy.deepcopy(w.fresh)
    fresh["inputs_new_sha256"].pop("data/fwd/sber_m10/MANIFEST.json")
    with pytest.raises(FV.Refusal, match="inputs_new_sha256"):
        FV.build(w.v1, PT.lock_info(w.lock1, w.root), w.prev, fresh, FIX, STAMP)


@pytest.mark.parametrize("status", ["начато", "выполнено", None])
def test_pervyj_zamok_ne_sboj_otkaz(tmp_path, status):
    w = make_povtor_world(tmp_path)
    write_lock1(w.root, status=status)
    with pytest.raises(FV.Refusal, match="статусе"):
        FV.write(path=tmp_path / "FREEZE_v2.json", v1_path=w.ctx.prev_freeze, lock1_path=w.lock1, fix_commit=FIX, root=w.root, fresh_fn=lambda c: w.fresh)


def test_fajl_models_v2_s_drugim_sha_otkaz_a_skripty_mogut_menyatsya(tmp_path):
    w = make_povtor_world(tmp_path)
    fresh = copy.deepcopy(w.fresh)
    fresh["code_sha256"]["models_v2/m.py"] = "9" * 64
    with pytest.raises(FV.Refusal, match=r"models_v2/m\.py"):
        FV.build(w.v1, PT.lock_info(w.lock1, w.root), w.prev, fresh, FIX, STAMP)
    fresh = copy.deepcopy(w.fresh)
    fresh["code_sha256"]["models_v2/novyj.py"] = "9" * 64  # новый модуль в замыкании — тоже научный код изменился
    with pytest.raises(FV.Refusal, match="novyj"):
        FV.build(w.v1, PT.lock_info(w.lock1, w.root), w.prev, fresh, FIX, STAMP)
    assert FV.build(w.v1, PT.lock_info(w.lock1, w.root), w.prev, w.fresh, FIX, STAMP)["code_sha256"] == w.fresh["code_sha256"]


@pytest.mark.parametrize("key", PT.SCI)
def test_podmena_nauchnogo_klyucha_lovitsya(tmp_path, key):
    w = make_povtor_world(tmp_path)
    mut = copy.deepcopy(w.v2)
    mut[key] = {"podmena": 1} if not isinstance(mut[key], list) else mut[key] + ["x"]
    mut["diff_v1"] = PT.diff_list(w.v1, mut)  # даже честный diff_v1 не спасает: научный ключ ≠ v1
    assert any(f"научный ключ {key} отличается" in m for m in PT.check_diff(w.v1, mut))


def test_diff_v1_lzhyot_i_netekhnicheskij_klyuch(tmp_path):
    w = make_povtor_world(tmp_path)
    assert PT.check_diff(w.v1, w.v2) == []
    lie = {**w.v2, "diff_v1": [x for x in w.v2["diff_v1"] if x["key"] != "code_sha256"]}  # скрытое отличие кода
    assert any("diff_v1" in m for m in PT.check_diff(w.v1, lie))
    extra = {**w.v2, "tajnyj": 1}
    extra["diff_v1"] = PT.diff_list(w.v1, extra)
    assert any("нетехнический ключ отличается от v1: tajnyj" in m for m in PT.check_diff(w.v1, extra))
    out = {**w.v2, "outputs": {**w.v2["outputs"], "report": "other"}}
    out["diff_v1"] = PT.diff_list(w.v1, out)
    assert any("outputs" in m for m in PT.check_diff(w.v1, out))
