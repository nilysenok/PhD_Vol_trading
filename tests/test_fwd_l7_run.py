"""Л7.3–Л7.4: повтор под замком r2 и режим без `--povtor` (синтетика, tmp; реальные замок, FREEZE, данные не трогаются).
Замок r2 ставится ДО чтения новых данных (№ 30: `Src` повтора смотрит на замок r2, а не на первый); первый замок и v1 побайтно не изменены после успеха, сбоя, нарушения заморозки;
без `--povtor` при существующем первом замке отказ, как прежде."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from test_fwd_l6_open import ROOT, Spy, fake_result, new_src_paths, snapshot  # noqa: I001 (первым: sys.path на scripts/fwd)
from test_fwd_l7_world import make_povtor_world  # noqa: E402
import otkrytie_harj_fwd as OP  # noqa: E402
import otkrytie_proverki as PV  # noqa: E402
import otkrytie_raschet as RS  # noqa: E402

from models_v2.s17 import fwd_src as S  # noqa: E402


def sh(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frozen(w) -> tuple:
    return sh(w.lock1), sh(w.ctx.prev_freeze), sh(w.ctx.freeze), w.lock1.stat().st_mtime_ns


def test_povtor_zamok_r2_do_chteniya_pervyj_zamok_i_v1_cely(tmp_path):
    w = make_povtor_world(tmp_path)
    before = frozen(w)
    spy = Spy(w.ctx, fake_result())
    assert OP.run(w.ctx, run_nodes=spy, dry=False) == 0
    assert spy.calls == [True] and frozen(w) == before  # к чтению замок r2 есть (№ 30); первый замок и v1 не тронуты
    lk = json.loads(w.ctx.lock.read_text(encoding="utf-8"))
    assert lk["status"] == "выполнено" and lk["povtor"] is True and lk["freeze_sha256"] == sh(w.ctx.freeze) and not (w.ctx.lock.stat().st_mode & 0o222)
    assert lk["prev_lock_sha256"] == sh(w.lock1) and lk["prev_freeze_sha256"] == sh(w.ctx.prev_freeze) and w.ctx.lock.name == "OPENED_HARJ_FWD_r2.json"
    assert len(lk["outputs_sha256"]) == len(OP.NEW_NODES) + len(OP.CSV) + 1
    snap = snapshot(w.root)
    spy2 = Spy(w.ctx, fake_result())
    assert OP.run(w.ctx, run_nodes=spy2, dry=False) == 2 and spy2.calls == [] and snapshot(w.root) == snap  # второго повтора нет


def test_povtor_sboj_posle_zamka_r2_pervyj_zamok_cel_vtorogo_povtora_net(tmp_path):
    w = make_povtor_world(tmp_path)
    before = frozen(w)
    assert OP.run(w.ctx, run_nodes=Spy(w.ctx, boom=ValueError("снова сбой")), dry=False) == 1
    lk = json.loads(w.ctx.lock.read_text(encoding="utf-8"))
    assert lk["status"] == "сбой" and "снова сбой" in lk["prichina"] and not list(w.ctx.rep_dir.glob("FWD_*")) and frozen(w) == before
    snap = snapshot(w.root)
    assert OP.run(w.ctx, run_nodes=Spy(w.ctx, fake_result()), dry=False) == 2 and snapshot(w.root) == snap


def test_povtor_narushenie_zamorozki_pervyj_zamok_cel(tmp_path):
    w = make_povtor_world(tmp_path)
    before, base = frozen(w), w.ctx.closure
    spy = Spy(w.ctx, fake_result())

    def run_nodes(c, fz):
        w.ctx.closure = lambda: base() | {"models_v2/s17/fwd_pozdnij.py"}
        return spy(c, fz)

    assert OP.run(w.ctx, run_nodes=run_nodes, dry=False) == 3
    assert json.loads(w.ctx.lock.read_text(encoding="utf-8"))["status"].startswith("нарушение заморозки") and frozen(w) == before


def test_mutatsiya_zamok_posle_chteniya_povtora_lovitsya(tmp_path):
    """Первый замок есть, замка r2 нет: реальный `Src` повтора (lock = r2) отказывает читать цены; `Src` на первом замке (мутант) читал бы."""
    w = make_povtor_world(tmp_path)
    paths = new_src_paths(w.ctx)
    assert paths.lock == w.ctx.lock != w.lock1 and w.lock1.exists() and not w.ctx.lock.exists()
    cut = pd.Timestamp("2026-03-02")
    with pytest.raises(S.SealedError):
        S.Src(paths).bars("SBER", cut)
    assert S.Src(S.Paths(**{**paths.__dict__, "lock": w.lock1})).bars("SBER", cut)["close"].tolist() == [300.0, 301.0, 302.0, 303.0]  # мутант: смотрит на первый замок

    def reader(c, fz):
        assert S.Src(paths).bars("SBER", cut)["close"].tolist() == [300.0, 301.0, 302.0, 303.0]  # под замком r2 — читается
        return fake_result()

    assert OP.run(w.ctx, run_nodes=reader, dry=False) == 0


def test_real_nodes_src_povtora_smotrit_na_zamok_r2(tmp_path):
    w = make_povtor_world(tmp_path)
    src = RS.real_nodes(w.ctx).Src()
    assert src.p.lock == w.ctx.lock != S.Paths().lock and not src.is_open()  # мутант (замок по умолчанию = первый) здесь был бы «открыт»
    OP.acquire_lock(w.ctx, {})
    assert RS.real_nodes(w.ctx).Src().is_open()


def test_bez_povtora_pri_sushchestvuyushchem_pervom_zamke_otkaz_kak_prezhde(tmp_path):
    from test_fwd_l6_open import make_world
    from test_fwd_l7_world import write_lock1
    ctx = make_world(tmp_path)
    write_lock1(ctx.root)
    assert not ctx.povtor and ctx.freeze.name == "FREEZE.json" and ctx.lock.name == "OPENED_HARJ_FWD.json" and ctx.lock.exists()
    snap, spy = snapshot(ctx.root), Spy(ctx, fake_result())
    assert any("замок OPENED_HARJ_FWD.json уже есть" in m for m in PV.preflight(ctx))
    assert OP.run(ctx, run_nodes=spy, dry=False) == 2 and spy.calls == [] and snapshot(ctx.root) == snap


def test_bez_povtora_v2_ne_chitaetsya_i_razbor_argumentov(tmp_path, monkeypatch):
    got = []
    monkeypatch.setattr(OP, "import_chain", lambda: None)
    monkeypatch.setattr(OP, "run", lambda ctx, dry: got.append((ctx, dry)) or 0)
    for argv in (["--otkryt"], ["--povtor"], ["--povtor", "--otkryt"], []):
        assert OP.main(argv) == 0
    (c1, d1), (c2, d2), (c3, d3), (c4, d4) = got
    assert (c1.povtor, d1, c1.freeze.name, c1.lock.name) == (False, False, "FREEZE.json", "OPENED_HARJ_FWD.json")
    assert (c2.povtor, d2, c2.freeze.name, c2.lock.name) == (True, True, "FREEZE_v2.json", "OPENED_HARJ_FWD_r2.json")
    assert (c3.povtor, d3) == (True, False) and (c4.povtor, d4) == (False, True)
    assert c2.prev_lock == c1.lock == ROOT / "data/fwd/models/OPENED_HARJ_FWD.json" and c2.prev_freeze == c1.freeze


@pytest.mark.skipif(not (ROOT / "data/fwd/models/OPENED_HARJ_FWD.json").exists(), reason="первого замка нет")
def test_realnyj_pervyj_zamok_i_v1_tolko_chtenie_sha():
    """Только sha256 байтов (значения не разбираются): первый замок и v1 в том виде, как их зафиксировал Ж-249."""
    assert sh(ROOT / "data/fwd/models/OPENED_HARJ_FWD.json").startswith("1869be4539c2")
    assert sh(ROOT / "docs/otchety/harj_fwd/FREEZE.json").startswith("4c9085d64b19")
