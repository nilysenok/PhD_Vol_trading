"""W8-B, оценка «от начала до конца» на копии (гейт B.G3; mutations.md № 30, 103, 123, 125, 158): запись во временную папку — ключ `W8.stage1` (с `written_at`, хэшами входов/выходов/кода) пишется ПЕРВЫМ, до таблиц с p;
`check` на копии чист; подмена числа в таблице, Σ вкладов, N Холма, константы порогов, F в днях gap = 0, подмена CHECKPOINT ловятся; `check` только читает (хэши до и после); расхождение входов — SystemExit.
Реальные выходы только читаются (autouse-хэши, сравнение содержимого до и после); реальный `check()` — чист."""
from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_w8_eval as E
from models_v2.s16 import x1_w8_eval_t as T


@pytest.fixture(autouse=True)
def real_files_untouched():
    def st():
        s = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (E.REG.TRIALS, E.CKPT) if p.exists()}
        s.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(E.REP.glob("W8_*")) if p.is_file()})
        s["W8"] = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(E.OUT.rglob("*")) if p.is_file()} if E.OUT.exists() else {}
        return s
    before = st()
    yield
    assert st() == before


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w8e")
    out, rep = tmp / "W8", tmp / "rep"
    out.mkdir()
    rep.mkdir()
    for p in E.OUT.glob("*.parquet"):
        shutil.copy(p, out / p.name)
    shutil.copy(E.CKPT, tmp / "CHECKPOINT.json")
    order = []
    real = E.merge_checkpoint

    def spy(ckpt, t, tx, out_=E.OUT, now=None):
        order.append(sorted(p.name for p in rep.glob("W8_*")))                         # какие таблицы уже лежат в момент записи ключа
        return real(ckpt, t, tx, out_, now)
    mp = pytest.MonkeyPatch()
    mp.setattr(E, "merge_checkpoint", spy)
    try:
        t = E.write(rep, tmp / "CHECKPOINT.json", out)
    finally:
        mp.undo()
    return dict(tmp=tmp, out=out, rep=rep, ck=tmp / "CHECKPOINT.json", order=order, t=t)


def chk(env, **kw):
    return E.check(env["rep"], env["ck"], env["out"])


def test_stage_key_written_before_any_table_with_p_and_check_on_copy_is_clean(env, capsys) -> None:
    assert env["order"] == [[]]                                                       # № 30: ключ записан, когда ни одной таблицы с p ещё нет
    cp = json.loads(env["ck"].read_text(encoding="utf-8"))["W8.stage1"]
    assert cp["n_holm"] == 814 and cp["n_holm_wide"] == 3333 and cp["configurations"] == 2 and cp["shifts"] == 19 and cp["written_at"]
    assert set(cp["outputs"]) == set(E.OUTS) and len(cp["inputs"]) == 2 + 19 + 1 and "w5g.parquet" in cp["inputs"] and "x1_w8_eval.py" in cp["code"]
    assert cp["inputs"]["w8g.parquet"] == E._sha(env["out"] / "w8g.parquet") and cp["decision"] == env["t"]["decision"]
    assert chk(env) == [] and "решение:" in capsys.readouterr().out


def test_copy_tables_equal_real_tables_and_real_check_is_clean(env) -> None:
    for n in E.OUTS:
        assert (env["rep"] / n).read_bytes() == (E.REP / n).read_bytes(), n           # пересчёт на копии = зафиксированные реальные таблицы
    assert E.check() == []                                                            # B.G3 на реальных файлах: только чтение


def test_check_only_reads_hash_before_and_after(env) -> None:
    h0 = E.tree_hash(env["rep"], env["ck"], env["out"])
    assert chk(env) == [] and E.tree_hash(env["rep"], env["ck"], env["out"]) == h0 and len(h0) == 8 + 1 + 21


def test_check_that_writes_is_caught(env, monkeypatch) -> None:
    f = env["rep"] / "W8_TABLE.csv"
    keep = f.read_bytes()
    real = E.compute

    def writing(inp):
        f.write_bytes(keep + b"#x\n")                                                  # мутация № 123: check пишет в таблицу
        return real(inp)
    try:
        monkeypatch.setattr(E, "compute", writing)
        assert any("--check изменил выходы" in b for b in chk(env))
    finally:
        monkeypatch.undo()
        f.write_bytes(keep)
    assert chk(env) == []


def test_substituted_number_in_table_and_guard_are_caught(env) -> None:
    tab, grd = env["rep"] / "W8_TABLE.csv", env["rep"] / "W8_GUARD.csv"
    for f, old, new in ((tab, next(ln for ln in tab.read_text(encoding="utf-8").splitlines() if ln.startswith("main,X1,")), None), (grd, "0.415", "0.600")):
        keep = f.read_text(encoding="utf-8")
        assert old in keep
        try:
            f.write_text(keep.replace(old, old.rsplit(",", 1)[0] + ",0.123456" if new is None else new, 1), encoding="utf-8")      # QLIKE X1 / порог У1 подменены
            assert any("%s: текст ≠ пересчёту" % f.name in b for b in chk(env)), f.name
        finally:
            f.write_text(keep, encoding="utf-8")
    assert chk(env) == []


def test_checkpoint_substitution_and_threshold_n_identity_mutations_are_caught(env, monkeypatch) -> None:
    cp = json.loads(env["ck"].read_text(encoding="utf-8"))
    bad = json.loads(json.dumps(cp))
    bad["W8.stage1"]["outputs"]["W8_GUARD.csv"] = "0" * 64                            # хэш выхода в ключе подменён
    env["ck"].write_text(json.dumps(bad, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    try:
        assert any("CHECKPOINT W8.stage1 ≠ пересчёту" in b for b in chk(env))
    finally:
        env["ck"].write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(T, "THR", {"У1": 0.9, "У2": 0.50, "У3": 0.26, "У6": 0.10})   # мутация: порог подогнан
    assert any("пороги ≠ константам спецификации" in b for b in chk(env))
    monkeypatch.undo()
    monkeypatch.setattr(T, "N_HOLM", 4)                                               # мутация № 103
    assert any("Холм: N ≠ 814" in b for b in chk(env)) and chk(env)
    monkeypatch.undo()
    monkeypatch.setattr(T, "identity_gap", lambda y: 1.0)                             # Σ вкладов ≠ Δ
    assert any("Σ вкладов ≠ Δ" in b for b in chk(env))
    monkeypatch.undo()
    assert chk(env) == []


def test_f_in_gap_zero_days_is_caught_and_mismatching_inputs_refuse(env) -> None:
    p = env["out"] / "w8g.parquet"
    keep = p.read_bytes()
    f = pd.read_parquet(p)
    z = int(np.flatnonzero((f["gap"] == 0).to_numpy())[-5])
    f.loc[z, "F"] = f.loc[z, "X1"] * 1.5                                              # поправка в день gap = 0
    f.to_parquet(p, index=False)
    try:
        bad = chk(env)
        assert any("в днях gap = 0 F W8-G ≠ X1" in b for b in bad)
        g = pd.read_parquet(p)
        g.loc[3, "rv"] *= 1.01                                                        # rv не совпадает с окном M → отказ
        g.to_parquet(p, index=False)
        with pytest.raises(SystemExit, match="расходятся"):
            chk(env)
    finally:
        p.write_bytes(keep)
    assert chk(env) == []


def test_load_inputs_requires_registered_ids_first(env, tmp_path, monkeypatch) -> None:
    t = tmp_path / "ALL_TRIALS.csv"
    t.write_text("id,name,formula,params,status,registered_at\nY1,a,b,c,d,10.11.2026 04:21:03\n", encoding="utf-8")
    real = E.REG.require
    monkeypatch.setattr(E.REG, "require", lambda cfg, trials=t: real(cfg, trials))     # реестр без W8 → отказ до чтения рядов (реестр до прогноза)
    with pytest.raises(SystemExit, match="нет в реестре"):
        E.load_inputs(env["out"])
