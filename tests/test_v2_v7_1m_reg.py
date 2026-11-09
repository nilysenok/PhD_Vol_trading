"""V7 лист 1a-M, G19/G20: реестр до счёта (11 строк плана, редакция 2), отказы бегуна, дополняемость реестра, `--check` ничего не пишет, модули 1a не читают данные SBER. Временные файлы — в tmp_path."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models_v2.s14 import run_v7 as RUN, v7_0_io as IO, v7_1m_reg as RG, v7_1m_stage as ST

PKG = Path(RG.__file__).parent


@pytest.fixture()
def reg(tmp_path, monkeypatch):
    t, c = tmp_path / "ALL_TRIALS.csv", tmp_path / "CHECKPOINT.json"
    monkeypatch.setattr(RG, "TRIALS", t)
    monkeypatch.setattr(RG, "CHECKPOINT", c)
    return t, c


def test_eleven_rows_status_time_conditions_and_encoder_rule(reg):
    t, _ = reg
    assert RG.ensure(when="2026-11-09T17:00:00+03:00") == "создан"
    rows = RG.read(t)
    assert rows[0] == RG.HEADER and len(rows) == 12 and [r[0] for r in rows[1:]] == ["Ж-233:%d" % i for i in range(1, 12)]
    assert all(r[7] == "зарегистрирована" and r[6] == "2026-11-09T17:00:00+03:00" for r in rows[1:])
    cfg = {r[0]: json.loads(r[4]) for r in rows[1:]}
    assert "условная" in cfg["Ж-233:9"]["condition"] and "условная" in cfg["Ж-233:11"]["condition"]
    assert all(cfg["Ж-233:%d" % i]["condition"] == "безусловная" for i in (1, 2, 3, 4, 5, 6, 7, 8, 10))
    assert all("лучший из 1 и 2 по QLIKE 2018–2021" in cfg["Ж-233:%d" % i]["encoder_rule"] for i in range(3, 12))
    assert cfg["Ж-233:6"]["router_in"] == "(Z,z)" and cfg["Ж-233:5"]["router_in"] == "Z" and cfg["Ж-233:7"]["router_in"] == "z" and "Ж-233:12" not in cfg
    assert RG.bad(t) == []


def test_registry_not_rewritten_by_rerun_and_only_appended(reg):
    t, _ = reg
    RG.ensure(when="T1")
    raw = t.read_bytes()
    assert RG.ensure(when="T2") == "без изменений" and t.read_bytes() == raw                  # повторный запуск не переписывает
    lines = raw.decode().splitlines(keepends=True)
    t.write_bytes("".join(lines[:5] + lines[6:]).encode())                                    # нет строки 5 (Ж-233:5)
    head = t.read_bytes()
    assert RG.ensure(when="T3") == "дополнен"
    new = t.read_bytes()
    assert new.startswith(head) and new[len(head):].decode().startswith("Ж-233:5,") and new[len(head):].decode().count("\n") == 1 and "T3" in new[len(head):].decode()


def test_bad_flags_changed_duplicate_foreign_and_reserve_rows(reg):
    t, _ = reg
    RG.ensure(when="T")
    good = t.read_text(encoding="utf-8")
    t.write_text(good.replace("Residual MoE K=2, маршрутизатор B", "Residual MoE K=2, маршрутизатор X"), encoding="utf-8")
    assert any("Ж-233:6" in b for b in RG.bad(t))
    t.write_text(good + good.splitlines()[2] + "\n", encoding="utf-8")
    assert any("дубль" in b for b in RG.bad(t))
    t.write_text(good + "Ж-233:12,Ж-233,x,reserve,{},trial,T,зарегистрирована\n", encoding="utf-8")
    assert any("вне плана" in b for b in RG.bad(t))
    t.write_text(good.replace("зарегистрирована", "", 1), encoding="utf-8")
    assert RG.bad(t) == []                                                                    # статус не входит в текст конфигурации; его проверяет gate
    with pytest.raises(RG.Refusal, match="статус"):
        RG.gate("1", trials=t)


def test_gate_refuses_reserve_foreign_conditional_and_wrong_status(reg):
    t, c = reg
    with pytest.raises(RG.Refusal, match="реестр"):
        RG.gate("6")                                                                          # реестра нет
    RG.ensure(when="T")
    assert RG.gate("6")["key"] == "Ж-233:6"
    for cfg, text in (("12", "резерв"), ("13", "не из реестра"), ("0", "не из реестра"), ("A0", "не из реестра")):
        with pytest.raises(RG.Refusal, match=text):
            RG.gate(cfg)
    for cfg, key in (("9", "cond_9"), ("11", "cond_11")):
        with pytest.raises(RG.Refusal, match="условная"):
            RG.gate(cfg)
        c.write_text(json.dumps({key: "true"}), encoding="utf-8")
        with pytest.raises(RG.Refusal, match="условная"):                                       # не булево true
            RG.gate(cfg)
        c.write_text(json.dumps({key: False}), encoding="utf-8")
        with pytest.raises(RG.Refusal, match="условная"):
            RG.gate(cfg)
        c.write_text(json.dumps({key: True}), encoding="utf-8")
        assert RG.gate(cfg)["key"] == "Ж-233:" + cfg
    c.write_text(json.dumps({"cond_9": True}), encoding="utf-8")
    with pytest.raises(RG.Refusal, match="условная"):
        RG.gate("11")                                                                         # условие 9 не открывает 11
    t.write_text(t.read_text(encoding="utf-8").replace("зарегистрирована", "посчитана", 1), encoding="utf-8")
    with pytest.raises(RG.Refusal, match="статус"):
        RG.gate("1")


def test_mutation_gate_ignoring_condition_or_reserve_is_caught(reg, monkeypatch):
    RG.ensure(when="T")
    monkeypatch.setattr(RG, "CONDITION_KEY", {})                                              # мутант: условие снято
    assert RG.gate("9")["key"] == "Ж-233:9"                                                    # без условия отказа нет -> обычный тест отказа упал бы
    monkeypatch.setattr(RG, "RESERVE", "99")                                                  # мутант: резерв не тот
    with pytest.raises(RG.Refusal, match="не из реестра"):
        RG.gate("12")                                                                         # отказ идёт уже по другой причине, не «резерв»: сообщение тестом различается


def test_runner_refuses_by_stage_and_config(reg, capsys, monkeypatch):
    RG.ensure(when="T")
    assert RUN.main(["--stage", "2", "--config", "12", "--check"]) == 3 and "резерв" in capsys.readouterr().out
    assert RUN.main(["--stage", "3", "--check"]) == 3                                         # 8 допущена, 9 — отказ (условие)
    out = capsys.readouterr().out
    assert "допущены реестром ['8']" in out and "отказано ['9']" in out
    assert RUN.main(["--stage", "1b", "--check"]) == 0 and RUN.main(["--stage", "1b"]) == 2 and "не подключён" in capsys.readouterr().out
    called = []
    monkeypatch.setitem(ST.RUNNERS, "5", lambda: called.append(5))
    assert RUN.main(["--stage", "2", "--config", "5"]) == 0 and called == [5]


def test_runner_refuses_when_registry_is_missing_or_broken(reg, capsys):
    t, _ = reg
    assert RUN.main(["--stage", "2", "--check"]) == 3
    RG.ensure(when="T")
    t.write_text(t.read_text(encoding="utf-8").replace("Ж-233:6,", "Ж-233:66,"), encoding="utf-8")
    assert RUN.main(["--stage", "2", "--config", "6", "--check"]) == 3


def test_stage_1a_check_writes_nothing_and_reads_no_sber_data(reg, monkeypatch, tmp_path):
    t, c = reg
    rep = tmp_path / "rep"
    rep.mkdir()
    monkeypatch.setattr(IO, "REP", rep)
    monkeypatch.setattr(IO, "OUT", tmp_path / "out")
    monkeypatch.setattr(IO, "STATE", tmp_path / "state.md")
    monkeypatch.setattr(RG, "TRIALS", rep / "ALL_TRIALS.csv")
    monkeypatch.setattr(RG, "CHECKPOINT", rep / "CHECKPOINT.json")
    for name in ("load", "read_parquet"):
        monkeypatch.setattr(np if name == "load" else pd, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError("чтение данных в 1a")))
    assert RUN.main(["--stage", "1a"]) == 0                                                   # запись: реестр и ключ 1a (+ ключа 0 в tmp нет)
    snap = {p.name: p.read_bytes() for p in rep.iterdir()}
    assert set(snap) == {"ALL_TRIALS.csv", "CHECKPOINT.json"} and "1a" in json.loads(snap["CHECKPOINT.json"])
    cp = json.loads(snap["CHECKPOINT.json"])
    cp["0"] = {"x": 1}
    (rep / "CHECKPOINT.json").write_text(json.dumps(cp, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    snap = {p.name: p.read_bytes() for p in rep.iterdir()}
    assert RUN.main(["--stage", "1a", "--check"]) == 0
    assert {p.name: p.read_bytes() for p in rep.iterdir()} == snap                            # --check ничего не пишет
    assert RUN.main(["--stage", "1a"]) == 0 and json.loads((rep / "CHECKPOINT.json").read_text())["0"] == {"x": 1}      # ключ 0 не трогается; реестр не переписан
    assert (rep / "ALL_TRIALS.csv").read_bytes() == snap["ALL_TRIALS.csv"]


def test_mutation_check_that_writes_is_caught_by_digest(reg, monkeypatch, tmp_path):
    rep = tmp_path / "rep"
    rep.mkdir()
    monkeypatch.setattr(IO, "REP", rep)
    monkeypatch.setattr(IO, "OUT", tmp_path / "out")
    monkeypatch.setattr(IO, "STATE", tmp_path / "state.md")
    monkeypatch.setattr(RG, "TRIALS", rep / "ALL_TRIALS.csv")
    monkeypatch.setattr(RG, "CHECKPOINT", rep / "CHECKPOINT.json")
    assert RUN.main(["--stage", "1a"]) == 0
    monkeypatch.setattr(ST, "check_1a", lambda: (rep / "leak.txt").write_text("x") and [])    # мутант: «проверка» пишет файл
    assert RUN.main(["--stage", "1a", "--check"]) == 1


def test_check_1a_flags_stale_checkpoint_and_missing_registry(reg):
    t, c = reg
    assert any("ключа 1a" in b for b in ST.check_1a())
    ST.write_1a()
    cp = json.loads(c.read_text(encoding="utf-8"))
    assert ST.check_1a() == ["CHECKPOINT.json: нет ключа 0"]
    cp["0"] = {"x": 1}
    c.write_text(json.dumps(cp), encoding="utf-8")
    assert ST.check_1a() == []
    cp["1a"]["N_holm"] = 722
    c.write_text(json.dumps(cp), encoding="utf-8")
    assert any("ключ 1a ≠ пересчёту" in b for b in ST.check_1a())


def test_modules_1a_do_not_touch_sber_data_at_import_level():
    """Ни один модуль v7_1m_* не вызывает чтение данных на уровне модуля; имена файлов X_w*/store_raw — только внутри функций (стадии 1b+)."""
    for f in sorted(PKG.glob("v7_1m_*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)) or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
                continue
            src = ast.get_source_segment(f.read_text(encoding="utf-8"), node) or ""
            assert not any(k in src for k in ("read_parquet", "np.load", ".npz", "store_raw", "open(")), (f.name, src[:80])
    assert not any(".npz" in p.read_text(encoding="utf-8") or "store_raw" in p.read_text(encoding="utf-8") for p in PKG.glob("v7_1m_*.py"))      # файлы последовательностей листа 0 модули 1a не называют вовсе


def test_real_registry_and_checkpoint_1a_are_consistent_if_written():
    if not RG.TRIALS.exists():
        pytest.skip("лист 1a ещё не записан")
    assert RG.bad() == [] and len(RG.read()) == 12
    assert ST.check_1a() == []
