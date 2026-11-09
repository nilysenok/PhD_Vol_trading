"""V7 лист 3, G42: запись и `--check` конфигурации 8 на синтетике (боевые `run_prod`, `record`, `check_cfg`, `write_cfg8`; артефакты Stage A подменены загрузчиком, выходы и реестр — во временном каталоге).
Проверка читает и ничего не пишет; ловит подмену прогноза (sha256 и побитная переоценка), подмену w, ℓ не той даты, F ≠ H при G = 0, π вне симплекса. Бегун отказывает 9 (условие) и допускает только 8; запись добавляет строки реестра."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from models_v2.s14 import v7_0_io as IO, v7_1b_cfgrun as CR, v7_1m_data as DATA, v7_1m_reg as RG, v7_1m_stage as ST, v7_1m_syn as SY, v7_3_cfg as C8, v7_3_run as R3

S = SY.make(720, 1)
TAB, SEEDS = S["tab"], (0,)


def fake_load(path):
    year = int(path.name.removesuffix(".npz").split("_")[2])
    days = TAB.index[TAB.index < pd.Timestamp(year + 1, 1, 1)]
    return dict(days=days.to_numpy("datetime64[ns]").astype(np.int64), z=S["z"][:len(days)])


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v73")
    mp = pytest.MonkeyPatch()
    mp.setattr(IO, "ROOT", tmp)
    mp.setattr(IO, "OUT", tmp / "out")
    mp.setattr(R3.R2, "same_dates", lambda df: True)
    df, per = C8.run_prod(TAB, "tcn", load=fake_load, seeds=SEEDS)
    yield R3.record(df, per, "tcn", 1.0), df, tmp
    mp.undo()


@pytest.fixture
def env(made, monkeypatch):
    rec, df, tmp = made
    monkeypatch.setattr(IO, "ROOT", tmp)
    monkeypatch.setattr(IO, "OUT", tmp / "out")
    monkeypatch.setattr(R3.R2, "same_dates", lambda d: True)
    return rec, df


def chk(rec):
    return R3.check_cfg({"cfg8": rec}, "tcn", TAB, load=fake_load, seeds=SEEDS)


def swap(df, rec, mut):
    bad = mut(df.copy())
    bad.to_parquet(R3.path(), index=False)
    return dict(rec, sha256_content=CR.content_hash(bad), sha256_file=IO.sha_file(R3.path()))


def test_honest_file_passes_and_record_has_78_params_grid_and_seeds(env):
    rec, _ = env
    assert chk(rec) == [] and rec["params"] == 78 and rec["router"] == "B" and rec["grid"] == [[0.1, 0.0], [0.1, 0.1], [1.0, 0.0], [1.0, 0.1]] and rec["fitted_share"] > 0.5


def test_check_writes_nothing(env, monkeypatch):
    rec, _ = env
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("проверка пишет"))
    for obj, name in ((pd.DataFrame, "to_parquet"), (pd.DataFrame, "to_csv"), (IO, "merge_checkpoint"), (IO, "write_csv"), (np, "savez"), (np, "savez_compressed")):
        monkeypatch.setattr(obj, name, boom)
    before = IO.outputs_digest()
    assert chk(rec) == [] and IO.outputs_digest() == before


def test_mutations_substituted_forecast_w_ell_F_and_pi_are_caught(env):
    rec, df = env
    try:
        bad = df.copy()
        bad.loc[len(bad) - 1, "G"] += 1e-9
        bad.to_parquet(R3.path(), index=False)
        assert any("sha256" in m for m in chk(rec))                                                          # подмена прогноза — sha256
        cases = {"(G)": lambda d: d.assign(G=d["G"] + 1e-9),                                                  # подмена внутри блока при подогнанном sha — пересчёт переоценки
                 "(w)": lambda d: d.assign(w=d["w"] + 1e-9),                                           # подмена w
                 "ℓ": lambda d: d.assign(ell=d["ell"].shift(-1).fillna(0.0)),                                  # ℓ даты d + 1
                 "F ≠ H": lambda d: d.assign(G=d["G"].where(d.index > 2, 0.0), F=d["F"].where(d.index > 2, d["H"] * 1.01)),
                 "симплекс": lambda d: d.assign(pi1=d["pi1"] + 0.1 * (d["fitted"] > 0))}
        for word, mut in cases.items():
            assert any(word in m for m in chk(swap(df, rec, mut))), word
    finally:
        df.to_parquet(R3.path(), index=False)


def test_mutation_other_encoder_seed_changes_forecast_so_recount_catches(env):
    rec, _ = env
    shifted = lambda p: dict(fake_load(p), z=fake_load(p)["z"] + 0.5)
    assert any("побитно" in m for m in R3.check_cfg({"cfg8": rec}, "tcn", TAB, load=shifted, seeds=SEEDS))


@pytest.fixture
def trials(tmp_path, monkeypatch):
    t, c = tmp_path / "ALL_TRIALS.csv", tmp_path / "CHECKPOINT.json"
    monkeypatch.setattr(RG, "TRIALS", t)
    monkeypatch.setattr(RG, "CHECKPOINT", c)
    RG.ensure(when="T")
    return t, c


def test_gate_refuses_9_by_condition_and_admits_8_and_mutation_without_condition_admits_9(trials, monkeypatch):
    _, c = trials
    assert RG.gate("8")["key"] == "Ж-233:8"
    with pytest.raises(RG.Refusal, match="условная"):
        RG.gate("9")
    c.write_text(json.dumps({"2": {"cond9": {"cond9": False}}}), encoding="utf-8")                           # ложное условие не включает cond_9
    with pytest.raises(RG.Refusal):
        RG.gate("9")
    monkeypatch.setattr(RG, "CONDITION_KEY", {})                                                             # мутант: условие снято
    assert RG.gate("9")["key"] == "Ж-233:9"


def test_registry_rows_result_8_and_not_run_9_are_accepted_and_wrong_status_is_not(trials):
    t, _ = trials
    assert RG.add_result("8", dict(a=1), t) == "дописана" and RG.add_result("9", dict(a=1), t, suffix=RG.NOT_RUN_SUFFIX) == "дописана" and RG.add_result("9", dict(a=1), t, suffix=RG.NOT_RUN_SUFFIX) == "уже есть"
    rows = {r[0]: r for r in RG.read(t)[1:]}
    assert rows["Ж-233:9:не_считается"][7].startswith("не считается") and rows["Ж-233:8:результат"][7] == "посчитана" and RG.bad(t) == []
    t.write_text(t.read_text(encoding="utf-8").replace("не считается: условие ложно (2.cond9)", "посчитана"), encoding="utf-8")
    assert RG.bad(t)                                                                                         # «9 посчитана» — нарушение реестра


def test_refusal_ok_only_for_9_with_false_cond9_and_ready_8(trials):
    _, c = trials
    c.write_text(json.dumps({"2": {"cond9": {"cond9": False}}, "3": {"cfg8": {}}}), encoding="utf-8")
    assert R3.ready(c) and R3.refusal_ok(["9"], c) and not R3.refusal_ok(["8", "9"], c) and not R3.refusal_ok([], c)
    c.write_text(json.dumps({"2": {"cond9": {"cond9": True}}, "3": {"cfg8": {}}}), encoding="utf-8")
    assert not R3.refusal_ok(["9"], c)
    c.write_text(json.dumps({"2": {"cond9": {"cond9": False}}}), encoding="utf-8")
    assert not R3.ready(c) and not R3.refusal_ok(["9"], c)


def test_runner_stage3_check_calls_8_check_only_when_ready_and_refuses_9(trials, monkeypatch):
    _, c = trials
    called = []
    monkeypatch.setattr(R3, "check_run", lambda ok, refused: called.append((ok, refused)) or 0)
    assert ST.run_configs("3", None, True) == 3 and called == []                                             # 8 ещё не посчитана — прежнее поведение (отказ 9, rc 3)
    c.write_text(json.dumps({"2": {"cond9": {"cond9": False}}, "3": {"cfg8": {}}}), encoding="utf-8")
    assert ST.run_configs("3", None, True) == 0 and called == [(["8"], ["9"])]


def test_check_run_rejects_admitted_9_or_missing_refusal_and_passes_honest(trials, monkeypatch, capsys):
    _, c = trials
    c.write_text(json.dumps({"2": {"cond9": {"cond9": False}}, "3": {"cfg8": {}}}), encoding="utf-8")
    monkeypatch.setattr(R3, "check", lambda cp: [])
    monkeypatch.setattr(R3.CR, "read_cp", lambda: {"3": {}})
    assert R3.check_run(["8"], ["9"]) == 0
    assert R3.check_run(["8", "9"], []) == 1 and "отказ должен быть ровно у 9" in capsys.readouterr().out         # мутант: бегун допустил 9
    assert R3.check_run(["8", "9"], []) == 1


def test_write_cfg8_writes_file_checkpoint_and_two_registry_rows(trials, tmp_path, monkeypatch):
    t, c = trials
    c.write_text(json.dumps({"2": {"cond9": {"cond9": False}}}), encoding="utf-8")
    monkeypatch.setattr(IO, "REP", tmp_path)
    monkeypatch.setattr(IO, "ROOT", tmp_path)
    monkeypatch.setattr(IO, "OUT", tmp_path / "out")
    monkeypatch.setattr(R3.R2, "same_dates", lambda d: True)
    monkeypatch.setattr(R3.C3, "kind3", lambda cp: "tcn")
    monkeypatch.setattr(DATA, "load_tab", lambda: TAB)
    orig = C8.run_prod
    monkeypatch.setattr(C8, "run_prod", lambda tab, kind: orig(tab, kind, load=fake_load, seeds=SEEDS))
    assert R3.write_cfg8() == 0
    cp = json.loads(c.read_text(encoding="utf-8"))
    assert cp["2"]["cond9"]["cond9"] is False and cp["3"]["cfg8"]["sha256_file"] == IO.sha_file(R3.path()) and R3.refusal_ok(["9"], c)
    keys = [r[0] for r in RG.read(t)[1:]]
    assert "Ж-233:8:результат" in keys and "Ж-233:9:не_считается" in keys and RG.bad(t) == []
    assert R3.write_cfg8() == 0 and keys == [r[0] for r in RG.read(t)[1:]]                                   # повтор — пропуск, строки не дублируются
