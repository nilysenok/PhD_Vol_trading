"""W4-A, регистратор `x1_w4_reg` (W4_SPEC раздел 7): 19 строк одной меткой; отказ при parquet в W4 (реестр позже прогноза) и при повторе; `require` без id — отказ;
проверка по id, а не по позиции (класс № 164); прочие строки реестра побайтно прежние. Работа только на копии во временном каталоге; реальный реестр и W4 не трогаются."""
from __future__ import annotations

import json
import re

import pytest

from models_v2.s16 import x1_w4_reg as W

IDS = ["W3D%d" % i for i in range(1, 12)] + ["W4-S1", "W4-S2", "W4-Ka", "W4-Kb1", "W4-Kb2", "W4-X", "W4-Xa", "W4-Xb"]    # раздел 7 спецификации, независимая копия
HEAD = "id,name,formula,params,status,registered_at\n"
OTHER = ['Y1,"М1: затухание, по состоянию","F = X1·exp(g), a, b","a_n, a_e",зарегистрирована,10.11.2026 04:21:03\n',
         'Z1,чужая строка,формула с ""кавычками"",—,условная,10.11.2026 16:00:00\n']
NOW = "11.11.2026 13:00:00"


@pytest.fixture()
def env(tmp_path):
    t, w4 = tmp_path / "ALL_TRIALS.csv", tmp_path / "W4"
    t.write_bytes((HEAD + "".join(OTHER)).encode("utf-8"))
    return t, w4, tmp_path / "CHECKPOINT.json"


def reg(env, now=NOW, ckpt=False):
    t, w4, c = env
    return W.register(now, trials=t, w4=w4, ckpt=c if ckpt else None)


def rows(t):
    return {ln.split(",", 1)[0]: ln for ln in t.read_text(encoding="utf-8").splitlines()[1:]}


def test_register_19_rows_one_stamp_and_other_rows_bytewise(env) -> None:
    t, _, _ = env
    before = t.read_bytes()
    assert reg(env) == 19
    after = t.read_bytes()
    assert after.startswith(before)                                                       # прежние строки побайтно на месте, добавлено только в конец
    r = rows(t)
    assert [i for i in r if i not in ("Y1", "Z1")] == IDS and list(W.IDS) == IDS
    assert {ln.rsplit(",", 1)[1] for i, ln in r.items() if i in IDS} == {NOW}              # одна registered_at у всех 19
    text = t.read_text(encoding="utf-8")
    assert len(re.findall(r"^(W3D[0-9]+|W4-)", text, flags=re.M)) == 19                     # A.G3
    assert W.check(trials=t, w4=env[1]) == []


def test_register_without_trailing_newline_keeps_old_rows(env) -> None:
    t, _, _ = env
    t.write_bytes((HEAD + OTHER[0].rstrip("\n")).encode("utf-8"))
    before = t.read_bytes()
    reg(env)
    assert t.read_bytes().startswith(before) and len(rows(t)) == 1 + 19


def test_statuses_and_conditional_rows(env) -> None:
    reg(env)
    r = rows(env[0])
    assert all("после просмотра" in r["W3D%d" % i] for i in (8, 9, 10, 11)) and "после просмотра" not in r["W3D1"]
    assert r["W4-S1"].endswith("зарегистрирована," + NOW) and r["W4-S2"].endswith("зарегистрирована," + NOW)
    assert all("условная" in r[i] for i in ("W4-X", "W4-Xa", "W4-Xb"))
    assert "контроль" in r["W4-Ka"] and "контроль" in r["W4-Kb1"] and "контроль" in r["W4-Kb2"]


def test_default_stamp_is_single_even_if_clock_moves(env, monkeypatch) -> None:
    ticks = iter("11.11.2026 13:00:%02d" % i for i in range(100))
    monkeypatch.setattr(W, "now_text", lambda: next(ticks))                               # часы идут вперёд при каждом вызове
    reg(env, now=None)
    assert len({ln.rsplit(",", 1)[1] for i, ln in rows(env[0]).items() if i in IDS}) == 1


def test_refuses_when_forecast_parquet_exists(env) -> None:
    t, w4, c = env
    w4.mkdir()
    (w4 / "mplus_S1.parquet").write_bytes(b"x")
    before = t.read_bytes()
    with pytest.raises(SystemExit, match="позже прогноза"):
        reg(env, ckpt=True)
    assert t.read_bytes() == before and not c.exists()                                    # реестр и CHECKPOINT не тронуты


def test_parquet_in_subdirectory_or_other_suffix_does_not_block(env) -> None:
    _, w4, _ = env
    (w4 / "sub").mkdir(parents=True)
    (w4 / "sub" / "x.parquet").write_bytes(b"x")                                          # спецификация: data/regen/harj_x1/W4/*.parquet — только корень
    (w4 / "notes.txt").write_text("x")
    assert reg(env) == 19


def test_second_registration_refused_and_nothing_appended(env) -> None:
    t, _, _ = env
    reg(env)
    once = t.read_bytes()
    with pytest.raises(SystemExit, match="уже зарегистрированы"):
        reg(env, now="11.11.2026 14:00:00")
    assert t.read_bytes() == once


def test_partially_registered_is_refused_without_partial_write(env) -> None:
    t, _, _ = env
    t.write_bytes(t.read_bytes() + b"W4-S1,x,x,x,x,10.11.2026 01:00:00\n")
    before = t.read_bytes()
    with pytest.raises(SystemExit, match="W4-S1"):
        reg(env)
    assert t.read_bytes() == before


def test_require_refuses_without_id_and_accepts_every_registered(env) -> None:
    t, _, _ = env
    for i in ("W4-S1", "W4-X", "W3D1"):
        with pytest.raises(SystemExit, match="нет в реестре"):
            W.require(i, trials=t)                                                        # до регистрации: чужие Y1/Z1 есть, W4 нет
    assert W.require("Y1", trials=t) is None                                              # require смотрит только id
    reg(env)
    for i in IDS:
        assert W.require(i, trials=t) is None
    for bad in ("W4-S3", "W4-s1", "W4-X ", "", "W3D12", "W4"):
        with pytest.raises(SystemExit, match="нет в реестре"):
            W.require(bad, trials=t)
    with pytest.raises(SystemExit):
        W.require("W4-S1", trials=t.parent / "missing.csv")                               # реестра нет совсем


def test_require_and_check_do_not_depend_on_position(env) -> None:
    t, w4, _ = env
    reg(env)
    lines = t.read_text(encoding="utf-8").splitlines(keepends=True)
    head, body = lines[0], lines[1:]
    mixed = head + "".join(body[2:6] + OTHER + body[6:] + body[:2]) + 'Q9,позже,f,p,s,12.11.2026 01:00:00\n'
    t.write_text(mixed, encoding="utf-8")                                                 # строки W3/W4 вперемешку с чужими, чужая строка последняя
    assert W.check(trials=t, w4=w4) == []
    for i in IDS:
        W.require(i, trials=t)
    t.write_text(head + 'Q9,чужая последняя,f,p,s,1\n' + "".join(OTHER) + "".join(body[4:]), encoding="utf-8")   # двух первых строк (W3D1, W3D2) нет
    bad = W.check(trials=t, w4=w4)
    assert bad and "17 из 19" in bad[0]
    with pytest.raises(SystemExit):
        W.require("W3D1", trials=t)


def test_check_catches_two_stamps_duplicates_and_parquet(env) -> None:
    t, w4, _ = env
    reg(env)
    good = t.read_text(encoding="utf-8")
    t.write_text(good.replace("W4-Xb,", "W4-Xb,", 1).rsplit(NOW, 1)[0] + "11.11.2026 13:00:01\n", encoding="utf-8")       # у последней строки другая метка
    assert any("registered_at не одна" in b for b in W.check(trials=t, w4=w4))
    t.write_text(good + good.splitlines()[-1] + "\n", encoding="utf-8")                   # дубль id
    assert any("строк W3/W4" in b for b in W.check(trials=t, w4=w4))
    t.write_text(good, encoding="utf-8")
    w4.mkdir()
    (w4 / "ka.parquet").write_bytes(b"x")
    assert any("parquet" in b for b in W.check(trials=t, w4=w4))


def test_checkpoint_merge_keeps_other_keys_and_refusal_leaves_it(env) -> None:
    t, w4, c = env
    other = {"ZE.stage": {"ok": True, "n": 751}, "Y.keys": [1, 2, "с кириллицей"]}
    c.write_text(json.dumps(other, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    reg(env, ckpt=True)
    cp = json.loads(c.read_text(encoding="utf-8"))
    assert {k: cp[k] for k in other} == other                                             # прочие ключи прежние
    assert cp["W4.registered"]["registered_at"] == NOW and cp["W4.registered"]["n"] == 770 and cp["W4.registered"]["n_wide"] == 3245 and cp["W4.registered"]["rows"] == 19
    assert "W3" in cp and cp["W3"]["forecasts_written"] == 0
    snap = c.read_bytes()
    with pytest.raises(SystemExit):
        reg(env, ckpt=True)
    assert c.read_bytes() == snap
