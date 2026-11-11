"""W5-A, регистратор `x1_w5_reg` (W5_SPEC раздел 7): 28 строк одной меткой; отказ при parquet в W5 (реестр позже прогноза, № 30) и при повторе; `require` без id — отказ;
проверка по id, а не по позиции (№ 164); прочие строки реестра побайтно прежние; регистратор W4 не ломается. Работа только на копии во временном каталоге."""
from __future__ import annotations

import json
import re

import pytest

from models_v2.s16 import x1_w4_reg as W4
from models_v2.s16 import x1_w5_reg as W

LEVEL = ["absCM", "absK23", "dow", "e", "err", "ev_any", "ev_cbr", "ev_cpius", "ev_exp", "ev_fomc", "ev_rebal", "gap", "lvl", "pi", "sgnCM", "tCM", "tK1M", "v"]
IDS = ["W5D%d" % i for i in range(1, 25)] + ["W5-G", "W5-Ga", "W5-Gb", "W5-GK2"]            # раздел 7 спецификации, независимая копия
HEAD = "id,name,formula,params,status,registered_at\n"
OTHER = ['Y1,"М1: затухание, по состоянию","F = X1·exp(g), a, b","a_n, a_e",зарегистрирована,10.11.2026 04:21:03\n',
         'W4-S1,чужая W4,формула с ""кавычками"",—,условная,11.11.2026 12:39:42\n']
NOW = "11.11.2026 15:00:00"


@pytest.fixture()
def env(tmp_path):
    t, w5 = tmp_path / "ALL_TRIALS.csv", tmp_path / "W5"
    t.write_bytes((HEAD + "".join(OTHER)).encode("utf-8"))
    return t, w5, tmp_path / "CHECKPOINT.json"


def reg(env, now=NOW, ckpt=False):
    t, w5, c = env
    return W.register(now, trials=t, w5=w5, ckpt=c if ckpt else None)


def rows(t):
    return {ln.split(",", 1)[0]: ln for ln in t.read_text(encoding="utf-8").splitlines()[1:]}


def mine(r):
    return {i: ln for i, ln in r.items() if i in IDS}


def test_register_28_rows_one_stamp_and_other_rows_bytewise(env) -> None:
    t, _, _ = env
    before = t.read_bytes()
    assert reg(env) == 28
    assert t.read_bytes().startswith(before)                                              # прежние строки побайтно, добавлено только в конец
    r = rows(t)
    assert [i for i in r if i in IDS] == IDS and list(W.IDS) == IDS and len(W.ROWS) == 28
    assert {ln.rsplit(",", 1)[1] for ln in mine(r).values()} == {NOW}                      # одна registered_at у всех 28
    assert len(re.findall(r"^(W5D[0-9]+|W5-)", t.read_text(encoding="utf-8"), flags=re.M)) == 28    # A.G3
    assert W.check(trials=t, w5=env[1]) == []


def test_n_program_and_wide_arithmetic() -> None:
    assert (W.N_PROGRAM, W.N_WIDE) == (770 + 24 + 4, 3245 + 2 * 28) == (798, 3301)


def test_row_content_levels_trust_and_statuses(env) -> None:
    reg(env)
    r = rows(env[0])
    for i, s in enumerate(LEVEL):
        assert ("состояние %s" % s) in r["W5D%d" % (i + 1)]
    for i, s in enumerate(["absCM", "e", "pi", "sgnCM", "tCM", "tK1M"]):
        assert ("состояние %s" % s) in r["W5D%d" % (19 + i)] and "π_G" in r["W5D%d" % (19 + i)]
    assert all("критериев не получает" in r["W5D%d" % i] for i in range(1, 25))
    assert r["W5-G"].endswith("зарегистрирована," + NOW) and "нет сетки" in r["W5-G"]
    assert "контроль" in r["W5-Ga"] and "контроль" in r["W5-Gb"]
    assert "справочно" in r["W5-GK2"] and "критериев не получает" in r["W5-GK2"]
    assert "252" in r["W5-Gb"]


def test_default_stamp_is_single_even_if_clock_moves(env, monkeypatch) -> None:
    ticks = iter("11.11.2026 15:00:%02d" % i for i in range(100))
    monkeypatch.setattr(W, "now_text", lambda: next(ticks))                               # часы идут вперёд при каждом вызове
    reg(env, now=None)
    assert len({ln.rsplit(",", 1)[1] for ln in mine(rows(env[0])).values()}) == 1


def test_register_without_trailing_newline_keeps_old_rows(env) -> None:
    t, _, _ = env
    t.write_bytes((HEAD + OTHER[0].rstrip("\n")).encode("utf-8"))
    before = t.read_bytes()
    reg(env)
    assert t.read_bytes().startswith(before) and len(rows(t)) == 1 + 28


def test_refuses_when_forecast_parquet_exists_no_partial_write(env) -> None:
    t, w5, c = env
    w5.mkdir()
    (w5 / "w5g.parquet").write_bytes(b"x")
    before = t.read_bytes()
    with pytest.raises(SystemExit, match="позже прогноза"):
        reg(env, ckpt=True)
    assert t.read_bytes() == before and not c.exists()                                    # реестр и CHECKPOINT не тронуты


def test_parquet_in_subdirectory_or_other_suffix_does_not_block(env) -> None:
    _, w5, _ = env
    (w5 / "sub").mkdir(parents=True)
    (w5 / "sub" / "x.parquet").write_bytes(b"x")
    (w5 / "notes.txt").write_text("x")
    assert reg(env) == 28


def test_second_registration_refused_and_nothing_appended(env) -> None:
    t, _, _ = env
    reg(env)
    once = t.read_bytes()
    with pytest.raises(SystemExit, match="уже зарегистрированы"):
        reg(env, now="11.11.2026 16:00:00")
    assert t.read_bytes() == once


@pytest.mark.parametrize("one", ["W5D1", "W5D24", "W5-G", "W5-GK2"])
def test_partially_registered_is_refused_without_partial_write(env, one) -> None:
    t, _, _ = env
    t.write_bytes(t.read_bytes() + ("%s,x,x,x,x,10.11.2026 01:00:00\n" % one).encode())
    before = t.read_bytes()
    with pytest.raises(SystemExit, match=one):
        reg(env)
    assert t.read_bytes() == before


def test_require_refuses_without_id_and_accepts_every_registered(env) -> None:
    t, _, _ = env
    for i in ("W5-G", "W5D1", "W5-GK2"):
        with pytest.raises(SystemExit, match="нет в реестре"):
            W.require(i, trials=t)                                                        # до регистрации: чужие Y1/W4-S1 есть, W5 нет
    assert W.require("Y1", trials=t) is None
    reg(env)
    for i in IDS:
        assert W.require(i, trials=t) is None
    for bad in ("W5-g", "W5-G ", "", "W5D25", "W5D0", "W5", "W5-Gc"):
        with pytest.raises(SystemExit, match="нет в реестре"):
            W.require(bad, trials=t)
    with pytest.raises(SystemExit):
        W.require("W5-G", trials=t.parent / "missing.csv")                                # реестра нет совсем


def test_require_and_check_do_not_depend_on_position(env) -> None:
    t, w5, _ = env
    reg(env)
    lines = t.read_text(encoding="utf-8").splitlines(keepends=True)
    head, body = lines[0], lines[1:]
    t.write_text(head + "".join(body[5:9] + OTHER + body[9:] + body[:5]) + 'Q9,позже,f,p,s,12.11.2026 01:00:00\n', encoding="utf-8")    # вперемешку, чужая строка последняя
    assert W.check(trials=t, w5=w5) == []
    for i in IDS:
        W.require(i, trials=t)
    t.write_text(head + 'Q9,чужая последняя,f,p,s,1\n' + "".join(OTHER) + "".join(body[7:]), encoding="utf-8")        # пяти первых строк нет
    bad = W.check(trials=t, w5=w5)
    assert bad and "28" in bad[0]
    with pytest.raises(SystemExit):
        W.require("W5D1", trials=t)


def test_check_catches_two_stamps_duplicates_missing_and_parquet(env) -> None:
    t, w5, _ = env
    reg(env)
    good = t.read_text(encoding="utf-8")
    t.write_text(good.rsplit(NOW, 1)[0] + "11.11.2026 15:00:01\n", encoding="utf-8")        # у последней строки другая метка
    assert any("registered_at не одна" in b for b in W.check(trials=t, w5=w5))
    t.write_text(good + good.splitlines()[-1] + "\n", encoding="utf-8")                   # дубль id
    assert any("строк W5" in b for b in W.check(trials=t, w5=w5))
    t.write_text(good.replace("W5-GK2,", "W5-GK3,"), encoding="utf-8")                    # id подменён: число строк то же, состав другой
    assert any("строк W5" in b for b in W.check(trials=t, w5=w5))
    t.write_text(good, encoding="utf-8")
    w5.mkdir()
    (w5 / "w5g.parquet").write_bytes(b"x")
    assert any("parquet" in b for b in W.check(trials=t, w5=w5))


def test_w4_registrar_unbroken_alongside_w5(env) -> None:
    t, w5, _ = env
    t.write_bytes(HEAD.encode() + OTHER[0].encode())                                      # без чужой строки с id W4-S1
    assert W4.register("11.11.2026 12:39:42", trials=t, w4=env[0].parent / "W4", ckpt=None) == 19
    assert reg(env) == 28
    assert W4.check(trials=t, w4=env[0].parent / "W4") == [] and W.check(trials=t, w5=w5) == []
    assert len(W4.IDS) == 19 and not set(W4.IDS) & set(W.IDS)                             # id не пересекаются


def test_checkpoint_merge_keeps_other_keys_and_refusal_leaves_it(env, tmp_path, monkeypatch) -> None:
    t, w5, c = env
    rep, code = tmp_path / "rep", tmp_path / "code"
    rep.mkdir(), code.mkdir()
    for n in ("W5_DIAG.csv", "W5_TABLE0.csv", "W5_GAPS.csv", "W5_SPEC.md"):
        (rep / n).write_text("содержимое " + n, encoding="utf-8")
    (code / "x1_w5_a.py").write_text("a", encoding="utf-8")
    (code / "x1_w4_a.py").write_text("b", encoding="utf-8")
    other = {"W4.registered": {"n": 770}, "ZE.stage": {"ok": True, "n": 751}, "Y.keys": [1, 2, "с кириллицей"]}
    c.write_text(json.dumps(other, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    real = W.merge_checkpoint
    monkeypatch.setattr(W, "merge_checkpoint", lambda ck, now: real(ck, now, rep=rep, code=code))
    reg(env, ckpt=True)
    cp = json.loads(c.read_text(encoding="utf-8"))
    assert {k: cp[k] for k in other} == other                                             # прочие ключи (в т. ч. W4.registered) прежние
    r = cp["W5.registered"]
    assert (r["registered_at"], r["n"], r["n_wide"], r["rows"]) == (NOW, 798, 3301, 28)
    assert cp["W5"]["configurations"] == 0 and cp["W5"]["forecasts_written"] == 0
    assert sorted(cp["W5"]["sha256"]) == ["W5_DIAG.csv", "W5_GAPS.csv", "W5_SPEC.md", "W5_TABLE0.csv"]
    assert list(cp["W5"]["code"]) == ["x1_w5_a.py"]                                       # только x1_w5_*.py
    snap = c.read_bytes()
    with pytest.raises(SystemExit):
        reg(env, ckpt=True)
    assert c.read_bytes() == snap
