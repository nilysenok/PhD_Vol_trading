"""W8-A, регистратор `x1_w8_reg` (W8_SPEC разд. 8): 12 строк одной меткой; отказ при parquet в W8 (реестр позже прогноза, № 30) и при повторе; `require` без id — отказ; проверка по id, а не по позиции
(№ 164); прочие строки реестра побайтно прежние; режим «после прогноза»; ключ CHECKPOINT сливается, прочие ключи прежние. Работа только на копиях во временном каталоге; только monkeypatch (№ 43)."""
from __future__ import annotations

import json
import re

import pytest

from models_v2.s16 import x1_w5_reg as W5
from models_v2.s16 import x1_w8_reg as W

FEATS = ["type", "len", "holiday", "lvl", "shock", "e", "imoex_absr", "imoex_rv", "season", "cold"]
IDS = ["W8D%d" % i for i in range(1, 11)] + ["W8-G", "W8-Gb"]                           # разд. 8 спецификации, независимая копия
HEAD = "id,name,formula,params,status,registered_at\n"
OTHER = ['Y1,"М1: затухание, по состоянию","F = X1·exp(g), a, b","a_n, a_e",зарегистрирована,10.11.2026 04:21:03\n',
         'W5-G,чужая W5,формула с ""кавычками"",—,условная,11.11.2026 12:39:42\n']
NOW = "11.11.2026 15:00:00"


@pytest.fixture()
def env(tmp_path):
    t, w8 = tmp_path / "ALL_TRIALS.csv", tmp_path / "W8"
    t.write_bytes((HEAD + "".join(OTHER)).encode("utf-8"))
    return t, w8, tmp_path / "CHECKPOINT.json"


def reg(env, now=NOW, ckpt=False):
    t, w8, c = env
    return W.register(now, trials=t, w8=w8, ckpt=c if ckpt else None)


def rows(t):
    return {ln.split(",", 1)[0]: ln for ln in t.read_text(encoding="utf-8").splitlines()[1:]}


def mine(r):
    return {i: ln for i, ln in r.items() if i in IDS}


def test_register_12_rows_one_stamp_and_other_rows_bytewise(env) -> None:
    t, w8, _ = env
    before = t.read_bytes()
    assert reg(env) == 12
    assert t.read_bytes().startswith(before)                                              # прежние строки побайтно, добавлено только в конец
    r = rows(t)
    assert [i for i in r if i in IDS] == IDS and list(W.IDS) == IDS and len(W.ROWS) == 12
    assert {ln.rsplit(",", 1)[1] for ln in mine(r).values()} == {NOW}                      # одна registered_at у всех 12
    assert len(re.findall(r"^(W8D[0-9]+|W8-)", t.read_text(encoding="utf-8"), flags=re.M)) == 12      # A.G3
    assert W.check(trials=t, w8=w8) == []


def test_n_program_and_wide_arithmetic() -> None:
    assert (W.N_PROGRAM, W.N_WIDE) == (798 + 4 + 10 + 2, 3301 + 8 + 2 * 12) == (814, 3333)
    assert not set(W.IDS) & set(W5.IDS)                                                   # id W5 и W8 не пересекаются


def test_row_content_features_statuses_and_diagnostic_mark(env) -> None:
    reg(env)
    r = rows(env[0])
    for i, s in enumerate(FEATS):
        assert ("признаки по порядку: " + ", ".join(FEATS)) in r["W8D%d" % (i + 1)] and ("по признаку %s," % s) in r["W8D%d" % (i + 1)]
    assert all("критериев не получает" in r["W8D%d" % i] and "ПОСЛЕ расчёта" in r["W8D%d" % i] for i in range(1, 11))
    assert r["W8-G"].endswith("зарегистрирована," + NOW) and "нет сетки" in r["W8-G"] and "β W5-G" in r["W8-G"]
    assert r["W8-Gb"].endswith("контроль," + NOW) and "252" in r["W8-Gb"]


def test_default_stamp_is_single_even_if_clock_moves_and_has_format(env, monkeypatch) -> None:
    ticks = iter("11.11.2026 15:00:%02d" % i for i in range(100))
    monkeypatch.setattr(W, "now_text", lambda: next(ticks))                               # часы идут вперёд при каждом вызове
    reg(env, now=None)
    stamps = {ln.rsplit(",", 1)[1] for ln in mine(rows(env[0])).values()}
    assert len(stamps) == 1 and re.fullmatch(r"\d\d\.\d\d\.\d{4} \d\d:\d\d:\d\d", stamps.pop())


def test_register_without_trailing_newline_keeps_old_rows(env) -> None:
    t, _, _ = env
    t.write_bytes((HEAD + OTHER[0].rstrip("\n")).encode("utf-8"))
    before = t.read_bytes()
    reg(env)
    assert t.read_bytes().startswith(before) and len(rows(t)) == 1 + 12


def test_refuses_when_forecast_parquet_exists_no_partial_write(env) -> None:
    t, w8, c = env
    w8.mkdir()
    (w8 / "w8g.parquet").write_bytes(b"x")
    before = t.read_bytes()
    with pytest.raises(SystemExit, match="позже прогноза"):
        reg(env, ckpt=True)
    assert t.read_bytes() == before and not c.exists()                                    # реестр и CHECKPOINT не тронуты


def test_parquet_in_subdirectory_or_other_suffix_does_not_block(env) -> None:
    _, w8, _ = env
    (w8 / "sub").mkdir(parents=True)
    (w8 / "sub" / "x.parquet").write_bytes(b"x")
    (w8 / "notes.txt").write_text("x")
    assert reg(env) == 12


def test_second_registration_refused_and_nothing_appended(env) -> None:
    t, _, _ = env
    reg(env)
    once = t.read_bytes()
    with pytest.raises(SystemExit, match="уже зарегистрированы"):
        reg(env, now="11.11.2026 16:00:00")
    assert t.read_bytes() == once


@pytest.mark.parametrize("one", ["W8D1", "W8D10", "W8-G", "W8-Gb"])
def test_partially_registered_is_refused_without_partial_write(env, one) -> None:
    t, _, _ = env
    t.write_bytes(t.read_bytes() + ("%s,x,x,x,x,10.11.2026 01:00:00\n" % one).encode())
    before = t.read_bytes()
    with pytest.raises(SystemExit, match=one):
        reg(env)
    assert t.read_bytes() == before


def test_require_refuses_without_id_and_accepts_every_registered(env) -> None:
    t, _, _ = env
    for i in ("W8-G", "W8D1", "W8-Gb"):
        with pytest.raises(SystemExit, match="нет в реестре"):
            W.require(i, trials=t)                                                        # до регистрации: чужие Y1/W5-G есть, а W8 нет
    assert W.require("Y1", trials=t) is None
    reg(env)
    for i in IDS:
        assert W.require(i, trials=t) is None
    for bad in ("W8-g", "W8-G ", "", "W8D11", "W8D0", "W8", "W8-Gc"):
        with pytest.raises(SystemExit, match="нет в реестре"):
            W.require(bad, trials=t)
    with pytest.raises(SystemExit):
        W.require("W8-G", trials=t.parent / "missing.csv")                                # реестра нет совсем


def test_require_and_check_do_not_depend_on_position(env) -> None:
    t, w8, _ = env
    reg(env)
    lines = t.read_text(encoding="utf-8").splitlines(keepends=True)
    head, body = lines[0], lines[1:]
    t.write_text(head + "".join(body[5:9] + OTHER + body[9:] + body[:5]) + 'Q9,позже,f,p,s,12.11.2026 01:00:00\n', encoding="utf-8")    # вперемешку, чужая строка последняя
    assert W.check(trials=t, w8=w8) == []
    for i in IDS:
        W.require(i, trials=t)
    t.write_text(head + 'Q9,чужая последняя,f,p,s,1\n' + "".join(OTHER) + "".join(body[7:]), encoding="utf-8")        # пяти первых строк нет
    bad = W.check(trials=t, w8=w8)
    assert bad and "12" in bad[0]
    with pytest.raises(SystemExit):
        W.require("W8D1", trials=t)


def test_check_catches_two_stamps_duplicates_missing_id_and_parquet(env) -> None:
    t, w8, _ = env
    reg(env)
    good = t.read_text(encoding="utf-8")
    t.write_text(good.rsplit(NOW, 1)[0] + "11.11.2026 15:00:01\n", encoding="utf-8")        # у последней строки другая метка
    assert any("registered_at не одна" in b for b in W.check(trials=t, w8=w8))
    t.write_text(good + good.splitlines()[-1] + "\n", encoding="utf-8")                   # дубль id
    assert any("строк W8" in b for b in W.check(trials=t, w8=w8))
    t.write_text(good.replace("W8-Gb,", "W8-Gc,"), encoding="utf-8")                      # id подменён: число строк то же, состав другой
    assert any("строк W8" in b for b in W.check(trials=t, w8=w8))
    t.write_text(good, encoding="utf-8")
    w8.mkdir()
    (w8 / "w8g.parquet").write_bytes(b"x")
    assert any("parquet" in b for b in W.check(trials=t, w8=w8))


def test_check_after_mode_registered_before_parquet_and_before_stage_key(env) -> None:
    t, w8, c = env
    reg(env)
    assert any("нет parquet" in b for b in W.check(trials=t, w8=w8, after=True, ckpt=c))        # после прогноза без прогноза — сравнивать не с чем
    w8.mkdir()
    (w8 / "w8g.parquet").write_bytes(b"x")                                                # mtime = сейчас (после 15:00 11.11.2026)
    assert W.check(trials=t, w8=w8, after=True, ckpt=c) == []                             # регистрация раньше parquet; ключа стадии нет
    c.write_text(json.dumps({"W8.stage1": {"written_at": "11.11.2026 14:59:59"}}), encoding="utf-8")
    assert any("не раньше времени записи ключа W8.stage1" in b for b in W.check(trials=t, w8=w8, after=True, ckpt=c))      # ключ стадии записан до регистрации
    c.write_text(json.dumps({"W8.stage1": {"written_at": "11.11.2026 15:00:01"}}), encoding="utf-8")
    assert W.check(trials=t, w8=w8, after=True, ckpt=c) == []
    t.write_text(t.read_text(encoding="utf-8").replace(NOW, "31.12.2099 00:00:00"), encoding="utf-8")        # мутация: реестр позже прогноза (№ 30)
    assert any("не раньше mtime самого раннего parquet" in b for b in W.check(trials=t, w8=w8, after=True, ckpt=c))


def test_checkpoint_merge_keeps_other_keys_and_refusal_leaves_it(env, tmp_path, monkeypatch) -> None:
    t, w8, c = env
    rep = tmp_path / "rep"
    rep.mkdir()
    for n in ("W8_SPEC.md", "W8_DECOMP.csv"):
        (rep / n).write_text("содержимое " + n, encoding="utf-8")
    other = {"W5.registered": {"n": 798}, "ZE.stage": {"ok": True, "n": 751}, "Y.keys": [1, 2, "с кириллицей"]}
    c.write_text(json.dumps(other, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    real = W.merge_checkpoint
    monkeypatch.setattr(W, "merge_checkpoint", lambda ck, now: real(ck, now, rep=rep))
    reg(env, ckpt=True)
    cp = json.loads(c.read_text(encoding="utf-8"))
    assert {k: cp[k] for k in other} == other                                             # прочие ключи прежние
    r = cp["W8.registered"]
    assert (r["registered_at"], r["n"], r["n_wide"], r["rows"]) == (NOW, 814, 3333, 12)
    assert len(r["spec_sha256"]) == 64 and len(r["decomp_sha256"]) == 64 and r["spec_sha256"] != r["decomp_sha256"]
    assert set(cp) == set(other) | {"W8.registered"}                                      # ключ стадии W8.stage1 не создан регистратором
    snap = c.read_bytes()
    with pytest.raises(SystemExit):
        reg(env, ckpt=True)
    assert c.read_bytes() == snap
