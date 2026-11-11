"""W6-reg, регистратор `x1_w6_reg`: 4 строки одной меткой с пометкой «ПОСЛЕ расчёта»; прочие строки побайтно; отказ при повторе; проверка по id, а не по позиции.
Работа только на копии во временном каталоге."""
from __future__ import annotations

import json

import pytest

from models_v2.s16 import x1_w6_reg as W

IDS = ["W6D-GJS", "W6D-ZJS", "W6D-U1", "W6D-U2"]
HEAD = "id,name,formula,params,status,registered_at\n"
OTHER = ['Y1,"М1: затухание","F = X1·exp(g)",—,зарегистрирована,10.11.2026 04:21:03\n',
         'W5D1,чужая,формула,—,"диагностика W5, критериев не получает",11.11.2026 13:52:24\n']
NOW = "11.11.2026 15:00:00"


@pytest.fixture()
def env(tmp_path):
    t = tmp_path / "ALL_TRIALS.csv"
    t.write_bytes((HEAD + "".join(OTHER)).encode("utf-8"))
    return t, tmp_path / "CHECKPOINT.json"


def rows(t):
    return {ln.split(",", 1)[0]: ln for ln in t.read_text(encoding="utf-8").splitlines()[1:]}


def test_four_rows_one_stamp_and_other_rows_bytewise(env) -> None:
    t, _ = env
    before = t.read_bytes()
    assert W.register(NOW, trials=t, ckpt=None) == 4
    assert t.read_bytes().startswith(before)
    r = rows(t)
    mine = {i: r[i] for i in IDS}
    assert len(r) == 6 and all(NOW in ln and "ПОСЛЕ расчёта" in ln for ln in mine.values())
    assert W.check(trials=t) == []


def test_repeat_registration_refused_and_file_unchanged(env) -> None:
    t, _ = env
    W.register(NOW, trials=t, ckpt=None)
    after = t.read_bytes()
    with pytest.raises(SystemExit):
        W.register(NOW, trials=t, ckpt=None)
    assert t.read_bytes() == after


def test_check_by_id_not_position(env) -> None:
    t, _ = env
    W.register(NOW, trials=t, ckpt=None)
    lines = t.read_text(encoding="utf-8").splitlines(keepends=True)
    t.write_text(lines[0] + "".join(reversed(lines[1:])), encoding="utf-8")           # порядок строк другой
    assert W.check(trials=t) == []
    t.write_text(t.read_text(encoding="utf-8") + 'W9X,пост,—,—,x,11.11.2026 16:00:00\n', encoding="utf-8")   # чужая строка в конце не мешает
    assert W.check(trials=t) == []


def test_check_catches_missing_row_second_stamp_and_missing_mark(env) -> None:
    t, _ = env
    assert W.check(trials=t)                                                            # пусто — замечание
    W.register(NOW, trials=t, ckpt=None)
    good = t.read_text(encoding="utf-8")
    t.write_text(good.replace("W6D-U1", "W6D-U9"), encoding="utf-8")
    assert any("4" in m for m in W.check(trials=t))
    t.write_text(good.replace(NOW, "11.11.2026 15:00:01", 1), encoding="utf-8")
    assert any("registered_at" in m for m in W.check(trials=t))
    t.write_text(good.replace("ПОСЛЕ расчёта", "до расчёта", 1), encoding="utf-8")
    assert any("пометки" in m for m in W.check(trials=t))


def test_checkpoint_merge_only_w6_key(env, tmp_path) -> None:
    t, c = env
    doc = tmp_path / "d.md"
    doc.write_text("x", encoding="utf-8")
    c.write_text(json.dumps({"A": {"n": 1}, "W5.registered": {"n": 798}}, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    W.register(NOW, trials=t, ckpt=c, doc=doc)
    cp = json.loads(c.read_text(encoding="utf-8"))
    assert cp["A"] == {"n": 1} and cp["W5.registered"] == {"n": 798}
    w = cp["W6.registered"]
    assert (w["n"], w["n_wide"], w["rows"], w["registered_at"]) == (802, 3309, 4, NOW)
