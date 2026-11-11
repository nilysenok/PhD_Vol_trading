"""W9-reg, регистратор `x1_w9_reg`: 13 строк (6 по объявлению + 7 «ПОСЛЕ ПРОСМОТРА») одной меткой; прочие строки и ключи CHECKPOINT побайтно; отказ при повторе;
проверка по id, а не по позиции; N = 827 / 3359 = 814 + 13 / 3333 + 26. Работа только на копиях во временном каталоге; реальные файлы читаются."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from models_v2.s16 import x1_w9_reg as W

DECL6 = ["W9A", "W9D1", "W9D2", "W9D3", "W9B1", "W9B2"]
POST7 = ["W9-PH-E", "W9-PH-E0", "W9-PH-R", "W9-PH-R0", "W9-PH-R126", "W9-PH-R504", "W9-PH-R+W5G"]
HEAD = "id,name,formula,params,status,registered_at\n"
OTHER = ['Y1,"М1: затухание","F = X1·exp(g)",—,зарегистрирована,10.11.2026 04:21:03\n',
         'W8D1,чужая,формула,—,"диагностика W8, критериев не получает",11.11.2026 15:27:28\n']
NOW = "11.11.2026 16:30:00"


@pytest.fixture()
def env(tmp_path: Path):
    t = tmp_path / "ALL_TRIALS.csv"
    t.write_bytes(("# шапка реестра\n" + HEAD + "".join(OTHER)).encode("utf-8"))
    c = tmp_path / "CHECKPOINT.json"
    c.write_text(json.dumps({"W8.registered": {"n": 814, "n_wide": 3333}, "A": {"x": [1, 2]}}, indent=1, sort_keys=True), encoding="utf-8")
    return t, c


def test_constants_and_composition() -> None:
    assert list(W.IDS) == DECL6 + POST7 and len(W.IDS) == 13
    assert (W.N_PROGRAM, W.N_WIDE) == (814 + 13, 3333 + 2 * 13) == (827, 3359)
    assert [r[0] for r in W.ROWS if W.POST in r[4]] == POST7
    assert all(W.MARK in r[4] and "решение дирижёра 16:22" in r[4] for r in W.ROWS)
    assert all("критериев по объявлению не прошла" in r[4] for r in W.ROWS[:6])


def test_real_checkpoint_base_is_w8_814_3333() -> None:
    cp = json.loads(W.CKPT.read_text(encoding="utf-8"))
    assert cp["W8.registered"]["n"] == 814 and cp["W8.registered"]["n_wide"] == 3333      # база N для прибавления 13 строк


def test_13_rows_one_stamp_other_rows_bytewise(env) -> None:
    t, _ = env
    before = t.read_bytes()
    assert W.register(NOW, trials=t, ckpt=None) == 13
    assert t.read_bytes().startswith(before)                                    # прочие строки и шапка побайтно
    cur = pd.read_csv(t, comment="#", dtype=str, keep_default_na=False)
    mine = cur[cur["id"].isin(W.IDS)]
    assert len(cur) == 15 and len(mine) == 13 and set(mine["registered_at"]) == {NOW}
    assert W.check(trials=t) == []


def test_checkpoint_merge_only_w9_key(env) -> None:
    t, c = env
    before = json.loads(c.read_text(encoding="utf-8"))
    W.register(NOW, trials=t, ckpt=c)
    after = json.loads(c.read_text(encoding="utf-8"))
    assert set(after) - set(before) == {"W9.registered"}
    assert all(after[k] == before[k] for k in before)
    k = after["W9.registered"]
    assert (k["n"], k["n_wide"], k["rows"], k["registered_at"]) == (827, 3359, 13, NOW)
    assert (k["rows_declared"], k["rows_post_viewing"]) == (6, 7)


def test_repeat_registration_refused_and_file_unchanged(env) -> None:
    t, c = env
    W.register(NOW, trials=t, ckpt=c)
    after, cp = t.read_bytes(), c.read_bytes()
    with pytest.raises(SystemExit):
        W.register(NOW, trials=t, ckpt=c)
    assert t.read_bytes() == after and c.read_bytes() == cp


def test_check_by_id_not_position(env) -> None:
    t, _ = env
    W.register(NOW, trials=t, ckpt=None)
    lines = t.read_text(encoding="utf-8").splitlines(keepends=True)
    t.write_text("".join(lines[:2]) + "".join(reversed(lines[2:])), encoding="utf-8")       # порядок строк другой, шапка на месте
    assert W.check(trials=t) == []
    t.write_text(t.read_text(encoding="utf-8") + "W9X,позже,—,—,x,11.11.2026 17:00:00\n", encoding="utf-8")   # чужая строка в конце не мешает
    assert W.check(trials=t) == []


def test_check_catches_missing_second_stamp_missing_mark_and_wrong_post(env) -> None:
    t, _ = env
    W.register(NOW, trials=t, ckpt=None)
    good = t.read_text(encoding="utf-8")
    lines = good.splitlines(keepends=True)
    t.write_text("".join(ln for ln in lines if not ln.startswith("W9A,")), encoding="utf-8")
    assert any("строк W9" in b for b in W.check(trials=t))                      # строка выпала
    t.write_text(good.replace(NOW, "11.11.2026 16:31:00", 1), encoding="utf-8")
    assert any("registered_at" in b for b in W.check(trials=t))                 # две метки
    t.write_text(good.replace(W.MARK, "иное", 1), encoding="utf-8")
    assert any("нет пометки" in b for b in W.check(trials=t))
    t.write_text(good.replace(W.POST, "после", 1), encoding="utf-8")            # у одной из 7 снят «ПОСЛЕ ПРОСМОТРА»
    assert any(W.POST in b for b in W.check(trials=t))
    t.write_text(good + "".join(ln for ln in lines if ln.startswith("W9A,")), encoding="utf-8")   # дубль id
    assert any("строк W9" in b for b in W.check(trials=t))


def test_real_registry_not_changed_by_check_and_no_clash() -> None:
    real = W.TRIALS.read_bytes()
    ids = set(pd.read_csv(W.TRIALS, comment="#", dtype=str, keep_default_na=False)["id"])
    assert W.check() == [] or not (ids & set(W.IDS)) or len(ids & set(W.IDS)) == 13      # на реальном: все 13 или ни одной
    assert W.TRIALS.read_bytes() == real
