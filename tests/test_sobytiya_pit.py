"""fastbt/sobytiya_pit.py: события point-in-time (Ж-070, Д3). Синтетические файлы, сеть не нужна.

Мутации подкладываются в копию исходника (замена строки + exec) и обязаны ронять проверку `_proverka`.
"""
from datetime import date
from pathlib import Path
from types import ModuleType

import pytest

from fastbt import sobytiya_pit as real

SRC = Path(real.__file__).read_text(encoding="utf-8")
HEAD = "ticker,tip,data_obyavleniya,data_sobytiya,istochnik\n"


def _csv(tmp_path, *rows):
    p = tmp_path / "events.csv"
    p.write_text(HEAD + "".join(",".join(r) + "\n" for r in rows), encoding="utf-8")
    return p


def _mutant(old, new):
    assert old in SRC, old
    m = ModuleType("mut")
    m.__file__ = real.__file__
    exec(compile(SRC.replace(old, new, 1), "mut", "exec"), m.__dict__)
    return m


def _proverka(mod, tmp_path):
    """True, если все инварианты point-in-time выполнены."""
    p = _csv(tmp_path,
             ("AAA", "x", "2020-03-10", "2020-03-12", "s"),     # объявлено d+1, событие d+3 (d = 03-09)
             ("AAA", "y", "2020-03-09", "2020-03-09", "s"),     # объявлено в день события: граница <=
             ("*", "z", "2020-03-05", "2020-03-09", "s"))       # всем тикерам
    d, d1 = date(2020, 3, 9), date(2020, 3, 10)
    try:
        tips = lambda x: sorted(r["tip"] for r in x)  # noqa: E731
        assert "x" not in tips(mod.sobytiya_na(d, "AAA", vpered=5, path=p))
        assert "x" in tips(mod.sobytiya_na(d1, "AAA", vpered=5, path=p))
        assert tips(mod.sobytiya_na(d, "AAA", path=p)) == ["y", "z"]
        assert tips(mod.sobytiya_na(d, "BBB", path=p)) == ["z"]
        for bad in (date(2022, 1, 1), date(2023, 5, 1)):
            with pytest.raises(PermissionError):
                mod.sobytiya_na(bad, path=p)
        q = _csv(tmp_path, ("AAA", "w", "2021-12-30", "2022-01-03", "s"))
        with pytest.raises(PermissionError):
            mod.sobytiya_na(date(2021, 12, 31), path=q)
    except (AssertionError, pytest.fail.Exception):
        return False
    return True


def test_real_proverka_prohodit(tmp_path):
    assert _proverka(real, tmp_path)


def test_obyavleno_pozzhe_ne_otdayotsya(tmp_path):
    p = _csv(tmp_path, ("AAA", "x", "2020-03-10", "2020-03-10", "s"))
    assert real.sobytiya_na("2020-03-09", "AAA", vpered=3, path=p) == []
    assert len(real.sobytiya_na("2020-03-10", "AAA", path=p)) == 1


@pytest.mark.parametrize("old,new", [
    ('r["data_obyavleniya"] <= d\n', 'r["data_sobytiya"] <= d\n'),          # сравнение по дате события
    ('r["data_obyavleniya"] <= d\n', 'r["data_obyavleniya"] < d\n'),        # < вместо <=
    ('r["ticker"] in ("*", ticker)', 'r["ticker"] == ticker'),              # * не раздаётся
    ("if d >= END:", "if False:"),                                          # запрос >= 2022
    ('if r["data_sobytiya"] >= END or r["data_obyavleniya"] >= END:', "if False:"),   # строки >= 2022
], ids=["po_data_sobytiya", "strogo_menshe", "zvezda", "chtenie_2022", "stroki_2022"])
def test_mutatsii_lovyatsya(tmp_path, old, new):
    assert not _proverka(_mutant(old, new), tmp_path)


def test_realnyj_fajl_bez_2022_i_digest():
    assert real.digest() == real.digest() and len(real.digest()) == 40
    assert len(real.sobytiya_na("2021-01-01", vpered=400)) > 0    # файл читается без PermissionError
