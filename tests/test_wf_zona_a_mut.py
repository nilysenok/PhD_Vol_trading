"""Тестировщик PR #733: дыры, не пойманные tests/test_wf_zona_a.py при подложенных ошибках (мутации а-е: 1 дыра, д).
Синтетика, данные не читаются. python3 -m pytest tests/test_wf_zona_a_mut.py -q"""
import pandas as pd
import pytest

from fastbt import e1_sektor, period_v2, zona

D = pd.Timestamp


@pytest.fixture(autouse=True)
def _dev():
    yield
    zona.set_zone("dev")


def test_granitsa_wf_rovno_pervoe_marta_2026(tmp_path, monkeypatch):
    p = tmp_path / "o.json"
    p.write_text("{}")
    monkeypatch.setattr(period_v2, "OTMETKA", p)
    assert zona.END_WF == D("2026-03-01")           # мутация END_WF=03-02 падает и здесь, не только через check()
    with zona.zone_ctx("wf"):
        zona.check(D("2026-02-28 23:59"), "t")
        with pytest.raises(PermissionError):
            zona.check(D("2026-03-01"), "t")
        e1_sektor.load_indexes([], main=tmp_path, end=D("2026-03-01"))              # граница включена как «исключительно»
        with pytest.raises(PermissionError):
            e1_sektor.load_indexes([], main=tmp_path, end=D("2026-03-02"))          # мутация (е) с другим END: на день дальше


def test_put_wf_ne_zalezaet_ot_peremennoj_test_zony(monkeypatch, tmp_path):
    p = tmp_path / "o.json"
    p.write_text("{}")
    monkeypatch.setattr(period_v2, "OTMETKA", p)
    monkeypatch.setenv("FASTBT_Z2_DAILY", "/test_locked/x.parquet")
    with zona.zone_ctx("wf"):
        assert zona.path("daily", "dflt") == "dflt"     # (г) с переменной: возвращался бы test_locked


def test_hard_v_dev_i_test1_ne_wf():
    for z in ("dev", "test1"):
        with zona.zone_ctx(z):
            assert zona.hard() == zona.HARD != zona.END_WF


def test_zona_ne_menyaetsya_i_hooki_ne_zovutsya_pri_otkaze(tmp_path, monkeypatch):
    monkeypatch.setattr(period_v2, "OTMETKA", tmp_path / "net.json")
    chislo = []
    zona.on_change(lambda: chislo.append(1))
    try:
        with pytest.raises(period_v2.ZamokVladeltsa):
            zona.set_zone("wf")
        assert zona.name() == "dev" and chislo == []     # сброс кэшей загрузчиков не должен идти при отказе
    finally:
        zona._hooks.pop()


@pytest.mark.xfail(strict=True, reason="находка: period_v2.proverit_chtenie принимает пустой файл (exists), скрипты сервера "
                   "требуют непустой (-s). Безопаснее «непустой»: случайный touch не снимает замок. Решает автор/владелец.")
def test_pustaya_otmetka_ne_prinimaetsya(tmp_path, monkeypatch):
    p = tmp_path / "wf_zamok_snyat.json"
    p.write_text("")
    monkeypatch.setattr(period_v2, "OTMETKA", p)
    with pytest.raises(period_v2.ZamokVladeltsa):
        zona.set_zone("wf")
