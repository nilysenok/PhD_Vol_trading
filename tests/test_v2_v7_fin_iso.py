"""V7-FIN (G50–G52): числа вне таблиц — только объявленные определения; `--check` ТОЛЬКО ЧИТАЕТ (хэш выходов до = после; запись в `--check` ловится); реестр, ключ FIN и REPRODUCTION; код FIN не вписывает результаты в тексты."""
from __future__ import annotations

import json
import re
from pathlib import Path

from models_v2.s14 import v7_0_io as IO, v7_fin_asm as A, v7_fin_check as CK, v7_fin_data as D, v7_fin_docheck as DC, v7_fin_stage as ST, v7_fin_text as TX

T = A.load()


def test_numbers_outside_tables_are_only_declared_definitions():
    assert DC.check_static(T) == [] and TX.STATIC_DECIMALS >= {"0,05", "0,0005"}


def test_mutation_static_number_in_text_is_caught(monkeypatch):
    monkeypatch.setattr(TX, "CAVEATS", TX.CAVEATS + "\n- Порог 0,77.")
    assert DC.check_static(T) != []


def test_mutation_number_typed_into_generator_function_is_caught(monkeypatch):
    """Число вписано в код генератора (раздел 11): искажение таблиц его не меняет — оно обнаруживается как «вне таблиц»."""
    from models_v2.s14 import v7_fin_doc2 as W2
    orig = W2.sec11
    monkeypatch.setattr(W2, "sec11", lambda t: orig(t) + "\nКвантиль 0,6180.")
    assert DC.check_static(T) != []


def test_perturbation_has_a_different_fixed_point_than_second_one():
    assert abs(0.1371 / (1 - 1.31) - 0.0913 / (1 - 0.77)) > 0.01


def test_generator_sources_have_no_result_decimals_in_text_literals():
    """В исходниках разделов нет десятичных чисел через запятую внутри строк, кроме определений (ловит ручную вписку до сборки)."""
    ok = TX.STATIC_DECIMALS
    for name in ("v7_fin_doc.py", "v7_fin_doc2.py", "v7_fin_asm.py"):
        src = (Path(D.__file__).parent / name).read_text(encoding="utf-8")
        assert not [x for x in re.findall(r"\d+,\d+", src) if x not in ok], name


def test_real_check_is_read_only_and_ok(capsys):
    before = IO.outputs_digest()
    assert ST.run(True) == 0 and "OK: FIN" in capsys.readouterr().out
    assert IO.outputs_digest() == before


def test_mutation_check_that_writes_a_file_is_caught(monkeypatch, tmp_path, capsys):
    r = ST.compute()
    monkeypatch.setattr(ST, "compute", lambda: r)
    monkeypatch.setattr(IO, "OUT", tmp_path)

    def writer(_r):
        (tmp_path / "stray.txt").write_text("запись из --check")
        return []
    monkeypatch.setattr(CK, "all_bad", writer)
    assert ST.run(True) == 1 and "изменил выходы" in capsys.readouterr().out


def test_registry_checkpoint_and_reproduction_are_consistent():
    assert CK.registry(ST.compute()) == []
    fin = D.read_cp()["FIN"]
    assert fin["configs_spent_by_FIN"] == 0 and fin["places"] == 11 and fin["computed"] == 9 and fin["not_run_by_condition"] == [9, 11] and fin["reserve_used"] is False
    assert set(fin["tables_sha"]) == set(D.TABLES) and fin["tables_sha"] == ST.tables_sha()
    txt = A.REPRO.read_text(encoding="utf-8")
    assert all("--stage %s --check" % s in txt for s in A.STAGES) and "FIN" in A.STAGES


def test_mutation_registry_with_duplicate_key_or_reserve_is_caught(monkeypatch, tmp_path):
    r = ST.compute()
    tr = (IO.REP / "ALL_TRIALS.csv").read_text(encoding="utf-8").splitlines()
    for extra, tag in ((tr[1], "дубль"), ("Ж-233:12,Ж-233,x,reserve,{},trial,2026-11-09T21:00:00+03:00,зарегистрирована", "резерв")):
        fake = tmp_path / ("trials_%s" % tag)
        fake.mkdir()
        (fake / "ALL_TRIALS.csv").write_text("\n".join(tr + [extra]) + "\n", encoding="utf-8")
        monkeypatch.setattr(CK, "TRIALS", fake / "ALL_TRIALS.csv")
        assert any("G51" in x for x in CK.registry(r)), tag
    monkeypatch.undo()
    assert not [x for x in CK.registry(r) if "дубли" in x or "резерв" in x]


def test_other_checkpoint_keys_untouched_by_fin():
    cp = D.read_cp()
    assert ST.fin_entry(cp, len(json.dumps(1)))["other_keys_sha"] == cp["FIN"]["other_keys_sha"] and {"0", "1a", "1b", "2", "3", "4", "DX"} <= set(cp)
