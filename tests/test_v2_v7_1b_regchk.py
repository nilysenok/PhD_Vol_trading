"""V7 лист 1b-C: проверка реестра после дописывания результатов (замена заглушки `len(RG.read()) == 12`). Мутации: изменена строка регистрации; результат незарегистрированной конфигурации; дубль результата;
строка регистрации после результатов. Реальный реестр читается только на копию в tmp_path."""
from __future__ import annotations

import pytest

from models_v2.s14 import v7_1b_regchk as RC, v7_1m_reg as RG


@pytest.fixture()
def trials(tmp_path):
    if not RG.TRIALS.exists():
        pytest.skip("реестр не записан")
    lines = RG.TRIALS.read_bytes().splitlines(keepends=True)[: RC.REG_LINES]
    p = tmp_path / "ALL_TRIALS.csv"
    p.write_bytes(b"".join(lines))
    return p


def test_registration_alone_is_intact_and_results_ok(trials):
    assert RC.registration_intact(trials) and RC.results_only_registered(trials)
    RG.add_result("1", dict(a=1), trials, when="2026-11-09T17:00:00+03:00")
    RG.add_result("3", dict(a=3), trials, when="2026-11-09T18:00:00+03:00")
    assert RC.registration_intact(trials) and RC.results_only_registered(trials) and RG.bad(trials) == []


def test_mutation_registration_row_changed_is_caught(trials):
    RG.add_result("1", dict(a=1), trials, when="2026-11-09T17:00:00+03:00")
    raw = trials.read_bytes()
    for old, new in ((b"\xd0\x96-233:5,", b"\xd0\x96-233:5 ,"), (b"\xd0\xb7\xd0\xb0\xd1\x80\xd0\xb5\xd0\xb3\xd0\xb8\xd1\x81\xd1\x82\xd1\x80\xd0\xb8\xd1\x80\xd0\xbe\xd0\xb2\xd0\xb0\xd0\xbd\xd0\xb0", b"\xd0\xbf\xd0\xbe\xd1\x81\xd1\x87\xd0\xb8\xd1\x82\xd0\xb0\xd0\xbd\xd0\xb0")):
        assert old in raw
        trials.write_bytes(raw.replace(old, new, 1))
        assert not RC.registration_intact(trials)
    trials.write_bytes(raw)
    assert RC.registration_intact(trials)


def test_mutation_result_of_unregistered_config_or_duplicate_is_caught(trials):
    RG.add_result("1", dict(a=1), trials, when="2026-11-09T17:00:00+03:00")
    raw = trials.read_bytes()
    lines = raw.decode("utf-8").splitlines(keepends=True)
    trials.write_bytes((raw.decode("utf-8") + lines[-1].replace("Ж-233:1:результат", "Ж-233:99:результат")).encode("utf-8"))
    assert not RC.results_only_registered(trials)
    trials.write_bytes((raw.decode("utf-8") + lines[-1]).encode("utf-8"))
    assert not RC.results_only_registered(trials)


def test_mutation_registration_row_after_results_is_caught(trials):
    raw = trials.read_bytes().decode("utf-8").splitlines(keepends=True)
    extra = raw[-1].replace("Ж-233:11,", "Ж-233:11x,")
    trials.write_bytes("".join(raw + [extra]).encode("utf-8"))
    assert not RC.results_only_registered(trials)
