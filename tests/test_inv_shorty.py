"""Тест fastbt/inv_shorty.py: синтетика с известным ответом, мутация «рабочие дни», сверка с отчётом."""
from datetime import date, timedelta

import pytest

from fastbt import inv_shorty as m


def _row(d0, d1, side, div="0"):
    return {"data_vhoda": d0, "data_vykhoda": d1, "napravlenie": side,
            "dividend_D087_dolya_vkhoda": div}


# Пятница -> следующий вторник: 4 календарных дня, но 2 рабочих.
ROWS = [
    _row("2018-01-05", "2018-01-09", "short", "0.0087"),
    _row("2018-01-01", "2018-01-11", "long", "0.5"),  # лонг не считается
]


def test_calendar_days_short_only():
    assert m.plata_za_short(ROWS) == pytest.approx(-100 * 4 * 0.10 / 365)


def test_empty_ledger_is_zero():
    assert m.plata_za_short([]) == 0.0
    assert m.dobavka_D([]) == 0.0


def test_dobavka_short_only():
    # 0,0087 * 0,13/0,87 = 0,0013 -> -0,13 п. п.; дивиденд лонга игнорируется
    assert m.dobavka_D(ROWS) == pytest.approx(-0.13)


def _workdays(row):
    a = date.fromisoformat(row["data_vhoda"])
    b = date.fromisoformat(row["data_vykhoda"])
    return sum(1 for i in range((b - a).days) if (a + timedelta(i)).weekday() < 5)


def test_mutation_workdays_caught(monkeypatch):
    good = m.plata_za_short(ROWS)
    monkeypatch.setattr(m, "dni", _workdays)
    assert m.plata_za_short(ROWS) != pytest.approx(good)


def test_sverka_s_otchetom():
    """П. 8 отчёта: № 3 и № 4 по реальным журналам, допуск 0,05 п. п. (округление до 0,1)."""
    exp3 = [-1.5, -2.0, -2.3, -38.9, -27.6, -57.4]
    exp4 = [0, 0, 0, -1.1, -2.3, -9.5]
    for k in range(1, 7):
        rows = m.chitat(m.DIAG / f"S{k}_sdelki_posle370.csv")
        assert m.plata_za_short(rows) == pytest.approx(exp3[k - 1], abs=0.05)
        assert m.dobavka_D(rows) == pytest.approx(exp4[k - 1], abs=0.05)
