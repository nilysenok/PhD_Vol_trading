"""V7 лист 1b-B: выбор энкодера для конфигурации 3 — только по датам целей 2018–2021 (мутация «окно с 2022+»), общие даты, ничья; строки результата в ALL_TRIALS (дописываются, регистрация не меняется);
хэш содержимого прогнозов; запись choice3 не переписывается."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s14 import v7_1b_cfg as CFG, v7_1b_cfgrun as CR, v7_1m_reg as RG


def frame(shift_early, shift_late, seed=0):
    """Прогноз = RV·exp(ошибка): в 2018–2021 смещение shift_early, в 2022+ — shift_late (чем ближе к 0, тем лучше)."""
    d = pd.bdate_range("2018-01-03", "2025-12-30")
    tp = d + pd.Timedelta(days=1)
    rv = np.exp(np.random.default_rng(seed).normal(0, 0.2, len(d)))
    sh = np.where(tp.year <= 2021, shift_early, shift_late)
    return pd.DataFrame({"d": d, "t_plus": tp, "rv": rv, "H": rv, "g": 0.0, "F": rv * np.exp(sh)})


def test_choose_by_2018_2021_only_and_mutation_window_2022_plus_is_caught(monkeypatch):
    dfs = dict(tcn=frame(0.30, 0.02), cnn=frame(0.05, 0.40, seed=0))          # CNN лучше в 2018–2021, TCN лучше в 2022+
    ch = CFG.choose(dfs)
    assert ch["choice"] == "cnn" and ch["qlike"]["cnn"] < ch["qlike"]["tcn"] and ch["window"] == "t⁺ 2018–2021"
    monkeypatch.setattr(CFG, "CHOICE_WIN", (2022, 2025))
    assert CFG.choose(dfs)["choice"] == "tcn"                                 # мутант (выбор по 2022+) даёт другой ответ — тест его ловит
    monkeypatch.setattr(CFG, "CHOICE_WIN", (2018, 2026))
    assert CFG.choose(dfs)["choice"] == "tcn"                                 # окно «всё» тоже ловится: 2022+ перевешивает


def test_choice_does_not_depend_on_forecasts_after_2021():
    a, b = frame(0.30, 0.02), frame(0.05, 0.40)
    base = CFG.choose(dict(tcn=a, cnn=b))
    a2, b2 = a.copy(), b.copy()
    late = a2["t_plus"].dt.year >= 2022
    a2.loc[late, "F"], b2.loc[late, "F"] = a2.loc[late, "rv"] * 50, b2.loc[late, "F"] * 1e-3
    assert CFG.choose(dict(tcn=a2, cnn=b2)) == base                           # 2022+ полностью заменены — выбор и QLIKE те же


def test_common_dates_and_tie_goes_to_tcn():
    a, b = frame(0.1, 0.1), frame(0.1, 0.1)
    assert CFG.choose(dict(tcn=a, cnn=b))["choice"] == "tcn"
    b2 = b.copy()
    b2.loc[b2["t_plus"].dt.year == 2019, "F"] = np.nan                         # даты, где у одной конфигурации нет F, в сравнение не входят
    ch = CFG.choose(dict(tcn=a, cnn=b2))
    assert ch["n_dates"] == int(((a["t_plus"].dt.year <= 2021) & (a["t_plus"].dt.year != 2019)).sum())


def test_content_hash_sensitive_to_one_ulp_and_order_of_rows():
    a = frame(0.1, 0.1)
    h = CR.content_hash(a)
    b = a.copy()
    b.loc[10, "F"] = np.nextafter(b.loc[10, "F"], np.inf)
    assert CR.content_hash(b) != h and CR.content_hash(a.iloc[::-1].reset_index(drop=True)) != h and CR.content_hash(a.copy()) == h


@pytest.fixture()
def reg(tmp_path):
    t = tmp_path / "ALL_TRIALS.csv"
    RG.ensure(t, when="T")
    return t


def test_result_rows_are_appended_registration_untouched_and_not_rewritten(reg):
    head = reg.read_bytes()
    assert RG.add_result("1", dict(q=0.1), reg, when="T2") == "дописана" and RG.add_result("2", dict(q=0.2), reg, when="T2") == "дописана"
    new = reg.read_bytes()
    assert new.startswith(head) and RG.bad(reg) == []                          # строки регистрации побайтно прежние; результат реестр не нарушает
    assert RG.add_result("1", dict(q=999), reg) == "уже есть" and reg.read_bytes() == new
    assert RG.ensure(reg, when="T3") == "без изменений"
    RG.gate("1", trials=reg)                                                   # допуск конфигурации по-прежнему работает


def test_bad_still_flags_foreign_rows_wrong_status_or_unknown_base(reg):
    RG.add_result("1", dict(q=0.1), reg, when="T")
    good = reg.read_text(encoding="utf-8")
    reg.write_text(good.replace(",посчитана", ",зарегистрирована"), encoding="utf-8")
    assert any("вне плана" in b for b in RG.bad(reg))
    reg.write_text(good.replace("Ж-233:1:результат", "Ж-233:12:результат"), encoding="utf-8")
    assert any("вне плана" in b for b in RG.bad(reg))
    reg.write_text(good + good.splitlines()[-1] + "\n", encoding="utf-8")
    assert any("дубль" in b for b in RG.bad(reg))
