"""Тесты Л-8 (walk-forward, PBO_wf). Синтетика с известным ответом, без данных репозитория.

Правка тестировщика (П-4, фикстуры под настоящий Л-7; требования тестов не ослаблены, утверждения те же):
1. Убран двойник vybor_region и ветка except ImportError: тесты идут на настоящем fastbt.progon_region.
2. rows=[{}]*n заменены на ROWS: настоящему Л-7 нужны klyuch/pt и активность (sdelok_y >= 30 в каждом году, storony_y).
   Четыре бинарные оси дают 16 столбцов; радиус 1 покрывает обе значения оси, и любой регион = все 16 (>= MIN_CHLENOV = 10).
3. Столбцов везде 16 (было 4-12): регион из < 10 членов Л-7 не допускает, и ряд был бы пустым.
4. Дрейф сигнала повышен (0,004 -> 0,02; 0,01 -> 0,05):ансамбль из 16 столбцов разбавляет сигнал одного
   столбца в 16 раз, прежний сигнал не проходил доли + кварталов >= 0,8 и регион не выбирался.
6. Ж-090 (целевая этапа 1 = mSR за окно при mSR каждого года > 0): tsel "mediana" -> "msr" в OSNOVNOE, в ожидаемом регионе и в
   мутации OOS-старта; в test_pravila_osnovnoe добавлено: множество tsel правил = {"msr", "dolya"}. Требования те же.
5. test_mutation_window_captures_oos_quarter: утверждение «OOS Шарп < 1,5» заменено сверкой OOS-ряда с рядом региона, выбранного
   по W кварталам строго до q. Причина: регион настоящего Л-7 здесь = все 16 столбцов, спайк квартала q законно входит в ансамбль,
   и «< 1,5» перестало отличать утечку от честного ряда (Шарп 17 без утечки). Вторая часть (ValueError на окне с q) прежняя.
"""
import itertools

import numpy as np
import pandas as pd
import pytest

from fastbt import progon_wf as pw

DATY = pd.bdate_range("2016-01-01", "2019-12-31").values.astype("datetime64[D]")
T = len(DATY)
KD = pw.kvartal_dnya(DATY)
NC = 16
ROWS = [dict(klyuch={**{a: float(v) for a, v in zip("abcd", p)}, "t_vyhod": "open2", "vhod": "0.5"}, status="ok",  # П-7, Ж-086: оси обязательны
             sdelok_y={str(y): 60 for y in range(2016, 2020)}, storony_y={str(y): 40 for y in range(2016, 2020)})
        for p in itertools.product((0, 1), repeat=4)]


def _noise(n=NC, seed=0, sd=0.01):
    assert n == NC, "ROWS заданы на 16 столбцов"
    return np.random.default_rng(seed).normal(0, sd, (T, n))


def test_kvartaly_granicy():
    assert KD[0] == 0 and KD[-1] == 15
    assert set(np.unique(KD)) == set(range(16))


def test_signal_only_2016_2017_gives_oos_near_zero():
    srs = []
    for seed in range(5):
        M = _noise(16, seed)
        M[KD < 8, 0] += 0.05  # сигнал только в 2016–2017
        srs.append(pw._sr(pw.wf(M, ROWS, DATY, pw.OSNOVNOE)["oos_ryad"]))
    assert abs(np.mean(srs)) < 0.8  # без сигнала в OOS ряд — шум, не 2+ как внутри выборки
    M = _noise(16, 9)
    M[KD < 8, 0] += 0.05
    assert pw._sr(M[KD < 8, 0]) > 3


def test_stationary_signal_oos_positive():
    M = _noise(16, 1)
    M[:, 3] += 0.02
    r = pw.wf(M, ROWS, DATY, pw.OSNOVNOE)
    assert pw._sr(r["oos_ryad"]) > 1.5


def test_oos_common_for_all_rules_and_no_leak_window():
    M = _noise(16, 2)
    n_oos = int(np.sum(KD >= 8))
    for p in pw.PRAVILA:
        r = pw.wf(M, ROWS, DATY, p)
        assert len(r["oos_ryad"]) == n_oos
        assert list(r["oos_by_quarter"]) == [f"{y}Q{k}" for y in (2018, 2019) for k in range(1, 5)]
    assert pw.okno_otbora(4, 8) == [4, 5, 6, 7] and pw.okno_otbora(8, 8) == list(range(8))
    assert pw.okno_otbora(4, 8) != pw.okno_otbora(8, 8)


def test_pravila_osnovnoe():
    assert len(pw.PRAVILA) == 8 and len(set(pw.PRAVILA)) == 8
    assert pw.OSNOVNOE == (8, 1, "msr") and pw.OSNOVNOE in pw.PRAVILA
    assert {t for _, _, t in pw.PRAVILA} == {"msr", "dolya"}


def test_selection_never_sees_oos_quarter(monkeypatch):
    seen = []
    orig = pw.vybor_region

    def spy(M, rows, daty, kvartaly_idx=None, radius=1, tsel="msr"):
        seen.append(list(kvartaly_idx))
        return orig(M, rows, daty, kvartaly_idx=kvartaly_idx, radius=radius, tsel=tsel)

    monkeypatch.setattr(pw, "vybor_region", spy)
    pw.wf(_noise(16, 3), ROWS, DATY, (4, 1, "dolya"))
    assert [s[-1] + 1 for s in seen] == list(range(8, 16))
    assert all(len(s) == 4 for s in seen)


def test_mutation_window_captures_oos_quarter(monkeypatch):
    """Окно, захватившее q, ловится защитой wf (утечка: спайк внутри q)."""
    M = _noise(16, 4)
    for q in pw.OOS_KV:
        M[KD == q, q - 8] += 0.05  # у каждого q свой столбец-спайк, виден только внутри q
    from fastbt.progon_region import ryad_regiona, vybor_region
    oos = pw.wf(M, ROWS, DATY, pw.OSNOVNOE)["oos_ryad"]
    exp = []
    for q in pw.OOS_KV:  # ожидаемый OOS: регион, выбранный только по W кварталам строго до q
        reg = vybor_region(M, ROWS, DATY, kvartaly_idx=list(range(q - 8, q)), radius=1, tsel="msr")
        exp.append(np.zeros(int((KD == q).sum())) if reg is None else ryad_regiona(M, reg["chleny"])[KD == q])
    assert np.allclose(oos, np.concatenate(exp))
    monkeypatch.setattr(pw, "okno_otbora", lambda W, q: list(range(q - W + 1, q + 1)))
    with pytest.raises(ValueError):
        pw.wf(M, ROWS, DATY, pw.OSNOVNOE)


def test_mutation_oos_starts_2017_for_w4_is_caught(monkeypatch):
    n_oos = int(np.sum(KD >= 8))
    monkeypatch.setattr(pw, "OOS_KV", tuple(range(4, 16)))
    r = pw.wf(_noise(16, 5), ROWS, DATY, (4, 1, "msr"))
    assert len(r["oos_ryad"]) != n_oos  # проверка длины общего OOS ловит мутацию


def test_tablica_shape():
    t = pw.tablica(_noise(16, 6), ROWS, DATY)
    assert len(t) == 8 and list(t["pravilo"]) == pw.PRAVILA
    assert {"mediana_oos_kv", "dolya_pol_oos_kv", "sharpe_oos", "region"} <= set(t.columns)
    assert t["dolya_pol_oos_kv"].between(0, 1).all()


def test_pbo_wf_known_answers():
    n = 800
    rng = np.random.default_rng(7)
    # правило 0 всегда лучшее и на train, и на test -> PBO = 0
    oos = rng.normal(0, 0.01, (n, 8))
    oos[:, 0] += 0.05
    assert pw.pbo_wf(oos) == 0.0
    # чистый шум: лучший на train не лучший на test (суммы половин связаны) -> PBO выше, чем у доминирующего
    assert pw.pbo_wf(rng.normal(0, 0.01, (n, 8))) >= 0.5


def test_pbo_wf_rejects_seven_rules():
    oos = np.random.default_rng(8).normal(0, 0.01, (400, 8))
    with pytest.raises(ValueError):
        pw.pbo_wf(oos[:, :7])
