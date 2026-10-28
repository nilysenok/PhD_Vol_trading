"""Тесты Л-8 (walk-forward, PBO_wf). Синтетика с известным ответом, без данных репозитория."""
import sys
import types

import numpy as np
import pandas as pd
import pytest

try:  # Л-7 может быть ещё не слита: тогда простой двойник с тем же контрактом
    import fastbt.progon_region  # noqa: F401
except ImportError:
    _m = types.ModuleType("fastbt.progon_region")

    def _vybor(M, rows, daty, kvartaly_idx=None, radius=1, tsel="mediana"):
        kd = pw.kvartal_dnya(daty)
        best, sc = None, -np.inf
        for j in range(M.shape[1]):
            v = [pw._sr(M[kd == k, j]) for k in kvartaly_idx]
            s = float(np.median(v)) if tsel == "mediana" else float(np.mean(np.array(v) > 0))
            if s > sc:
                best, sc = j, s
        return {"centr": best, "chleny": [best], "ryad": M[:, best], "score": sc, "osi": []}

    _m.vybor_region = _vybor
    sys.modules["fastbt.progon_region"] = _m

from fastbt import progon_wf as pw  # noqa: E402

DATY = pd.bdate_range("2016-01-01", "2019-12-31").values.astype("datetime64[D]")
T = len(DATY)
KD = pw.kvartal_dnya(DATY)


def _noise(n=12, seed=0, sd=0.01):
    return np.random.default_rng(seed).normal(0, sd, (T, n))


def test_kvartaly_granicy():
    assert KD[0] == 0 and KD[-1] == 15
    assert set(np.unique(KD)) == set(range(16))


def test_signal_only_2016_2017_gives_oos_near_zero():
    srs = []
    for seed in range(5):
        M = _noise(12, seed)
        M[KD < 8, 0] += 0.01  # сигнал только в 2016–2017
        srs.append(pw._sr(pw.wf(M, [{}] * 12, DATY, pw.OSNOVNOE)["oos_ryad"]))
    assert abs(np.mean(srs)) < 0.8  # без сигнала в OOS ряд — шум, не 2+ как внутри выборки
    M = _noise(12, 9)
    M[KD < 8, 0] += 0.01
    assert pw._sr(M[KD < 8, 0]) > 3


def test_stationary_signal_oos_positive():
    M = _noise(12, 1)
    M[:, 3] += 0.004
    r = pw.wf(M, [{}] * 12, DATY, pw.OSNOVNOE)
    assert pw._sr(r["oos_ryad"]) > 1.5


def test_oos_common_for_all_rules_and_no_leak_window():
    M = _noise(6, 2)
    n_oos = int(np.sum(KD >= 8))
    for p in pw.PRAVILA:
        r = pw.wf(M, [{}] * 6, DATY, p)
        assert len(r["oos_ryad"]) == n_oos
        assert list(r["oos_by_quarter"]) == [f"{y}Q{k}" for y in (2018, 2019) for k in range(1, 5)]
    assert pw.okno_otbora(4, 8) == [4, 5, 6, 7] and pw.okno_otbora(8, 8) == list(range(8))
    assert pw.okno_otbora(4, 8) != pw.okno_otbora(8, 8)


def test_pravila_osnovnoe():
    assert len(pw.PRAVILA) == 8 and len(set(pw.PRAVILA)) == 8
    assert pw.OSNOVNOE == (8, 1, "mediana") and pw.OSNOVNOE in pw.PRAVILA


def test_selection_never_sees_oos_quarter(monkeypatch):
    seen = []
    orig = pw.vybor_region

    def spy(M, rows, daty, kvartaly_idx=None, radius=1, tsel="mediana"):
        seen.append(list(kvartaly_idx))
        return orig(M, rows, daty, kvartaly_idx=kvartaly_idx, radius=radius, tsel=tsel)

    monkeypatch.setattr(pw, "vybor_region", spy)
    pw.wf(_noise(4, 3), [{}] * 4, DATY, (4, 1, "dolya"))
    assert [s[-1] + 1 for s in seen] == list(range(8, 16))
    assert all(len(s) == 4 for s in seen)


def test_mutation_window_captures_oos_quarter(monkeypatch):
    """Окно, захватившее q, ловится защитой wf (утечка: спайк внутри q)."""
    M = _noise(8, 4)
    for q in pw.OOS_KV:
        M[KD == q, q - 8] += 0.05  # у каждого q свой столбец-спайк, виден только внутри q
    assert pw._sr(pw.wf(M, [{}] * 8, DATY, pw.OSNOVNOE)["oos_ryad"]) < 1.5
    monkeypatch.setattr(pw, "okno_otbora", lambda W, q: list(range(q - W + 1, q + 1)))
    with pytest.raises(ValueError):
        pw.wf(M, [{}] * 8, DATY, pw.OSNOVNOE)


def test_mutation_oos_starts_2017_for_w4_is_caught(monkeypatch):
    n_oos = int(np.sum(KD >= 8))
    monkeypatch.setattr(pw, "OOS_KV", tuple(range(4, 16)))
    r = pw.wf(_noise(4, 5), [{}] * 4, DATY, (4, 1, "mediana"))
    assert len(r["oos_ryad"]) != n_oos  # проверка длины общего OOS ловит мутацию


def test_tablica_shape():
    t = pw.tablica(_noise(5, 6), [{}] * 5, DATY)
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
