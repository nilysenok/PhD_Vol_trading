"""Y-A (Ж-243), признаки дня d: карта слотов против паспорта V7; s_d — известный ответ; PIT (бар дня t⁺ в s_d ловится); Δ_d по календарю t⁺ (мутация ловится); e_d = колонка s X1 = de_e1_state.states.
Тесты на реальных файлах только читают."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_ya_feat as FE

X1_FILE = "data/regen/harj_deep/E1/x1.parquet"


@pytest.fixture(scope="module")
def real():
    from models_v2.s15 import de0_data as D
    x1 = pd.read_parquet(X1_FILE)
    return dict(x1=x1, store=FE.load_store(), de_store=D.load_store(), raw=np.load(D.OUT7 / "store_raw.npz", allow_pickle=True))


def test_slot_map_matches_v7_passport(real) -> None:
    import importlib
    seq = importlib.import_module("models_v2.s14.v7_0_seq")
    assert (FE.T_OPEN, FE.SLOT_MIN, seq.T0, seq.SLOTS) == (600, 10, 600, 53)
    assert (FE.slot_of("10:00"), FE.slot_of("17:40"), FE.slot_of("18:30"), FE.slot_of("18:40")) == (0, 46, 51, 52)
    assert list(FE.CLOSE_SLOTS) == [46, 47, 48, 49, 50, 51]
    ch = [str(c) for c in real["raw"]["channels"]]
    tod = real["raw"]["raw"][:, :, ch.index("tod")]          # доля сессии = (старт − 10:00)/520 минут, записана складом
    for k in range(53):
        v = tod[:, k][np.isfinite(tod[:, k])]
        assert np.allclose(v, 10 * k / 520.0, atol=1e-6)


def test_day_features_known_answer_and_zero_cases() -> None:
    r2, pres = np.ones(53), np.ones(53, bool)
    r2[FE.CLOSE_SLOTS] = 4.0
    s, n, nl, z = FE.day_features(r2, pres)
    assert (n, nl, z) == (53, 6, False)
    assert s == pytest.approx(np.log(24.0 * 53 / 6) - np.log(47.0 + 24.0), abs=1e-12)
    p2 = pres.copy()
    p2[FE.CLOSE_SLOTS] = False                               # нет части закрытия
    assert FE.day_features(r2, p2)[0] == 0.0 and FE.day_features(r2, p2)[3] is True
    r0 = r2.copy()
    r0[FE.CLOSE_SLOTS] = 0.0                                 # RV^last = 0
    assert FE.day_features(r0, pres)[0] == 0.0 and FE.day_features(r0, pres)[3] is True
    r1 = r2.copy()
    r1[52] = 50.0                                            # слот 52 (18:40) не в части закрытия, но в RV_d
    assert FE.day_features(r1, pres)[0] == pytest.approx(np.log(24.0 * 53 / 6) - np.log(47.0 + 24.0 + 49.0), abs=1e-12)


def synth_store(seed: int = 0, n: int = 30) -> dict:
    rng = np.random.default_rng(seed)
    return dict(days=pd.bdate_range("2019-01-09", periods=n), r2=rng.random((n, 53)) + 0.1, present=np.ones((n, 53), bool))


def test_pit_day_d_only_and_t_plus_bar_mutation_is_caught(monkeypatch) -> None:
    st = synth_store()
    idx = st["days"][5:25]
    tp = idx + pd.offsets.BDay(1)
    base = FE.features(st, idx, tp)["s"]
    st2 = {**st, "r2": st["r2"].copy()}
    st2["r2"][6:] *= 7.0                                     # бары ПОСЛЕ d меняются (в том числе день t⁺)
    st2["r2"][6:, :46] *= 3.0
    after = FE.features(st2, idx[:1], tp[:1])["s"]           # день d = индекс 5; бары дня 6 (= t⁺) изменены
    assert after[0] == base[0]                               # честный признак дня d не видит t⁺
    monkeypatch.setattr(FE, "feat_row", lambda i: i + 1)     # мутация: бары дня t⁺ в s_d
    assert FE.features(st2, idx[:1], tp[:1])["s"][0] != base[0]
    assert not np.allclose(FE.features(st, idx, tp)["s"], base)


def test_delta_known_answer_and_calendar_mutation(real, monkeypatch) -> None:
    d = pd.DatetimeIndex(["2019-03-04", "2019-03-08", "2019-03-07", "2018-12-28"])
    tp = pd.DatetimeIndex(["2019-03-05", "2019-03-11", "2019-03-11", "2019-01-09"])
    assert FE.delta_days(d, tp).tolist() == [0.0, 2.0, 3.0, 11.0]
    x1 = real["x1"]
    dd = FE.delta_days(x1["d"], x1["t_plus"])
    assert dd.min() == 0 and dd.max() >= 4 and (dd == 2).sum() > 300       # реальные праздники растягивают разрыв
    monkeypatch.setattr(FE, "delta_days", lambda d, tp: np.where(pd.DatetimeIndex(d).dayofweek == 4, 2.0, 0.0))     # мутация: Δ не из t⁺, а по дню недели
    assert not np.array_equal(FE.delta_days(x1["d"], x1["t_plus"]), dd)


def test_e_d_is_x1_column_equal_to_de_states_and_mutation_is_caught(real, monkeypatch) -> None:
    from models_v2.s15 import de_e1_state as S
    x1 = real["x1"]
    want = S.states(real["de_store"], pd.DatetimeIndex(x1["d"]))
    assert np.array_equal(FE.e_state(x1), want)
    monkeypatch.setattr(FE, "e_state", lambda x: np.roll(x["s"].to_numpy(int), 1))      # мутация: e_d не из X1
    assert not np.array_equal(FE.e_state(x1), want)


def test_real_features_rv_d_matches_target_and_reports_zero_days(real) -> None:
    x1 = real["x1"]
    ft = FE.features(real["store"], x1["d"], x1["t_plus"])
    target = dict(zip(pd.DatetimeIndex(x1["t_plus"]), x1["rv"]))
    rel = [abs(target[d] - r) / target[d] for d, r in zip(pd.DatetimeIndex(x1["d"]), ft["rv_d"]) if d in target]
    assert len(rel) == len(x1) - 2 and max(rel) < 1e-6        # RV_d склада = цель RV(t⁺) строки предыдущего дня
    assert int(ft["zero"].sum()) == 0 and set(np.unique(ft["nlast"])) == {6}
    assert np.isfinite(ft["s"]).all() and ft["n"].min() >= 41
