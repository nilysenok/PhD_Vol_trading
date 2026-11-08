"""V6-0, G5: паспорт M10 SBER и тождество цели Σr² = RV. Мутации: выброшенный бар ломает тождество; смена расписания считается, разовая дыра — нет; отсутствие колонки объёма меняет вычислимость E."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s10 import hp_paths as P
from models_v2.s12 import v5_0_bars as BA
from models_v2.s13 import v6_0_load as LD, v6_0_pass as PA


@pytest.fixture(scope="module")
def real():
    w, _ = LD.load()
    return w, PA.passport(w)


def test_g5_identity_real_data(real):
    _, (tab, res) = real
    y = tab[tab["section"] == "year"]
    assert res["max_daily"] <= PA.TOL_ID and res["max_target"] <= PA.TOL_ID
    assert y["year"].astype(int).tolist() == list(range(2014, 2027)) and y["bars_mode"].astype(int).tolist() == [52] + [53] * 12 and (y["share_zero_volume_bars"] == 0).all()
    assert y.set_index(y["year"].astype(int)).loc[2015, "share_not_mode"] > 0.3 and int(y["schedule_changes"].sum()) == 1            # смена расписания 2014→2015 (52 → 53 бара)
    assert int(tab[tab["section"] == "total"]["n_target_days"].iloc[0]) == 2271


def test_inputs_a_to_f_all_present_with_status(real):
    t = real[1][0]
    x = t[t["section"] == "inputs"].set_index("direction")
    assert list(x.index) == list("ABCDEF") and set(x["computable"]) <= {"да", "нет", "с оговоркой"} and x.loc["C", "computable"] == "с оговоркой" and x.loc["E", "computable"] == "да"


def _bars(days, n=53, zero_vol=0):
    rng = np.random.default_rng(0)
    rows = []
    for d in days:
        px = 100.0 + np.cumsum(rng.normal(0, 0.05, n)).round(2)
        for i in range(n):
            o = px[i - 1] if i else 100.0
            rows.append(dict(begin=pd.Timestamp(d) + pd.Timedelta(hours=10, minutes=10 * i), open=o, high=max(o, px[i]), low=min(o, px[i]), close=px[i], volume=0 if i < zero_vol else 1000 + i, value=1.0))
    return pd.DataFrame(rows)


def test_mutation_dropped_bar_breaks_identity_on_real_day():
    bars = pd.read_parquet(P.bars_path("SBER"))
    day = bars[bars["begin"].dt.normalize() == pd.Timestamp("2019-03-15")]
    main = day[(day["begin"].dt.strftime("%H:%M") >= "10:00") & (day["begin"].dt.strftime("%H:%M") <= "18:40")]
    ok = BA.day_table(day)
    ref = float(pd.read_parquet(P.OUT / "daily_SBER.parquet").loc["2019-03-15", "rv"])
    assert PA.identity_max(ok["rv_bars"], pd.Series([ref])) <= PA.TOL_ID
    cut = day.drop(main.index[20])                                              # мутация: выброшен бар внутри сессии
    assert PA.identity_max(BA.day_table(cut)["rv_bars"], pd.Series([ref])) > 1e-4


def test_year_rows_known_answer_synthetic():
    bars = _bars(["2019-01-09", "2019-01-10"], zero_vol=10)
    days = BA.day_table(bars)
    days["rv"] = days["rv_bars"]
    k1 = pd.DataFrame({"t_plus": pd.to_datetime(["2019-01-10", "2019-01-11"]), "rv": [days["rv"].iloc[1], 1.0]}, index=pd.to_datetime(["2019-01-09", "2019-01-10"]))
    y = PA.year_rows(bars, days, k1).iloc[0]
    assert y["n_sessions"] == 2 and y["n_bars"] == 106 and y["bars_mode"] == 53 and y["share_not_mode"] == 0.0
    assert np.isclose(y["share_zero_volume_bars"], 20 / 106)                       # по 10 нулевых баров в каждой из двух сессий and y["max_rel_vs_daily"] < 1e-12 and y["price_step"] == 0.01


def test_regime_changes_counts_persistent_shifts_not_single_gaps():
    assert PA.regime_changes(np.array([53] * 30 + [50] * 30 + [53] * 25)) == 2
    assert PA.regime_changes(np.array([53] * 40 + [52] + [53] * 40)) == 0
    assert PA.regime_changes(np.array([53] * 30 + [50] * 5 + [53] * 30)) == 0           # разовая серия < 20 сессий не режим


def test_inputs_without_volume_column_degrades_E():
    bars = _bars(["2019-01-09"]).drop(columns=["volume", "value"])
    yrs = pd.DataFrame({"share_zero_volume_bars": [0.0], "share_days_with_gaps": [0.0], "schedule_changes": [0]})
    x = PA.inputs_rows(bars, yrs).set_index("direction")
    assert x.loc["E", "computable"] == "с оговоркой" and "отсутствует" in x.loc["E", "note"]
    full = PA.inputs_rows(_bars(["2019-01-09"]), yrs).set_index("direction")
    assert full.loc["E", "computable"] == "да"


def test_tick_known_answer():
    assert np.isclose(PA.tick(np.array([100.00, 100.01, 100.03, 100.01])), 0.01)
