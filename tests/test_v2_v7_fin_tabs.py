"""V7-FIN (G49): сводные таблицы. Известный ответ на синтетике (доли дней, кризисные дни, преимущество, годы); реальные таблицы FIN_* = таблицам листов; журнал [V7] выбирается дословно; перепутанный знак ловится."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s14 import v7_0_io as IO, v7_fin_data as D, v7_fin_tabs as TB


def synth(n_per=60):
    """Даты t⁺ 2018…2025 по n_per на год; rv = 1; cfg8 = rv (лучшая), K2/K3/NTP2/A2 = 2·rv, остальные = 3·rv; K1 растёт со временем, в 2019 — гигантский выброс."""
    ti = np.concatenate([np.datetime64("%d-03-01" % y) + np.arange(n_per) for y in range(2018, 2026)])
    rv = np.ones(len(ti))
    s = {k: 3.0 * rv for k in D.SER}
    for k in ("K2", "K3", "V4:NTP2", "V6:A2"):
        s[k] = 2.0 * rv
    s["cfg8"] = 1.1 * rv
    s["K1"] = np.arange(len(ti), dtype=float) + 1.0
    s["K1"][(pd.DatetimeIndex(ti).year == 2019)] = 1e6
    return dict(rv=rv, tp=ti, s=s)


def test_after_known_answer_shares_and_crisis_window_only_after_2021():
    d = synth()
    a = TB.after(d).set_index("series")
    assert a.loc["cfg8", "share_better_K2_2022_2025"] == 1.0 and a.loc["cfg8", "share_better_A2_2018_2021"] == 1.0
    assert a.loc["K2", "share_better_K2_2022_2025"] == 0.0 and a.loc["cfg1", "share_better_K3_2022_2025"] == 0.0     # мутация «>» вместо «<» поменяла бы 1 и 0
    n22 = 4 * 60
    k = int(round(0.05 * n22))
    assert a.loc["K1", "n_2018_2021"] == 240 and a.loc["K1", "n_2022_2025"] == n22 and abs(a.loc["K1", "n_crisis"] - k) <= 1
    assert a.loc["K1", "qlike_crisis"] > a.loc["K1", "qlike_2022_2025"]                                           # наибольшие прогнозы K1 → наибольшая QLIKE; выброс 2019 не входит
    assert a.loc["cfg8", "qlike_2022"] < a.loc["K2", "qlike_2022"] < a.loc["cfg1", "qlike_2022"] and a.loc["cfg8", "ratio_2022_2025_to_2018_2021"] == pytest.approx(1.0)


def test_mutation_crisis_over_all_years_would_take_2019_outlier():
    d = synth()
    ti = pd.DatetimeIndex(d["tp"])
    k1 = d["s"]["K1"]
    m22 = (ti.year >= 2022) & (ti.year <= 2025)
    ok = m22 & (k1 >= np.quantile(k1[m22], 0.95))
    bad = k1 >= np.quantile(k1, 0.95)
    assert (ti[ok].year >= 2022).all() and (ti[bad].year == 2019).any() and TB.after(d).set_index("series").loc["K1", "n_crisis"] == int(ok.sum())


def test_advantage_requires_all_four_bases_below_zero():
    rows = [dict(config="a", window="2018_2021", base=b, hi90=-0.1) for b in D.BASES] + [dict(config="b", window="2018_2021", base=b, hi90=-0.1 if b != "K3" else 0.2) for b in D.BASES]
    rows += [dict(config="c", window="2022_2025", base=b, hi90=-1.0) for b in D.BASES]
    assert TB.advantage({"FIN_TESTS": pd.DataFrame(rows)}) == ["a"]


def test_best_and_years_known_answer():
    allm = pd.DataFrame([dict(series=s, window=w, qlike=q) for s, w, q in [("cfg1", "2018_2021", 0.3), ("cfg8", "2018_2021", 0.2), ("K1", "2018_2021", 0.1), ("cfg8", "2018", 0.5), ("cfg1", "2018", 0.6)]])
    assert TB.best({"FIN_ALL": allm}, "2018_2021") == "cfg8"          # K1 — контроль, в «лучшие конфигурации» не входит
    y = TB.years(pd.DataFrame([dict(series=s, window=str(w), qlike=float(w) + i) for i, s in enumerate(D.SER) for w in range(2018, 2027)]))
    assert list(y["year"]) == list(range(2018, 2027)) and list(y.columns) == ["year"] + D.SER and y.loc[0, "K1"] == 2018.0


def test_departures_extracts_headers_with_bullets_and_skips_other_programs(tmp_path):
    j = tmp_path / "j.md"
    j.write_text("шапка\n↺ — [V7] агент А:\n- пункт один\n  - вложенный\nтекст\n↺ — [V5, к Ж-234] чужое\nМоя ошибка [V7]: просчёт.\n**Моя ошибка [V7] (дирижёр).**\n- следствие\n", encoding="utf-8")
    d = D.departures(j)
    assert list(d["line"]) == [2, 7, 8] and list(d["kind"]) == ["отступление", "ошибка дирижёра", "ошибка дирижёра"]
    assert "пункт один" in d.iloc[0]["text"] and "вложенный" in d.iloc[0]["text"] and "текст" not in d.iloc[0]["text"] and "следствие" in d.iloc[2]["text"]


def test_real_journal_has_all_v7_departures_and_the_conductor_errors():
    d = D.departures()
    assert len(d) >= 15 and (d["kind"] == "ошибка дирижёра").sum() >= 3 and d["text"].str.contains("93 параметра").any() and d["text"].str.contains("каким энкодером считать z").any()


def test_registry_real_eleven_places_nine_computed_two_not_run_v1_archived():
    r = D.registry().set_index("place")
    assert len(r) == 11 and int((r["status"] == "посчитана").sum()) == 9 and r.loc[9, "status"].startswith("не считается") and r.loc[11, "status"].startswith("не считается")
    assert bool(r.loc[1:7, "v1_archived"].all()) and not r.loc[8:11, "v1_archived"].any()


def test_real_tables_equal_sheets_and_passthrough_counts():
    t = D.sheets()
    assert len(t["FIN_PAIRS"]) == sum(len(IO.read_csv(n + "_PAIRS.csv")) for n in "234") and len(t["FIN_GO"]) == 7 * 7 + 7 + 7
    assert set(t["FIN_GO"].query("cond == 0")["config"]) == set(D.CFG)
    disk = IO.read_csv("4_METRICS.csv")
    assert set(disk["series"]) == set(D.SER)


def test_weights_stage_a_year_of_cfg10_has_no_stage_b_rows():
    w = D.sheets()["FIN_WEIGHTS"]
    x = w[(w["config"] == "cfg10") & (w["quantity"] == "pi0") & (w["period"] == "2018")]
    assert int(x["n"].iloc[0]) == 0 and np.isnan(float(x["mean"].iloc[0]))
