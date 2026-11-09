"""T0 (Ж-242): таблица QLIKE — известный ответ на синтетике и три обязательные мутации: (1) простое среднее показателей периодов вместо дневных потерь; (2) не общие даты; (3) подменённое число.
Сеть не нужна; тесты на реальных файлах только читают сохранённые прогнозы."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_t0 as X


def synth(f_year1: float = 4.0, seed: int = 0) -> dict:
    """8 рядов на рабочих днях 2018-01-03…2026-02-26; K1: F = 2·RV в 2018 (по t⁺) и f_year1·RV позже — разные потери по годам при разном n годов."""
    d = pd.bdate_range("2018-01-03", "2026-02-26")
    tp = pd.DatetimeIndex(list(d[1:]) + [pd.Timestamp("2026-02-27")])
    rng = np.random.default_rng(seed)
    rv = np.exp(rng.normal(0, 0.3, len(d)))
    fr = {}
    for k in X.SPEC:
        F = rv * np.exp(rng.normal(0.1, 0.2, len(d)))
        if k == "K1":
            F = rv * np.where(tp.year == 2018, 2.0, f_year1)
        fr[k] = pd.DataFrame(dict(d=d, t_plus=tp, rv=rv, F=F))
    return fr


def table(fr: dict) -> pd.DataFrame:
    return X.qlike_table(fr, X.common(fr), {k: len(v) for k, v in fr.items()})


def l_of(r: float) -> float:
    return r - np.log(r) - 1.0          # l = RV/F − ln(RV/F) − 1, r = RV/F


def test_known_answer_daily_losses_not_mean_of_periods() -> None:
    fr = synth()
    t = table(fr).set_index("series")
    tp = pd.DatetimeIndex(fr["K1"]["t_plus"])
    n18 = int((tp.year == 2018).sum())
    want = (n18 * l_of(0.5) + (len(tp) - n18) * l_of(0.25)) / len(tp)         # среднее дневных l
    assert t.at["K1", "q_all"] == pytest.approx(want, abs=1e-12)
    assert t.at["K1", "q_2018"] == pytest.approx(l_of(0.5), abs=1e-12)
    means = np.mean([t.at["K1", "q_%d" % y] for y in X.YEARS])
    assert abs(want - means) > 1e-3                      # при неравных n годов среднее показателей периодов — другое число


def test_mutation_mean_of_period_scores_is_caught(monkeypatch) -> None:
    fr, orig = synth(), X.win_q

    def mutant(rv, f, tp, lo, hi):
        if (lo, hi) == X.WINS["all"]:
            return float(np.mean([orig(rv, f, tp, y, y) for y in X.YEARS]))
        return orig(rv, f, tp, lo, hi)

    tp = pd.DatetimeIndex(fr["K1"]["t_plus"])
    n18 = int((tp.year == 2018).sum())
    want = (n18 * l_of(0.5) + (len(tp) - n18) * l_of(0.25)) / len(tp)
    assert table(fr).set_index("series").at["K1", "q_all"] == pytest.approx(want, abs=1e-12)
    monkeypatch.setattr(X, "win_q", mutant)
    assert abs(table(fr).set_index("series").at["K1", "q_all"] - want) > 1e-3          # мутант не совпал с известным ответом


def test_common_dates_intersection_and_hole() -> None:
    fr = synth()
    fr["V7:cfg8"] = fr["V7:cfg8"].iloc[100:].reset_index(drop=True)                       # ряд позже начинается
    fr["K2"] = fr["K2"].drop(index=500).reset_index(drop=True)                             # дыра в середине
    al = X.common(fr)
    want = pd.DatetimeIndex(fr["V7:cfg8"]["d"]).difference([fr["K1"]["d"].iloc[500]])
    assert all(pd.DatetimeIndex(v["d"]).equals(want) for v in al.values())
    t = table(fr)
    assert set(t["n_all"]) == {len(want)} and set(t["first_all"]) == {str(al["K1"]["t_plus"].min().date())}


def test_mutation_not_common_dates_is_caught(monkeypatch) -> None:
    fr = synth()
    fr["V7:cfg8"] = fr["V7:cfg8"].iloc[100:].reset_index(drop=True)
    table(fr)                                                                           # честный путь проходит
    monkeypatch.setattr(X, "common", lambda f: {k: v.copy() for k, v in f.items()})     # мутант: каждый ряд на своих датах
    with pytest.raises(SystemExit):
        table(fr)


def test_target_mismatch_stops() -> None:
    fr = synth()
    fr["X1"].loc[10, "rv"] *= 1.01
    with pytest.raises(SystemExit):
        X.common(fr)


def test_mutation_substituted_number_is_caught() -> None:
    t = table(synth())
    text = X.csv_text(t)
    assert X.diff(t, text, "T") == []
    v = "%r" % float(t["q_all"].iloc[0])
    assert v in text
    assert X.diff(t, text.replace(v, "%r" % (float(t["q_all"].iloc[0]) + 1e-4), 1), "T") == ["T: q_all"]


@pytest.fixture(scope="module")
def real():
    return X.compute()


def test_real_period_and_disk_tables_match(real) -> None:
    t, s, md = real
    assert set(t["common_first_d"]) == {"2018-01-03"} and set(t["first_all"]) == {"2018-01-04"}      # 2017 в общий период не входит
    T = t.set_index("series")
    assert [bool(T.at[k, "has_2017"]) for k in ("K1", "K2", "K3", "V6:A2", "V4:NTP2", "V7:cfg6", "V7:cfg8", "X1")] == [True] * 5 + [False] * 3
    assert X.diff(t, (X.REP / "P0_TABLE.csv").read_text(encoding="utf-8"), "t") == []
    assert X.diff(s, (X.REP / "P0_STATS.csv").read_text(encoding="utf-8"), "s") == []
    assert (X.REP / "P0_TABLE.md").read_text(encoding="utf-8") == md


def test_check_writes_nothing_and_catches_substitution(real, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(X, "compute", lambda: real)
    X.write(tmp_path)
    assert X.check(tmp_path) == []
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    p = tmp_path / "P0_TABLE.csv"
    txt = p.read_text(encoding="utf-8")
    v = "%r" % float(real[0]["q_all"].iloc[0])
    p.write_text(txt.replace(v, "%r" % (float(real[0]["q_all"].iloc[0]) + 1e-3), 1), encoding="utf-8")
    assert any("q_all" in b for b in X.check(tmp_path))
    p.write_text(txt, encoding="utf-8")
    (tmp_path / "P0_TABLE.md").write_text(real[2].replace("0.2", "0.3", 1), encoding="utf-8")
    assert X.check(tmp_path) == ["P0_TABLE.md: текст ≠ генератору"]
    assert {q.name: q.read_bytes() for q in tmp_path.iterdir() if q.name != "P0_TABLE.md"} == {k: v for k, v in before.items() if k != "P0_TABLE.md"}
