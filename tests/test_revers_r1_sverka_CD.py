"""Тесты сверки «предзащита → s9» для C/D (scripts/revers/r1_sverka_CD.py): синтетика с известным ответом и мутации.
Данных проекта не читают. Запуск: FASTBT_MAIN=<своя копия> python3 -m pytest tests/test_revers_r1_sverka_CD.py -q"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "revers"))
import r1_sverka_CD as SV  # noqa: E402
from fastbt import msp, s9  # noqa: E402

DATES = pd.DatetimeIndex(["2022-01-03", "2022-01-04", "2022-01-05", "2022-01-07", "2022-01-10", "2022-01-11"])


def _pozicii(sdvig: int = 0) -> pd.DataFrame:
    """2 тикера × 2 стратегии; у (T2, S2) нет первых двух дней (EW по имеющимся парам). sdvig=1 — позиция знает завтра."""
    rng = np.random.default_rng(7)
    rows = []
    for t in ("T1", "T2"):
        lr = rng.normal(0, 0.01, len(DATES))
        for s in ("S1", "S2"):
            pos = np.sign(np.roll(lr, -sdvig)) if sdvig else rng.choice([-1.0, 0.0, 1.0], len(DATES))
            k0 = 2 if (t, s) == ("T2", "S2") else 0
            for i in range(k0, len(DATES)):
                rows.append(dict(date=DATES[i], ticker=t, strategy=s, position=pos[i], daily_gross_return=pos[i] * lr[i], lr=lr[i]))
    return pd.DataFrame(rows)


def _engine(p: pd.DataFrame, kr: np.ndarray):
    """s9.Engine без кэша: только поля, которые читают port и _finish (строки упорядочены по паре, затем по дате)."""
    e = object.__new__(s9.Engine)
    p = p.sort_values(["ticker", "strategy", "date"]).reset_index(drop=True)
    e.dates = DATES
    pair = p.groupby(["ticker", "strategy"], sort=False).ngroup().to_numpy()
    e.r = {"date": DATES.get_indexer(p["date"]), "pair": pair}
    e.first = np.r_[True, pair[1:] != pair[:-1]]
    e.lr_row = p["lr"].to_numpy()
    e.cnt = np.bincount(e.r["date"], minlength=len(DATES)).astype(float)
    e.kr, e.rf, e.cal = kr, kr / 365.0, msp.cal_days(DATES)
    return e, p["position"].to_numpy()


def test_sloy_raven_s9_port():
    """Слой лестницы (msp.portfolio, pairs, дни вперёд) = s9.Engine.port; мутация «дни назад, без выхода Н5» различается."""
    p, kr = _pozicii(), np.array([0.085, 0.085, 0.095, 0.095, 0.095, 0.2])
    rate = pd.Series(kr, index=DATES)
    e, pos = _engine(p, kr)
    a = e.port(pos, comm=SV.COMM)["excess"].to_numpy()
    b = msp.portfolio(p, rate, SV.COMM, agg="pairs")["excess"].to_numpy()
    assert np.max(np.abs(a - b)) < 1e-15
    mut = msp.portfolio(p, SV.s_dnyami(rate, True), SV.COMM, agg="pairs")["excess"].to_numpy()
    assert np.max(np.abs(a - mut)) > 1e-6


def test_mutaciya_sdvig_pozicii_na_den_menyaet_stroku():
    """Позиция, сдвинутая на +1 день (знает доходность дня), обязана менять Шарп слоя."""
    rate = pd.Series(0.1, index=DATES)
    x0 = SV.sharpy(msp.portfolio(_pozicii(), rate, SV.COMM, agg="pairs")["excess"])
    x1 = SV.sharpy(msp.portfolio(_pozicii(sdvig=1), rate, SV.COMM, agg="pairs")["excess"])
    assert abs(x1["ves"] - x0["ves"]) > 0.1


def test_csv_ostanavlivaetsya_na_granice(tmp_path):
    """Строка ≥ 2025 и всё после неё не читаются (битая строка после границы не мешает); даты не по порядку — отказ."""
    f = tmp_path / "m.csv"
    f.write_text("date,total_ret,excess_ret\n2024-12-27,0.001,0.0005\n2024-12-30,0.002,0.001\n2025-01-03,1,1\nбитая,x,y\n")
    d = SV.csv_do_gran(f)
    assert list(d.index) == [pd.Timestamp("2024-12-27"), pd.Timestamp("2024-12-30")] and d.index.max() < SV.GRAN
    f.write_text("date,total_ret,excess_ret\n2024-12-30,0.002,0.001\n2024-12-27,0.001,0.0005\n")
    with pytest.raises(ValueError):
        SV.csv_do_gran(f)


def test_stavka_vosst_izvestnyj_otvet():
    """total − excess = rf·дни (d_prev, d]/365 ⇒ восстановленная ставка равна исходной; первая дата — 1 день."""
    r = np.array([0.085, 0.085, 0.2, 0.2, 0.2, 0.2])
    cal = msp.cal_days(DATES, forward=False)
    a = pd.DataFrame({"total_ret": 0.001 + r * cal / 365, "excess_ret": np.full(6, 0.001)}, index=DATES)
    s = SV.stavka_vosst(a)
    assert np.allclose(s.to_numpy(), r, atol=1e-14) and s.attrs["cal"] == "back"


def test_zapolnit_a_tolko_propushchennye_edinicy():
    """C = A только для единиц без строк C; имеющиеся строки C не меняются (мутация «перезаписать C из A» ловится)."""
    p = _pozicii().assign(test_year=2022)
    pa = p.assign(approach="A")
    pc = p[~((p.ticker == "T1") & (p.strategy == "S1"))].assign(approach="C", position=lambda x: x["position"] * 0.5)
    f, n = SV.zapolnit_a(pc, pa)
    assert n == len(DATES) and len(f) == len(p) and set(f["approach"]) == {"C"}
    est = f.iloc[:len(pc)]
    assert np.array_equal(est["position"].to_numpy(), pc["position"].to_numpy())
    dop = f.iloc[len(pc):]
    assert set(zip(dop.ticker, dop.strategy)) == {("T1", "S1")}


def _st():
    s = lambda v: {"ves": v, 2022: v + 0.1, 2023: v - 0.2, 2024: v * 0.5}        # noqa: E731
    return [("0", "печать", s(2.7), "x"), ("1", "ставка", s(2.69), "x"), ("5а", "позиции", s(-0.5), "x"), ("6", "див", s(-0.52), "x")]


def test_lestnica_zamykaetsya_i_mutacii_lovyatsya():
    """ΣΔ + остаток = итог − строка 0 и Δ = разность Шарпов; подмена Δ шага или остатка — ворота падают."""
    df = SV.tablica("C", _st())
    assert SV.proverit(df) == []
    st = df[df["sharpe"].notna()]
    assert abs(st["delta"].sum() - (st["sharpe"].iloc[-1] - st["sharpe"].iloc[0])) < 1e-12
    for stroka, kol in ((2, "delta"), (1, "delta_2023")):
        m = df.copy()
        m.loc[stroka, kol] += 0.01
        assert SV.proverit(m), f"мутация {kol} в строке {stroka} не поймана"
    m = df.copy()
    m.loc[m["shag"] == "ост", "delta"] = 0.05
    assert SV.proverit(m)


def test_tablica_v_repozitorii_polnaya_i_zamknuta():
    """Выходная таблица: оба объекта, каждый пункт Ж-124 п. 2 (в) + σ̂ и h — шаг или строка «не измеряется»; ворота проходят;
    итог = a_cd_test1.csv (TEST_C −0,5253, D −0,8263)."""
    df = pd.read_csv(SV.TAB / "sverka_predzashchita_CD.csv", comment="#")
    assert SV.proverit(df) == []
    for ob, itog in (("C", -0.525349713666074), ("D", -0.8262801824898581)):
        g = df[df["obekt"] == ob]
        t = " ".join(g["izmenenie"])
        for slovo in ("сетка", "веса", "ФДР", "ставк", "дивиденды", "комиссия", "период калибровки", "остановка торгов",
                      "шорт", "состав", "источник σ̂", "горизонт h"):
            assert slovo in t, (ob, slovo)
        assert abs(g[g["shag"] == "6"]["sharpe"].iloc[0] - itog) < 1e-12
