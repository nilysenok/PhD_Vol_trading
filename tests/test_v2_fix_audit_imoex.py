"""Замечания проверяющего Ж-200, п. 4–5: шапка с sha IMOEX; утечки и инвариантность по барам после d с синтетическим IMOEX.

Мутации (в копии): imoex-признаки видят день d+1 (сдвиг -1 строка) — ловятся; шапка «imoex=нет» при строках IMOEX — ловится.
"""
import itertools

import numpy as np
import pandas as pd
import pytest

from models_v2 import har, prov, run_fix_report
from models_v2.s2 import features as ft
from models_v2.s3 import resid
from models_v2.s4 import design, fit, mkt, runner, synth

HP = fit.HP(3, 10.0, 2.0, 0.1, 0.8)
D = 560
IMX = prov.ROOT / "data/iss_check/index_10m/IMOEX.parquet"


# ---- п. 4: шапка ----

def test_restamp_changes_only_header_line(tmp_path):
    p = tmp_path / "t.csv"
    body = "a,b\n1,2\n3,4\n"
    p.write_text("# x; imoex=нет оставшееся\n" + body, encoding="utf-8")
    run_fix_report.restamp(p, "cadc3e41cda11533")
    head, _, rest = p.read_text(encoding="utf-8").partition("\n")
    assert head == "# x; imoex=cadc3e41cda11533 оставшееся" and rest == body


def test_restamp_refuses_file_without_marker(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("# x; imoex=cadc3e41cda11533\na\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        run_fix_report.restamp(p, "ffff")


def test_report_tables_with_imoex_rows_carry_imoex_sha():
    """Все таблицы отчёта по строкам с IMOEX несут sha входа, а не «imoex=нет»."""
    if not (run_fix_report.OUT / "fix_forecasts_c.parquet").exists():
        pytest.skip("этап c не выполнен")
    sha = prov.sha_file(IMX)[:16]
    assert sha.startswith("cadc3e41") and run_fix_report.imoex_sha() == sha
    for n in ("fix_criterion.csv", "fix_metrics.csv", "fix_dm.csv", "fix_i6_choice.csv", "fix_i2_choice_hard.csv", "fix_leak_c.csv"):
        head = (run_fix_report.TAB / n).open(encoding="utf-8").readline()
        assert "imoex=" + sha in head and "imoex=нет" not in head, n


# ---- п. 5: IMOEX в утечках ----

def world(bars_a, bars_b, bars_i):
    F = ft.build(bars_a)
    M, _ = mkt.block(None, F.X[:, [0, 1, 2]], F.seg, F.dates, [mkt.paper(bars_a), mkt.paper(bars_b)], bars_i)
    lb = resid.harj_ab(har.Panel(F.dates, F.seg, F.rv, F.X[:, :4]))[1]
    return F, M, lb


@pytest.fixture(scope="module")
def bars():
    return synth.random_bars(700, 11), synth.random_bars(700, 12), synth.random_bars(700, 13, vol=0.001)


def predict(F, M, lb, form, w, d):
    dz = design.make(F, form, design.columns(M, "imoex"), lb if form == "resid" else None)
    return runner.block_fit(F, dz, HP, d, [d], w, None, "work")["f"]


def damaged(bars, d):
    return world(*[synth.damage_after(b, d) for b in bars])


@pytest.mark.parametrize("form,w", list(itertools.product(design.FORMS, (False, True))))
def test_forecast_with_imoex_bitwise_invariant_to_bars_after_d(bars, form, w):
    base = world(*bars)
    dmg = damaged(bars, pd.Timestamp(base[0].dates[D]))
    assert "imoex_d" in base[1].columns and base[1]["imoex_d"].iloc[D] == base[1]["imoex_d"].iloc[D]
    assert predict(*base, form, w, D).tobytes() == predict(*dmg, form, w, D).tobytes()


def test_damaging_only_imoex_bars_does_not_change_forecast(bars):
    a, b, i = bars
    base = world(a, b, i)
    dmg = world(a, b, synth.damage_after(i, pd.Timestamp(base[0].dates[D])))
    assert predict(*base, "level", False, D).tobytes() == predict(*dmg, "level", False, D).tobytes()


def forced(F, M, d):
    """Дизайн «level», где цель = imoex_d строки t: модель обязана использовать столбец IMOEX."""
    dz = design.make(F, "level", design.columns(M, "imoex"))
    j = ft.N_BASE + list(design.columns(M, "imoex").columns).index("imoex_d")
    dz.y = np.nan_to_num(dz.X[:, j], nan=0.0)
    return runner.block_fit(F, dz, HP, d, [d], False, None, "work")["f"]


def test_imoex_columns_are_not_vacuous(bars):
    """Контроль: когда цель зависит от imoex_d, порча баров IMOEX ДО d меняет прогноз, а после d — нет."""
    a, b, i = bars
    base = world(a, b, i)
    ref = forced(base[0], base[1], D)
    before = world(a, b, synth.damage_after(i, pd.Timestamp(base[0].dates[D - 30])))
    after = world(a, b, synth.damage_after(i, pd.Timestamp(base[0].dates[D])))
    assert forced(before[0], before[1], D).tobytes() != ref.tobytes()
    assert forced(after[0], after[1], D).tobytes() == ref.tobytes()


def test_mutation_imoex_features_plus_one_day_caught(bars):
    base = world(*bars)
    dmg = damaged(bars, pd.Timestamp(base[0].dates[D]))
    for _, M, _ in (base, dmg):
        for c in mkt.IDX:
            M[c] = M[c].shift(-1)                      # признаки IMOEX строки d видят d+1
    assert predict(*base, "level", False, D).tobytes() != predict(*dmg, "level", False, D).tobytes()


def test_imoex_rows_invariant_and_future_imoex_caught():
    n, d = 60, 40
    dts, R = synth.days(n), synth.random_returns(n, 3)
    ri = synth.random_returns(n, 4)
    rs = [synth.from_returns(dts, synth.random_returns(n, s)) for s in (1, 2)]
    ib = synth.from_returns(dts, ri)
    sb = np.log(mkt.paper(rs[0])["day"][["rv", "rv_w", "rv_m"]].reindex(dts).to_numpy(float))

    def rows_upto(shift=0):
        out = []
        for ibars in (ib, synth.damage_after(ib, dts[d])):
            M, _ = mkt.block(None, sb, np.zeros(n, int), dts, [mkt.paper(x) for x in rs], ibars)
            cols = M[list(mkt.IDX)].shift(-shift)
            out.append(cols.iloc[:d + 1].to_numpy().tobytes())
        return out[0] == out[1]

    assert rows_upto(0) and not rows_upto(1)
