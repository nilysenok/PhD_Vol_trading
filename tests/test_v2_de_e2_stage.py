"""E2: порядок записи (`E2.eval` раньше таблиц с p), генератор отчёта, калибровка p, строка реестра, `--check` только читает и ловит подмену (синтетика, временный каталог)."""
import json

import torch  # noqa: F401  — первым (OpenMP)
import numpy as np
import pandas as pd
import pytest

from models_v2.s15 import de0_io as IO, de_e2_doc as DOC, de_e2_eval as EV, de_e2_stage as S2, de_reg as R


def _synth():
    idx = pd.bdate_range("2018-01-03", "2026-02-26")
    n = len(idx)
    rng = np.random.default_rng(5)
    rv = np.exp(-1.0 + 0.4 * rng.standard_normal(n))
    ser = {k: rv * np.exp(sg * rng.standard_normal(n)) for k, sg in (("K1", .5), ("K2", .35), ("K3", .33), ("V4:NTP2", .3), ("V6:A2", .3), ("V6:A0", .31), ("V6:A0g", .3), ("cfg8", .28), ("cfg4", .29), ("X1", .28), ("X2", .27))}
    ser["M"], ser["K8"] = np.sqrt(ser["K2"] * ser["K3"]), np.sqrt(ser["K1"] * np.sqrt(ser["K2"] * ser["K3"]))
    tp = idx[1:].append(pd.DatetimeIndex([idx[-1] + pd.offsets.BDay(1)]))
    p, thr = rng.random(n), np.full(n, np.quantile(rv, 0.9))
    x2 = pd.DataFrame(dict(d=idx, t_plus=tp, rv=rv, H=ser["K1"], e=(rng.random(n) < .1).astype(int), rs_neg_share=rng.random(n), p=p, cn=np.exp(rng.standard_normal(n) * .1), cs=np.exp(rng.standard_normal(n) * .1), F=ser["X2"], beta=1.0, thr=thr))[S2.COLS]
    t = EV.main_tables(rv, ser, tp)
    st, cal, bt = EV.state_qlike(rv, ser, tp, x2["e"].to_numpy()), EV.calibration(tp, x2["p"], rv, x2["thr"]), EV.beta_table(x2)
    return x2, t, st, cal, bt, dict(n_dates=n, first="2018-01-03", last="2026-02-26", input_sha256={}, configs_spent=2)


@pytest.fixture(scope="module")
def synth():
    return _synth()


@pytest.fixture()
def env(tmp_path, monkeypatch, synth):
    monkeypatch.setattr(IO, "REP", tmp_path)
    monkeypatch.setattr(IO, "CP_FILE", tmp_path / "CHECKPOINT.json")
    monkeypatch.setattr(R, "TRIALS", tmp_path / "ALL_TRIALS.csv")
    monkeypatch.setattr(S2, "OUT", tmp_path / "x")
    monkeypatch.setattr(S2, "X2_FILE", tmp_path / "x" / "x2.parquet")
    monkeypatch.setattr(S2, "REPORT", tmp_path / "E2_REPORT.md")
    monkeypatch.setattr(S2, "compute", lambda inp=None: synth)
    R.register("01.01.2020 10:00:00")
    return tmp_path


def test_eval_key_written_before_tables_with_p_and_result_row_added(env, monkeypatch):
    calls = []
    real_cp = IO.merge_checkpoint
    monkeypatch.setattr(IO, "merge_checkpoint", lambda k, v: (calls.append(("cp", k, v)), real_cp(k, v)))
    monkeypatch.setattr(S2, "_emit_csv", lambda name, df: (calls.append(("csv", name)), (IO.REP / (name + ".csv")).write_text("x")))
    S2.write()
    names = [c[1] for c in calls]
    assert names.index("E2.x2") < names.index("E2.eval") < names.index("E2_PAIRS") < names.index("E2.rules")
    ev = [c for c in calls if c[:2] == ("cp", "E2.eval")][0][2]
    assert "p_one" not in json.dumps(ev) and "p_holm" not in json.dumps(ev)
    reg = R.read()
    assert list(reg["id"]).count("X2") == 2 and reg.iloc[-1]["status"] == "посчитана" and "6.2 против A0g" in reg.iloc[-1]["formula"]
    R.guard("X2")                                                                      # первая строка X2 по-прежнему «зарегистрирована»
    S2.add_result_row(S2.EV.own_base(S2.compute()[1]["PAIRS"], "V6:A0g"), False)
    assert list(R.read()["id"]).count("X2") == 2                                       # повторно не дописывается
    assert list(pd.read_parquet(S2.X2_FILE).columns) == S2.COLS


def test_check_reads_only_and_catches_substituted_number_table_and_order(env):
    S2.write()
    before = S2.outputs_hash()
    assert S2.check() == [] and S2.outputs_hash() == before
    rep = S2.REPORT.read_text(encoding="utf-8")
    num = [w for w in rep.split() if w.replace(".", "").isdigit()][0]
    S2.REPORT.write_text(rep.replace(num, num + "9", 1), encoding="utf-8")
    assert any("E2_REPORT.md" in b for b in S2.check())
    S2.REPORT.write_text(rep, encoding="utf-8")
    p = IO.REP / "E2_CALIB.csv"
    p.write_text(p.read_text(encoding="utf-8").replace(",", ";", 1), encoding="utf-8")
    assert any("E2_CALIB" in b for b in S2.check())
    S2.write()
    cp = IO.read_cp()
    cp["E2.eval"]["written_at"] = "31.12.2099 00:00:00"
    IO.CP_FILE.write_text(json.dumps(cp), encoding="utf-8")
    assert any("порядок" in b for b in S2.check())


def test_x2_is_registered_before_forecast_in_check(env):
    S2.write()
    R.TRIALS.write_text(R.TRIALS.read_text(encoding="utf-8").replace("01.01.2020 10:00:00", "31.12.2099 00:00:00"), encoding="utf-8")
    assert any("позже прогноза" in b for b in S2.check())


def test_close_equal_infinities_and_catches_difference():
    inf = float("inf")
    assert S2._close({"5": [inf, False], "x": float("nan")}, {"5": [inf, False], "x": float("nan")})
    assert not S2._close([inf], [1.0]) and not S2._close([1.0], [1.0 + 1e-6]) and not S2._close([inf], [-inf])


def test_unregistered_x2_is_refused(env):
    R.TRIALS.write_text("# x\n" + ",".join(R.COLS) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        S2.write()


def test_report_first_line_comes_from_tables_and_substitution_changes_text(synth):
    x2, t, st, cal, bt, num = synth
    a, c = EV.own_base(t["PAIRS"], "V6:A0g"), EV.own_base(t["PAIRS"], "cfg4")
    txt = DOC.build(t, a, c, st, cal, bt, num)
    assert txt.splitlines()[0].startswith("**X2 (хвостовая смесь), 2018–2021, правило 6.2.** против A0g: **%s**" % a["rule62"])
    t2 = {k: v.copy() for k, v in t.items()}
    t2["GO"].loc[0, "value"] += 0.5
    assert DOC.build(t2, a, c, st, cal, bt, num) != txt


def test_calibration_known_answer_and_anticalibration_is_caught():
    idx = pd.bdate_range("2018-01-03", "2025-12-31")
    n = len(idx)
    rng = np.random.default_rng(9)
    p = rng.random(n)
    rv = (rng.random(n) < p).astype(float)
    thr = np.full(n, 0.5)
    ok = EV.calibration(idx, p, rv, thr)
    bad = EV.calibration(idx, 1 - p, rv, thr)
    e_ok, e_bad = (c[(c["window"] == "all") & (c["decile"] == "итого")]["abs_gap"].iloc[0] for c in (ok, bad))
    assert e_ok < 0.05 < 0.3 < e_bad
    d10 = ok[(ok["window"] == "all") & (ok["decile"] == "10")].iloc[0]
    assert d10["mean_p"] > 0.9 and d10["freq"] > 0.85 and ok[ok["window"] == "all"]["n"].iloc[:-1].sum() == n
