"""E4: порядок записи (`E4.eval` раньше таблиц с p), генератор отчёта, строка реестра, `--check` только читает и ловит подмену, реестр раньше прогноза (синтетика, временный каталог)."""
import json

import torch  # noqa: F401  — первым (OpenMP)
import numpy as np
import pandas as pd
import pytest

from models_v2.s15 import de0_io as IO, de_e4_doc as DOC, de_e4_eval as EV, de_e4_stage as S4, de_reg as R


def _synth():
    idx = pd.bdate_range("2018-01-03", "2026-02-26")
    n = len(idx)
    rng = np.random.default_rng(7)
    rv = np.exp(-1.0 + 0.4 * rng.standard_normal(n))
    ser = {k: rv * np.exp(sg * rng.standard_normal(n)) for k, sg in (("K1", .5), ("K2", .35), ("K3", .33), ("V4:NTP2", .3), ("V6:A2", .3), ("V6:A0", .31), ("V6:A0g", .3), ("cfg8", .28), ("X1", .28), ("ENS", .3), ("X4", .29))}
    ser["M"], ser["K8"] = np.sqrt(ser["K2"] * ser["K3"]), np.sqrt(ser["K1"] * np.sqrt(ser["K2"] * ser["K3"]))
    tp = idx[1:].append(pd.DatetimeIndex([idx[-1] + pd.offsets.BDay(1)]))
    x4 = pd.DataFrame(dict(d=idx, t_plus=tp, rv=rv, H=ser["K1"], **{k: 0.05 * rng.standard_normal(n) for k in S4.GN}, gbar=0.1 * rng.standard_normal(n), sd=rng.random(n) * 0.1, sbar=0.05, act=rng.random(n) < .8, nval=126, kappa=rng.choice([.5, 1., 2.], n), alpha=0.2 + 0.8 * rng.random(n), F=ser["X4"], F_ens=ser["ENS"]))
    t = EV.main_tables(rv, ser, tp)
    st, at = EV.state_qlike(rv, ser, tp, (rng.random(n) < .1).astype(int)), EV.alpha_table(x4)
    return x4, t, st, at, dict(n_dates=n, first="2018-01-03", last="2026-02-26", input_sha256={"a0g": "abc"}, configs_spent=4, windows=[None, 750, 1000, 1250, 1500], kappas=[0.5, 1.0, 2.0], val=126)


@pytest.fixture(scope="module")
def synth():
    return _synth()


@pytest.fixture()
def env(tmp_path, monkeypatch, synth):
    monkeypatch.setattr(IO, "REP", tmp_path)
    monkeypatch.setattr(IO, "CP_FILE", tmp_path / "CHECKPOINT.json")
    monkeypatch.setattr(R, "TRIALS", tmp_path / "ALL_TRIALS.csv")
    monkeypatch.setattr(S4, "OUT", tmp_path / "x")
    monkeypatch.setattr(S4, "X4_FILE", tmp_path / "x" / "x4.parquet")
    monkeypatch.setattr(S4, "REPORT", tmp_path / "E4_REPORT.md")
    monkeypatch.setattr(S4, "compute", lambda inp=None, fit=None: synth)
    R.register("01.01.2020 10:00:00")
    return tmp_path


def test_eval_key_before_tables_with_p_and_result_row_added(env, monkeypatch):
    calls = []
    real_cp = IO.merge_checkpoint
    monkeypatch.setattr(IO, "merge_checkpoint", lambda k, v: (calls.append(("cp", k, v)), real_cp(k, v)))
    monkeypatch.setattr(S4, "_emit_csv", lambda name, df: (calls.append(("csv", name)), (IO.REP / (name + ".csv")).write_text("x")))
    S4.write()
    names = [c[1] for c in calls]
    assert names.index("E4.x4") < names.index("E4.eval") < names.index("E4_PAIRS") < names.index("E4.rules")
    ev = [c for c in calls if c[:2] == ("cp", "E4.eval")][0][2]
    assert "p_one" not in json.dumps(ev) and "p_holm" not in json.dumps(ev)
    reg = R.read()
    assert list(reg["id"]).count("X4") == 2 and reg.iloc[-1]["status"] == "посчитана" and "6.2 против ENS" in reg.iloc[-1]["formula"]
    assert list(pd.read_parquet(S4.X4_FILE).columns) == S4.COLS


def test_check_reads_only_and_catches_substitution_table_order(env):
    S4.write()
    before = S4.outputs_hash()
    assert S4.check() == [] and S4.outputs_hash() == before
    rep = S4.REPORT.read_text(encoding="utf-8")
    num = [w for w in rep.split() if w.replace(".", "").isdigit()][0]
    S4.REPORT.write_text(rep.replace(num, num + "9", 1), encoding="utf-8")
    assert any("E4_REPORT.md" in b for b in S4.check())
    S4.REPORT.write_text(rep, encoding="utf-8")
    p = IO.REP / "E4_ALPHA_YEARS.csv"
    p.write_text(p.read_text(encoding="utf-8").replace(",", ";", 1), encoding="utf-8")
    assert any("E4_ALPHA_YEARS" in b for b in S4.check())
    S4.write()
    cp = IO.read_cp()
    cp["E4.eval"]["written_at"] = "31.12.2099 00:00:00"
    IO.CP_FILE.write_text(json.dumps(cp), encoding="utf-8")
    assert any("порядок" in b for b in S4.check())


def test_x4_registered_before_forecast_and_unregistered_refused(env):
    S4.write()
    R.TRIALS.write_text(R.TRIALS.read_text(encoding="utf-8").replace("01.01.2020 10:00:00", "31.12.2099 00:00:00"), encoding="utf-8")
    assert any("позже прогноза" in b for b in S4.check())
    R.TRIALS.write_text("# x\n" + ",".join(R.COLS) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        S4.write()


def test_report_first_line_from_tables_and_substitution_changes_text(synth):
    x4, t, st, at, num = synth
    se, sa = EV.own_base(t["PAIRS"], "ENS"), EV.own_base(t["PAIRS"], "V6:A0g")
    txt = DOC.build(t, se, sa, st, at, num)
    assert txt.splitlines()[0].startswith("**X4 (затвор по неопределённости), 2018–2021, правило 6.2** — против ансамбля без затвора: **%s**" % se["rule62"])
    t2 = {k: v.copy() for k, v in t.items()}
    t2["GO"].loc[0, "value"] += 0.5
    assert DOC.build(t2, se, sa, st, at, num) != txt
    assert list(at.columns).count("mean_alpha") == 1 and ((x4["alpha"] > 0) & (x4["alpha"] <= 1)).all()
