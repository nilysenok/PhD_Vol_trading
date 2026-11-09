"""E3: порядок записи (`E3.eval` раньше таблиц с p), генератор отчёта, строка реестра, `--check` только читает и ловит подмену; g_s — сохранённая A0g (sha256), не переобучается (синтетика, временный каталог)."""
import json

import torch  # noqa: F401  — первым (OpenMP)
import numpy as np
import pandas as pd
import pytest

from models_v2.s15 import de0_io as IO, de_e3_doc as DOC, de_e3_eval as EV, de_e3_stage as S3, de_reg as R


def _synth():
    idx = pd.bdate_range("2018-01-03", "2026-02-26")
    n = len(idx)
    rng = np.random.default_rng(5)
    rv = np.exp(-1.0 + 0.4 * rng.standard_normal(n))
    ser = {k: rv * np.exp(sg * rng.standard_normal(n)) for k, sg in (("K1", .5), ("K2", .35), ("K3", .33), ("V4:NTP2", .3), ("V6:A2", .3), ("V6:A0", .31), ("V6:A0g", .3), ("cfg8", .28), ("X1", .28), ("X3", .27))}
    ser["M"], ser["K8"] = np.sqrt(ser["K2"] * ser["K3"]), np.sqrt(ser["K1"] * np.sqrt(ser["K2"] * ser["K3"]))
    tp = idx[1:].append(pd.DatetimeIndex([idx[-1] + pd.offsets.BDay(1)]))
    ga = 0.1 * rng.standard_normal(n)
    x3 = pd.DataFrame(dict(d=idx, t_plus=tp, rv=rv, H=ser["K1"], F0=ser["V6:A0g"], gs=np.log(ser["V6:A0g"] / ser["K1"]), ga=ga, a=rng.random(n), lam=np.where(rng.random(n) < .5, 1.0, 10.0), n_rows=250, F=ser["X3"]))[S3.COLS]
    t = EV.main_tables(rv, ser, tp)
    st, at = EV.state_qlike(rv, ser, tp, (rng.random(n) < .1).astype(int)), EV.a_table(x3)
    return x3, t, st, at, dict(n_dates=n, first="2018-01-03", last="2026-02-26", input_sha256={}, configs_spent=3, win=250, hold=126, a0g_sha256="abc")


@pytest.fixture(scope="module")
def synth():
    return _synth()


@pytest.fixture()
def env(tmp_path, monkeypatch, synth):
    monkeypatch.setattr(IO, "REP", tmp_path)
    monkeypatch.setattr(IO, "CP_FILE", tmp_path / "CHECKPOINT.json")
    monkeypatch.setattr(R, "TRIALS", tmp_path / "ALL_TRIALS.csv")
    monkeypatch.setattr(S3, "OUT", tmp_path / "x")
    monkeypatch.setattr(S3, "X3_FILE", tmp_path / "x" / "x3.parquet")
    monkeypatch.setattr(S3, "REPORT", tmp_path / "E3_REPORT.md")
    monkeypatch.setattr(S3, "compute", lambda inp=None, expected_sha=None: synth)
    R.register("01.01.2020 10:00:00")
    return tmp_path


def test_eval_key_written_before_tables_with_p_and_result_row_added(env, monkeypatch):
    calls = []
    real_cp = IO.merge_checkpoint
    monkeypatch.setattr(IO, "merge_checkpoint", lambda k, v: (calls.append(("cp", k, v)), real_cp(k, v)))
    monkeypatch.setattr(S3, "_emit_csv", lambda name, df: (calls.append(("csv", name)), (IO.REP / (name + ".csv")).write_text("x")))
    S3.write()
    names = [c[1] for c in calls]
    assert names.index("E3.x3") < names.index("E3.eval") < names.index("E3_PAIRS") < names.index("E3.rules")
    ev = [c for c in calls if c[:2] == ("cp", "E3.eval")][0][2]
    assert "p_one" not in json.dumps(ev) and "p_holm" not in json.dumps(ev)
    reg = R.read()
    assert list(reg["id"]).count("X3") == 2 and reg.iloc[-1]["status"] == "посчитана" and "6.2 против A0g" in reg.iloc[-1]["formula"]
    R.guard("X3")
    S3.add_result_row(S3.EV.own_base(S3.compute()[1]["PAIRS"], "V6:A0g"), False)
    assert list(R.read()["id"]).count("X3") == 2
    assert list(pd.read_parquet(S3.X3_FILE).columns) == S3.COLS


def test_check_reads_only_and_catches_substituted_number_table_and_order(env):
    S3.write()
    before = S3.outputs_hash()
    assert S3.check() == [] and S3.outputs_hash() == before
    rep = S3.REPORT.read_text(encoding="utf-8")
    num = [w for w in rep.split() if w.replace(".", "").isdigit()][0]
    S3.REPORT.write_text(rep.replace(num, num + "9", 1), encoding="utf-8")
    assert any("E3_REPORT.md" in b for b in S3.check())
    S3.REPORT.write_text(rep, encoding="utf-8")
    p = IO.REP / "E3_A_YEARS.csv"
    p.write_text(p.read_text(encoding="utf-8").replace(",", ";", 1), encoding="utf-8")
    assert any("E3_A_YEARS" in b for b in S3.check())
    S3.write()
    cp = IO.read_cp()
    cp["E3.eval"]["written_at"] = "31.12.2099 00:00:00"
    IO.CP_FILE.write_text(json.dumps(cp), encoding="utf-8")
    assert any("порядок" in b for b in S3.check())


def test_x3_is_registered_before_forecast_in_check(env):
    S3.write()
    R.TRIALS.write_text(R.TRIALS.read_text(encoding="utf-8").replace("01.01.2020 10:00:00", "31.12.2099 00:00:00"), encoding="utf-8")
    assert any("позже прогноза" in b for b in S3.check())


def test_unregistered_x3_is_refused(env):
    R.TRIALS.write_text("# x\n" + ",".join(R.COLS) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        S3.write()


def test_report_first_line_comes_from_tables_and_substitution_changes_text(synth):
    x3, t, st, at, num = synth
    sb = EV.own_base(t["PAIRS"], "V6:A0g")
    txt = DOC.build(t, sb, st, at, num)
    assert txt.splitlines()[0].startswith("**X3 (устойчивая + адаптивная поправка), 2018–2021, правило 6.2 против A0g: %s**" % sb["rule62"])
    t2 = {k: v.copy() for k, v in t.items()}
    t2["GO"].loc[0, "value"] += 0.5
    assert DOC.build(t2, sb, st, at, num) != txt


def test_gs_is_saved_a0g_sha_checked_and_not_refit(tmp_path, monkeypatch):
    d = pd.bdate_range("2016-12-30", periods=50)
    H = np.exp(np.random.default_rng(2).standard_normal(50) * 0.1)
    F = H * np.exp(np.linspace(-0.2, 0.2, 50))
    f = tmp_path / "A0g.parquet"
    pd.DataFrame(dict(d=d, H=H, F=F)).to_parquet(f, index=False)
    monkeypatch.setattr(S3, "A0G_FILE", f)
    F0, sha = S3.load_a0g(d, H)
    assert np.array_equal(F0, F) and sha == IO.sha_file(f)                            # g_s = ln(F/H) берётся из файла, ничего не обучается
    S3.load_a0g(d, H, expected_sha=sha)
    pd.DataFrame(dict(d=d, H=H, F=F * 1.001)).to_parquet(f, index=False)               # файл изменён после записи хэша
    with pytest.raises(SystemExit, match="sha256"):
        S3.load_a0g(d, H, expected_sha=sha)
    with pytest.raises(SystemExit, match="даты"):
        S3.load_a0g(d, H * 1.01)
