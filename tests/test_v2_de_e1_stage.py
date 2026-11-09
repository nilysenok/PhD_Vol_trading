"""E1: порядок записи (`E1.eval` раньше таблиц с p), генератор отчёта, `--check` только читает и ловит подмену (синтетика, временный каталог)."""
import json

import torch  # noqa: F401  — первым (OpenMP)
import numpy as np
import pandas as pd
import pytest

from models_v2.s15 import de0_io as IO, de_e1_doc as DOC, de_e1_eval as EV, de_e1_stage as S1, de_e1_state as S, de_reg as R


def _synth():
    idx = pd.bdate_range("2018-01-03", "2026-02-26")
    n = len(idx)
    rng = np.random.default_rng(5)
    rv = np.exp(-1.0 + 0.4 * rng.standard_normal(n))
    ser = {k: rv * np.exp(sg * rng.standard_normal(n)) for k, sg in (("K1", .5), ("K2", .35), ("K3", .33), ("V4:NTP2", .3), ("V6:A2", .3), ("V6:A0", .31), ("V6:A0g", .3), ("cfg8", .28))}
    ser["M"], ser["K8"] = np.sqrt(ser["K2"] * ser["K3"]), np.sqrt(ser["K1"] * np.sqrt(ser["K2"] * ser["K3"]))
    tp = idx[1:].append(pd.DatetimeIndex([idx[-1] + pd.offsets.BDay(1)]))
    s = (rng.random(n) < 0.15).astype(int)
    r = S.run(rv, ser["M"], ser["cfg8"], s, idx.to_numpy(), tp.to_numpy())
    x1 = pd.DataFrame(dict(d=idx, t_plus=tp, rv=rv, M=ser["M"], C=ser["cfg8"], s=s, pi=r["pi"], F=r["F"], pi0=r["pis"][:, 0], pi1=r["pis"][:, 1], a0=r["a0"], a1=r["a1"]))[S1.COLS]
    ser["X1"] = r["F"]
    t = EV.main_tables(rv, ser, tp)
    num = dict(n_dates=n, first="2018-01-03", last="2026-02-26", n_extreme_days=int(s.sum()), n_no_bar_days=0, input_sha256={}, configs_spent=1)
    return x1, t, EV.state_qlike(rv, ser, tp, s), EV.pi_table(x1), num


@pytest.fixture(scope="module")
def synth():
    return _synth()


@pytest.fixture()
def env(tmp_path, monkeypatch, synth):
    monkeypatch.setattr(IO, "REP", tmp_path)
    monkeypatch.setattr(IO, "CP_FILE", tmp_path / "CHECKPOINT.json")
    monkeypatch.setattr(R, "TRIALS", tmp_path / "ALL_TRIALS.csv")
    monkeypatch.setattr(S1, "OUT", tmp_path / "x")
    monkeypatch.setattr(S1, "X1_FILE", tmp_path / "x" / "x1.parquet")
    monkeypatch.setattr(S1, "REPORT", tmp_path / "E1_REPORT.md")
    monkeypatch.setattr(S1, "compute", lambda inp=None: synth)
    R.register("01.01.2020 10:00:00")
    return tmp_path


def test_holm_uses_738_and_3181(synth):
    t = synth[1]["PAIRS"]
    k = t["window"] == "2018_2021"
    from models_v2.s13 import v6_1_eval as EVX
    assert np.allclose(t.loc[k, "p_holm_N"], EVX.holm_both(t.loc[k, "p_one"].to_numpy(), {"N": 738, "N_wide": 3181})[0])
    assert (EV.N_HOLM, EV.N_WIDE) == (738, 3181)


def test_eval_key_written_before_tables_with_p(env, monkeypatch):
    calls = []
    monkeypatch.setattr(S1, "_cp", lambda k, v: (calls.append(("cp", k, v)), IO.merge_checkpoint(k, v)))
    monkeypatch.setattr(S1, "_emit_csv", lambda name, df: (calls.append(("csv", name)), (IO.REP / (name + ".csv")).write_text("x")))
    S1.write()
    names = [c[1] for c in calls]
    assert names.index("E1.x1") < names.index("E1.eval") < names.index("E1_PAIRS") < names.index("E1.rules")
    ev = [c for c in calls if c[:2] == ("cp", "E1.eval")][0][2]
    assert "p_one" not in json.dumps(ev) and "p_holm" not in json.dumps(ev)
    assert (S1.X1_FILE.exists()) and list(pd.read_parquet(S1.X1_FILE).columns) == S1.COLS


def test_check_reads_only_and_catches_substituted_number_table_and_order(env):
    S1.write()
    before = S1.outputs_hash()
    assert S1.check() == [] and S1.outputs_hash() == before                    # --check ничего не пишет
    rep = S1.REPORT.read_text(encoding="utf-8")
    num = [w for w in rep.split() if w.replace(".", "").isdigit()][0]
    S1.REPORT.write_text(rep.replace(num, num + "9", 1), encoding="utf-8")        # число в отчёте подменено
    assert any("E1_REPORT.md" in b for b in S1.check())
    S1.REPORT.write_text(rep, encoding="utf-8")
    p = IO.REP / "E1_TESTS.csv"
    p.write_text(p.read_text(encoding="utf-8").replace(",", ";", 1), encoding="utf-8")       # таблица подменена
    assert any("E1_TESTS" in b for b in S1.check())
    S1.write()
    cp = IO.read_cp()
    cp["E1.eval"]["written_at"] = "31.12.2099 00:00:00"                           # E1.eval «после» таблиц с p
    IO.CP_FILE.write_text(json.dumps(cp), encoding="utf-8")
    assert any("порядок" in b for b in S1.check())


def test_report_first_line_comes_from_table_and_substitution_changes_text(synth):
    x1, t, st, pit, num = synth
    sb = EV.state_benefit(t["PAIRS"])
    txt = DOC.build(t, sb, st, pit, num)
    assert txt.splitlines()[0].startswith("**Польза состояния (X1 против NTP2, 2018–2021, правило 6.2): %s**" % sb["rule62"])
    t2 = {k: v.copy() for k, v in t.items()}
    t2["GO"].loc[0, "value"] += 0.5
    assert DOC.build(t2, sb, st, pit, num) != txt


def test_x1_is_registered_before_forecast_in_check(env):
    S1.write()
    ck = IO.read_cp()["E1.x1"]
    R.guard("X1", cp=IO.read_cp(), computed_at=ck["computed_at"])
    R.TRIALS.write_text(R.TRIALS.read_text(encoding="utf-8").replace("01.01.2020 10:00:00", "31.12.2099 00:00:00"), encoding="utf-8")     # запись реестра «после» прогноза
    assert any("позже прогноза" in b for b in S1.check())
