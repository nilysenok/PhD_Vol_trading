"""V4-NTP (G30–G31): состояния по PIT-доверию NTP3 (известный ответ, группы не по факту), изоляция диагностики, ключ кэша (Д5), реестр (replace_rows)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s11 import v4_c_cache as CC, v4_ntp_stage as ST, v4_ntpd_check as CK, v4_ntpd_states as STS, v4_trials as TR


def mk(n=700, seed=0, informative=True):
    """3 бумаги M16; в группе π > 0,5 гибрид точен (QLIKE 0), ML шумит — HAR-J информативен; в группе π ≤ 0,5 наоборот. informative=False — π не связано с исходом (случайно)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2017-01-02", periods=n + 1)
    fr = []
    for t in ("AFLT", "ALRS", "HYDR"):
        pi = rng.uniform(0, 1, n)
        rv = np.exp(rng.normal(0, 0.3, n))
        noisy = rv * np.exp(rng.normal(0, 0.5, n))
        hy = pi > 0.5
        c8 = np.where(hy, rv, noisy) if informative else noisy
        m = np.where(hy, noisy, rv) if informative else noisy * np.exp(rng.normal(0, 0.01, n))
        fr.append(pd.DataFrame({"ticker": t, "date": days[:-1], "t_plus": days[1:], "group": "M16", "rv_tplus": rv, "k1": c8, "k2": m, "k3": m, "C8": c8, "pi": pi}))
    c = pd.concat(fr, ignore_index=True)
    return {"c": c.drop(columns="pi"), "fc": {"NTP3": c[["ticker", "date", "pi"]]}}


def test_states_known_answer_informative_group_detected_other_not():
    st = STS.states(mk())
    g = st[(st["window"] == "2017_2021") & (st["comparison"] == "C8-M")].set_index("group")
    assert g.loc["pi_gt_half", "delta_mean"] < 0 and bool(g.loc["pi_gt_half", "adds_info"])
    assert g.loc["pi_le_half", "delta_mean"] > 0 and not bool(g.loc["pi_le_half", "adds_info"])
    assert len(st) == 3 * 3 * 2 and abs(g.loc["pi_gt_half", "share_rows"] - 0.5) < 0.05
    assert "добавляет информацию" in STS.answer(st)


def test_states_pit_groups_unrelated_to_outcome_add_nothing():
    st = STS.states(mk(informative=False))
    assert not st[st["group"].isin(["pi_gt_half", "pi_le_half"])]["adds_info"].any()
    assert "не добавляет" in STS.answer(st)


def test_groups_are_taken_from_pit_pi_of_forecast_date_mutation_by_outcome_caught():
    r = mk(informative=False)
    f = STS.frame(r)
    fc = r["fc"]["NTP3"]
    assert np.array_equal(f.sort_values(["ticker", "date"])["pi"].to_numpy(), fc.sort_values(["ticker", "date"])["pi"].to_numpy())
    out = {**r, "fc": {"NTP3": fc.assign(pi=np.where(f.sort_values(["ticker", "date"])["d_c8"].to_numpy() < 0, 0.9, 0.1))}}   # группа «по факту»: подглядывание в исход
    assert STS.states(out).query("group == 'pi_gt_half' and comparison == 'C8-M' and window == '2017_2021'")["adds_info"].iloc[0]


def test_isolation_imports_ntpd_detected_and_repo_clean():
    assert CK.imports_ntpd("from models_v2.s11 import v4_ntpd_states as S") and CK.imports_ntpd("import models_v2.s11.v4_ntpd_report")
    assert not CK.imports_ntpd("from models_v2.s11 import v4_ntp_state as S")
    assert CK.g29_isolation() == []


def test_cache_key_covers_c8_forecast_and_ntp_code(monkeypatch):
    got = {}
    monkeypatch.setattr(CC, "keys", lambda kind, extra_in=None, extra_code=None: got.update(i=extra_in, c=extra_code) or {"key": "x"})
    ST.keys()
    assert CC.OUT / "C8.parquet" in got["i"] and {"v4_ntp_gate.py", "v4_ntp_state.py", "v4_ntp_run.py"} <= {p.name for p in got["c"]}


def test_cache_key_changes_when_input_or_code_changes(tmp_path):
    f1, f2, c = tmp_path / "a.parquet", tmp_path / "b.parquet", tmp_path / "v4_ntp_x.py"
    f1.write_bytes(b"1")
    f2.write_bytes(b"2")
    c.write_text("x=1")
    k = CC.keys("NTP", extra_in=[f1], extra_code=[c])["key"]
    assert CC.keys("NTP", extra_in=[f2], extra_code=[c])["key"] != k
    c.write_text("x=2")
    assert CC.keys("NTP", extra_in=[f1], extra_code=[c])["key"] != k


def test_trials_rows_replace_only_ntp_and_keep_other_stages(tmp_path):
    f = tmp_path / "ALL_TRIALS.csv"
    f.write_text(TR.HEADER + "Ж-222:B1:aaa,Ж-222,x,y,\"{}\",trial\nЖ-222:CS2:bbb,Ж-222,x,y,\"{}\",trial\n")
    TR.replace_rows(["Ж-222:%s:" % c for c in ST.CFGS], ST.trials_rows("k" * 64), path=f)
    TR.replace_rows(["Ж-222:%s:" % c for c in ST.CFGS], ST.trials_rows("k" * 64), path=f)       # повтор — без дублей
    keys = [x.split(",")[0].split(":")[1] for x in f.read_text().splitlines()[1:]]
    assert sorted(keys) == ["B1", "CS2", "NTP1", "NTP2", "NTP3", "NTP4"]
