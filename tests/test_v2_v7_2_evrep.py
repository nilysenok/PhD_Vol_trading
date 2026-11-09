"""V7 лист 2b, G28–G30: отчёт `LEAF2_REPORT.md` целиком из таблиц; CHECKPOINT `2.eval` — числа без p-значений и записан ДО таблиц с p; `check_eval` сверяет с файлами и ничего не пишет.
Мутации: число в отчёте подменено; число вписано вручную (не следует за таблицей); p-значения в `2.eval`; `2.eval` после таблиц; файл таблицы или отчёта изменён. Синтетика, файлы SBER не читаются."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from models_v2.s14 import v7_0_io as IO, v7_2_ev as E2, v7_2_evrep as REP, v7_2_evrun as R, v7_2_evspec as SP, v7_2_run as R2

IDX = pd.bdate_range("2018-01-03", "2026-02-26")
N = len(IDX)


@pytest.fixture(scope="module")
def world():
    g = np.random.default_rng(1)
    rv = np.exp(g.normal(0, 0.5, N))
    s = {k: rv * np.exp(g.normal(0, 0.2, N)) for k in E2.CTRL + E2.CFGS}
    v1 = {k: rv * np.exp(g.normal(0, 0.25, N)) for k in E2.CFGS}
    frames = {}
    for c in SP.MOE:
        pi = g.dirichlet([8, 8, 8], N)
        g1, g2 = g.normal(0, 0.15, N), g.normal(0, 0.15, N)
        H = s["K1"]
        G = pi[:, 1] * g1 + pi[:, 2] * g2
        frames[c] = pd.DataFrame({"d": IDX, "t_plus": IDX, "rv": rv, "H": H, "G": G, "F": H * np.exp(G), "fitted": 1.0, "pi0": pi[:, 0], "pi1": pi[:, 1], "pi2": pi[:, 2], "g1": g1, "g2": g2})
    inp = dict(idx=IDX, rv=rv, tp=IDX.to_numpy("datetime64[ns]"), series=s, v1=v1, frames=frames, shas={"V6:A0": "x"})
    t = R.compute(inp)
    cp2 = dict(rules=R.rules(t), cond9=SP.cond9(frames["cfg6"], t["PAIRS"]))
    return inp, t, cp2


def test_render_is_deterministic_and_contains_table_numbers(world):
    inp, t, cp2 = world
    a, b = REP.render(t, inp, cp2), REP.render(t, inp, cp2)
    assert a == b and a.count("\n") > 200
    m = t["METRICS"]
    assert REP.f(m[(m["series"] == "cfg6") & (m["window"] == "2018_2021")]["qlike"].iloc[0], 6) in a
    assert "потолок" in a and "Холм N=733" in a and "задним числом" in a and "версия 1" in a


def test_mutation_number_substituted_in_report_is_caught(world):
    inp, t, cp2 = world
    good = REP.render(t, inp, cp2)
    v = REP.f(t["METRICS"].query("series == 'K2' and window == '2018_2021'")["qlike"].iloc[0], 6)
    assert v in good and good.replace(v, v[:-1] + ("1" if v[-1] != "1" else "2"), 1) != good


def test_mutation_number_typed_by_hand_does_not_follow_table(world):
    """Любая ячейка таблицы, попавшая в отчёт, меняет текст: чисел «вписанных вручную» нет."""
    inp, t, cp2 = world
    base = REP.render(t, inp, cp2)
    cells = (("METRICS", "qlike"), ("TESTS", "delta_pct"), ("TESTS", "lo90"), ("TESTS", "p_one"), ("TESTS", "p_holm_N"), ("PAIRS", "delta_pct"), ("PAIRS", "lo90"), ("PAIRS", "p_one"), ("PAIRS", "p_holm_N"),
             ("PAIRS", "p_holm_Nwide"), ("GO", "value"), ("MCS", "mean_qlike"), ("STRESS", "qlike"), ("SPEC", "share_pi_gt03"), ("SPEC", "gain_on"), ("SPEC", "gain_rest_lo90"), ("SPEC", "spearman_pi_gain_hindsight"),
             ("SPEC", "switch_freq"), ("WEIGHTS", "mean_pi"), ("V1", "qlike_v1"), ("V1", "d_k3_v2"))
    for key, col in cells:
        t2 = {k: v.copy() for k, v in t.items()}
        t2[key][col] = t2[key][col].astype(float) + 0.123456
        assert REP.render(t2, inp, cp2) != base, (key, col)
    assert REP.render(t, inp, dict(cp2, cond9=dict(cp2["cond9"], share_pi1_gt03=0.5))) != base


def test_numbers_have_no_pvalues_and_json_roundtrip(world):
    inp, t, _ = world
    n = R.numbers(t, inp)
    dump = json.dumps(n, ensure_ascii=False)
    assert "p_one" not in dump and "p_holm" not in dump and "p_dm" not in dump and json.loads(dump) == n
    assert any(k.startswith("H2|cfg6|cfg4|") for k in n["pairs"]) and len(n["qlike"]) == 16 and set(n["go"]) == set(E2.CFGS)
    assert "p_dm_one" not in dump and "shares_ok" not in dump and "cond9" not in n         # условие 9 содержит p — только после eval


class Fake:
    """Подмена записи: события в порядке вызова; CHECKPOINT ключа `2` — в словаре (через JSON)."""

    def __init__(self):
        self.events, self.cp = [], {}

    def update_2(self, sub, val):
        self.events.append("cp:" + sub)
        self.cp[sub] = json.loads(json.dumps(val))


@pytest.fixture
def written(world, monkeypatch, tmp_path):
    inp, t, _ = world
    fk = Fake()
    monkeypatch.setattr(R, "load", lambda: inp)
    monkeypatch.setattr(R2, "update_2", fk.update_2)
    monkeypatch.setattr(R.CR, "read_cp", lambda: {"2": fk.cp})
    monkeypatch.setattr(IO, "REP", tmp_path)
    orig = IO.write_csv
    monkeypatch.setattr(IO, "write_csv", lambda df, name, note, status=None: fk.events.append("csv:" + name) or orig(df, name, note, status))
    monkeypatch.setattr(IO, "code_hash", lambda: "c")
    monkeypatch.setattr(IO, "inputs_hash", lambda: "i")
    assert R.write_eval() == 0
    return fk, tmp_path, inp


def test_eval_is_written_before_any_table_with_p_values_and_cond9_after_tables(written):
    fk, _, _ = written
    ev = fk.events
    assert ev[0] == "cp:eval" and ev.index("cp:eval") < ev.index("csv:2_TESTS.csv") < ev.index("csv:2_PAIRS.csv") < ev.index("cp:rules") < ev.index("cp:cond9")
    assert all(ev.index("cp:eval") < ev.index("csv:" + n) for n in R.NAMES.values())


def test_check_eval_ok_on_written_and_catches_changes_and_writes_nothing(written, monkeypatch):
    fk, d, inp = written
    digest = lambda: hashlib.sha256(b"".join(p.read_bytes() for p in sorted(d.glob("*")))).hexdigest()
    before = digest()
    monkeypatch.setattr(IO, "write_csv", lambda *a, **k: (_ for _ in ()).throw(AssertionError("check пишет таблицу")))
    assert R.check_eval({"2": fk.cp}) == [] and digest() == before                          # проверка ничего не пишет
    rep = d / "LEAF2_REPORT.md"
    txt = rep.read_text(encoding="utf-8")
    rep.write_text(txt.replace("0.1", "0.2", 1), encoding="utf-8")                           # подменено число в отчёте
    assert any("LEAF2_REPORT" in b for b in R.check_eval({"2": fk.cp}))
    rep.write_text(txt, encoding="utf-8")
    tab = d / "2_PAIRS.csv"
    raw = tab.read_text(encoding="utf-8")
    tab.write_text(raw.replace("да", "нет", 1) if "да" in raw else raw.replace("нет", "да", 1), encoding="utf-8")        # подменена ячейка таблицы
    assert any("2_PAIRS.csv" in b for b in R.check_eval({"2": fk.cp}))
    tab.write_text(raw, encoding="utf-8")
    bad = {"2": dict(fk.cp, cond9=dict(fk.cp["cond9"], cond9=not fk.cp["cond9"]["cond9"]))}
    assert any("cond9" in b for b in R.check_eval(bad))
    assert any("eval" in b for b in R.check_eval({"2": {k: v for k, v in fk.cp.items() if k != "eval"}}))


def test_run_stage2_check_includes_eval_check(monkeypatch):
    """`--stage 2 --check` вызывает сверку оценки (мутация: сверка не подключена — замечание не доходит до rc)."""
    monkeypatch.setattr(R2, "check", lambda cp, ids=E2.CFGS[3:] and ("4", "5", "6", "7"): [])
    monkeypatch.setattr(R2.CR, "read_cp", lambda: {"2": {}})
    monkeypatch.setattr(R, "check_eval", lambda cp: ["2.eval: пересчёт чисел ≠ CHECKPOINT"])
    monkeypatch.setattr(R2.IO, "outputs_digest", lambda: "same")
    assert R2.check_run(("4", "5", "6", "7")) == 1
    monkeypatch.setattr(R, "check_eval", lambda cp: [])
    assert R2.check_run(("4", "5", "6", "7")) == 0
