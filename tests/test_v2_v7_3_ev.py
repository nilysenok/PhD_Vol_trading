"""V7 лист 3, G43: оценка конфигурации 8 и `LEAF3_REPORT.md` целиком из таблиц; CHECKPOINT `3.eval` — числа без p-значений и записан ДО таблиц с p; `check_eval` сверяет с файлами и ничего не пишет.
Известные ответы: 8 лучше 6 ⇒ Δ < 0 и правило 6.2 «да» (мутация «X и Y перепутаны» — «нет»); веса по годам. Мутации: число в отчёте подменено / вписано вручную, p в `3.eval`, `3.eval` после таблиц, файл таблицы или отчёта изменён.
Синтетика, файлы SBER не читаются."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from models_v2.s14 import v7_0_io as IO, v7_2_ev as E2, v7_3_ev as E3, v7_3_evrep as REP, v7_3_evrun as R, v7_3_run as R3

IDX = pd.bdate_range("2018-01-03", "2026-02-26")
N = len(IDX)


@pytest.fixture(scope="module")
def world():
    g = np.random.default_rng(2)
    rv = np.exp(g.normal(0, 0.5, N))
    s = {k: rv * np.exp(g.normal(0, 0.2, N)) for k in E2.CTRL + E3.CFGS}
    s["cfg8"] = rv * np.exp(g.normal(0, 0.1, N))                                               # 8 заметно лучше 6
    pi = g.dirichlet([8, 8, 8], N)
    H = s["K1"]
    G = np.log(s["cfg8"] / H)
    df8 = pd.DataFrame({"d": IDX, "t_plus": IDX, "rv": rv, "H": H, "ell": g.normal(0, 0.1, N), "w": 0.5 + 0.01 * (IDX.year - 2018).to_numpy(), "G": G, "F": s["cfg8"], "fitted": 1.0,
                        "pi0": pi[:, 0], "pi1": pi[:, 1], "pi2": pi[:, 2], "g1": g.normal(0, 0.1, N), "g2": g.normal(0, 0.1, N)})
    inp = dict(idx=IDX, rv=rv, tp=IDX.to_numpy("datetime64[ns]"), series=s, df8=df8, shas={"V6:A0": "x"})
    t = R.compute(inp)
    return inp, t, dict(rules=R.rules(t))


def test_pair_8_vs_6_known_answer_and_swapped_pair_mutation_is_caught(world, monkeypatch):
    inp, t, _ = world
    p = t["PAIRS"]
    r = p[(p["window"] == "2018_2021")].iloc[0]
    assert (r["x"], r["y"]) == ("cfg8", "cfg6") and r["delta_q"] < 0 and r["delta_pct"] < 0 and r["rule62"] == "да" and p[p["window"] == "2022_2025"]["rule62"].iloc[0] == "—"
    assert list(p["y"].unique()) == ["cfg6", "cfg4"] and set(p["window"]) == set(E2.PAIR_WIN)                                              # стресс: без 1 % крайних, без 2018 (2019–2021)
    monkeypatch.setattr(E3, "PAIRS8", (("H8 якорь", "cfg6", "cfg8"),))                         # мутант: пара перепутана
    q = E3.main_tables(inp["rv"], inp["series"], inp["tp"], inp["df8"])["PAIRS"]
    assert q[q["window"] == "2018_2021"].iloc[0]["delta_q"] > 0 and q[q["window"] == "2018_2021"].iloc[0]["rule62"] == "нет"


def test_weights_by_year_known_answer_and_only_fitted_rows(world):
    inp, t, _ = world
    w = t["WEIGHTS"]
    assert np.isclose(w[(w["quantity"] == "w") & (w["period"] == "2020")]["mean"].iloc[0], 0.52) and np.isclose(w[(w["quantity"] == "w") & (w["period"] == "до 2021 (≤ 2021)")]["mean"].iloc[0], 0.515, atol=2e-3)
    assert np.isclose(w[(w["period"] == "2018") & (w["quantity"] != "w")]["mean"].sum(), 1.0)                         # π_0 + π_1 + π_2 = 1
    df = inp["df8"].copy()
    df.loc[df["t_plus"].dt.year == 2020, "fitted"] = 0.0
    w2 = E3.weights_table(df)
    assert w2[(w2["quantity"] == "w") & (w2["period"] == "2020")]["n"].iloc[0] == 0                                  # строки без переоценки в среднее не входят


def test_render_is_deterministic_and_contains_table_numbers(world):
    inp, t, cp3 = world
    a = REP.render(t, inp, cp3)
    assert a == REP.render(t, inp, cp3) and a.count("\n") > 60 and "потолок" in a and "Холм N=733" in a and "9 не считается" in a
    m = t["METRICS"]
    assert REP.f(m[(m["series"] == "cfg8") & (m["window"] == "2018_2021")]["qlike"].iloc[0], 6) in a


def test_mutation_number_substituted_and_typed_by_hand_do_not_survive(world):
    inp, t, cp3 = world
    base = REP.render(t, inp, cp3)
    v = REP.f(t["METRICS"].query("series == 'K2' and window == '2018_2021'")["qlike"].iloc[0], 6)
    assert v in base and base.replace(v, v[:-1] + ("1" if v[-1] != "1" else "2"), 1) != base                                  # подмена числа в тексте отличима
    cells = (("METRICS", "qlike"), ("TESTS", "delta_pct"), ("TESTS", "lo90"), ("TESTS", "p_one"), ("TESTS", "p_holm_N"), ("PAIRS", "delta_pct"), ("PAIRS", "lo90"), ("PAIRS", "p_one"), ("PAIRS", "delta_q"),
             ("PAIRS", "p_holm_N"), ("PAIRS", "p_holm_Nwide"), ("GO", "value"), ("MCS", "mean_qlike"), ("STRESS", "qlike"), ("WEIGHTS", "mean"))
    for key, col in cells:                                                                                                    # любая ячейка таблицы, попавшая в отчёт, меняет текст: чисел «вписанных вручную» нет
        t2 = {k: x.copy() for k, x in t.items()}
        t2[key][col] = t2[key][col].astype(float) + 0.123456
        assert REP.render(t2, inp, cp3) != base, (key, col)
    r = dict(cp3["rules"]["cfg8"], p_holm_N_vs_K2_K3=[0.5, 0.5])
    assert REP.render(t, inp, dict(rules=dict(cp3["rules"], cfg8=r))) != base


def test_numbers_have_no_pvalues_and_json_roundtrip(world):
    inp, t, _ = world
    n = R.numbers(t, inp)
    dump = json.dumps(n, ensure_ascii=False)
    assert "p_one" not in dump and "p_holm" not in dump and "p_dm" not in dump and json.loads(dump) == n
    assert any(k.startswith("H8 якорь|cfg8|cfg6|") for k in n["pairs"]) and len(n["qlike"]) == 17 and set(n["go_cfg8"]) == set("0123456") and "w|2020" in n["weights"]


class Fake:
    def __init__(self):
        self.events, self.cp = [], {}

    def update_3(self, sub, val):
        self.events.append("cp:" + sub)
        self.cp[sub] = json.loads(json.dumps(val))


@pytest.fixture
def written(world, monkeypatch, tmp_path):
    inp, _, _ = world
    fk = Fake()
    monkeypatch.setattr(R, "load", lambda: inp)
    monkeypatch.setattr(R, "update_3", fk.update_3)
    monkeypatch.setattr(R.CR, "read_cp", lambda: {"3": fk.cp})
    monkeypatch.setattr(IO, "REP", tmp_path)
    orig = IO.write_csv
    monkeypatch.setattr(IO, "write_csv", lambda df, name, note, status=None: fk.events.append("csv:" + name) or orig(df, name, note, status))
    monkeypatch.setattr(IO, "code_hash", lambda: "c")
    monkeypatch.setattr(IO, "inputs_hash", lambda: "i")
    assert R.write_eval() == 0
    return fk, tmp_path


def eval_first(ev) -> bool:
    """`3.eval` записан раньше любой таблицы с p-значениями, `3.rules` — после таблиц."""
    return ev[0] == "cp:eval" and all(ev.index("cp:eval") < ev.index("csv:" + n) for n in R.NAMES.values()) and ev.index("csv:3_PAIRS.csv") < ev.index("cp:rules")


def test_eval_is_written_before_any_table_with_p_values_and_rules_after_tables(written):
    assert eval_first(written[0].events)


def test_mutation_eval_after_tables_is_caught_on_real_write(world, monkeypatch, tmp_path):
    """Мутант: `3.eval` откладывается до конца записи (после таблиц) — детектор порядка это видит."""
    inp, _, _ = world
    fk = Fake()
    monkeypatch.setattr(R, "load", lambda: inp)
    monkeypatch.setattr(R, "update_3", lambda sub, val: fk.update_3(sub, val) if sub != "eval" else None)
    monkeypatch.setattr(R.CR, "read_cp", lambda: {"3": {"rules": {"cfg8": {"status": "x"}}}})
    monkeypatch.setattr(IO, "REP", tmp_path)
    orig = IO.write_csv
    monkeypatch.setattr(IO, "write_csv", lambda df, name, note, status=None: fk.events.append("csv:" + name) or orig(df, name, note, status))
    monkeypatch.setattr(IO, "code_hash", lambda: "c")
    monkeypatch.setattr(IO, "inputs_hash", lambda: "i")
    monkeypatch.setattr(R.REP, "render", lambda *a: "")
    R.write_eval()
    fk.update_3("eval", {})
    assert not eval_first(fk.events)


def test_check_eval_ok_on_written_and_catches_changes_and_writes_nothing(written, monkeypatch):
    fk, d = written
    digest = lambda: hashlib.sha256(b"".join(p.read_bytes() for p in sorted(d.glob("*")))).hexdigest()
    before = digest()
    monkeypatch.setattr(IO, "write_csv", lambda *a, **k: (_ for _ in ()).throw(AssertionError("check пишет таблицу")))
    assert R.check_eval({"3": fk.cp}) == [] and digest() == before
    rep = d / "LEAF3_REPORT.md"
    txt = rep.read_text(encoding="utf-8")
    rep.write_text(txt.replace("0.1", "0.2", 1), encoding="utf-8")                                                           # подменено число в отчёте
    assert any("LEAF3_REPORT" in b for b in R.check_eval({"3": fk.cp}))
    rep.write_text(txt, encoding="utf-8")
    tab = d / "3_PAIRS.csv"
    raw = tab.read_text(encoding="utf-8")
    tab.write_text(raw.replace("да", "нет", 1) if "да" in raw else raw.replace("нет", "да", 1), encoding="utf-8")            # подменена ячейка таблицы
    assert any("3_PAIRS.csv" in b for b in R.check_eval({"3": fk.cp}))
    tab.write_text(raw, encoding="utf-8")
    assert any("eval" in b for b in R.check_eval({"3": {k: v for k, v in fk.cp.items() if k != "eval"}}))
    bad = dict(fk.cp, rules=dict(fk.cp["rules"], cfg8=dict(fk.cp["rules"]["cfg8"], go=not fk.cp["rules"]["cfg8"]["go"])))
    assert any("rules" in b for b in R.check_eval({"3": bad}))


def test_stage3_check_includes_eval_check_when_eval_is_written(monkeypatch):
    monkeypatch.setattr(R3, "check", lambda cp: [])
    monkeypatch.setattr(R3, "refusal_ok", lambda refused, cp_path=None: True)
    monkeypatch.setattr(R3.CR, "read_cp", lambda: {"3": {"eval": {}}})
    monkeypatch.setattr(R3.IO, "outputs_digest", lambda: "same")
    monkeypatch.setattr(R, "check_eval", lambda cp: ["3.eval: пересчёт чисел ≠ CHECKPOINT"])
    assert R3.check_run(["8"], ["9"]) == 1
    monkeypatch.setattr(R, "check_eval", lambda cp: [])
    assert R3.check_run(["8"], ["9"]) == 0
