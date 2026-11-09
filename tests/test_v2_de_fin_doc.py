"""FIN: итоговый документ и стадия — генератор из таблиц, сверка чисел, мутации: подмена числа, «лучше / хуже» и знак Δ, «подтверждено» при p Холма ≥ 0,05, X5 как посчитанная, `--check` пишет файл (синтетика во временном каталоге)."""
import json
import re

import numpy as np
import pandas as pd
import pytest

from models_v2.s15 import de0_io as IO, de_e5_stage as S5, de_fin_doc as D, de_fin_doc2 as D2, de_fin_stage as SF, de_fin_tabs as T, de_reg as R


def synth(seed: int = 1):
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(sum((list(pd.bdate_range("%d-01-03" % y, periods=70)) for y in range(2018, 2027)), []))
    tp, n = idx + pd.Timedelta(days=1), len(idx)
    lv = 0.3 * np.sin(np.arange(n) / 15)
    rv = np.exp(lv + rng.normal(0, 0.5, n))
    f = lambda s: np.exp(lv + rng.normal(0, s, n))
    ser = {k: f(0.4) for k in T.CTRL}
    s = T.assemble(ser, {k: f(0.3) for k in ("X1", "X2", "X3", "X4", "ENS")})
    return idx, T.all_tables(rv, s, tp, rng.integers(0, 2, n))


def reg_frame(x5="не считается"):
    rows = [(i, "зарегистрирована") for i in ("X1", "X2", "X3", "X4", "X5")] + [(i, "посчитана") for i in ("X1", "X2", "X3", "X4")] + [("X5", x5)]
    return pd.DataFrame([dict(id=i, name="n", formula="X%s: результат 1.23 %% К1" % i[1:], condition="", status=s, registered_at="01.01.2020 10:00:00") for i, s in rows])


def causes():
    return pd.DataFrame([dict(hybrid="A2", adv_2018_2021=0.0085, adv_2022_2025=-0.0307, d_adv=-0.0392, part_base=-0.0471, part_gu=0.0195, part_go=-0.0116)])


@pytest.fixture(scope="module")
def base():
    idx, t = synth()
    e5 = dict(condition=False, best="X3", window="2018_2021", qlike=dict(X2=0.17, X3=0.16, X4=0.165), rule62="нет", delta_pct=2.0, lo90=0.5, hi90=3.5, p_one=0.99, years_neg=0, n_years=4, base="V6:A0g")
    ctx = dict(t=t, num=dict(n_dates=len(idx), first="2018-01-03", last="2026-02-26"), leaf=(100, 0), e5=e5, reg=reg_frame(), causes=causes(), xsha={"x%d.parquet" % i: "ab" * 32 for i in range(1, 5)}, tab_sha={k: "cd" * 32 for k in SF.NAMES})
    return ctx, D2.build(ctx)


def test_doc_has_first_line_sections_status_and_every_decimal_recorded(base):
    ctx, doc = base
    lines = doc.splitlines()
    assert lines[0].startswith("**Дополнительная прогностическая ценность") and lines[4] == D.STATUS
    heads = [ln for ln in lines if ln.startswith("## ")]
    assert len(heads) == 11 and all(p in h for p, h in zip(D2.SECTIONS, heads))
    D2.build(ctx)
    assert D.doc_decimals(doc) == D.recorded() and "bps" not in doc.lower()
    assert "X5 — не считается" in doc and "посчитаны 4" in doc


def test_mutation_substituted_number_is_caught(base):
    ctx, doc = base
    m = re.search(r"DM p = (\d\.\d+)", doc)
    bad = doc.replace(m.group(0), "DM p = 0.001", 1)
    D2.build(ctx)
    assert bad != doc and D.doc_decimals(bad) != D.recorded()          # подмена вне таблиц ловится сверкой чисел
    t2 = {k: v.copy() if isinstance(v, pd.DataFrame) else v for k, v in ctx["t"].items()}
    t2["PAIRS"].loc[0, "delta_pct"] += 1.0                              # подмена в таблице → другой документ
    assert D2.build(dict(ctx, t=t2)) != doc


def test_mutation_flipped_better_worse_and_delta_sign_are_caught(base, monkeypatch):
    ctx, doc = base
    r = D.pair(ctx["t"], "X1", "K2", "2018_2021")
    assert D.better(float(r["delta_q"])) == ("лучше" if r["delta_q"] < 0 else "хуже") and D.pair_txt(r, "K2").endswith("(QLIKE %s)" % D.better(float(r["delta_q"])))
    monkeypatch.setattr(D, "better", lambda dq: "хуже" if dq < 0 else "лучше")
    assert D2.build(ctx) != doc
    monkeypatch.undo()
    t2 = {k: v.copy() if isinstance(v, pd.DataFrame) else v for k, v in ctx["t"].items()}
    t2["PAIRS"]["delta_pct"] = -t2["PAIRS"]["delta_pct"]                # перевёрнутый знак Δ
    assert D2.build(dict(ctx, t=t2)) != doc


def test_confirmed_needs_p_holm_below_alpha_and_mutation_is_caught(base, monkeypatch):
    ctx, doc = base
    t2 = {k: v.copy() if isinstance(v, pd.DataFrame) else v for k, v in ctx["t"].items()}
    p = t2["PAIRS"]
    k = p["window"] != "2022_2025"
    p.loc[k, "rule62"], p.loc[k, "p_holm_N"] = "да", 0.2                # правило «да», но p Холма ≥ 0,05
    honest = D2.build(dict(ctx, t=t2))
    assert "| подтверждено |" not in honest and "ценность над" in honest.splitlines()[0] and ": нет.**" in honest.splitlines()[0]
    monkeypatch.setattr(D, "ALPHA", 2.0)                                # мутант: порог Холма снят
    mutant = D2.build(dict(ctx, t=t2))
    assert "| подтверждено |" in mutant and ": есть.**" in mutant.splitlines()[0]
    monkeypatch.undo()
    p.loc[k, "p_holm_N"] = 0.01
    assert ": есть.**" in D2.build(dict(ctx, t=t2)).splitlines()[0]    # честный положительный случай


def test_x5_shown_as_computed_is_caught_by_registry_check():
    assert SF.registry_bad(reg_frame()) == []
    assert SF.registry_bad(reg_frame("посчитана")) != []                 # X5 как посчитанная
    assert SF.registry_bad(reg_frame()[lambda d: d["status"] != "посчитана"]) != []


@pytest.fixture()
def stage(tmp_path, monkeypatch, base):
    ctx, _ = base
    rep = tmp_path / "harj_deep"
    rep.mkdir()
    monkeypatch.setattr(IO, "REP", rep)
    monkeypatch.setattr(R, "TRIALS", rep / "ALL_TRIALS.csv")
    monkeypatch.setattr(IO, "CP_FILE", rep / "CHECKPOINT.json")
    R.register("01.01.2020 10:00:00")
    for i, c in enumerate(S5.CONF):
        pd.DataFrame([dict(series=c, window="2018_2021", qlike=0.16 + 0.005 * i)]).to_csv(rep / ("E%d_METRICS.csv" % (i + 2)), index=False)
        pd.DataFrame([dict(x=c, y="V6:A0g", window="2018_2021", rule62="нет", delta_pct=1.0, lo90=0.0, hi90=2.0, p_one=0.9, years_neg=0, n_years=4)]).to_csv(rep / ("E%d_PAIRS.csv" % (i + 2)), index=False)
    S5.write()
    cur = R.read()
    rows = pd.DataFrame([dict(id=i, name="r", formula="f", condition="", status="посчитана", registered_at="02.01.2020 10:00:00") for i in ("X2", "X3", "X4")])
    R.TRIALS.write_text(R.TRIALS.read_text(encoding="utf-8") + rows.to_csv(index=False, header=False, lineterminator="\n"), encoding="utf-8")
    causes().to_csv(rep / "DE0_CAUSES.csv", index=False)
    files = {}
    for i in range(1, 5):
        files["X%d" % i] = rep / ("x%d.parquet" % i)
        files["X%d" % i].write_bytes(b"x%d" % i)
    monkeypatch.setattr(SF, "x_files", lambda: files)
    monkeypatch.setattr(SF, "compute", lambda inp=None: (ctx["t"], ctx["num"], (100, 0), []))
    return rep


def test_write_then_check_clean_and_check_only_reads(stage):
    SF.write()
    h = SF.outputs_hash()
    assert SF.check() == [] and SF.outputs_hash() == h
    assert json.loads((stage / "CHECKPOINT.json").read_text(encoding="utf-8"))["FIN"]["leaf_mismatch"] == 0


def test_check_catches_edited_doc_edited_table_and_missing_section(stage):
    SF.write()
    doc = SF.doc_path()
    txt = doc.read_text(encoding="utf-8")
    doc.write_text(txt.replace("## 8. Абляции", "## 8. Другое"), encoding="utf-8")
    assert any("документ" in b for b in SF.check())
    doc.write_text(txt, encoding="utf-8")
    t = stage / "FIN_TESTS.csv"
    orig = t.read_text(encoding="utf-8")
    t.write_text(re.sub(r"(\n[^\n]*?,)(-?\d+\.\d+)", lambda m: m.group(1) + "7" + m.group(2), orig, count=1), encoding="utf-8")
    assert any("FIN_TESTS" in b for b in SF.check())


def test_mutation_check_that_writes_a_file_is_caught(stage, monkeypatch):
    SF.write()

    def writing_check():
        (IO.REP / "FIN_LEAK.csv").write_text("x", encoding="utf-8")        # «--check» пишет файл
        return []

    monkeypatch.setattr(SF.S5, "check", writing_check)
    assert any("изменил выходы" in b for b in SF.check())


def test_write_refuses_when_condition_x5_not_false(stage):
    cp = json.loads((stage / "CHECKPOINT.json").read_text(encoding="utf-8"))
    cp["E5"]["condition"] = True
    (stage / "CHECKPOINT.json").write_text(json.dumps(cp), encoding="utf-8")
    with pytest.raises(SystemExit, match="E5"):
        SF.write()
