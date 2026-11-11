"""S-Z-EVAL (Ж-258, ZE.G2), тестировщик: `z_doc`, `z_doc_fmt`, `z_doc_s1/s2` — отчёт `S_Z_REPORT.md` собирается только из таблиц.
Боевые `S_Z_*.csv`, `S_Z_REPORT.md`, CHECKPOINT только читаются; все мутации — на копиях в tmp. Проверка `z_doc.check` обязана вернуть непустой список на каждом мутанте (подменённое число, «лучше / хуже», «эквивалентно», пометка, порядок рядов, границы дат)."""
import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import test_s18_z_eval_fx as FX
from models_v2.s11 import v4_ntp_state as NS
from models_v2.s18 import z_doc as D
from models_v2.s18 import z_doc_fmt as DF
from models_v2.s18 import z_doc_s1 as S1
from models_v2.s18 import z_eval as E
from models_v2.s18 import z_eval_cfg as C
from models_v2.s18 import z_eval_tab as T

REP, CKPT = E.REP, E.CKPT
FILES = list(E.OUTS) + [E.REPORT]


@pytest.fixture
def cp(tmp_path):
    for n in FILES:
        shutil.copy(REP / n, tmp_path / n)
    shutil.copy(CKPT, tmp_path / "CHECKPOINT.json")
    return tmp_path, tmp_path / "CHECKPOINT.json"


def sha_all(d: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}


def edit(p: Path, f) -> None:
    p.write_text(f(p.read_text(encoding="utf-8")), encoding="utf-8")


def test_real_copy_is_clean_and_check_reads_only(cp, monkeypatch):
    d, ck = cp
    h0 = sha_all(d)

    def no_write(*a, **k):
        raise AssertionError("запись в режиме --check (№ 123)")
    monkeypatch.setattr(Path, "write_text", no_write)
    monkeypatch.setattr(Path, "write_bytes", no_write)
    assert D.check(d, ck) == []
    monkeypatch.undo()
    assert sha_all(d) == h0


def test_mut_check_that_writes_is_caught_by_the_read_only_spy(cp, monkeypatch):
    d, ck = cp
    orig = D.check

    def writing_check(rep, ckpt):
        (rep / E.REPORT).write_text("x", encoding="utf-8")
        return orig(rep, ckpt)

    def no_write(*a, **k):
        raise AssertionError("запись в режиме --check")
    monkeypatch.setattr(Path, "write_text", no_write)
    with pytest.raises(AssertionError):
        writing_check(d, ck)


def test_report_numbers_equal_tables_first_line_and_first_table_order(cp):
    d, ck = cp
    txt = (d / E.REPORT).read_text(encoding="utf-8")
    assert txt == D.current(d, ck) and txt.startswith("# " + E.MARK + "\n")
    i = [txt.find("| %s |" % s) for s in C.SHOWN]
    assert all(a >= 0 for a in i) and i == sorted(i)                                    # K1 … S-RCKZs в порядке задания
    assert txt.find("## 1.") < txt.find("## 2.") and "t⁺ 2018-01-04…2026-11-10, n = 2198" in txt
    tb = D.tabs(d)["table"]
    r = tb[tb["series"] == "S-RCKZ"].iloc[0]
    assert "| S-RCKZ | %.6f | %s | %s | %d |" % (r["qlike"], r["first_tplus"], r["last_tplus"], r["n"]) in txt
    dec = D.tabs(d)["dec"].iloc[-1]
    assert "**Z-DEC (код %s):** %s" % (dec["value"], dec["text"]) in txt


def test_mut_substituted_number_in_csv_or_report_caught(cp):
    d, ck = cp
    edit(d / "S_Z_TABLE.csv", lambda s: re.sub(r"(?m)^(main,S-RCKZ,\d+,[^,]+,[^,]+,)([0-9.]+)", lambda m: m.group(1) + "%.12g" % (float(m.group(2)) + 1e-4), s, count=1))
    assert D.check(d, ck) == ["отчёт ≠ пересборке из таблиц"]
    shutil.copy(REP / "S_Z_TABLE.csv", d / "S_Z_TABLE.csv")
    assert D.check(d, ck) == []
    edit(d / E.REPORT, lambda s: re.sub(r"(?m)^(\| S-RCKZ \| )(\d\.\d{6})", lambda m: m.group(1) + "%.6f" % (float(m.group(2)) + 1e-6), s, count=1))
    assert D.check(d, ck) == ["отчёт ≠ пересборке из таблиц"]
    shutil.copy(REP / E.REPORT, d / E.REPORT)
    shutil.copy(REP / E.REPORT, d / E.REPORT)
    v = D.tabs(d)["stats"]
    w = v[(v["kind"] == "main") & (v["hypothesis"] == "S-RCKZ|S-RCK")]["verdict"].iloc[0]
    assert w.startswith("численно лучше")
    edit(d / "S_Z_STATS.csv", lambda s: s.replace(w, "лучше S-RCK; подтверждено (p Холма N < 0,05 и p SPA < 0,05)", 1))      # вердикт в таблице подменён, отчёт прежний
    assert "отчёт ≠ пересборке из таблиц" in D.check(d, ck)


def test_mut_flipped_better_worse_in_report_caught(cp):
    d, ck = cp
    txt = (d / E.REPORT).read_text(encoding="utf-8")
    assert "численно не лучше" in txt or "численно лучше" in txt
    edit(d / E.REPORT, lambda s: s.replace("численно не лучше", "@@", 1).replace("численно лучше", "численно не лучше", 1).replace("@@", "численно лучше", 1))
    assert "отчёт ≠ пересборке из таблиц" in D.check(d, ck)


def test_mut_equivalent_and_stray_confirmed_words_caught(cp):
    d, ck = cp
    t = D.tabs(d)
    assert D.words("Спокойный текст; минус — лучше.", t) == []
    assert D.words("кандидат лучшего ряда не выбирался", t) == []                       # «лучшего» — не вердикт
    assert D.words("различие эквивалентно нулю при p ≥ 0,05", t) == ["слово «эквивалент» в отчёте"]
    assert D.words("S-RCKZ подтверждено лучше", t) == ["слово «подтверждено» / «лучше» вне формулировок разд. 7.5"]
    v = t["stats"]["verdict"].dropna().iloc[0]
    assert D.words("— " + v + " —", t) == []                                            # формулировка разд. 7.5 из столбца verdict разрешена
    edit(d / E.REPORT, lambda s: s + "\nS-RCKZ эквивалентен S-RCK при p ≥ 0,05.\n")
    assert any("эквивалент" in b for b in D.check(d, ck))


def test_mut_first_line_dates_and_row_order_caught(cp):
    d, ck = cp
    edit(d / E.REPORT, lambda s: s.replace("# " + E.MARK, "# результат", 1))
    assert "первая строка не пометка" in D.check(d, ck)
    shutil.copy(REP / E.REPORT, d / E.REPORT)
    edit(d / E.REPORT, lambda s: s.replace("t⁺ 2018-01-04…2026-11-10, n = 2198", "n = 2198"))
    assert "нет границ дат и n" in D.check(d, ck)
    shutil.copy(REP / E.REPORT, d / E.REPORT)

    def swap(s):
        a, b = (re.search(r"(?m)^\| %s \|.*$" % k, s).group(0) for k in ("K2", "K3"))
        return s.replace(a, "@@a", 1).replace(b, a, 1).replace("@@a", b, 1)
    edit(d / E.REPORT, swap)
    assert "первая таблица: порядок рядов ≠ заданию" in D.check(d, ck)


def test_mut_report_missing_or_table_missing_is_not_clean(cp):
    d, ck = cp
    (d / E.REPORT).unlink()
    assert "отчёт ≠ пересборке из таблиц" in D.check(d, ck)


# ───────── форматирование и первая таблица на известном ответе ─────────
def test_fmt_helpers_known_answers():
    assert DF.cap("2018-01-04", "2026-11-10", 2198) == "t⁺ 2018-01-04…2026-11-10, n = 2198"
    assert DF.sgn(-0.12345678) == "-0.1235" and DF.sgn(np.nan) == "—" and DF.pv(0.04999) == "0.0500" and DF.pv(np.nan) == "—"
    assert DF.ci(-0.1, -0.2, 0.05) == "-0.1000 [-0.2000; +0.0500]" and DF.ci(-0.1, np.nan, np.nan) == "-0.1000" and DF.fv("—") == "—" and DF.fv(0.5) == "+0.5000"
    df = pd.DataFrame({"a": [1.0, np.nan], "b": ["x", "y"]})
    assert DF.md(df, {"a": "%.2f", "b": "%s"}).splitlines() == ["| a | b |", "|---|---|", "| 1.00 | x |", "| — | y |"]
    assert DF.pick(df, b="y")["b"] == "y"
    with pytest.raises(ValueError):
        DF.pick(df, b="z")
    with pytest.raises(ValueError):
        DF.pick(pd.DataFrame({"b": ["x", "x"]}), b="x")


def test_first_table_columns_dash_for_base_and_sign_of_delta():
    with pytest.MonkeyPatch.context() as mp:
        FX.patch_boot(mp)
        rv, tp, F = FX.world(n=400, gain={"S-RCKZ": 0.05})
        ls = T.losses(rv, F)
        tb = T.table(rv, tp, F, ls)
    txt = S1.first_table(tb)
    rows = {ln.split("|")[1].strip(): [c.strip() for c in ln.split("|")[2:-1]] for ln in txt.splitlines()[2:]}
    assert list(rows) == list(C.SHOWN) and txt.splitlines()[0].count("|") == len(S1.COLS) + 1
    assert rows["S-RCK"][4] == "—" and rows["W5-G"][5] == "—" and rows["X1"][6] == "—"              # база к самой себе — прочерк (столбцы: QLIKE, даты, n, Δ к S-RCK, W5-G, X1, K2, K3)
    z = tb.set_index("series").loc["S-RCKZ"]
    assert rows["S-RCKZ"][4].startswith("%+.4f" % z["d_S-RCK_pct"]) and "[" in rows["S-RCKZ"][4] and "[" not in rows["S-RCKZ"][6]
    assert float(rows["S-RCKZ"][4].split()[0]) < 0 and rows["S-RCKZ"][1] == tb["first_tplus"].iloc[0] and rows["S-RCKZ"][3] == "400"


# ───────── реальные выходы: QLIKE на всех датах, Σ вкладов, N, пометки ─────────
def test_real_tables_qlike_sum_of_years_n_marks_and_dec_consistency():
    t = D.tabs()
    tb, st, y, dec = t["table"], t["stats"], t["years"], t["dec"].set_index("id")
    inp = C.load_inputs()
    for _, r in tb.iterrows():
        assert r["n"] == 2198 and abs(r["qlike"] / float(np.mean(NS.qlike(inp["rv"], inp["F"][r["series"]]))) - 1) < 1e-10      # без обрезки, v4_ntp_state.qlike; CSV хранит 12 значащих цифр
    for (h, b, kd), g in y[y["base"] != "-"].groupby(["series", "base", "kind"]):
        assert abs(g["contrib_pct"].sum() - g["delta_total_pct"].iloc[0]) <= 1e-12
    m = st[st["kind"] == "main"]
    assert len(m) == 6 and (st["n_holm"] == 855).all() and (st["n_holm_wide"] == 3415).all() and (m["m_family"] == 6).all() and (st["B"] == 10000).all()
    assert (m["lag"] == 8).all() and (m["n"] == 2198).all()
    for n in E.OUTS:
        assert (REP / n).read_text(encoding="utf-8").startswith("# " + E.MARK + "\n")
    q = tb.set_index("series")["qlike"]
    assert bool(dec.loc["H-Z1", "passed"]) == bool(q["S-RCKZ"] < q["S-RCK"]) and bool(dec.loc["H-Z2", "passed"]) == bool(q["S-RCKZ"] < q["S-RCKZs"])
    ok = all(bool(dec.loc[i, "passed"]) for i in ("H-Z1", "H-Z2", "У к W5-G", "У к S-RCK"))
    conf = bool(dec.loc["подтверждение H-Z1", "passed"])
    assert str(dec.loc["Z-DEC", "value"]) == ("1" if ok and conf else "2" if ok else "3")
