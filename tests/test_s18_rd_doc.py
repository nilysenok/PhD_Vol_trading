"""S-RCD-EVAL (Ж-259, RE.G2), тестировщик: `rd_doc`, `rd_doc_s1`, `rd_doc_s2` — отчёт `S_RCD_REPORT.md` собирается только из таблиц.
Боевые `S_RCD_*.csv`, `S_RCD_REPORT.md`, CHECKPOINT только читаются; все мутации — на копиях в tmp. `rd_doc.check` обязан вернуть непустой список на каждом мутанте
(подменённое число в CSV и в отчёте, «лучше / хуже» и код RCD-DEC, «эквивалентно», пометка, порядок рядов и разделов, границы дат, числа гейта M.G4)."""
import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import test_s18_rd_eval_fx as FX
from models_v2.s11 import v4_ntp_state as NS
from models_v2.s18 import rd_doc as D
from models_v2.s18 import rd_doc_s1 as S1
from models_v2.s18 import rd_eval as E
from models_v2.s18 import rd_eval_cfg as C
from models_v2.s18 import rd_eval_tab as T
from models_v2.s18.z_doc_fmt import cap

REP, CKPT = E.REP, E.CKPT
FILES = list(E.OUTS) + [E.REPORT]
OK = "отчёт ≠ пересборке из таблиц"


@pytest.fixture
def cp(tmp_path):
    for n in FILES:
        shutil.copy(REP / n, tmp_path / n)
    shutil.copy(CKPT, tmp_path / "CHECKPOINT.json")
    return tmp_path, tmp_path / "CHECKPOINT.json"


def sha_all(d: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}


def edit(p: Path, f) -> None:
    old = p.read_text(encoding="utf-8")
    new = f(old)
    assert new != old, "мутация не применилась к %s" % p.name
    p.write_text(new, encoding="utf-8")


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
    i = [txt.find("| %s |" % s) for s in D.ORDER]
    assert tuple(D.ORDER) == tuple(C.SHOWN) and all(a >= 0 for a in i) and i == sorted(i)
    assert txt.find("## 1.") < txt.find("| K1 |") < txt.find("## 2.") and "t⁺ 2018-01-04…2026-11-10, n = 2198" in txt
    t = D.tabs(d)
    tb = t["table"]
    r = tb[tb["series"] == "S-RCD"].iloc[0]
    assert "| S-RCD | %.6f | %s | %s | %d |" % (r["qlike"], r["first_tplus"], r["last_tplus"], r["n"]) in txt
    dec = t["dec"].iloc[-1]
    assert "**RCD-DEC (код %s):** %s" % (dec["value"], dec["text"]) in txt
    assert "N программы %d / %d" % (858, 3421) in txt


def test_mut_substituted_number_in_csv_or_report_caught(cp):
    d, ck = cp
    edit(d / "S_RCD_TABLE.csv", lambda s: re.sub(r"(?m)^(main,S-RCD,\d+,[^,]+,[^,]+,)([0-9.]+)", lambda m: m.group(1) + "%.12g" % (float(m.group(2)) + 1e-4), s, count=1))
    assert D.check(d, ck) == [OK]
    shutil.copy(REP / "S_RCD_TABLE.csv", d / "S_RCD_TABLE.csv")
    assert D.check(d, ck) == []
    edit(d / E.REPORT, lambda s: re.sub(r"(?m)^(\| S-RCD \| )(\d\.\d{6})", lambda m: m.group(1) + "%.6f" % (float(m.group(2)) + 1e-6), s, count=1))
    assert D.check(d, ck) == [OK]
    shutil.copy(REP / E.REPORT, d / E.REPORT)
    edit(d / "S_RCD_DECOMP.csv", lambda s: re.sub(r"(?m)^(e_D_M,S-RCK,,)(-?[0-9.]+)", lambda m: m.group(1) + "%.12g" % (float(m.group(2)) + 1e-3), s, count=1))
    assert OK in D.check(d, ck)                                                          # число разложения подменено, отчёт прежний


def test_mut_verdict_and_decision_code_swapped_in_tables_caught(cp):
    d, ck = cp
    v = D.tabs(d)["stats"]
    w = v[(v["kind"] == "main") & (v["hypothesis"] == "S-RCD|S-CDF")]["verdict"].iloc[0]
    assert w.startswith("численно лучше") and "не подтверждено" in w
    edit(d / "S_RCD_STATS.csv", lambda s: s.replace(w, "лучше S-CDF; подтверждено (p Холма N < 0,05 и p SPA < 0,05)", 1))
    assert OK in D.check(d, ck)
    shutil.copy(REP / "S_RCD_STATS.csv", d / "S_RCD_STATS.csv")
    dec = D.tabs(d)["dec"].iloc[-1]
    assert str(dec["value"]) in "234"                                                     # RCD-DEC «1» (превзошёл) на реальных числах не выведен
    edit(d / "S_RCD_DEC.csv", lambda s: s.replace('выведено кодом",%s,' % dec["value"], 'выведено кодом",1,', 1))
    assert OK in D.check(d, ck)


def test_mut_flipped_better_worse_in_report_caught(cp):
    d, ck = cp
    edit(d / E.REPORT, lambda s: s.replace("численно лучше", "@@", 1).replace("численно не лучше", "численно лучше", 1).replace("@@", "численно не лучше", 1) if "численно не лучше" in s else s.replace("численно лучше", "численно не лучше", 1))
    assert OK in D.check(d, ck)
    shutil.copy(REP / E.REPORT, d / E.REPORT)
    edit(d / E.REPORT, lambda s: s.replace("**RCD-DEC (код 2)", "**RCD-DEC (код 1)", 1))
    assert OK in D.check(d, ck)


def test_mut_equivalent_and_stray_confirmed_words_caught(cp):
    d, ck = cp
    t = D.tabs(d)
    assert D.words("Спокойный текст; минус — лучше.", t) == []
    assert D.words("кандидат лучшего ряда не выбирался", t) == []                       # «лучшего» — не вердикт
    assert D.words("различие эквивалентно нулю при p ≥ 0,05", t) == ["слово «эквивалент» в отчёте"]
    assert D.words("S-RCD подтверждено лучше", t) == ["слово «подтверждено» / «лучше» вне формулировок разд. 7.5"]
    v = t["stats"]["verdict"].dropna().iloc[0]
    assert D.words("— " + v + " —", t) == []                                            # формулировка разд. 7.5 из столбца verdict разрешена
    edit(d / E.REPORT, lambda s: s + "\nS-RCD эквивалентен S-RCK при p ≥ 0,05.\n")
    assert any("эквивалент" in b for b in D.check(d, ck))
    shutil.copy(REP / E.REPORT, d / E.REPORT)
    edit(d / E.REPORT, lambda s: s + "\nS-RCD подтверждено лучше S-RCK.\n")
    assert any("подтверждено" in b for b in D.check(d, ck))


def test_mut_first_line_dates_row_order_sections_and_gate_disclosure_caught(cp):
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
    shutil.copy(REP / E.REPORT, d / E.REPORT)
    edit(d / E.REPORT, lambda s: s.replace("## 7. Решение RCD-DEC", "## 8. Решение RCD-DEC", 1))
    assert "порядок разделов ≠ Ж-259 (числа и вклад механизма → проверки → следующий шаг)" in D.check(d, ck)
    shutil.copy(REP / E.REPORT, d / E.REPORT)
    edit(d / E.REPORT, lambda s: s.replace("порог не менялся", "порог изменён"))
    assert "нет раскрытия красного гейта M.G4 (а) с порогом" in D.check(d, ck)


def test_mut_report_missing_is_not_clean(cp):
    d, ck = cp
    (d / E.REPORT).unlink()
    assert OK in D.check(d, ck)


def test_gate_m4_numbers_come_from_gates_file_not_typed(cp, monkeypatch, tmp_path):
    d, ck = cp
    g = D.gate_m4()
    assert set(g) == {"cd", "thr", "ck"} and all(v.startswith("+") for v in g.values())
    src = D.GATES.read_text(encoding="utf-8")
    alt = tmp_path / "GATES_alt.md"
    alt.write_text(re.sub(r"(НЕ ЗАКРЫТ \(а\): λ = 0,1: )\+[0-9,]+", r"\g<1>+9,99", src, count=1), encoding="utf-8")
    assert D.gate_m4(alt)["cd"] == "+9,99"
    real_gate = D.gate_m4
    monkeypatch.setattr(D, "gate_m4", lambda p=alt: real_gate(p))
    assert OK in D.check(d, ck)                                                       # числа гейта — из файла: подмена там меняет пересборку, отчёт прежний
    monkeypatch.undo()
    bad = tmp_path / "GATES_bad.md"
    bad.write_text("| M.G4 | нет красного |\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        D.gate_m4(bad)
    bad.write_text("без строки\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        D.gate_m4(bad)


def test_first_table_columns_dash_for_base_and_sign_of_delta(monkeypatch):
    FX.patch_boot(monkeypatch)
    inp = FX.inputs({FX.R: 0.05, FX.CDF: 0.0, FX.RCK: 0.0})
    ls = T.losses(inp["rv"], inp["F"])
    tb = T.table(inp["rv"], inp["tp"], inp["F"], ls)
    txt = S1.first_table(tb)
    rows = {ln.split("|")[1].strip(): [c.strip() for c in ln.split("|")[2:-1]] for ln in txt.splitlines()[2:]}
    assert list(rows) == list(C.SHOWN) and txt.splitlines()[0].count("|") == len(S1.COLS) + 1
    assert rows["S-CDF"][4] == "—" and rows["S-RCK"][5] == "—" and rows["W5-G"][6] == "—" and rows["X1"][7] == "—" and rows["K2"][8] == "—" and rows["K3"][9] == "—"     # база к самой себе — прочерк
    z = tb.set_index("series").loc["S-RCD"]
    assert rows["S-RCD"][4].startswith("%+.4f" % z["d_S-CDF_pct"]) and "[" in rows["S-RCD"][4] and "[" not in rows["S-RCD"][7]
    assert float(rows["S-RCD"][4].split()[0]) < 0 and float(rows["S-RCD"][5].split()[0]) < 0 and rows["S-RCD"][3] == str(C.N_D)
    assert cap(tb["first_tplus"].iloc[0], tb["last_tplus"].iloc[0], tb["n"].iloc[0]) == "t⁺ 2018-01-04…2026-11-10, n = 2198"
    from models_v2.s18 import a_eval_stat as S
    S.idx_for.cache_clear()


def test_real_tables_qlike_sum_of_years_n_marks_and_dec_consistency():
    t = D.tabs()
    tb, st, y, dec = t["table"], t["stats"], t["years"], t["dec"].set_index("id")
    inp = C.load_inputs()
    for _, r in tb.iterrows():
        assert r["n"] == 2198 and abs(r["qlike"] / float(np.mean(NS.qlike(inp["rv"], inp["F"][r["series"]]))) - 1) < 1e-10      # без обрезки; CSV хранит 12 значащих цифр
    for (h, b, kd), g in y[y["base"] != "-"].groupby(["series", "base", "kind"]):
        assert abs(g["contrib_pct"].sum() - g["delta_total_pct"].iloc[0]) <= 1e-10               # CSV хранит 12 значащих цифр: допуск 1e-10 (в памяти `p_dec` держит 1e-12)
    m = st[st["kind"] == "main"]
    assert len(m) == 6 and (st["n_holm"] == 858).all() and (st["n_holm_wide"] == 3421).all() and (m["m_family"] == 6).all() and (st["B"] == 10000).all()
    assert (m["lag"] == 8).all() and (m["n"] == 2198).all()
    for n in E.OUTS:
        assert (REP / n).read_text(encoding="utf-8").startswith("# " + E.MARK + "\n")
    q = tb.set_index("series")["qlike"]
    assert bool(dec.loc["H-RCD1", "passed"]) == bool(q["S-RCD"] < q["S-CDF"]) and bool(dec.loc["S-RCD против S-RCK", "passed"]) == bool(q["S-RCD"] < q["S-RCK"])
    d0 = 100.0 * (q["S-RCD"] - q["S-CDF"]) / q["S-CDF"]
    assert abs(float(dec.loc["H-RCD1", "value"]) - d0) < 1e-6                           # база H-RCD1 — S-CDF, процент от неё
    ok = all(bool(dec.loc[i, "passed"]) for i in ("H-RCD1", "У к W5-G", "У к S-RCK", "S-RCD против S-RCK"))
    conf = bool(dec.loc["подтверждение над S-RCK", "passed"])
    assert str(dec.loc["RCD-DEC", "value"]) == ("1" if ok and conf else "2" if bool(dec.loc["H-RCD1", "passed"]) and bool(dec.loc["У к W5-G", "passed"]) else "3" if bool(dec.loc["H-RCD1", "passed"]) else "4")
    for p, k in ((st, "verdict"),):
        for _, r in m.iterrows():
            conf_r = bool(r["delta_pct"] < 0 and r["p_holm_N"] < 0.05 and r["p_spa_c"] < 0.05)
            assert r[k].startswith("лучше") == conf_r and ("эквивал" not in r[k])
