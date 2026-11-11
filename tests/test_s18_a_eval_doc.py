"""S-A-EVAL, отчёт `S_A_REPORT.md` (`a_doc`, гейт E.G3; mutations.md № 125, 140, 158): числа отчёта = таблицам (подмена числа в тексте и в таблице ловится), первая строка — «исследовательский … не независимый OOS»,
таблица Ж-254 п. 16 первая, в каждом разделе с таблицами — даты и n, «подтверждено» — по СВОЕЙ строке и только при Δ < 0, p Холма (N = 831) < 0,05 и p SPA < 0,05, слова «лучше / хуже» ↔ знак Δ, решение = строка GUARD,
`--check` только читает. Таблицы — синтетика через боевые функции `a_eval_*` (D_M 2198 дат), параметры — сохранённые ряды A/*.parquet (только чтение); всё пишется во временную папку, реальные файлы только читаются (autouse, содержимое).
Только monkeypatch (№ 43)."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import a_doc as D
from models_v2.s18 import a_doc_fmt as FM
from models_v2.s18 import a_doc_s1 as S1
from models_v2.s18 import a_doc_s2 as S2
from models_v2.s18 import a_eval as E
from models_v2.s18 import a_eval_par as PAR
from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import a_eval_tab as T
from models_v2.s18 import a_eval_yr as Y

MARK_RE = re.compile(r"исследовательский.*не независимый OOS")
ROWS16 = ["HAR-J (K1)", "XGBoost (K2)", "LightGBM (K3)", "X1", "W5-G", "W8-G", "S-A0", "S-A1", "S-A2"]               # порядок п. 16 спецификации, независимая копия
KEY = {"written_at": "11.11.2026 18:00:00", "inputs": {"data/a/k1.parquet": "ab" * 32}, "code": {"a_eval.py": "cd" * 32}}


@pytest.fixture(autouse=True)
def real_files_untouched():
    def st():
        ps = [E.CKPT, E.R.TRIALS] + [E.REP / n for n in E.OUTS] + [E.REP / "S_A_REPORT.md"] + sorted(E.A_DIR.glob("*.parquet"))
        return {str(p): p.read_bytes() for p in ps if p.exists() and p.is_file()}
    before = st()
    yield
    assert st() == before


def synth_tables() -> tuple:
    n = 2198
    rng = np.random.default_rng(4)
    tp = pd.bdate_range("2018-01-04", periods=n)
    Lv = np.exp(rng.normal(-8, 0.3, n))
    rv = Lv * np.exp(rng.normal(0, 0.45, n) + 0.1)

    def q(mult, s):
        return np.exp(np.random.default_rng(s).normal(0, 0.1, n)) * Lv * mult
    F = {"K1": q(1.9, 5), "K2": q(1.7, 3), "K3": q(1.8, 4), "X1": q(1.45, 1), "W5-G": q(1.22, 2), "W8-G": q(1.25, 6), "B_mix": q(1.6, 7), "ZB": q(1.4, 8)}
    for i, c in enumerate(T.CANDS + ("S-A2p",)):
        F[c] = q(1.32 + 0.01 * i, 10 + i)
        F["X1_A(%s)" % c] = q(1.24 + 0.01 * i, 20 + i)
    ls = T.losses(rv, F)
    spa = T.spa_blocks(ls)
    spa["K2"] = {**spa["K2"], "p_c": 0.5}                                                  # SPA семейства K2 не значим при значимом Холме строк: есть строки «Холм да, SPA нет»
    spa["W5-G"] = {**spa["W5-G"], "p_c": 0.01}                                             # SPA семейства W5-G значим при Холме 1,0: строка «SPA да, Холма нет» (Δ < 0 у S-A0)
    frames = {v: pd.read_parquet(E.A_DIR / f) for v, f in E.VFILES.items()}
    t = dict(table=T.table(rv, tp, F), stats=T.stats(rv, tp, F, ls, spa), years=Y.years(tp, ls), conc=Y.conc(tp, ls), spa=T.spa_table(spa), power=Y.power_table(tp, ls), params=PAR.params(frames))
    t["guard"] = Y.guard(t["table"], t["years"], t["conc"])
    t["mcs"] = S.mcs(rv, {k: F[k] for k in ("S-A0", "S-A1", "S-A2", "W5-G", "X1", "W8-G", "K2", "K3", "B_mix")})
    return t, tp


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    t, tp = synth_tables()
    root = tmp_path_factory.mktemp("sa_doc")
    rep = root / "harj_struct"
    rep.mkdir()
    (root / "harj_x1").mkdir()
    for k in FM.KEYS:
        (rep / ("S_A_%s.csv" % k.upper())).write_text(E.head(tp) + t[k].to_csv(index=False, lineterminator="\n", float_format="%.12g"), encoding="utf-8")
    (root / "harj_x1/CHECKPOINT.json").write_text(json.dumps({"S.A.eval": KEY}, indent=1, sort_keys=True), encoding="utf-8")
    T1 = FM.read_all(rep)
    (rep / "S_A_REPORT.md").write_text(D.build(T1, KEY), encoding="utf-8")
    return root, tp


@pytest.fixture
def world(base, tmp_path, monkeypatch):
    root, tp = base
    w = tmp_path / "w"
    shutil.copytree(root, w)
    rep = w / "harj_struct"
    monkeypatch.setattr(D, "REP", rep)
    monkeypatch.setattr(D, "REPORT", rep / "S_A_REPORT.md")
    monkeypatch.setattr(D, "CKPT", w / "harj_x1/CHECKPOINT.json")
    orig = FM.read_all
    monkeypatch.setattr(FM, "read_all", lambda rep_=rep: orig(rep_))
    return rep, tp


def tamper(path: Path, old: str, new: str, count: int = 1) -> None:
    s = path.read_text(encoding="utf-8")
    assert old in s, old
    path.write_text(s.replace(old, new, count), encoding="utf-8")


def sections(text: str) -> list:
    return text.split("\n## ")[1:]


def test_honest_report_passes_every_check_and_marks(world) -> None:
    rep, tp = world
    text = (rep / "S_A_REPORT.md").read_text(encoding="utf-8")
    assert D.check() == [] and MARK_RE.search(text.splitlines()[0]) and text.splitlines()[0] == D.MARK
    secs = sections(text)
    assert secs[0].startswith("1. Таблица") and len(secs) == 8                              # таблица Ж-254 п. 16 — первой, 8 разделов в порядке спецификации
    names = [ln.split(" | ")[0].lstrip("| ").strip() for ln in secs[0].split("\n\n")[2].splitlines()[2:]]
    idx = [next(i for i, n in enumerate(names) if n.startswith(k)) for k in ROWS16]
    assert idx == sorted(idx) and "S-A2p" in " ".join(names)
    line = "D_M: t⁺ %s…%s, n = %d." % (tp.min().date(), tp.max().date(), len(tp))
    for s in secs:
        if any(ln.startswith("| ") for ln in s.splitlines()):
            assert line in s.split("\n| ")[0], s[:20]                                      # даты и n перед таблицами каждого раздела
    s = pd.read_csv(rep / "S_A_STATS.csv", comment="#")
    assert "Независимое подтверждение: нет" in text and D.confirmed(FM.read_all(rep)) == int(stats_confirmed(s).sum()) >= 1


def test_substituted_number_in_text_or_table_is_caught(world) -> None:
    rep, _ = world
    t = pd.read_csv(rep / "S_A_TABLE.csv", comment="#")
    q = "%.6f" % float(t[t["series"] == "X1"]["qlike"].iloc[0])
    rp = rep / "S_A_REPORT.md"
    assert (",".join(q.split("."))) in rp.read_text(encoding="utf-8")
    tamper(rp, q.replace(".", ","), q.replace(".", ",")[:-1] + ("0" if q[-1] != "0" else "1"))        # число в тексте подменено
    assert any("≠ перестроению" in b for b in D.check())


def test_substituted_number_in_table_with_old_report_is_caught(world) -> None:
    rep, _ = world
    p = rep / "S_A_STATS.csv"
    s = pd.read_csv(p, comment="#")
    row = s[(s["kind"] == "main") & (s["hypothesis"] == "S-A2|X1")].iloc[0]
    d = "%.12g" % row["delta_pct"]
    tamper(p, "," + d + ",", ",-99.5,")                                                    # Δ в таблице подменено, отчёт прежний
    assert any("≠ перестроению" in b for b in D.check())


def test_first_line_and_order_of_first_table_mutations_are_caught(world) -> None:
    rep, _ = world
    rp = rep / "S_A_REPORT.md"
    keep = rp.read_text(encoding="utf-8")
    rp.write_text(keep.replace(D.MARK, "результат", 1), encoding="utf-8")
    assert any("первая строка" in b for b in D.check())
    rp.write_text("\n" + keep, encoding="utf-8")                                           # пометка не в первой строке
    assert any("первая строка" in b for b in D.check())
    rp.write_text(keep, encoding="utf-8")
    parts = keep.split("\n## ")
    assert len(parts) == 9
    rp.write_text("\n## ".join([parts[0], parts[2], parts[1]] + parts[3:]), encoding="utf-8")       # разделы 1 и 2 поменяны: статистика первой
    assert any("первой не таблица" in b for b in D.check())
    lines = keep.split("\n")
    i, j = (next(k for k, ln in enumerate(lines) if ln.startswith(p)) for p in ("| X1 (контрольная версия) |", "| S-A0 (новый гибрид) |"))
    lines[i], lines[j] = lines[j], lines[i]                                                # ряд S-A0 стоит перед X1: порядок п. 16 нарушен
    rp.write_text("\n".join(lines), encoding="utf-8")
    assert any("порядке п. 16" in b for b in D.check())


def test_dates_and_n_line_removed_from_a_section_is_caught(world) -> None:
    rep, tp = world
    rp = rep / "S_A_REPORT.md"
    text = rp.read_text(encoding="utf-8")
    line = "D_M: t⁺ %s…%s, n = %d." % (tp.min().date(), tp.max().date(), len(tp))
    sec6 = text.split("\n## 6. ")[1].split("\n## 7.")[0]
    rp.write_text(text.replace(sec6, sec6.replace(line + " ", "", 1), 1), encoding="utf-8")
    assert any("без строки D_M" in b for b in D.check())


def stats_confirmed(s: pd.DataFrame) -> pd.Series:
    m = (s["kind"] == "main")
    return m & (s["delta_pct"] < 0) & (s["p_holm_N"] < 0.05) & (s["p_spa_c"] < 0.05)


def test_confirmed_only_on_own_row_with_holm_and_spa_below_005(world) -> None:
    rep, _ = world
    s = pd.read_csv(rep / "S_A_STATS.csv", comment="#")
    m = s[s["kind"] == "main"]
    ok = m[stats_confirmed(s)[m.index]]
    holm_only = m[(m["delta_pct"] < 0) & (m["p_holm_N"] < 0.05) & (m["p_spa_c"] >= 0.05)]
    spa_only = m[(m["delta_pct"] < 0) & (m["p_holm_N"] >= 0.05) & (m["p_spa_c"] < 0.05)]
    assert len(ok) >= 1 and len(holm_only) >= 1 and len(spa_only) >= 1                     # предусловие: все три случая есть, иначе тест пуст
    txt = (rep / "S_A_REPORT.md").read_text(encoding="utf-8")
    lines = [ln for ln in txt.splitlines() if ln.startswith("| ") and "подтверждено" in ln.replace("не подтверждено", "")]
    assert len(lines) == len(ok) and D.check() == []
    rp = rep / "S_A_REPORT.md"
    for row in (holm_only.iloc[0], spa_only.iloc[0]):                                      # «подтверждено» при невыполненном одном из двух p
        keep = txt
        bad = [ln for ln in txt.splitlines() if ln.startswith("| ") and row["verdict"] in ln][0]
        rp.write_text(txt.replace(bad, bad.replace(row["verdict"], "численно лучше базы; подтверждено (p Холма N < 0,05 и p SPA < 0,05)"), 1), encoding="utf-8")
        assert any("подтверждено" in b for b in D.check())
        rp.write_text(keep, encoding="utf-8")


def test_confirmation_moved_to_another_row_with_same_count_is_caught_by_row(world) -> None:
    rep, _ = world
    p = rep / "S_A_STATS.csv"
    s = pd.read_csv(p, comment="#")
    m = s[s["kind"] == "main"]
    ok = m[stats_confirmed(s)[m.index]].iloc[0]
    no = m[(m["delta_pct"] >= 0)].iloc[0]                                                  # строка «численно не лучше»: подтверждения быть не может
    head = [ln for ln in p.read_text(encoding="utf-8").splitlines(keepends=True) if ln.startswith("#")]
    s.loc[s["hypothesis"].eq(ok["hypothesis"]) & s["kind"].eq("main"), "verdict"] = no["verdict"]                      # вердикты двух строк поменяны местами — число «подтверждено» прежнее
    s.loc[s["hypothesis"].eq(no["hypothesis"]) & s["kind"].eq("main"), "verdict"] = ok["verdict"]
    p.write_text("".join(head) + s.to_csv(index=False, lineterminator="\n", float_format="%.12g"), encoding="utf-8")
    rp = rep / "S_A_REPORT.md"
    rp.write_text(D.build(FM.read_all(rep), KEY), encoding="utf-8")
    assert any("подтверждено" in b or "своей строке" in b for b in D.check())                  # по своей строке, а не по счёту строк в таблице


def parse_ablation_rows(text: str) -> list:
    sec = text.split("\n## 4. ")[1].split("\n## 5.")[0]
    out = []
    for ln in sec.splitlines():
        c = [x.strip() for x in ln.strip("| ").split(" | ")]
        if len(c) == 8 and c[-1] in ("лучше", "хуже", "равно"):
            out.append((float(c[4].replace(",", ".")), c[-1]))
    return out


def test_better_worse_words_match_delta_sign_and_flip_is_caught(world, monkeypatch) -> None:
    rep, _ = world
    text = (rep / "S_A_REPORT.md").read_text(encoding="utf-8")
    rows = parse_ablation_rows(text)
    assert len(rows) == 8 and {w for _, w in rows} >= {"лучше", "хуже"}                    # предусловие: есть оба слова
    assert all(w == ("лучше" if d < 0 else "хуже" if d > 0 else "равно") for d, w in rows)
    orig = S2.word
    monkeypatch.setattr(S2, "word", lambda d: "хуже" if d < 0 else "лучше")               # мутация: слова перевёрнуты
    flipped = D.build(FM.read_all(rep), KEY)
    assert not all(w == ("лучше" if d < 0 else "хуже" if d > 0 else "равно") for d, w in parse_ablation_rows(flipped))
    (rep / "S_A_REPORT.md").write_text(flipped, encoding="utf-8")
    monkeypatch.setattr(S2, "word", orig)
    assert any("≠ перестроению" in b for b in D.check())                                   # подменённый текст не равен честной сборке


def test_decision_equals_guard_row_and_substituted_decision_is_caught(world) -> None:
    rep, _ = world
    g = pd.read_csv(rep / "S_A_GUARD.csv", comment="#")
    dec = g[g["id"] == "решение"]["value"].iloc[0]
    text = (rep / "S_A_REPORT.md").read_text(encoding="utf-8")
    assert "**" + dec + "**" in text and Y.decide(g) == dec
    p = rep / "S_A_GUARD.csv"
    tamper(p, dec, "новая контрольная версия: S-A2 (решение разработки, не «подтверждено»); X1 и W5-G остаются контрольными версиями; W8-G — экспериментальная" if not dec.startswith("новая") else "не прошёл — S-A2: У3-S|М1 (+9.0000)")
    assert any("≠" in b or "решение" in b for b in D.check())                              # строка «решение» подменена при прежнем отчёте


def test_check_reads_only_hash_of_outputs_before_and_after(world) -> None:
    rep, _ = world
    files = sorted(rep.glob("S_A_*")) + [D.CKPT]
    h0 = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    assert D.check() == []
    assert {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files} == h0


def test_missing_eval_key_refuses_to_build_or_check(world) -> None:
    rep, _ = world
    D.CKPT.write_text(json.dumps({"прочий": 1}), encoding="utf-8")
    with pytest.raises(SystemExit):
        D.check()
