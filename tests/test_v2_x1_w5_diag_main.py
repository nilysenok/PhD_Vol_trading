"""W5-A, `x1_w5_diag` (+ `_t`, `_v`): пересчёт = `W5_DIAG.csv` и `W5_TABLE0.csv` побитно; метка gap = 1[Δ_d > 0] на реальном календаре; PIT-терцили и PIT-события (№ 153, 35);
вердикт выведен из CSV; `--check` только читает (№ 123); таблица 0 и шум — известный ответ на синтетике. Реальные файлы только читаются; подмены — monkeypatch и копии в tmp."""
from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_w5_diag as W
from models_v2.s16 import x1_w5_diag_t as T
from models_v2.s16 import x1_w5_diag_v as V

ROOT = W.ROOT
REP, REG = ROOT / "docs/otchety/harj_x1", ROOT / "data/regen/harj_x1"
REAL = (W.OUT, W.OUT0, REP / "ALL_TRIALS.csv", REP / "CHECKPOINT.json")


def tree_state() -> dict:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for d in (REP, REG) for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.fixture(autouse=True)
def real_files_untouched():
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in REAL if p.exists()}
    w5 = sorted(str(p) for p in (REG / "W5").rglob("*")) if (REG / "W5").exists() else []
    yield
    assert {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in REAL if p.exists()} == before        # реальные DIAG, TABLE0, ALL_TRIALS, CHECKPOINT прежние
    assert (sorted(str(p) for p in (REG / "W5").rglob("*")) if (REG / "W5").exists() else []) == w5 == []  # W5 без прогнозов


@pytest.fixture(scope="module")
def real():
    x, st = W.build()
    return x, st, W.compute(x, st)


def data_lines(p) -> list:
    return [ln for ln in Path(p).read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]


def test_recompute_equals_files_bitwise_and_block_sizes(real) -> None:
    x, st, df = real
    assert W.diag_lines(df) == data_lines(W.OUT)                                          # A.G1: W5_DIAG.csv
    assert T.table(*T.series(x)).to_csv(index=False, lineterminator="\n").splitlines() == data_lines(W.OUT0)       # W5_TABLE0.csv
    assert df["block"].value_counts().to_dict() == {"A_loss": 210, "B_level": 90, "B_trust": 30, "C_event": 84, "D_gap_posthoc": 93, "E_noise": 48}     # G2 спецификации
    assert len(x) == 2198 and x["t_plus"].min() == pd.Timestamp("2018-01-04") and x["t_plus"].max() == pd.Timestamp("2026-11-10")


def test_table0_rows_equal_w1_values(real) -> None:
    t = T.table(*T.series(real[0]))
    q = {k: t[(t["series"] == k) & (t["window"] == "M")]["qlike"].iloc[0] for k in ("K2", "X1", "ZB", "ZB0", "cfg8")}
    assert q == pytest.approx({"K2": 0.218812, "X1": 0.216427, "ZB": 0.213328, "ZB0": 0.217287, "cfg8": 0.220194}, abs=5e-7)      # G4 спецификации
    m = t[t["window"] == "M"].set_index("series")
    assert m.loc["M+S1", "d_X1_pct"] == pytest.approx(-0.69, abs=0.005) and m.loc["X1", "d_X1_pct"] == 0 and m.loc["K2", "d_K2_pct"] == 0


def test_series_stops_when_cfg8_differs_from_column_c(real) -> None:
    x = real[0].copy()
    x["C"] = x["C"] * 1.001                                                               # мутация: колонка C не равна cfg8
    with pytest.raises(SystemExit, match="cfg8"):
        T.series(x)
    x = real[0].copy()
    x.loc[5, "rv"] *= 1.5
    with pytest.raises(SystemExit, match="RV"):
        T.series(x)                                                                       # RV окна ≠ RV прогнозов W4


# ---------- метка gap ----------

def test_gap_label_is_delta_positive_on_real_calendar(real) -> None:
    x, st, _ = real
    delta = ((x["t_plus"] - x["d"]).dt.days - 1).to_numpy()
    assert (st["gap"] == np.where(delta > 0, "1", "0")).all()                            # gap = 1[Δ_d > 0], Δ = (t⁺ − d) − 1 календарных дней
    row = lambda d: int(np.flatnonzero(x["d"] == d)[0])
    assert st["gap"][row("2022-02-22")] == "1"                                            # праздник 23.02 посреди недели: вторник → четверг
    assert st["gap"][row("2024-06-11")] == "1"                                            # 12 июня
    assert st["gap"][row("2022-02-18")] == "1"                                            # пятница → понедельник (Δ = 2) тоже gap
    assert st["gap"][row("2018-01-05")] == "1" and delta[row("2018-01-05")] == 3          # пятница → вторник после праздничной недели
    assert st["gap"][row("2022-02-24")] == "0"                                            # соседние рабочие дни без перерыва
    assert delta.min() == 0 and set(delta) >= {0, 1, 2, 3}


def test_gap_label_known_answer_synthetic_calendar() -> None:
    d = pd.to_datetime(["2024-01-09", "2024-01-12", "2024-02-21", "2024-05-06", "2024-05-07"])   # Вт, Пт, Ср, Пн, Вт
    tp = pd.to_datetime(["2024-01-10", "2024-01-15", "2024-02-26", "2024-05-07", "2024-05-10"])  # Ср, Пн(Δ=2), Пн (праздник 23.02: Δ=4), Вт, Пт (8–9 мая: Δ=2)
    g = (((tp - d).days - 1) > 0)
    assert list(g) == [False, True, True, False, True]
    from models_v2.s16 import x1_w5_gaps as G
    odd = G.odd_pairs(pd.DataFrame(dict(d=d, t_plus=tp)))
    assert list(odd["d"].astype(str)) == ["2024-02-21", "2024-05-07"]                    # пятница → понедельник исключена; Ср → Пн и Вт → Пт — перерывы не по выходным
    assert list(odd["delta_days"]) == [4, 2] and odd["nonworking_between"].iloc[1] == "2024-05-08 Wed|2024-05-09 Thu"


# ---------- PIT ----------

def terc_whole(v, minpast=60):
    """Мутация: пороги по всей выборке (с днём i и будущим)."""
    v = np.asarray(v, float)
    q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
    g = np.full(len(v), "na", object)
    for i in range(minpast, len(v)):
        g[i] = "T1" if v[i] <= q1 else ("T2" if v[i] <= q2 else "T3")
    return g


def test_terc_known_answer_na_and_strictly_before_day_i() -> None:
    g = W.terc_past(np.arange(100.0))
    assert (g[:60] == "na").all() and (g[60:] == "T3").all()                              # каждое новое значение выше всех прошлых
    v = [1.0, 2.0, 3.0, 2.5]
    assert W.terc_past(v, minpast=3)[3] == "T3"                                           # прошлое [1,2,3]: q2 = 2,33 < 2,5; с днём i порог был бы 2,5 → T2
    inc = np.quantile(np.asarray(v), [1 / 3, 2 / 3])
    assert not (2.5 > inc[1])                                                             # мутация «день i в пороге» дала бы другой ответ
    assert (W.terc_past([5.0] * 10, minpast=3)[3:] == "T1").all()                         # константа: v ≤ q1 → T1


def test_terc_pit_future_corruption_does_not_change_past_and_mutation_is_caught() -> None:
    rng = np.random.default_rng(1)
    v = rng.normal(size=400)
    bad = v.copy()
    bad[300:] = rng.normal(5.0, 3.0, size=100)                                            # порча будущего
    assert (W.terc_past(v)[:300] == W.terc_past(bad)[:300]).all()                         # группы прошлых дат прежние
    assert not (terc_whole(v)[:300] == terc_whole(bad)[:300]).all()                       # мутация «по всей выборке» меняет группы прошлого → тест её ловит
    assert not (W.terc_past(v)[300:] == W.terc_past(bad)[300:]).all()                     # а группы порченных дат — меняются


def test_real_terciles_do_not_see_future(real) -> None:
    x, st, _ = real
    lcm = np.log(x["C"] / x["M"]).to_numpy()
    k = 1500
    bad = lcm.copy()
    bad[k:] = bad[k:][::-1] * 3 + 1
    for name, f in (("tCM", lambda a: a), ("absCM", np.abs)):
        assert (W.terc_past(f(bad))[:k] == st[name][:k]).all()
    assert (st["tCM"][:60] == "na").all() and (st["tCM"][60:] != "na").all()


def fake_cal(tables):
    def _cal(name):
        c = tables.get(name, pd.DataFrame(dict(date=[], known_from=[])))
        c = c.copy()
        c["date"], c["known_from"] = pd.to_datetime(c["date"]), pd.to_datetime(c["known_from"])
        c["known_to"] = pd.to_datetime(c["known_to"]) if "known_to" in c else pd.NaT
        return c
    return _cal


def x3():
    return pd.DataFrame(dict(d=pd.to_datetime(["2024-01-09", "2024-01-12", "2024-01-15"]), t_plus=pd.to_datetime(["2024-01-10", "2024-01-15", "2024-01-16"])))


TABLES = {
    "cbr.csv": pd.DataFrame(dict(date=["2024-01-10", "2024-01-15", "2024-01-16"], known_from=["2024-01-01", "2024-01-01", "2024-01-16"], known_to=[None, "2024-01-12", None])),
    "fomc.csv": pd.DataFrame(dict(date=["2024-01-13", "2024-01-10", "2024-01-16", "2024-01-15"], known_from=["2024-01-01"] * 4)),
    "cpi_us.csv": pd.DataFrame(dict(date=["2024-01-10", "2024-01-16"], known_from=["2024-01-10", "2024-01-01"])),
    "expirations.csv": pd.DataFrame(dict(date=["2024-01-10"], known_from=["2024-01-01"], known_to=["2024-01-09"])),
}


def test_events_pit_and_fomc_interval_known_answer(monkeypatch) -> None:
    monkeypatch.setattr(W, "_cal", fake_cal(TABLES))
    ev = W.events(x3())
    assert list(ev["ev_cbr"]) == [True, False, False]       # строка 1: версия 15.01 заменена (known_to 12.01 ≤ d); строка 2: известна только с t⁺ (known_from > d, № 35)
    assert list(ev["ev_fomc"]) == [False, True, True]       # FOMC в [d, t⁺): дата = t⁺ не считается, дата = d считается
    assert list(ev["ev_cpius"]) == [False, False, True]     # CPI с known_from = 10.01 > d = 09.01 не отдаётся на d
    assert list(ev["ev_exp"]) == [False, False, False]      # версия снята до d (known_to ≤ d)
    assert list(ev["ev_rebal"]) == [False] * 3 and list(ev["ev_any"]) == [True, True, True]          # строка 0 — ЦБ, 1 — FOMC, 2 — FOMC и CPI


def test_events_mutations_future_known_from_and_fomc_at_t_plus_are_caught(monkeypatch) -> None:
    honest = fake_cal(TABLES)
    monkeypatch.setattr(W, "_cal", honest)
    base = W.events(x3())
    monkeypatch.setattr(W, "_known", lambda c, d: c["date"])                              # мутация: known_from/known_to игнорируются (событие из будущего отдано на d)
    mut = W.events(x3())
    assert list(mut["ev_cbr"]) != list(base["ev_cbr"]) and list(mut["ev_cpius"]) != list(base["ev_cpius"]) and list(mut["ev_exp"]) != list(base["ev_exp"])
    t2 = dict(TABLES, **{"fomc.csv": pd.DataFrame(dict(date=["2024-01-10"], known_from=["2024-01-01"]))})   # FOMC ровно в t⁺ строки 0
    monkeypatch.undo()
    monkeypatch.setattr(W, "_cal", fake_cal(t2))
    assert list(W.events(x3())["ev_fomc"]) == [False, False, False]                       # мутация «FOMC в день t⁺» дала бы True в строке 0


# ---------- вердикт из CSV ----------

def write_diag(p: Path, spec: dict) -> Path:
    """spec: {(block, state): {window: (d_ctrl, d_x1, 'a:1|b:2')}}"""
    rows = []
    for (blk, s), wins in spec.items():
        for w, (dc, dx, grp) in wins.items():
            rows.append(dict(block=blk, state=s, group=grp, window=w, d_state_vs_ctrl_pct=dc, d_state_vs_X1_pct=dx))
    p.write_text("# комментарий шапки\n" + pd.DataFrame(rows).to_csv(index=False), encoding="utf-8")
    return p


def passing(g1="a:1.000|b:2.000", g2="a:1.100|b:3.000"):
    return {"h1->h2": (-1.0, -1.0, "x:0"), "h2->h1": (-1.0, -1.0, "x:0"), "h1->h2_no2022": (-0.1, 0.0, "x:0"),
            "h1->h1_insample": (-1.0, -1.0, g1), "h2->h2_insample": (-1.0, -1.0, g2)}


def test_verdict_known_answer_each_criterion_blocks(tmp_path) -> None:
    bad_c2 = passing()
    bad_c2["h2->h1"] = (-1.0, 0.1, "x:0")
    bad_c3 = passing()
    bad_c3["h1->h2_no2022"] = (0.2, -1.0, "x:0")
    bad_c1 = passing()
    bad_c1["h1->h2"] = (0.0, -1.0, "x:0")                                                 # строго < 0: ноль не проходит
    spec = {("B_level", "ok"): passing(), ("B_level", "c1"): bad_c1, ("B_level", "c2"): bad_c2, ("B_level", "c3"): bad_c3,
            ("B_level", "dir"): passing(g2="a:3.000|b:1.000"), ("B_trust", "okT"): passing(g1="a:0.0|b:1.0|c:0.5", g2="a:0.1|b:0.9|c:0.4")}
    r, passed, total = V.verdict(write_diag(tmp_path / "D.csv", spec))
    assert passed == ["B_level/ok", "B_trust/okT"] and total == 6
    assert dict(zip(r["state"], r["PASS"])) == {"ok": True, "c1": False, "c2": False, "c3": False, "dir": False, "okT": True}
    assert not r.set_index("state").loc["dir", "direction"] and r.set_index("state").loc["dir", "h1->h2"]


def test_verdict_main_exit_code_needs_24_states(tmp_path, capsys) -> None:
    assert V.main(write_diag(tmp_path / "D.csv", {("B_level", "ok"): passing()})) == 1   # проверено 1, не 24
    out = capsys.readouterr().out
    assert "прошли: B_level/ok" in out and "всего проверено: 1" in out


def test_real_verdict_derived_from_csv_and_substituted_number_changes_it(tmp_path) -> None:
    r, passed, total = V.verdict(W.OUT)
    assert passed == ["B_level/gap"] and total == 24                                      # A.G2
    t = pd.read_csv(W.OUT, comment="#")
    m = (t["block"] == "B_level") & (t["state"] == "gap") & (t["window"] == "h2->h1")
    assert m.sum() == 1
    t.loc[m, "d_state_vs_X1_pct"] = 0.5                                                   # подмена числа в таблице
    bad = tmp_path / "W5_DIAG.csv"
    t.to_csv(bad, index=False)
    assert V.verdict(bad)[1] == []                                                        # вердикт выведен из CSV, а не записан в код
    t2 = pd.read_csv(W.OUT, comment="#")
    m2 = (t2["block"] == "B_level") & (t2["state"] == "dow") & (t2["window"].isin(["h1->h2", "h2->h1", "h1->h2_no2022"]))
    t2.loc[m2, ["d_state_vs_X1_pct", "d_state_vs_ctrl_pct"]] = -1.0
    t2.to_csv(bad, index=False)
    assert "B_level/dow" in V.verdict(bad)[1] or "B_level/gap" in V.verdict(bad)[1]      # у dow порядок групп в половинах может не совпасть, у gap — тот же
    assert "B_level/gap" in V.verdict(bad)[1]


def test_check_verdict_cli_prints_expected_lines() -> None:
    r = subprocess.run([sys.executable, "-W", "ignore", "-m", "models_v2.s16.x1_w5_diag", "--check-verdict"], cwd=ROOT, capture_output=True, text=True, timeout=50)
    assert r.returncode == 0 and "прошли: B_level/gap" in r.stdout and "всего проверено: 24" in r.stdout


# ---------- `--check` только читает ----------

def copies(tmp_path):
    d, t = tmp_path / "W5_DIAG.csv", tmp_path / "W5_TABLE0.csv"
    d.write_bytes(W.OUT.read_bytes())
    t.write_bytes(W.OUT0.read_bytes())
    return d, t


def test_check_ok_and_reads_only_hash_of_all_outputs(real) -> None:
    before = tree_state()
    assert len(before) > 100
    assert W.check() == []
    assert tree_state() == before                                                         # хэши всех файлов docs/otchety/harj_x1 и data/regen/harj_x1 равны


def test_check_catches_substituted_number_in_diag_and_in_table0(real, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(W, "build", lambda: real[:2])
    monkeypatch.setattr(W, "compute", lambda x, st: real[2])
    d, t = copies(tmp_path)
    assert W.check(d, t) == []
    lines = d.read_text(encoding="utf-8").splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("B_level,gap,") and "h1->h2," in ln)
    lines[i] = lines[i].replace("-", "+", 1)
    d.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert any("W5_DIAG" in b for b in W.check(d, t))
    d, t = copies(tmp_path)
    s = t.read_text(encoding="utf-8")
    t.write_text(s.replace("\nK2,", "\nK3,", 1), encoding="utf-8")                      # подмена в строке данных (в шапке-комментарии не считается)
    assert any("W5_TABLE0" in b for b in W.check(d, t))


def test_check_that_writes_is_caught_by_hash_inside_and_outside(real, tmp_path, monkeypatch) -> None:
    d, t = copies(tmp_path)
    monkeypatch.setattr(W, "build", lambda: real[:2])
    orig = W.compute

    def writing(x, st):
        d.write_text(d.read_text(encoding="utf-8") + "# проверка дописала\n", encoding="utf-8")                # мутация: check пишет
        return real[2]
    monkeypatch.setattr(W, "compute", writing)
    snap = hashlib.sha256(d.read_bytes()).hexdigest()
    assert any("изменил файл" in b for b in W.check(d, t))                               # встроенное сравнение sha до/после
    assert hashlib.sha256(d.read_bytes()).hexdigest() != snap                            # и внешний хэш выхода это видит
    assert orig.__name__ == "compute"


def write_calls(src: str) -> list:
    return [n.func.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("write_text", "write_bytes", "open", "to_csv", "dump")]


def func_src(src: str, name: str) -> str:
    return next(ast.unparse(f) for f in ast.parse(src).body if isinstance(f, ast.FunctionDef) and f.name == name)


def test_readers_have_no_write_calls_and_scanner_sees_mutation() -> None:
    src = Path(W.__file__).read_text(encoding="utf-8")
    for f in ("events", "terc_past", "build", "_data_lines", "_file_sha", "_cal", "_known"):
        assert write_calls(func_src(src, f)) == [], f
    assert [c for c in write_calls(func_src(src, "check")) if c != "to_csv"] == []        # to_csv в check — только в текст для сравнения, не в файл
    assert "write_text" in write_calls(func_src(src, "main"))                             # единственный писатель — main (флаг --write)
    assert write_calls(func_src(src.replace("    before = (_file_sha(diag)", "    Path(diag).write_text('x')\n    before = (_file_sha(diag)", 1), "check"))


# ---------- таблица 0 и шум: известный ответ ----------

def synth(n_per=30):
    n = 9 * n_per
    yr = 2018 + np.arange(n) // n_per
    x = pd.DataFrame(dict(rv=np.ones(n), t_plus=pd.to_datetime(["%d-06-15" % y for y in yr])))
    one = np.ones(n)
    s = {"K1": 3 * one, "K2": 2 * one, "K3": 4 * one, "M": 2 * one, "X1": 2 * one, "ZB": 4 * one, "ZB0": 2 * one, "cfg8": 2 * one}
    return x, s


def test_table0_known_answer_windows_and_signs() -> None:
    x, s = synth()
    t = T.table(x, s)
    ql = lambda f: 1 / f - np.log(1 / f) - 1
    r = t[(t["series"] == "ZB") & (t["window"] == "M")].iloc[0]
    assert r["n"] == 270 and r["qlike"] == pytest.approx(ql(4.0)) and r["d_X1_pct"] == pytest.approx(100 * (ql(4.0) / ql(2.0) - 1)) and r["d_K3_pct"] == pytest.approx(0.0)
    r = t[(t["series"] == "K1") & (t["window"] == "h1_2018_2021")].iloc[0]
    assert r["n"] == 120 and str(r["t_plus_from"]) == "2018-06-15" and str(r["t_plus_to"]) == "2021-06-15"
    assert t[(t["series"] == "X1") & (t["window"] == "h2_2022_2026")]["n"].iloc[0] == 150
    assert (t[t["series"] == "X1"]["d_X1_pct"] == 0).all() and (t[t["series"] == "K2"]["d_K2_pct"] == 0).all()


def test_noise_known_answer_constant_difference_has_zero_se() -> None:
    x, s = synth()
    nz = T.noise(x, s)
    ql = lambda f: 1 / f - np.log(1 / f) - 1
    r = nz[(nz["state"] == "ZB-X1") & (nz["window"] == "M")].iloc[0]
    assert r["block"] == "E_noise" and r["n"] == 270 and r["se_pct"] == pytest.approx(0.0, abs=1e-12)
    assert r["d_pct"] == pytest.approx(100 * (ql(4.0) - ql(2.0)) / ql(2.0)) and r["share_days_changed"] == 1.0
    assert nz[(nz["state"] == "ZB0-X1") & (nz["window"] == "2020")]["share_days_changed"].iloc[0] == 0.0       # ZB0 = X1: разность нулевая
    assert len(nz) == 4 * 12
