"""V7 лист 4a, G46: оценка конфигурации 10, условие 11, дрейф энкодера, `LEAF4_REPORT.md` целиком из таблиц, порядок записи CHECKPOINT (`4.eval` до таблиц с p; `4.cond11` до любого счёта 11), отказ 11, `--check` ничего не пишет.
Известные ответы: 10 лучше 6 ⇒ Δ < 0, правило 6.2 «да», cond11 истинно (мутации «пара перепутана», «окно 2022+» — ложно); выравнивание компонент z; потеря предлога нулевой сети = среднее y² по валидным барам.
Мутации: число в отчёте подменено / вписано вручную, p в `4.eval`, `4.eval` после таблиц, `4.cond11` после счёта 11, 11 при ложном условии, `cond_11` при ложном `4.cond11`, файл таблицы или отчёта изменён, диагностика без выравнивания. Синтетика, файлы SBER не читаются."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from models_v2.s14 import v7_0_io as IO, v7_1e_net as NET, v7_1e_pre as PRE, v7_1m_reg as RG, v7_2_ev as E2, v7_4_chk as CK, v7_4_dx as DX, v7_4_ev as E4, v7_4_evrep as REP, v7_4_evrun as R

IDX = pd.bdate_range("2018-01-03", "2026-02-26")
N = len(IDX)
W0 = "2026-11-09T19:00:00+03:00"


def make_world(better10=True):
    g = np.random.default_rng(2)
    rv = np.exp(g.normal(0, 0.5, N))
    s = {k: rv * np.exp(g.normal(0, 0.2, N)) for k in E2.CTRL + E4.CFGS}
    s["cfg10"] = rv * np.exp(g.normal(0, 0.1 if better10 else 0.5, N))
    pi = g.dirichlet([8, 8, 8], N)
    a18 = np.asarray(IDX < pd.Timestamp("2019-01-01"))
    df10 = pd.DataFrame({"d": IDX, "t_plus": IDX, "rv": rv, "H": s["K1"], "G": np.log(s["cfg10"] / s["K1"]), "F": s["cfg10"], "fitted": 1.0, "pi0": pi[:, 0], "pi1": pi[:, 1], "pi2": pi[:, 2],
                         "g1": g.normal(0, 0.1, N), "g2": g.normal(0, 0.1, N), "stageA_2018": a18.astype(float)})
    inp = dict(idx=IDX, rv=rv, tp=IDX.to_numpy("datetime64[ns]"), series=s, df10=df10, shas={"V6:A0": "x"})
    t = R.compute(inp)
    return inp, t


def drift_frame():
    g = np.random.default_rng(4)
    rows = [dict(year=y, seed=s, pretext_A_train=0.5 + 0.01 * s, pretext_B_train=0.45, pretext_A_year=0.6, pretext_B_year=0.62 + 0.01 * (y - 2019), corr1=0.9, corr2=0.8, corr3=0.7, corr4=0.6,
                 corr_mean_year=0.75 - 0.01 * (y - 2019), corr_mean_past=0.8) for y in range(2019, 2027) for s in range(5)]
    return pd.DataFrame(rows)[DX.COLS].assign(pretext_A_year=lambda d: d["pretext_A_year"] + g.normal(0, 1e-3, len(d)))


@pytest.fixture(scope="module")
def world():
    inp, t = make_world()
    cp4 = dict(rules=R.rules(t), cond11=dict(E4.cond11(t["PAIRS"]), written_at=W0), cfg10=dict(lam_e=0.01, stageA_2018_rows=int(inp["df10"]["stageA_2018"].sum()), written_at=W0))
    return inp, t, cp4


def test_pair_10_vs_6_known_answer_cond11_true_and_mutations_are_caught(world, monkeypatch):
    inp, t, _ = world
    p = t["PAIRS"]
    r = p[p["window"] == "2018_2021"].iloc[0]
    assert (r["x"], r["y"]) == ("cfg10", "cfg6") and r["delta_q"] < 0 and r["delta_pct"] < 0 and r["rule62"] == "да" and p[p["window"] == "2022_2025"]["rule62"].iloc[0] == "—"
    assert list(p["y"].unique()) == ["cfg6", "cfg8"] and E4.cond11(p)["cond11"] is True and E4.cond11(p)["n_years"] == 4
    monkeypatch.setattr(E4, "PAIRS10", (("H10 совместное", "cfg6", "cfg10"),) + E4.PAIRS10[1:])                  # мутант: пара перепутана
    q = E2.pairs_table(inp["rv"], inp["series"], inp["tp"], E4.PAIRS10)
    assert q[q["window"] == "2018_2021"].iloc[0]["rule62"] == "нет"
    monkeypatch.setattr(E4, "COND_WIN", "2022_2025")                                                              # мутант: окно 2022+ (правило там «—»)
    assert E4.cond11(p)["cond11"] is False


def test_cond11_false_when_10_is_worse_than_6():
    _, t = make_world(better10=False)
    assert E4.cond11(t["PAIRS"])["cond11"] is False and t["PAIRS"][t["PAIRS"]["window"] == "2018_2021"].iloc[0]["delta_q"] > 0


def test_weights_only_stage_b_rows_and_pi_sums_to_one(world):
    inp, t, _ = world
    w = t["WEIGHTS"]
    assert (w[w["period"] == "2018"]["n"] == 0).all() and np.isnan(w[w["period"] == "2018"]["mean"]).all()          # 2018 — прогноз Stage A, в веса не входит
    assert np.isclose(w[w["period"] == "2020"]["mean"].sum(), 1.0) and set(w["quantity"]) == {"pi0", "pi1", "pi2"}
    df = inp["df10"].copy()
    df.loc[df["t_plus"].dt.year == 2020, "fitted"] = 0.0
    assert E4.weights_table(df).query("period == '2020'")["n"].eq(0).all()


def test_numbers_have_no_pvalues_and_json_roundtrip(world):
    inp, t, _ = world
    n = R.numbers(t, inp)
    dump = json.dumps(n, ensure_ascii=False)
    assert "p_one" not in dump and "p_holm" not in dump and json.loads(dump) == n
    assert any(k.startswith("H10 совместное|cfg10|cfg6|") for k in n["pairs"]) and len(n["qlike"]) == 18 and set(n["go_cfg10"]) == set("0123456") and "pi0|2020" in n["weights"]


def test_render_deterministic_contains_numbers_and_every_cell_changes_text(world):
    inp, t, cp4 = world
    dr = drift_frame()
    base = REP.render(t, inp, cp4, dr)
    assert base == REP.render(t, inp, cp4, dr) and base.count("\n") > 60 and "потолок" in base and "11 здесь не считается" in base
    v = REP.f(t["METRICS"].query("series == 'K2' and window == '2018_2021'")["qlike"].iloc[0], 6)
    assert v in base and base.replace(v, v[:-1] + ("1" if v[-1] != "1" else "2"), 1) != base
    for key, col in (("METRICS", "qlike"), ("TESTS", "delta_pct"), ("TESTS", "p_one"), ("PAIRS", "delta_pct"), ("PAIRS", "p_holm_N"), ("PAIRS", "delta_q"), ("GO", "value"), ("MCS", "mean_qlike"), ("STRESS", "qlike")):
        t2 = {k: x.copy() for k, x in t.items()}
        t2[key][col] = t2[key][col].astype(float) + 0.123456
        assert REP.render(t2, inp, cp4, dr) != base, (key, col)
    for col in ("pretext_B_year", "corr_mean_year", "corr1"):                                                       # числа дрейфа — из таблицы
        assert REP.render(t, inp, cp4, dr.assign(**{col: dr[col] + 0.0123})) != base, col
    assert REP.render(t, inp, dict(cp4, cond11=dict(cp4["cond11"], p_one=0.777)), dr) != base


class Fake:
    def __init__(self):
        self.events, self.cp, self.added = [], {}, []

    def update_4(self, sub, val):
        self.events.append("cp:" + sub)
        self.cp[sub] = json.loads(json.dumps(val))


def patch_all(monkeypatch, tmp_path, fk, inp):
    fk.cp["cfg10"] = dict(lam_e=0.01, stageA_2018_rows=int(inp["df10"]["stageA_2018"].sum()), written_at=W0)
    monkeypatch.setattr(R, "load", lambda: inp)
    monkeypatch.setattr(R.R4, "update_4", fk.update_4)
    monkeypatch.setattr(R.CR, "read_cp", lambda: {"4": fk.cp})
    monkeypatch.setattr(R, "drift_frame", drift_frame)
    monkeypatch.setattr(R, "drift_spot", lambda dr, *a: [])                                   # пересчёт строки дрейфа — на данных SBER, здесь заглушён (проверяется в боевом `--check`)
    monkeypatch.setattr(R.RG, "add_result", lambda cfg, payload, suffix=None, **k: fk.added.append((cfg, suffix)) or "дописана")
    monkeypatch.setattr(IO, "REP", tmp_path)
    orig = IO.write_csv
    monkeypatch.setattr(IO, "write_csv", lambda df, name, note, status=None: fk.events.append("csv:" + name) or orig(df, name, note, status))
    monkeypatch.setattr(IO, "code_hash", lambda: "c")
    monkeypatch.setattr(IO, "inputs_hash", lambda: "i")


@pytest.fixture
def written(world, monkeypatch, tmp_path):
    fk = Fake()
    patch_all(monkeypatch, tmp_path, fk, world[0])
    assert R.write_eval() == 0
    return fk, tmp_path


def test_eval_before_tables_rules_after_cond11_after_rules_and_true_cond_does_not_write_not_run_row(written):
    ev = written[0].events
    assert ev[0] == "cp:eval" and all(ev.index("cp:eval") < ev.index("csv:" + n) for n in R.NAMES.values()) and ev.index("csv:4_PAIRS.csv") < ev.index("cp:rules") < ev.index("cp:cond11") < ev.index("csv:4_DRIFT.csv") < ev.index("cp:drift")
    assert written[0].added == [] and written[0].cp["cond11"]["cond11"] is True


def test_false_cond11_writes_not_run_row_for_11_with_own_suffix(monkeypatch, tmp_path):
    inp, _ = make_world(better10=False)
    fk = Fake()
    patch_all(monkeypatch, tmp_path, fk, inp)
    R.write_eval()
    assert fk.cp["cond11"]["cond11"] is False and fk.added == [("11", RG.NOT_RUN11_SUFFIX)]
    assert RG.SUFFIX_STATUS[RG.NOT_RUN11_SUFFIX] == "не считается: условие ложно (4.cond11)" and RG.NOT_RUN11_SUFFIX != RG.NOT_RUN_SUFFIX


def test_mutation_eval_after_tables_is_caught(world, monkeypatch, tmp_path):
    fk = Fake()
    patch_all(monkeypatch, tmp_path, fk, world[0])
    monkeypatch.setattr(R.R4, "update_4", lambda sub, val: fk.update_4(sub, val) if sub != "eval" else None)
    monkeypatch.setattr(R.REP, "render", lambda *a: "")
    R.write_eval()
    fk.update_4("eval", {})
    ev = fk.events
    assert not (ev[0] == "cp:eval" and all(ev.index("cp:eval") < ev.index("csv:" + n) for n in R.NAMES.values()))


def test_check_eval_ok_on_written_catches_changes_and_writes_nothing(written, monkeypatch):
    fk, d = written
    digest = lambda: hashlib.sha256(b"".join(p.read_bytes() for p in sorted(d.glob("*")))).hexdigest()
    before = digest()
    monkeypatch.setattr(IO, "write_csv", lambda *a, **k: (_ for _ in ()).throw(AssertionError("check пишет таблицу")))
    assert R.check_eval({"4": fk.cp}) == [] and digest() == before
    rep = d / "LEAF4_REPORT.md"
    txt = rep.read_text(encoding="utf-8")
    rep.write_text(txt.replace("0.1", "0.2", 1), encoding="utf-8")
    assert any("LEAF4_REPORT" in b for b in R.check_eval({"4": fk.cp}))
    rep.write_text(txt, encoding="utf-8")
    tab = d / "4_PAIRS.csv"
    raw = tab.read_text(encoding="utf-8")
    tab.write_text(raw.replace("да", "нет", 1) if "да" in raw else raw.replace("нет", "да", 1), encoding="utf-8")
    assert any("4_PAIRS.csv" in b for b in R.check_eval({"4": fk.cp}))
    tab.write_text(raw, encoding="utf-8")
    dr = d / R.DRIFT
    draw = dr.read_text(encoding="utf-8")
    dr.write_text(draw.replace("0.62", "0.63", 1), encoding="utf-8")
    assert any("4_DRIFT.csv" in b for b in R.check_eval({"4": fk.cp}))
    dr.write_text(draw, encoding="utf-8")
    assert any("eval" in b for b in R.check_eval({"4": {k: v for k, v in fk.cp.items() if k != "eval"}}))
    monkeypatch.setattr(R, "drift_spot", lambda dr, *a: ["4_DRIFT.csv: строка не равна пересчёту"])
    assert any("не равна пересчёту" in b for b in R.check_eval({"4": fk.cp}))                                         # пересчёт строки дрейфа расходится — ловится
    monkeypatch.setattr(R, "drift_spot", lambda dr, *a: [])
    assert any("cond11" in b for b in R.check_eval({"4": dict(fk.cp, cond11=dict(fk.cp["cond11"], cond11=False))}))
    assert any("порядок" in b for b in R.check_eval({"4": dict(fk.cp, cond11=dict(fk.cp["cond11"], written_at="2000-01-01T00:00:00+03:00"))}))


def test_order_cond11_before_11_and_false_condition_blocks_11_and_cond_11():
    ok = dict(cond11=dict(cond11=True, written_at="2026-11-09T20:00:00+03:00"), cfg11=dict(written_at="2026-11-09T21:00:00+03:00"))
    assert CK.order_problems({"4": ok}) == []
    late = dict(ok, cond11=dict(ok["cond11"], written_at="2026-11-09T22:00:00+03:00"))                                # мутант: cond11 записан после счёта 11
    assert any("после счёта 11" in b for b in CK.order_problems({"4": late}))
    assert any("без 4.cond11" in b for b in CK.order_problems({"4": dict(cfg11=ok["cfg11"])}))
    false = dict(ok, cond11=dict(ok["cond11"], cond11=False))                                                         # мутант: 11 посчитана при ложном условии
    assert any("при ложном" in b for b in CK.order_problems({"4": false}))
    assert any("cond_11" in b for b in CK.order_problems({"4": dict(cond11=false["cond11"]), "cond_11": True}))      # допуск 11 при ложном условии
    assert CK.order_problems({"4": dict(cond11=false["cond11"])}) == []                                              # ложное условие без счёта 11 — норма


def test_runner_refuses_11_without_cond_11_and_mutation_without_key_is_caught(tmp_path, monkeypatch):
    trials, cp = tmp_path / "t.csv", tmp_path / "cp.json"
    RG.ensure(trials, "2026-11-09T00:00:00+03:00")
    cp.write_text(json.dumps({"4": {"cond11": {"cond11": False}}}), encoding="utf-8")
    with pytest.raises(RG.Refusal, match="cond_11"):
        RG.gate("11", trials, cp)
    cp.write_text(json.dumps({"cond_11": True}), encoding="utf-8")
    assert RG.gate("11", trials, cp)["key"] == "Ж-233:11"
    monkeypatch.setattr(RG, "CONDITION_KEY", {"9": "cond_9"})                                                         # мутант: условие 11 не проверяется
    cp.write_text(json.dumps({}), encoding="utf-8")
    assert RG.gate("11", trials, cp)["key"] == "Ж-233:11"                                                            # без правки это был бы отказ (выше)
    with pytest.raises(RG.Refusal, match="резерв"):
        RG.gate("12", trials, cp)


def test_check_run_requires_exactly_10_ok_and_11_refused_and_includes_eval_check(monkeypatch):
    monkeypatch.setattr(CK, "check", lambda cp: [])
    monkeypatch.setattr(CK.CR, "read_cp", lambda: {"4": {"eval": {}}})
    monkeypatch.setattr(CK.IO, "outputs_digest", lambda: "same")
    monkeypatch.setattr(R, "check_eval", lambda cp: [])
    assert CK.check_run(["10"], ["11"]) == 0
    assert CK.check_run(["10", "11"], []) == 1 and CK.check_run(["10"], []) == 1                                       # 11 допущена бегуном — ошибка
    monkeypatch.setattr(R, "check_eval", lambda cp: ["4.eval: пересчёт чисел ≠ CHECKPOINT"])
    assert CK.check_run(["10"], ["11"]) == 1
    seq = iter(["a", "b"])
    monkeypatch.setattr(R, "check_eval", lambda cp: [])
    monkeypatch.setattr(CK.IO, "outputs_digest", lambda: next(seq))
    assert CK.check_run(["10"], ["11"]) == 1                                                                           # выходы изменены проверкой


def test_aligned_corr_known_answer_permuted_and_flipped_components_and_no_alignment_mutation():
    g = np.random.default_rng(0)
    za = g.normal(size=(300, 4))
    zb = -za[:, [2, 0, 3, 1]] + 0.01 * g.normal(size=(300, 4))                                                        # перестановка и смена знака — тот же z
    assert (DX.aligned_corr(zb, za) > 0.99).all()
    diag = np.abs(np.corrcoef(za.T, zb.T)[:4, 4:]).diagonal()                                                         # мутант: без выравнивания
    assert (diag < 0.2).all()
    zc = zb.copy()
    zc[::7] = np.nan
    assert (DX.aligned_corr(zc, za) > 0.99).all()                                                                     # даты с NaN отбрасываются


def test_pretext_loss_known_answer_zero_network_equals_mean_squared_label():
    g = np.random.default_rng(1)
    n = 12
    pack = dict(X=g.normal(size=(n, 1, 53, 11)).astype(np.float32), mask=np.ones((n, 1, 53), bool), r2=np.abs(g.normal(size=(n, 53))) * 1e-3)
    base = NET.build("tcn").double()
    for p in base.parameters():
        p.data.zero_()
    rows = np.arange(n)
    y, ok = PRE.targets(pack, rows, 1e-4)
    exp = float((ok * y.astype(float) ** 2).sum() / ok.sum())
    assert np.isclose(DX.pretext_loss(base, pack, rows, 1e-4), exp, rtol=1e-6)
    base.head.bias.data.fill_(1.0)                                                                                    # сеть, ставящая 1: потеря = среднее (y − 1)²
    assert np.isclose(DX.pretext_loss(base, pack, rows, 1e-4), float((ok * (y.astype(float) - 1) ** 2).sum() / ok.sum()), rtol=1e-6)
    ref = PRE._loss(base, NET.to_seq(pack["X"], pack["mask"]).double(), torch.as_tensor(y).double(), torch.as_tensor(ok))
    assert np.isclose(DX.pretext_loss(base, pack, rows, 1e-4), float(ref), rtol=1e-6)


def test_by_year_means_over_seeds_and_differences():
    t = drift_frame()
    b = DX.by_year(t)
    assert set(b) == {str(y) for y in range(2019, 2027)} and np.isclose(b["2019"]["pretext_B_minus_A_train"], t[t["year"] == 2019]["pretext_B_train"].mean() - t[t["year"] == 2019]["pretext_A_train"].mean())
    assert np.isclose(b["2021"]["corr_mean_year"], 0.73)
