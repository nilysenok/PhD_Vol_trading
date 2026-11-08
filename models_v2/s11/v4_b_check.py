"""V4-B: проверки гейтов G9–G14 для `run_v4 --stage B --check`. Каждая функция возвращает список нарушений (пусто — гейт закрыт)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import v4_b_cache as CA, v4_b_data as D, v4_b_decl as DECL, v4_b_ops as OP, v4_b_rule as RU, v4_b_run as R, v4_b_stats as ST, v4_io as IO

CSVS = ["B5_RULE.csv", "B_COMPARISON.csv", "B_PER_TICKER.csv", "B_TESTS.csv", "B_MCS.csv", "B_WEIGHTS.csv", "B_SBER.csv", "B_LOTO.csv", "B_REPORT.md"]


def g9(o: dict) -> list:
    bad, cp = [], json.loads(DECL.CP.read_text())
    ct = DECL.controls(D.eval_frame(o["P"]))
    for k, v in ct.items():
        if abs(cp["B_controls_crit_panel"][k] - v) > 1e-12:
            bad.append("G9: контроль %s в CHECKPOINT не совпадает с crit_panel" % k)
    if abs(cp["go_threshold_B"] - (min(ct.values()) - S.MARGIN)) > 1e-12:
        bad.append("G9: go_threshold_B != min(контролей) − 0,0005")
    if cp["N_wide"] != cp["N_declared_before_stats"] + DECL.optuna_trials()[0] or "N_wide_rule" not in cp:
        bad.append("G9: N_wide / правило подсчёта")
    files = [IO.REP / f for f in CSVS if (IO.REP / f).exists()] + [IO.REP / "ALL_TRIALS.csv"] + sorted(CA.OUT.glob("B_*.parquet"))
    return bad + ["G9: файл создан не позже объявления порога: " + x for x in DECL.declared_before(files)]


def g10(o: dict) -> list:
    bad, P = [], o["P"]
    for cfg in o["cfgs"]:
        f = pd.read_parquet(CA.OUT / ("B_%s.parquet" % cfg))
        w = pd.read_parquet(CA.OUT / ("B_%s_weights.parquet" % cfg))
        if f["ticker"].nunique() != 17 or len(f) != len(P) or not (f[["ticker", "date"]].sort_values(["ticker", "date"]).to_numpy() == P[["ticker", "date"]].sort_values(["ticker", "date"]).to_numpy()).all():
            bad.append("G10: %s — состав бумаг/дат не равен H∩K2∩K3" % cfg)
        if not (np.isfinite(f["F"]) & (f["F"] > 0)).all() or not w["converged"].all():
            bad.append("G10: %s — нефинитные прогнозы или несошедшаяся оценка" % cfg)
        if R.pit_violation(cfg, P[P["date"] < "2019-07-01"], "2019-06"):
            bad.append("G10: %s — обрезка: RV после d меняет прогноз дня d" % cfg)
    return bad + identities(P)


def identities(P: pd.DataFrame) -> list:
    H, M = P["h"].to_numpy(), OP.mean_m(P["k2"], P["k3"])
    w = np.random.default_rng(1).uniform(0.05, 0.95, len(H))
    f = lambda c, **k: OP.forward(c, H, M, w, **k)[0]
    bad = []
    if not (np.allclose(f("B4", p=1.0), f("B1"), rtol=1e-10, atol=0) and np.allclose(f("B4", p=-1.0), f("B3"), rtol=1e-10, atol=0) and np.allclose(f("B4", p=1e-10), f("B2"), rtol=1e-10, atol=0)):
        bad.append("G10: тождества p -> 0, 1, −1 не сошлись на реальных H, M")
    one = np.ones(len(H))
    if not all(np.array_equal(OP.forward(c, H, M, one, p=0.5)[0], H) for c in ("B1", "B2", "B3", "B4")):
        bad.append("G10: w = 1 не даёт H побитно")
    return bad


def g11(o: dict) -> list:
    r, st = IO.read_csv("B5_RULE.csv"), RU.rule_stats(o["fc"]["B2"])
    trials = pd.read_csv(IO.REP / "ALL_TRIALS.csv")
    computed = [k.split(":")[1] for k in trials["key"] if k.startswith("Ж-222:B")]
    bad = [] if abs(float(r["c"].iloc[0]) - st["c"]) < 1e-9 and bool(r["passes"].iloc[0]) == RU.rule_passes(st) else ["G11: B5_RULE.csv не совпадает с пересчётом"]
    if not RU.consistent(st, computed) or ("B5" in o["cfgs"]) != RU.rule_passes(st):
        bad.append("G11: B5 посчитан не по правилу")
    return bad


def g12(o: dict) -> list:
    bad, n = [], len(o["cfgs"])
    for name, rows in (("B_COMPARISON.csv", (4 + n) * 4), ("B_TESTS.csv", n * 4 * 4), ("B_SBER.csv", (4 + n) * 4)):
        if len(IO.read_csv(name)) != rows:
            bad.append("G12: %s — строк %d, ожидалось %d" % (name, len(IO.read_csv(name)), rows))
    comp = IO.read_csv("B_COMPARISON.csv")
    k = D.eval_frame(o["P"], {c: o["fc"][c] for c in o["cfgs"]})
    crit = S.crit_panel(k, DECL.CTRL + o["cfgs"]).set_index("series")["crit_panel"]
    if any(abs(float(comp[comp["series"] == s]["crit_panel"].iloc[0]) - v) > 1e-12 for s, v in crit.items()):
        bad.append("G12: crit_panel в B_COMPARISON не равен hp_stats_sel.crit_panel")
    t = IO.read_csv("B_TESTS.csv")
    m = t[t["kind"] == "main"].dropna(subset=["p_holm"])
    if (m["p_holm_wide"] < m["p_holm"] - 1e-15).any() or m.empty:
        bad.append("G12: p_holm_wide < p_holm или нет основного семейства")
    s = IO.read_csv("B_SBER.csv")
    sa = s[(s["window"] == "all") & (s["series"] == "k1")]
    if sa.empty or int(sa["n"].iloc[0]) != 2271 or abs(float(sa["ref_a1_roll_126"].iloc[0]) - 0.20814) > 1e-4:
        bad.append("G12: B_SBER — не 2271 даты или нет ориентира")
    pt = IO.read_csv("B_PER_TICKER.csv")
    if pt["ticker"].nunique() != 17 or IO.read_csv("B_MCS.csv").empty or IO.read_csv("B_WEIGHTS.csv").empty:
        bad.append("G12: B_PER_TICKER/B_MCS/B_WEIGHTS неполны")
    return bad


def g13(o: dict) -> list:
    lt = IO.read_csv("B_LOTO.csv")
    want = set(dict.fromkeys([o["best"], "B2"]))
    bad = [] if want <= set(lt["config"]) else ["G13: нет LOTO для лучшей B и B2"]
    L = lt[lt["held_out"].isin(IO.M16 + ["M16"])]
    if set(L["held_out"]) != set(IO.M16 + ["M16"]) or not L["delta_zero_check"].astype(str).eq("True").all():
        bad.append("G13: LOTO неполон или δ ≠ 0 у исключённой")
    if "GCHE" not in set(lt["held_out"]) or (lt["series"].astype(str).str.startswith("нет M").sum() != 10):
        bad.append("G13: перенос GCHE / строки «нет M» по остальным E11")
    return bad


def g14(o: dict) -> list:
    txt = (IO.REP / "B_REPORT.md").read_text()
    crit = S.crit_panel(D.eval_frame(o["P"], {c: o["fc"][c] for c in o["cfgs"]}), DECL.CTRL + o["cfgs"]).set_index("series")["crit_panel"]
    best = min(o["cfgs"], key=lambda c: crit[c])
    go = bool(crit[best] <= json.loads(DECL.CP.read_text())["go_threshold_B"])
    first = txt.splitlines()[0].strip().lower()
    bad = [] if first.startswith("go" if go else "stop") and best in txt.splitlines()[0] else ["G14: первая строка B_REPORT не соответствует правилу (%s, %s)" % ("go" if go else "stop", best)]
    if len(txt.split()) > 1200 or "O1" not in txt:
        bad.append("G14: B_REPORT > 1200 слов или нет разбора против O1")
    return bad


def all_checks(o: dict) -> list:
    return g9(o) + g10(o) + g11(o) + g12(o) + g13(o) + g14(o)
