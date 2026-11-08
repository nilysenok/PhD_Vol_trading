"""V4-FIN: итоговый документ, часть 1 — блоки, ЦЕЛИКОМ собираемые из `FIN_*.csv` (главная строка, таблица трёх лучших, ссылочные ряды, все 46 архитектур, окна, 17 бумаг, SBER, годы, тесты, MCS).
Каждое число этих блоков — из таблиц; `--check` пересобирает главную строку и блоки из CSV на диске и ищет их в документе (G47). Текстовые части — `v4_fin_text*.py`."""
from __future__ import annotations

import json

import pandas as pd

from models_v2.s11 import v4_fin_data as FD, v4_io as IO

REFS = ["k1", "k2", "k3", "d2"]
NAMES = {"k1": "K1 (HAR-J)", "k2": "K2 (XGBoost)", "k3": "K3 (LightGBM)", "d2": "D2 (LGBM-D2)"}
WN = {"2017": "2017", "2018": "2018", "2017_2021": "2017–2021", "2022_2025": "2022–2025", "2026": "2026", "all": "вся"}


def tables_from_disk() -> dict:
    t = {k: IO.read_csv(f) for k, f in (("all", "FIN_ALL.csv"), ("tickers", "FIN_TICKERS.csv"), ("sber", "FIN_SBER.csv"), ("tests", "FIN_TESTS.csv"), ("mcs", "FIN_MCS.csv"), ("years", "FIN_YEARS.csv"))}
    t["cp"] = json.loads((IO.REP / "CHECKPOINT.json").read_text())
    a = t["all"][(t["all"]["window"] == "all") & t["all"]["series"].isin(FD.CFGS)]
    t["top"] = list(a.sort_values(["crit_panel", "series"])["series"].head(3))
    return t


def nm(s: str) -> str:
    return NAMES.get(s, s)


def yn(v) -> str:
    return "—" if pd.isna(v) else ("да" if str(v).lower() == "true" else "нет")


def idx(t: dict) -> pd.DataFrame:
    return t["all"].set_index(["series", "window"])


def word(x: float) -> str:
    return "лучше" if x < 0 else "хуже"


def n_go(t: dict) -> int:
    a = t["all"][(t["all"]["window"] == "all") & t["all"]["series"].isin(FD.CFGS)]
    return int(sum(str(v).lower() == "true" for v in a["go"]))


def n_both(t: dict) -> int:
    return len(both_list(t))


def both_list(t: dict) -> list:
    a = idx(t)
    return [c for c in FD.CFGS if all(a.loc[(c, w), "d_qlike_%s_pct_of_k1" % r] < 0 for w in ("2017_2021", "2022_2025") for r in ("k2", "k3"))]


def min_holm(t: dict) -> float:
    """Минимальный p_holm (N = N_declared) трёх лучших и K8 против K2, K3, D2 в основном семействе (2022–2025, вся история)."""
    d = t["tests"]
    return float(d[(d["kind"] == "main") & d["comparator"].isin(["k2", "k3", "d2"]) & d["config"].isin(t["top"])]["p_holm"].min())


def both_block(t: dict) -> str:
    """Конфигурации, лучшие и K2, и K3 на окне выбора и на 2022–2025 (Δ QLIKE < 0 во всех четырёх клетках): их Δ и условия «go»."""
    a, rows = idx(t), []
    for c in both_list(t):
        v = ["%+.2f" % a.loc[(c, w), "d_qlike_%s_pct_of_k1" % r] for w in ("2017_2021", "2022_2025") for r in ("k2", "k3")]
        rows.append("| %s | %s | %.5f | %s | %s |" % (c, FD.STAGE_OF[c], a.loc[(c, "all"), "crit_panel"], " | ".join(v), yn(a.loc[(c, "all"), "go"])))
    return _hdr(["конфигурация", "этап", "crit_panel", "Δ K2, 2017–21", "Δ K3, 2017–21", "Δ K2, 2022–25", "Δ K3, 2022–25", "go"]) + "\n".join(rows) + "\n"


def first_line(t: dict) -> str:
    a, b, thr = idx(t), t["top"][0], t["cp"]["go_threshold_B"]
    g = lambda s, w, c: float(a.loc[(s, w), c])
    d = lambda w, r: g(b, w, "d_qlike_%s_pct_of_k1" % r)
    return ("Устойчивого преимущества гибрида над самостоятельными XGBoost (K2) и LightGBM (K3) нет: «go» (три условия Ж-225 п. 10) прошли %d из 46 конфигураций; лучшая по crit_panel — %s: %.5f против K1 %.5f, K2 %.5f, K3 %.5f, D2 %.5f (порог go %.5f), "
            "простое среднее QLIKE 2017–2021 %.5f против K2 %.5f и K3 %.5f, но на 2022–2025 она %s K2 на %.2f %% и %s K3 на %.2f %% уровня K1 (вся история: %s %.2f %% и %s %.2f %%); лучше K2 и K3 одновременно на окне выбора и на 2022–2025: %d из 46 конфигураций (разности малы, на 2022–2025 — доли процента), а у трёх лучших минимальный p_holm к K2, K3, D2 (N = %d) равен %.3f."
            % (n_go(t), b, g(b, "all", "crit_panel"), g("k1", "all", "crit_panel"), g("k2", "all", "crit_panel"), g("k3", "all", "crit_panel"), g("d2", "all", "crit_panel"), thr, g(b, "all", "qlike_simple_2017_2021"),
               g("k2", "all", "qlike_simple_2017_2021"), g("k3", "all", "qlike_simple_2017_2021"), word(d("2022_2025", "k2")), abs(d("2022_2025", "k2")), word(d("2022_2025", "k3")), abs(d("2022_2025", "k3")),
               word(d("all", "k2")), abs(d("all", "k2")), word(d("all", "k3")), abs(d("all", "k3")), n_both(t), t["cp"]["N_declared_before_stats"], min_holm(t)))


def _hdr(cols: list) -> str:
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"


def _row(a, s: str) -> list:
    g = lambda w, c: float(a.loc[(s, w), c])
    cond = [yn(a.loc[(s, "all"), c]) for c in ("c1", "c2", "c3", "go")]
    return ["%.5f" % g("all", "crit_panel"), *cond, "%.5f" % g("all", "qlike_simple_2017_2021")] + ["%+.2f" % g(w, "d_qlike_%s_pct_of_k1" % r) for w in ("all", "2022_2025") for r in REFS]


HEAD = ["crit_panel", "(1)", "(2)", "(3)", "go", "простое ср. 2017–2021"] + ["Δ %s, %s" % (r.upper(), w) for w in ("вся", "2022–25") for r in REFS]


def top3_block(t: dict) -> str:
    a = idx(t)
    return _hdr(["№", "архитектура", "этап"] + HEAD) + "".join("| %d | %s | %s | %s |\n" % (i + 1, s, FD.STAGE_OF[s], " | ".join(_row(a, s))) for i, s in enumerate(t["top"]))


def ref_block(t: dict) -> str:
    a = idx(t)
    return _hdr(["ряд"] + HEAD) + "".join("| %s | %s |\n" % (nm(s), " | ".join(_row(a, s))) for s in REFS)


def all_block(t: dict) -> str:
    a = idx(t)
    order = sorted(FD.CFGS, key=lambda s: (float(a.loc[(s, "all"), "crit_panel"]), s))
    return _hdr(["архитектура", "этап"] + HEAD) + "".join("| %s | %s | %s |\n" % (s, FD.STAGE_OF[s], " | ".join(_row(a, s))) for s in order)


def windows_block(t: dict) -> str:
    """Δ QLIKE к K1, K2, K3, D2 (% уровня K1 окна) для трёх лучших и K8 по окнам 2017, 2018, 2017–2021, 2022–2025, 2026."""
    a, rows = idx(t), []
    for s in list(t["top"]) + [x for x in ("K8",) if x not in t["top"]]:
        for w in ("2017", "2018", "2017_2021", "2022_2025", "2026"):
            rows.append("| %s | %s | %.5f | %s |" % (s, WN[w], a.loc[(s, w), "qlike"], " | ".join("%+.2f" % a.loc[(s, w), "d_qlike_%s_pct_of_k1" % r] for r in REFS)))
    return _hdr(["архитектура", "окно", "QLIKE (M16)"] + ["Δ %s" % r.upper() for r in REFS]) + "\n".join(rows) + "\n"


def tickers_block(t: dict, w: str) -> str:
    d = t["tickers"][t["tickers"]["window"] == w]
    cols = REFS + ["K8", "C1", "C4"] + [x for x in t["top"] if x not in ("K8", "C1", "C4")]
    p = d.pivot(index="ticker", columns="series", values="qlike")[cols]
    p.loc["M16 (среднее)"] = d[d["group"] == "M16"].pivot(index="ticker", columns="series", values="qlike")[cols].mean()
    return _hdr(["бумага"] + cols) + "".join("| %s | %s |\n" % (i, " | ".join("%.4f" % v for v in r)) for i, r in p.iterrows())


def sber_block(t: dict) -> str:
    s = t["sber"].set_index(["series", "window"])
    ref = float(s.loc[("k1", "all"), "ref_a1_roll_126"])
    order = sorted(REFS + FD.CFGS, key=lambda c: float(s.loc[(c, "all"), "qlike"]))
    return _hdr(["ряд", "QLIKE, вся (2271 дата)", "Δ к ориентиру %.5f" % ref, "QLIKE 2022–2025"]) + "".join(
        "| %s | %.4f | %+.4f | %.4f |\n" % (c, s.loc[(c, "all"), "qlike"], s.loc[(c, "all"), "qlike"] - ref, s.loc[(c, "2022_2025"), "qlike"]) for c in order)


def years_block(t: dict) -> str:
    y = t["years"]
    cols = REFS + ["K8", "C8"] + [x for x in t["top"] if x not in ("K8", "C8")]
    p = y[y["series"].isin(cols)].pivot(index="year", columns="series", values="ann_qlike")[cols]
    return _hdr(["год цели"] + cols) + "".join("| %d | %s |\n" % (i, " | ".join("%.4f" % v for v in r)) for i, r in p.iterrows())


def tests_block(t: dict) -> str:
    d = t["tests"][t["tests"]["kind"] == "main"]
    return _hdr(["конфигурация", "база", "окно", "Δ QLIKE", "DM p", "бутстрап 95 % ДИ", "p_holm (N=%d)" % t["cp"]["N_declared_before_stats"], "p_holm_wide (N=%d)" % t["cp"]["N_wide"]]) + "".join(
        "| %s | %s | %s | %+.5f | %.3f | [%+.5f; %+.5f] | %.3f | %.3f |\n" % (r.config, r.comparator, WN[r.window], r.delta_mean, r.dm_p, r.boot_lo, r.boot_hi, r.p_holm, r.p_holm_wide) for r in d.itertuples())


def mcs_block(t: dict) -> str:
    m, rows = t["mcs"], []
    for w in ("2017_2021", "2022_2025", "all"):
        d = m[(m["window"] == w) & m["in_mcs90"].astype(str).eq("True")].sort_values("mean_qlike")
        rows.append("| %s | %d из %d | %s |" % (WN[w], len(d), (m["window"] == w).sum(), ", ".join("%s (%.4f)" % (r.model, r.mean_qlike) for r in d.itertuples())))
    return _hdr(["окно", "в MCS 90 %", "ряды (средний панельный QLIKE)"]) + "\n".join(rows) + "\n"


def numbers_in(s: str) -> int:
    import re
    return len(re.findall(r"[-+]?\d+[.,]\d+|\b\d+\b", s))


def missing_claims(txt: str, t: dict) -> list:
    """Фрагменты, пересобранные из таблиц, которых нет в тексте документа (подменённое число -> фрагмент не найден)."""
    return [c for c in claims(t) if c not in txt]


def claims(t: dict) -> list:
    """Проверяемые фрагменты документа: главная строка, таблицы трёх лучших и ссылочных рядов, все 46 архитектур, окна."""
    return [first_line(t), top3_block(t), ref_block(t), all_block(t), windows_block(t), both_block(t)]
