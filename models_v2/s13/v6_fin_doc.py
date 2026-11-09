"""V6 лист FIN, G52: итоговый документ — блоки, ЦЕЛИКОМ собираемые из таблиц на диске (`FIN_*.csv`, `*_GO`, `C_UNSTABLE`, `D_REGIME`, `BEST_SBER`, `ALL_TRIALS.csv`, `CHECKPOINT.json`, `MAP_DONE_V6.md`). Ни одно результатное число не вписано руками:
десятичные числа текста — только из этих функций (проверка `v6_fin_docheck`). Формат: десятичная запятая, минус «−»; Δ — разность средних QLIKE в процентах уровня QLIKE K1 того же окна (минус = лучше)."""
from __future__ import annotations

import json

import pandas as pd

from models_v2.s13 import v6_0_io as IO, v6_fin_data as D

WN = {"2017_2021": "2017–2021", "2022_2025": "2022–2025", "2026": "2026", "all": "вся история"}
BASES = ["K1", "K2", "K3", "K8", "V4:NTP2"]
DIAG = "A0g"
PRE = [c for c in D.CFG if c != DIAG]  # 17 предрегистрированных; A0g — диагностический контроль (Ж-230), в выбор «лучшей конструкции» не входит
LBL = "A0g — диагностический контроль (Ж-230)"
FIN_CSV = D.TABLES
ALIAS = {"FIN_ALL": "all", "FIN_YEARS": "years", "FIN_TESTS": "tests", "FIN_MCS": "mcs", "FIN_ABLATION": "abl", "FIN_STRESS": "stress", "FIN_ENCOMPASS": "enc", "FIN_INFO_ALL": "info", "FIN_LEVELS": "levels"}


def load() -> dict:
    t = {ALIAS[k]: IO.read_csv(k + ".csv") for k in FIN_CSV}
    t["go"] = pd.concat([IO.read_csv(c + "_GO.csv").assign(config=c) for c in D.CFG], ignore_index=True)
    t["unst"], t["regime"], t["best"] = IO.read_csv("C_UNSTABLE.csv"), IO.read_csv("D_REGIME.csv"), IO.read_csv("BEST_SBER.csv")
    t["trials"] = pd.read_csv(IO.REP / "ALL_TRIALS.csv")
    t["cp"] = json.loads((IO.REP / "CHECKPOINT.json").read_text())
    t["map"] = (IO.REP / "MAP_DONE_V6.md").read_text(encoding="utf-8")
    return t


def f(x: float, nd: int = 4) -> str:
    return ("%.*f" % (nd, x)).replace(".", ",").replace("-", "−")


def s(x: float, nd: int = 2) -> str:
    return ("%+.*f" % (nd, x)).replace(".", ",").replace("-", "−")


def iv(r, nd: int = 2) -> str:
    return "%s [%s; %s]" % (s(r.delta_pct, nd), s(r.lo90, nd), s(r.hi90, nd))


def yn(v) -> str:
    return "—" if pd.isna(v) else ("да" if str(v).lower() in ("true", "да") else "нет")


def word(x: float) -> str:
    return "лучше" if x < 0 else "хуже"


def hdr(cols: list) -> str:
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"


def nm(x: str) -> str:
    return {"D2ctl": "D2 (контроль)"}.get(x, x)


def m(t: dict) -> pd.DataFrame:
    return t["all"].set_index(["series", "window"])


def q(t: dict, series: str, win: str) -> float:
    return float(m(t).loc[(series, win), "qlike"])


def tt(t: dict, c: str, base: str, win: str):
    d = t["tests"]
    return d[(d["section"] == "delta_dm") & (d["config"] == c) & (d["base"] == base) & (d["window"] == win)].iloc[0]


def best(t: dict, win: str) -> str:
    """Лучшая конструкция V6 — только из 17 предрегистрированных (A0g — диагностический контроль, Ж-230)."""
    return min(PRE, key=lambda c: (q(t, c, win), c))


def best_with_diag(t: dict, win: str) -> str:
    """То же по всем 18 местам бюджета — для числовой сверки «выбор не меняется»."""
    return min(D.CFG, key=lambda c: (q(t, c, win), c))


def clear_vs_boosts(t: dict, c: str, win: str) -> bool:
    return all(tt(t, c, b, win).hi90 < 0 for b in ("K2", "K3"))


def point_better(t: dict, c: str, win: str) -> bool:
    return all(tt(t, c, b, win).delta_pct < 0 for b in ("K2", "K3"))


def level5(t: dict, c: str) -> str:
    d = t["levels"]
    return str(d[(d["config"] == c) & (d["level"] == 5)].iloc[0]["basis"])


def first_line(t: dict) -> str:
    b17, ba = best(t, "2017_2021"), best(t, "all")
    if any(clear_vs_boosts(t, c, w) for c in D.CFG for w in ("2017_2021", "all")):
        raise ValueError("первая строка документа написана для результата «интервалов целиком ниже 0 против K2 и K3 нет»")
    pts = [c for c in PRE if point_better(t, c, "2017_2021")]
    pa = [c for c in PRE if point_better(t, c, "all")]
    cell = lambda c, w: "к K2 %s, к K3 %s" % (iv(tt(t, c, "K2", w)), iv(tt(t, c, "K3", w)))
    return ("Нет: ни у одной из %d конфигураций V6 (%d предрегистрированных и диагностический контроль A0g, Ж-230; лучшая конструкция выбирается из предрегистрированных) нет статистически подтверждённого преимущества над самостоятельными XGBoost (K2) и LightGBM (K3) на SBER; уровень 5: «статистически неподтверждённое историческое улучшение». "
            "Δ — в %% уровня QLIKE K1, 90 %% интервал. Точечно лучше обоих бустингов в окне выбора 2017–2021 — %d конфигураций из %d (лучшая %s, QLIKE %s: %s; интервалы включают 0), на всей истории — %d (лучшая по QLIKE %s, %s: %s); "
            "в 2022–2025 %s хуже обоих (Δ к K2 %s, к K3 %s). Лучший прежний ряд V4:NTP2 лучше любой конфигурации V6 на всей истории (%s против %s) и хуже %s в 2017–2021 (%s против %s). Числа — потолок / development walk-forward, не независимый OOS."
            % (len(D.CFG), len(PRE), len(pts), len(PRE), b17, f(q(t, b17, "2017_2021")), cell(b17, "2017_2021"), len(pa), ba, f(q(t, ba, "all")), cell(ba, "all"), b17, s(tt(t, b17, "K2", "2022_2025").delta_pct), s(tt(t, b17, "K3", "2022_2025").delta_pct),
               f(q(t, "V4:NTP2", "all")), f(q(t, ba, "all")), b17, f(q(t, "V4:NTP2", "2017_2021")), f(q(t, b17, "2017_2021"))))


def best_block(t: dict, c: str) -> str:
    rows = []
    for w in ("2017_2021", "all", "2022_2025"):
        r = [tt(t, c, b, w) for b in BASES]
        rows.append("| %s | %s | %s |" % (WN[w], f(q(t, c, w)), " | ".join(iv(x) + " (p %s)" % f(x.p_one, 3) for x in r)))
    return hdr(["окно", "QLIKE %s" % c] + ["Δ к %s, %%" % b for b in BASES]) + "\n".join(rows) + "\n"


def enc_block(t: dict, cfgs: list) -> str:
    e = t["enc"].set_index(["config", "window", "base"])
    rows = ["| %s | %s |" % (c, " | ".join("%s [%s; %s]" % (s(e.loc[(c, w, b), "w"]), s(e.loc[(c, w, b), "lo90"]), s(e.loc[(c, w, b), "hi90"])) for w in ("2017_2021", "all") for b in ("K8", "K3"))) for c in cfgs]
    return hdr(["конфигурация"] + ["w к %s, %s" % (b, WN[w]) for w in ("2017_2021", "all") for b in ("K8", "K3")]) + "\n".join(rows) + "\n"


def mcs_block(t: dict) -> str:
    mc, rows = t["mcs"], []
    for w in ("2017_2021", "2022_2025", "all"):
        d = mc[(mc["window"] == w) & mc["in_mcs90"].astype(str).str.lower().eq("true")].sort_values("mean_qlike")
        rows.append("| %s | %d из %d | %s |" % (WN[w], len(d), (mc["window"] == w).sum(), ", ".join("%s (%s)" % (nm(r.model), f(r.mean_qlike)) for r in d.itertuples())))
    return hdr(["окно", "в MCS 90 %", "ряды (средний QLIKE)"]) + "\n".join(rows) + "\n"
