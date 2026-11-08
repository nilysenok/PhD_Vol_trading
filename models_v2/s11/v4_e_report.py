"""V4-E, G25: E_REPORT.md. Первая строка — «go»/«stop» по правилу (лучшая E ≤ порог), пересчитывается проверкой по crit_panel; далее — E1 против K1, E2 − E1, b в шоках, годы, SBER, LOTO, реестр."""
from __future__ import annotations

import pandas as pd

from models_v2.s11 import v4_e_run as R

REF = ("k1", "k2", "k3", "d2")


def verdict(crit: pd.Series, thr: float) -> tuple:
    """-> (go?, лучшая E, её crit_panel)."""
    best = min(R.CFGS, key=lambda x: float(crit[x]))
    return bool(crit[best] <= thr), best, float(crit[best])


def first_line(o: dict) -> str:
    cs, cp = o["crit"].set_index("series")["crit_panel"], o["cp"]
    go, best, v = verdict(cs, cp["go_threshold_B"])
    ctl = min(REF, key=lambda x: float(cs[x]))
    q = o["comp"].set_index(["series", "window"])["qlike"]
    d = lambda w: "/".join("%.2f" % (100.0 * (q[(best, w)] - q[(r, w)]) / q[("k1", w)]) for r in REF)
    g = o["go"]["cfg"][best]
    yn = lambda x: "да" if x else "нет"
    go3 = go and g["go"]
    gr = o["go"]["ref"]
    return ("%s — этап E: лучшая %s, (1) crit_panel %.5f ≤ порога %.5f: %s (лучший контроль %s %.5f; отрыв %.5f = %.2f %% уровня K1 %.5f); (2) без 2017 %.5f ≤ min(K2,K3,D2) %.5f − 0,0005: %s; "
            "(3) без 1 %% крайних дат %.5f ≤ min(K2,K3,D2) %.5f − 0,0005: %s; Δ QLIKE к K1/K2/K3/D2 (%% уровня K1, M16): вся история %s, 2022–2025 %s" % (
                "go" if go3 else "stop", best, v, cp["go_threshold_B"], yn(g["c1"]), ctl, cs[ctl], v - cs[ctl], 100.0 * (v - cs[ctl]) / cs["k1"], cs["k1"], g["crit_no2017"],
                min(x["crit_no2017"] for x in gr.values()), yn(g["c2"]), g["crit_trim1"], min(x["crit_trim1"] for x in gr.values()), yn(g["c3"]), d("all"), d("2022_2025")))


def _tab(df: pd.DataFrame, fmt="%.5f") -> str:
    return "\n".join("| " + " | ".join(str(x) if isinstance(x, str) else (fmt % x if x == x else "—") for x in r) + " |" for r in df.itertuples(index=False))


def build(o: dict) -> str:
    cs = o["crit"].set_index("series")["crit_panel"]
    y = o["years"][(o["years"]["role"] == "выбор")].pivot(index="series", columns="year", values="ann_qlike")
    ys = ["k1", "d2", "B2", "C8"] + R.CFGS
    sb = o["sber"][o["sber"]["window"] == "all"].set_index("series")["qlike"]
    sh = o["shocks"].pivot(index="config", columns="window", values="mean_b")
    lo = o["loto"][(o["loto"]["held_out"] == "M16") & (o["loto"]["window"] == "all")]
    e12 = cs["E2"] - cs["E1"]
    L = [first_line(o), "", "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; первая цифра — потолок. Объявление — GATES_V4.md («Этап E»); израсходовано 6 конфигураций (всего 21 из 60).", "",
         "## crit_panel (M16 × цели 2017–2021, выбор)", "| ряд | crit_panel |", "|---|---|", _tab(pd.DataFrame({"s": ["k1", "k2", "k3", "d2", "B2", "C8"] + R.CFGS, "v": [cs[x] for x in ["k1", "k2", "k3", "d2", "B2", "C8"] + R.CFGS]})), "",
         "## Калибровка одного HAR-J (E1) и вклад ML (E2 − E1)",
         "E1 − K1: %+.5f (%+.2f %% уровня K1). E2 − E1: %+.5f (%+.2f %%). E2 против G = B2: %+.5f. Калибровка одного HAR-J %s; ML (E2 − E1) %s. Доля дней с |b| = 0,5 (обрезка) у E1: %.3f." % (
             cs["E1"] - cs["k1"], 100 * (cs["E1"] - cs["k1"]) / cs["k1"], e12, 100 * e12 / cs["k1"], cs["E2"] - cs["B2"], "выигрыш к K1 даёт" if cs["E1"] < cs["k1"] else "выигрыша к K1 не даёт",
             "добавляет качество" if e12 < 0 else "качества не добавляет", float((o["fc"]["E1"]["b"].abs() >= 0.5 - 1e-12).mean())), "",
         "## По годам цели (средний QLIKE M16; 2017–2021 выбор, 2022+ описательно)", "| ряд | " + " | ".join(str(c) for c in y.columns) + " |", "|---|" + "---|" * len(y.columns), "\n".join(
             "| %s | " % s + " | ".join("%.4f" % v for v in y.loc[s]) + " |" for s in ys), "",
         "## b в шоках (среднее по панели, описательно)", "| конфигурация | " + " | ".join(sh.columns) + " |", "|---|" + "---|" * len(sh.columns), "\n".join("| %s | " % i + " | ".join("%+.4f" % v for v in sh.loc[i]) + " |" for i in sh.index), "",
         "## SBER (2271 дата, ориентир 0,2081 — потолок: c, λ выбраны на всей истории)", " ".join("%s %.4f;" % (s, sb[s]) for s in ["k1", "k2", "k3", "d2", "B2", "C8"] + R.CFGS), "",
         "## LOTO лучшей E (%s)" % o["best"], "M16, вся история: LOTO %.4f / пул %.4f / K1 %.4f (Δ к пулу %+.4f)." % (lo["q_loto"].iloc[0], lo["q_pool"].iloc[0], lo["q_k1"].iloc[0], lo["d_loto_pool"].iloc[0]), "",
         "## Ужесточённое «go» (Ж-224 п. 10): условия 1–3 по каждой E", "| E | crit_panel | (1) | без 2017 | (2) | без 1 % крайних | (3) | go |", "|---|---|---|---|---|---|---|---|",
         "\n".join("| %s | %.5f | %s | %.5f | %s | %.5f | %s | %s |" % (e, cs[e], r["c1"], r["crit_no2017"], r["c2"], r["crit_trim1"], r["c3"], r["go"]) for e, r in o["go"]["cfg"].items()),
         "Контроли без 2017 / без 1 %%: %s." % "; ".join("%s %.5f / %.5f" % (k_, v["crit_no2017"], v["crit_trim1"]) for k_, v in o["go"]["ref"].items()), "",
         "## Реестр", "Повторный запуск стадии больше не удаляет строки других стадий (`v4_trials.replace_rows`); в ALL_TRIALS — строки Ж-222:B*, C*, E*.", "",
         "Файлы: E_COMPARISON, E_PER_TICKER, E_TESTS, E_MCS, E_YEARS, E_SBER, E_LOTO, E_BIAS (.csv); %s." % o["note"]]
    return "\n".join(L) + "\n"
