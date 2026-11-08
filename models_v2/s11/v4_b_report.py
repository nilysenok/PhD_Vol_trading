"""V4-B, G14: B_REPORT.md — первая строка «go»/«stop» по заранее записанному правилу; все числа берутся из таблиц стадии (не вводятся вручную)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s11 import v4_b_stats as ST, v4_io as IO

DEVIATIONS = [
    "Кривизна штрафа. «κ = 250 пар-эквивалентов» реализовано как штраф (κ·s²/2)·Σδ² в сумме потерь по строкам, где s² = mean ln²(M/H) на закрытых парах — кривизна QLIKE одной пары по весу; тот же s² для приора (250 пар). Для b (B5) s² = 1. Масштаб не подбирался.",
    "Панельный параметр (w_p, b_p) входит в оценку вместе с δ_i и тоже стянут к приору (вес приора 250 пар против κ·n бумаг): при 17 бумагах стягивание w_p ≈ 1/18 расстояния до 0,5.",
    "«До первых 60 закрытых пар — приор» применено к панели целиком (число закрытых пар всех бумаг < 60 → w = 0,5, p = 0, b = 0, B6 — 1/3 каждому); бумага без закрытых пар получает панельные значения.",
    "Начало расчёта: первый месяц (2016-12/2017-01) — приор, закрытых пар нет; оценка с февраля 2017. «Закрытая» пара месяца — t⁺ строго раньше первого торгового дня месяца.",
    "N_wide: в файлах Optuna h = 1 нет колонки бумаги — посчитаны все 2400 строк-проб (верхняя оценка).",
    "Метрики RMSE/MAE/R² — единое определение проекта (формулы `hp_hyb_out`: RMSE, MAE по RV; R² по RV и по ln RV); пересчитаны листом C-a 08.11, прежние колонки (rmse_ln, mae_ln, r2_lvl) — в `archive_pre_metrics/` с SHA.",
    "ALL_TRIALS: правка `v4_trials.verify` (разрешены строки `Ж-222:B*`), иначе `run_v4 --stage audit --check` падал бы на допустимых строках этапа B; других правок audit/A нет.",
]


def delta_pct(comp: pd.DataFrame, cfg: str, w: str) -> dict:
    c = comp[comp["window"] == w].set_index("series")["qlike"]
    return {x: 100.0 * float((c[cfg] - c[x]) / c["k1"]) for x in ("k1", "k2", "k3", "d2")}


def oracle_share(o: dict) -> list:
    a = IO.read_csv("A_ORACLE.csv")
    f = o["fc"][o["best"]]
    out = []
    for w in ("2017_2021", "all"):
        r = a[(a["ticker"] == "M16") & (a["window"] == w) & (a["m"] == "KG")].iloc[0]
        lo, hi = IO.WINDOWS[w]
        qs = [float(np.mean(metrics.qlike(g["rv"], g["F"]))) for _, g in f[f["ticker"].isin(IO.M16) & (f["t_plus"].dt.year >= lo) & (f["t_plus"].dt.year <= hi)].groupby("ticker")]
        share = (r["q_best"] - np.mean(qs)) / (r["q_best"] - r["o1"])
        out.append("%s: лучшая одиночная (H или M) %.4f, O1 (постоянный вес задним числом) %.4f, лучшая B %.4f → достигнуто %.0f %% потолка постоянного веса" % (w, r["q_best"], r["o1"], np.mean(qs), 100 * share))
    return out


def build(o: dict) -> str:
    cp, crit, comp = o["cp"], o["crit"].set_index("series")["crit_panel"], o["comp"]
    best, thr = o["best"], o["cp"]["go_threshold_B"]
    ctrl = {k: crit[k] for k in ("k1", "k2", "k3", "d2")}
    go, bc = crit[best] <= thr, min(ctrl, key=ctrl.get)
    head = "%s — этап B: лучшая %s, crit_panel %.5f против порога %.5f (лучший контроль %s %.5f; отрыв %.5f = %.2f %% уровня K1 %.5f; нужно ≥ 0,0005)" % (
        "go" if go else "stop", best, crit[best], thr, bc, ctrl[bc], ctrl[bc] - crit[best], 100 * (ctrl[bc] - crit[best]) / ctrl["k1"], ctrl["k1"])
    L = [head, "", "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; первая цифра — потолок; окна 2022+ в выборе не участвуют.", "", "## Отступления и толкования (до выводов)"] + ["- " + d for d in DEVIATIONS]
    L += ["", "## crit_panel (M16 × цели 2017–2021, `hp_stats_sel.crit_panel`)", "", "| ряд | crit_panel |", "|---|---|"] + ["| %s | %.5f |" % (s, v) for s, v in crit.items()]
    L += ["", "## Δ QLIKE лучшей B к контролям, % уровня K1 (M16; минус — лучше)", "", "| окно | к K1 | к K2 | к K3 | к D2 |", "|---|---|---|---|---|"]
    for w in ("all", "2022_2025"):
        d = delta_pct(comp, best, w)
        L.append("| %s | %.2f | %.2f | %.2f | %.2f |" % (w, d["k1"], d["k2"], d["k3"], d["d2"]))
    L += ["", "Все конфигурации (Δ к K1, %, окно all): " + "; ".join("%s %.2f" % (c, delta_pct(comp, c, "all")["k1"]) for c in o["cfgs"]) + "."]
    L += ["", "## Холм (основное семейство M16 × {2022_2025, all}, N=%d; N_wide=%d)" % (cp["N_declared_before_stats"], cp["N_wide"]), ""] + holm_lines(o)
    L += ["", "## Разбор против oracle O1 (диагностика, не критерий)", ""] + ["- " + x for x in oracle_share(o)]
    L += ["", "## SBER (отдельно, в выбор не входит; 2271 дата; ориентир 0,2081 — потолок)", ""] + sber_lines(o)
    L += ["", "## B5", "", "c = %.4f, t(НУ 5) = %.2f, ln(RV/H) = %.4f, ln(RV/M) = %.4f; правило %s → B5 %s." % (o["st"]["c"], o["st"]["t_nw5"], o["st"]["mean_log_rv_h"], o["st"]["mean_log_rv_m"],
                                                                                    "выполнено" if "B5" in o["cfgs"] else "не выполнено", "посчитан" if "B5" in o["cfgs"] else "не считался (1 конфигурация в резерв)")]
    L += ["", "## LOTO и перенос", ""] + loto_lines(o)
    L += ["", "Израсходовано конфигураций этапа: %d из 6 (бюджет программы 60); время счёта %s." % (len(o["cfgs"]), "см. CHECKPOINT.json (B_runtime_sec)")]
    return "\n".join(L) + "\n"


def holm_lines(o: dict) -> list:
    t = o["tests"]
    t = t[(t["config"] == o["best"]) & (t["kind"] == "main")]
    return ["- %s vs %s, %s: Δ=%.5f, DM p=%.4f, p_holm=%.3f, p_holm_wide=%.3f, бутстрап ДИ [%.5f; %.5f]" % (r.config, r.comparator, r.window, r.delta_mean, r.dm_p, r.p_holm, r.p_holm_wide, r.boot_lo, r.boot_hi) for r in t.itertuples()]


def sber_lines(o: dict) -> list:
    s = o["sber"]
    s = s[s["window"] == "all"].set_index("series")["qlike"]
    return ["- " + "; ".join("%s %.4f" % (k, v) for k, v in s.items()) + "; ориентир A1_roll_126 0,2081 (потолок: c, λ выбраны на всей истории)."]


def loto_lines(o: dict) -> list:
    lt = o["loto"]
    lt = lt[(lt["held_out"] == "M16") & lt["window"].isin(["all", "2022_2025"])]
    L = ["- %s, %s: LOTO %.4f, пул %.4f, K1 %.4f; Δ к пулу %.5f, к K1 %.5f (%.2f %% K1)" % (r.config, r.window, r.q_loto, r.q_pool, r.q_k1, r.d_loto_pool, r.d_loto_k1, r.d_loto_k1_pct_of_k1) for r in lt.itertuples()]
    g = o["gche"]
    g = g[(g["held_out"] == "GCHE") & (g["window"] == "all")]
    return L + ["- GCHE (перенос панельных весов, окно all): " + "; ".join("%s %s %.4f" % (r.config, r.series, r.qlike) for r in g.itertuples()) + "; прочие E11 — нет M, не применимо."]
