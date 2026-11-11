"""W1: текст `W1_REPORT.md` — генератор из таблиц (числа не вписываются руками; результатные числа берутся только из таблиц). Формулировки по правилам, записанным в коде:
«выигрыша нет» не пишется при Δ < 0 и p ≥ 0,05; «превосходит», «доказано» — только при p Холма < 0,05."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s16.x1_t0_doc import f6, tab

STATUS = "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; диагностика после просмотра; Ж-250. Обучения нет, новых прогнозов нет, испытаний не регистрируется: только сохранённые ряды."
WN = {"M": "окно M (общий период K1, K2, K3, X1)", "H": "окно H (все ряды T0 + ZB, ZB0, до 2026-02-27)", "2017": "окно 2017", "F": "окно F (продолжение 2026)"}


def pc(x: float) -> str:
    return "%+.2f %%" % x


def sec_join(j: pd.DataFrame) -> list:
    L = ["## 1. Склейка архива и продолжения 2026", "", "Стоп-условия пройдены: дублей d и t⁺ нет; RV дня t⁺ одинакова у всех рядов на общих датах (при расхождении генератор останавливается с перечнем дат).", "",
         tab(["ряд", "архив, строк", "новых", "t⁺ с", "t⁺ по", "дубли d/t⁺", "стык: последний архивный → первый новый t⁺", "стык подряд по календарю", "d → t⁺ вне календаря", "RV ≠ независимого ряда", "нет в независимом ряду (t⁺)", "архив = fwd-файл (макс. откл.)"],
             [(r.series, r.n_archive, r.n_new, r.tp_first, r.tp_last, "%d/%d" % (r.dup_d, r.dup_tp), "%s → %s" % (r.last_arch_tp, r.first_new_tp or "—"), "да" if r.stitch_ok else "НЕТ", r.calendar_bad, r.rv_vs_daily_bad,
               r.rv_ref_missing or "—", "—" if pd.isna(r.archive_vs_fwd_maxdiff) else "%.1e" % r.archive_vs_fwd_maxdiff) for r in j.itertuples()]), "",
         "Независимый ряд RV — дневной архив (`daily_SBER`) и запечатанная RV продолжения (замок открыт, Л6). Ряды с меньшим покрытием не обрезают главное окно; отдельное окно получает ряд, не покрывающий общий период ядра.", "", "**Время выпуска и правила обновления** (архив и продолжение — одно правило; архивные строки побитно сверены в Ж-249, `FWD_CHECKS.csv`):", ""]
    L += ["- %s: %s." % (r.series, r.update_rule) for r in j.itertuples()]
    return L + ["- Время выпуска: %s." % j["release"].iloc[0], ""]


def sec_table(t: pd.DataFrame) -> list:
    L = ["## 2. QLIKE (среднее ДНЕВНЫХ потерь l = RV/F − ln(RV/F) − 1 на общих датах окна; Δ — в % QLIKE базы, минус — лучше базы)", ""]
    for w, name in WN.items():
        x = t[t["window"] == w]
        if len(x):
            L += ["**%s: t⁺ %s – %s, n = %d.**" % (name, x["first_tplus"].iloc[0], x["last_tplus"].iloc[0], x["n"].iloc[0]), "",
                  tab(["ряд", "QLIKE", "Δ к K2", "Δ к K3", "Δ к X1", "MCS 90 %", "p MCS", "версия (источник)", "sha256 архива", "sha256 продолжения"],
                      [(r.series, f6(r.qlike), "—" if np.isnan(r.d_k2_pct) else pc(r.d_k2_pct), "—" if np.isnan(r.d_k3_pct) else pc(r.d_k3_pct), "—" if np.isnan(r.d_x1_pct) else pc(r.d_x1_pct),
                        "в" if r.mcs90 else "вне", "%.3f" % r.mcs_p, r.origin, r.archive_sha256[:12], r.new_sha256[:12] or "—") for r in x.itertuples()]), ""]
    return L


def sec_stats(s: pd.DataFrame) -> list:
    L = ["## 3. Статистика пар (90 %% блочный бутстрап: блок 20, B 2000, seed 0; DM Ньюи–Уэст лаг 5, односторонний «модель лучше»; Холм с N программы = %d, N_wide = %d)" % (s["n_holm"].iloc[0], s["n_holm_wide"].iloc[0]), ""]
    for w in ("M", "H", "F"):
        x = s[s["window"] == w]
        L += ["**%s (n = %d)**" % (WN[w], x["n"].iloc[0]), "", tab(["ряд", "база", "Δ % базы", "90 % интервал Δ", "p DM", "p Холма N", "p Холма N_wide", "вывод"],
              [(r.hybrid, r.base, pc(r.delta_pct), "[%+.2f; %+.2f]" % (r.lo90, r.hi90), "%.4f" % r.p_one, "%.4f" % r.p_holm_N, "%.4f" % r.p_holm_Nwide, r.verdict) for r in x.itertuples()]), ""]
    return L


def sec_years(y: pd.DataFrame, c: pd.DataFrame) -> list:
    L = ["## 4. Вклад лет (c_y = Σ(l_X − l_K) / (n · Q_K), % базы; Σ c_y = полному Δ; 2026 — две части)", ""]
    for s, b in (("X1", "K2"), ("X1", "K3"), ("ZB", "K2"), ("ZB", "K3")):
        x, k = y[(y["series"] == s) & (y["base"] == b)], c[(c["series"] == s) & (c["base"] == b)].set_index("key")
        L += ["**%s против %s: полный Δ %s.**" % (s, b, pc(x["delta_total_pct"].iloc[0])), "", tab(["период", "n", "QLIKE %s" % s, "QLIKE %s" % b, "Δ года", "вклад c_y", "Δ без периода"],
              [(r.period, r.n, f6(r.q_series), f6(r.q_base), pc(r.delta_pct), "%+.3f" % r.contrib_pct, pc(float(k.at[r.period, "value"]))) for r in x.itertuples()]), "",
              "Лет с выигрышем: %d из %d; доля выигрыша от лучшего периода %.0f %%, от двух лучших %.0f %%; сумма вкладов ухудшающих периодов %+.3f п. п. (выигрыш всего %.3f п. п.)." % (
                  k.at["years_with_gain", "value"], k.at["periods", "value"], 100 * k.at["share_best_year", "value"], 100 * k.at["share_two_best_years", "value"], k.at["worsening_sum_pct", "value"], k.at["gain_total_pct", "value"]), ""]
    return L


def top_groups(st: pd.DataFrame, pair: str, k: int = 3) -> pd.DataFrame:
    x = st[(st["pair"] == pair) & ~st["group"].str.startswith("ORACLE") & (st["group"] != "нет порога")]
    return x[x["contrib_pct"] > 0].sort_values("contrib_pct", ascending=False).head(k)


def sec_states(st: pd.DataFrame, bias: pd.DataFrame) -> list:
    L = ["## 5. Причинно доступные состояния S1…S8 (окно M; X1 − K; вклад = Σ потерь группы / (n · Q_K); Σ вкладов по состоянию = полному Δ)", "", "Полная таблица — `W1_STATES.csv` (все группы всех состояний; группа «нет порога» — пока прошлых значений меньше порога выборки).", ""]
    for pair in ("X1|K2", "X1|K3"):
        T = top_groups(st, pair)
        L += ["**Группы с наибольшим положительным вкладом в проигрыш X1 против %s:**" % pair.split("|")[1], "", tab(["состояние", "группа", "доля пар", "средняя разность потерь", "вклад в Δ, п. п.", "лет с тем же знаком"],
              [(r.state + ": " + r.desc, r.group, "%.1f %%" % (100 * r.share), "%+.5f" % r.mean_diff, "%+.3f" % r.contrib_pct, "%d из %d" % (r.years_same_sign, r.years_with_group)) for r in T.itertuples()]), ""]
    o = st[st["group"].str.startswith("ORACLE")]
    L += ["**Потолок, использует исход (не прогноз): «лучший из X1 и K2 в группе состояния ex post», Δ к K2:** " + "; ".join("%s %s" % (r.state, pc(r.contrib_pct)) for r in o.itertuples()) + ".", ""]
    b = bias[bias["kind"] == "год t⁺"]
    L += ["## 6. Смещение (ln(RV/F): минус — прогноз выше исхода; доля недопрогноза — доля дат с RV > F; в таблице `W1_BIAS.csv` — также S1, S3)", "",
          tab(["год t⁺"] + ["%s: ln(RV/F) / недопрогноз / доля потерь от недопрогноза" % s for s in ("X1", "K2", "K3")],
              [[g] + ["%+.3f / %.0f %% / %.0f %%" % tuple(v) for v in [(float(q.mean_ln_rv_f), 100 * float(q.share_under), 100 * float(q.loss_share_under)) for q in [b[(b["series"] == s) & (b["group"] == g)].iloc[0] for s in ("X1", "K2", "K3")]]]
               for g in sorted(b["group"].unique())]), ""]
    return L


def sec_answers(s: pd.DataFrame, y: pd.DataFrame, st: pd.DataFrame, c: pd.DataFrame) -> list:
    L = ["## 7. Ответы (из таблиц)", ""]
    for b in ("K2", "K3"):
        r = s[(s["window"] == "M") & (s["hybrid"] == "X1") & (s["base"] == b)].iloc[0]
        L.append("- **X1 против %s, окно M.** Численно: %s (Δ %s, 90 %% интервал [%+.2f; %+.2f]). Статистически: %s (p DM %.4f, p Холма N %.4f)." % (
            b, "лучше" if r.delta_pct < 0 else "не лучше", pc(r.delta_pct), r.lo90, r.hi90, "подтверждено" if r.verdict.endswith("(p Холма < 0,05)") else "не подтверждено", r.p_one, r.p_holm_N))
    for b in ("K2", "K3"):
        x = y[(y["series"] == "X1") & (y["base"] == b)].sort_values("contrib_pct")
        L.append("- **Годы против %s.** Создают преимущество (вклад < 0): %s. Уменьшают (вклад > 0): %s." % (b, ", ".join("%s %+.3f" % (r.period, r.contrib_pct) for r in x[x["contrib_pct"] < 0].itertuples()) or "—", ", ".join("%s %+.3f" % (r.period, r.contrib_pct) for r in x[x["contrib_pct"] > 0].itertuples()) or "—"))
    for b in ("K2", "K3"):
        lo = c[(c["series"] == "X1") & (c["base"] == b) & (c["item"] == "loyo")]
        w = lo.loc[lo["value"].idxmax()]
        L.append("- **Концентрация против %s.** Лучший для X1 период — %s; без него Δ %s (с ним — %s)." % (b, w["key"], pc(w["value"]), pc(float(c[(c["series"] == "X1") & (c["base"] == b) & (c["key"] == "delta_total_pct")]["value"].iloc[0]))))
    for pair in ("X1|K2", "X1|K3"):
        T = top_groups(st, pair)
        L.append("- **Состояния, объясняющие проигрыш против %s.** %s." % (pair.split("|")[1], "; ".join("%s «%s»: доля пар %.1f %%, вклад %+.3f п. п., знак тот же в %d из %d лет" % (r.state, r.group, 100 * r.share, r.contrib_pct, r.years_same_sign, r.years_with_group) for r in T.itertuples())))
    L += ["", "Оговорки: все числа — историческая проверка и диагностика после просмотра; группы выбраны по вкладу уже на этих же данных (потолок, не отбор). Численное улучшение при p Холма ≥ 0,05 — не доказанное превосходство и не отсутствие выигрыша.", ""]
    return L


def build(t: dict) -> str:
    L = ["# W1_REPORT — сопоставимая таблица SBER, h = 1: архив + продолжение 2026 (лист W1, Ж-250 пп. 1–2)", "", STATUS, "",
         "Таблицы: `W1_JOIN.csv`, `W1_TABLE.csv`, `W1_STATS.csv`, `W1_YEARS.csv`, `W1_CONC.csv`, `W1_STATES.csv`, `W1_BIAS.csv`. Генератор: `models_v2/s16/x1_w1*.py`; режим `--check` только читает.", ""]
    return "\n".join(L + sec_join(t["join"]) + sec_table(t["table"]) + sec_stats(t["stats"]) + sec_years(t["years"], t["conc"]) + sec_states(t["states"], t["bias"]) + sec_answers(t["stats"], t["years"], t["states"], t["conc"])) + "\n"
