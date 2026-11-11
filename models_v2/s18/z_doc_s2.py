"""S-Z-EVAL: разделы 4–8 отчёта `S_Z_REPORT.md` (защитные условия, годы, срезы, статистика, параметры) — строки, собранные из таблиц; чисел, вписанных вручную, нет. В каждой таблице — границы дат и n."""
from __future__ import annotations

import pandas as pd

from models_v2.s18.z_doc_fmt import cap, fv, md, pv, sgn

PAIRS = (("S-RCKZ", "S-RCK"), ("S-RCKZ", "W5-G"), ("X1_R(S-RCKZ)", "X1"), ("S-RCKZ", "S-RCKZs"))
SLICES = (("no_day", "без t⁺ = 2022-02-24"), ("no_year", "без 2022 года"), ("no_top_day", "без самого влиятельного дня"), ("no_top_year", "без самого влиятельного календарного года"), ("trim", "симметричная обрезка 22 + 22 дат (только диагностика; даты выбраны по разности потерь пары)"))


def guard_lines(t: dict, tb: pd.DataFrame) -> list:
    g, n = t["guard"], tb["n"].iloc[0]
    return ["Пороги этапа S без изменений (значения — в столбце threshold); два набора — к W5-G и к S-RCK. D_M: %s; h1 — t⁺ 2018–2021, h2 — 2022–2026." % cap(tb["first_tplus"].iloc[0], tb["last_tplus"].iloc[0], n), "",
            md(g.assign(value=g["value"].map(fv), threshold=g["threshold"].map(fv)), {"set": "%s", "id": "%s", "condition": "%s", "value": "%s", "threshold": "%s", "passed": "%s"}), ""]


def years_lines(t: dict) -> list:
    y, out = t["years"], []
    for h, b in PAIRS[:2]:
        w = y[(y["series"] == h) & (y["base"] == b)]
        out += ["**%s против %s** (вклад = 100·Σ(l_h − l_b)/Σ l_b; Σ вкладов лет = Δ; минус — лучше):" % (h, b), md(w, {"period": "%s", "n": "%d", "q_series": "%.6f", "q_base": "%.6f", "delta_pct": "%+.4f", "contrib_pct": "%+.4f"}), ""]
    return out


def slices_lines(t: dict, tb: pd.DataFrame) -> list:
    c, rows = t["conc"], []
    for h, b in PAIRS[:2] + PAIRS[3:]:
        s = c[(c["series"] == h) & (c["base"] == b)]
        for it, lab in SLICES:
            v, n = s[s["item"] == it].iloc[0], s[(s["item"] == "n_slice") & (s["key"] == it)].iloc[0]
            rows.append({"пара": "%s к %s" % (h, b), "срез": lab, "ключ": "" if it == "trim" else v["key"], "дат": "%d" % n["value"], "контроль и знаменатель": "контроль %s; знаменатель — QLIKE %s на оставшихся %d датах" % (b, b, n["value"]), "Δ, %": sgn(v["value"])})
        top = s[s["item"] == "share_top1pct"].iloc[0]
        rows.append({"пара": "%s к %s" % (h, b), "срез": "доля выигрыша от 22 лучших дат (диагностика)", "ключ": top["key"], "дат": "%d" % tb["n"].iloc[0], "контроль и знаменатель": "контроль %s; суммарный выигрыш на всех датах" % b, "Δ, %": pv(top["value"])})
    return ["Срезы — только диагностика; основной Δ считается на всех датах D_M (%s)." % cap(tb["first_tplus"].iloc[0], tb["last_tplus"].iloc[0], tb["n"].iloc[0]), "", md(pd.DataFrame(rows), {c: "%s" for c in rows[0]}), ""]


def stats_lines(t: dict, cp: dict) -> list:
    st, k = t["stats"], cp["S.Z.eval"]
    m = st[st["kind"] == "main"]
    out = ["Лаг DM NW-%d, бутстрап стационарный (средний блок %d, B = %d, seed %d); Холм m = %d, N = %d, N_wide = %d; SPA — студентизированный (Hansen 2005), p_c; D_M: %s." % (k["lag_main"], k["boot_block"], k["boot"], k["boot_seed"], k["m_family"], k["n_holm"], k["n_holm_wide"], cap(m["first"].iloc[0], m["last"].iloc[0], m["n"].iloc[0])), "",
           md(m, {"hypothesis": "%s", "delta_pct": "%+.4f", "lo90": "%+.4f", "hi90": "%+.4f", "p_boot": "%.4f", "p_dm": "%.4f", "p_dm_lag5": "%.4f", "p_holm_N": "%.4f", "p_holm_Nwide": "%.4f", "p_spa_c": "%.4f", "p_spa_block": "%s", "verdict": "%s"}), "",
           "Половины (h1 — t⁺ 2018–2021, h2 — 2022–2026), без Холма:", md(st[st["kind"] == "window"], {"hypothesis": "%s", "window": "%s", "n": "%d", "first": "%s", "last": "%s", "delta_pct": "%+.4f", "lo90": "%+.4f", "hi90": "%+.4f", "p_dm": "%.4f"}), "",
           "Справочно (сырые p, критериев нет; S-RZ — абляция «только z»):", md(st[st["kind"] == "reference"], {"hypothesis": "%s", "n": "%d", "first": "%s", "last": "%s", "delta_pct": "%+.4f", "lo90": "%+.4f", "hi90": "%+.4f", "p_dm": "%.4f"}), ""]
    sp, mc, pw = t["spa"], t["mcs"], t["power"]
    out += ["SPA и StepM ((а) {S-RCKZ} к S-RCK; (б) {S-RCKZ, S-RZ} к W5-G; (в) {X1_R(S-RCKZ), X1_R(S-RZ)} к X1):", md(sp, {"kind": "%s", "block": "%s", "benchmark": "%s", "candidates": "%s", "T": "%.4f", "p_l": "%.4f", "p_c": "%.4f", "p_u": "%.4f", "t_stat": "%+.4f", "p_stepm": "%.4f"}), "",
            "MCS 90 %% (T_max), 12 рядов, только конечные положительные: %s." % cap(m["first"].iloc[0], m["last"].iloc[0], mc["n_used"].iloc[0]), md(mc, {"series": "%s", "mean_qlike": "%.6f", "p": "%.4f", "in_mcs": "%s"}), "",
            "Мощность (σ_LR — NW-8; n для мощности 0,5 / 0,8 при одностороннем α = 0,05; недостаточная мощность не означает, что улучшение невозможно):",
            md(pw, {"series": "%s", "base": "%s", "window": "%s", "n": "%d", "sigma_lr": "%.5f", "effect_pct": "%+.4f", "n_p05": "%.0f", "n_p08": "%.0f"}), ""]
    return out


def params_lines(t: dict) -> list:
    p = t["params"]
    mon = p[p["kind"].isin(["months", "lam", "clip", "n_clip", "z"])].pivot_table(index="variant", columns=["kind", "item"], values="value", sort=False)
    mon.columns = ["%s:%s" % c for c in mon.columns]
    thz = p[p["kind"] == "thz"]
    zt = p[p["kind"] == "ztilde_forecast"]
    return ["Помесячный выбор (λ*, w_λ и G по блокам, c, m, κ) — в `S_Z_SEL.csv`; здесь — сводка. Месяцев переоценки — активных (|P| ≥ 630); λ* и доля θ ≠ 0, c, m, κ, `n_clip` (доля обрезанных z̃ пула), даты с `z_ok` = 0 и `n_excl` > 0:", md(mon.reset_index(), {"variant": "%s"} | {c: "%.4g" for c in mon.columns}), "",
            "θ_Z по месяцам с λ* > 0 (среднее и разброс по месяцам; разброс по головам). θ разных лет несравнимы: знак и порядок компонент между годами не сохраняются (разд. 5.2):",
            md(thz.pivot_table(index=["variant", "key"], columns="item", values="value", sort=False).reset_index(), {"variant": "%s", "key": "%s", "mean_over_months": "%+.5f", "std_over_months": "%.5f", "mean_thsd_heads": "%.5f"}), "",
            "z̃ на днях прогноза активных месяцев по годам (З-5; μ, σ — по закрытому пулу месяца): доля обрезанных |z̃| > 3 и std z̃ после обрезки:",
            md(zt.pivot_table(index="key", columns="item", values="value", sort=True).reset_index().rename(columns={"key": "год"}), {"год": "%s", "share_clipped": "%.4f", "std_z0": "%.4f", "std_z1": "%.4f", "std_z2": "%.4f", "std_z3": "%.4f"}), "",
            "Распределение ln(C_Z/C_S-RCK) на активных датах (ex post, η_Z отдельно не сохранён) и корреляция с ln(RV/C_S-RCK) (ex post):",
            md(t["mech"][t["mech"]["section"].isin(["lnratio", "corr_ex_post"])], {"section": "%s", "item": "%s", "key": "%s", "value": "%.5f"}), ""]
