"""S-RCD-EVAL: разделы 3–5 отчёта `S_RCD_REPORT.md` (H-RCD1 и защита, годы и половины, срезы; статистика, нулевое распределение, параметры по месяцам, RCD-DEC) — строки, собранные из таблиц; чисел, вписанных вручную, нет.
В каждой таблице — границы дат и n; в каждом срезе — контроль, число дат и знаменатель."""
from __future__ import annotations

import pandas as pd

from models_v2.s18 import rd_eval_null as RN
from models_v2.s18.rd_doc_s1 import dm
from models_v2.s18.z_doc_fmt import cap, fv, md, pick, pv, sgn

PAIRS = (("S-RCD", "S-CDF"), ("S-RCD", "S-RCK"), ("S-RCD", "W5-G"))
SLICES = (("no_day", "без t⁺ = 2022-02-24"), ("no_year", "без 2022 года"), ("no_top_day", "без самого влиятельного дня"), ("no_top_year", "без самого влиятельного календарного года"), ("trim", "симметричная обрезка 22 + 22 дат (только диагностика; даты выбраны по разности потерь пары)"))


def h1_lines(t: dict, tb: pd.DataFrame) -> list:
    d, st = t["dec"].set_index("id"), t["stats"]
    h = pick(st, kind="main", hypothesis="S-RCD|S-CDF")
    return ["**H-RCD1** (QLIKE(S-RCD) < QLIKE(S-CDF), D_M: %s, все даты, обычная QLIKE): Δ = %s %% QLIKE(S-CDF), 90 %% интервал [%s; %s], p_boot %s, DM NW-8 %s, p Холма (N = %d) %s, p SPA (а) %s — **%s**; условие H-RCD1 %s." % (
        cap(h["first"], h["last"], h["n"]), sgn(h["delta_pct"]), sgn(h["lo90"]), sgn(h["hi90"]), pv(h["p_boot"]), pv(h["p_dm"]), int(h["n_holm"]), pv(h["p_holm_N"]), pv(h["p_spa_c"]), h["verdict"], "выполнено численно" if d.loc["H-RCD1", "passed"] else "не выполнено"), ""]


def guard_lines(t: dict, tb: pd.DataFrame) -> list:
    g = t["guard"]
    return ["Пороги этапа S без изменений (в столбце threshold); два набора — к W5-G и к S-RCK (S-RCK — кандидат-контроль, календарных переключений нет). D_M: %s; h1 — t⁺ 2018–2021, h2 — 2022–2026." % cap(tb["first_tplus"].iloc[0], tb["last_tplus"].iloc[0], tb["n"].iloc[0]), "",
            md(g.assign(value=g["value"].map(fv), threshold=g["threshold"].map(fv)), {"set": "%s", "id": "%s", "condition": "%s", "value": "%s", "threshold": "%s", "passed": "%s"}), ""]


def years_lines(t: dict) -> list:
    y, out = t["years"], []
    for h, b in PAIRS:
        w = y[(y["series"] == h) & (y["base"] == b)]
        out += ["**%s против %s** (вклад = 100·Σ(l_h − l_b)/Σ l_b; Σ вкладов лет = Δ; половины h1 — t⁺ 2018–2021, h2 — 2022–2026; минус — лучше):" % (h, b), md(w, {"period": "%s", "n": "%d", "q_series": "%.6f", "q_base": "%.6f", "delta_pct": "%+.4f", "contrib_pct": "%+.4f"}), ""]
    return out


def slices_lines(t: dict, tb: pd.DataFrame) -> list:
    c, rows = t["conc"], []
    for h, b in PAIRS:
        s = c[(c["series"] == h) & (c["base"] == b)]
        for it, lab in SLICES:
            v, n = s[s["item"] == it].iloc[0], s[(s["item"] == "n_slice") & (s["key"] == it)].iloc[0]
            rows.append({"пара": "%s к %s" % (h, b), "срез": lab, "ключ": "" if it == "trim" else v["key"], "дат": "%d" % n["value"], "контроль и знаменатель": "контроль %s; знаменатель — QLIKE %s на оставшихся %d датах" % (b, b, n["value"]), "Δ, %": sgn(v["value"])})
        top = s[s["item"] == "share_top1pct"].iloc[0]
        rows.append({"пара": "%s к %s" % (h, b), "срез": "доля выигрыша от 22 лучших дат (диагностика)", "ключ": top["key"], "дат": "%d" % tb["n"].iloc[0], "контроль и знаменатель": "контроль %s; суммарный выигрыш на всех датах" % b, "Δ, %": pv(top["value"])})
    return ["Срезы — только диагностика; основной Δ считается на всех датах D_M (%s)." % cap(tb["first_tplus"].iloc[0], tb["last_tplus"].iloc[0], tb["n"].iloc[0]), "", md(pd.DataFrame(rows), {c: "%s" for c in rows[0]}), ""]


def stats_lines(t: dict, cp: dict) -> list:
    st, k = t["stats"], cp["S.RCD.eval"]
    m = st[st["kind"] == "main"]
    cols = {"hypothesis": "%s", "delta_pct": "%+.4f", "lo90": "%+.4f", "hi90": "%+.4f", "p_boot": "%.4f", "p_dm": "%.4f", "p_dm_lag5": "%.4f", "p_holm_N": "%.4f", "p_holm_Nwide": "%.4f", "p_spa_c": "%.4f", "p_spa_block": "%s", "verdict": "%s"}
    small = {"hypothesis": "%s", "window": "%s", "n": "%d", "first": "%s", "last": "%s", "delta_pct": "%+.4f", "lo90": "%+.4f", "hi90": "%+.4f", "p_dm": "%.4f"}
    out = ["Лаг DM NW-%d, бутстрап стационарный (средний блок %d, B = %d, seed %d); Холм m = %d, N = %d, N_wide = %d; SPA — студентизированный (Hansen 2005), p_c; D_M: %s." % (k["lag_main"], k["boot_block"], k["boot"], k["boot_seed"], k["m_family"], k["n_holm"], k["n_holm_wide"], cap(m["first"].iloc[0], m["last"].iloc[0], m["n"].iloc[0])), "",
           md(m, cols), "", "Половины (без Холма):", md(st[st["kind"] == "window"], small), "",
           "Контрасты разложения (сырые p; Холм N по своим 4 p, справочно):", md(st[st["kind"] == "decomp"], {"hypothesis": "%s", "n": "%d", "first": "%s", "last": "%s", "delta_pct": "%+.4f", "lo90": "%+.4f", "hi90": "%+.4f", "p_dm": "%.4f", "p_holm_m": "%.4f", "p_holm_N": "%.4f"}), ""]
    sp, mc, pw = t["spa"], t["mcs"], t["power"]
    return out + ["SPA и StepM ((а) {S-RCD} к S-CDF; (б) {S-RCD} к S-RCK; (в) {S-RCD, S-CDF, S-CKF} к W5-G; (г) {X1_R(S-RCD), X1_V(S-CDF), X1_V(S-CKF)} к X1):", md(sp, {"kind": "%s", "block": "%s", "benchmark": "%s", "candidates": "%s", "T": "%.4f", "p_l": "%.4f", "p_c": "%.4f", "p_u": "%.4f", "t_stat": "%+.4f", "p_stepm": "%.4f"}), "",
                  "MCS 90 %% (T_max), 15 рядов, только конечные положительные: %s." % cap(m["first"].iloc[0], m["last"].iloc[0], mc["n_used"].iloc[0]), md(mc, {"series": "%s", "mean_qlike": "%.6f", "p": "%.4f", "in_mcs": "%s"}), "",
                  "Мощность (σ_LR — NW-8; n для мощности 0,5 / 0,8 при одностороннем α = 0,05; недостаточная мощность не означает, что улучшение невозможно):", md(pw, {"series": "%s", "base": "%s", "window": "%s", "n": "%d", "sigma_lr": "%.5f", "effect_pct": "%+.4f", "n_p05": "%.0f", "n_p08": "%.0f"}), ""]


def null_lines(t: dict) -> list:
    s, m = RN.read()
    q = lambda i, st: float(m[(m["item"] == i) & (m["stat"] == st)]["value"].iloc[0])
    lo, hi = float(dm(t, "noise", "abs_q05")["value"]), float(dm(t, "noise", "abs_q95")["value"])
    rows = [{"величина": lab, "реальный Δ по модулю, %": "%.4f" % abs(dm(t, "noise_real", k)["value"]), "5 % модуля нуля": "%.4f" % lo, "95 % модуля нуля": "%.4f" % hi, "правило": dm(t, "noise_rule", k)["key"]}
            for lab, k in (("Δ(S-RCD к S-CDF), D_M", "RCD_vs_CDF|D_M"), ("Δ(S-RCD к S-CDF), активные даты", "RCD_vs_CDF|active"), ("Δ(S-RCD к S-RCK), D_M", "RCD_vs_RCK|D_M"), ("Δ(S-RCD к S-RCK), активные даты", "RCD_vs_RCK|active"))]
    pm = pick(t["params"], variant="S-RCD", kind="months", item="share_theta_nonzero")
    return ["Объявленная диагностика (дирижёр, 12.11 02:47, до QLIKE; не условие RCD-DEC): полная процедура S-RCD (`rd_walk`, режим R, форма C-D) на синтетике (цель ln RV = ln F⁰ + шум без связи с признаками, логнормальный σ 0,5 и 1 %% скачков ×30), seeds шума %d–%d, n = %d шумов." % (RN.SEEDS[0], RN.SEEDS[-1], len(s)),
            "QLIKE(R) − QLIKE(θ = 0), %% QLIKE(θ = 0) на датах активных месяцев: среднее %s, СКО %s, 5 %% %s, 95 %% %s; доля месяцев θ ≠ 0 — среднее %.3f (реальный S-RCD: %.3f)." % (sgn(q("qlike_rel_pct", "mean")), sgn(q("qlike_rel_pct", "std")), sgn(q("qlike_rel_pct", "q05")), sgn(q("qlike_rel_pct", "q95")), q("share_theta_nonzero", "mean"), pm["value"]), "",
            md(pd.DataFrame(rows), {c: "%s" for c in rows[0]}), "",
            "Правило: |Δ| внутри [5 %%; 95 %%] распределения |Δ_null| по %d шумам ⇒ «не отличим от шума процедуры»; Δ_null — QLIKE(R) − QLIKE(θ = 0) на активных датах." % len(s),
            "Оговорка о соразмерности: нуль — разность одной процедуры с θ = 0, а реальные контрасты (S-RCD к S-CDF, к S-RCK) — разности двух обученных процедур, их шум примерно в √2 шире; при %d шумах квантили 5 и 95 %% — это фактически минимум и максимум |Δ_null|. Поэтому вывод справочный, не тест." % len(s),
            "", "Нулевое распределение по seeds шума:", md(s, {"seed": "%d", "n_active": "%d", "qlike_rel_pct": "%+.4f", "share_theta_nonzero": "%.3f"}), ""]


def params_lines(t: dict) -> list:
    p = t["params"]
    mon = p[p["kind"].isin(["months", "lam", "clip", "n_small_pool", "spread"])].pivot_table(index="variant", columns=["kind", "item"], values="value", sort=False)
    mon.columns = ["%s:%s" % c for c in mon.columns]
    ab = p[p["kind"].isin(["l0", "at_bound"])].pivot_table(index=["variant", "kind"], columns="item", values="value", sort=False)
    ba = lambda v, k: float(pick(p, variant=v, kind="at_bound", item=k)["value"]) * 100.0
    act, nzm = pick(p, variant="S-RCD", kind="months", item="active")["value"], pick(p, variant="S-RCD", kind="months", item="share_theta_nonzero")["value"]
    prose = "ρ_c у S-RCD на границе [0; 1] в %.1f %% месяцев с θ ≠ 0 (%d активных месяцев, доля θ ≠ 0 — %.3f); ρ_p — в %.1f %%; у S-CDF ρ_c — в %.1f %%, ρ_p — в %.1f %% (все месяцы S-CDF с θ ≠ 0)." % (ba("S-RCD", "rho_c"), act, nzm, ba("S-RCD", "rho_p"), ba("S-CDF", "rho_c"), ba("S-CDF", "rho_p"))
    return [prose, "", "Помесячный выбор (λ*, w_λ и G по блокам, c, m, κ; val_* у S-CDF и S-CKF) — в `S_RCD_SEL.csv`; здесь сводка по месяцам переоценки (только активные):", md(mon.reset_index(), {"variant": "%s"} | {c: "%.4g" for c in mon.columns}), "",
            "‖θ‖₀ по месяцам (l0: min / медиана / max) и доля месяцев с параметром на границе формы (at_bound; у C-D ρ ∈ [0; 1]):", md(ab.reset_index(), {c: ("%s" if c in ("variant", "kind") else "%.4g") for c in ab.reset_index().columns}), ""]


def dec_lines(t: dict) -> list:
    d = t["dec"].set_index("id")
    rows = t["dec"].iloc[:-1].copy()
    return [md(rows.assign(value=rows["value"].map(fv)), {"id": "%s", "condition": "%s", "value": "%s", "passed": "%s"}), "",
            "**RCD-DEC (код %s):** %s" % (d.loc["RCD-DEC", "value"], d.loc["RCD-DEC", "text"]), "",
            "Доля s и подтверждённость H-RCD1 показаны при любом исходе; S-RCK не заменяется без решения владельца (Ж-259 п. 3)."]
