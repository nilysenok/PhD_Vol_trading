"""S-R-EVAL, документ (Ж-254 п. 16): разделы 1–3 отчёта `S_R_REPORT.md` — таблица рядов, процедура (П1, П2, абляции), статистика. Текст строится только из таблиц `S_R_*.csv` (`r_doc_fmt.read_all`).
Публичное API: `LABEL` (подписи рядов в порядке п. 16), `sec1(T)`, `sec2(T)`, `sec3(T)`."""
from __future__ import annotations

from models_v2.s18.r_doc_fmt import c_, dm_line, md, stat, word, yn

LABEL = {"K1": "HAR-J (K1)", "K2": "XGBoost (K2)", "K3": "LightGBM (K3)", "X1": "X1 (контрольная версия)", "W5-G": "W5-G (контрольная версия)", "W8-G": "W8-G (экспериментальный контроль)",
         "S-A2": "S-A2 (пилот A, прежняя процедура)", "S-CK": "S-CK (компонентный гибрид, прежняя процедура)", "S-RA2": "S-RA2 (A2 + R, основной кандидат)", "S-RCK": "S-RCK (C-K + R, кандидат)",
         "S-RA2b": "S-RA2b (абляция: R без обрезки)", "S-RCKb": "S-RCKb (абляция: R без обрезки)", "S-RA2c": "S-RA2c (абляция: обрезка без блоков)", "S-RCKc": "S-RCKc (абляция: обрезка без блоков)",
         "B_mix": "B_mix = √(K2·K3) (справочно)", "ZB": "ZB (справочно)"}
BASES, CANDS = ("X1", "W5-G", "K2", "K3"), ("S-RA2", "S-RCK")
MAIN = [k for k in LABEL if k not in ("B_mix", "ZB")]


def sec1(T: dict) -> str:
    t = T["table"].set_index("series")
    rows = [[LABEL[k], c_(t.at[k, "qlike"], 6)] + ["—" if k == b else c_(t.at[k, "d_%s_pct" % b], 4, True) for b in BASES] + ["%s…%s" % (t.at[k, "first_tplus"], t.at[k, "last_tplus"]), int(t.at[k, "n"])] for k in MAIN]
    a = md(["Ряд", "QLIKE", "Δ к X1, %", "Δ к W5-G, %", "Δ к K2, %", "Δ к K3, %", "даты t⁺", "n"], rows)
    lay = [[k, c_(t.at[k, "qlike"], 6), c_(t.at[k, "d_X1_pct"], 4, True), c_(t.at[k, "d_W5-G_pct"], 4, True), "%s…%s" % (t.at[k, "first_tplus"], t.at[k, "last_tplus"]), int(t.at[k, "n"])] for k in t.index if k.startswith("X1_")]
    b = md(["Слой доверия X1 над C", "QLIKE", "Δ к X1 (М1), %", "Δ к W5-G, %", "даты t⁺", "n"], lay)
    return ("## 1. Таблица на общем периоде D_M\n\n" + dm_line(T) + " QLIKE — среднее дневных потерь без обрезки; Δ в % QLIKE базы, минус — лучше. S-A2 и S-CK — прежние ряды пилотов A и C (старая процедура), показаны для сравнения.\n\n" + a +
            "\n\nОтдельно: слой доверия X1 над каналом C (до слоя перерыва W5) — для М1 (X1_R против X1); X1_A(S-A2) и X1_V(S-CK) — слои прежних пилотов.\n\n" + dm_line(T) + "\n\n" + b)


def sec2(T: dict) -> str:
    rows = [[r.pid, r.candidate, r.old, c_(r.qlike_new, 6), c_(r.qlike_old, 6), c_(r.delta_pct, 4, True), c_(r.delta_trim_pct, 4, True), yn(r.cond_qlike), yn(r.cond_trim), yn(r.passed), c_(r.p_holm_N, 4), c_(r.p_spa_c, 4),
             "подтверждено" if str(r.confirmed) == "True" else "не подтверждено", word(r.delta_pct)] for r in T["proc"].itertuples()]
    a = md(["П", "кандидат", "прежний ряд", "QLIKE нового", "QLIKE прежнего", "Δ, %", "Δ_trim, %", "QLIKE ниже", "Δ_trim < 0", "П выполнено", "p Холма N", "p SPA", "вывод П", "новый против прежнего"], rows)
    ab = []
    for r in T["stats"][T["stats"]["kind"] == "ablation"].itertuples():
        ab.append([r.hypothesis, c_(r.delta_pct, 4, True), "[%s; %s]" % (c_(r.lo90, 3, True), c_(r.hi90, 3, True)), c_(r.p_boot, 4), c_(r.p_dm, 4)])
    pt = T["proc"].set_index("pid")
    need = "; ".join("%s: R < Rb %s, R < Rc %s ⇒ «нужны оба элемента» %s" % (p, yn(pt.at[p, "r_lt_rb"]), yn(pt.at[p, "r_lt_rc"]), yn(pt.at[p, "both_needed"])) for p in pt.index)
    return ("## 2. Процедура: П1, П2 и абляции\n\n" + dm_line(T) + " П — QLIKE(F_R) < QLIKE(F_прежний) ∧ Δ_trim(F_R против F_прежний) < 0; «подтверждено» = П ∧ p Холма (N программы = 846, m = 2) < 0,05 ∧ p SPA < 0,05.\n\n" + a +
            "\n\nВывод «нужны оба элемента» — только при R < Rb и R < Rc (показ, не условие): " + need + ".\n\nАблации (сырые p, справочно; Δ — первый ряд против второго, минус — первый лучше):\n\n" + dm_line(T) + "\n\n" +
            md(["Сравнение", "Δ, %", "90 % интервал", "p_boot", "p DM NW-8"], ab))


def _hrow(T, hid, kind="main") -> list:
    s = stat(T, hid, kind)
    return [hid, c_(s["delta_pct"], 4, True), "[%s; %s]" % (c_(s["lo90"], 3, True), c_(s["hi90"], 3, True)), c_(s["p_boot"], 4), c_(s["p_dm"], 4), c_(s["p_dm_lag5"], 4), c_(s["p_holm_m"], 4), c_(s["p_holm_N"], 4), c_(s["p_holm_Nwide"], 4),
            c_(s["p_spa_c"], 4), c_(s["p_stepm"], 4), s["verdict"]]


def sec3(T: dict) -> str:
    hdr = ["Гипотеза", "Δ, %", "90 % интервал", "p_boot", "p DM NW-8", "p DM NW-5 (справ.)", "Холм m = 8", "Холм N = 846", "Холм N_wide = 3397", "p SPA (c)", "p StepM", "вывод"]
    a = md(hdr, [_hrow(T, "%s|%s" % (c, b)) for c in CANDS for b in BASES])
    p2 = md(["П"] + hdr[1:7] + ["Холм m = 2", "Холм N = 846", "Холм N_wide = 3397", "p SPA (c)", "p StepM", "вывод"], [_hrow(T, "П%d" % i, "proc") for i in (1, 2)])
    sp = T["spa"]
    fam = md(["База (бенчмарк)", "кандидаты", "T", "p_l", "p_c (основной)", "p_u"], [[r.base, r.candidates.replace("|", ", "), c_(r.T, 4), c_(r.p_l, 4), c_(r.p_c, 4), c_(r.p_u, 4)] for r in sp[sp["kind"].isin(["family", "proc"])].itertuples()])
    rej = [r.candidates for r in sp[sp["kind"] == "candidate"].itertuples() if str(r.stepm_reject) == "True"]
    m = T["mcs"]
    inn = [r.series for r in m.itertuples() if str(r.in_mcs) == "True"]
    return ("## 3. Статистика\n\n" + dm_line(T) + " DM — Ньюи–Уэст, лаг 8 (NW-94, основной; NW-5 — справочно), односторонний «кандидат лучше»; интервал и p_boot — стационарный бутстрап (средний блок 20, B = 10 000, seed 0); "
            "Холм — по p DM NW-8: m = 8 и m = 2 (семейства), N = 846 и N_wide = 3397 (счётчик программы; недостающие до N p = 1); SPA — студентизированный Hansen (2005), основной p_c; StepM — Romano–Wolf. Против X1 сравнивается слой X1_R (М1), против остальных — F.\n\n" +
            a + "\n\nПроцедура (П1, П2):\n\n" + dm_line(T) + "\n\n" + p2 + "\n\nSPA по базам (кандидаты {S-RA2, S-RCK}; бенчмарк согласован со слоем) и по П (один кандидат против прежнего ряда).\n\n" + dm_line(T) + "\n\n" + fam +
            "\n\nStepM (α = 0,05): " + ("отвергнуты — " + ", ".join(rej) if rej else "превосходящих кандидатов нет (ни одна гипотеза не отвергнута)") + ".\n\n"
            "MCS 90 %% (T_max, B = 10 000, только конечные положительные прогнозы; n = %d): в множество входят %d из %d рядов (%s); наименьший p = %s." % (int(m["n_used"].iloc[0]), len(inn), len(m), ", ".join(inn), c_(m["p"].min(), 4)))
