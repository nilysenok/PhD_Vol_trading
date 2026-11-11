"""S-A-EVAL, документ (Ж-254 п. 16; `S_A_SPEC.md`, разд. 8): разделы 1–3 отчёта `S_A_REPORT.md` — таблица рядов, статистика и мощность, годы и вклады. Текст строится только из таблиц `S_A_*.csv` (`a_doc_fmt.read_all`).
Публичное API: `sec1(T)` — таблица рядов (HAR-J / XGBoost / LightGBM / X1 / W5-G / W8-G как экспериментальный контроль / S-A0 / S-A1 / S-A2 / S-A2p) и слой X1_A против X1 (М1);
`sec2(T)` — 12 гипотез (интервалы, p DM NW-8 и NW-5, p_boot, Холм m = 12 / N = 831 / N_wide = 3367, SPA, StepM), семейства SPA, MCS, мощность; `sec3(T)` — вклады периодов и половин (Σ = Δ)."""
from __future__ import annotations

from models_v2.s18.a_doc_fmt import c_, dm_line, md, one, stat, yrs

LABEL = {"K1": "HAR-J (K1)", "K2": "XGBoost (K2)", "K3": "LightGBM (K3)", "X1": "X1 (контрольная версия)", "W5-G": "W5-G (контрольная версия)", "W8-G": "W8-G (экспериментальный контроль)",
         "S-A0": "S-A0 (новый гибрид)", "S-A1": "S-A1 (новый гибрид)", "S-A2": "S-A2 (новый гибрид, основная гипотеза)", "S-A2p": "S-A2p (контроль: время дня переставлено)",
         "B_mix": "B_mix = √(K2·K3) (справочно)", "ZB": "ZB (справочно)"}
BASES = ("X1", "W5-G", "K2", "K3")
CANDS = ("S-A0", "S-A1", "S-A2")


def sec1(T: dict) -> str:
    t = T["table"].set_index("series")
    rows = [[LABEL[k], c_(t.at[k, "qlike"], 6)] + ["—" if k == b else c_(t.at[k, "d_%s_pct" % b], 4, True) for b in BASES] + ["%s…%s" % (t.at[k, "first_tplus"], t.at[k, "last_tplus"]), int(t.at[k, "n"])]
            for k in LABEL]
    a = md(["Ряд", "QLIKE", "Δ к X1, %", "Δ к W5-G, %", "Δ к K2, %", "Δ к K3, %", "даты t⁺", "n"], rows)
    lay = []
    for v in CANDS:
        s = stat(T, v + "|X1")
        lay.append(["X1_A(%s)" % v, c_(s["qlike_h"], 6), c_(s["delta_pct"], 4, True), "[%s; %s]" % (c_(s["lo90"], 3, True), c_(s["hi90"], 3, True)), c_(s["p_dm"], 4), "%s…%s" % (s["first"], s["last"]), int(s["n"])])
    p = t.loc["X1_A(S-A2p)"]
    lay.append(["X1_A(S-A2p) (контроль)", c_(p["qlike"], 6), c_(p["d_X1_pct"], 4, True), "—", "—", "%s…%s" % (p["first_tplus"], p["last_tplus"]), int(p["n"])])
    b = md(["Слой доверия X1 над C_A", "QLIKE", "Δ к X1 (М1), %", "90 % интервал", "p DM (NW-8)", "даты t⁺", "n"], lay)
    return ("## 1. Таблица на общем периоде D_M\n\n" + dm_line(T) + " QLIKE — среднее дневных потерь; Δ в % QLIKE базы, минус — лучше.\n\n" + a +
            "\n\nОтдельно: слой доверия X1 над новой компонентой C_A (до слоя перерыва W5) против X1 — критерий М1.\n\n" + dm_line(T) + "\n\n" + b)


def _hrow(T, c, b) -> list:
    s = stat(T, "%s|%s" % (c, b))
    return [("X1_A(%s)" % c if b == "X1" else c) + " к " + b, c_(s["delta_pct"], 4, True), "[%s; %s]" % (c_(s["lo90"], 3, True), c_(s["hi90"], 3, True)), c_(s["p_boot"], 4), c_(s["p_dm"], 4), c_(s["p_dm_lag5"], 4),
            c_(s["p_holm_m12"], 4), c_(s["p_holm_N"], 4), c_(s["p_holm_Nwide"], 4), c_(s["p_spa_c"], 4), c_(s["p_stepm"], 4), s["verdict"]]


def _power(T) -> str:
    p = T["power"]
    rows = [[r.candidate, r.pair, r.window, int(r.n), int(r.lag), c_(r.sigma_lr, 4), c_(r.effect_pct, 4, True), c_(r.n_p05, 0), c_(r.n_p08, 0), c_(r.n_p05_hyp1pct, 0), c_(r.n_p08_hyp1pct, 0)] for r in p.itertuples()]
    return md(["Кандидат", "пара", "окно", "n", "лаг", "σ_LR", "эффект, %", "n для мощности 0,5", "n для 0,8", "n при −1 %: 0,5", "n при −1 %: 0,8"], rows)


def sec2(T: dict) -> str:
    hdr = ["Гипотеза", "Δ, %", "90 % интервал", "p_boot", "p DM NW-8", "p DM NW-5 (справ.)", "Холм m = 12", "Холм N = 831", "Холм N_wide = 3367", "p SPA (c)", "p StepM", "вывод"]
    a = md(hdr, [_hrow(T, c, b) for c in CANDS for b in BASES])
    sp = T["spa"]
    fam = sp[sp["kind"] == "family"]
    b = md(["База (бенчмарк)", "кандидаты", "T", "p_l", "p_c (основной)", "p_u"], [[r.base, r.candidates.replace("|", ", "), c_(r.T, 4), c_(r.p_l, 4), c_(r.p_c, 4), c_(r.p_u, 4)] for r in fam.itertuples()])
    cand = sp[sp["kind"] == "candidate"]
    rej = [r.candidates for r in cand.itertuples() if str(r.stepm_reject) == "True"]
    m = T["mcs"]
    inn = [r.series for r in m.itertuples() if str(r.in_mcs) == "True"]
    return ("## 2. Статистика\n\n" + dm_line(T) + " DM — Ньюи–Уэст, лаг 8 (правило NW-94, основной; NW-5 — справочно, как W1–W8), односторонний «кандидат лучше»; интервал и p_boot — стационарный бутстрап (средний блок 20, B = 10 000, seed 0); "
            "Холм — по p DM NW-8: m = 12 (семейство этапа), N = 831 и N_wide = 3367 (счётчик программы; недостающие до N p = 1); SPA — студентизированный Hansen (2005), основной p_c; StepM — Romano–Wolf. "
            "Против X1 сравнивается слой X1_A (М1), против остальных — F_A.\n\n" + a + "\n\nSPA по базам (кандидаты {A0, A1, A2}; бенчмарк согласован со слоем кандидатов).\n\n" + dm_line(T) + "\n\n" + b +
            "\n\nStepM (α = 0,05): " + ("отвергнуты — " + ", ".join(rej) if rej else "превосходящих кандидатов нет (ни одна гипотеза не отвергнута)") + ".\n\n"
            "MCS 90 %% (T_max, B = 10 000, только конечные положительные прогнозы; n = %d): в множество входят %d из %d рядов (%s); наименьший p = %s.\n\n" % (int(m["n_used"].iloc[0]), len(inn), len(m), ", ".join(inn), c_(m["p"].min(), 4)) +
            "Мощность: σ_LR разности потерь (NW-94), наблюдаемый эффект, n для мощности 0,5 и 0,8 при α = 0,05 односторонний; «—» — эффект не в пользу кандидата или нулевой (n не определено); справа — гипотетический эффект −1 % QLIKE базы. "
            "Недостаточная мощность не доказывает, что улучшение невозможно.\n\n" + dm_line(T) + " Окно «без 2022» — n = %d.\n\n" % int(T["power"][T["power"]["window"] == "без 2022"]["n"].iloc[0]) + _power(T))


def _ytab(T, cand) -> str:
    per = [p for p in T["years"]["period"].unique()]
    rows = []
    for p in per:
        r1, r2 = yrs(T, "X1_A(%s)" % cand, "X1", p), yrs(T, cand, "W5-G", p)
        rows.append([p, int(r1["n"]), c_(r1["delta_pct"], 4, True), c_(r1["contrib_pct"], 4, True), c_(r2["delta_pct"], 4, True), c_(r2["contrib_pct"], 4, True)])
    y = T["years"]
    for k, (h, b) in (("М1", ("X1_A(%s)" % cand, "X1")), ("М2", (cand, "W5-G"))):
        s = y[(y["series"] == h) & (y["base"] == b) & (y["kind"] == "period")]["contrib_pct"].sum()
        rows.append(["Σ вкладов периодов (%s)" % k, "", "", c_(s, 4, True) if k == "М1" else "", "", c_(s, 4, True) if k == "М2" else ""])
    return md(["Период (t⁺)", "n", "Δ М1, %", "вклад М1, п. п.", "Δ М2, %", "вклад М2, п. п."], rows)


def sec3(T: dict) -> str:
    d1, d2 = (one(T["years"], series="X1_A(S-A2)", base="X1", period="2018")["delta_total_pct"], one(T["years"], series="S-A2", base="W5-G", period="2018")["delta_total_pct"])
    return ("## 3. Годы и вклады\n\n" + dm_line(T) + " Вклад периода c_p = 100·Σ(l_кандидат − l_база)/(n·Q_база), п. п. QLIKE базы; Σ вкладов периодов = Δ на D_M (тождество проверяется кодом, допуск 1e-12); "
            "половины: h1 = 2018–2021 (n = 1010), h2 = 2022–2026 (n = 1188). М1 — X1_A против X1, М2 — F_A против W5-G. Победа в каждом году не требуется; худшие годы видны в таблицах.\n\n"
            "Δ полного периода: S-A2 М1 = %s %%, М2 = %s %%.\n\n" % (c_(d1, 4, True), c_(d2, 4, True)) + "### S-A2\n\n" + _ytab(T, "S-A2") + "\n\n### S-A1\n\n" + _ytab(T, "S-A1") +
            "\n\nS-A0 и остальные базы (K2, K3) — в `S_A_YEARS.csv`.")
