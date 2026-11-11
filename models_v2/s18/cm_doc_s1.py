"""S-C-EVAL, документ (Ж-254 п. 16): разделы 1–3 отчёта `S_C_REPORT.md` — таблица рядов, статистика и мощность, годы и концентрация. Текст строится только из таблиц `S_C_*.csv` (`cm_doc_fmt.read_all`).
Публичное API: `LABEL` (подписи рядов в порядке п. 16), `sec1(T)` — таблица (K1 / K2 / K3 / X1 / W5-G / W8-G как экспериментальный контроль / S-A2 / S-CD / S-CK / S-CKA / S-C7 / S-CKL) и слой X1_C против X1;
`sec2(T)` — 8 гипотез (интервалы, p DM NW-8 и NW-5, Холм m = 8 / N = 836 / N_wide = 3377, SPA, StepM), S-CD против баз, семейства SPA, MCS, мощность; `sec3(T)` — годы, половины и концентрация."""
from __future__ import annotations

from models_v2.s18.cm_doc_fmt import c_, cnc, dm_line, md, one, skey, stat, yrs

LABEL = {"K1": "HAR-J (K1)", "K2": "XGBoost (K2)", "K3": "LightGBM (K3)", "X1": "X1 (контрольная версия)", "W5-G": "W5-G (контрольная версия)", "W8-G": "W8-G (экспериментальный контроль)",
         "S-A2": "S-A2 (пилот A, не прошёл)", "S-CD": "S-CD (прямая коррекция, контроль структуры)", "S-CK": "S-CK (компонентный гибрид, основной кандидат)", "S-CKA": "S-CKA (+ компонентная потеря)",
         "S-C7": "S-C7 (контроль K7-класса)", "S-CKL": "S-CKL (контроль: разложение сдвинуто на 252 сессии)", "B_mix": "B_mix = √(K2·K3) (справочно)", "ZB": "ZB (справочно)"}
BASES, CANDS, VARS = ("X1", "W5-G", "K2", "K3"), ("S-CK", "S-CKA"), ("S-CD", "S-CK", "S-CKA", "S-C7", "S-CKL")


def sec1(T: dict) -> str:
    t = T["table"].set_index("series")
    rows = [[LABEL[k], c_(t.at[k, "qlike"], 6)] + ["—" if k == b else c_(t.at[k, "d_%s_pct" % b], 4, True) for b in BASES] + ["%s…%s" % (t.at[k, "first_tplus"], t.at[k, "last_tplus"]), int(t.at[k, "n"])] for k in LABEL]
    a = md(["Ряд", "QLIKE", "Δ к X1, %", "Δ к W5-G, %", "Δ к K2, %", "Δ к K3, %", "даты t⁺", "n"], rows)
    lay = []
    for v in VARS:
        p = t.loc["X1_C(%s)" % v]
        lay.append(["X1_C(%s)" % v, c_(p["qlike"], 6), c_(p["d_X1_pct"], 4, True), c_(p["d_W5-G_pct"], 4, True), "%s…%s" % (p["first_tplus"], p["last_tplus"]), int(p["n"])])
    b = md(["Слой доверия X1 над C_V", "QLIKE", "Δ к X1 (М1), %", "Δ к W5-G, %", "даты t⁺", "n"], lay)
    return ("## 1. Таблица на общем периоде D_M\n\n" + dm_line(T) + " QLIKE — среднее дневных потерь; Δ в % QLIKE базы, минус — лучше. S-A2 — пилот A (не прошёл), показан для сравнения.\n\n" + a +
            "\n\nОтдельно: слой доверия X1 над новым каналом C_V (до слоя перерыва W5) против X1 — критерий М1.\n\n" + dm_line(T) + "\n\n" + b)


def _hrow(T, hid, kind, lab) -> list:
    s = stat(T, hid, kind)
    return [lab, c_(s["delta_pct"], 4, True), "[%s; %s]" % (c_(s["lo90"], 3, True), c_(s["hi90"], 3, True)), c_(s["p_boot"], 4), c_(s["p_dm"], 4), c_(s["p_dm_lag5"], 4), c_(s["p_holm_m"], 4), c_(s["p_holm_N"], 4),
            c_(s["p_holm_Nwide"], 4), c_(s["p_spa_c"], 4), c_(s["p_stepm"], 4), s["verdict"]]


def _power(T) -> str:
    rows = [[r.candidate, r.pair, r.window, int(r.n), int(r.lag), c_(r.sigma_lr, 4), c_(r.effect_pct, 4, True), c_(r.n_p05, 0), c_(r.n_p08, 0), c_(r.n_p05_hyp1pct, 0), c_(r.n_p08_hyp1pct, 0)] for r in T["power"].itertuples()]
    return md(["Кандидат", "пара", "окно", "n", "лаг", "σ_LR", "эффект, %", "n для мощности 0,5", "n для 0,8", "n при −1 %: 0,5", "n при −1 %: 0,8"], rows)


def sec2(T: dict) -> str:
    hdr = ["Гипотеза", "Δ, %", "90 % интервал", "p_boot", "p DM NW-8", "p DM NW-5 (справ.)", "Холм m = 8", "Холм N = 836", "Холм N_wide = 3377", "p SPA (c)", "p StepM", "вывод"]
    a = md(hdr, [_hrow(T, "%s|%s" % (c, b), "main", ("X1_C(%s)" % c if b == "X1" else c) + " к " + b) for c in CANDS for b in BASES])
    sp = T["spa"]
    fam = sp[sp["kind"] == "family"]
    f = md(["База (бенчмарк)", "кандидаты", "T", "p_l", "p_c (основной)", "p_u"], [[r.base, r.candidates.replace("|", ", "), c_(r.T, 4), c_(r.p_l, 4), c_(r.p_c, 4), c_(r.p_u, 4)] for r in fam.itertuples()])
    rej = [r.candidates for r in sp[sp["kind"] == "candidate"].itertuples() if str(r.stepm_reject) == "True"]
    m = T["mcs"]
    inn = [r.series for r in m.itertuples() if str(r.in_mcs) == "True"]
    cd = md([x.replace("m = 8", "m = 4") for x in hdr[:-1]] + ["справочно"], [_hrow(T, "S-CD|%s" % b, "cd_base", ("X1_C(S-CD)" if b == "X1" else "S-CD") + " к " + b)[:-1] + ["сырые p; Холм N по 4 p; кандидатом не является"] for b in BASES])
    return ("## 2. Статистика\n\n" + dm_line(T) + " DM — Ньюи–Уэст, лаг 8 (правило NW-94, основной; NW-5 — справочно, как W1–W8), односторонний «кандидат лучше»; интервал и p_boot — стационарный бутстрап (средний блок 20, B = 10 000, seed 0); "
            "Холм — по p DM NW-8: m = 8 (кандидаты × базы), N = 836 и N_wide = 3377 (счётчик программы; недостающие до N p = 1); SPA — студентизированный Hansen (2005), основной p_c; StepM — Romano–Wolf. "
            "Против X1 сравнивается слой X1_C (М1), против остальных — F.\n\n" + a + "\n\nSPA по базам (кандидаты {S-CK, S-CKA, S-CD}; бенчмарк согласован со слоем кандидатов).\n\n" + dm_line(T) + "\n\n" + f +
            "\n\nStepM (α = 0,05): " + ("отвергнуты — " + ", ".join(rej) if rej else "превосходящих кандидатов нет (ни одна гипотеза не отвергнута)") + ".\n\n"
            "S-CD против баз — справочно (сырые p и Холм N программы по 4 p):\n\n" + dm_line(T) + "\n\n" + cd + "\n\n"
            "MCS 90 %% (T_max, B = 10 000, только конечные положительные прогнозы; n = %d): в множество входят %d из %d рядов (%s); наименьший p = %s.\n\n" % (int(m["n_used"].iloc[0]), len(inn), len(m), ", ".join(inn), c_(m["p"].min(), 4)) +
            "Мощность: σ_LR разности потерь (NW-94), наблюдаемый эффект, n для мощности 0,5 и 0,8 при α = 0,05 односторонний; «—» — эффект не в пользу кандидата или нулевой (n не определено); справа — гипотетический эффект −1 % QLIKE базы. "
            "Недостаточная мощность не доказывает, что улучшение невозможно.\n\n" + dm_line(T) + " Окно «без 2022» — n = %d.\n\n" % int(T["power"][T["power"]["window"] == "без 2022"]["n"].iloc[0]) + _power(T))


def _ytab(T, cand) -> str:
    rows = []
    for p in T["years"]["period"].unique():
        r1, r2 = yrs(T, "X1_C(%s)" % cand, "X1", p), yrs(T, cand, "W5-G", p)
        rows.append([p, int(r1["n"]), c_(r1["delta_pct"], 4, True), c_(r1["contrib_pct"], 4, True), c_(r2["delta_pct"], 4, True), c_(r2["contrib_pct"], 4, True)])
    y = T["years"]
    for k, (h, b) in (("М1", ("X1_C(%s)" % cand, "X1")), ("М2", (cand, "W5-G"))):
        s = y[(y["series"] == h) & (y["base"] == b) & (y["kind"] == "period")]["contrib_pct"].sum()
        rows.append(["Σ вкладов периодов (%s)" % k, "", "", c_(s, 4, True) if k == "М1" else "", "", c_(s, 4, True) if k == "М2" else ""])
    return md(["Период (t⁺)", "n", "Δ М1, %", "вклад М1, п. п.", "Δ М2, %", "вклад М2, п. п."], rows)


def _ctab(T) -> str:
    rows = []
    for c, pid, h, b in [(c, p, ("X1_C(%s)" % c if bb == "X1" else c), bb) for c in CANDS for p, bb in (("М1", "X1"), ("М2", "W5-G"))] + [("S-CK", "К1", "S-CK", "S-CD")]:
        y1, y2, y3 = skey(T, h, b, "no_top_day"), skey(T, h, b, "var_share_top_day"), skey(T, h, b, "no_top_year")
        rows.append([c, pid, c_(cnc(T, h, b, "summary", "delta_total_pct"), 4, True), c_(cnc(T, h, b, "no_day", "2022-02-24"), 4, True), c_(cnc(T, h, b, "no_year", "2022"), 4, True),
                     c_(cnc(T, h, b, "no_top_day", y1), 4, True) + " (" + y1 + ")", c_(cnc(T, h, b, "no_top_year", y3), 4, True) + " (" + y3 + ")", c_(cnc(T, h, b, "var_share_top_day", y2), 3) + " (" + y2 + ")",
                     c_(cnc(T, h, b, "summary", "share_best_year"), 3), c_(cnc(T, h, b, "summary", "share_two_best_years"), 3)])
    return md(["Кандидат", "пара", "Δ, %", "без 24.02.2022", "без 2022", "без самого влиятельного дня", "без самого влиятельного года", "доля дисперсии крупнейшего дня", "доля выигрыша: лучший период", "два лучших"], rows)


def sec3(T: dict) -> str:
    h = {k: one(T["years"], series=s, base=b, period="2018")["delta_total_pct"] for k, (s, b) in (("1", ("X1_C(S-CK)", "X1")), ("2", ("S-CK", "W5-G")))}
    return ("## 3. Годы, половины и концентрация\n\n" + dm_line(T) + " Вклад периода c_p = 100·Σ(l_кандидат − l_база)/(n·Q_база), п. п. QLIKE базы; Σ вкладов периодов = Δ на D_M (тождество проверяется кодом, допуск 1e-12); "
            "половины h1 = 2018–2021 (n = 1010), h2 = 2022–2026 (n = 1188) — строки периода «h1…», «h2…». М1 — X1_C против X1, М2 — F против W5-G. Победа в каждом году не требуется; худшие годы видны в таблицах.\n\n"
            "Δ полного периода S-CK: М1 = %s %%, М2 = %s %%.\n\n" % (c_(h["1"], 4, True), c_(h["2"], 4, True)) + "### S-CK\n\n" + _ytab(T, "S-CK") + "\n\n### S-CKA\n\n" + _ytab(T, "S-CKA") +
            "\n\nКонцентрация и чувствительность — только диагностика, не новая выборка; основная оценка — все 2198 дат. Срезы: без t⁺ = 2022-02-24; без всего 2022; без самого влиятельного дня (max |l_кандидат − l_база|); без самого влиятельного календарного года; "
            "доля дисперсии разности от крупнейшего дня (d_(1) − d̄)²/Σ(d_t − d̄)²; доля выигрыша от лучшего периода и от двух лучших (определена при суммарном выигрыше периодов > 0).\n\n" + dm_line(T) + "\n\n" + _ctab(T) +
            "\n\nВклады каждого периода для К1–К3, S-CD и абляции — в `S_C_YEARS.csv` и `S_C_CONC.csv`.")
