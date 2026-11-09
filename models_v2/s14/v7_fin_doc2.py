"""V7 лист FIN: разделы 8–15 итогового документа (абляции, веса, устойчивость, после 2021, причины, отступления, перспективы, реестр и хэши). Числа — из таблиц `t` и CHECKPOINT; словесные выводы — по знаку и по условиям (`word`, `claims_fail`)."""
from __future__ import annotations

from models_v2.s14 import v7_fin_data as D, v7_fin_text as TX
from models_v2.s14.v7_fin_doc import f, go_count, nm, s, tbl, word

HYP = ["H1", "H2", "H3", "4–7 против баз", "H8 якорь", "8 против 4 (справочно)", "H10 совместное", "10 против 8 (справочно)"]
PERIODS = [str(y) for y in range(2018, 2027)] + ["до 2021 (≤ 2021)", "после 2021 (≥ 2022)"]


def dx(t: dict, key: str) -> float:
    x = t["FIN_DX"]
    return float(x[x["key"] == key]["value"].iloc[0])


def pair_rows(t: dict, win: str) -> list:
    p, rows = t["FIN_PAIRS"], []
    for h in HYP:
        for r in p[(p["hyp"] == h) & (p["window"] == win)].itertuples():
            rows.append([h, nm(r.x), nm(r.y), "%s [%s; %s]" % (s(r.delta_pct), s(r.lo90), s(r.hi90)), word(r.delta_pct), f(r.p_one, 3), "%d из %d" % (r.years_neg, r.n_years), r.rule62, f(r.p_holm_N, 3)])
    return rows


def sec8(t: dict) -> str:
    head = ["гипотеза", "X", "Y", "Δ, % K1 [90 %]", "X", "DM p", "лет Δ < 0", "правило 6.2", "Холм N"]
    p = t["FIN_PAIRS"]
    yes = p[(p["window"] == "2018_2021") & (p["rule62"] == "да")]
    lst = ", ".join("%s: %s против %s" % (r.hyp, nm(r.x), nm(r.y)) for r in yes.itertuples()) or "ни одной"
    return ("H1 — энкодер против A0, A0g (конфигурации 1–3); H2 — маршрутизатор против плоского корректора (6 против 4); H3 — вход маршрутизатора A, B, C (5, 6, 7); якорь — 8 против 6; совместное обучение — 10 против 6; остальное — справочно. "
            "Правило 6.2 по всем парам (окно 2018–2021) выполнено в %d из %d: %s.\n\n**2018–2021** (`FIN_PAIRS.csv`; «X» — конфигурация X лучше или хуже Y по знаку Δ):\n\n%s\n**2022–2025** (правило 6.2 не применяется, «—»):\n\n%s"
            % (len(yes), int((p["window"] == "2018_2021").sum()), lst, tbl(head, pair_rows(t, "2018_2021")), tbl(head, pair_rows(t, "2022_2025"))))


def sec9(t: dict) -> str:
    w = t["FIN_WEIGHTS"].drop_duplicates(["config", "quantity", "period"])
    pv = w.pivot(index=["config", "quantity"], columns="period", values="mean").reindex(columns=PERIODS)
    rows = [[c, k] + [f(v, 3) for v in r] for (c, k), r in pv.iterrows()]
    sp = t["FIN_SPEC"]
    sp = sp[sp["window"] == "2018_2021"]
    srows = [[r.config, int(r.k), f(r.share_pi_gt03, 3), f(r.switch_freq, 3), "%s [%s; %s]" % (s(r.gain_on, 4), s(r.gain_on_lo90, 4), s(r.gain_on_hi90, 4)), r.specialized] for r in sp.itertuples()]
    return ("Средние веса по годам цели, до и после 2021: π_k — эксперт k (π_0 — нулевой), w — вес якоря (конфигурация 8); 2018 у конфигурации 10 — прогноз Stage A без весов Stage B (`FIN_WEIGHTS.csv`):\n\n%s\n"
            "![динамика весов](harj_hybrid_v7/FIN_WEIGHTS.png)\n\n**Специализация экспертов 6.3** (задним числом, описательно; окно 2018–2021, `FIN_SPEC.csv`; «выигрыш» — разность QLIKE с экспертом и без на днях π_k > 0,3):\n\n%s"
            % (tbl(["конфигурация", "величина"] + PERIODS, rows), tbl(["конфигурация", "эксперт", "доля дней π > 0,3", "частота переключений", "выигрыш [90 %]", "специализирован"], srows)))


def sec10(t: dict) -> str:
    st = t["FIN_STRESS"]
    lab = ["2018_2021", "без 1 % крайних", "2019_2021"]
    rows = []
    for k in D.SER:
        g = st[st["series"] == k].set_index("stress")
        rows.append([nm(k)] + ["%s / %s" % (f(g.loc[x, "qlike"]), s(g.loc[x, "gap_best_k2k3"], 4)) for x in lab])
    ls = t["FIN_LAMSWEEP"]
    lrows = [["cfg%d" % r.cfg, f(r.lam_g, 2), f(r.red_in_pct, 2), f(r.val_q_mean)] for r in ls.itertuples()]
    return ("QLIKE / разность к лучшему из K2 и K3 (минус — лучше; `FIN_STRESS.csv`); «2019–2021» — без 2018:\n\n%s\n**Соседние λ_g** (внутренняя решётка последних переоценок января 2019–2021, конфигурации 1 и 4; среднее по трём годам; снижение QLIKE H на обучающих парах, %% и средняя QLIKE на валидации; `FIN_LAMSWEEP.csv`):\n\n%s"
            % (tbl(["ряд", "2018–2021", "без 1 % крайних дат", "без 2018"], rows), tbl(["конфигурация", "λ_g", "снижение, %", "QLIKE валидации"], lrows)))


def sec11(t: dict) -> str:
    a = t["FIN_AFTER"].set_index("series")
    rows = [[nm(k), f(r.qlike_2018_2021), f(r.qlike_2022_2025), f(r.ratio_2022_2025_to_2018_2021, 3), f(r.qlike_2022), f(r.qlike_2023_2025), f(r.qlike_crisis)] + [f(r["share_better_%s_2022_2025" % b], 3) for b in ("K2", "K3", "NTP2", "A2")] for k, r in a.iterrows()]
    k1 = a.loc["K1"]
    return ("Ошибка HAR-J (K1): QLIKE %s на 2018–2021 и %s на 2022–2025 (отношение %s); у K2 — %s и %s. Кризисные дни — 5 %% дат окна 2022–2025 с наибольшим прогнозом K1 (%d дат; правило объявлено в листе FIN до просмотра, не подбиралось). "
            "Вклад 2022 года: разрыв конфигурации 6 к K2 на 2022–2025 — %s, без 2022 — %s (доля 2022 в разрыве %s; конфигурация 4: %s, без 2022 — %s, доля %s). Доли дней 2022–2025, когда ряд лучше XGBoost (K2), LightGBM (K3), NTP2, A2 — последние четыре столбца (`FIN_AFTER.csv`):\n\n%s"
            % (f(k1.qlike_2018_2021), f(k1.qlike_2022_2025), f(k1.ratio_2022_2025_to_2018_2021, 3), f(a.loc["K2", "qlike_2018_2021"]), f(a.loc["K2", "qlike_2022_2025"]), int(k1.n_crisis), s(dx(t, "gap_total.6"), 4), s(dx(t, "gap_without_2022.6"), 4),
               f(dx(t, "gap_share_2022.6"), 3), s(dx(t, "gap_total.4"), 4), s(dx(t, "gap_without_2022.4"), 4), f(dx(t, "gap_share_2022.4"), 3),
               tbl(["ряд", "QLIKE 2018–2021", "QLIKE 2022–2025", "отношение", "QLIKE 2022", "QLIKE 2023–2025", "кризис", "лучше K2", "лучше K3", "лучше NTP2", "лучше A2"], rows)))


def sec12(t: dict) -> str:
    v0 = lambda k: t["FIN_DX"].set_index("key").loc["verdict." + k, "text"]
    v = lambda k: v0(k) + (" [%s]" % TX.DX_NOTE if k in ("b", "d") and v0(k).startswith("подтверждена") else "")   # замечание З-2
    return ("Только измеренное (лист DX, задним числом, не отбор; правила объявлены до счёта; `FIN_DX.csv`):\n\n"
            "- **(а) причина «энкодеру не хватает выразительности»** — %s: энкодер лучше наивных предсказателей (среднее, EWMA, профиль, константа) в %s %% ячеек год × seed; отношение MSE к лучшему наивному %s.\n"
            "- **(б) причина «у z нет переносимого сигнала сверх Z»** — %s: частный R² z сверх Z и контролей после выбора α — %s; доля дисперсии z, объясняемая контролями, — %s.\n"
            "- **(в) причина «недостаточная регуляризация»** — %s: самый малый λ_g выбирается в %s случаев; средний |G| = %s; снижение QLIKE H на обучающих парах, %%: %s (λ_g = 0) и %s (λ_g = 1).\n"
            "- **(г) причина «z дублирует Z и H»; вклад 2022 года** — %s: PIT R² z → ln(RV/K2) %s (на разработке %s); разрыв конфигурации 4 к K2 — %s (доля 2022 %s), конфигурации 6 — %s (доля %s).\n\n"
            "**Оговорка о z (замечание З-5).** %s\n"
            % (v("a"), f(100 * dx(t, "enc_share_beats"), 1), f(dx(t, "enc_mse_ratio_best_naive"), 3), v("b"), f(dx(t, "expost_partial_r2_best"), 4), f(dx(t, "z_r2_on_controls"), 3),
               v("c"), f(dx(t, "lam_share_lowest"), 3), f(dx(t, "g_mean_abs_median"), 3), f(dx(t, "sweep_red.0"), 2), f(dx(t, "sweep_red.1"), 2), v("d"), f(dx(t, "pit_r2_all"), 4), f(dx(t, "pit_r2_dev"), 4), s(dx(t, "gap_total.4"), 4), f(dx(t, "gap_share_2022.4"), 3), s(dx(t, "gap_total.6"), 4), f(dx(t, "gap_share_2022.6"), 3), TX.Z_IN_SAMPLE))


def sec13(t: dict) -> str:
    v1, rows = t["FIN_V1"], []
    for r in v1[v1["window"].isin(["2018_2021", "2022_2025"])].itertuples():
        rows.append([r.config, r.window, f(r.qlike_v1), s(r.d_k3_v1) + " [%s; %s]" % (s(r.lo90_v1), s(r.hi90_v1)), f(r.qlike_v2), s(r.d_k3_v2) + " [%s; %s]" % (s(r.lo90_v2), s(r.hi90_v2))])
    dep = t["FIN_DEPARTURES"]
    j = "".join("- журнал, строка %d (%s): %s\n" % (r.line, r.kind, r.text) for r in dep.itertuples())
    return ("%s\n\n**Версия 1 и версия 2 рядом** (QLIKE и Δ к K3 в %% уровня K1 [90 %%]; основная — версия 2, `FIN_V1.csv`):\n\n%s\n**Записи журнала [V7] дословно (все отступления и ошибки дирижёра):**\n\n%s\n**Оговорки:**\n\n%s"
            % (TX.DEPARTURES, tbl(["конфигурация", "окно", "QLIKE v1", "Δ к K3, v1", "QLIKE v2", "Δ к K3, v2"], rows), j, TX.CAVEATS))


def head14(t: dict, yes) -> str:
    """Заголовок «Подтверждено» только при p Холма (N = 733) < 0,05 у пар с правилом 6.2; иначе — формулировка замечания З-1 (p из таблицы)."""
    pm = float(yes["p_holm_N"].min()) if len(yes) else 1.0
    if pm < 0.05:
        return "**Подтверждено**"
    return "**Правило 6.2 выполнено (статистически не подтверждено: p Холма N = 733 — %s, порог 0,05)**" % f(pm, 3)


def sec14(t: dict) -> str:
    p, x = t["FIN_PAIRS"], t["FIN_TESTS"]
    yes = p[(p["window"] == "2018_2021") & (p["rule62"] == "да")]
    cnt = lambda w: int(((x["window"] == w) & x["base"].isin(D.BASES) & (x["hi90"] < 0)).sum())
    w8 = t["FIN_WEIGHTS"]
    w8 = w8[(w8["config"] == "cfg8") & (w8["quantity"] == "w")].set_index("period")["mean"]
    by = "; ".join("%s — %d из %d" % (h, int(((p["hyp"] == h) & (p["window"] == "2018_2021") & (p["rule62"] == "да")).sum()), int(((p["hyp"] == h) & (p["window"] == "2018_2021")).sum())) for h in HYP[:3] + HYP[4:5] + HYP[6:7])
    return ("%s (правилами, объявленными до счёта): правило 6.2 выполнено в %d парах: %s; по гипотезам (пар с «да» из всех пар): %s; «go» 6.1 — у %d из %d конфигураций.\n\n"
            "**Не подтверждено:** преимущество над K2, K3, NTP2, A2 — пар (конфигурация, база) с интервалом Δ целиком ниже нуля: %d из %d на 2018–2021 и %d из %d на 2022–2025; p Холма (N = 733) — наименьший по всем этим сравнениям %s на 2018–2021.\n\n"
            "**Где источник информации:** вес якоря на бустинг M у конфигурации 8 растёт от %s (2018) до %s (2026); частный R² z сверх Z — %s (раздел 12); то есть измеренный выигрыш 8 против 6 согласуется с долей бустингов в базе, а не с новым сигналом энкодера.\n\n"
            "**Оговорка о z (замечание З-5).** %s\n\n"
            "**Какая проверка отделила бы от шума:** фиксированная конфигурация 8 и её контроли (якорь с постоянным w, перестановка z во времени, то же без z) на других бумагах панели по правилам V7 без подбора, с числом дат, достаточным для различимого эффекта; без обращения к окну 2022+ для выбора. Это проверка, а не обещание результата."
            % (head14(t, yes), len(yes), "; ".join("%s: %s против %s" % (r.hyp, nm(r.x), nm(r.y)) for r in yes.itertuples()) or "нет", by, go_count(t), len(D.CFG), cnt("2018_2021"), len(D.CFG) * len(D.BASES), cnt("2022_2025"), len(D.CFG) * len(D.BASES),
               f(float(x[(x["window"] == "2018_2021") & x["base"].isin(D.BASES)]["p_holm_N"].min()), 3), f(w8["2018"], 2), f(w8["2026"], 2), f(dx(t, "expost_partial_r2_best"), 4), TX.Z_IN_SAMPLE))


def sec15(t: dict, cp: dict, shas: dict) -> str:
    keys = [[k, v.get("code", "—") if isinstance(v, dict) else "—", v.get("inputs", "—") if isinstance(v, dict) else "—", ", ".join(sorted(v)) if isinstance(v, dict) else "—"] for k, v in sorted(cp.items()) if k != "FIN"]
    fc = [[k, cp[st][k]["sha256_file"][:16]] for st, k in (("1b", "cfg1"), ("1b", "cfg2"), ("1b", "cfg3"), ("2", "cfg4"), ("2", "cfg5"), ("2", "cfg6"), ("2", "cfg7"), ("3", "cfg8"), ("4", "cfg10"))]
    return ("Реестр — `ALL_TRIALS.csv` (раздел 4; только дополняется, версия 1 — строки «результат v1 архивирован»); конфигураций израсходовано листом FIN: 0. Ключи CHECKPOINT (`CHECKPOINT.json`; ключ `FIN` добавляет эта стадия; остальные не менялись):\n\n%s\n"
            "Хэши parquet прогнозов (sha256, 16 знаков):\n\n%s\nХэши входов: %s; реестр ALL_TRIALS sha256 %s; прогнозы K2/K3/NTP2/A2 — `CONTROLS_V7.csv`."
            % (tbl(["ключ", "код", "входы", "содержимое"], keys), tbl(["файл", "sha256"], fc), shas["inputs"], shas["trials"]))
