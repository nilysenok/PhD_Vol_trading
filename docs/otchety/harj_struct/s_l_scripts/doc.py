"""S-L: сборка `docs/otchety/harj_struct/S_L_DIAG.csv` и `S_L_REPORT.md` из `<sp>/s_l/{l12,l3,l3_months}.csv` и `S_D_{ELEMENTS,PRED}.csv`; решение по таблице 7.2 `S_D_SPEC.md` выводится кодом.
Запуск из корня: PYTHONPATH=. python3 -W ignore <sp>/s_l/doc.py [--check] (--check — только чтение: пересборка в памяти и сравнение с файлами)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = Path.cwd()
HS = ROOT / "docs/otchety/harj_struct"
OUT_CSV, OUT_MD = HS / "S_L_DIAG.csv", HS / "S_L_REPORT.md"
INPUTS = ["data/regen/harj_struct/R/rck.parquet", "data/regen/harj_struct/R/ra2.parquet", "data/regen/harj_struct/A/a2.parquet", "data/regen/harj_x1/W5/w5g.parquet",
          "docs/otchety/harj_struct/S_D_ELEMENTS.csv", "docs/otchety/harj_struct/S_D_PRED.csv"]
L3_MAIN, L3_THR, R_THR = ("S-A2 (старый)", "full", "C"), 0.80, 0.5
SER = ("W5-G", "X1", "cfg8", "B_mix")


def sha(p) -> str:
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()[:16]


def load():
    a = pd.read_csv(HERE / "l12.csv", dtype={"group": str})
    b = pd.read_csv(HERE / "l3.csv")
    m = pd.read_csv(HERE / "l3_months.csv")
    b["var"], b["group"] = b["part"] + "|" + b["level"], "-"
    m2 = m.melt(id_vars=["series", "part", "level", "refit", "n_pool"], value_vars=["gain_total", "gain_rest99", "share_top1"], var_name="item")
    m2 = m2.assign(section="L3_month", var=m2["part"] + "|" + m2["level"], group=m2["refit"].astype(str), n=m2["n_pool"], note="")
    cols = ["section", "series", "var", "group", "item", "n", "value", "note"]
    el = pd.read_csv(HS / "S_D_ELEMENTS.csv", comment="#")
    pr = pd.read_csv(HS / "S_D_PRED.csv", comment="#")
    return pd.concat([a[cols], b[cols], m2[cols]], ignore_index=True).fillna({"note": ""}), el, pr


def g(df, **kw) -> float:
    s = df
    for k, v in kw.items():
        s = s[s[k].astype(str) == str(v)]
    if len(s) != 1:
        raise SystemExit("ожидалась одна строка: %s, найдено %d" % (kw, len(s)))
    return float(s["value"].iloc[0])


def decide(df, el, pr) -> list:
    """Строки таблицы 7.2 по объявленным условиям. (i) и (iii) — коды разд. 5 из S_D_ELEMENTS; L2 — чтение О-5 по W5-G; L3 — О-8."""
    ci = el.loc[el["series"] == "S-RCKf", "code"].iloc[0]
    ciii = el.loc[el["series"] == "S-RCKm", "code"].iloc[0]
    rd = {a: int(g(df, section="L2", series="W5-G", var="pi×" + a, item="reading_O5")) for a in ("v1", "v2")}
    l3 = g(df, section="L3", series=L3_MAIN[0], var=L3_MAIN[1] + "|" + L3_MAIN[2], item="frac_rest99_not_better")
    l3m = g(df, section="L3", series=L3_MAIN[0], var="memory_only|C", item="frac_rest99_not_better")
    pf2 = pr.loc[pr["id"] == "P-f2", "fulfilled"].iloc[0]
    rows = [
        ("1", "(i) необходим или «хуже без него» и L2: π не объясняется v₁, v₂", "(i) = %s; L2 чтение О-5: v₁ → %d, v₂ → %d (1 = не объясняется)" % (ci, rd["v1"], rd["v2"]),
         ci in ("needed", "worse_noise") and rd["v1"] == 1 and rd["v2"] == 1, "S-RL1 — сдвиг уровня C одним параметром, процедура R на уровне F"),
        ("2", "(i) необходим и L2: π ≈ режим v₁ (или v₂)", "(i) = %s; v₁ → %d, v₂ → %d (2 = π ≈ режим)" % (ci, rd["v1"], rd["v2"]),
         ci == "needed" and (rd["v1"] == 2 or rd["v2"] == 2), "S-RLv"),
        ("3", "(i) не обнаружен", "(i) = %s" % ci, ci == "not_found", "S-RL1 с потерей уровня C"),
        ("4", "(iii) вредит", "(iii) = %s" % ciii, ciii == "harms", "m := 1 в преемнике"),
        ("5", "L3: у памяти A2 нет выигрыша на 99 %% пар в ≥ %.0f %% месяцев" % (100 * L3_THR), "старый S-A2, полное θ, уровень C (О-8): %.3f" % l3,
         l3 >= L3_THR, "линия обучаемой памяти закрывается"),
    ]
    return rows, dict(ci=ci, ciii=ciii, l3=l3, l3m=l3m, pf2=pf2)


def n_base() -> int:
    """Текущий N программы — максимум `n` по ключам S.*.registered в CHECKPOINT (только чтение)."""
    import json
    cp = json.loads((ROOT / "docs/otchety/harj_x1/CHECKPOINT.json").read_text(encoding="utf-8"))
    return max(v["n"] for k, v in cp.items() if k.startswith("S.") and k.endswith(".registered") and isinstance(v, dict) and "n" in v)


def f(v, k=4) -> str:
    return "—" if not np.isfinite(v) else ("%." + str(k) + "f") % v


def render(df, el, pr) -> tuple:
    rows, info = decide(df, el, pr)
    dec = pd.DataFrame([dict(section="DECISION_7.2", series="-", var="row " + r[0], group="-", item="holds", n=0, value=float(r[3]), note=r[1] + " | " + r[2] + " | → " + r[4]) for r in rows])
    full = pd.concat([df, dec], ignore_index=True)
    head = ["# S-L (Ж-256 п. 5; S_D_SPEC.md разд. 7.2–7.3): диагностика без обучения по сохранённым рядам и θ; DEVELOPMENT / HISTORICAL WALK-FORWARD; исследовательский, не OOS",
            "# L1/L2: окно M после разогрева 252 строк, PIT-квинтили по датам < d, обрезка c_d = Q₀,₉₉ RV/F по парам t⁺ < d; строки L2_ceiling — потолок (k_g по исходу), не прогноз; L3: выигрыш в выборке по пулу месяца, θ сохранённые",
            "# входы sha256[:16]: " + "; ".join("%s %s" % (Path(p).name, sha(p)) for p in INPUTS) + "; генераторы <sp>/s_l/{l12,l3,doc}.py"]
    csv = "\n".join(head) + "\n" + full.to_csv(index=False, float_format="%.10g")
    return csv, md(df, el, pr, rows, info)


def md(df, el, pr, rows, info) -> str:
    L = []
    w = L.append
    nL1 = int(g(df, section="L1", item="days_in_L1"))
    win = df[(df["section"] == "L1") & (df["item"] == "days_in_L1")]["note"].iloc[0]
    w("диагностика без обучения (S-L, Ж-256 п. 5), исследовательский результат на раскрытой истории, не OOS; S-RCK — кандидат, X1 и W5-G — контрольные версии; строки «потолок» используют исход и прогнозом не являются")
    w("")
    w("# S-L — почему помогает сдвиг уровня и не помогает память: L1, L2, L3 и выбор по таблице 7.2 `S_D_SPEC.md`")
    w("")
    w("Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Ничего не обучалось: только сохранённые ряды (`x1`, `W5/w5g`, `A/a2`, `R/rck`, `R/ra2`) и сохранённые θ месяцев, прямой проход `a_model.forward` / `r_loss.lnc_of`. Объявления О-1…О-8 записаны до счёта (`<sp>/s_l/GATES.md`). Числа — `S_L_DIAG.csv`.")
    w("")
    w("## 1. L1 — PIT-таблица `S_D_EXPLAIN.md` разд. 2.3")
    w("")
    w("%s, n = %d (разогрев 252 строк). Ячейка — среднее min(RV/F, c_d), c_d = Q₀,₉₉ RV/F по парам t⁺ < d. Квинтили π, v₁ = ln(RV_d/RV_m,d), v₂ = ln(B_mix/cfg8) — пороги по датам < d (порча значений после d группу дня d не меняет: %s)." % (
        win.replace("окно L1: ", "Окно d: "), nL1, ", ".join("%s %s" % (k, df[(df["section"] == "check") & (df["var"] == k)]["note"].iloc[0]) for k in ("pi", "v1", "v2"))))
    w("")
    w("| переменная | группа | n | W5-G | X1 | cfg8 | B_mix |")
    w("|---|---|---|---|---|---|---|")
    w("| все | — | %d | %s |" % (nL1, " | ".join(f(g(df, section="L1", series=s, var="all", group="all", item="mean_clipped_ratio"), 3) for s in SER)))
    for var, gl in (("pi", "π"), ("v1", "v₁"), ("v2", "v₂"), ("s_d", "s_d"), ("gap", "gap^PIT"), ("year", "год")):
        grp = sorted(df[(df["section"] == "L1") & (df["series"] == "W5-G") & (df["var"] == var) & (df["item"] == "mean_clipped_ratio")]["group"].unique(), key=lambda t: int(t))
        for gv in grp:
            n = int(df[(df["section"] == "L1") & (df["series"] == "W5-G") & (df["var"] == var) & (df["group"] == gv) & (df["item"] == "mean_clipped_ratio")]["n"].iloc[0])
            lab = ("Q%s" % gv) if var in ("pi", "v1", "v2") else gv
            w("| %s | %s | %d | %s |" % (gl, lab, n, " | ".join(f(g(df, section="L1", series=s, var=var, group=gv, item="mean_clipped_ratio"), 3) for s in SER)))
    w("")
    p1, p5 = (g(df, section="L1", series="W5-G", var="pi", group=q, item="mean_clipped_ratio") for q in ("1", "5"))
    p4 = g(df, section="L1", series="W5-G", var="pi", group="4", item="mean_clipped_ratio")
    qmin = {s: 1 + int(np.argmin([g(df, section="L1", series=s, var="pi", group=str(q), item="mean_clipped_ratio") for q in range(1, 6)])) for s in SER}
    hi_lo = all(g(df, section="L1", series=s, var="pi", group="5", item="mean_clipped_ratio") < g(df, section="L1", series=s, var="pi", group="1", item="mean_clipped_ratio") for s in SER)
    w("- Направление таблицы не-PIT %s: у %s четырёх рядов Q5 по π ниже Q1 (W5-G: Q1 %s → Q4 %s, Q5 %s). Квинтиль минимума: %s." % (
        "подтверждается" if hi_lo else "не подтверждается", "всех" if hi_lo else "не всех", f(p1, 3), f(p4, 3), f(p5, 3), ", ".join("%s Q%d" % (s, q) for s, q in qmin.items())))
    w("- Корреляция π с v₁: Пирсон %s, Спирмен %s; с v₂: %s и %s. π почти не связан с переменными режима." % tuple(f(g(df, section="L1", var=v, item=i), 3) for v in ("pi~v1", "pi~v2") for i in ("corr_pearson", "corr_spearman")))
    w("- 2018 — один день (29.12.2018), в выводах не используется.")
    w("")
    w("## 2. L2 — заменяет ли режим π")
    w("")
    w("Разброс S = r̄(Q5) − r̄(Q1) среднего обрезанного отношения; «внутри» — среднее той же разности по 5 квинтилям другой переменной; R = внутри / без условия. Чтение О-5 (моё, порог спецификацией не задан): R_π|v ≥ %.1f — «π не объясняется v»." % R_THR)
    w("")
    w("| ряд | пара | S_π | S_π внутри v | R_π\\|v | S_v | S_v внутри π | R_v\\|π | чтение О-5 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for s in SER:
        for a in ("v1", "v2"):
            k = lambda it: g(df, section="L2", series=s, var="pi×" + a, item=it)
            note = df[(df["section"] == "L2") & (df["series"] == s) & (df["var"] == "pi×" + a) & (df["item"] == "reading_O5")]["note"].iloc[0]
            w("| %s | π × %s | %s | %s | %s | %s | %s | %s | %s |" % (s, a.replace("v1", "v₁").replace("v2", "v₂"), f(k("S_pi_marginal")), f(k("S_pi_within_v")), f(k("R_pi_given_v"), 2), f(k("S_v_marginal")), f(k("S_v_within_pi")), f(k("R_v_given_pi"), 2), note.replace("v1", "v₁").replace("v2", "v₂")))
    w("")
    w("- S_v считается как Q5 − Q1. У v₁ зависимость U-образная (W5-G: Q1 %s, Q3 %s, Q5 %s), поэтому R_v|π мало говорит о силе v₁. На строку таблицы это не влияет: решает R_π|v." % tuple(f(g(df, section="L1", series="W5-G", var="v1", group=q, item="mean_clipped_ratio"), 3) for q in ("1", "3", "5")))
    w("")
    w("5 × 5, W5-G, π × v₁ (n в скобках):")
    w("")
    w("| π \\ v₁ | Q1 | Q2 | Q3 | Q4 | Q5 |")
    w("|---|---|---|---|---|---|")
    for i in range(1, 6):
        cells = []
        for j in range(1, 6):
            r = df[(df["section"] == "L2") & (df["series"] == "W5-G") & (df["var"] == "pi×v1") & (df["group"] == "pi%d|v1%d" % (i, j))].iloc[0]
            cells.append("%s (%d)" % (f(r["value"], 3), r["n"]))
        w("| Q%d | %s |" % (i, " | ".join(cells)))
    w("")
    w("**Потолок** — оптимальный по QLIKE постоянный множитель k_g = mean(RV/F) в каждой группе. Он использует исход, это не прогноз. Выигрыш — %% QLIKE ряда на тех же %d днях, минус — лучше:" % nL1)
    w("")
    w("| ряд | одна группа | π (5) | v₁ (5) | v₂ (5) | π × v₁ (25) | π × v₂ (25) |")
    w("|---|---|---|---|---|---|---|")
    for s in SER:
        w("| %s | %s |" % (s, " | ".join(f(g(df, section="L2_ceiling", series=s, var=v, item="gain_pct_of_qlike"), 3) for v in ("one", "pi", "v1", "v2", "pi×v1", "pi×v2"))))
    w("")
    w("- Для W5-G потолок по π (%s %%) больше потолков по v₁ (%s %%) и v₂ (%s %%). При этом 25 групп — это 25 констант, подобранных по исходу на тяжёлом хвосте, поэтому потолки 5 × 5 сильно завышены." % tuple(f(g(df, section="L2_ceiling", series="W5-G", var=v, item="gain_pct_of_qlike"), 3) for v in ("pi", "v1", "v2")))
    w("- Дополнительно, вне разд. 7.2 и без влияния на строку таблицы: не прокси ли π времени. Разброс по π внутри лет 2019–2026 для W5-G: R_π|год = %s. Но знак S_π по годам меняется:" % f(g(df, section="L2_extra", series="W5-G", var="pi×year", item="R_pi_given_year"), 2))
    yrs = df[(df["section"] == "L2_extra") & (df["series"] == "W5-G") & (df["item"] == "S_pi_in_year")]
    w("  " + "; ".join("%s %s" % (r["group"], f(r["value"], 3)) for _, r in yrs.iterrows()) + ".")
    w("  Средний π по годам: " + "; ".join("%s %s" % (r["group"], f(r["value"], 3)) for _, r in df[(df["section"] == "L1") & (df["item"] == "mean_pi")].iterrows() if r["group"] != "2018") + ".")
    w("")
    w("## 3. L3 — почему память не добавляет")
    w("")
    w("Выигрыш в выборке месяца: gain = ℓ(θ = 0) − ℓ(θ̂) по закрытому пулу месяца. «Выигрыш на 99 %» — Σgain без ⌈1 %⌉ лучших пар > 0. Основной уровень — цель обучения ряда (О-7). У старого S-A2 θ — среднее seeds, а прогноз считался по среднему ln C seeds.")
    w("")
    w("| ряд | θ | уровень | месяцев | доля «есть выигрыш на 99 %» | доля «нет» | доля «есть выигрыш в целом» | доля 1 % лучших, медиана |")
    w("|---|---|---|---|---|---|---|---|")
    for (s, p, lev) in (("S-A2 (старый)", "full", "C"), ("S-A2 (старый)", "full", "F"), ("S-A2 (старый)", "memory_only", "C"), ("S-A2 (старый)", "memory_only", "F"), ("S-A2 (старый)", "no_memory", "C"), ("S-A2 (старый)", "no_memory", "F"),
                        ("S-RCK", "full", "F"), ("S-RCK", "full", "C"), ("S-RA2", "full", "F"), ("S-RA2", "full", "C")):
        k = lambda it: g(df, section="L3", series=s, var=p + "|" + lev, item=it)
        lab = {"full": "полное", "memory_only": "только память (ω, ψ, c)", "no_memory": "без памяти (δ, δw, δκ)"}[p]
        main = (s, p, lev) in (L3_MAIN, ("S-RCK", "full", "F"), ("S-RA2", "full", "F"))
        w("| %s | %s | %s%s | %d | %s | %s | %s | %s |" % (s, lab, lev, " (цель)" if main else "", int(k("months")), f(k("frac_rest99_better"), 3), f(k("frac_rest99_not_better"), 3), f(k("frac_total_better"), 3), f(k("share_top1_median"), 3)))
    w("")
    e = lambda it: g(df, section="L3", series="S-A2 (старый)", var="full|C", item=it)
    w("- Сверка: S-RCK (F) %s — как в `S_R_REPORT.md` разд. 4 (0,441); S-RA2 (F) %s — как там (0,000)." % (f(g(df, section="L3", series="S-RCK", var="full|F", item="frac_rest99_better"), 3), f(g(df, section="L3", series="S-RA2", var="full|F", item="frac_rest99_better"), 3)))
    w("- Приближение старого S-A2: прямой проход по среднему θ seeds против сохранённого C_A. Медиана отн. ошибки %s, максимум %s; в %d из 106 месяцев ошибка > 10⁻³. Без этих месяцев (n = 87) доля «нет» для полного θ: C %s, F %s. Вывод тот же." % (
        f(e("approx_median_rel_C_meantheta_vs_saved"), 8), f(e("approx_max_rel_C_meantheta_vs_saved"), 3), int(e("approx_months_rel_gt_1e-3")),
        f(g(df, section="L3", series="S-A2 (старый)", var="full|C", item="frac_rest99_not_better|approx_err<=1e-3"), 3), f(g(df, section="L3", series="S-A2 (старый)", var="full|F", item="frac_rest99_not_better|approx_err<=1e-3"), 3)))
    yr = df[(df["section"] == "L3") & (df["series"] == "S-A2 (старый)") & (df["var"] == "full|C") & df["item"].str.startswith("frac_rest99_better|")]
    mo = df[(df["section"] == "L3_month") & (df["series"] == "S-A2 (старый)") & (df["var"] == "full|C") & (df["item"] == "gain_rest99")]
    late = mo[mo["group"] >= "2022"]
    w("- Старый S-A2, полное θ, уровень C, доля «есть выигрыш на 99 %%» по годам переоценки: %s. С 2022 года выигрыш на 99 %% пар — в %d из %d месяцев." % (
        "; ".join("%s %s" % (r["item"].split("|")[1], f(r["value"], 2)) for _, r in yr.iterrows()), int((late["value"] > 0).sum()), len(late)))
    nm = min(g(df, section="L3", series="S-A2 (старый)", var=v, item="frac_rest99_not_better") for v in ("memory_only|C", "memory_only|F", "no_memory|C", "no_memory|F"))
    w("- Разложение при фиксированных θ — не абляция. «Только память» и «без памяти» по отдельности выигрыша на 99 %% почти не дают: доля месяцев «нет» ≥ %s. У полного θ доля месяцев с выигрышем — %s. Выигрыш на основной массе пар идёт от совместной настройки памяти с уровнем и корректором; доли не аддитивны." % (
        f(nm, 3), f(g(df, section="L3", series="S-A2 (старый)", var="full|C", item="frac_rest99_better"), 3)))
    yk = df[(df["section"] == "L3") & (df["series"] == "S-RCK") & (df["var"] == "full|F") & df["item"].str.startswith("frac_rest99_better|")]
    w("- Для сравнения — сдвиг уровня S-RCK (уровень F, месяцы θ ≠ 0), доля «есть выигрыш на 99 %%» по годам: %s (n месяцев: %s)." % (
        "; ".join("%s %s" % (r["item"].split("|")[1], f(r["value"], 2)) for _, r in yk.iterrows()), ", ".join(str(int(r["n"])) for _, r in yk.iterrows())))
    ra2 = df[(df["section"] == "L3_month") & (df["series"] == "S-RA2") & (df["var"] == "full|F") & (df["item"] == "gain_rest99")]
    w("- Ответ L3. У S-RA2 выигрыша на 99 %% пар нет в %d из %d месяцев. У старой процедуры он был в отдельных годах (выше) и с 2022 года почти исчез. Поэтому для 2022+ различие «сигнала нет» и «R стягивает сигнал» не нужно: сигнала нет и до R. Параметры памяти без уровня и корректора выигрыша на основной массе пар не дают (доля «нет» ≥ %s)." % (
        int((ra2["value"] <= 0).sum()), len(ra2), f(nm, 3)))
    w("")
    w("## 4. Решение по таблице 7.2 (выводится кодом из `S_D_ELEMENTS.csv` и `S_L_DIAG.csv`)")
    w("")
    w("| строка | условие | значения | выполняется | следствие |")
    w("|---|---|---|---|---|")
    for r in rows:
        w("| %s | %s | %s | **%s** | %s |" % (r[0], r[1], r[2], "да" if r[3] else "нет", r[4]))
    w("")
    ei = el[el["series"] == "S-RCKf"].iloc[0]
    w("**Как читать исход (i) = «%s».** В разд. 5 четыре вердикта; «вклад элемента не обнаружен» — только при Δ ≤ 0. У (i) Δ = %+.4f %%, 90 %% интервал [%+.4f; %+.4f], вердикт «%s».\n- Строка 1 таблицы 7.2 явно включает «или «хуже без него»»; строка 3 требует «(i) не обнаружен».\n- По тексту разд. 5 и 7 исход читается как «хуже без него» → строка 1. Строка 2 требует строгого «необходим» и не выполняется.\n- В чтении (i) двусмысленности нет." % (
        info["ci"], ei["delta_pct"], ei["lo90"], ei["hi90"], ei["verdict"]))
    w("- Альтернатива для дирижёра, если он сочтёт (i) «не обнаруженным»: строка 3 — S-RL1 с потерей уровня C, связь с π не регистрируется. Текст спецификации эту альтернативу не поддерживает.")
    w("")
    p2 = pr[pr["id"] == "P-f2"].iloc[0]
    w("**Где спецификация двусмысленна.** P-f2: значение %.4f при пороге %.4f, итог «%s» (`S_D_PRED.csv`). По разд. 2.4 тогда «выбор разд. 7 идёт по ветке «механизм не подтверждён»», но в таблице 7.2 такой строки нет. Два чтения, выбор не делаю:" % (p2["value"], p2["threshold"], p2["fulfilled"]))
    w("- **(А)** Ветка не меняет строку: разд. 2.4 прямо говорит, что невыполненное предсказание вердикт не меняет. S-RL1 и есть прямая проверка механизма H-Lπ с контролями (а), (б) — регистрировать S-RL1 (`S_L_SPEC.md`).")
    w("- **(Б)** «Механизм не подтверждён» блокирует доработку, основанную на этом механизме. S-RL1 (H-Lπ) не регистрируется до решения дирижёра; следующий шаг — без регистрации.")
    w("- **Строка 4 не выполняется:** (iii) = «%s», поэтому m в преемнике — как в S-RCK." % info["ciii"])
    w("- **Строка 5 при объявленном чтении О-8 (полное θ старого S-A2, уровень C) не выполняется:** %s < %.2f. Линия обучаемой памяти по критерию 7.2 не закрывается." % (f(info["l3"], 3), L3_THR))
    w("  - Если под «памятью A2» понимать только параметры памяти (ω, ψ, c) при фиксированном θ, доля = %s ≥ %.2f, и линия закрывалась бы. Это разложение, не абляция, и до счёта оно критерием не объявлялось. Выбор чтения — за дирижёром." % (f(info["l3m"], 3), L3_THR))
    mo = df[(df["section"] == "L3_month") & (df["item"] == "gain_rest99")]
    ra2 = mo[(mo["series"] == "S-RA2") & (mo["var"] == "full|F")]
    late = mo[(mo["series"] == "S-A2 (старый)") & (mo["var"] == "full|C") & (mo["group"] >= "2022")]
    w("  - Отрицательный результат с цифрами записан при любом чтении: выигрыша на 99 %% пар нет у R-памяти (S-RA2) в %d из %d месяцев, у старой — в %d из %d месяцев с 2022 года." % (
        int((ra2["value"] <= 0).sum()), len(ra2), int((late["value"] <= 0).sum()), len(late)))
    nb, add = n_base(), 3
    w("")
    w("**Следующая доработка:** при чтении (А) — одна регистрация: S-RL1 с контролями (а) и (б) по `S_L_SPEC.md`; N = %d + %d = %d. При чтении (Б) регистрации нет." % (nb, add, nb + add))
    w("")
    w("## 5. Не проверено и ограничения")
    w("- L1/L2 описательные. Статистики (бутстрапа) для разностей групп нет. Потолки — по исходу.")
    rr = [g(df, section="L2", series="W5-G", var="pi×" + a, item="R_pi_given_v") for a in ("v1", "v2")]
    w("- Порог %.1f в чтении О-5 задал архитектор: в спецификации его нет. У W5-G R_π|v = %s и %s." % (R_THR, f(rr[0], 2), f(rr[1], 2)))
    pos = [r["group"] for _, r in yrs.iterrows() if r["value"] > 0]
    w("- Связь «высокий π → прогноз завышен» по годам неустойчива по знаку: S_π > 0 в %d из %d лет (%s). Это главный риск S-RL1." % (len(pos), len(yrs), ", ".join(pos)))
    w("- L3: у старого S-A2 θ — среднее seeds, а прогноз считался по среднему ln C. Устойчивость к этому приближению показана выше.")
    w("")
    w("## 6. Воспроизводимость")
    w("- `PYTHONPATH=. python3 -W ignore <sp>/s_l/l12.py`, затем `… l3.py`, затем `… doc.py`; `doc.py --check` пересобирает в памяти и сравнивает. `<sp>` = `/private/tmp/claude-501/-Users-nilysenok-Desktop--------PhD-Vol-trading/5cfa4f4e-a264-412c-a614-0a8af6830acb/scratchpad`.")
    w("- Входы (sha256, 16 знаков): " + "; ".join("`%s` %s" % (p, sha(p)) for p in INPUTS) + ".")
    return "\n".join(L) + "\n"


def main():
    df, el, pr = load()
    csv, text = render(df, el, pr)
    if "--check" in sys.argv:
        bad = [p.name for p, t in ((OUT_CSV, csv), (OUT_MD, text)) if not p.exists() or p.read_text(encoding="utf-8") != t]
        rows, _ = decide(df, el, pr)
        print("строки 7.2, выполняются: %s" % ", ".join(r[0] for r in rows if r[3]))
        print("--check: %s" % ("ok" if not bad else "FAIL " + ", ".join(bad)))
        sys.exit(1 if bad else 0)
    OUT_CSV.write_text(csv, encoding="utf-8")
    OUT_MD.write_text(text, encoding="utf-8")
    print("записано: %s (%d строк данных), %s" % (OUT_CSV.name, csv.count("\n") - 4, OUT_MD.name))


if __name__ == "__main__":
    main()
