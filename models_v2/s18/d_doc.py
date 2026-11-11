"""S-D-EVAL (E.G2): отчёт `docs/otchety/harj_struct/S_D_REPORT.md` из таблиц `S_D_*.csv` и ключей CHECKPOINT (`S.D.registered`, `S.D.eval`). Числа берутся ТОЛЬКО из таблиц; вердикты и предсказания — из колонок таблиц (код `d_eval_tab`, `d_eval_par`).
Публичное API: `read_tables(rep) -> dict`; `render(t, cp) -> str`; `build(rep, ckpt)` пишет отчёт; `check(rep, ckpt) -> list` — только чтение: файл = перестроению из CSV и ключей; первая строка — пометка; границы дат и n перед каждой таблицей; контроль и знаменатель срезов названы.
Запуск: `python3 -W ignore -m models_v2.s18.d_doc [--build | --check]`."""
from __future__ import annotations

import json
import sys

import pandas as pd

from models_v2.s18 import d_doc_fmt as F
from models_v2.s18 import d_eval as E

REPORT = E.REP / "S_D_REPORT.md"
FIRST = "диагностика вклада элементов R, исследовательский результат, не OOS (абляции S-RCKm, S-RCKf не кандидаты; S-RCK — кандидат, X1 и W5-G — контрольные версии)"


def read_tables(rep=E.REP) -> dict:
    return {k: pd.read_csv(rep / n, comment="#") for k, n in zip(E.KEYS, E.OUTS)}


def sec_main(t: dict) -> list:
    tb = t["table"].set_index("series")
    rows = [[s, "%.6f" % tb.at[s, "qlike"]] + ["%+.4f" % tb.at[s, "d_vs_%s_pct" % b] if s != b else "—" for b in ("S-RCK", "W5-G", "X1")] for s in ("S-RCK", "S-RCKm", "S-RCKf", "W5-G", "X1", "S-RCKb", "K2", "K3")]
    return ["## 1. QLIKE на всех общих датах", "", F.bounds(t["table"]), "", "Обычная QLIKE без обрезки; Δ — в % QLIKE ряда-базы из заголовка колонки, минус — лучше.", ""] + F.md(["ряд", "QLIKE", "Δ к S-RCK, %", "Δ к W5-G, %", "Δ к X1, %"], rows)


def sec_delta(t: dict) -> list:
    d = t["delta"]
    rows = [[r.series, r.base, "%+.4f" % r.delta_pct, "[%+.4f; %+.4f]" % (r.lo90, r.hi90), "%.4f" % r.p_boot, "%.4f" % r.p_dm_nw8] for r in d.itertuples()]
    return ["## 2. Разности Δ с 90 % интервалами", "", F.bounds(d), "", "Стационарный бутстрап: B = %d, средний блок %d, seed %d; p DM NW-%d (односторонний «ряд лучше») — справочно; Холма и SPA нет." % (d.boot_B.iloc[0], d.boot_block.iloc[0], d.boot_seed.iloc[0], d.nw_lag.iloc[0]), ""] + F.md(["ряд", "база", "Δ, %", "90 % интервал", "p бутстрап", "p DM"], rows)


def sec_verdict(t: dict) -> list:
    el = t["elements"]
    rows = [[r.element, "%s против %s" % (r.series, r.base) if r.series == r.series else "—", "%+.4f" % r.delta_pct if r.delta_pct == r.delta_pct else "—", "%s" % r.code, r.verdict, r.main] for r in el.itertuples()]
    ci = el.code.iloc[1]
    note = "Разд. 7 спецификации: исход (i) «%s» относится к строке «(i) необходим или «хуже без него»»; следующая доработка выбирается только после S-L, регистрации нет." % ci if ci in ("needed", "worse_noise") else "Разд. 7: строка (i) «не обнаружен» — S-RL1 с потерей уровня C; регистрации нет."
    return ["## 3. Вклад элементов и вердикты (критерий разд. 5, выведен кодом из таблицы)", "", F.bounds(t["table"]), "", "Вклад = насколько хуже без элемента: Δ_x = 100·(QLIKE(без элемента) − QLIKE(S-RCK))/QLIKE(S-RCK); (ii) — база S-RCKb, без вердикта разд. 5.", ""] + F.md(["элемент", "сравнение", "Δ, %", "код", "вердикт", "основной"], rows) + ["", note]


def sec_pred(t: dict) -> list:
    rows = [[r.id, r.statement, "%+.4f" % r.value, "%+.4f" % r.threshold, r.fulfilled] for r in t["pred"].itertuples()]
    return ["## 4. Предсказания механизма (объявлены до счёта)", "", F.bounds(t["table"]), "", "Невыполненное предсказание вердикт не меняет; оно означает, что механизм `S_D_EXPLAIN.md` разд. 2.3 неполон (ветка «механизм не подтверждён»).", ""] + F.md(["id", "утверждение", "значение", "порог", "итог"], rows)


def sec_mech(t: dict) -> list:
    m = t["mech"]
    g = lambda s, i: m[(m.series == s) & (m["item"] == i)]["value"].iloc[0]
    rows = [[s, "%d" % g(s, "n_active_months"), "%d" % g(s, "n_months_theta0"), "%d / %d / %d" % (g(s, "n_months_lam_0.1"), g(s, "n_months_lam_0.01"), g(s, "n_months_lam_0.001")), "%d" % g(s, "n_days_C_ne_cfg8"), "%+.4f" % g(s, "mean_ln_C_cfg8"),
             "%+.4f" % g(s, "Ffr_vs_W5G_pct"), "%+.4f" % g(s, "reaction_pct"), "%+.4f" % g(s, "full|F_vs_W5G_pct"), "%+.4f" % g(s, "level_only|F_vs_W5G_pct")] for s in ("S-RCK", "S-RCKm", "S-RCKf")]
    return ["## 5. Механизм: месяцы, сдвиг уровня, замороженный π, разложение при фиксированных θ", "", "Окно D_M: t⁺ 2018-01-04…2026-11-10, n = 2198; 75 активных месяцев с 2020-08-03; δ = ln(C/cfg8) на днях C ≠ cfg8; «только уровень» — θ_c0, θ_p0 в 𝓜 при пересчитанных слоях (не абляция).", ""] + F.md(
        ["ряд", "мес. акт.", "мес. θ = 0", "λ* 0,1 / 0,01 / 0,001", "дней C ≠ cfg8", "среднее δ", "F^fr к W5-G, %", "реакция слоя, п. п.", "F к W5-G при θ, %", "только уровень, %"], rows)


def sec_years(t: dict) -> list:
    y = t["years"]
    cols = [(x, b) for x in ("S-RCKm", "S-RCKf") for b in ("S-RCK", "W5-G")]
    per = sorted(y["period"].unique())
    rows = [[p] + ["%+.4f" % y[(y.series == x) & (y.base == b) & (y.period == p)]["contrib_pct"].iloc[0] for x, b in cols] for p in per]
    rows.append(["Σ = Δ"] + ["%+.4f" % y[(y.series == x) & (y.base == b)]["delta_total_pct"].iloc[0] for x, b in cols])
    return ["## 6. Вклады лет", "", "Окно D_M: t⁺ 2018-01-04…2026-11-10, n = %d; вклад = 100·Σ_год(l_ряд − l_база)/Σ l_база, п. п. QLIKE базы; Σ по годам = Δ (10⁻¹²)." % y.groupby(["series", "base"])["n"].sum().iloc[0], ""] + F.md(["год"] + ["%s к %s" % c for c in cols], rows)


def sec_slices(t: dict, cp: dict) -> list:
    s = t["slices"]
    rows = [[r.series, r.base, r.slice, r.kind, "%d" % r.n, "%d" % r.n_removed, "%.6f" % r.q_base, "%+.4f" % r.delta_pct] for r in s.itertuples()]
    ov = t["mech"][t["mech"].section == "trim_overlap"].set_index("item")["value"]
    return ["## 7. Срезы (диагностика)", "", "Контроль в срезах — **S-RCK** (для (ii) — S-RCKb; для сравнений с W5-G и X1 — они сами). Знаменатель Δ — QLIKE базы на **оставшихся** датах: 2197 («без t⁺ 2022-02-24») и 1991 («без 2022 года») у S-RCK; "
            "исключённые даты одинаковы во всех сравнениях листа. Обрезка 22 + 22 — отдельно: даты выбраны по разности потерь d_t = ℓ(ряд) − ℓ(база) своей пары, поэтому наборы исключённых дат у разных пар разные "
            "(S-RCKm и S-RCKf к S-RCK: общих %d из %d, Жаккар %.3f); обрезка не входит в основную оценку." % (ov["n_common"], ov["n_each"], ov["jaccard"]), "", "Окно D_M: t⁺ 2018-01-04…2026-11-10, n = %d до исключений." % s["n"].max(), ""] + F.md(["ряд", "база (контроль)", "срез", "вид", "n", "исключено", "QLIKE базы на оставшихся", "Δ, %"], rows)


def sec_meta(cp: dict) -> list:
    r, e = cp["S.D.registered"], cp["S.D.eval"]
    return ["## 8. Регистрация и протокол", "", "- Строки S-RCKm, S-RCKf зарегистрированы одной записью `%s` (до первого parquet `D/`); N программы = %d, N_wide = %d (абляции входят по одной)." % (r["registered_at"], r["n"], r["n_wide"]),
            "- Ключ `S.D.eval` записан `%s`, до таблиц; B = %d, блок %d, seed %d, NW-%d, K обрезки = %d." % (e["written_at"], e["boot"], e["boot_block"], e["boot_seed"], e["lag_main"], e["ktrim"]),
            "- Отступления: имя отчёта `S_D_REPORT.md` и таблиц `S_D_*.csv` вместо `S_D_ABL.*` по поручению; обрезка и срезы — диагностика, основная Δ по всем 2198 датам.",
            "- Не сделано: Холм, SPA, MCS (спецификация: сырые p, справочно); лист S-L и регистрация следующей доработки — после этого отчёта; λ* = 0,1 — край сетки (ограничение, сетка не расширялась)."]


def render(t: dict, cp: dict) -> str:
    parts = [[FIRST], sec_main(t), sec_delta(t), sec_verdict(t), sec_pred(t), sec_mech(t), sec_years(t), sec_slices(t, cp), sec_meta(cp)]
    return "\n".join(l for p in parts for l in p + [""]) .rstrip("\n") + "\n"


def build(rep=E.REP, ckpt=E.CKPT) -> None:
    (rep / "S_D_REPORT.md").write_text(render(read_tables(rep), json.loads(ckpt.read_text(encoding="utf-8"))), encoding="utf-8")


def check(rep=E.REP, ckpt=E.CKPT) -> list:
    p, h0 = rep / "S_D_REPORT.md", E.tree_hash(rep, ckpt)
    txt = p.read_text(encoding="utf-8") if p.exists() else ""
    want = render(read_tables(rep), json.loads(ckpt.read_text(encoding="utf-8")))
    bad = [] if txt == want else ["отчёт ≠ перестроению из таблиц"]
    bad += ["первая строка не пометка"] * (not txt.startswith("диагностика вклада элементов R, исследовательский результат, не OOS"))
    bad += ["перед таблицей нет границ дат и n"] * (txt.count("D_M: t⁺ 2018-01-04…2026-11-10, n = ") < 7)
    bad += ["не названы контроль и знаменатель срезов"] * (not ("Контроль в срезах — **S-RCK**" in txt and "Знаменатель Δ" in txt and "2197" in txt and "1991" in txt))
    return bad + ([] if E.tree_hash(rep, ckpt) == h0 else ["--check изменил выходы"])


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
        print("S_D_REPORT.md записан")
    else:
        b = check()
        print("S-D doc --check: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
