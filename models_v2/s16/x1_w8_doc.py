"""W8-B, отчёт `W8_REPORT.md` — генератор из CSV-таблиц оценки (числа не вписываются руками; всё результатное читается из `W8_{TABLE,STATS,YEARS,CELLS,CONC,GUARD,MCS,BETA}.csv`).
Публичное API:
- `FIRST` — первая строка отчёта: три пометки разд. 6 `W8_SPEC.md` (`x1_w8_eval.MARKS`).
- `build(rep=REP) -> str`: порядок разд. 9 «Порядок отчёта»: 1. общая таблица окна M (K1, K2, K3, X1, W5-G, W8-G; справочно W8-Gb, M, ZB); 2. Δ к W5-G, X1, K2, K3 с 90 % интервалом, p DM, p Холма (N = 814, N_wide = 3333);
  3. годы, вклады по году × типу перерыва, путь β_{m,G}; 4. У1–У6 с числом, порогом и формулой, решение выведено кодом из `W8_GUARD.csv`; 5. чувствительность (только показ). «Подтверждено» — только при p Холма < 0,05.
- `problems(text, rep=REP) -> list`: текст равен перерисовке из таблиц, первая строка — пометки, «подтверждено» только при p Холма < 0,05 по своей строке, «доказательство» рядом с У1–У6 только в отрицании, нет «ожидаемого процента».
- `write(rep=REP)`; запуск: `python3 -W ignore -m models_v2.s16.x1_w8_doc [--write | --check]`; `--check` ничего не пишет (хэш отчёта до и после)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from models_v2.s16 import x1_w8_eval as E, x1_w8_eval_t as T
from models_v2.s16.x1_t0_doc import f6, tab

REP = E.REP
FIRST = "**" + "; ".join(E.MARKS) + "**"
STATUS = "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS (Ж-250 п. 6). Ступень 1 W8-B; код — `models_v2/s16/x1_w8_*.py`; X1, W5-G, K1, K2, K3 — сохранённые ряды, не менялись."
TYPES = T.TYPES
CONFIRMED = re.compile(r"(?<!не )подтверждено \(p Холма < 0,05\)")
NEGATION = "не является статистическим доказательством"


def pc(x) -> str:
    return "%+.2f %%" % x


def read(rep: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(rep / name, comment="#")


def num(v) -> str:
    try:
        return "%+.4f" % float(v)
    except (TypeError, ValueError):
        return str(v)


def sec_table(t: pd.DataFrame) -> list:
    x = t.iloc[0]
    return ["## 1. Общая таблица окна M: t⁺ %s – %s, n = %d" % (x.first_tplus, x.last_tplus, x.n), "",
            "QLIKE — среднее дневных потерь l = RV/F − ln(RV/F) − 1 на одних датах. Основные: K1, K2, K3, X1, W5-G, W8-G; справочно: W8-Gb, M, ZB.", "",
            tab(["ряд", "роль", "QLIKE"], [(r.series, "основной" if r.role == "main" else "справочно", f6(r.qlike)) for r in t.itertuples()]), ""]


def sec_stats(s: pd.DataFrame) -> list:
    L = ["## 2. Δ W8-G к W5-G, X1, K2, K3: 90 %% интервал блочного бутстрапа (блок 20, B 2000, seed 0), p DM (НУ, лаг 5, односторонний), p Холма (N = %d, N_wide = %d)" % (T.N_HOLM, T.N_WIDE), "",
         "Минус — W8-G лучше базы. «Подтверждено» — только при p Холма < 0,05; неопределённость разницы — интервал и p. Холм — по четырём основным строкам.", ""]
    m = s[s["kind"].isin(["main", "reference"])]
    L += [tab(["ряд", "база", "Δ", "90 % интервал", "p DM", "p Холма N", "p Холма N_wide", "вывод"],
              [(r.hybrid, r.base, pc(r.delta_pct), "[%+.2f; %+.2f]" % (r.lo90, r.hi90), "%.4f" % r.p_one, "—" if pd.isna(r.p_holm_N) else "%.4f" % r.p_holm_N, "—" if pd.isna(r.p_holm_Nwide) else "%.4f" % r.p_holm_Nwide, r.verdict)
               for r in m.itertuples()]), ""]
    p = s[s["kind"] == "perm_p"].iloc[0]
    return L + ["Сдвиги ряда z (19 сдвигов 252 + 100·(j − 1); в N не входят): %s = %.4f." % (p.hybrid, p.p_one), ""]


def sec_years(y: pd.DataFrame, c: pd.DataFrame, b: pd.DataFrame) -> list:
    order = list(dict.fromkeys(y["period"]))
    q = y[y["base"] == "-"].pivot(index="period", columns="series", values="q_series")
    L = ["## 3. Годы, вклады по году × типу перерыва, путь β", "", "QLIKE по периодам:", "",
         tab(["период", "n"] + list(T.MAIN), [(p, int(y[(y["period"] == p) & (y["series"] == "K1")]["n"].iloc[0])) + tuple(f6(q.at[p, k]) for k in T.MAIN) for p in order]), ""]
    for base in T.BASES:
        z = y[(y["series"] == "W8-G") & (y["base"] == base)].set_index("period")
        L += ["Δ и вклад W8-G к %s (c_y = Σ(l_W8 − l_база)/(n·Q_база), Σ c_y = Δ окна M = %s):" % (base, pc(z["delta_total_pct"].iloc[0])), "",
              tab(["период", "Δ периода", "вклад, п. п."], [(p, pc(z.at[p, "delta_pct"]), "%+.4f" % z.at[p, "contrib_pct"]) for p in order]), ""]
    for base in ("W5-G", "X1"):
        k = c[(c["base"] == base) & (c["type"].isin(TYPES + ("all",)))]
        rows = []
        for p in list(dict.fromkeys(k["period"])):
            v = k[k["period"] == p].set_index("type")
            rows.append((p,) + tuple("%d / %+.4f" % (v.at[t, "n"], v.at[t, "contrib_pp_period"]) for t in TYPES) + (pc(v.at["all", "delta_pct"]),))
        L += ["Вклад W8-G к %s по типу перерыва (n / п. п. QLIKE базы периода; Σ по типам = Δ периода):" % base, "", tab(["период"] + list(TYPES) + ["Δ периода"], rows), ""]
    inc = [(G, b.loc[b["incl_" + G] == 1, "refit_date"].min() if (b["incl_" + G] == 1).any() else "нет", int(b["incl_" + G].sum())) for G in ("T1", "T2", "T3")]
    return L + ["Месяцы включения групп (n ≥ 30 пар gap = 1 группы в закрытом пуле; всего %d месяцев переоценки):" % len(b), "",
                tab(["группа", "первый месяц включения", "месяцев включена"], inc), ""]


def sec_guard(g: pd.DataFrame) -> list:
    r, d = g[g["id"] != "решение"], g[g["id"] == "решение"].iloc[0]
    return ["## 4. Основной критерий и защитные условия У1–У6 (в зарегистрированном виде)", "",
            "Пороги и формулы объявлены до счёта и не менялись. Прохождение порогов не является статистическим доказательством; неопределённость разницы — раздел 2.", "",
            tab(["id", "условие", "значение", "порог", "формула порога", "выполнено"], [(x.id, x.condition, num(x.value), x.threshold, x.formula, "да" if x.passed else "нет") for x in r.itertuples()]), "",
            "**Решение (выведено кодом из таблицы):** %s." % d["value"], ""]


def sec_sens(s: pd.DataFrame, c: pd.DataFrame, m: pd.DataFrame) -> list:
    v = s[s["kind"] == "sens"]
    L = ["## 5. Чувствительность (только показ; β по срезам не переоценивается)", "",
         tab(["вариант", "база", "n", "QLIKE W8-G", "QLIKE базы", "Δ"], [(r.variant, r.base, r.n, f6(r.qlike_h), f6(r.qlike_b), pc(r.delta_pct)) for r in v.itertuples()]), "",
         "Концентрация выигрыша (вклады периодов; `W8_CONC.csv`):", ""]
    sm = c[c["item"] == "summary"].pivot(index="base", columns="key", values="value")
    L += [tab(["база", "периодов с выигрышем", "доля лучшего", "доля двух лучших", "сумма ухудшений, п. п."],
              [(b, int(sm.at[b, "years_with_gain"]), "—" if pd.isna(sm.at[b, "share_best_year"]) else "%.3f" % sm.at[b, "share_best_year"],
                "—" if pd.isna(sm.at[b, "share_two_best_years"]) else "%.3f" % sm.at[b, "share_two_best_years"], "%+.4f" % sm.at[b, "worsening_sum_pct"]) for b in T.BASES]), "",
          "MCS 90 % по {W8-G, W5-G, X1, ZB, K2, K3, M}:", "", tab(["ряд", "QLIKE", "p MCS", "в MCS"], [(r.series, f6(r.mean_qlike), "%.3f" % r.p, "да" if r.in_mcs else "нет") for r in m.itertuples()]), ""]
    return L


def build(rep: Path = REP) -> str:
    L = [FIRST, "", "# W8-B: поправка β по терцилю RV IMOEX дня d — результаты ступени 1", "", STATUS, ""]
    st = read(rep, "W8_STATS.csv")
    L += sec_table(read(rep, "W8_TABLE.csv")) + sec_stats(st) + sec_years(read(rep, "W8_YEARS.csv"), read(rep, "W8_CELLS.csv"), read(rep, "W8_BETA.csv"))
    L += sec_guard(read(rep, "W8_GUARD.csv")) + sec_sens(st, read(rep, "W8_CONC.csv"), read(rep, "W8_MCS.csv"))
    return "\n".join(L).rstrip("\n") + "\n"


def _confirmed(text: str, s: pd.DataFrame) -> list:
    """«Подтверждено» — по СВОЕЙ строке: строка с p Холма ≥ 0,05 не может его нести; вне строк таблицы Δ оно не допускается."""
    bad, lines = [], text.splitlines()
    for r in s[s["kind"] == "main"].itertuples():
        for ln in (x for x in lines if x.startswith("| %s | %s |" % (r.hybrid, r.base))):
            if CONFIRMED.search(ln) and not (r.p_holm_N < 0.05):
                bad.append("«подтверждено» у строки %s к %s при p Холма %.4f ≥ 0,05" % (r.hybrid, r.base, r.p_holm_N))
    rest = [x for x in lines if not any(x.startswith("| %s | %s |" % (r.hybrid, r.base)) for r in s.itertuples() if r.kind in ("main", "reference"))]
    return bad + (["«подтверждено» вне строк таблицы Δ"] if any(CONFIRMED.search(x) for x in rest) else [])


def _proof(text: str) -> list:
    sec4 = text.split("## 4.")[1].split("## 5.")[0] if "## 4." in text and "## 5." in text else ""
    return [m.group(0) for m in re.finditer(r"[^.\n]*доказ[^.\n]*\.?", text, re.I)
            if (m.group(0) in sec4 or re.search(r"У[1-6]|порог", m.group(0))) and NEGATION not in m.group(0)]


def problems(text: str, rep: Path = REP) -> list:
    bad = [] if text == build(rep) else ["отчёт ≠ перерисовке из таблиц"]
    bad += [] if text.startswith(FIRST) else ["первая строка не содержит пометки разд. 6"]
    bad += _confirmed(text, read(rep, "W8_STATS.csv"))
    g = read(rep, "W8_GUARD.csv")
    want = T.decide(g[g["id"] != "решение"])
    bad += [] if "**Решение (выведено кодом из таблицы):** %s." % want in text and g.loc[g["id"] == "решение", "value"].iloc[0] == want else ["решение ≠ выведенному из W8_GUARD.csv"]
    bad += ["«доказательство» вне отрицания рядом с У1–У6 / в разделе защитных условий: %s" % s.strip() for s in _proof(text)]
    bad += ["есть «ожидаемый процент»"] if re.search(r"ожидаем", text, re.I) else []
    return bad


if __name__ == "__main__":
    if "--write" in sys.argv:
        (REP / "W8_REPORT.md").write_text(build(), encoding="utf-8")
        print("W8_REPORT.md записан")
    elif "--check" in sys.argv:
        p = REP / "W8_REPORT.md"
        h0 = p.read_bytes() if p.exists() else b""
        b = problems(p.read_text(encoding="utf-8")) if p.exists() else ["нет W8_REPORT.md"]
        b += [] if not p.exists() or p.read_bytes() == h0 else ["--check изменил отчёт"]
        print("W8 doc --check: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        sys.exit("укажите --write или --check")
