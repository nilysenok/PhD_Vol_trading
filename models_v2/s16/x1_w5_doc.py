"""W5-B, отчёт `W5_REPORT.md` — генератор из CSV-таблиц оценки (числа не вписываются руками; всё результатное читается из `W5_{TABLE,STATS,YEARS,GAPDAYS,GUARD,SENS,CONC,MCS}.csv` и шапки `W5_GAP_PIT.csv`).
Публичное API:
- `FIRST` — первая строка отчёта: три пометки 11.5; `W4_STATUS` — формулировка статуса W4 (Ж-251 п. 1, дословно шапка спецификации).
- `build(rep=REP) -> str`: текст отчёта в порядке 11.6 (общая таблица без «ожидаемого процента»; Δ с 90 % интервалом и p DM; p Холма, N = 798; годы; дни gap = 1 и gap = 0; У1–У5 с числом, порогом и формулой;
  чувствительность; MCS; перестановки; сверка gap^PIT и gap^real отдельной строкой; решение выведено из GUARD). «Подтверждено» пишется только при p Холма < 0,05; прохождение У1–У5 статистическим доказательством не называется.
- `problems(text, rep=REP) -> list`: `--check` — текст равен перерисовке из таблиц (все числа = таблицам), первая строка — пометки, W4 — формулировкой шапки, «подтверждено» только при p Холма < 0,05, нет слов об ожидаемом проценте.
- `write(rep=REP)`; запуск: `python3 -W ignore -m models_v2.s16.x1_w5_doc [--write | --check]`; `--check` ничего не пишет (хэш отчёта до и после)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w5_eval as E, x1_w5_eval_t as T, x1_w5_gap_cal as CAL
from models_v2.s16.x1_t0_doc import f6, tab

REP = E.REP
FIRST = "**" + "; ".join(m[2:] for m in CAL.MARKS) + "**"
W4_STATUS = ("Предварительный вариант S1 улучшил QLIKE относительно M на 1,65 % на всём окне. Он не прошёл зарегистрированный порог ухудшения 2018–2021: +0,89 % при пороге +0,50 %. "
             "Ступень 1 не пройдена, полный гибрид W4 не рассчитывался. Порог W4 не меняется, W4 не перезапускается.")
STATUS = "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS (Ж-250 п. 6). Ступень 1 W5-B; код — `models_v2/s16/x1_w5_*.py`; сохранённые ряды и прежние результаты не менялись."


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
    L = ["## 1. Общая таблица окна M: t⁺ %s – %s, n = %d" % (x.first_tplus, x.last_tplus, x.n), "",
         "QLIKE — среднее дневных потерь l = RV/F − ln(RV/F) − 1 на одних датах. Основные: K1, K2, K3, X1, W5-G; справочно: M, W5-Ga, W5-Gb, W5-GK2.", "",
         tab(["ряд", "роль", "QLIKE"], [(r.series, "основной" if r.role == "main" else "справочно", f6(r.qlike)) for r in t.itertuples()]), ""]
    return L


def sec_stats(s: pd.DataFrame) -> list:
    L = ["## 2. Δ W5-G к X1, K2, K3: 90 %% интервал блочного бутстрапа (блок 20, B 2000, seed 0), p DM (НУ, лаг 5, односторонний), p Холма (N = %d, N_wide = %d)" % (T.N_HOLM, T.N_WIDE), "",
         "Минус — W5-G лучше базы. «Подтверждено» — только при p Холма < 0,05; неопределённость разницы — интервал и p.", ""]
    m = s[s["kind"].isin(["main", "reference"])]
    L += [tab(["W5-G / ряд", "база", "Δ", "90 % интервал", "p DM", "p Холма N", "p Холма N_wide", "вывод"],
              [(r.hybrid, r.base, pc(r.delta_pct), "[%+.2f; %+.2f]" % (r.lo90, r.hi90), "%.4f" % r.p_one, "—" if pd.isna(r.p_holm_N) else "%.4f" % r.p_holm_N, "—" if pd.isna(r.p_holm_Nwide) else "%.4f" % r.p_holm_Nwide, r.verdict) for r in m.itertuples()]), ""]
    p = s[s["kind"] == "perm_p"].iloc[0]
    return L + ["Перестановки (19 сдвигов меток 252 + 100·(j − 1); в N не входят): %s = %.4f." % (p.hybrid, p.p_one), ""]


def sec_years(y: pd.DataFrame) -> list:
    w = y[(y["series"] == "W5-G") & (y["base"] != "-")]
    q = y[y["base"] == "-"].pivot(index="period", columns="series", values="q_series")
    order = list(dict.fromkeys(y["period"]))
    L = ["## 3. Годы и половины: QLIKE и вклады (c_y = Σ(l_W5 − l_база)/(n·Q_база), Σ c_y = Δ)", "",
         tab(["период", "n"] + list(T.MAIN), [(p, int(y[(y["period"] == p) & (y["series"] == "K1")]["n"].iloc[0])) + tuple(f6(q.at[p, k]) for k in T.MAIN) for p in order]), ""]
    for b in ("X1", "K2", "K3"):
        z = w[w["base"] == b].set_index("period")
        L += ["Δ и вклад W5-G к %s (Δ окна M = %s):" % (b, pc(z["delta_total_pct"].iloc[0])), "",
              tab(["период", "Δ периода", "вклад, п. п."], [(p, pc(z.at[p, "delta_pct"]), "%+.4f" % z.at[p, "contrib_pct"]) for p in order]), ""]
    return L


def sec_gap(g: pd.DataFrame, gp_head: list) -> list:
    return ["## 4. Дни gap = 1 и gap = 0 отдельно (в днях gap = 0 разность потерь — ноль)", "",
            tab(["подмножество", "n", "доля", "QLIKE X1", "QLIKE W5-G", "Δ внутри", "вклад в Δ окна", "макс. модуль разности потерь"],
                [(r.subset, r.n, "%.3f" % r.share, f6(r.q_x1), f6(r.q_w5g), pc(r.delta_pct), "%+.4f" % r.contrib_pct, "%.6f" % r.max_abs_loss_diff) for r in g.itertuples()]), "",
            "**Метка gap^PIT и сверка с gap^real (`W5_GAP_PIT.csv`):**", ""] + ["- " + ln.lstrip("# ") for ln in gp_head[3:]] + [""]


def sec_guard(g: pd.DataFrame) -> list:
    r = g[g["id"] != "решение"]
    d = g[g["id"] == "решение"].iloc[0]
    return ["## 5. Защитные условия У1–У5 (в зарегистрированном виде)", "",
            "Пороги и формулы объявлены до счёта и не менялись. Прохождение порогов не является статистическим доказательством; неопределённость разницы — раздел 2.", "",
            tab(["id", "условие", "значение", "порог", "формула порога", "выполнено"], [(x.id, x.condition, num(x.value), x.threshold, x.formula, "да" if x.passed else "нет") for x in r.itertuples()]), "",
            "**Решение (выведено кодом из таблицы):** %s." % d["value"], ""]


def sec_rest(s: pd.DataFrame, m: pd.DataFrame) -> list:
    return ["## 6. Чувствительность (только диагностика; β по срезам не переоценивается)", "",
            tab(["вариант", "исключено", "n", "QLIKE X1", "QLIKE W5-G", "Δ к X1"], [(r.variant, r.excluded, r.n, f6(r.q_x1), f6(r.q_w5g), pc(r.delta_pct)) for r in s.itertuples()]), "",
            "## 7. MCS 90 % по {W5-G, X1, ZB, K2, K3, M}", "", tab(["ряд", "QLIKE", "p MCS", "в MCS"], [(r.series, f6(r.mean_qlike), "%.3f" % r.p, "да" if r.in_mcs else "нет") for r in m.itertuples()]), ""]


def build(rep: Path = REP) -> str:
    gp = [ln for ln in (rep / "W5_GAP_PIT.csv").read_text(encoding="utf-8").splitlines() if ln.startswith("#")]
    L = [FIRST, "", "# W5-B: поправка уровня X1 после перерыва торгов — результаты ступени 1", "", STATUS, "", "**Статус W4.** " + W4_STATUS, ""]
    L += sec_table(read(rep, "W5_TABLE.csv")) + sec_stats(read(rep, "W5_STATS.csv")) + sec_years(read(rep, "W5_YEARS.csv")) + sec_gap(read(rep, "W5_GAPDAYS.csv"), gp)
    L += sec_guard(read(rep, "W5_GUARD.csv")) + sec_rest(read(rep, "W5_SENS.csv"), read(rep, "W5_MCS.csv"))
    return "\n".join(L).rstrip("\n") + "\n"


def problems(text: str, rep: Path = REP) -> list:
    bad = [] if text == build(rep) else ["отчёт ≠ перерисовке из таблиц"]
    bad += [] if text.startswith(FIRST) else ["первая строка не содержит пометки 11.5"]
    bad += [] if "Статус W4." in text and W4_STATUS in text else ["статус W4 не формулировкой шапки"]
    bad += _confirmed(text, read(rep, "W5_STATS.csv"))
    bad += ["«доказательство» вне отрицания рядом с У1–У5 / в разделе защитных условий: %s" % s.strip() for s in _proof(text)]
    bad += ["есть «ожидаемый процент»"] if re.search(r"ожидаем", text, re.I) else []
    return bad


CONFIRMED = re.compile(r"(?<!не )подтверждено \(p Холма < 0,05\)")
NEGATION = "не является статистическим доказательством"


def _confirmed(text: str, s: pd.DataFrame) -> list:
    """«Подтверждено» — по СВОЕЙ строке: строка с p Холма ≥ 0,05 (или без p) не может его нести; вне строк основной таблицы оно не допускается."""
    bad, lines = [], text.splitlines()
    for r in s[s["kind"] == "main"].itertuples():
        for ln in (x for x in lines if x.startswith("| %s | %s |" % (r.hybrid, r.base))):
            if CONFIRMED.search(ln) and not (r.p_holm_N < 0.05):
                bad.append("«подтверждено» у строки %s к %s при p Холма %.4f ≥ 0,05" % (r.hybrid, r.base, r.p_holm_N))
    rest = [x for x in lines if not any(x.startswith("| %s | %s |" % (r.hybrid, r.base)) for r in s.itertuples() if r.kind in ("main", "reference"))]
    return bad + (["«подтверждено» вне строк таблицы Δ"] if any(CONFIRMED.search(x) for x in rest) else [])


def _proof(text: str) -> list:
    """Предложения со словом «доказ…» (доказательство, доказан…) в разделе 5 или с У1–У5 / «порог», кроме отрицания «не является статистическим доказательством»."""
    sec5 = text.split("## 5.")[1].split("## 6.")[0] if "## 5." in text and "## 6." in text else ""
    out = []
    for m in re.finditer(r"[^.\n]*доказ[^.\n]*\.?", text, re.I):
        near = m.group(0) in sec5 or re.search(r"У[1-5]|порог", m.group(0))
        if near and NEGATION not in m.group(0):
            out.append(m.group(0))
    return out


if __name__ == "__main__":
    if "--write" in sys.argv:
        (REP / "W5_REPORT.md").write_text(build(), encoding="utf-8")
        print("W5_REPORT.md записан")
    elif "--check" in sys.argv:
        p = REP / "W5_REPORT.md"
        h0 = p.read_bytes()
        b = problems(p.read_text(encoding="utf-8")) if p.exists() else ["нет W5_REPORT.md"]
        b += [] if p.read_bytes() == h0 else ["--check изменил отчёт"]
        print("W5 doc --check: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        sys.exit("укажите --write или --check")
