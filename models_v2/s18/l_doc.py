"""S-L-EVAL: отчёт `S_L_EVAL_REPORT.md` — собирается ТОЛЬКО из таблиц `S_L_EVAL_*.csv` и ключей CHECKPOINT (ни одно число не вписано вручную). `--write` пишет, `--check` пересобирает в памяти и сравнивает (только чтение).
Запуск: `python3 -W ignore -m models_v2.s18.l_doc [--write | --check]`."""
from __future__ import annotations

import datetime as dt
import json
import sys

import pandas as pd

from models_v2.s18 import l_eval as E
from models_v2.s18 import l_reg as LR
from models_v2.s18 import l_reg2 as LR2
from models_v2.s18 import l_walk as LW

FIRST = "исследовательский результат, не OOS; S-RL1a, S-RL1b, S-RL1b2 — контроли, не кандидаты"


def tabs(rep=E.REP) -> dict:
    return {k: pd.read_csv(rep / n, comment="#") for k, n in zip(E.KEYS, E.OUTS)}


def md(df: pd.DataFrame, fmt: dict) -> str:
    cols = list(fmt)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join((f % r[c]) if isinstance(f, str) and pd.notna(r[c]) and not isinstance(r[c], str) else str(r[c]) for c, f in fmt.items()) + " |")
    return "\n".join(out)


def tm(s: float) -> str:
    return dt.datetime.fromtimestamp(s).strftime("%d.%m.%Y %H:%M:%S")


def build(t: dict, cp: dict) -> str:
    v, tb, dl = t["verdict"].set_index("id"), t["table"], t["delta"]
    pt = tb.copy()
    pt.columns = [c.replace("d_vs_", "Δ к ").replace("_pct", "") for c in pt.columns]
    cols = {"series": "%s", "qlike": "%.6f"} | {c: "%+.4f" for c in pt.columns if c.startswith("Δ к ")}
    d = dl[dl["base"].isin(["W5-G", "X1", "K2", "K3", "S-RCK", "S-RL1a", "S-RL1b2", "S-RL1b"]) | (dl["series"] == "X1_R(S-RL1)")]
    dd = {"series": "%s", "base": "%s", "delta_pct": "%+.4f", "lo90": "%+.4f", "hi90": "%+.4f", "p_dm_nw8": "%.4f", "p_holm_N": "%.4f", "p_spa_c": "%.4f"}
    y = t["years"]
    ys = y[(y["series"] == "S-RL1") & (y["base"] == "W5-G")][["period", "n", "q_series", "q_base", "contrib_pct"]]
    mm = t["mech"]
    ay = mm[(mm["section"] == "a_by_year") & (mm["series"] == "S-RL1")].copy()
    ay[["year", "stat"]] = ay["item"].str.split("|", expand=True)
    ayp = ay.pivot(index="year", columns="stat", values="value").reset_index()[["year", "n_months", "n_lam_pos", "a_mean", "a_median"]]
    sl = t["slices"]
    sl = sl[sl["series"].isin(["S-RL1", "S-RL1b2", "S-RL1b", "S-RL1a"]) & sl["base"].isin(["W5-G", "S-RL1b2", "S-RL1b"])]
    g = t["guard"]
    de = mm[(mm["section"] == "decomp")][["series", "item", "value"]]
    reg, reg2, ev = cp[LR.KEY]["registered_at"], cp[LR2.KEY]["registered_at"], cp[E.KEY]["written_at"]
    ptimes = {k: tm((LW.L_DIR / (k + ".parquet")).stat().st_mtime) for k in LW.VARS_L}
    n0 = tb["n"].iloc[0]
    L = ["# %s" % FIRST, "",
         "Выведено из таблиц `S_L_EVAL_*.csv` (`l_doc`); N программы %d / %d; D_M: t⁺ %s…%s, n = %d. Обычная QLIKE (среднее дневных потерь, без обрезки); Δ в %% QLIKE базы, минус — лучше." % (cp[E.KEY]["n_holm"], cp[E.KEY]["n_holm_wide"], tb["first_tplus"].iloc[0], tb["last_tplus"].iloc[0], n0), "",
         "## 1. Критерий «взаимодействие подтверждено численно» (Ж-257: основной контроль (б) — S-RL1b2)",
         "- Основная версия: **%s**. Справочно, по циклическому S-RL1b (спецификация): **%s**." % (v.loc["interaction", "value"], v.loc["interaction_ref", "value"]),
         "- Условия: 1 — %s (%s); 2 — %s (%s); 3 — %s (%s); справочное 2 — %s." % (tuple(x for i in ("C1", "C2", "C3") for x in (v.loc[i, "condition"], "выполнено" if v.loc[i, "passed"] else "нет")) + (v.loc["C2ref", "condition"],)),
         "- Статистически: %s — «подтверждено» только при Δ < 0, p Холма < 0,05 и p SPA < 0,05: **%s**." % (v.loc["statistical", "value"], "да" if v.loc["statistical", "passed"] else "нет"),
         "- Пороги этапа S: **%s**. Это статус кандидата не меняет (замену S-RCK решает дирижёр)." % v.loc["этап S", "value"],
         "- Против S-RCK (5.4): %s." % v.loc["5.4", "condition"], "",
         "## 2. QLIKE и Δ к базам (t⁺ %s…%s, n = %d)" % (tb["first_tplus"].iloc[0], tb["last_tplus"].iloc[0], n0), md(pt, cols), "",
         "## 3. Δ с 90 %%-интервалами (стационарный бутстрап B = %d, блок %d, seed %d; DM NW-%d)" % (cp[E.KEY]["boot"], cp[E.KEY]["boot_block"], cp[E.KEY]["boot_seed"], cp[E.KEY]["lag_main"]),
         "Холм и SPA — только у семейства S-RL1 против X1 (слой), W5-G, K2, K3 (m = 4).", md(d.assign(series=d["series"] + ""), dd), "",
         "## 4. Годы: S-RL1 против W5-G (вклад = 100·Σ(l_h − l_b)/Σ l_b, Σ = Δ; минус — лучше)", md(ys, {"period": "%s", "n": "%d", "q_series": "%.6f", "q_base": "%.6f", "contrib_pct": "%+.4f"}), "",
         "## 5. Оценка a по годам (S-RL1, месяцы с λ* > 0)", md(ayp, {"year": "%s", "n_months": "%d", "n_lam_pos": "%d", "a_mean": "%+.4f", "a_median": "%+.4f"}), "",
         "## 6. Срезы (контроль и знаменатель названы в строке; обрезка — только диагностика, даты выбраны по разности потерь пары)",
         md(sl, {"series": "%s", "base": "%s", "slice": "%s", "n": "%d", "q_base": "%.6f", "q_series": "%.6f", "delta_pct": "%+.4f"}), "",
         "## 7. Пороги этапа S", md(g, {"pair": "%s", "id": "%s", "value": "%.4f", "threshold": "%s", "passed": "%s"}), "",
         "## 8. Предсказания P-L1…P-L4 и разложение «замороженный π + реакция слоя»",
         md(t["verdict"][t["verdict"]["section"] == "5.5"], {"id": "%s", "condition": "%s", "value": "%s", "threshold": "%s"}), "", md(de, {"series": "%s", "item": "%s", "value": "%+.4f"}), "",
         "## 9. Порядок и оговорки",
         "- Регистрация S-RL1, S-RL1a, S-RL1b: %s; S-RL1b2 (Ж-257 п. 3, до счёта rl1b2): %s; ключ `S.L.eval`: %s; parquet: %s." % (reg, reg2, ev, ", ".join("%s %s" % (k, s) for k, s in ptimes.items())),
         "- Циклический S-RL1b переносит последние 252 значения π в начало истории (будущее в обучении ранних месяцев); он оставлен как зарегистрированный по спецификации, основной контроль — S-RL1b2 без переноса (первые 252 пары вне пула обучения, не заполняются).",
         "- λ* = 0,1 — край сетки (не расширялась). Все числа — разработка на раскрытой истории: исследовательский результат, не OOS."]
    return "\n".join(L) + "\n"


def current(rep=E.REP, ckpt=E.CKPT) -> str:
    return build(tabs(rep), json.loads(ckpt.read_text(encoding="utf-8")))


def check(rep=E.REP, ckpt=E.CKPT) -> list:
    p = rep / E.REPORT
    bad = [] if p.exists() and p.read_text(encoding="utf-8") == current(rep, ckpt) else ["отчёт ≠ пересборке из таблиц"]
    txt = p.read_text(encoding="utf-8") if p.exists() else ""
    bad += ["первая строка не пометка"] * (not txt.startswith("# " + FIRST))
    bad += ["нет границ дат и n"] * ("2018-01-04…2026-11-10, n = 2198" not in txt)
    bad += ["в срезах не названы контроль и знаменатель"] * ("без t⁺ = 2022-02-24" not in txt or "обрезка — только диагностика" not in txt)
    return bad


if __name__ == "__main__":
    if "--write" in sys.argv:
        (E.REP / E.REPORT).write_text(current(), encoding="utf-8")
        print("записано", E.REPORT)
    elif "--check" in sys.argv:
        b = check()
        print("S-L doc --check: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        sys.exit("укажите --write или --check")
