"""V4-C, G18: C_REPORT.md — первая строка: лучшая C-a по crit_panel против порога go и против B6; все числа берутся из таблиц стадии. Stop/go этапа C — после C-b."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_b_report as BR, v4_c_cache as CC, v4_c_stats as CS, v4_z_feat as ZF

DEVIATIONS = [
    "Масштаб штрафов как в B: κ = λ = 250 «пар-эквивалентов» умножены на s² = mean ln²(M/H) (кривизна одной пары по весу), применены к параметрам логита (α, δ_i, β) буквально; для логита кривизна одной пары меньше "
    "в σ'² ≤ 1/16 раз, то есть штраф относительно сильнее, чем для веса в B (на синтетике с известным гейтом истинный β = 1,2 оценивается как ≈ 0,9: сжатие, знак и порядок сохранены; тест `test_known_gate_sign_and_order_recovered`). "
    "Масштаб объявлен до счёта и не подбирался; вариант «точный пара-эквивалент» (÷16) не считался — выбор по результатам запрещён, решение за дирижёром (лист C-b).",
    "Тождество с B2 (G16) проверено как тождество отображения: при β = 0, δ = 0, ℓ-константе F = B2 при w_H = 1 − σ(α) до 1e-8 и как совпадение оптимумов без штрафов; штрафы в B2 (на вес) и в C (на логит) разные, "
    "поэтому оценённые параметры C с β = 0 не равны оценкам B2 — это не дефект.",
    "Строка обучения с нефинитным признаком (первые строки бумаги, дыры календаря: 34 строки lerr/sur_mkt, 17 — ewqd) в оценку не входит; в прогнозе такой признак после стандартизации = 0 (среднее окна), число строк — в C_WEIGHTS.",
    "Вклад HAR-surprise и Intraday Shape по отдельности: конфигураций сверх C1–C4 нет (запрещено); даётся диагностика C4 с обнулением коэффициентов признаков БЕЗ переоценки (`C4_diag_*`), N не расходует; "
    "это не вклад «блока при честной переоценке» — переоценка без блока потребовала бы новой конфигурации (лист C-b).",
    "Проверка сдвига Д1: профиль каждого признака — корреляция с ln RV цели при сдвигах −2…+1 по цели ≤ 2021 (среднее по бумагам); максимум профиля обязан быть на последнем известном дне (−1), утечка — максимум на 0/+1 с превышением > 0,05. "
    "Для ell и |ell| (разность прогнозов не имеет персистентности уровня; у ell максимум на 0 — это предсказательное содержание M, не утечка) проверка Д1 — через связанные величины: профили ln H, ln K2, ln K3 (`v4_audit.shift_ok`). "
    "Для всех признаков дополнительно — обрезка (подмена данных после d не меняет признак d).",
    "CHECKPOINT: ключ `used` (программа, 0) не менялся; расход листа — `C_configs_spent` = 4 и `used_after_C_a` = 9; в `v4_b_stage` и `v4_trials.verify` — минимальные правки (повторный счёт B не затирает строки Ж-222:C*; verify допускает их); "
    "формулы RMSE/MAE/R² B заменены на hp_hyb_out (`v4_b_stats.metr`), прогнозы B пересчитаны тем же кодом без изменений crit_panel.",
]


def crit_lines(crit: pd.Series, thr: float) -> list:
    return ["| ряд | crit_panel | к порогу go (−) |", "|---|---|---|"] + ["| %s | %.5f | %+.5f |" % (s, v, v - thr) for s, v in crit.items()]


def delta_lines(comp: pd.DataFrame, best: str) -> list:
    L = ["| окно | к K1 | к K2 | к K3 | к D2 | к B6 |", "|---|---|---|---|---|---|"]
    for w in ("all", "2022_2025"):
        c = comp[comp["window"] == w].set_index("series")["qlike"]
        d = {x: 100.0 * float((c[best] - c[x]) / c["k1"]) for x in ("k1", "k2", "k3", "d2", "B6")}
        L.append("| %s | %.2f | %.2f | %.2f | %.2f | %.2f |" % (w, d["k1"], d["k2"], d["k3"], d["d2"], d["B6"]))
    return L


def block_lines(crit: pd.Series, comp: pd.DataFrame, wt: pd.DataFrame) -> list:
    L = ["| шаг | Δ crit_panel | % уровня K1 |", "|---|---|---|"] + ["| %s | %+.5f | %+.2f |" % (r.step, r.d_crit, r.d_crit_pct_of_k1) for r in CS.block_steps(crit).itertuples()]
    L += ["", "Диагностика C4 (обнуление коэффициентов без переоценки; не конфигурации): "]
    a = comp[comp["window"] == "all"].set_index("series")["qlike"]
    for name, fs in (("C4_diag_no_har_surprise", "sur_mkt"), ("C4_diag_no_intraday_shape", "lasth, herf")):
        L.append("- %s (%s): crit_panel %.5f (Δ к C4 %+.5f), QLIKE M16 всей истории %.5f (Δ %+.5f)" % (name, fs, crit[name], crit[name] - crit["C4"], a[name], a[name] - a["C4"]))
    p = wt[(wt["ticker"] == "PANEL") & wt["b_sur_mkt"].notna()]
    L.append("- средние β C4 по месяцам (стандартизованные z): " + "; ".join("%s %+.3f" % (f, p["b_" + f].mean()) for f in ZF.BLOCKS["z_new"] + ["ell", "abs_ell"]))
    return L


def weight_lines(fc: dict, best: str) -> list:
    b2 = pd.read_parquet(CC.OUT / "B_B2.parquet")
    wb = (1.0 - b2.groupby(b2["date"].dt.year)["w"].mean())
    wc = {c: CS.weight_by_year(fc[c]) for c in (best, "C4") if c in fc}
    L = ["| год | " + " | ".join("w на M, %s" % c for c in wc) + " | w на M, B2 (дрейф B) |", "|---|" + "---|" * (len(wc) + 1)]
    for y in wb.index:
        L.append("| %d | " % y + " | ".join("%.3f" % s.get(y, np.nan) for s in wc.values()) + " | %.3f |" % wb[y])
    return L


def build(o: dict) -> str:
    cp, crit, comp, best = o["cp"], o["crit"].set_index("series")["crit_panel"], o["comp"], o["best"]
    thr, b6 = cp["go_threshold_B"], crit["B6"]
    head = ("Лучшая C-a: %s, crit_panel %.5f против порога go %.5f (%s; stop/go этапа C — после C-b) и против B6 %.5f (Δ %+.5f; %s)" %
            (best, crit[best], thr, "ниже порога" if crit[best] <= thr else "выше порога на %.5f" % (crit[best] - thr), b6, crit[best] - b6, "лучше B6" if crit[best] < b6 else "хуже B6"))
    L = [head, "", "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; первая цифра — потолок; окна 2022+ в выборе не участвуют.", "", "## Отступления и толкования (до выводов)"] + ["- " + d for d in DEVIATIONS]
    L += ["", "## crit_panel (M16 × цели 2017–2021, `hp_stats_sel.crit_panel`; порог go = %.5f)" % thr, ""] + crit_lines(crit, thr)
    L += ["", "## Δ QLIKE лучшей C-a (%s), %% уровня K1 (M16; минус — лучше)" % best, ""] + delta_lines(comp, best)
    L += ["", "## Вклад блоков признаков (C1 → C4) и отдельно HAR-surprise / Intraday Shape", ""] + block_lines(crit, comp, pd.concat(o["wt"].values()))
    L += ["", "## Средний вес на M по годам (сравнение с дрейфом B2)", ""] + weight_lines(o["fc"], best)
    L += ["", "## Холм (основное семейство M16 × {2022_2025, all}, N=%d; N_wide=%d)" % (cp["N_declared_before_stats"], cp["N_wide"]), ""] + BR.holm_lines({**o, "best": best})
    L += ["", "## SBER (отдельно, в выбор не входит; 2271 дата; ориентир 0,2081 — потолок)", ""] + BR.sber_lines(o)
    L += ["", "## LOTO (панель на 16, поперечные Z по 16, δ = 0 у исключённой)", ""] + BR.loto_lines({"loto": o["loto"], "gche": pd.DataFrame(columns=["held_out", "window"])})[:-1]
    conv = pd.concat(o["wt"].values())
    conv = conv[conv["ticker"] == "PANEL"]
    L += ["", "Сходимость оптимизатора (L-BFGS-B, max|∇| < 1e-6): %.1f %% месячных оценок (%d из %d)." % (100 * conv["converged"].mean(), int(conv["converged"].sum()), len(conv))]
    L += ["Израсходовано конфигураций листа: 4 (C1–C4); после листа 9 из 60 (B — 5, C-a — 4); время счёта — `CHECKPOINT.json` (C_runtime_sec)."]
    return "\n".join(L) + "\n"
