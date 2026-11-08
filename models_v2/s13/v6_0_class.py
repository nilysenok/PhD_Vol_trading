"""V6-0, G2: класс сопоставимости ряда SBER — «PIT-сопоставим» или «потолок: <причина>» — с доказательством (строка кода или отчёта).
Определение (объявлено листом 0). PIT-сопоставим: (а) каждый параметр на дате d оценён по парам с t⁺ ≤ d и (б) сам ряд (набор признаков, окно, c, λ, конфигурация) НЕ выбран по метрике на датах SBER, на которых он сравнивается.
Потолок: хотя бы одно из условий не доказано. Ряды SBER-программ Ж-210/211/212/214 причинны по весам и параметрам, но отобраны по `crit_wf` на тех же датах SBER (перебор сеток, финалист семейства) — для «go» это потолки.
Исключение — замороженные контроли программы (K1, K2, K3, M, K8, D2) и их точные дубли; K2/K3 несут оговорку (дизайн выбран на SBER 2017–2021)."""
from __future__ import annotations

PIT, CEIL = "PIT-сопоставим", "потолок"
E_HF = "docs/otchety/hybrid_final_v1/GATES_HF.md:55 «финалист семейства — min crit_wf (фиксированные c/λ — потолок)»; LEVELS.csv ceiling_note «потолок: выбор по crit_wf на всей истории»"
E_HF14 = "docs/otchety/hybrid_final_v1/GATES_HF.md:14 «наборы выбраны по EX1 … потолок (отбор групп видел историю)»"
E_HFWF = "docs/otchety/hybrid_final_v1/GATES_HF.md:12-13, 30: c, λ — по закрытым прогнозам (target_date ≤ d), тест «закрыто+1» models_v2/s7/hf_rules.py::wf_select"
E_HS = "docs/otchety/hybrid_struct_v1/FORECAST_COMPARISONS.csv, шапка «финалисты направлений — потолок (выбор по crit_wf на всей истории)»"
E_V2 = "docs/otchety/harj_hybrid_v2/GATES_V2.md:153 У30 «финалист ступени — минимум crit_wf среди кандидатов»; statistical_validation/FINALISTS.csv"
E_M16 = "models_v2/s10/hp_stats_sel.py:49-51 crit_window: группа M16 (SBER в выбор не входит); models_v2/s12/v5_fin_tabs.py:2 «SBER в выбор (M16) не входит»"
E_K23 = "docs/otchety/hybrid_struct_v1/INVENTORY.md: XJ = x2_best_j (K2), LF = C0_inh_F4 (K3); дизайн Ж-204/206/210 выбран на SBER 2017–2021 (sber_h1_deep_v1)"
CTRL = {"K1": "HAR-J(б) PIT: расширяющееся окно OLS, models_v2/har.py (пары t+1 ≤ i)", "K2": E_K23, "K3": E_K23, "M": "M = √(K2·K3): оговорка K2/K3", "K8": "K8 = √(H·M): оговорка K2/K3", "D2": "панельный LGBM-D2, выбор по M16 (" + E_M16 + ")"}


def classify(series: str, program: str, family: str, equal_to: str = "") -> tuple:
    """-> (класс, причина, доказательство)."""
    c = series.split(":", 1)[-1]
    if equal_to:
        return PIT, "точный дубль замороженного контроля %s (max|Δ| = 0)" % equal_to, "сверка v6_0_prior.dup_of"
    if program == "J210" and family == "comp":
        return CEIL, "базовая модель Ж-204/206 (XF/B2X/B2L): дизайн выбран на SBER", E_K23
    if program == "J210" and c.startswith("ctl_"):
        return CEIL, "контроль Ж-210 на XJ/LF/XF (дизайн моделей выбран на SBER; среднее без параметров)", E_K23
    if program == "J210":
        return CEIL, "правило весов Ж-210 причинно, но семейство перебрано, финалист по crit_wf на SBER", E_HF
    if family == "J211bound" and ("_wf_" in c or c.startswith("R23_wf")):
        return CEIL, "c, λ выбираются по закрытым парам (причинно), но база/набор Ж-204 и сетка видели SBER; в «go» не берётся", E_HFWF
    if family == "J211bound":
        return CEIL, "фиксированные c/λ из сетки, выбор по crit_wf на всей истории", E_HF
    if family == "J211feat":
        return CEIL, "набор признаков выбран по EX1/crit_wf на истории SBER", E_HF14
    if family == "J211w":
        return CEIL, "EMA/ограничение скорости веса: перебор, финалист по crit_wf на SBER", E_HF
    if program == "J212":
        return CEIL, "направление Ж-212 перебрано на SBER, финалисты по crit_wf", E_HS
    if program == "V2":
        return CEIL, "ступени Ж-214 перебраны на SBER, финалист по crit_wf", E_V2
    if program == "DEEP":
        return CEIL, "модель программы sber_h1_deep_v1: дизайн выбран на SBER", E_K23
    if program in ("PAN", "V4", "V5"):
        return PIT, "параметры по закрытым парам (тесты PIT программы); дизайн выбран по M16, SBER — диагностика", E_M16
    raise ValueError("нет правила класса: %s %s %s" % (series, program, family))
