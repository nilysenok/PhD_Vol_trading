"""V4-C, лист C-b, G21: раздел C-b в C_REPORT.md. Первая строка — «go»/«stop» этапа C по заранее записанному правилу (лучшая из C1–C10 ≤ go_threshold_B); все числа берутся из таблиц стадии."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_b_report as BR, v4_c_cache as CC, v4_c_report as CA, v4_c_stats as CS, v4_cb_run as R

DEVIATIONS = [
    "C7 — собственная реализация (без LightGBM): гессиан — гауссово-ньютоновская часть e·u² (полный гессиан может быть отрицательным), лист = −ΣG/(ΣH + L), L = 250·s²/16 (250 пар-эквивалентов; константа не подбиралась), пороги — по 32 "
    "квантильным корзинам, нефинитный признак в прогнозе = медиана окна, переобучение раз в год: весь 2017 — приор (w = 0,5), закрытых пар до 02.01.2017 нет; колонка `month` для C7 содержит год, сходимость L-BFGS к C7 не относится (converged = True, nit = 200).",
    "C5: f — кубический B-сплайн по ℓ, 2 внутренних узла по квантилям 1/3 и 2/3 ℓ окна обучения, без свободного члена (5 столбцов); ℓ вне окна прижат к его границам (константное продолжение); гребень на коэффициентах — как на β в C-a (на стандартизованных столбцах).",
    "C6: z_perf_EWMA = признак `ewqd` (EWMA полураспад 20 пар разности QLIKE(H) − QLIKE(M)); γ ≥ 0 — граница L-BFGS-B на коэффициенте при стандартизованном `ewqd` (знак сохраняется); β₁, β₂ при ℓ и |ℓ| — без ограничений.",
    "C8: один старт (a = 0, b = −1, веса режимов 0,5): при равных стартах и b = 0 градиент по b равен нулю (седло симметрии calm ↔ crisis), начальное b = −1 фиксирует смысл «выше рыночный уровень — меньше доля calm»; локальный оптимум не исключён; "
    "гребень и отклонения по бумагам — как в C-a (κ = λ = 250·s²).",
    "C9: softmax по {H, K2, K3}, штраф на δ_ik и гребень на α_k, β_k (избыточность softmax снимается гребнем); колонка `w` — суммарный вес K2 + K3 (сопоставима с весом на M в C1–C8), веса экспертов — `wH`, `wK2`, `wK3` в C9.parquet.",
    "C10: штраф на η_i с тем же κ·s², что на δ_i; тождество C10 при η = 0 и C5/линейная часть → C1 проверяется через контур C1x (`v4_cb_run.SPECS`), не конфигурация и в реестр не входит.",
    "Прежние модули (`v4_c_*`, `v4_z_*`, `v4_b_*`, `v4_trials`, `run_v4` кроме добавленной стадии `Cb`) не менялись; C1–C4 не пересчитывались (кэш), побитное совпадение со снимком до листа — в EVIDENCE G20.",
]


def year_lines(yt: pd.DataFrame, series: list) -> list:
    L = ["| ряд | " + " | ".join("%d: Δ к K1 / к D2" % y for y in range(2017, 2022)) + " |", "|---|" + "---|" * 5]
    for s in series:
        t = yt[(yt["series"] == s) & (yt["role"] == "выбор")].set_index("year")
        L.append("| %s | " % s + " | ".join("%+.4f / %+.4f" % (t.loc[y, "d_ann_k1"], t.loc[y, "d_ann_d2"]) for y in range(2017, 2022)) + " |")
    return L


def weight_lines(fc: dict, series: list) -> list:
    b2 = pd.read_parquet(CC.OUT / "B_B2.parquet")
    wb = 1.0 - b2.groupby(b2["date"].dt.year)["w"].mean()
    wc = {c: CS.weight_by_year(fc[c]) for c in series}
    L = ["| год | " + " | ".join("w на M, %s" % c for c in wc) + " | B2 |", "|---|" + "---|" * (len(wc) + 1)]
    return L + ["| %d | " % y + " | ".join("%.3f" % s.get(y, np.nan) for s in wc.values()) + " | %.3f |" % wb[y] for y in wb.index]


def first_line(crit: pd.Series, comp: pd.DataFrame, best: str, thr: float) -> str:
    ctrl = {k: crit[k] for k in ("k1", "k2", "k3", "d2")}
    bc = min(ctrl, key=ctrl.get)
    d = {w: BR.delta_pct(comp, best, w) for w in ("all", "2022_2025")}
    return ("%s — этап C: лучшая из C1–C10 %s, crit_panel %.5f против порога go %.5f (лучший контроль %s %.5f; отрыв %.5f = %.2f %% уровня K1 %.5f; нужно ≥ 0,0005 ниже контроля); "
            "Δ QLIKE к K1/K2/K3/D2 (%% уровня K1, M16): вся история %.2f/%.2f/%.2f/%.2f, 2022–2025 %.2f/%.2f/%.2f/%.2f" % (
                "go" if crit[best] <= thr else "stop", best, crit[best], thr, bc, ctrl[bc], ctrl[bc] - crit[best], 100 * (ctrl[bc] - crit[best]) / ctrl["k1"], ctrl["k1"],
                *[d["all"][x] for x in ("k1", "k2", "k3", "d2")], *[d["2022_2025"][x] for x in ("k1", "k2", "k3", "d2")]))


def build(r: dict) -> str:
    cp, crit, comp, best = r["cp"], r["crit"].set_index("series")["crit_panel"], r["comp"], r["best"]
    thr, yt = cp["go_threshold_B"], r["years"]
    L = [first_line(crit, comp, best, thr), "", "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; первая цифра — потолок; окна 2022+ в выборе не участвуют.", "",
         "## Отступления и толкования C-b (до выводов)"] + ["- " + d for d in DEVIATIONS]
    L += ["", "## crit_panel (M16 × цели 2017–2021, `hp_stats_sel.crit_panel`; порог go = %.5f)" % thr, ""] + CA.crit_lines(crit, thr)
    L += ["", "## Δ QLIKE лучшей (%s), %% уровня K1 (M16; минус — лучше)" % best, ""] + CA.delta_lines(comp, best)
    L += ["", "Δ QLIKE к K1, % уровня K1 (вся история / 2022–2025): " + "; ".join("%s %.2f / %.2f" % (c, BR.delta_pct(comp, c, "all")["k1"], BR.delta_pct(comp, c, "2022_2025")["k1"]) for c in CS_ALL(r)) + "."]
    L += ["", "## По годам цели 2017–2021: Δ среднего QLIKE (M16) к K1 / к D2 (плюс — мета-слой хуже; 2017 — год без прошлых OOF-пар: приор, у C7 весь год)", ""] + year_lines(yt, ["B6"] + CS_ALL(r))
    L += ["", "crit_part по годам (pen_Y/4, 2018–2021; сумма = crit_panel): " + "; ".join("%s %s" % (s, " ".join("%.5f" % v for v in yt[(yt["series"] == s) & yt["crit_part"].notna()]["crit_part"])) for s in ("k1", "d2", best))]
    L += ["", "## Средний вес на M по годам (C9 — K2 + K3; сравнение с дрейфом B2)", ""] + weight_lines(r["fc"], sorted({"C1", best}, key=lambda x: int(x[1:])))
    L += ["", "## Холм (лучшая; основное семейство M16 × {2022_2025, all} × C1–C10, N=%d; N_wide=%d)" % (cp["N_declared_before_stats"], cp["N_wide"]), ""] + BR.holm_lines({**r, "best": best})
    L += ["", "## SBER (отдельно, в выбор не входит; 2271 дата; ориентир 0,2081 — потолок)", ""] + BR.sber_lines(r)
    L += ["", "## LOTO (панель на 16, поперечные Z по 16, параметры исключённой — панельные)", ""] + BR.loto_lines({"loto": r["loto"], "gche": pd.DataFrame(columns=["held_out", "window"])})[:-1]
    conv = pd.concat([r["wt_b"][c] for c in R.CFGS if c != "C7"])
    conv = conv[conv["ticker"] == "PANEL"]
    L += ["", "Сходимость оптимизатора (C5, C6, C8, C9, C10; max|∇| < 1e-6): %.1f %% оценок (%d из %d)." % (100 * conv["converged"].mean(), int(conv["converged"].sum()), len(conv))]
    L += ["Израсходовано конфигураций листа: 6 (C5–C10); после листа 15 из 60 (B — 5, C-a — 4, C-b — 6); время счёта — `CHECKPOINT.json` (C_b_runtime_sec)."]
    L += ["", "## Лист C-a (прежний отчёт, без изменений)", "", CA.build(r["ca"]).rstrip()]
    return "\n".join(L) + "\n"


def CS_ALL(r: dict) -> list:
    return [c for c in r["comp"]["series"].unique() if c.startswith("C") and "diag" not in c]
