"""V7-DX: числа для выводов, правила вывода по гипотезам (ОБЪЯВЛЕНЫ ДО СЧЁТА, 09.11.2026) и текст `DX_REPORT.md` целиком из таблиц `DX_*.csv` (никаких чисел вручную).
Правила (пороги — константы ниже):
 (а) выразительность: доля ячеек (kind, год, seed), где энкодер лучше всех трёх наивных на отложенных 10 %: ≥ 0,8 — «отброшена»; ≤ 0,2 — «подтверждена»; иначе «не различимо».
 (б) сигнал: частный R² задним числом (лучший штраф, среднее по seeds, 2018–2021): < 0,005 — «подтверждена» (если в выборке снижение ≥ 0,5 % — с пометкой «в выборке есть, не переносится»); ≥ 0,02 — «отброшена»; иначе «не различимо».
 (в) регуляризация: прирост снижения в выборке при λ_g = 0,01 против 0,1 (среднее по конфигурациям 1, 4 и датам): < 0,5 п.п. и при λ_g = 0 против 0,1 < 1 п.п. — «отброшена»; ≥ 2 п.п. — «не различимо» (в выборке прирост неотличим от переобучения, вне выборки не проверяется);
     иначе «не различимо». «Подтверждена» невозможна без проверки вне выборки — она запрещена листом.
 (г) дублирование: R² вне выборки PIT-регрессии ln(RV/K2) на z (2018–2021, среднее) ≤ 0,005 и R² компонент z на четырёх контролях ≥ 0,3 — «подтверждена»; ≥ 0,01 — «отброшена»; иначе «не различимо»."""
from __future__ import annotations

import numpy as np
import pandas as pd

A_HI, A_LO = 0.8, 0.2
B_NO, B_YES, B_INS = 0.005, 0.02, 0.5
C_NO1, C_NO0, C_YES = 0.5, 1.0, 2.0
D_NO, D_YES, D_CTRL = 0.005, 0.01, 0.3
DEV = (2018, 2021)


def numbers(t: dict) -> dict:
    """-> словарь чисел из таблиц (для выводов, отчёта и CHECKPOINT)."""
    e, s, x, sw, pc, pit, gap = t["ENC"], t["SIG"], t["SIG_EXPOST"], t["LAMSWEEP"], t["PCORR"], t["PIT"], t["GAP"]
    dev = lambda d, c: d[(d[c] >= DEV[0]) & (d[c] <= DEV[1])]
    sd = s[s["dev_window"]]
    g = sw.groupby("lam_g")["red_in_pct"].mean()
    best = x.groupby("alpha")["partial_r2"].mean()
    zr = dev(pc[pc["var"].str.startswith("z")], "year")
    gr = gap[gap["cfg"].isin(["4", "6"])]
    n = dict(enc_share_beats=float(e["enc_beats_all_naive"].mean()), enc_mse_ratio_best_naive=float((e["mse_enc"] / e[["mse_mean", "mse_ewma", "mse_profile"]].min(axis=1)).median()),
             enc_share_top4_median=float(e["share_top4"].median()), enc_eff_rank_median=float(e["eff_rank"].median()), enc_dead_axes_max=int(e["n_dead_axes"].max()),
             stab_aligned_mean=float(t["ENC_STAB"]["aligned_mean"].mean()), stab_unaligned_mean=float(t["ENC_STAB"]["unaligned_mean"].mean()),
             sig_in={c: float(sd[sd["cfg"] == c]["red_in_pct"].mean()) for c in ("1", "4", "6")}, sig_out={c: float(sd[sd["cfg"] == c]["red_out_pct"].mean()) for c in ("1", "4", "6")},
             const_in={c: float(sd[sd["cfg"] == c]["const_red_in_pct"].mean()) for c in ("1", "4", "6")}, const_out={c: float(sd[sd["cfg"] == c]["const_red_out_pct"].mean()) for c in ("1", "4", "6")},
             expost_partial_r2_best=float(best.max()), expost_best_alpha=float(best.idxmax()), expost_r2_z_best=float(x.groupby("alpha")["r2_z"].mean().max()),
             sweep_red={"%g" % k: float(v) for k, v in g.items()}, lam_share_lowest=float(t["LAM"].query("year == 'all' and cfg in ['1','4','6'] and lam_g == 0.1")["share"].sum() / 3),
             g_share_gt09_seed_max=float(t["G"]["share_gt09_seed"].max()), g_mean_abs_median=float(t["G"]["mean_abs_g"].median()),
             pit_r2_dev=float(dev(pit, "year_t_plus")["r2_oos"].mean()), pit_r2_all=float(pit["r2_oos"].mean()), z_r2_on_controls=float(zr["r2_on_controls"].mean()),
             gap_share_2022={c: float(gr[(gr["cfg"] == c) & (gr["year"] == 2022)]["share_of_total"].iloc[0]) for c in ("4", "6")},
             gap_total={c: float(gr[gr["cfg"] == c]["total_gap"].iloc[0]) for c in ("4", "6")}, gap_without_2022={c: float(gr[gr["cfg"] == c]["gap_without_2022"].iloc[0]) for c in ("4", "6")})
    return n


def verdicts(n: dict) -> dict:
    a = n["enc_share_beats"]
    va = "отброшена" if a >= A_HI else "подтверждена" if a <= A_LO else "не различимо"
    r, ins = n["expost_partial_r2_best"], max(n["sig_in"].values())
    vb = "подтверждена" if r < B_NO else "отброшена" if r >= B_YES else "не различимо"
    if vb == "подтверждена" and ins >= B_INS:
        vb += " (к H снижение есть, но частный R² z сверх Z ≈ 0: выгода к H объясняется контролями Z и уровнем H — таблицы (б) и (г))"
    sw = n["sweep_red"]
    d1, d0 = sw["0.01"] - sw["0.1"], sw["0"] - sw["0.1"]
    vc = "отброшена" if (d1 < C_NO1 and d0 < C_NO0) else "не различимо"
    p, c = n["pit_r2_dev"], n["z_r2_on_controls"]
    vd = "подтверждена" if (p <= D_NO and c >= D_CTRL) else "отброшена" if p >= D_YES else "не различимо"
    return dict(a=va, b=vb, c=vc, d=vd)


NAMES = dict(a="(а) недостаточная выразительность энкодера", b="(б) отсутствие предсказуемого сигнала", c="(в) избыточная регуляризация", d="(г) дублирование информации HAR-J и бустингов")


def head_line(v: dict) -> str:
    ok = [NAMES[k] for k, x in v.items() if x.startswith("подтверждена")]
    no = [NAMES[k] for k, x in v.items() if x.startswith("отброшена")]
    un = [NAMES[k] for k, x in v.items() if x.startswith("не различимо")]
    return "Причина (по правилам, объявленным до счёта): подтверждена — %s; отброшена — %s; не различимо — %s." % ("; ".join(ok) or "нет", "; ".join(no) or "нет", "; ".join(un) or "нет")


def md(df: pd.DataFrame, cols=None, fmt="%.4f") -> str:
    d = df[list(cols)] if cols else df
    f = lambda x: (fmt % x) if isinstance(x, (float, np.floating)) else str(x)
    return "\n".join(["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)] + ["| " + " | ".join(f(x) for x in r) + " |" for r in d.itertuples(index=False)])


def render(t: dict, status: str) -> str:
    n = numbers(t)
    v = verdicts(n)
    L = ["# DX — причины отсутствия новой информации (V7, SBER, h = 1)", "", "Статус: " + status, "", head_line(v), "",
         "Числа — из таблиц `DX_*.csv` (генератор `v7_dx_rep.render`). Сравнения «в выборке» — на обучающих парах; «вне выборки» — сохранённые прогнозы; верхняя оценка — ex post, не для отбора. Окно выводов 2018–2021; 2022+ — только справочно.", ""]
    L += ["## (а) Энкодер: %s" % v["a"], "",
          "- Доля ячеек (kind, год, seed), где энкодер лучше всех трёх наивных на отложенных 10 %%: %.3f (правило: ≥ %.1f — отброшена, ≤ %.1f — подтверждена)." % (n["enc_share_beats"], A_HI, A_LO),
          "- Медиана MSE энкодера / лучшего наивного: %.3f; медиана доли дисперсии 4 компонент из 16: %.3f; эффективный ранг (медиана): %.2f; мёртвых осей (макс.): %d." % (n["enc_mse_ratio_best_naive"], n["enc_share_top4_median"], n["enc_eff_rank_median"], n["enc_dead_axes_max"]),
          "- Средняя |corr| компонент z между seeds одного года: с выравниванием %.3f, без выравнивания %.3f." % (n["stab_aligned_mean"], n["stab_unaligned_mean"]), "",
          md(t["ENC"].groupby(["kind", "year"], as_index=False)[["mse_enc", "mse_mean", "mse_ewma", "mse_profile", "mse_const", "share_top4", "eff_rank"]].mean()), ""]
    L += ["## (б) Сигнал: %s" % v["b"], "",
          "- Снижение QLIKE к H, %% (2018–2021, средние по годам Y → Y+1): в выборке %s; вне выборки %s." % (_d(n["sig_in"]), _d(n["sig_out"])),
          "- Контроль без модели «константа» (F = H·среднее rv/H закрытых пар; добавлен после первого просмотра, правила вывода не менялись), %%: в выборке %s; вне выборки %s." % (_d(n["const_in"]), _d(n["const_out"])),
          "- Частный R² задним числом (z энкодера 2021, гребень α = %g): %.4f; R²(z) %.4f (ex post, не для отбора)." % (n["expost_best_alpha"], n["expost_partial_r2_best"], n["expost_r2_z_best"]), "",
          md(t["SIG"], ["cfg", "year", "n_in", "red_in_pct", "const_red_in_pct", "year_out", "n_out", "red_out_pct", "const_red_out_pct", "dev_window"]), "", md(t["SIG_EXPOST"].groupby("alpha", as_index=False)[["r2_Z", "r2_z", "r2_zZ", "partial_r2"]].mean()), ""]
    L += ["## (в) Регуляризация: %s" % v["c"], "",
          "- Снижение QLIKE в выборке по λ_g (среднее по конфигурациям 1, 4 и датам 2019-01, 2020-01, 2021-01), %%: %s. Решётка плана воспроизводит выбор побитно во всех %d ячейках: %s." % (_d(n["sweep_red"]), len(t["LAMSWEEP"]) // 4, bool(t["LAMSWEEP"]["lattice_repro"].all())),
          "- Доля выборов минимального λ_g = 0,1 (1, 4, 6): %.3f; доля |G| > 0,9 по seeds (макс. по конфигурациям и годам): %.4f; медиана средних |G|: %.4f." % (n["lam_share_lowest"], n["g_share_gt09_seed_max"], n["g_mean_abs_median"]), "",
          md(t["LAMSWEEP"], ["cfg", "date", "lam_g", "red_in_pct", "val_q_mean", "mean_epochs"]), "", md(t["LAM"].query("year == 'all'"), ["cfg", "lam_g", "lam_lb", "n", "share", "cp_match"]), "", md(t["G"], ["cfg", "year", "mean_abs_g", "q90", "q99", "share_gt09_avg", "share_gt09_seed"]), ""]
    L += ["## (г) Дублирование: %s" % v["d"], "",
          "- R² вне выборки PIT-регрессии ln(RV/K2) на z: 2018–2021 %.4f; 2018–2025 %.4f. R² компонент z на четырёх контролях (2018–2021): %.3f." % (n["pit_r2_dev"], n["pit_r2_all"], n["z_r2_on_controls"]),
          "- Разрыв к K2 (QLIKE, плюс = хуже K2), 2018–2025: конфигурация 4 %.4f, 6 %.4f; без 2022: %.4f и %.4f; доля 2022 в разрыве: %.2f и %.2f." % (n["gap_total"]["4"], n["gap_total"]["6"], n["gap_without_2022"]["4"], n["gap_without_2022"]["6"], n["gap_share_2022"]["4"], n["gap_share_2022"]["6"]), "",
          md(t["PIT"].groupby("year_t_plus", as_index=False)["r2_oos"].mean()), "", md(t["PCORR"].query("year >= 2018 and year <= 2021").groupby("var", as_index=False).mean(numeric_only=True).drop(columns=["year", "seed"])), "", md(t["GAP"], ["cfg", "year", "n", "delta_vs_K2", "contribution", "share_of_total"]), ""]
    return "\n".join(L) + "\n"


def _d(d: dict) -> str:
    return ", ".join("%s: %.2f" % (k, x) for k, x in d.items())
