"""Рисунки 4.0-4.8 из финальных дневных рядов МСП (см. figures_final.py)."""
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from figures_final_common import (BLUE1, BLUE2, BLUE3, GREY, RED, SN, boot_deltas, comma_axes,
                                  comm_rates, dec, imoex_ret, key_rate_daily, load_msp,
                                  load_strat, plt, save, sharpe)

AP = "ABCD"
PRINT_SH = dict(A=1.32, B=1.47, C=2.51, D=2.54)
PRINT_DSH = dict(B=0.15, C=1.19, D=1.22)
COMM_RU = dict(inst="Инст.", std="Розн.")


def _xs(ctx, ap):
    return load_msp(ctx, ap).excess_ret.values


def fig_4_0(ctx):
    kr = key_rate_daily(ctx)
    kr = kr["2022-01-01":]
    y = kr["2022-01-01":"2025-12-31"]
    mean = float(y.mean())  # среднее по календарным дням 2022-2025 (ступени)
    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    ax.step(kr.index, kr.values, where="post", color=BLUE1, linewidth=1.5)
    ax.axhline(mean, color=GREY, linestyle="--", linewidth=1)
    ax.text(kr.index[-1], mean + 0.4, f"Среднее 2022–2025 = {dec(mean, 1)}%", fontsize=9,
            color=GREY, ha="right", va="bottom")
    ax.set_xlabel("Год"); ax.set_ylabel("Ключевая ставка, %")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y")); ax.xaxis.set_major_locator(mdates.YearLocator())
    comma_axes(ax, axis="y")
    ctx.chk("4.0", "средняя ставка 2022-2025, %", mean, 14.3, "подпись «среднее 14,3%»; ряд ЦБ (ступени)", 0.05)
    save(ctx, fig, "fig_4_0_key_rate", kr.rename("key_rate_pct").rename_axis("date").reset_index())
    ctx.ok("4.0", f"ряд до {kr.index[-1].date()}, среднее {mean:.2f}")


def fig_4_1(ctx):
    rates, rows = comm_rates(ctx), []
    data = np.array([[sharpe(load_strat(ctx, i, "A", c).excess_ret.values) for i in range(1, 7)]
                     for c in ("inst", "std")])
    lab = [f"{COMM_RU[c]} ({dec(rates.get(c, np.nan) * 100, 2)}%)" for c in ("inst", "std")]
    fig, ax = plt.subplots(figsize=(6.3, 2.0))
    im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0, vmax=data.max() * 1.1)
    ax.set_xticks(range(6)); ax.set_xticklabels(SN); ax.set_yticks(range(2)); ax.set_yticklabels(lab)
    for i in range(2):
        for j in range(6):
            ax.text(j, i, dec(data[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if data[i, j] > data.max() * 0.6 else "black")
    ax.set_title("Шарп стратегий (дневной таймфрейм, подход A)"); ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    for c, r in zip(("inst", "std"), data):
        rows += [dict(comm=c, strategy=s, sharpe_excess=v) for s, v in zip(SN, r)]
    save(ctx, fig, "fig_4_1_heatmap_daily_A", pd.DataFrame(rows))
    ctx.ok("4.1", "строки: инст./розн.; строки «без комиссий» нет в рядах")


def fig_4_2(ctx):
    ctx.skip("4.2", "нет часовых дневных рядов доходности стратегий (только hourly_wf_A_results.csv, без B/C/D)")


def fig_4_3(ctx):
    cats = {"Контртренд\n(S1, S2)": (1, 2), "Тренд\n(S3, S4)": (3, 4), "Диапазон\n(S5, S6)": (5, 6)}
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    rows, x, w = [], np.arange(3), 0.35
    for k, (c, col) in enumerate(zip(("inst", "std"), (BLUE1, BLUE3))):
        v = [np.mean([sharpe(load_strat(ctx, i, "A", c).excess_ret.values) for i in ids]) for ids in cats.values()]
        ax.bar(x + k * w, v, w, label=f"{COMM_RU[c]}", color=col)
        rows += [dict(comm=c, category=n.replace("\n", " "), sharpe_mean=a) for n, a in zip(cats, v)]
    ax.set_xticks(x + w / 2); ax.set_xticklabels(list(cats), fontsize=9)
    ax.set_ylabel("Шарп"); ax.set_title("Дневной"); ax.legend(fontsize=9, frameon=False); comma_axes(ax, axis="y")
    fig.tight_layout()
    save(ctx, fig, "fig_4_3_category_comparison", pd.DataFrame(rows))
    ctx.ok("4.3", "только дневная панель; часовой нет ряда")


def fig_4_4(ctx):
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    x, w, rows = np.arange(6), 0.2, []
    for k, (ap, col) in enumerate(zip(AP, (BLUE3, BLUE2, BLUE1, GREY))):
        v = [sharpe(load_strat(ctx, i, ap).excess_ret.values) for i in range(1, 7)]
        ax.bar(x + k * w, v, w, label=f"Подход {ap}", color=col)
        rows += [dict(approach=ap, strategy=s, sharpe_excess=a) for s, a in zip(SN, v)]
    ax.set_xticks(x + 1.5 * w); ax.set_xticklabels(SN); ax.set_ylabel("Шарп")
    ax.set_title("Шарп стратегий по подходам (дневной)"); ax.legend(fontsize=9); comma_axes(ax, axis="y")
    save(ctx, fig, "fig_4_4_ew_portfolios_abcd", pd.DataFrame(rows))
    ctx.ok("4.4", "по рядам S{i}_{A..D}_inst")


def fig_4_5(ctx):
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    x, w, rows = np.arange(6), 0.35, []
    for k, (ap, col) in enumerate((("A", BLUE3), ("C", BLUE1))):
        v = [sharpe(load_strat(ctx, i, ap).excess_ret.values) for i in range(1, 7)]
        ax.bar(x + (k - 0.5) * w, v, w, label=f"Подход {ap}", color=col)
        rows += [dict(approach=ap, strategy=s, sharpe_excess=a) for s, a in zip(SN, v)]
    ax.set_xticks(x); ax.set_xticklabels(SN); ax.set_ylabel("Шарп"); ax.legend(fontsize=8)
    ax.set_title("Подход A и подход C (масштабирование по σ), дневной", fontsize=10); comma_axes(ax, axis="y")
    save(ctx, fig, "fig_4_5_a_vs_best_bcd", pd.DataFrame(rows))
    ctx.ok("4.5", "только дневная панель")


def fig_4_6(ctx):
    ctx.skip("4.6", "нет рядов МСП по методам взвешивания на финальных рядах: extended_daily_returns даёт EW-A 1,285 вместо 1,32 (другой прогон)")


def fig_4_7(ctx):
    im = imoex_ret(ctx)
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    rows = {}
    for ap, col, ls in (("A", BLUE3, "-"), ("C", BLUE1, "-")):
        d = load_msp(ctx, ap)
        rows[f"msp_{ap}_cum_pct"] = (np.cumprod(1 + d.total_ret.values) - 1) * 100
        ax.plot(d.index, rows[f"msp_{ap}_cum_pct"], color=col, linewidth=1.2, label=f"МСП-{ap}")
    idx = d.index
    rows["imoex_cum_pct"] = (np.cumprod(1 + im.reindex(idx).fillna(0).values) - 1) * 100
    ax.plot(idx, rows["imoex_cum_pct"], color=RED, linewidth=1.2, linestyle="--", label="IMOEX")
    ax.axhline(0, color="black", linewidth=0.5); ax.set_ylabel("Кумулятивная доходность, %")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y")); ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    comma_axes(ax, axis="y")
    save(ctx, fig, "fig_4_7_cumulative_returns", pd.DataFrame(rows, index=idx).rename_axis("date").reset_index())
    ctx.ok("4.7", "ряды скользящей калибровки (те же, что дают 1,32/2,51)")


def fig_4_8(ctx):
    a = _xs(ctx, "A")
    reps, labs = {}, []
    for ap in "BCD":
        t = _xs(ctx, ap)
        d = boot_deltas(sharpe, t, a)
        obs = sharpe(t) - sharpe(a)
        reps[f"{ap}_vs_A"] = d
        labs.append(f"{ap} против A\n{'+' if obs >= 0 else '−'}{dec(abs(obs))}")
        ctx.chk("4.8", f"ΔШарп {ap} против A", obs, PRINT_DSH[ap], "печать бутстрапа (fix_all_bootstrap_ce)")
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    parts = ax.violinplot([reps[k] for k in reps], positions=range(3), showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(BLUE2); pc.set_alpha(0.7)
    ax.axhline(0, color=RED, linestyle="--", linewidth=1, label="H₀: Δ = 0")
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("Δ Шарп"); ax.set_title("Бутстреп-распределения Δ Шарп (дневной, 10 000 реплик)", fontsize=10)
    ax.legend(fontsize=9); comma_axes(ax, axis="y")
    save(ctx, fig, "fig_4_8_bootstrap_distributions", pd.DataFrame(reps))
    ctx.ok("4.8", "настоящий бутстреп: seed 42, 10 000 реплик")
