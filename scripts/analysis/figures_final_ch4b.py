"""Рисунки 4.9-4.16 из финальных дневных рядов МСП (см. figures_final.py)."""
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from figures_final_common import (BLUE1, BLUE2, BLUE3, BPY, GREY, RED, SN, comma_axes, dce, dec,
                                  econ_costs, imoex_ret, load_msp, load_strat, mdd_pct, plt, save,
                                  sharpe)

KS = (1, 2, 3)  # множители масштабирования экспозиции (сетка рисунка)
PRINT_SH = dict(A=1.32, B=1.47, C=2.51, D=2.54)


def scaled(d, k):
    """Ряд МСП при масштабе k: торговая часть x k, свободный капитал под ставку, плечо под ставку."""
    exp, tr = d.exposure.values, d.trading_ret.values
    rf = np.where(exp < 1, (d.total_ret.values - tr) / np.maximum(1 - exp, 1e-12), 0.0)
    fdr = np.maximum(1 - k * exp, 0) * rf
    lev = np.maximum(k * exp - 1, 0) * rf
    return k * tr, fdr, lev, k * tr + fdr - lev, float(np.mean(k * exp > 1))


def fig_4_9(ctx):
    r = pd.concat({s: load_strat(ctx, i, "A").excess_ret for i, s in enumerate(SN, 1)}, axis=1)
    c = r.corr().values
    rbar = (c.sum() - 6) / 30
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-0.5, vmax=1.0); ax.grid(False)
    ax.set_xticks(range(6)); ax.set_xticklabels(SN, fontsize=8); ax.set_yticks(range(6)); ax.set_yticklabels(SN, fontsize=8)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, dec(c[i, j]), ha="center", va="center", fontsize=7, color="white" if abs(c[i, j]) > 0.5 else "black")
    ax.set_title(f"Дневной (ρ̄ = {dec(rbar)})", fontsize=10); fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Корреляционная матрица стратегий (подход A)", fontsize=11)
    save(ctx, fig, "fig_4_9_correlation_matrices", r.corr().rename_axis("strategy").reset_index())
    ctx.ok("4.9", "только дневная панель")


def fig_4_10(ctx):
    names, tr, fd, rows = [], [], [], []
    for ap in "ABCD":
        d = load_msp(ctx, ap)
        t, f = d.trading_ret.mean() * BPY * 100, (d.total_ret - d.trading_ret).mean() * BPY * 100
        names.append(f"МСП-{ap}"); tr.append(t); fd.append(f)
        rows.append(dict(msp=ap, trading_pct=t, fdr_pct=f, total_pct=t + f, exposure_pct=d.exposure.mean() * 100))
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    x = np.arange(4)
    ax.bar(x, tr, 0.6, label="Торговая доходность", color=BLUE1)
    ax.bar(x, fd, 0.6, bottom=tr, label="ФДР (депозит ЦБ)", color=BLUE3)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("Годовая доходность, %")
    ax.set_title("Декомпозиция доходности МСП (дневной)"); ax.legend(fontsize=9); comma_axes(ax, axis="y")
    save(ctx, fig, "fig_4_10_return_decomposition", pd.DataFrame(rows))
    ctx.ok("4.10", "торговая часть и ФДР = total_ret - trading_ret из рядов")


def _cum_figure(ctx, name, lines, ncol, title_note):
    im = imoex_ret(ctx)
    idx = lines[0][1].index
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    rows = {}
    for lab, d, col, ls in lines:
        rows[lab] = (np.cumprod(1 + d.values) - 1) * 100
        ax.plot(idx, rows[lab], color=col, linewidth=1.2, linestyle=ls, label=lab)
    rows["IMOEX"] = (np.cumprod(1 + im.reindex(idx).fillna(0).values) - 1) * 100
    ax.plot(idx, rows["IMOEX"], color=RED, linewidth=1.2, linestyle="--", label="IMOEX")
    ax.axhline(0, color="black", linewidth=0.5); ax.set_ylabel("Кумулятивная доходность, %")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y")); ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=ncol, frameon=False)
    comma_axes(ax, axis="y")
    save(ctx, fig, name, pd.DataFrame(rows, index=idx).rename_axis("date").reset_index())


def fig_4_12(ctx):
    m = {ap: load_msp(ctx, ap) for ap in "ABCD"}
    mean = (m["B"].total_ret + m["C"].total_ret + m["D"].total_ret) / 3
    _cum_figure(ctx, "fig_4_12_msp_vs_imoex",
                [("МСП-A (1×)", m["A"].total_ret, BLUE3, "-"), ("МСП-MEAN (1×)", mean, BLUE2, "-"),
                 ("МСП-BEST = D (1×)", m["D"].total_ret, BLUE1, "-")], 4, "")
    for ap in "ABCD":
        ctx.chk("4.12/4.7", f"Шарп МСП-{ap}", sharpe(m[ap].excess_ret.values), PRINT_SH[ap], "печать Шарпа МСП")
    ctx.ok("4.12", "MEAN = среднее дневных доходностей B, C, D; BEST = D (максимальный Шарп 2,54)")


def fig_4_13(ctx):
    d, im = load_msp(ctx, "D"), imoex_ret(ctx)
    r_im = im.reindex(d.index).fillna(0).values
    dd = {}
    for lab, r in (("МСП-BEST (D)", d.total_ret.values), ("IMOEX", r_im)):
        cum = np.cumsum(r)
        dd[lab] = (cum - np.maximum.accumulate(cum)) * 100
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    ax.fill_between(d.index, 0, dd["МСП-BEST (D)"], alpha=0.5, color=BLUE1, label=f"МСП-BEST (MDD = {dec(mdd_pct(d.total_ret.values), 1)}%)")
    ax.fill_between(d.index, 0, dd["IMOEX"], alpha=0.3, color=RED, label=f"IMOEX (MDD = {dec(mdd_pct(r_im), 1)}%)")
    ax.set_ylabel("Просадка, %"); ax.set_title("Профиль просадок МСП-BEST и IMOEX")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y")); ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(fontsize=9); comma_axes(ax, axis="y")
    save(ctx, fig, "fig_4_13_drawdown_profile", pd.DataFrame(dd, index=d.index).rename_axis("date").reset_index())
    p = ctx.series.parent / "msp_results.csv"
    if p.exists():
        t = pd.read_csv(p)
        t = t[(t.level == "msp") & (t.comm_label == "inst")].set_index("approach").max_dd
        for ap in "ABCD":
            ctx.chk("4.13", f"MDD МСП-{ap}, %", mdd_pct(load_msp(ctx, ap).total_ret.values), t[ap], "msp/msp_results.csv (max_dd)")
    ctx.ok("4.13", "просадки из дневных рядов, MDD по накопленной сумме доходностей")


def fig_4_11_14(ctx):
    rows = []
    for ap in "AD":
        d = load_msp(ctx, ap)
        for k in KS:
            t, f, lv, tot, share = scaled(d, k)
            rows.append(dict(msp=ap, k=k, trading_pct=t.mean() * BPY * 100, fdr_pct=f.mean() * BPY * 100,
                             lev_cost_pct=-lv.mean() * BPY * 100, total_pct=tot.mean() * BPY * 100,
                             mdd_pct=mdd_pct(tot), share_days_lev=share))
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for ap, col, mk in (("A", BLUE3, "s"), ("D", BLUE1, "o")):
        s = df[df.msp == ap]
        ax.plot(s.mdd_pct, s.total_pct, mk + "-", color=col, linewidth=1.5, markersize=7, label="МСП-A" if ap == "A" else "МСП-BEST (D)")
        for _, r in s.iterrows():
            ax.annotate(f"{int(r.k)}×", (r.mdd_pct, r.total_pct), textcoords="offset points", xytext=(6, 4), fontsize=9, color=col)
    ax.set_xlabel("Максимальная просадка (MDD), %"); ax.set_ylabel("Годовая доходность, %")
    ax.legend(fontsize=9, loc="upper left"); comma_axes(ax)
    save(ctx, fig, "fig_4_11_efficient_frontier", df)
    s = df[df.msp == "D"]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x = np.arange(len(KS))
    ax.bar(x, s.trading_pct, 0.6, label="Торговая доходность", color=BLUE1)
    ax.bar(x, s.fdr_pct, 0.6, bottom=s.trading_pct, label="ФДР", color=BLUE3)
    ax.bar(x, s.lev_cost_pct, 0.6, label="Стоимость плеча", color=RED)
    ax.axhline(0, color="black", linewidth=0.5); ax.set_xticks(x); ax.set_xticklabels([f"{k}×" for k in KS])
    ax.set_xlabel("k"); ax.set_ylabel("Годовая доходность, %"); ax.set_title("Дневной, МСП-BEST (D)")
    ax.legend(fontsize=8, loc="upper left"); comma_axes(ax, axis="y")
    save(ctx, fig, "fig_4_14_scaling_decomposition", df)
    ctx.ok("4.11", "сетка k=1,2,3; плечо тарифицируется по ставке ЦБ без надбавки (экспозиция в рядах < 1: плечо не возникает)")
    ctx.ok("4.14", "только дневная панель; МСП-BEST = D")


def fig_4_15_16(ctx):
    c1, c2 = econ_costs()
    g = np.arange(1, 16)
    a = load_msp(ctx, "A").total_ret.values
    pair = {ap: load_msp(ctx, ap).total_ret.values for ap in "CD"}
    rows = {"gamma": g}
    fig, ax1 = plt.subplots(figsize=(6.3, 3.5)); ax2 = ax1.twinx()
    for ap, col, mk in (("C", BLUE1, "o"), ("D", BLUE2, "s")):
        dl = np.array([dce(pair[ap], a, int(x)) for x in g]); rows[f"dCE_{ap}_pct"] = dl
        ax1.plot(g, dl, mk + "-", color=col, linewidth=1.5, markersize=4, label=f"ΔCE, {ap} против A")
        for cost, ls, nm in ((c1, "--", "1-й год"), (c2, ":", "2-й год и далее")):
            k = np.where(dl > 0, cost / np.where(dl > 0, dl / 100, 1), np.nan); rows[f"kmin_{ap}_{nm}_mln"] = k
            ax2.plot(g, k, ls, color=col, alpha=0.4, linewidth=1, label=f"Мин. капитал, {ap}, {nm}")
    ax1.set_xlabel("Коэффициент неприятия риска γ"); ax1.set_ylabel("ΔCE, % годовых")
    ax2.set_ylabel("Мин. капитал, млн руб."); ax2.grid(False)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False); comma_axes(ax1, ax2, axis="y")
    save(ctx, fig, "fig_4_15_delta_gamma", pd.DataFrame(rows))
    ctx.chk("4.15", "ΔCE(C, γ=10), % годовых", rows["dCE_C_pct"][9], 2.07, "печать ΔCE (fix_all_bootstrap_ce)")
    ctx.ok("4.15", f"ΔCE из рядов (геометрическое CE); затраты {c1}/{c2} млн из stage8_economic.py; economic/*.csv нули/inf (расчёт на пустых данных)")
    labs, er, rk, rs, tot, rr = ["C против A (γ=5)", "D против A (γ=5)"], [], [], [], [], []
    for ap in "CD":
        er.append((pair[ap].mean() - a.mean()) * BPY * 100)
        rk.append(-(5 / 2) * (pair[ap].var(ddof=0) - a.var(ddof=0)) * BPY * 100)
        tot.append(dce(pair[ap], a, 5)); rs.append(tot[-1] - er[-1] - rk[-1])
    fig, ax = plt.subplots(figsize=(6.3, 2.6)); y = np.arange(2)
    ax.barh(y, er, 0.4, label="Компонента доходности ΔE(r)", color=BLUE1)
    ax.barh(y, rk, 0.4, left=er, label="Компонента риска −(γ/2)Δσ²", color=BLUE3)
    ax.barh(y, rs, 0.4, left=np.array(er) + np.array(rk), label="Прочее (старшие моменты)", color=GREY)
    for i in range(2):
        ax.text(max(er[i], tot[i]) + 0.1, i, f"ΔCE = {dec(tot[i])}", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.set_xlabel("Δ, процентные пункты годовых")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False); comma_axes(ax, axis="x")
    save(ctx, fig, "fig_4_16_delta_decomposition", pd.DataFrame(dict(pair=labs, dE_pct=er, risk_pct=rk, other_pct=rs, dCE_pct=tot)))
    ctx.ok("4.16", "ΔE(r), −(γ/2)Δσ², остаток и ΔCE из рядов C, D против A")
