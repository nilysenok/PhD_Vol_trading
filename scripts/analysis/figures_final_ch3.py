"""Рисунки 3.5-3.11 из сохранённых прогнозов (тест 2019; walk-forward 2017-2025)."""
from math import erfc, sqrt

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from figures_final_common import dec, plt, save

C = dict(hyb="#2E7D32", lgb="#43A047", xgb="#66BB6A", har="#1565C0", lstm="#E65100", gru="#EF6C00", grey="#757575")
M19 = [("Гибридная", "hybrid", "hyb"), ("LightGBM", "lightgbm", "lgb"), ("XGBoost", "xgboost", "xgb"),
       ("HAR-J", "har", "har"), ("LSTM", "lstm", "lstm"), ("GRU", "gru", "gru")]
OLD19_H1 = [0.276, 0.296, 0.277, 0.305, 0.372, 0.399]  # значения старого рисунка 3.5 (для сверки)
HY = "V1_Adaptive"
WFM = [("HAR-J", "HAR-J", "har"), ("XGBoost", "XGBoost", "xgb"), ("LightGBM", "LightGBM", "lgb"), ("Гибрид", HY, "hyb")]
PRINT_WF = {"HAR-J": (0.524, 0.796, 0.961), "XGBoost": (0.400, 0.956, 1.127),
            "LightGBM": (0.378, 0.821, 0.870), HY: (0.380, 0.762, 0.888)}
PRINT_DM = {HY: (13.55, 3.41, 7.03), "LightGBM": (12.11, -1.48, 4.61), "XGBoost": (11.95, -5.49, -4.63)}
HOR = (1, 5, 22)


def qloss(a, p):
    p = np.clip(p, 1e-10, None)
    r = a / p
    return r - np.log(r) - 1


def q2019(ctx, key, h):
    d = pd.read_parquet(ctx.t2019 / f"{key}_h{h}.parquet")
    d = d[d.rv_actual > 0]
    return float(qloss(d.rv_actual.values, d.rv_pred.values).mean()), len(d)


def wf_loss(ctx, key, h):
    """Строки walk-forward 2017-2025 (rv>0); h=5 - исправленный файл прогнозов."""
    d = pd.read_parquet(ctx.wf / f"walkforward_all_h{h}{'_corrected' if h == 5 else ''}.parquet")
    d = d[(d.rv_actual > 0) & d.year.between(2017, 2025)]
    return d.assign(q=qloss(d.rv_actual.values, d[f"pred_{key}"].values)).set_index(["date", "ticker"]).sort_index()


def wf_year_mean(l):
    """Печатное определение: простое среднее годовых QLIKE 2017-2025 (пул по тикерам)."""
    return l.q.groupby(l.index.get_level_values("date").year).mean()


def _bars(ctx, name, fig_id, labels, vals, cols, title, ylab_n, size=(8, 5)):
    fig, ax = plt.subplots(figsize=size)
    x = np.arange(len(labels))
    ax.bar(x, vals, 0.55, color=cols, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.004, dec(v, 3), ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("QLIKE"); ax.set_ylim(0, max(vals) * 1.18)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", visible=False)
    fig.text(0.5, 0.95, title, ha="center", va="top", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    save(ctx, fig, name, pd.DataFrame(dict(model=labels, qlike_2019_h1=vals)))


def fig_3_5_to_3_8(ctx):
    q = {(k, h): q2019(ctx, k, h) for _, k, _ in M19 for h in HOR}
    for (nm, k, _), ref in zip(M19, OLD19_H1):
        ctx.chk("3.5", f"QLIKE 2019 H=1, {nm}", q[(k, 1)][0], ref, "значение старого рисунка 3.5")
    fig = plt.figure(figsize=(11, 8)); ax = fig.add_subplot(111, projection="3d")
    for p in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        p.fill = False
    rows = []
    for j in reversed(range(len(M19))):
        nm, k, c = M19[j]
        for i, h in enumerate(HOR):
            v = q[(k, h)][0]
            ax.bar3d(i - 0.3, j - 0.175, 0, 0.6, 0.35, v, color=C[c], alpha=0.92, edgecolor="k", lw=0.4)
            ax.text(i, j, v + 0.015, dec(v, 3), ha="center", va="bottom", fontsize=8.5, fontweight="bold", zorder=100)
            rows.append(dict(model=nm, horizon=h, qlike=v, n=q[(k, h)][1]))
    ax.set_xticks(range(3)); ax.set_xticklabels([f"H={h}" for h in HOR]); ax.set_xlabel("Горизонт прогнозирования", labelpad=10)
    ax.set_yticks(range(len(M19))); ax.set_yticklabels([m[0] for m in M19], fontsize=9); ax.set_ylabel("Модель", labelpad=10)
    ax.set_zlim(0, max(r["qlike"] for r in rows) * 1.15); ax.view_init(elev=25, azim=-55)
    fig.subplots_adjust(left=0, right=0.88, bottom=0, top=0.88); fig.text(0.92, 0.45, "QLIKE", rotation=90, va="center")
    ns = [r["n"] for r in rows]
    fig.text(0.5, 0.92, "Рисунок 3.5 — Сравнение QLIKE моделей\nпо горизонтам прогнозирования", ha="center", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.02, f"Тестовый период 2019 (N={min(ns)}–{max(ns)})", ha="center", fontsize=9, color="0.4")
    save(ctx, fig, "fig_3_5_all_models_3d", pd.DataFrame(rows))
    v1 = lambda k: q[(k, 1)][0]  # noqa: E731
    _bars(ctx, "fig_3_6_boosting_vs_har", 3.6, ["XGBoost", "LightGBM", "HAR-J"], [v1("xgboost"), v1("lightgbm"), v1("har")],
          [C["xgb"], C["lgb"], C["har"]], "Рисунок 3.6 — QLIKE моделей бустинга и HAR-J\n(тест 2019, H=1)", 0)
    _bars(ctx, "fig_3_7_rnn_vs_boosting", 3.7, ["XGBoost", "LightGBM", "HAR-J", "LSTM", "GRU"],
          [v1(k) for k in ("xgboost", "lightgbm", "har", "lstm", "gru")], [C[c] for c in ("xgb", "lgb", "har", "lstm", "gru")],
          "Рисунок 3.7 — QLIKE рекуррентных сетей, бустинга и HAR-J\n(тест 2019, H=1)", 0, (10, 5.5))
    _bars(ctx, "fig_3_8_model_classes", 3.8, ["Бустинг", "HAR-J", "RNN"],
          [(v1("xgboost") + v1("lightgbm")) / 2, v1("har"), (v1("lstm") + v1("gru")) / 2], [C["xgb"], C["har"], C["lstm"]],
          "Рисунок 3.8 — Средний QLIKE по категориям моделей\n(тест 2019, H=1)", 0)
    for f in ("3.5", "3.6", "3.7", "3.8"):
        ctx.ok(f, "тест 2019 из data/predictions/test_2019; GJR-GARCH не показан: прогнозы garch_h*.parquet дают QLIKE 1,80/1,05/0,73, а не 0,501/0,567/0,601 старого рисунка")


def fig_3_9_3_11(ctx):
    res, rows = {}, []
    for nm, k, _ in WFM:
        for h in HOR:
            res[(k, h)] = wf_year_mean(wf_loss(ctx, k, h))
            v = float(res[(k, h)].mean()); rows.append(dict(model=nm, horizon=h, qlike=v, n_years=len(res[(k, h)])))
            ctx.chk("3.9", f"QLIKE {nm} H={h}", v, PRINT_WF[k][HOR.index(h)], "печать (среднее годовых 2017–2025)")
    fig, ax = plt.subplots(figsize=(12, 6)); x = np.arange(3); w = 0.18
    for i, (nm, k, c) in enumerate(WFM):
        vals = [r["qlike"] for r in rows if r["model"] == nm]
        b = ax.bar(x + (i - 1.5) * w, vals, w, color=C[c], label=nm, zorder=3)
        for bb, v in zip(b, vals):
            ax.text(bb.get_x() + w / 2, v + 0.012, dec(v, 3), ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"H={h}" for h in HOR]); ax.set_xlabel("Горизонт прогнозирования")
    ax.set_ylabel("QLIKE (среднее по годам)"); ax.set_ylim(0, max(r["qlike"] for r in rows) * 1.15); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=10); ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.96, "Рисунок 3.9 — Walk-forward QLIKE по горизонтам прогнозирования\n(среднее 2017–2025, 17 тикеров)", ha="center", va="top", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    save(ctx, fig, "fig_3_9_walkforward_qlike", pd.DataFrame(rows))
    ctx.ok("3.9", "walkforward_all_h{1,22}.parquet и walkforward_all_h5_corrected.parquet; гибрид = V1")
    # 3.10 DM (простой t, пул (дата, тикер) 2017-2025; t>0 - модель лучше HAR-J)
    t, p, drow = np.zeros((3, 3)), np.zeros((3, 3)), []
    pairs = [("HAR-J vs Гибрид", HY), ("HAR-J vs LightGBM", "LightGBM"), ("HAR-J vs XGBoost", "XGBoost")]
    for i, (pl, k) in enumerate(pairs):
        for j, h in enumerate(HOR):
            d = (wf_loss(ctx, "HAR-J", h).q - wf_loss(ctx, k, h).q).dropna().values
            t[i, j] = d.mean() / (d.std(ddof=1) / sqrt(len(d))); p[i, j] = erfc(abs(t[i, j]) / sqrt(2))
            ctx.chk("3.10", f"DM t, {pl}, H={h}", t[i, j], PRINT_DM[k][j], "печать DM (простой t, пул 2017–2025)")
            drow.append(dict(pair=pl, horizon=h, t=t[i, j], p=p[i, j], n=len(d)))
    cm = LinearSegmentedColormap.from_list("dm", [(0, "#C62828"), (0.35, "#FFCDD2"), (0.45, "#E0E0E0"), (0.55, "#E0E0E0"), (0.65, "#C8E6C9"), (1, "#2E7D32")])
    vm = max(16, np.ceil(np.abs(t).max()))
    fig, ax = plt.subplots(figsize=(9, 5)); im = ax.imshow(t, cmap=cm, vmin=-vm, vmax=vm, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"H={h}" for h in HOR]); ax.set_yticks(range(3)); ax.set_yticklabels([a for a, _ in pairs])
    for i in range(3):
        for j in range(3):
            s = "***" if p[i, j] < 0.001 else "**" if p[i, j] < 0.01 else "*" if p[i, j] < 0.05 else ""
            ax.text(j, i, dec(t[i, j]) + s, ha="center", va="center", fontsize=12, fontweight="bold", color="white" if abs(t[i, j]) > 8 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("t-статистика DM-теста")
    fig.text(0.45, 0.96, "Рисунок 3.10 — DM-тест: HAR-J против ML-моделей\n(walk-forward 2017–2025, простой t по паре (дата, тикер))", ha="center", va="top", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    save(ctx, fig, "fig_3_10_dm_test_heatmap", pd.DataFrame(drow))
    ctx.ok("3.10", "t по рядам; гибрид H=5 не даёт печатных 3,41 (см. сверку)")
    years = list(range(2017, 2026)); fig, ax = plt.subplots(figsize=(12, 6)); yrow = {"year": years}
    for h, (col, ls, mk) in zip(HOR, ((C["hyb"], "-", "o"), (C["har"], "--", "s"), ("#B71C1C", "-.", "D"))):
        v = res[(HY, h)].reindex(years).values; yrow[f"H{h}"] = v
        ax.plot(years, v, color=col, ls=ls, marker=mk, lw=2, markersize=7, label=f"H={h}")
        for yr, a in zip(years, v):
            ax.text(yr, a + 0.03, dec(a, 3), ha="center", fontsize=8, color=col)
    ax.set_xticks(years); ax.set_xlabel("Год (тестовый)"); ax.set_ylabel("QLIKE"); ax.legend(loc="upper left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.96, "Рисунок 3.11 — QLIKE гибридной модели по годам\n(walk-forward, 17 тикеров)", ha="center", va="top", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    save(ctx, fig, "fig_3_11_hybrid_by_year", pd.DataFrame(yrow))
    ctx.ok("3.11", "V1 (pred_V1_Adaptive), h=5 из исправленного файла")
