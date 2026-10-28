"""Рисунки 3.5-3.11 из тестовых прогнозов финала третьего круга (тест главы 3 выполнен 26.10 08:57).

Вход: <MAIN>/data/regen/test_locked/final3/final_h{1,5,22}_test.parquet — только чтение; тест не
перезапускается. Финал: h=1 D22_h1, h=5 D12_h5, h=22 HAR-J с поправкой (FINAL_DESIGN.md, поле 10).
Определения — как scripts/analysis/final_test.py: фильтр rv_actual > 1e-12, QLIKE = mean(y/p - ln(y/p) - 1);
«2019» — год таблиц 3.4/3.6; «среднее» — простое среднее годовых QLIKE 2017–2026 (табл. 3.7);
DM — основной способ (дневной ряд средней по тикерам разности потерь, NW лаг h-1; табл. 3.9).
Сверка — с docs/tables/table_3_*_test.csv.
"""
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from figures_final_common import TABLES, TEST_SRC, dec, plt, save
from ftp_stats import daily, nw_var, qloss

C = dict(fin="#2E7D32", lgb="#43A047", xgb="#66BB6A", har="#1565C0", harc="#64B5F6")
HOR = (1, 5, 22)
FIN = "Финальная"
COLS = {FIN: "rv_pred", "LightGBM": "lgb_raw", "XGBoost": "xgb_raw", "HAR-J": "har_raw", "HAR-J с поправкой": "har_corr"}
CK = {FIN: "fin", "LightGBM": "lgb", "XGBoost": "xgb", "HAR-J": "har", "HAR-J с поправкой": "harc"}
TAB_NAME = {FIN: "Гибридная"}          # имя финала в таблицах final_test.py
FIN_H = {1: "D22_h1", 5: "D12_h5", 22: "HAR-J с поправкой"}
REFS = (("HAR-J", "HAR-J как в диссертации"), ("HAR-J с поправкой", "HAR-J с поправкой"))


def load_test(h):
    d = pd.read_parquet(TEST_SRC / f"final_h{h}_test.parquet")
    return d[d.rv_actual > 1e-12].sort_values(["date", "ticker"]).reset_index(drop=True)


def q(d, col):
    return float(qloss(d.rv_actual.values, d[col].values).mean())


def tab(name):
    return pd.read_csv(TABLES / f"table_3_{name}_test.csv")


def _bars(ctx, name, labels, vals, title, size=(8, 5)):
    fig, ax = plt.subplots(figsize=size)
    x = np.arange(len(labels))
    ax.bar(x, vals, 0.55, color=[C[CK.get(l, "har")] for l in labels], zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.004, dec(v, 3), ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("QLIKE"); ax.set_ylim(0, max(vals) * 1.18)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", visible=False)
    fig.text(0.5, 0.95, title, ha="center", va="top", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    save(ctx, fig, name, pd.DataFrame(dict(model=labels, qlike_2019_h1=vals)))


def fig_3_5_to_3_8(ctx, D):
    q19 = {(m, h): q(D[h][D[h].year == 2019], c) for m, c in COLS.items() for h in HOR}
    t6, t4 = tab("6").set_index("Модель"), tab("4").set_index("Модель")
    for m in COLS:
        for h in HOR:
            ctx.chk("3.5", f"QLIKE 2019 H={h}, {m}", q19[(m, h)], t6.loc[TAB_NAME.get(m, m), f"H = {h}"], "table_3_6_test.csv", 1e-4)
    models = list(COLS)
    fig = plt.figure(figsize=(11, 8)); ax = fig.add_subplot(111, projection="3d")
    for p in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        p.fill = False
    rows = []
    for j in reversed(range(len(models))):
        m = models[j]
        for i, h in enumerate(HOR):
            v = q19[(m, h)]
            ax.bar3d(i - 0.3, j - 0.175, 0, 0.6, 0.35, v, color=C[CK[m]], alpha=0.92, edgecolor="k", lw=0.4)
            ax.text(i, j, v + 0.012, dec(v, 3), ha="center", va="bottom", fontsize=8.5, fontweight="bold", zorder=100)
            rows.append(dict(model=m, horizon=h, qlike=v, n=int((D[h].year == 2019).sum())))
    ax.set_xticks(range(3)); ax.set_xticklabels([f"H={h}" for h in HOR]); ax.set_xlabel("Горизонт прогнозирования", labelpad=10)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=9); ax.set_ylabel("Модель", labelpad=10)
    ax.set_zlim(0, max(r["qlike"] for r in rows) * 1.15); ax.view_init(elev=25, azim=-55)
    fig.subplots_adjust(left=0, right=0.88, bottom=0, top=0.88); fig.text(0.92, 0.45, "QLIKE", rotation=90, va="center")
    fig.text(0.5, 0.92, "Рисунок 3.5 — Сравнение QLIKE моделей\nпо горизонтам прогнозирования (тест, 2019 г.)", ha="center", fontsize=12, fontweight="bold")
    save(ctx, fig, "fig_3_5_all_models_3d", pd.DataFrame(rows))
    v1 = lambda m: q19[(m, 1)]  # noqa: E731
    for m in ("XGBoost", "LightGBM", "HAR-J"):
        ctx.chk("3.6", f"QLIKE 2019 H=1, {m}", v1(m), t4.loc[m, "H = 1"], "table_3_4_test.csv", 1e-4)
    _bars(ctx, "fig_3_6_boosting_vs_har", ["XGBoost", "LightGBM", "HAR-J"], [v1(m) for m in ("XGBoost", "LightGBM", "HAR-J")],
          "Рисунок 3.6 — QLIKE моделей бустинга и HAR-J\n(тест, 2019 г., H=1)")
    _bars(ctx, "fig_3_8_model_classes", ["Бустинг", "HAR-J"], [(v1("XGBoost") + v1("LightGBM")) / 2, v1("HAR-J")],
          "Рисунок 3.8 — Средний QLIKE по категориям моделей\n(тест, 2019 г., H=1)", (6, 5))
    note = "тест final3, 2019 г.; бустинг без поправки (как табл. 3.4/3.6); LSTM/GRU и GJR-GARCH в тест не входят"
    for f in ("3.5", "3.6"):
        ctx.ok(f, note)
    ctx.ok("3.8", note + " — класс RNN убран")
    ctx.skip("3.7", "LSTM/GRU не входят в финальный тест (табл. 3.5 в final_test.py не строится); без RNN рисунок совпадает с 3.6")


def fig_3_9(ctx, D):
    t7 = tab("7").set_index("Модель")
    models = ["HAR-J", "HAR-J с поправкой", "XGBoost", "LightGBM", FIN]
    rows = []
    for m in models:
        for h in HOR:
            yq = D[h].groupby("year").apply(lambda g: q(g, COLS[m]))
            v = float(yq.mean()); rows.append(dict(model=m, horizon=h, qlike=v, n_years=len(yq)))
            ctx.chk("3.9", f"QLIKE {m} H={h}", v, t7.loc[TAB_NAME.get(m, m), f"H = {h}"], "table_3_7_test.csv (среднее годовых 2017–2026)", 1e-4)
    fig, ax = plt.subplots(figsize=(12, 6)); x = np.arange(3); w = 0.16
    for i, m in enumerate(models):
        vals = [r["qlike"] for r in rows if r["model"] == m]
        b = ax.bar(x + (i - 2) * w, vals, w, color=C[CK[m]], label=m, zorder=3)
        for bb, v in zip(b, vals):
            ax.text(bb.get_x() + w / 2, v + 0.008, dec(v, 3), ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"H={h}\n(финал: {FIN_H[h]})" for h in HOR]); ax.set_xlabel("Горизонт прогнозирования")
    ax.set_ylabel("QLIKE (среднее по годам)"); ax.set_ylim(0, max(r["qlike"] for r in rows) * 1.15); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.96, "Рисунок 3.9 — QLIKE на тесте по горизонтам прогнозирования\n(среднее годовых 2017–2026, 17 тикеров)", ha="center", va="top", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    save(ctx, fig, "fig_3_9_walkforward_qlike", pd.DataFrame(rows))
    ctx.ok("3.9", "тест final3, 2017–2026 (2026 — январь–3 февраля); бустинг без поправки")


def dm_main(d, ref, h):
    """Основной DM final_test.py: дневной ряд средней по тикерам (L_эталон − L_финал), NW лаг h−1."""
    l = qloss(d.rv_actual.values, d[ref].values) - qloss(d.rv_actual.values, d.rv_pred.values)
    s = daily(d.date.values, l)
    v = nw_var(s, h - 1)
    t = s.mean() / np.sqrt(v) if v > 1e-300 else np.nan
    return float(t), float(stats.norm.sf(t)) if np.isfinite(t) else np.nan, len(s)


def fig_3_10(ctx, D):
    t9 = tab("9")
    t9 = t9[t9.способ.str.startswith("основной")].set_index(["H", "эталон"])
    t, p, rows = np.full((2, 3), np.nan), np.full((2, 3), np.nan), []
    for i, (m, en) in enumerate(REFS):
        for j, h in enumerate(HOR):
            t[i, j], p[i, j], n = dm_main(D[h], COLS[m], h)
            ref = t9.loc[(h, en), "t"]
            if np.isfinite(t[i, j]) and pd.notna(ref):
                ctx.chk("3.10", f"DM t, {m} vs финал, H={h}", t[i, j], ref, "table_3_9_test.csv (основной способ, t округлено до 0,01)", 0.006)
            rows.append(dict(pair=f"{m} vs финал", horizon=h, t=t[i, j], p_one=p[i, j], n_days=n))
    cm = LinearSegmentedColormap.from_list("dm", [(0, "#C62828"), (0.35, "#FFCDD2"), (0.45, "#E0E0E0"), (0.55, "#E0E0E0"), (0.65, "#C8E6C9"), (1, "#2E7D32")])
    vm = max(4, np.ceil(np.nanmax(np.abs(t))))
    fig, ax = plt.subplots(figsize=(9, 4)); im = ax.imshow(np.nan_to_num(t), cmap=cm, vmin=-vm, vmax=vm, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"H={h}\n({FIN_H[h]})" for h in HOR]); ax.set_yticks(range(2)); ax.set_yticklabels([f"{m} vs финал" for m, _ in REFS])
    for i in range(2):
        for j in range(3):
            if not np.isfinite(t[i, j]):
                ax.text(j, i, "—\n(финал = эталон)", ha="center", va="center", fontsize=9)
                continue
            s = "***" if p[i, j] < 0.001 else "**" if p[i, j] < 0.01 else "*" if p[i, j] < 0.05 else ""
            ax.text(j, i, dec(t[i, j]) + s, ha="center", va="center", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("t-статистика DM (t > 0 — финал лучше)")
    fig.text(0.45, 0.98, "Рисунок 3.10 — DM-тест: эталоны HAR-J против финальной модели\n(тест 2017–2026, дневной ряд средней по тикерам, NW лаг H−1)", ha="center", va="top", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.88])
    save(ctx, fig, "fig_3_10_dm_test_heatmap", pd.DataFrame(rows))
    ctx.ok("3.10", "основной способ табл. 3.9; звёзды — одностороннее p (H1: финал лучше); h=22: финал = HAR-J с поправкой")


def fig_3_11(ctx, D):
    tg = tab("final_po_godam")
    tg = tg[(tg.Модель == "Гибридная") & (tg.Год != "все годы (пул)")].assign(Год=lambda x: x.Год.astype(int)).set_index(["h", "Год"]).QLIKE
    years = sorted(D[1].year.unique()); yrow = {"year": years}
    fig, ax = plt.subplots(figsize=(12, 6))
    for h, (col, ls, mk) in zip(HOR, ((C["fin"], "-", "o"), (C["har"], "--", "s"), ("#B71C1C", "-.", "D"))):
        v = [q(D[h][D[h].year == y], "rv_pred") for y in years]; yrow[f"H{h}"] = v
        for y, a in zip(years, v):
            ctx.chk("3.11", f"QLIKE финал H={h}, {y}", a, tg.loc[(h, y)], "table_3_final_po_godam_test.csv", 1e-6)
        ax.plot(years, v, color=col, ls=ls, marker=mk, lw=2, markersize=7, label=f"H={h} ({FIN_H[h]})")
        for yr, a in zip(years, v):
            ax.text(yr, a + 0.04, dec(a, 3), ha="center", fontsize=8, color=col)
    ax.set_xticks(years); ax.set_xticklabels([str(y) if y < 2026 else "2026*" for y in years])
    ax.set_xlabel("Год (тестовый)"); ax.set_ylabel("QLIKE"); ax.legend(loc="upper left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.96, "Рисунок 3.11 — QLIKE финальной модели по годам\n(тест, 17 тикеров; * 2026 — январь–3 февраля)", ha="center", va="top", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    save(ctx, fig, "fig_3_11_hybrid_by_year", pd.DataFrame(yrow))
    ctx.ok("3.11", "финал по h (D22_h1 / D12_h5 / HAR-J с поправкой), тест 2017–2026")


def run(ctx):
    ctx.period = "тест главы 3 (2017–2026)"
    ctx.footer = "Тест главы 3 (прогон 26.10.2026 08:57), data/regen/test_locked/final3; финал: H=1 D22_h1, H=5 D12_h5, H=22 HAR-J с поправкой"
    D = {h: load_test(h) for h in HOR}
    fig_3_5_to_3_8(ctx, D); fig_3_9(ctx, D); fig_3_10(ctx, D); fig_3_11(ctx, D)
    return D
