#!/usr/bin/env python3
"""Generate all 4 figures for section 3.2 (final version)."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Font & style ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# ── Color palette ─────────────────────────────────────────────
C = {
    'HAR-J':      '#1f4e79',
    'GARCH-GJR':  '#8b1a1a',
    'XGBoost':    '#2e7d32',
    'LightGBM':   '#e65100',
    'LSTM':       '#757575',
    'GRU':        '#9e9e9e',
    'Гибридная':  '#4a148c',
}


# ══════════════════════════════════════════════════════════════
# ГРАФИК 1 — fig_3_5_test2019_h1.png
# ══════════════════════════════════════════════════════════════
def plot_fig1():
    models = ['Гибридная', 'XGBoost', 'LightGBM', 'HAR-J', 'LSTM', 'GRU', 'GARCH-GJR']
    values = [0.276, 0.277, 0.290, 0.305, 0.372, 0.399, 0.501]
    colors = [C[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, values, color=colors, width=0.60,
                  edgecolor='black', linewidth=0.4)

    ax.set_ylabel('QLIKE', fontsize=12)
    ax.set_title('Сравнение QLIKE моделей: тестовый период 2019 (H=1)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(0, 0.60)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle=':', color='grey')
    ax.tick_params(axis='x', length=0, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    plt.tight_layout()
    fp = f'{OUT_DIR}/fig_3_5_test2019_h1.png'
    fig.savefig(fp, facecolor='white')
    print(f'Saved {fp}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# ГРАФИК 2 — fig_3_6_test2019_3d.png
# ══════════════════════════════════════════════════════════════
def plot_fig2():
    models = ['Гибридная', 'LightGBM', 'XGBoost', 'HAR-J', 'LSTM', 'GRU', 'GARCH-GJR']
    horizons = ['H=1', 'H=5', 'H=22']
    qlike = np.array([
        [0.276, 0.373, 0.441],   # Гибридная
        [0.290, 0.382, 0.455],   # LightGBM
        [0.277, 0.396, 0.470],   # XGBoost
        [0.305, 0.424, 0.467],   # HAR-J
        [0.372, 0.435, 0.485],   # LSTM
        [0.399, 0.411, 0.503],   # GRU
        [0.501, 0.567, 0.601],   # GARCH-GJR
    ])
    colors = [C[m] for m in models]

    n_models = len(models)
    n_horizons = len(horizons)

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection='3d')

    dx, dy = 0.45, 0.45

    # Draw back-to-front for correct occlusion
    for i in range(n_models - 1, -1, -1):
        for j in range(n_horizons - 1, -1, -1):
            ax.bar3d(j - dx/2, i - dy/2, 0, dx, dy, qlike[i, j],
                     color=colors[i], alpha=0.90,
                     edgecolor='black', linewidth=0.3, shade=True)

    ax.set_xticks(range(n_horizons))
    ax.set_xticklabels(horizons, fontsize=10)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(models, fontsize=9)
    ax.set_zlabel('QLIKE', fontsize=12, labelpad=10)
    ax.set_xlabel('Горизонт прогнозирования', fontsize=12, labelpad=10)
    ax.set_ylabel('Модель', fontsize=12, labelpad=15)

    ax.set_zlim(0, 0.7)
    ax.set_ylim(-0.5, n_models - 0.5)
    ax.set_xlim(-0.5, n_horizons - 0.5)
    ax.view_init(elev=25, azim=-50)

    ax.set_title('Сравнение QLIKE всех моделей по горизонтам (тест 2019)',
                 fontsize=13, fontweight='bold', pad=20)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgrey')
    ax.yaxis.pane.set_edgecolor('lightgrey')
    ax.zaxis.pane.set_edgecolor('lightgrey')
    ax.grid(True, alpha=0.3, linestyle=':', color='grey')

    fp = f'{OUT_DIR}/fig_3_6_test2019_3d.png'
    fig.savefig(fp, facecolor='white')
    print(f'Saved {fp}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# ГРАФИК 3 — fig_3_7_categories_3d.png
# ══════════════════════════════════════════════════════════════
def plot_fig3():
    categories = ['Гибридная', 'Бустинг', 'RNN', 'Классическая']
    horizons = ['H=1', 'H=5', 'H=22']
    qlike = np.array([
        [0.276, 0.373, 0.441],   # Гибридная
        [0.284, 0.389, 0.463],   # Бустинг
        [0.386, 0.423, 0.494],   # RNN
        [0.403, 0.496, 0.534],   # Классическая
    ])
    colors = ['#4a148c', '#2e7d32', '#757575', '#1f4e79']

    n_cats = len(categories)
    n_horizons = len(horizons)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    dx, dy = 0.55, 0.55

    for i in range(n_cats - 1, -1, -1):
        for j in range(n_horizons - 1, -1, -1):
            ax.bar3d(j - dx/2, i - dy/2, 0, dx, dy, qlike[i, j],
                     color=colors[i], alpha=0.90,
                     edgecolor='black', linewidth=0.3, shade=True)

    ax.set_xticks(range(n_horizons))
    ax.set_xticklabels(horizons, fontsize=10)
    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(categories, fontsize=10)
    ax.set_zlabel('QLIKE', fontsize=12, labelpad=10)
    ax.set_xlabel('Горизонт прогнозирования', fontsize=12, labelpad=10)
    ax.set_ylabel('Категория модели', fontsize=12, labelpad=10)

    ax.set_zlim(0, 0.7)
    ax.set_ylim(-0.5, n_cats - 0.5)
    ax.set_xlim(-0.5, n_horizons - 0.5)
    ax.view_init(elev=25, azim=-50)

    ax.set_title('Сравнение QLIKE по категориям моделей (тест 2019)',
                 fontsize=13, fontweight='bold', pad=20)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgrey')
    ax.yaxis.pane.set_edgecolor('lightgrey')
    ax.zaxis.pane.set_edgecolor('lightgrey')
    ax.grid(True, alpha=0.3, linestyle=':', color='grey')

    fp = f'{OUT_DIR}/fig_3_7_categories_3d.png'
    fig.savefig(fp, facecolor='white')
    print(f'Saved {fp}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# ГРАФИК 4 — fig_3_8_hybrid_by_year.png
# ══════════════════════════════════════════════════════════════
def plot_fig4():
    years = list(range(2017, 2027))
    h1  = [0.2562, 0.3500, 0.2779, 0.3534, 0.2300, 0.6799, 0.3849, 0.3123, 0.5718, 0.4867]
    h5  = [4.3780, 0.4426, 0.3931, 0.4977, 0.3045, 3.5742, 0.4448, 0.5098, 0.6963, 0.5281]
    h22 = [0.3393, 0.4492, 0.4484, 1.0073, 0.4455, 3.4562, 0.5196, 0.6014, 0.7218, 0.6502]

    # Use symlog scale: linear near 0, log for large values
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(years, h1, '-o', color='#1f4e79', linewidth=2.0, markersize=5,
            label='Дневной (H=1)', zorder=3)
    ax.plot(years, h5, '-s', color='#2e7d32', linewidth=2.0, markersize=5,
            label='Недельный (H=5)', zorder=3)
    ax.plot(years, h22, '-^', color='#e65100', linewidth=2.0, markersize=5,
            label='Месячный (H=22)', zorder=3)

    # MOEX halt line
    ax.axvline(x=2022, color='grey', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)
    ax.text(2022.1, ax.get_ylim()[1] * 0.92, 'Остановка\nторгов MOEX',
            fontsize=8, color='grey', va='top')

    ax.set_yscale('symlog', linthresh=1.0, linscale=0.5)
    ax.set_yticks([0.2, 0.3, 0.5, 1.0, 2.0, 4.0])
    ax.set_yticklabels(['0.2', '0.3', '0.5', '1.0', '2.0', '4.0'])

    ax.set_xlabel('Год', fontsize=12)
    ax.set_ylabel('QLIKE', fontsize=12)
    ax.set_title('Динамика QLIKE гибридной модели по годам (walk-forward)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(years)
    ax.set_xticklabels(years, fontsize=10)
    ax.tick_params(axis='y', labelsize=10)

    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle=':', color='grey')

    plt.tight_layout()
    fp = f'{OUT_DIR}/fig_3_8_hybrid_by_year.png'
    fig.savefig(fp, facecolor='white')
    print(f'Saved {fp}')
    plt.close(fig)


# ── Run all ───────────────────────────────────────────────────
if __name__ == '__main__':
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    print('\nDone! All 4 figures saved.')
