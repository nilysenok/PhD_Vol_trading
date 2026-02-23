#!/usr/bin/env python3
"""Generate 4 figures for Section 3.1 of the dissertation.

fig_3_1a_h1.png   — vertical bar chart, H=1
fig_3_1b_h5.png   — vertical bar chart, H=5
fig_3_1c_h22.png  — vertical bar chart, H=22
fig_3_1d_3d_surface.png — 3D bar chart, all horizons
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# --- Font setup ---
try:
    fm.findfont('Times New Roman', fallback_to_default=False)
    font_name = 'Times New Roman'
except Exception:
    font_name = 'serif'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': [font_name, 'DejaVu Serif'],
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
})

OUT_DIR = 'results/final/figures'

# --- Data (test 2019, N=4284) ---
DATA = {
    'H=1':  {'HAR-J': 0.3052, 'GARCH-GJR': 1.8017},
    'H=5':  {'HAR-J': 0.4243, 'GARCH-GJR': 1.0514},
    'H=22': {'HAR-J': 0.4669, 'GARCH-GJR': 0.7320},
}

COLOR_HAR   = '#1a3a5c'   # dark blue
COLOR_GARCH = '#8b1a1a'   # dark red
COLORS = [COLOR_HAR, COLOR_GARCH]
MODELS = ['HAR-J', 'GARCH-GJR']


def make_bar_chart(horizon_key, subtitle, filename, ylim_top):
    """Create a single vertical bar chart for one horizon."""
    har_val = DATA[horizon_key]['HAR-J']
    garch_val = DATA[horizon_key]['GARCH-GJR']
    values = [har_val, garch_val]

    fig, ax = plt.subplots(figsize=(5, 4.5))

    bars = ax.bar(MODELS, values, color=COLORS, width=0.55,
                  edgecolor='black', linewidth=0.5)

    # Value labels above bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ylim_top * 0.015,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('QLIKE', fontsize=12)
    ax.set_title(f'Сравнение QLIKE моделей:\nгоризонт {subtitle}',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_ylim(0, ylim_top)

    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')
    ax.tick_params(axis='x', length=0)

    fig.savefig(f'{OUT_DIR}/{filename}')
    print(f'Saved: {OUT_DIR}/{filename}')
    plt.close(fig)


# =====================================================================
# FIGURES 1-3: vertical bar charts per horizon
# =====================================================================
make_bar_chart('H=1',  'H=1 (1 день)',  'fig_3_1a_h1.png',  ylim_top=2.1)
make_bar_chart('H=5',  'H=5 (5 дней)',  'fig_3_1b_h5.png',  ylim_top=1.25)
make_bar_chart('H=22', 'H=22 (22 дня)', 'fig_3_1c_h22.png', ylim_top=0.9)


# =====================================================================
# FIGURE 4: 3D bar chart — all horizons
# =====================================================================
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Positions
horizons = ['H=1', 'H=5', 'H=22']
x_pos = np.array([0, 1, 2])    # horizons
y_har = np.zeros(3)             # HAR-J at y=0
y_garch = np.ones(3)            # GARCH-GJR at y=1

har_vals   = [DATA[h]['HAR-J'] for h in horizons]
garch_vals = [DATA[h]['GARCH-GJR'] for h in horizons]

bar_width = 0.35
bar_depth = 0.4

# HAR-J bars
ax.bar3d(x_pos - bar_width / 2, y_har, np.zeros(3),
         bar_width, bar_depth, har_vals,
         color=COLOR_HAR, alpha=0.85, edgecolor='black', linewidth=0.3,
         label='HAR-J')

# GARCH-GJR bars
ax.bar3d(x_pos - bar_width / 2, y_garch, np.zeros(3),
         bar_width, bar_depth, garch_vals,
         color=COLOR_GARCH, alpha=0.85, edgecolor='black', linewidth=0.3,
         label='GARCH-GJR')


# Axes
ax.set_xticks(x_pos)
ax.set_xticklabels(horizons, fontsize=10)
ax.set_yticks([0.2, 1.2])
ax.set_yticklabels(['HAR-J', 'GARCH-GJR'], fontsize=10)
ax.set_zlabel('QLIKE', fontsize=12, labelpad=8)
ax.set_xlabel('Горизонт', fontsize=12, labelpad=8)
ax.set_ylabel('Модель', fontsize=12, labelpad=10)

ax.set_title('Сравнение QLIKE моделей HAR-J и GARCH-GJR\nпо горизонтам прогнозирования',
             fontsize=13, fontweight='bold', pad=15)

# View angle — all 6 bars visible
ax.view_init(elev=25, azim=-50)
ax.set_zlim(0, 2.0)

# Light grid on back walls
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('lightgray')
ax.yaxis.pane.set_edgecolor('lightgray')
ax.zaxis.pane.set_edgecolor('lightgray')
ax.zaxis._axinfo['grid'].update({'color': 'gray', 'linestyle': '--', 'linewidth': 0.4})
ax.xaxis._axinfo['grid'].update({'color': 'gray', 'linestyle': '--', 'linewidth': 0.4})
ax.yaxis._axinfo['grid'].update({'color': 'gray', 'linestyle': '--', 'linewidth': 0.4})

fig.savefig(f'{OUT_DIR}/fig_3_1d_3d_surface.png')
print(f'Saved: {OUT_DIR}/fig_3_1d_3d_surface.png')
plt.close(fig)

print('\nDone! All 4 figures saved.')


# =====================================================================
# VARIANT 2: 3-model comparison (HAR-J, GARCH-daily, GARCH-10min)
# =====================================================================
# GARCH-10min QLIKE values will be read from prediction files;
# fall back to placeholders if files not yet generated.

import os
_pred_dir = os.path.join('data', 'predictions', 'test_2019')


def _load_garch10min_qlike():
    """Load GARCH-10min QLIKE (Patton variant) from prediction files."""
    vals = {}
    for h in [1, 5, 22]:
        p = os.path.join(_pred_dir, f'garch_10min_h{h}.parquet')
        if not os.path.exists(p):
            return None
        import pandas as _pd
        df = _pd.read_parquet(p)
        actual = df['rv_actual'].values
        pred = np.maximum(df['rv_pred'].values, 1e-10)
        ratio = actual / pred
        # Patton (2011) QLIKE: always non-negative, consistent with DATA dict
        vals[f'H={h}'] = float(np.mean(ratio - np.log(ratio) - 1))
    return vals


_g10_qlike = _load_garch10min_qlike()
if _g10_qlike is None:
    print('\nVariant 2: garch_10min prediction files not found — skipping.')
else:
    OUT_DIR_V2 = 'results/final/figures/section_3_1/вариант_2'
    os.makedirs(OUT_DIR_V2, exist_ok=True)

    DATA_V2 = {
        'H=1':  {'HAR-J': 0.3052, 'GARCH-GJR': 1.8017,
                  'GARCH-10min': _g10_qlike['H=1']},
        'H=5':  {'HAR-J': 0.4243, 'GARCH-GJR': 1.0514,
                  'GARCH-10min': _g10_qlike['H=5']},
        'H=22': {'HAR-J': 0.4669, 'GARCH-GJR': 0.7320,
                  'GARCH-10min': _g10_qlike['H=22']},
    }

    COLOR_G10 = '#1a6b3a'   # dark green for GARCH-10min
    COLORS_V2 = [COLOR_HAR, COLOR_GARCH, COLOR_G10]
    MODELS_V2 = ['HAR-J', 'GARCH-GJR', 'GARCH-10min']

    def make_bar_chart_v2(horizon_key, subtitle, filename, ylim_top):
        """3-bar chart: HAR-J / GARCH-daily / GARCH-10min."""
        values = [DATA_V2[horizon_key][m] for m in MODELS_V2]

        fig, ax = plt.subplots(figsize=(6.5, 4.5))

        bars = ax.bar(MODELS_V2, values, color=COLORS_V2, width=0.50,
                      edgecolor='black', linewidth=0.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ylim_top * 0.015,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

        ax.set_ylabel('QLIKE', fontsize=12)
        ax.set_title(f'Сравнение QLIKE моделей:\nгоризонт {subtitle}',
                     fontsize=13, fontweight='bold', pad=10)
        ax.set_ylim(0, ylim_top)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')
        ax.tick_params(axis='x', length=0)

        fig.savefig(f'{OUT_DIR_V2}/{filename}')
        print(f'Saved: {OUT_DIR_V2}/{filename}')
        plt.close(fig)

    # Figures a-c: per-horizon bar charts
    make_bar_chart_v2('H=1',  'H=1 (1 день)',  'fig_3_1a_h1.png',  ylim_top=2.1)
    make_bar_chart_v2('H=5',  'H=5 (5 дней)',  'fig_3_1b_h5.png',  ylim_top=1.25)
    make_bar_chart_v2('H=22', 'H=22 (22 дня)', 'fig_3_1c_h22.png', ylim_top=0.9)

    # Figure d: 3D bar chart — all horizons, 3 models
    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection='3d')

    horizons_v2 = ['H=1', 'H=5', 'H=22']
    x_pos_v2 = np.array([0, 1, 2])

    y_positions = [0, 1, 2]   # HAR=0, GARCH-daily=1, GARCH-10min=2
    bar_w = 0.3
    bar_d = 0.35

    for mi, (model_name, color, y_base) in enumerate(
        zip(MODELS_V2, COLORS_V2, y_positions)
    ):
        vals = [DATA_V2[h][model_name] for h in horizons_v2]
        ax.bar3d(x_pos_v2 - bar_w / 2,
                 np.full(3, y_base), np.zeros(3),
                 bar_w, bar_d, vals,
                 color=color, alpha=0.85,
                 edgecolor='black', linewidth=0.3,
                 label=model_name)

    ax.set_xticks(x_pos_v2)
    ax.set_xticklabels(horizons_v2, fontsize=10)
    ax.set_yticks([y + 0.17 for y in y_positions])
    ax.set_yticklabels(MODELS_V2, fontsize=9)
    ax.set_zlabel('QLIKE', fontsize=12, labelpad=8)
    ax.set_xlabel('Горизонт', fontsize=12, labelpad=8)
    ax.set_ylabel('Модель', fontsize=12, labelpad=12)

    ax.set_title(
        'Сравнение QLIKE: HAR-J, GARCH-GJR, GARCH-10min\n'
        'по горизонтам прогнозирования',
        fontsize=13, fontweight='bold', pad=15,
    )

    ax.view_init(elev=25, azim=-50)
    ax.set_zlim(0, 2.0)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.zaxis._axinfo['grid'].update(
        {'color': 'gray', 'linestyle': '--', 'linewidth': 0.4})
    ax.xaxis._axinfo['grid'].update(
        {'color': 'gray', 'linestyle': '--', 'linewidth': 0.4})
    ax.yaxis._axinfo['grid'].update(
        {'color': 'gray', 'linestyle': '--', 'linewidth': 0.4})

    fig.savefig(f'{OUT_DIR_V2}/fig_3_1d_3d_surface.png')
    print(f'Saved: {OUT_DIR_V2}/fig_3_1d_3d_surface.png')
    plt.close(fig)

    print(f'\nVariant 2: all 4 figures saved to {OUT_DIR_V2}/')
