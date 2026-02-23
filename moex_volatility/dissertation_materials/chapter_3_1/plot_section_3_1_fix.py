#!/usr/bin/env python3
"""Generate 4 figures for Section 3.1 — fix: 'GARCH' instead of 'GARCH-GJR (10-мин)'.

fig_3_1a_h1.png   — vertical bar chart, H=1
fig_3_1b_h5.png   — vertical bar chart, H=5
fig_3_1c_h22.png  — vertical bar chart, H=22
fig_3_1d_3d.png   — 3D bar chart, all horizons
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

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

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Data (test 2019, Patton QLIKE) ---
DATA = {
    'H=1':  {'HAR-J': 0.3052, 'GARCH-GJR': 0.5006},
    'H=5':  {'HAR-J': 0.4243, 'GARCH-GJR': 0.5669},
    'H=22': {'HAR-J': 0.4669, 'GARCH-GJR': 0.6012},
}

COLOR_HAR   = '#1f4e79'
COLOR_GARCH = '#8b1a1a'
COLORS = [COLOR_HAR, COLOR_GARCH]
MODELS = ['HAR-J', 'GARCH-GJR']
MODEL_KEYS = ['HAR-J', 'GARCH-GJR']


# =====================================================================
# FIGURES 1-3: vertical bar charts per horizon
# =====================================================================
def make_bar_chart(horizon_key, subtitle, filename, ylim_top):
    values = [DATA[horizon_key][k] for k in MODEL_KEYS]

    fig, ax = plt.subplots(figsize=(5, 4.5))

    ax.bar(MODELS, values, color=COLORS, width=0.50,
           edgecolor='black', linewidth=0.5)

    ax.set_ylabel('QLIKE', fontsize=12)
    ax.set_title(f'Сравнение QLIKE моделей:\nгоризонт {subtitle}',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_ylim(0, ylim_top)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')
    ax.tick_params(axis='x', length=0)

    fig.savefig(f'{OUT_DIR}/{filename}')
    print(f'Saved: {OUT_DIR}/{filename}')
    plt.close(fig)


make_bar_chart('H=1',  'H=1 (1 день)',  'fig_3_1a_h1.png',  ylim_top=0.7)
make_bar_chart('H=5',  'H=5 (5 дней)',  'fig_3_1b_h5.png',  ylim_top=0.75)
make_bar_chart('H=22', 'H=22 (22 дня)', 'fig_3_1c_h22.png', ylim_top=0.8)


# =====================================================================
# FIGURE 4: 3D bar chart — all horizons
# =====================================================================
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

horizons = ['H=1', 'H=5', 'H=22']
x_pos = np.array([0, 1, 2])
y_har = np.zeros(3)
y_garch = np.ones(3)

har_vals   = [DATA[h]['HAR-J'] for h in horizons]
garch_vals = [DATA[h]['GARCH-GJR'] for h in horizons]

bar_width = 0.35
bar_depth = 0.4

ax.bar3d(x_pos - bar_width / 2, y_har, np.zeros(3),
         bar_width, bar_depth, har_vals,
         color=COLOR_HAR, alpha=0.85, edgecolor='black', linewidth=0.3,
         label='HAR-J')

ax.bar3d(x_pos - bar_width / 2, y_garch, np.zeros(3),
         bar_width, bar_depth, garch_vals,
         color=COLOR_GARCH, alpha=0.85, edgecolor='black', linewidth=0.3,
         label='GARCH-GJR')

ax.set_xticks(x_pos)
ax.set_xticklabels(horizons, fontsize=10)
ax.set_yticks([0.2, 1.2])
ax.set_yticklabels(['HAR-J', 'GARCH-GJR'], fontsize=10)
ax.set_zlabel('QLIKE', fontsize=12, labelpad=8)
ax.set_xlabel('Горизонт', fontsize=12, labelpad=8)
ax.set_ylabel('Модель', fontsize=12, labelpad=10)

ax.set_title('Сравнение QLIKE моделей HAR-J и GARCH-GJR\nпо горизонтам прогнозирования',
             fontsize=13, fontweight='bold', pad=15)

ax.view_init(elev=25, azim=-50)
ax.set_zlim(0, 0.7)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('lightgray')
ax.yaxis.pane.set_edgecolor('lightgray')
ax.zaxis.pane.set_edgecolor('lightgray')
ax.zaxis._axinfo['grid'].update({'color': 'gray', 'linestyle': '--', 'linewidth': 0.4})
ax.xaxis._axinfo['grid'].update({'color': 'gray', 'linestyle': '--', 'linewidth': 0.4})
ax.yaxis._axinfo['grid'].update({'color': 'gray', 'linestyle': '--', 'linewidth': 0.4})

fig.savefig(f'{OUT_DIR}/fig_3_1d_3d.png')
print(f'Saved: {OUT_DIR}/fig_3_1d_3d.png')
plt.close(fig)

print('\nDone! All 4 figures saved.')
