"""
Рисунки 3.5–3.7 — QLIKE по годам для горизонтов H=1, H=5, H=22
Линейные графики: HAR-J, XGBoost, LightGBM, Гибридная
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

years = list(range(2017, 2026))

# Data from walk-forward (strategies_v3.csv)
# Hybrid = best strategy per horizon: H=1 V2_MultiVal, H=5 V6_TrimmedW, H=22 V7_Rolling
data = {
    'HAR-J': {
        1:  [0.275, 0.418, 0.305, 0.464, 0.296, 1.307, 0.478, 0.438, 0.737],
        5:  [0.295, 0.482, 0.424, 0.572, 0.329, 3.264, 0.513, 0.547, 0.737],
        22: [0.318, 0.459, 0.467, 1.152, 0.492, 3.890, 0.502, 0.643, 0.725],
    },
    'XGBoost': {
        1:  [0.262, 0.342, 0.277, 0.350, 0.242, 0.751, 0.429, 0.337, 0.613],
        5:  [0.333, 0.432, 0.395, 0.897, 0.398, 4.456, 0.463, 0.522, 0.704],
        22: [0.348, 0.454, 0.469, 1.006, 0.560, 5.482, 0.461, 0.624, 0.744],
    },
    'LightGBM': {
        1:  [0.324, 0.361, 0.290, 0.309, 0.232, 0.626, 0.381, 0.314, 0.568],
        5:  [0.296, 0.464, 0.383, 0.789, 0.352, 4.139, 0.465, 0.508, 0.686],  # gbdt
        22: [0.364, 0.462, 0.455, 0.961, 0.476, 3.201, 0.626, 0.629, 0.659],
    },
    'Гибридная': {
        1:  [0.253, 0.338, 0.275, 0.325, 0.226, 0.655, 0.382, 0.313, 0.573],
        5:  [0.323, 0.434, 0.379, 0.519, 0.317, 3.153, 0.437, 0.482, 0.672],
        22: [0.302, 0.443, 0.434, 0.935, 0.430, 3.627, 0.451, 0.454, 0.671],
    },
}

# Style per model
styles = {
    'HAR-J':      {'color': '#999999', 'linestyle': '--', 'linewidth': 1.8, 'marker': 's', 'markersize': 5, 'zorder': 2},
    'XGBoost':    {'color': '#2166AC', 'linestyle': '-',  'linewidth': 1.8, 'marker': 'o', 'markersize': 5, 'zorder': 3},
    'LightGBM':   {'color': '#67A9CF', 'linestyle': '-',  'linewidth': 1.8, 'marker': '^', 'markersize': 5, 'zorder': 3},
    'Гибридная':  {'color': '#1B9E37', 'linestyle': '-',  'linewidth': 2.8, 'marker': 'D', 'markersize': 6, 'zorder': 4},
}

configs = [
    (1,  'fig_3_5_qlike_by_year_h1.png',  'Рисунок 3.5 — QLIKE по годам, горизонт H=1'),
    (5,  'fig_3_6_qlike_by_year_h5.png',  'Рисунок 3.6 — QLIKE по годам, горизонт H=5'),
    (22, 'fig_3_7_qlike_by_year_h22.png', 'Рисунок 3.7 — QLIKE по годам, горизонт H=22'),
]

for horizon, filename, title in configs:
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')

    for model_name in ['HAR-J', 'XGBoost', 'LightGBM', 'Гибридная']:
        vals = data[model_name][horizon]
        if vals is None:
            continue
        style = styles[model_name]
        ax.plot(years, vals, label=model_name, **style)

    ax.set_xlabel('Год', fontsize=12)
    ax.set_ylabel('QLIKE', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=0, fontsize=10)
    ax.tick_params(axis='y', labelsize=10)

    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    plt.tight_layout()
    outpath = f'dissertation_materials/chapter_3_2/{filename}'
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.15)
    print(f'Saved {filename}')
    plt.close()
