"""
Рисунок 3.5 — Сравнение QLIKE моделей по горизонтам прогнозирования
3D bar chart: 7 models × 3 horizons, sorted by avg QLIKE ascending
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# All models on test 2019 — sorted by ascending avg QLIKE (best first → worst last)
models = ['Гибридная', 'LightGBM', 'XGBoost', 'HAR-J', 'LSTM', 'GRU', 'GARCH-GJR']
horizons = ['H=1', 'H=5', 'H=22']

# Test 2019 QLIKE; GARCH-GJR = v2b (GJR + normal + overnight)
qlike = np.array([
    [0.276, 0.373, 0.441],   # Гибридная   avg=0.363
    [0.290, 0.382, 0.455],   # LightGBM    avg=0.376
    [0.277, 0.396, 0.470],   # XGBoost     avg=0.381
    [0.305, 0.424, 0.467],   # HAR-J       avg=0.399
    [0.372, 0.435, 0.485],   # LSTM        avg=0.431
    [0.399, 0.411, 0.503],   # GRU         avg=0.438
    [0.501, 0.567, 0.601],   # GARCH-GJR   avg=0.556
])

colors = [
    '#1B9E37',    # Hybrid - green
    '#67A9CF',    # LightGBM - light blue
    '#2166AC',    # XGBoost - dark blue
    '#999999',    # HAR-J - grey
    '#D6604D',    # LSTM - red
    '#F4A582',    # GRU - orange
    '#8b1a1a',    # GARCH-GJR - dark red
]
alphas = [0.95, 0.95, 0.95, 0.95, 0.85, 0.85, 0.85]

n_models = len(models)
n_horizons = len(horizons)

fig = plt.figure(figsize=(10, 7.5), facecolor='white')
ax = fig.add_subplot(111, projection='3d')

dx = 0.45
dy = 0.45

# Draw back-to-front for correct occlusion (highest y first)
for i in range(n_models - 1, -1, -1):
    for j in range(n_horizons - 1, -1, -1):
        x = j
        y = i
        dz = qlike[i, j]

        ax.bar3d(x - dx/2, y - dy/2, 0, dx, dy, dz,
                 color=colors[i], alpha=alphas[i],
                 edgecolor='black', linewidth=0.3, shade=True)

# Axes
ax.set_xticks(range(n_horizons))
ax.set_xticklabels(horizons, fontsize=11)
ax.set_yticks(range(n_models))
ax.set_yticklabels(models, fontsize=9)
ax.set_zlabel('QLIKE', fontsize=12, labelpad=10)
ax.set_xlabel('Горизонт прогнозирования', fontsize=12, labelpad=10)
ax.set_ylabel('Модель', fontsize=12, labelpad=15)

ax.set_zlim(0, 0.7)
ax.set_ylim(-0.5, n_models - 0.5)
ax.set_xlim(-0.5, n_horizons - 0.5)
ax.view_init(elev=25, azim=-50)

# Title
fig.suptitle('Рисунок 3.5 — Сравнение QLIKE моделей\nпо горизонтам прогнозирования',
             fontsize=14, fontweight='bold', y=0.98)

# Footnote
fig.text(0.50, 0.01,
         'Тестовый период 2019 (N=3 910–4 284)',
         ha='center', va='bottom', fontsize=9, fontstyle='italic', color='#666666')

# Clean panes
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('lightgrey')
ax.yaxis.pane.set_edgecolor('lightgrey')
ax.zaxis.pane.set_edgecolor('lightgrey')
ax.grid(True, alpha=0.3)

plt.savefig(f'{OUT_DIR}/fig_3_5_all_models_3d.png',
            dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
print('Saved fig_3_5_all_models_3d.png')
plt.close()
