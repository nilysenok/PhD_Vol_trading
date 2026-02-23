"""
Рисунок 3.8 — Средний QLIKE по категориям моделей и горизонтам прогнозирования
3D bar chart: 4 categories × 3 horizons, sorted by avg QLIKE ascending
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

# Categories on test 2019 — sorted by ascending avg QLIKE (best first → worst last)
categories = ['Гибридная', 'Бустинг', 'RNN', 'Классическая']
horizons = ['H=1', 'H=5', 'H=22']

# All test 2019 QLIKE; GARCH-GJR = v2b (GJR + normal + overnight)
# Классическая = avg(HAR-J, GARCH-GJR v2b)
# Бустинг = avg(XGBoost, LightGBM)
# RNN = avg(LSTM, GRU)
qlike = np.array([
    [0.276,           0.373,           0.441],              # Гибридная      avg=0.363
    [(0.277+0.290)/2, (0.396+0.382)/2, (0.470+0.455)/2],  # Бустинг        avg=0.378
    [(0.372+0.399)/2, (0.435+0.411)/2, (0.485+0.503)/2],  # RNN            avg=0.434
    [(0.305+0.501)/2, (0.424+0.567)/2, (0.467+0.601)/2],  # Классическая   avg=0.478
])

colors = [
    '#1B9E37',    # Hybrid - green
    '#2166AC',    # Boosting - blue
    '#D6604D',    # RNN - red
    '#999999',    # Classical - grey
]
alphas = [0.95, 0.95, 0.95, 0.85]

n_cats = len(categories)
n_horizons = len(horizons)

fig = plt.figure(figsize=(10, 7), facecolor='white')
ax = fig.add_subplot(111, projection='3d')

dx = 0.55
dy = 0.55

# Draw back-to-front for correct occlusion
for i in range(n_cats - 1, -1, -1):
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
ax.set_yticks(range(n_cats))
ax.set_yticklabels(categories, fontsize=10)
ax.set_zlabel('QLIKE', fontsize=12, labelpad=10)
ax.set_xlabel('Горизонт прогнозирования', fontsize=12, labelpad=10)
ax.set_ylabel('Категория модели', fontsize=12, labelpad=10)

ax.set_zlim(0, 0.7)
ax.set_ylim(-0.5, n_cats - 0.5)
ax.set_xlim(-0.5, n_horizons - 0.5)
ax.view_init(elev=25, azim=-50)

# Title
fig.suptitle('Рисунок 3.8 — Средний QLIKE по категориям моделей\nи горизонтам прогнозирования',
             fontsize=14, fontweight='bold', y=0.98)

# Footnote
fig.text(0.50, 0.01,
         'Тестовый период 2019; Классическая = среднее HAR-J и GARCH-GJR',
         ha='center', va='bottom', fontsize=9, fontstyle='italic', color='#666666')

# Clean panes
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('lightgrey')
ax.yaxis.pane.set_edgecolor('lightgrey')
ax.zaxis.pane.set_edgecolor('lightgrey')
ax.grid(True, alpha=0.3)

plt.savefig(f'{OUT_DIR}/fig_3_8_categories_3d.png',
            dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
print('Saved fig_3_8_categories_3d.png')
plt.close()
