"""Строка (в) эталона HAR-J: лучшая по val-QLIKE шага спецификация окна из списка этапа 2б (код оценки - etap2_okno.py:fit_predict,
ветка kanon-v22-pereschet): 1 E (расширяющееся), 2-4 R1/R2/R3 (скользящие 1/2/3 года от конца train_pure), 5 D2 (взвешенный OLS,
полураспад 2 года), 10 AveW (среднее прогнозов E, R1, R2, R3). Выбор на шаге t - по QLIKE val-года шага t (тест не используется).
Только валидация; без поправки Б0.1 (строка (в) - устойчивость, не эталон)."""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
import gibrid_har as GH

SPECS = {'1_E': None, '2_R1': 1, '3_R2': 2, '4_R3': 3, '5_D2': 'decay'}


def _fit(tp, harj, tg, decay):
    X = GH.clean(np.log(tp[harj].values.clip(1e-10))); y = np.log(tp[tg].values + 1e-10)
    w = None
    if decay:
        age = (pd.to_datetime(tp['date']).max() - pd.to_datetime(tp['date'])).dt.days.values / 365.25
        w = 0.5 ** (age / 2.0)
    return LinearRegression().fit(X, y, sample_weight=w)


def qlike(y, p):
    p = np.clip(p, 1e-10, None)
    return float(np.mean(y / p - np.log(y / p) - 1))


def run(A, harj):
    rows = []
    for h in (1, 5, 22):
        tg = f'rv_target_h{h}'
        for step, ty in enumerate([y for y in range(2017, 2026) if y in A.year.values], 1):
            full = A[A.year < ty]; vy = sorted(full.year.unique())[-1]
            tpa = full[full.year != vy].dropna(subset=[tg]); vi = full[full.year == vy].dropna(subset=[tg])
            vi = vi[vi[tg] > 1e-12]  # тот же фильтр, что в gibrid_v17.merged
            P = {}
            for name, k in SPECS.items():
                tp = tpa if k in (None, 'decay') else tpa[tpa.year > tpa.year.max() - k]
                m = _fit(tp, harj, tg, k == 'decay')
                P[name] = np.clip(np.exp(m.predict(GH.clean(np.log(vi[harj].values.clip(1e-10))))), 1e-10, None)
            P['10_AveW'] = np.mean([P[n] for n in ('1_E', '2_R1', '3_R2', '4_R3')], axis=0)
            q = {n: qlike(vi[tg].values, p) for n, p in P.items()}
            best = min(q, key=q.get)
            rows.append(dict(h=h, step=step, val_year=vy, chosen=best, q_chosen=q[best], **{f'q_{n}': v for n, v in q.items()}))
    return pd.DataFrame(rows)
