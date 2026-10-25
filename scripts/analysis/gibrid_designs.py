"""Выбор схемы на шаге (по val-QLIKE шага / минимакс по доступным val-годам) и четыре критерия владельца (25.10), только валидация.
c1: q_дизайн <= 1,02*q_HAR(с Б0.1oos) в каждом val-году (V0=HAR-J не сравнивается сама с собой); c2: q_гибрид <= 1,02*min(q_HAR,q_xgb,q_lgb) в каждом году;
c3: max_год q_гибрид/q_HAR < min(max_год q_xgb/q_HAR, max_год q_lgb/q_HAR) (у HAR отношение 1 по определению — не берётся);
c4: |Δ w_comp| <= 0,3 между соседними шагами (обе схемы не V1) и не более 3 смен ML у V1. Критерии считаются по каждому h; дизайн проходит, если все h."""
import subprocess
from pathlib import Path
import numpy as np, pandas as pd


def devlog(path, text):
    p = Path(path)
    n = 1 + sum(1 for l in p.read_text().splitlines() if l.startswith('| ') and l[2:3].isdigit()) if p.exists() else 1
    when = subprocess.run(['date', '+%Y-%m-%d %H:%M'], capture_output=True, text=True).stdout.strip()
    head = '' if p.exists() else '# Журнал разработки (dev_log)\n\n| № | Дата-время | Что меняли |\n|---|---|---|\n'
    with open(p, 'a') as f:
        f.write(head + f'| {n} | {when} | {text.splitlines()[0]} |\n\n' + '\n'.join(text.splitlines()[1:]) + '\n\n')


def select(res, mode, h):
    """res: список шагов {q:{схема:qlike}, W:{схема:w3}, pv, pt, vs, ts, base}. Возвращает таблицу выбора, val- и test-прогнозы."""
    rows, ov, ot = [], [], []
    for i, r in enumerate(res):
        q = r['q']
        if mode == 'minimax':  # минимум худшего val-года среди лет, доступных к шагу (шаги 1..i)
            best = min(q, key=lambda k: max(x['q'][k] for x in res[:i + 1]))
        else:
            best = min(q, key=q.get)
        w = r['W'][best]
        row = dict(r['base'], h=h, scheme=best, w_har=w[0], w_xgb=w[1], w_lgb=w[2], q_hyb=q[best])
        row.update({f'q_{k}': v for k, v in q.items()})
        rows.append(row)
        for df, p, lst in ((r['vs'], r['pv'][best], ov), (r['ts'], r['pt'][best], ot)):
            lst.append(df[['date', 'ticker', 'year', 'step', 'rv_actual']].assign(rv_pred=p))
    return pd.DataFrame(rows), pd.concat(ov), pd.concat(ot)


def flags(ch):
    """ch — таблица выбора одного h (строка = val-год). Возвращает dict флагов и показателей."""
    hy, har, xg, lg = ch.q_hyb.values, ch.q_har.values, ch.q_xgb_single.values, ch.q_lgb_single.values
    # эталон — одиночные модели с Б0.1oos (q_har, q_xgb_single, q_lgb_single). V0 с самой собой не сравнивается (решение 25.10).
    v0c = ch.scheme.iloc[0][3:].lower() if ch.scheme.iloc[0].startswith('V0_') else None
    others = {'har': har, 'xgb': xg, 'lgb': lg}
    if v0c:
        others.pop(v0c)
    single_min = np.minimum.reduce(list(others.values()))
    worst = (hy / har).max()
    ml_only = [(v / har).max() for k, v in others.items() if k != 'har']  # у HAR отношение к самому себе = 1 — не берётся
    worst_single = min(ml_only)
    c1 = True if v0c == 'har' else bool((hy <= 1.02 * har).all())  # c1: не хуже HAR-J более чем на 2% (симметрично c2)
    sc, ml = ch.scheme.values, ch.v1_ml.values
    W = ch[['w_har', 'w_xgb', 'w_lgb']].values
    # c4 (решение 25.10): скачок весов <= 0,3 только между соседними шагами, где обе схемы непрерывные (не V1);
    # для V1 (бинарный выбор ML) — число смен ML-компоненты среди шагов, где выбрана V1, <= 3 из 9
    jump = max([0.0] + [np.abs(W[i] - W[i - 1]).max() for i in range(1, len(ch)) if sc[i] != 'V1_Adaptive' and sc[i - 1] != 'V1_Adaptive'])
    v1m = ml[sc == 'V1_Adaptive']
    sw = int((v1m[1:] != v1m[:-1]).sum()) if len(v1m) > 1 else 0
    return dict(c1=c1, c2=bool((hy <= 1.02 * single_min).all()), c3=bool(worst < worst_single),
                c4=bool(jump <= 0.3 and sw <= 3), ml_switches=sw, worst_ratio=worst, worst_ratio_single=worst_single, max_w_jump=jump,
                mean_q=float(hy.mean()))


def v0(res, comp, h):
    """V0 = одна компонента (с поправкой, если res посчитан с ней): веса one-hot, переключений нет -> c4 выполнен тривиально."""
    rows, ov, ot = [], [], []
    w = {'har': (1, 0, 0), 'xgb': (0, 1, 0), 'lgb': (0, 0, 1)}[comp]
    for r in res:
        y, p = r['vs']['rv_actual'].values, np.clip(r['vs'][comp].values, 1e-10, None)
        rows.append(dict(r['base'], h=h, scheme='V0_' + comp.upper(), w_har=w[0], w_xgb=w[1], w_lgb=w[2],
                         q_hyb=float(np.mean(y / p - np.log(y / p) - 1))))
        for df, lst in ((r['vs'], ov), (r['ts'], ot)):
            lst.append(df[['date', 'ticker', 'year', 'step', 'rv_actual']].assign(rv_pred=df[comp].values))
    return pd.DataFrame(rows), pd.concat(ov), pd.concat(ot)
