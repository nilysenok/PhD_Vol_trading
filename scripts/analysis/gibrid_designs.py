"""Выбор схемы на шаге (по val-QLIKE шага / минимакс по доступным val-годам) и четыре критерия владельца (25.10), только валидация.
c1: q_гибрид < q_HAR в каждом val-году; c2: q_гибрид <= 1,02*min(q_HAR,q_xgb,q_lgb) в каждом году;
c3: max_год q_гибрид/q_HAR < min(max_год q_xgb/q_HAR, max_год q_lgb/q_HAR) (у HAR отношение 1 по определению — не берётся);
c4: |Δ w_comp| <= 0,3 между соседними шагами по всем трём компонентам. Критерии считаются по каждому h; дизайн проходит, если все h."""
import numpy as np, pandas as pd


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
    single_min = np.minimum.reduce([har, xg, lg])
    worst = (hy / har).max()
    worst_single = min((xg / har).max(), (lg / har).max())
    jump = max(np.abs(np.diff(ch[c].values)).max() if len(ch) > 1 else 0 for c in ('w_har', 'w_xgb', 'w_lgb'))
    return dict(c1=bool((hy < har).all()), c2=bool((hy <= 1.02 * single_min).all()), c3=bool(worst < worst_single),
                c4=bool(jump <= 0.3), worst_ratio=worst, worst_ratio_single=worst_single, max_w_jump=jump,
                mean_q=float(hy.mean()))
