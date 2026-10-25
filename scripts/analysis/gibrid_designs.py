"""Выбор схемы на шаге (по val-QLIKE шага / минимакс по доступным val-годам) и четыре критерия владельца (25.10), только валидация.
c1: q_дизайн <= 1,02*q_HAR(с Б0.1oos) в каждом val-году (V0=HAR-J не сравнивается сама с собой); c2: q_гибрид <= 1,02*min(q_HAR,q_xgb,q_lgb) в каждом году;
c3: max_год q_гибрид/q_HAR < min(max_год q_xgb/q_HAR, max_год q_lgb/q_HAR) (у HAR отношение 1 по определению — не берётся);
c4: |Δ w_comp| <= 0,3 между соседними шагами (обе схемы не V1) и не более 3 смен ML у V1. Критерии считаются по каждому h; дизайн проходит, если все h."""
import re, subprocess
from pathlib import Path
import numpy as np, pandas as pd
import gibrid_schemes as S

# Д8.8-Д8.11: состав - gibrid_compute. V10_Fix (Д8.10) исключён из кандидатов (решение дирижёра 25.10: веса по val 2017-2021
# применяются к тесту 2017-2020, val шага t = тест шага t-1 - заглядывание); остаётся справкой (столбцы q_V10_Fix)
CAND = ('V3_Regime', 'V7_HalfLife', 'V10_MinQ', 'V10_Pool', 'V11_Gap', 'Med3', 'Gate_L2', 'EQ_Equal', 'ML2_Pair')


def devlog(path, text):
    p = Path(path)
    # номер итерации - следующий за наибольшим в строках журнала «| № | дата |» (строки таблиц с дизайнами не считаются)
    n = 1 + max([int(m.group(1)) for m in re.finditer(r'^\| (\d+) \| \d{4}-\d\d-\d\d', p.read_text(), re.M)] + [0]) if p.exists() else 1
    when = subprocess.run(['date', '+%Y-%m-%d %H:%M'], capture_output=True, text=True).stdout.strip()
    head = '' if p.exists() else '# Журнал разработки (dev_log)\n\n| № | Дата-время | Что меняли |\n|---|---|---|\n'
    with open(p, 'a') as f:
        f.write(head + f'| {n} | {when} | {text.splitlines()[0]} |\n\n' + '\n'.join(text.splitlines()[1:]) + '\n\n')


def pick(res, i, mode):
    """Схема шага i по его val-году (как для теста): step - мин. q; minimax - мин. худшего года среди шагов 1..i; g3 - G3.
    Кандидаты step/minimax - набор Д8 (CAND); V5 и V7 с окном 20 - только справка."""
    if mode == 'g3':
        return 'G3_InvQ'
    keys = [k for k in CAND if k in res[i]['q']]
    if mode == 'minimax':
        return min(keys, key=lambda k: max(x['q'][k] for x in res[:i + 1]))
    return min(keys, key=lambda k: res[i]['q'][k])


def select(res, mode, h, ev='oos'):
    """res: шаги {q, W, pv, pt, vs, ts, base, po, qo}. Тест шага t - схема и параметры шага t (scheme_test).
    ev='oos' (по умолчанию, решение дирижёра 25.10): метрика val-года шага t - схема, выбранная на шаге t-1, с её параметрами
    шага t-1 (q_hyb, scheme, w_* - применённые; q_<схема> - вне выборки, qi_<схема> - в выборке); шаг 1 - q_hyb = NaN.
    ev='in' - справка: всё по тому же val-году шага. Возвращает таблицу выбора, val- (rv_pred, rv_pred_oos) и test-прогнозы."""
    rows, ov, ot = [], [], []
    for i, r in enumerate(res):
        q, best = r['q'], pick(res, i, mode)
        w = r['W'][best]
        row = dict(r['base'], h=h, scheme=best, scheme_test=best, w_har=w[0], w_xgb=w[1], w_lgb=w[2], q_hyb=q[best])
        row.update({f'q_{k}': v for k, v in q.items()}); row.update({f'qi_{k}': v for k, v in q.items()})
        po = np.full(len(r['vs']), np.nan)
        if ev == 'oos':
            row['q_hyb'] = np.nan
            if i > 0:
                bp = pick(res, i - 1, mode); wp = res[i - 1]['W'][bp]
                row.update(scheme=bp, w_har=wp[0], w_xgb=wp[1], w_lgb=wp[2], v1_ml=res[i - 1]['base']['v1_ml'], q_hyb=r['qo'][bp])
                row.update({f'q_{k}': v for k, v in r['qo'].items()})
                po = r['po'][bp]
        rows.append(row)
        ov.append(r['vs'][['date', 'ticker', 'year', 'step', 'rv_actual']].assign(rv_pred=r['pv'][best], rv_pred_oos=po))
        ot.append(r['ts'][['date', 'ticker', 'year', 'step', 'rv_actual']].assign(rv_pred=r['pt'][best]))
    return pd.DataFrame(rows), pd.concat(ov), pd.concat(ot)


def shrink(res, lam):
    """Рычаг Г4: усадка к равным весам. Прогноз схемы -> (1-lam)*прогноз + lam*среднее(HAR, XGB, LGB); веса так же.
    Смешивание по S.MODE (--blend): rv - арифметическое, log - геометрическое (и среднее трёх, и усадка). То же - для po (вне выборки)."""
    if lam == 0:
        return res
    out = []
    eq, mix = np.array([1 / 3] * 3), np.array([1 - lam, lam])
    for r in res:
        mv, mt = (S.comb(r[k][['har', 'xgb', 'lgb']].values, eq) for k in ('vs', 'ts'))
        sh = lambda P, m: None if P is None else {k: S.comb(np.column_stack([p, m]), mix) for k, p in P.items()}
        pv, pt, po = sh(r['pv'], mv), sh(r['pt'], mt), sh(r['po'], mv)
        y = r['vs']['rv_actual'].values
        qd = lambda P: None if P is None else {k: S.qlike(y, p) for k, p in P.items()}
        out.append(dict(r, q=qd(pv), qo=qd(po), pv=pv, pt=pt, po=po, W={k: (1 - lam) * np.asarray(w, dtype=float) + lam / 3 for k, w in r['W'].items()}))
    return out


def flags(ch):
    """ch — таблица выбора одного h (строка = val-год). Возвращает dict флагов и показателей."""
    hy, har, xg, lg = ch.q_hyb.values, ch.q_har.values, ch.q_xgb_single.values, ch.q_lgb_single.values
    # эталон — одиночные модели с Б0.1oos (q_har, q_xgb_single, q_lgb_single). V0 с самой собой не сравнивается (решение 25.10).
    v0c = ch.scheme.iloc[0][3:].lower() if ch.scheme.iloc[0].startswith('V0_') else None
    others = {'har': har, 'xgb': xg, 'lgb': lg}
    others.pop(v0c, None)  # V0_HCJ (HAR-CJ, Д8.6) сравнивается со всеми тремя: эталон критериев - прежний
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
    w = {'har': (1, 0, 0), 'xgb': (0, 1, 0), 'lgb': (0, 0, 1), 'hcj': (0, 0, 0)}[comp]  # hcj - вне тройки, веса 0
    for r in res:
        y, p = r['vs']['rv_actual'].values, np.clip(r['vs'][comp].values, 1e-10, None)
        rows.append(dict(r['base'], h=h, scheme='V0_' + comp.upper(), w_har=w[0], w_xgb=w[1], w_lgb=w[2],
                         q_hyb=float(np.mean(y / p - np.log(y / p) - 1))))
        for df, lst in ((r['vs'], ov), (r['ts'], ot)):
            lst.append(df[['date', 'ticker', 'year', 'step', 'rv_actual']].assign(rv_pred=df[comp].values))
    return pd.DataFrame(rows), pd.concat(ov), pd.concat(ot)
