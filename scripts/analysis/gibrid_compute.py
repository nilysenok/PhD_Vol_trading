"""Схемы гибрида на всех шагах для одной поправки (вынесено из gibrid_v17.py, Д8 25.10). Набор схем - не больше 15:
V0 x 3 (HAR-J, XGB, LGB) + V0_HCJ (HAR-CJ, Д8.6) - в gibrid_designs.v0; кандидаты выбора step/minimax (gibrid_designs.CAND): V1, V2, V3, V4,
V6, V7_HalfLife (Д8.3), V10_MinQ (Д8.2), V11_Gap (Д8.5), Med3 (Д8.6), Gate_L2 (Д8.7); G3_InvQ - выбор g3. Итого 15.
V5_Combined и V7 с окном 20 считаются только для справки (столбцы q_*), выбором не берутся. Смешивание - по S.MODE (--blend)."""
import numpy as np
import gibrid_schemes as S
import gibrid_bias as BI
import gibrid_d8 as D8
from gibrid_io import merged


def _prep(har, bx, bl, ext, part):
    d = merged(har, bx, bl)
    d = d.merge(ext['hcj'][part], on=['date', 'ticker'], how='left').merge(ext['state'], on=['date', 'ticker'], how='left')
    assert d['hcj'].notna().all(), 'HAR-CJ: нет прогноза на часть строк'
    return d


def compute(hv, ht, h, bx, bl, code, lag, log, ext):
    """Список шагов {q, W, pv, pt, vs, ts, base}. code: bez | ord | rob | sm | smrob | ord_in. ext: hcj (val/test), state."""
    val, test = _prep(hv, bx['val'], bl['val'], ext, 'val'), _prep(ht, bx['test'], bl['test'], ext, 'test')
    val['har_raw'] = val['har']
    if code != 'bez':
        BI.correct(val, test, BI.resid(val), code, h, log)
    for d in (val, test):
        d['v6'], W6 = S.v6(d)
        d['v7'], W7 = S.v7(d, 20, lag)
        d[['w6h', 'w6x', 'w6l']], d[['w7h', 'w7x', 'w7l']] = W6, W7
        for hl in D8.HL:
            d[f'v7hl{hl}'], Wh = S.v7(d, 20, lag, hl)
            d[[f'w7hl{hl}{c}' for c in 'hxl']] = Wh
    res = []
    for st in sorted(val['step'].unique()):
        vs, ts = val[val.step == st], test[test.step == st]
        y = vs['rv_actual'].values
        prev = val[val.step == st - 1] if st > 1 else None
        ratio = vs['rv_actual'].mean() / val[val.step <= st]['rv_actual'].mean()  # история = val-годы 1..st
        W, reg = S.weights_v2_v5(vs, prev, ratio)
        w1, meth, ml = S.v1(vs)
        w10, lam10 = D8.v10(vs)
        W = {'V1_Adaptive': np.array(w1), **W, 'V10_MinQ': w10}
        F = {k: _fixed(w) for k, w in W.items()}  # схема -> применение параметров шага к любому набору строк
        hl = min(D8.HL, key=lambda x: S.qlike(y, vs[f'v7hl{x}'].values))  # Д8.3: полураспад по val-году шага
        for k, c, p in (('V6_TrimmedW', 'w6', 'v6'), ('V7_Rolling_W20', 'w7', 'v7'), ('V7_HalfLife', f'w7hl{hl}', f'v7hl{hl}')):
            F[k] = _cols(p, c)
        th, tau = D8.v11_fit(vs)
        g = D8.gate_fit(vs)
        F['V11_Gap'] = lambda d, th=th, tau=tau: D8.v11_apply(d, th, tau)
        F['Med3'] = D8.med3
        F['Gate_L2'] = lambda d, g=g: D8.gate_apply(d, g)
        base = dict(step=st, test_year=int(ts['year'].iloc[0]), val_year=int(vs['year'].iloc[0]), v1_ml=ml, v1_method=meth,
                    regime=reg, regime_ratio=ratio, v1_w_har=w1[0], q_har=S.qlike(y, vs['har'].values), q_har_raw=S.qlike(y, vs['har_raw'].values),
                    q_xgb_single=S.qlike(y, vs['xgb'].values), q_lgb_single=S.qlike(y, vs['lgb'].values), q_hcj_single=S.qlike(y, vs['hcj'].values),
                    v7_hl=hl, v10_lam=lam10, v11_theta=th, v11_tau=tau, gate_b=' '.join(f'{b:.3f}' for b in g['b']))
        iq = 1.0 / np.array([base['q_har'], base['q_xgb_single'], base['q_lgb_single']])  # Г3: веса по обратному val-QLIKE компонент
        F['G3_InvQ'] = _fixed(iq / iq.sum())
        pv, pt, W = {}, {}, {}
        for k, f in F.items():
            (pv[k], w), pt[k] = f(vs), f(ts)[0]
            W[k] = w.mean(axis=0) if np.ndim(w) == 2 else np.asarray(w, dtype=float)  # построчные веса -> средние val-года (для c4)
        q = {k: S.qlike(y, p) for k, p in pv.items()}
        res.append(dict(q=q, W=W, pv=pv, pt=pt, vs=vs, ts=ts, base=base, F=F, po=None, qo=None))
    # оценка разработочного набора вне выборки (решение дирижёра 25.10): параметры шага t-1 -> val-год шага t; шаг 1 - без оценки
    for prv, r in zip(res[:-1], res[1:]):
        y = r['vs']['rv_actual'].values
        r['po'] = {k: f(r['vs'])[0] for k, f in prv['F'].items()}
        r['qo'] = {k: S.qlike(y, p) for k, p in r['po'].items()}
    return res


def _fixed(w):
    w = np.asarray(w, dtype=float)
    return lambda d: (S.blend(w, d), w)


def _cols(p, c):
    return lambda d: (d[p].values, d[[c + 'h', c + 'x', c + 'l']].values)
