"""Гибрид V1-V7 на прогнозах бустинга + пространство дизайнов (слепой тест: метрики только на разработочном наборе val 2016-2021).
python3 -W ignore scripts/analysis/gibrid_v17.py --boost P1|B --out data/regen [--regen <каталог файлов бустинга>] [--h 22] [--v7-lag h|1]
    [--devlog docs/otchety/dev_log.md] [--etalon ord|same] [--confirm --finalists auto|"дизайн1,дизайн2"]
Дизайн = бустинг-нога (single | top5 | seeds3) | выбор схемы (step | minimax | g3) | поправка (bez | B-ord | B-rob | B-sm | B-smrob) | усадка L0/L0.25/L0.5;
V0 = HAR-J / XGB / LGB по отдельности (с каждой поправкой). g3 = веса ~ 1/val-QLIKE трёх компонент (рычаг Г3). Схемы V1-V7: gibrid_schemes.py.
Вход: {regen}/{xgb|B_xgb}_h{h}_{val,test}.parquet, {P1|B_lgbm}_... (+ _top5_, _seeds4344_); HAR-J пересчитывается (gibrid_har.py).
Выход по h: designs_val_{b}_h{h}.csv, ranking_{b}_h{h}.csv, final_choice_{b}_h{h}.csv, hyb_{b}_final_h{h}_{val,test}.parquet, hyb_{b}_h{h}_{val,test}.parquet
(дизайн «как в коде»), hyb_{b}_h{h}_choice.csv, bias_factors_{b}_h{h}.csv, ref_{b}_h{h}.csv, har_specs_val_{b}_h{h}.csv, hyb_{b}_har_check_h{h}.csv."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import gibrid_har as GH
import gibrid_har_spec as HS
import gibrid_schemes as S
import gibrid_designs as D
import gibrid_report as RP
import gibrid_bias as BI
from gibrid_io import load_boost, merged

DEV = 6  # разработочный набор: шаги 1-6 (val 2016-2021); подтверждающий: шаги 7-9 (val 2022-2024), скрыт без --confirm
CR = ('c1', 'c2', 'c3', 'c4')
LAMS = (0.0, 0.25, 0.5)


def compute(hv, ht, h, bx, bl, code, lag, log):
    """Схемы V1-V7 и G3 на всех шагах -> список шагов (q, веса, прогнозы val/test). code: bez | ord | rob | sm | smrob | ord_in."""
    val, test = merged(hv, bx['val'], bl['val']), merged(ht, bx['test'], bl['test'])
    val['har_raw'] = val['har']
    if code != 'bez':
        BI.correct(val, test, BI.resid(val), code, h, log)
    for d in (val, test):
        d['v6'], W6 = S.v6(d)
        d['v7'], W7 = S.v7(d, 20, lag)
        d[['w6h', 'w6x', 'w6l']], d[['w7h', 'w7x', 'w7l']] = W6, W7
    res = []
    for st in sorted(val['step'].unique()):
        vs, ts = val[val.step == st], test[test.step == st]
        prev = val[val.step == st - 1] if st > 1 else None
        ratio = vs['rv_actual'].mean() / val[val.step <= st]['rv_actual'].mean()  # история = val-годы 1..st
        W, reg = S.weights_v2_v5(vs, prev, ratio)
        w1, meth, ml = S.v1(vs)
        W = {'V1_Adaptive': np.array(w1), **W}
        pv = {k: S.blend(w, vs) for k, w in W.items()}; pt = {k: S.blend(w, ts) for k, w in W.items()}
        for k, c, p in (('V6_TrimmedW', 'w6', 'v6'), ('V7_Rolling_W20', 'w7', 'v7')):
            pv[k], pt[k] = vs[p].values, ts[p].values
            W[k] = vs[[c + 'h', c + 'x', c + 'l']].mean().values
        y = vs['rv_actual'].values
        base = dict(step=st, test_year=int(ts['year'].iloc[0]), val_year=int(vs['year'].iloc[0]), v1_ml=ml, v1_method=meth,
                    regime=reg, regime_ratio=ratio, v1_w_har=w1[0], q_har=S.qlike(y, vs['har'].values), q_har_raw=S.qlike(y, vs['har_raw'].values),
                    q_xgb_single=S.qlike(y, vs['xgb'].values), q_lgb_single=S.qlike(y, vs['lgb'].values))
        iq = 1.0 / np.array([base['q_har'], base['q_xgb_single'], base['q_lgb_single']])  # Г3: веса по обратному val-QLIKE компонент
        W['G3_InvQ'] = iq / iq.sum(); pv['G3_InvQ'], pt['G3_InvQ'] = S.blend(W['G3_InvQ'], vs), S.blend(W['G3_InvQ'], ts)
        q = {k: S.qlike(y, p) for k, p in pv.items()}
        res.append(dict(q=q, W=W, pv=pv, pt=pt, vs=vs, ts=ts, base=base))
    return res


def hide(df, confirm):
    """Метрики подтверждающего набора (q_*, ratio_har) обнуляются, если не --confirm. Схемы/веса по шагам остаются (нужны для теста)."""
    if confirm:
        return df
    df = df.copy()
    df.loc[df.step > DEV, [c for c in df.columns if c.startswith('q_') or c == 'ratio_har']] = np.nan
    return df


def run_h(h, A, harj, a, out, regen):
    hv, ht, _ = GH.har_forecasts(h, A, harj)
    lag = h if a.v7_lag == 'h' else int(a.v7_lag)
    rows, outs, chs, blog, ref = [], {}, {}, [], []

    def add(ch, ov, ot, name):
        ch.insert(0, 'design', name)
        fl = D.flags(ch[ch.step <= DEV])  # критерии - только на разработочном наборе
        ch['ratio_har'] = ch.q_hyb / ch.q_har
        for k in CR + ('worst_ratio', 'worst_ratio_single', 'max_w_jump', 'ml_switches'):
            ch[k + '_h'] = fl[k]
        rows.append(ch); outs[name] = (ov, ot); chs[name] = ch

    for var in ['single'] + (['top5', 'seeds3'] if a.boost == 'B' else []):
        try:
            bx, bl = load_boost(a.boost, h, regen, var)
        except FileNotFoundError as e:
            print('нет файла, вариант', var, 'пропущен:', e, flush=True); continue
        etalon = None  # c1-c3 для всех дизайнов - одиночные модели с Б0.1 ord (--etalon ord) или с той же оценкой, что у дизайна (--etalon same)
        for code in ('ord', 'rob', 'sm', 'smrob', 'bez'):
            lg = []
            res = compute(hv, ht, h, bx, bl, code, lag, lg)
            if code == 'ord':
                etalon = [{k: r['base'][k] for k in ('q_har', 'q_xgb_single', 'q_lgb_single')} for r in res]
            elif a.etalon == 'ord' or code == 'bez':
                for r, e in zip(res, etalon):
                    r['base'].update(e)  # q_har_raw остаётся без поправки
            if var == 'single':
                blog += lg
            lab = 'bez' if code == 'bez' else 'B-' + code
            for lam in LAMS:
                rs = D.shrink(res, lam)
                for sel in ('step', 'minimax', 'g3'):
                    add(*D.select(rs, sel, h), f'{var}|{sel}|{lab}|L{lam:g}')
            for comp in ('har', 'xgb', 'lgb'):
                add(*D.v0(res, comp, h), f'{var}|V0_{comp.upper()}|{lab}|L0')
        if var == 'single':  # справка (разработочный набор): in-sample Б0.1 и V7 с lag=1 (исходная реализация)
            c1, c0, cl = (D.select(compute(hv, ht, h, bx, bl, b, l, []), 'step', h)[0] for b, l in (('ord_in', lag), ('bez', lag), ('bez', 1)))
            m = lambda c, col: c.loc[c.step <= DEV, col].mean()
            ref.append(dict(h=h, hyb_B01_insample=m(c1, 'q_hyb'), har_B01_insample=m(c1, 'q_har'), hyb_bez_lag_h=m(c0, 'q_hyb'),
                            V7_lag_h=m(c0, 'q_V7_Rolling_W20'), V7_lag1_orig=m(cl, 'q_V7_Rolling_W20'), hyb_bez_lag1=m(cl, 'q_hyb'),
                            V7_steps_lag_h=str([s for s in c0.step[c0.scheme == 'V7_Rolling_W20'] if s <= DEV]),
                            V7_steps_lag1=str([s for s in cl.step[cl.scheme == 'V7_Rolling_W20'] if s <= DEV])))
    if not rows:
        print('h', h, 'нет данных бустинга, пропуск'); return None
    R = pd.concat(rows, ignore_index=True)
    sp = HS.run(A, harj, (h,))
    pd.DataFrame([GH.check_saved(h, ht)]).to_csv(out / f'hyb_{a.boost}_har_check_h{h}.csv', index=False)
    hide(R, a.confirm).to_csv(out / f'designs_val_{a.boost}_h{h}.csv', index=False)
    pd.DataFrame(blog).to_csv(out / f'bias_factors_{a.boost}_h{h}.csv', index=False)
    pd.DataFrame(ref).to_csv(out / f'ref_{a.boost}_h{h}.csv', index=False)
    hide(sp, a.confirm).to_csv(out / f'har_specs_val_{a.boost}_h{h}.csv', index=False)
    code = 'single|step|bez|L0'
    hide(R[R.design == code], a.confirm).drop(columns=['design']).to_csv(out / f'hyb_{a.boost}_h{h}_choice.csv', index=False)
    for nm, tag in ((code, f'hyb_{a.boost}_h{h}'),):
        outs[nm][0].to_parquet(out / f'{tag}_val.parquet', index=False); outs[nm][1].to_parquet(out / f'{tag}_test.parquet', index=False)
    # правило выбора: прошедшие все критерии -> мин. средний val-QLIKE; иначе мин. нарушенных, затем мин. QLIKE; всё на 2016-2021
    Rd = R[R.step <= DEV]
    F = Rd.groupby('design', sort=False).agg(**{k: (k + '_h', 'first') for k in CR}, mean_q=('q_hyb', 'mean')).reset_index()
    F['h'] = h
    F['failed'] = F.apply(lambda r: ','.join(k for k in CR if not r[k]), axis=1)
    F['n_failed'] = F.apply(lambda r: sum(not r[k] for k in CR), axis=1)
    F['family'] = F.design.map(lambda n: n.split('|')[1])
    F = F.sort_values(['n_failed', 'mean_q']).reset_index(drop=True)
    F.to_csv(out / f'ranking_{a.boost}_h{h}.csv', index=False)
    win = F.head(1)
    win.to_csv(out / f'final_choice_{a.boost}_h{h}.csv', index=False)
    ov, ot = outs[win.design.iloc[0]]
    ov.to_parquet(out / f'hyb_{a.boost}_final_h{h}_val.parquet', index=False); ot.to_parquet(out / f'hyb_{a.boost}_final_h{h}_test.parquet', index=False)
    auto = F.groupby('family').head(1).head(5).design.tolist()  # финалисты auto: лучшие по правилу в разных семействах (step/minimax/g3/V0_*), до 5
    return dict(h=h, Rd=Rd, F=F, win=win, ref=ref, sp=sp[sp.step <= DEV], chs=chs, auto=auto)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boost', required=True, choices=['P1', 'B'])
    ap.add_argument('--out', default='data/regen')
    ap.add_argument('--regen', default='data/regen')
    ap.add_argument('--h', type=int, nargs='*', default=[1, 5, 22], help='горизонты (по одному: --h 22)')
    ap.add_argument('--v7-lag', default='h', help="'h' (по умолчанию) = без заглядывания в цель; 1 = как в коде")
    ap.add_argument('--etalon', default='ord', choices=['ord', 'same'], help='эталон c1-c3: Б0.1 ord для всех дизайнов / та же оценка, что у дизайна')
    ap.add_argument('--devlog', default='docs/otchety/dev_log.md')
    ap.add_argument('--confirm', action='store_true', help='подтверждающий набор 2022-2024: только для --finalists, один раз')
    ap.add_argument('--finalists', default='', help='auto или имена дизайнов через запятую, напр. "single|step|B-ord|L0,single|V0_HAR|B-ord|L0"')
    a = ap.parse_args()
    out, regen = Path(a.out), Path(a.regen)
    out.mkdir(parents=True, exist_ok=True)
    A, harj = GH.load_all()
    P = [p for p in (run_h(h, A, harj, a, out, regen) for h in a.h) if p]
    if not P:
        return
    cnt = RP.counter(a.devlog, sum(len(p['F']) for p in P), max(len(p['F']) for p in P))
    L = RP.lines(P, a.boost, a.v7_lag, cnt, a.etalon)
    if a.confirm:  # один раз, для финалистов: критерии 1-4 на шагах 7-9 (c4 внутри 7-9)
        L += ['', 'ПОДТВЕРЖДАЮЩИЙ НАБОР (val 2022-2024)', '| дизайн | h | c1 | c2 | c3 | c4 | средн. q |', '|---|---|---|---|---|---|---|']
        for p in P:
            names = p['auto'] if a.finalists == 'auto' else [x for x in a.finalists.split(',') if x]
            for nm in names:
                g = p['chs'][nm]; g = g[g.step > DEV]; fl = D.flags(g)
                L.append(f'| {nm.replace("|", "/")} | {p["h"]} | ' + ' | '.join('+' if fl[k] else '-' for k in CR) + f' | {fl["mean_q"]:.3f} |')
    D.devlog(a.devlog, '\n'.join(L))
    print(pd.concat([p['win'] for p in P])[['h', 'design', 'failed', 'mean_q']].to_string())


if __name__ == '__main__':
    main()
