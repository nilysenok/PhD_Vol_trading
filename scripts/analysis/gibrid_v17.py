"""Гибрид V1-V7 на прогнозах бустинга + дизайны (слепой тест: метрики только на валидационных годах шагов).
python3 scripts/analysis/gibrid_v17.py --boost P1|B --out data/regen [--regen data/regen] [--v7-lag h|1] [--devlog docs/otchety/dev_log.md]
Дизайн = вариант бустинга (single | top5 | seeds3, если файлы есть) x выбор схемы (step | minimax) x поправка Б0.1 (без | с).
Вход: {regen}/{xgb|B_xgb}_h{h}_{val,test}.parquet, {P1|B_lgbm}_...; HAR-J пересчитывается (gibrid_har.py).
Выход: hyb_{boost}_h{h}_{val,test}.parquet и hyb_{boost}_choice.csv (дизайн «как в коде»: single/step/без), designs_val.csv,
hyb_{boost}_har_check.csv, hyb_{boost}_final_h{h}_{val,test}.parquet + final_choice_{boost}.csv (правило 25.10: мин. нарушенных
критериев, затем мин. средний val-QLIKE; в набор входят V0 = HAR-J / XGB / LGB с поправкой), ref_{boost}.csv, bias_factors_{boost}.csv."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import gibrid_har as GH
import gibrid_har_spec as HS
import gibrid_schemes as S
import gibrid_designs as D
import gibrid_report as RP
from gibrid_io import load_boost, merged


def s2_val(val):
    """s2 — дисперсия остатков ln(rv)-ln(прогноз) на val-годе шага, по компонентам (только валидация, тест не используется)."""
    return {c: {st: float(np.var(np.log(g['rv_actual'].values + 1e-10) - np.log(g[c].values)))
                for st, g in val.groupby('step')} for c in ('har', 'xgb', 'lgb')}


def bias_correct(val, test, s2, mode, h, log):
    """Б0.1: F=exp(s2/2) для HAR-J и обеих ML. mode 1 — s2 val-года шага (in-sample для val); mode 2 — s2 val-года предыдущего шага
    (вне выборки; на шаге 1 F=1); на тесте в обоих режимах F по val-году своего шага."""
    for st in sorted(val['step'].unique()):
        vm, tm = val['step'] == st, test['step'] == st
        for c in ('har', 'xgb', 'lgb'):
            fo = np.exp(s2[c][st] / 2)
            fv = fo if mode == 1 else (np.exp(s2[c][st - 1] / 2) if st > 1 else 1.0)
            val.loc[vm, c] *= fv; test.loc[tm, c] *= fo
            log.append(dict(h=h, step=st, comp=c, s2_own=s2[c][st], F_in=fo, F_oos=(np.exp(s2[c][st - 1] / 2) if st > 1 else 1.0)))


def compute(hv, ht, h, bx, bl, bias, lag, log):
    """Все схемы V1-V7 на всех шагах -> список шагов (q, веса, прогнозы val/test)."""
    val, test = merged(hv, bx['val'], bl['val']), merged(ht, bx['test'], bl['test'])
    val['har_raw'] = val['har']
    if bias:
        bias_correct(val, test, s2_val(val), bias, h, log)
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
        q = {k: S.qlike(y, p) for k, p in pv.items()}
        base = dict(step=st, test_year=int(ts['year'].iloc[0]), val_year=int(vs['year'].iloc[0]), v1_ml=ml, v1_method=meth,
                    regime=reg, regime_ratio=ratio, v1_w_har=w1[0], q_har=S.qlike(y, vs['har'].values), q_har_raw=S.qlike(y, vs['har_raw'].values),
                    q_xgb_single=S.qlike(y, vs['xgb'].values), q_lgb_single=S.qlike(y, vs['lgb'].values))
        res.append(dict(q=q, W=W, pv=pv, pt=pt, vs=vs, ts=ts, base=base))
    return res


DEV = 6  # разработочный набор: шаги 1-6 (val 2016-2021); подтверждающий: шаги 7-9 (val 2022-2024), скрыт без --confirm
CR = ('c1', 'c2', 'c3', 'c4')


def hide(df, confirm):
    """Метрики подтверждающего набора (q_*, ratio_har) обнуляются, если не --confirm. Схемы/веса по шагам остаются (нужны для теста)."""
    if confirm:
        return df
    df = df.copy()
    df.loc[df.step > DEV, [c for c in df.columns if c.startswith('q_') or c == 'ratio_har']] = np.nan
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boost', required=True, choices=['P1', 'B'])
    ap.add_argument('--out', default='data/regen')
    ap.add_argument('--regen', default='data/regen')
    ap.add_argument('--v7-lag', default='h', help="'h' (по умолчанию) = без заглядывания в цель; 1 = как в коде")
    ap.add_argument('--devlog', default='docs/otchety/dev_log.md')
    ap.add_argument('--confirm', action='store_true', help='подтверждающий набор 2022-2024: только для --finalists, один раз')
    ap.add_argument('--finalists', default='', help='имена дизайнов через запятую, напр. "single|step|B0.1oos,top5|step|B0.1oos"')
    a = ap.parse_args()
    out, regen = Path(a.out), Path(a.regen)
    out.mkdir(parents=True, exist_ok=True)
    A, harj = GH.load_all()
    variants = ['single'] + (['top5', 'seeds3'] if a.boost == 'B' else [])
    chk, rows, outs, blog, ref, chs = [], [], {}, [], [], {}

    def add(ch, ov, ot, name, h):
        ch.insert(0, 'design', name)
        fl = D.flags(ch[ch.step <= DEV])  # критерии - только на разработочном наборе
        ch['ratio_har'] = ch.q_hyb / ch.q_har
        for k in CR + ('worst_ratio', 'worst_ratio_single', 'max_w_jump', 'ml_switches'):
            ch[k + '_h'] = fl[k]
        rows.append(ch); outs[(name, h)] = (ov, ot); chs[(name, h)] = ch

    for h in (1, 5, 22):
        hv, ht, _ = GH.har_forecasts(h, A, harj)
        chk.append(GH.check_saved(h, ht))
        lag = h if a.v7_lag == 'h' else int(a.v7_lag)
        for var in variants:
            try:
                bx, bl = load_boost(a.boost, h, regen, var)
            except FileNotFoundError as e:
                print('нет файла, вариант', var, 'пропущен:', e, flush=True); continue
            etalon = None  # c1-c3 для ВСЕХ дизайнов - относительно одиночных моделей с Б0.1oos (HAR-J, XGB, LGB)
            for bias in (2, 0):  # 2 Б0.1 с s2 val-года предыдущего шага (вне выборки; на шаге 1 F=1); 0 без поправки
                lg = []
                res = compute(hv, ht, h, bx, bl, bias, lag, lg)
                if bias == 2:
                    etalon = [{k: r['base'][k] for k in ('q_har', 'q_xgb_single', 'q_lgb_single')} for r in res]
                else:
                    for r, e in zip(res, etalon):
                        r['base'].update(e)  # q_har_raw остаётся без поправки
                if bias == 2 and var == 'single':
                    blog += lg
                for sel in ('step', 'minimax'):
                    add(*D.select(res, sel, h), f'{var}|{sel}|{"B0.1oos" if bias else "bez"}', h)
                if bias == 2:
                    for comp in ('har', 'xgb', 'lgb'):
                        add(*D.v0(res, comp, h), f'{var}|V0_{comp.upper()}|B0.1oos', h)
            if var == 'single':  # справка (разработочный набор): in-sample Б0.1 и V7 с lag=1 (исходная реализация)
                c1, c0, cl = (D.select(compute(hv, ht, h, bx, bl, b, l, []), 'step', h)[0] for b, l in ((1, lag), (0, lag), (0, 1)))
                m = lambda c, col: c.loc[c.step <= DEV, col].mean()
                ref.append(dict(h=h, hyb_B01_insample=m(c1, 'q_hyb'), har_B01_insample=m(c1, 'q_har'), hyb_bez_lag_h=m(c0, 'q_hyb'),
                                V7_lag_h=m(c0, 'q_V7_Rolling_W20'), V7_lag1_orig=m(cl, 'q_V7_Rolling_W20'), hyb_bez_lag1=m(cl, 'q_hyb'),
                                V7_steps_lag_h=str([s for s in c0.step[c0.scheme == 'V7_Rolling_W20'] if s <= DEV]),
                                V7_steps_lag1=str([s for s in cl.step[cl.scheme == 'V7_Rolling_W20'] if s <= DEV])))
        print('h', h, 'готово', flush=True)
    R = pd.concat(rows, ignore_index=True)
    hide(R, a.confirm).to_csv(out / 'designs_val.csv', index=False)
    pd.DataFrame(chk).to_csv(out / f'hyb_{a.boost}_har_check.csv', index=False)
    pd.DataFrame(blog).to_csv(out / f'bias_factors_{a.boost}.csv', index=False)
    pd.DataFrame(ref).to_csv(out / f'ref_{a.boost}.csv', index=False)
    sp = HS.run(A, harj)
    hide(sp, a.confirm).to_csv(out / f'har_specs_val_{a.boost}.csv', index=False)  # строка (в) эталона HAR-J
    code = 'single|step|bez'
    hide(R[R.design == code], a.confirm).drop(columns=['design']).to_csv(out / f'hyb_{a.boost}_choice.csv', index=False)
    for h in (1, 5, 22):
        ov, ot = outs[(code, h)]
        ov.to_parquet(out / f'hyb_{a.boost}_h{h}_val.parquet', index=False); ot.to_parquet(out / f'hyb_{a.boost}_h{h}_test.parquet', index=False)
    # правило выбора (решение 25.10): сначала прошедшие все критерии -> мин. средний val-QLIKE; иначе мин. нарушенных, затем мин. QLIKE; всё на 2016-2021
    Rd = R[R.step <= DEV]
    F = Rd.groupby(['h', 'design'], sort=False).agg(**{k: (k + '_h', 'first') for k in CR}, mean_q=('q_hyb', 'mean')).reset_index()
    F['failed'] = F.apply(lambda r: ','.join(k for k in CR if not r[k]), axis=1)
    F['n_failed'] = F.apply(lambda r: sum(not r[k] for k in CR), axis=1)
    win = F.sort_values(['h', 'n_failed', 'mean_q']).groupby('h').head(1)
    win.to_csv(out / f'final_choice_{a.boost}.csv', index=False)
    for r in win.itertuples():
        ov, ot = outs[(r.design, r.h)]
        ov.to_parquet(out / f'hyb_{a.boost}_final_h{r.h}_val.parquet', index=False); ot.to_parquet(out / f'hyb_{a.boost}_final_h{r.h}_test.parquet', index=False)
    L = RP.lines(Rd, win, ref, sp[sp.step <= DEV], a.boost, a.v7_lag, RP.counter(a.devlog, len(F), F.design.nunique()), sorted(Rd.val_year.unique()))
    if a.confirm:  # один раз, для финалистов: критерии 1-4 на шагах 7-9 (c4 внутри 7-9)
        L += ['', f'ПОДТВЕРЖДАЮЩИЙ НАБОР (val 2022-2024), финалисты: {a.finalists}', '| дизайн | h | c1 | c2 | c3 | c4 | средн. q |', '|---|---|---|---|---|---|---|']
        for nm in [x for x in a.finalists.split(',') if x]:
            for h in (1, 5, 22):
                g = chs[(nm, h)]; g = g[g.step > DEV]; fl = D.flags(g)
                L.append(f'| {nm.replace("|", "/")} | {h} | ' + ' | '.join('+' if fl[k] else '-' for k in CR) + f' | {fl["mean_q"]:.3f} |')
    D.devlog(a.devlog, '\n'.join(L))
    print(win[['h', 'design', 'failed', 'mean_q']].to_string())


if __name__ == '__main__':
    main()
