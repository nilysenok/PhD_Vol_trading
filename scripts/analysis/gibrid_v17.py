"""Гибрид V1-V7 на прогнозах бустинга + дизайны (слепой тест: метрики только на валидационных годах шагов).
python3 scripts/analysis/gibrid_v17.py --boost P1|B --out data/regen [--regen data/regen] [--v7-lag h|1] [--devlog docs/otchety/dev_log.md]
Дизайн = вариант бустинга (single | top5 | seeds3, если файлы есть) x выбор схемы (step | minimax) x поправка Б0.1 (без | с).
Вход: {regen}/{xgb|B_xgb}_h{h}_{val,test}.parquet, {P1|B_lgbm}_...; HAR-J пересчитывается (gibrid_har.py).
Выход: hyb_{boost}_h{h}_{val,test}.parquet и hyb_{boost}_choice.csv (дизайн «как в коде»: single/step/без), designs_val.csv,
hyb_{boost}_har_check.csv, hyb_{boost}_final_h{h}_{val,test}.parquet + final_choice_{boost}.csv (правило 25.10: мин. нарушенных
критериев, затем мин. средний val-QLIKE; в набор входят V0 = HAR-J / XGB / LGB с поправкой), ref_{boost}.csv, bias_factors_{boost}.csv."""
import argparse, subprocess
from pathlib import Path
import numpy as np, pandas as pd
import gibrid_har as GH
import gibrid_schemes as S
import gibrid_designs as D

SCHEMES = ['V1_Adaptive', 'V2_MultiVal', 'V3_Regime', 'V4_Shrinkage', 'V5_Combined', 'V6_TrimmedW', 'V7_Rolling_W20']
MAIN_REGEN = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')


def find(regen, name):
    for d in (regen, GH.ROOT / 'data/regen', MAIN_REGEN):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)


def _avg(df):  # ансамбль: среднее прогнозов по (date, ticker, step)
    return df.groupby(['date', 'ticker', 'step'], as_index=False).agg(
        year=('year', 'first'), rv_actual=('rv_actual', 'first'), rv_pred=('rv_pred', 'mean'))


def load_boost(boost, h, regen, variant):
    """single — одна точка; top5 — среднее топ-5 (B_*_h{h}_top5_*); seeds3 — seed 42 + B_*_h{h}_seeds4344_* (формат tune_B.py)."""
    res = []
    for n in {'P1': ('xgb', 'P1'), 'B': ('B_xgb', 'B_lgbm')}[boost]:
        d = {}
        for k in ('val', 'test'):
            if variant == 'single':
                d[k] = pd.read_parquet(find(regen, f'{n}_h{h}_{k}.parquet'))
            elif variant == 'top5':
                d[k] = _avg(pd.read_parquet(find(regen, f'{n}_h{h}_top5_{k}.parquet')))
            else:
                d[k] = _avg(pd.concat([pd.read_parquet(find(regen, f'{n}_h{h}_{k}.parquet')),
                                       pd.read_parquet(find(regen, f'{n}_h{h}_seeds4344_{k}.parquet'))]))
        res.append(d)
    return res


def merged(har, xgb, lgb):
    m = har.rename(columns={'rv_pred': 'har'})
    for n, d in (('xgb', xgb), ('lgb', lgb)):
        m = m.merge(d[['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': n}), on=['date', 'ticker'])
    return m[m['rv_actual'] > 1e-12].sort_values(['step', 'date', 'ticker']).reset_index(drop=True)  # фильтр: v3.py:65


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


def devlog(path, text):
    p = Path(path)
    n = 1 + sum(1 for l in p.read_text().splitlines() if l.startswith('| ') and l[2:3].isdigit()) if p.exists() else 1
    when = subprocess.run(['date', '+%Y-%m-%d %H:%M'], capture_output=True, text=True).stdout.strip()
    head = '' if p.exists() else '# Журнал разработки (dev_log)\n\n| № | Дата-время | Что меняли |\n|---|---|---|\n'
    with open(p, 'a') as f:
        f.write(head + f'| {n} | {when} | {text.splitlines()[0]} |\n\n' + '\n'.join(text.splitlines()[1:]) + '\n\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boost', required=True, choices=['P1', 'B'])
    ap.add_argument('--out', default='data/regen')
    ap.add_argument('--regen', default='data/regen')
    ap.add_argument('--v7-lag', default='h', help="'h' (по умолчанию) = без заглядывания в цель; 1 = как в коде")
    ap.add_argument('--devlog', default='docs/otchety/dev_log.md')
    a = ap.parse_args()
    out, regen = Path(a.out), Path(a.regen)
    out.mkdir(parents=True, exist_ok=True)
    A, harj = GH.load_all()
    variants = ['single'] + (['top5', 'seeds3'] if a.boost == 'B' else [])
    chk, rows, outs, blog, ref = [], [], {}, [], []
    CR = ('c1', 'c2', 'c3', 'c4')

    def add(ch, ov, ot, name, h):
        ch.insert(0, 'design', name)
        fl = D.flags(ch)
        ch['ratio_har'] = ch.q_hyb / ch.q_har
        for k in CR + ('worst_ratio', 'worst_ratio_single', 'max_w_jump', 'ml_switches'):
            ch[k + '_h'] = fl[k]
        rows.append(ch); outs[(name, h)] = (ov, ot)

    for h in (1, 5, 22):
        hv, ht, _ = GH.har_forecasts(h, A, harj)
        chk.append(GH.check_saved(h, ht))
        lag = h if a.v7_lag == 'h' else int(a.v7_lag)
        for var in variants:
            try:
                bx, bl = load_boost(a.boost, h, regen, var)
            except FileNotFoundError as e:
                print('нет файла, вариант', var, 'пропущен:', e, flush=True); continue
            for bias in (0, 2):  # 0 без поправки; 2 Б0.1 с s2 val-года предыдущего шага (вне выборки; на шаге 1 F=1)
                lg = []
                res = compute(hv, ht, h, bx, bl, bias, lag, lg)
                if bias == 2 and var == 'single':
                    blog += lg
                for sel in ('step', 'minimax'):
                    add(*D.select(res, sel, h), f'{var}|{sel}|{"B0.1oos" if bias else "bez"}', h)
                if bias == 2:
                    for comp in ('har', 'xgb', 'lgb'):
                        add(*D.v0(res, comp, h), f'{var}|V0_{comp.upper()}|B0.1oos', h)
            if var == 'single':  # справка: in-sample Б0.1 и V7 с lag=1 (исходная реализация)
                c1 = D.select(compute(hv, ht, h, bx, bl, 1, lag, []), 'step', h)[0]
                c0 = D.select(compute(hv, ht, h, bx, bl, 0, lag, []), 'step', h)[0]
                cl = D.select(compute(hv, ht, h, bx, bl, 0, 1, []), 'step', h)[0]
                ref.append(dict(h=h, hyb_B01_insample=c1.q_hyb.mean(), har_B01_insample=c1.q_har.mean(), hyb_bez_lag_h=c0.q_hyb.mean(),
                                V7_lag_h=c0.q_V7_Rolling_W20.mean(), V7_lag1_orig=cl.q_V7_Rolling_W20.mean(), hyb_bez_lag1=cl.q_hyb.mean(),
                                V7_steps_lag_h=str(c0.step[c0.scheme == 'V7_Rolling_W20'].tolist()),
                                V7_steps_lag1=str(cl.step[cl.scheme == 'V7_Rolling_W20'].tolist())))
        print('h', h, 'готово', flush=True)
    R = pd.concat(rows, ignore_index=True)
    R.to_csv(out / 'designs_val.csv', index=False)
    pd.DataFrame(chk).to_csv(out / f'hyb_{a.boost}_har_check.csv', index=False)
    pd.DataFrame(blog).to_csv(out / f'bias_factors_{a.boost}.csv', index=False)
    pd.DataFrame(ref).to_csv(out / f'ref_{a.boost}.csv', index=False)
    code = 'single|step|bez'
    R[R.design == code].drop(columns=['design']).to_csv(out / f'hyb_{a.boost}_choice.csv', index=False)
    for h in (1, 5, 22):
        ov, ot = outs[(code, h)]
        ov.to_parquet(out / f'hyb_{a.boost}_h{h}_val.parquet', index=False); ot.to_parquet(out / f'hyb_{a.boost}_h{h}_test.parquet', index=False)
    # правило выбора (решение 25.10): на каждом h наименьшее число нарушенных критериев, при равенстве — мин. средний val-QLIKE
    F = R.groupby(['h', 'design'], sort=False).agg(**{k: (k + '_h', 'first') for k in CR}, mean_q=('q_hyb', 'mean')).reset_index()
    F['failed'] = F.apply(lambda r: ','.join(k for k in CR if not r[k]), axis=1)
    F['n_failed'] = F.apply(lambda r: sum(not r[k] for k in CR), axis=1)
    win = F.sort_values(['h', 'n_failed', 'mean_q']).groupby('h').head(1)
    win.to_csv(out / f'final_choice_{a.boost}.csv', index=False)
    for r in win.itertuples():
        ov, ot = outs[(r.design, r.h)]
        ov.to_parquet(out / f'hyb_{a.boost}_final_h{r.h}_val.parquet', index=False); ot.to_parquet(out / f'hyb_{a.boost}_final_h{r.h}_test.parquet', index=False)
    ys = sorted(R.val_year.unique())
    L = [f'Гибрид V1-V7 + V0, дизайны на {a.boost} (gibrid_v17.py, только валидация; V7 lag={a.v7_lag}; Б0.1 только oos; c1-c2 относительно моделей с той же поправкой)',
         '| дизайн | h | ' + ' | '.join(str(y) for y in ys) + ' | средн. q | худш. отн. к HAR | c1 | c2 | c3 | c4 |', '|' + '---|' * (9 + len(ys))]
    for (d, h), g in R.groupby(['design', 'h'], sort=False):
        f = g.iloc[0]
        L.append(f'| {d.replace("|", "/")} | {h} | ' + ' | '.join(f'{r.q_hyb:.3f} ({r.ratio_har:.2f})' for r in g.itertuples())
                 + f' | {g.q_hyb.mean():.3f} | {f.worst_ratio_h:.2f} | ' + ' | '.join('+' if f[k + '_h'] else '-' for k in CR) + ' |')
    L += ['', 'Выбор по правилу (мин. нарушенных критериев, затем мин. средний val-QLIKE):', '| h | дизайн | не выполнены | средн. q |', '|---|---|---|---|']
    L += [f'| {r.h} | {r.design.replace("|", "/")} | {r.failed or "-"} | {r.mean_q:.3f} |' for r in win.itertuples()]
    L += ['', 'Справка (single/step, средний val-QLIKE по шагам):', '| h | гибрид Б0.1 in-sample | HAR Б0.1 in-sample | V7 lag=h | V7 lag=1 (исходная, с заглядыванием на h-1 дней) | шаги V7 lag=h | шаги V7 lag=1 |', '|---|---|---|---|---|---|---|']
    L += [f'| {r["h"]} | {r["hyb_B01_insample"]:.3f} | {r["har_B01_insample"]:.3f} | {r["V7_lag_h"]:.3f} | {r["V7_lag1_orig"]:.3f} | {r["V7_steps_lag_h"]} | {r["V7_steps_lag1"]} |' for r in ref]
    devlog(a.devlog, '\n'.join(L))
    print(win[['h', 'design', 'failed', 'mean_q']].to_string())


if __name__ == '__main__':
    main()
