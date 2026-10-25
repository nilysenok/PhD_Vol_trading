"""Гибрид V1-V7 на прогнозах бустинга + дизайны (слепой тест: метрики только на валидационных годах шагов).
python3 scripts/analysis/gibrid_v17.py --boost P1|B --out data/regen [--regen data/regen] [--v7-lag 1] [--devlog docs/otchety/dev_log.md]
Дизайн = вариант бустинга (single | top5 | seeds3, если файлы есть) x выбор схемы (step | minimax) x поправка Б0.1 (без | с).
Вход: {regen}/{xgb|B_xgb}_h{h}_{val,test}.parquet, {P1|B_lgbm}_...; HAR-J пересчитывается (gibrid_har.py).
Выход: hyb_{boost}_h{h}_{val,test}.parquet и hyb_{boost}_choice.csv (дизайн «как в коде»: single/step/без), designs_val.csv,
hyb_{boost}_har_check.csv, hyb_{boost}_{победитель}_h{h}_{val,test}.parquet (если есть дизайн, прошедший c1-c4)."""
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


def bias_correct(val, test, s2_har):
    """Б0.1: F=exp(s2/2); s2 HAR-J — остатки обучающего окна шага, бустинга — остатки val-года шага (ln)."""
    for st in val['step'].unique():
        vm, tm = val['step'] == st, test['step'] == st
        for c in ('xgb', 'lgb'):
            f = np.exp(np.var(np.log(val.loc[vm, 'rv_actual'].values + 1e-10) - np.log(val.loc[vm, c].values)) / 2)
            val.loc[vm, c] *= f; test.loc[tm, c] *= f
        f = np.exp(s2_har[st] / 2)
        val.loc[vm, 'har'] *= f; test.loc[tm, 'har'] *= f


def compute(hv, ht, s2h, bx, bl, bias, lag):
    """Все схемы V1-V7 на всех шагах -> список шагов (q, веса, прогнозы val/test)."""
    val, test = merged(hv, bx['val'], bl['val']), merged(ht, bx['test'], bl['test'])
    if bias:
        bias_correct(val, test, s2h)
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
                    regime=reg, regime_ratio=ratio, v1_w_har=w1[0], q_har=S.qlike(y, vs['har'].values),
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
    ap.add_argument('--v7-lag', type=int, default=1, help='1 = как в коде; h = без заглядывания в цель')
    ap.add_argument('--devlog', default='docs/otchety/dev_log.md')
    a = ap.parse_args()
    out, regen = Path(a.out), Path(a.regen)
    out.mkdir(parents=True, exist_ok=True)
    A, harj = GH.load_all()
    variants = ['single'] + (['top5', 'seeds3'] if a.boost == 'B' else [])
    chk, rows, outs = [], [], {}
    for h in (1, 5, 22):
        hv, ht, s2h = GH.har_forecasts(h, A, harj)
        chk.append(GH.check_saved(h, ht))
        for var in variants:
            try:
                bx, bl = load_boost(a.boost, h, regen, var)
            except FileNotFoundError as e:
                print('нет файла, вариант', var, 'пропущен:', e, flush=True); continue
            for bias in (0, 1):
                res = compute(hv, ht, s2h, bx, bl, bias, a.v7_lag)
                for sel in ('step', 'minimax'):
                    ch, ov, ot = D.select(res, sel, h)
                    name = f'{var}|{sel}|{"B0.1" if bias else "bez"}'
                    ch.insert(0, 'design', name)
                    fl = D.flags(ch)
                    ch['ratio_har'] = ch.q_hyb / ch.q_har
                    for k in ('c1', 'c2', 'c3', 'c4', 'worst_ratio', 'worst_ratio_single', 'max_w_jump'):
                        ch[k + '_h'] = fl[k]
                    rows.append(ch); outs[(name, h)] = (ov, ot)
            print('h', h, var, 'готово', flush=True)
    R = pd.concat(rows, ignore_index=True)
    R.to_csv(out / 'designs_val.csv', index=False)
    pd.DataFrame(chk).to_csv(out / f'hyb_{a.boost}_har_check.csv', index=False)
    code = 'single|step|bez'
    R[R.design == code].drop(columns=['design']).to_csv(out / f'hyb_{a.boost}_choice.csv', index=False)
    for h in (1, 5, 22):
        ov, ot = outs[(code, h)]
        ov.to_parquet(out / f'hyb_{a.boost}_h{h}_val.parquet', index=False); ot.to_parquet(out / f'hyb_{a.boost}_h{h}_test.parquet', index=False)
    G = R.groupby('design')
    S_ = pd.DataFrame({'mean_q': G.q_hyb.mean(), 'pass': G.apply(lambda g: all(g.groupby('h')[c + '_h'].first().all() for c in ('c1', 'c2', 'c3', 'c4')))})
    ok = S_[S_['pass']]
    win = ok.mean_q.idxmin() if len(ok) else None
    if win:
        for h in (1, 5, 22):
            ov, ot = outs[(win, h)]
            tag = win.replace('|', '_')
            ov.to_parquet(out / f'hyb_{a.boost}_{tag}_h{h}_val.parquet', index=False); ot.to_parquet(out / f'hyb_{a.boost}_{tag}_h{h}_test.parquet', index=False)
    L = [f'Гибрид V1-V7, дизайны на {a.boost} (gibrid_v17.py, только валидация; v7-lag={a.v7_lag}); победитель по c1-c4: {win}',
         '| дизайн | h | ' + ' | '.join(str(y) for y in sorted(R.val_year.unique())) + ' | средн. q | худш. отн. к HAR (одиночн.) | c1 | c2 | c3 | c4 |',
         '|' + '---|' * (9 + R.val_year.nunique())]
    for (d, h), g in R.groupby(['design', 'h'], sort=False):
        c = ' | '.join(f'{r.q_hyb:.3f} ({r.ratio_har:.2f})' for r in g.itertuples())
        f = g.iloc[0]
        L.append(f'| {d.replace("|", "/")} | {h} | {c} | {g.q_hyb.mean():.3f} | {f.worst_ratio_h:.2f} ({f.worst_ratio_single_h:.2f}) | ' + ' | '.join('+' if f[k + '_h'] else '-' for k in ('c1', 'c2', 'c3', 'c4')) + ' |')
    L.append('В ячейке: val-QLIKE (отношение к HAR-J); полная таблица — data/regen/designs_val.csv.')
    devlog(a.devlog, '\n'.join(L))
    print('победитель:', win)


if __name__ == '__main__':
    main()
