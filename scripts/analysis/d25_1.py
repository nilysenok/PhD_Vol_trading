"""Д25.1: нога RVI. y_h,i(t) = RVI²(t)·k_i(t).
RVI (close_RVI, % годовых, 30 календарных дней ~ 22 торговых) -> дневная дисперсия: v(t) = (RVI/100)²/252. Для h=22 окно RVI и цели совпадает
(30 к.д. ~ 22 т.д.), это средняя дневная дисперсия за 22 дня. Для h=5 масштаб пересчитывается через k_i, оценённый на 5-дневной цели (плоская
временная структура в пределах месяца; уровень дневной дисперсии тот же, k_i^(5) снимает разницу). Пропуски RVI — ffill по прошлому.
k_i^(h)(t) = медиана RV_i,h(τ)/v(τ) по 252 последним строкам тикера с τ ≤ t−h (цель RV_i,h(τ) = среднее rv за [τ+1, τ+h] известна к t;
эмбарго по концу цели), min 126 наблюдений; нет k -> прогноз HAR-J шага (доля строк — в лог).
Применения: (а) самостоятельная модель (нога в обоих слотах, V0_XGB гибрида); (б) компонента x10 объединителя Д22 (группа с суммой 1);
(в) нога Д19: ln y = ln(v·k) + ρ·r̂₁, ρ — OLS без константы по OOS-истории шагов < t (как Д19.1); (г) h=5.
python3 scripts/analysis/d25_1.py --build | --d22 [--seeds 42 43 44] | --leak | --eval | --yaml B... | --test B..."""
import argparse, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import gibrid_har as GH
import d19_signal as D19

R = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')
OUT, TL = R / 'D25', R / 'test_locked'
FEAT = R / 'features_pit_v3.parquet'
K = ['date', 'ticker', 'step']
WIN, MINP = 252, 126


def rvi_leg(A, h, lag=None, win=WIN):
    """Прогноз ноги RVI на каждую строку (date, ticker). lag — сдвиг (по умолчанию h: только известные цели); для подложенной утечки lag < h."""
    lag = h if lag is None else lag
    a = A[['date', 'ticker', 'close_RVI', f'rv_target_mean_h{h}']].copy().sort_values(['ticker', 'date'])
    a['v'] = (a.groupby('ticker').close_RVI.ffill() / 100) ** 2 / 252
    a['ratio'] = a[f'rv_target_mean_h{h}'] / a['v']
    a.loc[~np.isfinite(a.ratio) | (a.ratio <= 0), 'ratio'] = np.nan
    a['k'] = a.groupby('ticker').ratio.transform(lambda s: s.shift(lag).rolling(win, min_periods=MINP).median())
    a['leg'] = a.v * a.k
    return a[['date', 'ticker', 'v', 'k', 'leg']]


def frames(h, A, harj):
    """val/test-строки HAR-J шага (как в финалах) + нога RVI; пропуск k -> HAR-J."""
    hv, ht, _ = GH.har_forecasts(h, A, harj)
    L = rvi_leg(A, h)
    out = []
    for d in (hv, ht):
        d = d.rename(columns={'rv_pred': 'har'}).merge(L, on=['date', 'ticker'], how='left')
        d['nofill'] = d.leg.isna()
        d['rvi'] = d.leg.fillna(d.har)
        out.append(d)
    return out


def save(b, h, v, t, sfx=''):
    c = K + ['year', 'rv_actual', 'rv_pred']
    td = TL / f'D25_build/{b}'  # временный каталог сборки; в test_locked/D25_* переносится только у прошедших (--test)
    td.mkdir(parents=True, exist_ok=True)
    for slot in ('xgb', 'lgbm'):
        v[c].to_parquet(OUT / f'{b}{sfx}_{slot}_h{h}_val.parquet', index=False)
        if not sfx:
            t[c].to_parquet(td / f'{b}_{slot}_h{h}_test.parquet', index=False)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    A, harj = GH.load_pit(FEAT)
    hv1, _, _ = GH.har_forecasts(1, A, harj); hv1 = hv1.rename(columns={'rv_pred': 'har1'})
    t1 = pd.read_parquet(TL / 'final2/final_h1_test.parquet')[K + ['xgb_raw', 'har_raw']].rename(columns={'xgb_raw': 'hyb1', 'har_raw': 'har1'})
    rows, kk = [], []
    for h in (22, 5):
        V, T = frames(h, A, harj)
        print(f'h={h}: строк val {len(V)}, без k (-> HAR-J) {V.nofill.mean():.4f}; test без k {T.nofill.mean():.4f}', flush=True)
        kk.append(V[V.year <= 2021].groupby(['ticker', 'year']).k.median().rename(f'k_h{h}'))  # только разработочный набор
        # (а) самостоятельная нога
        save(f'D25_1a_h{h}', h, V.assign(rv_pred=V.rvi), T.assign(rv_pred=T.rvi))
        # (в) нога Д19: ln y = ln(v k) + rho r1
        for sd in ('top5', 's42', 's43', 's44'):
            W = V.merge(D19.r1_val(sd), on=K).merge(hv1[K + ['har1']], on=K)
            W['r1'] = np.log(W.hyb1) - np.log(W.har1)
            W['res'] = np.log(W.rv_actual.clip(1e-10)) - np.log(W.rvi)
            TT = T.merge(t1, on=K); TT['r1'] = np.log(TT.hyb1) - np.log(TT.har1)
            ok = W.rv_actual > 1e-12
            ols = lambda d: float((d.r1 * d.res).sum() / (d.r1 ** 2).sum())
            pv, pt = [], []
            for s in range(1, 10):
                rv_ = ols(W[ok & ((W.step < s) | ((s == 1) & (W.step == 1)))]); rt_ = ols(W[ok & (W.step <= s)])
                v = W[W.step == s]; t = TT[TT.step == s]
                pv.append(v.assign(rv_pred=np.exp(np.log(v.rvi) + rv_ * v.r1))); pt.append(t.assign(rv_pred=np.exp(np.log(t.rvi) + rt_ * t.r1)))
                rows.append(dict(h=h, seed=sd, step=s, rho_val=rv_, rho_test=rt_))
            pv, pt = pd.concat(pv), pd.concat(pt)
            sfx = '' if sd == 'top5' else f'_{sd}'
            save(f'D25_1v_h{h}', h, pv, pt, sfx)
            if h == 22:  # D25_1vx: XGB-слот = нога (в), LGB-слот = Д16.5 (как D19_1x)
                c = K + ['year', 'rv_actual', 'rv_pred']
                lv = pv[K + ['year', 'rv_actual']].merge(D19.d165_leg('val'), on=K).rename(columns={'x165': 'rv_pred'})[c]
                lt = pt[K + ['year', 'rv_actual']].merge(D19.d165_leg('test'), on=K).rename(columns={'x165': 'rv_pred'})[c]
                assert len(lv) == len(pv) and len(lt) == len(pt)
                b = 'D25_1vx_h22'
                pv[c].to_parquet(OUT / f'{b}{sfx}_xgb_h22_val.parquet', index=False); lv.to_parquet(OUT / f'{b}{sfx}_lgbm_h22_val.parquet', index=False)
                if not sfx:
                    td = TL / f'D25_build/{b}'; td.mkdir(parents=True, exist_ok=True)
                    pt[c].to_parquet(td / f'{b}_xgb_h22_test.parquet', index=False); lt.to_parquet(td / f'{b}_lgbm_h22_test.parquet', index=False)
    pd.DataFrame(rows).to_csv(OUT / 'D25_1_rho.csv', index=False)
    pd.concat(kk, axis=1).to_csv(OUT / 'D25_1_k_by_year.csv')
    print(pd.DataFrame(rows).query("seed=='top5'").round(3).to_string())


# ---------------------------------------------------------------- (б) компонента x10 объединителя Д22
def patch_d22(hs=(22, 5)):
    """Подключает x10 к d22_full: нога RVI считается из той матрицы, которую получает assemble (в детекторе утечек Д22 — испорченной)."""
    import d22_full as DF
    A, harj = GH.load_pit(FEAT)
    base = {h: rvi_leg(A, h)[['date', 'ticker', 'leg']] for h in hs}
    orig_asm, orig_xs = DF.assemble, DF.xs_for
    state = {'on': False, 'h': None, 'cache': {}}

    def leg_for(h, A_):
        if not {'close_RVI', f'rv_target_mean_h{h}'} <= set(A_.columns):
            return base[h]
        key = (h, id(A_))
        if key not in state['cache']:
            state['cache'][key] = rvi_leg(A_, h)[['date', 'ticker', 'leg']]
        return state['cache'][key]

    def asm(h, kind, seed, I, A_, hist, no_h1=False):
        d = orig_asm(h, kind, seed, I, A_, hist, no_h1)
        if state['on']:
            d = d.merge(leg_for(h, A_).rename(columns={'leg': 'p10'}), on=['date', 'ticker'], how='left')
            d['x10'] = np.log(d.pop('p10')); d['x10'] = d['x10'].fillna(d['x1'])
        return d

    def xs(h, k, drop=(), em=0):
        x = orig_xs(h, k, drop, em)
        return x + ['x10'] if state['on'] else x
    DF.assemble, DF.xs_for = asm, xs
    DF.GROUP = DF.GROUP + ('x10',)
    return DF, state


def leak_b():
    """Детектор утечек Д22 (d22_full.leak) на объединителе с x10: будущее (date > t) в матрице и входах заменяется шумом -> прогноз до t
    совпадает; plant=True (ln RV(t+1) в x6) детектор обязан поймать."""
    DF, state = patch_d22((5,))
    state['on'] = True
    a = DF.leak(h_list=(5,)); b = DF.leak(h_list=(5,), plant=True)
    assert (a.mismatch == 0).all(), a
    assert (b[b.kind == 'val'].mismatch > 0).any(), b
    txt = pd.concat([a, b]).to_string()
    open(OUT / 'leak_d25_1b.txt', 'w').write('Д25.1(б): объединитель Д22 + x10, h=5, seed 42\n' + txt + '\nбез подлога: mismatch = 0: OK; подлог пойман: OK\n'); print(txt)


def d22(seeds, hs=(22, 5)):
    DF, state = patch_d22(hs)
    I_v, I_t = DF.load_inputs('val'), DF.load_inputs('test')
    rows = []
    for h in hs:
        code = DF.CODE[h]
        for sd in seeds:
            for on in (False, True):
                state['on'] = on
                fv, ft, B, _ = DF.pipeline(h, sd, I_v, I_t)
                nm = f'D25_1b_h{h}' if on else f'D22ref_h{h}'
                fv.to_parquet(OUT / f'{nm}_s{sd}_final_val.parquet', index=False)
                if on:
                    td = TL / 'D25_build' / f'D25_1b_h{h}'; td.mkdir(parents=True, exist_ok=True)
                    ft.to_parquet(td / f'D25_1b_h{h}_s{sd}_final_test.parquet', index=False)
                    B.assign(seed=sd).to_csv(OUT / f'D25_1b_h{h}_s{sd}_beta.csv', index=False)
                # метрики только шагов 2-6 (year_q: lo=2, hi=6)
                for cd in (code, 'sm'):
                    rows.append(dict(h=h, seed=sd, s2=cd, **DF.summarize(nm, fv, cd)))
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(OUT / 'D25_1b_eval.csv', index=False)
    print(pd.DataFrame(rows).to_string())


def export_b(hs=(5, 22)):
    """D25_1b в формате gibrid_v17 (как d22_full.export): одиночная модель в обоих слотах, rv_pred — без s² (поправку делает гибрид),
    val шагов 1–9: D25/D25_1b_h{h}_{m}_h{h}[_seeds4344]_val; тест шагов 1–9 — test_locked/D25_build/D25_1b_h{h}/ (без метрик)."""
    cols = ['date', 'ticker', 'year', 'step', 'seed', 'rank', 'rv_actual', 'rv_pred']
    for h in hs:
        tag = f'D25_1b_h{h}'; td = TL / 'D25_build' / tag
        V = pd.concat([pd.read_parquet(OUT / f'{tag}_s{sd}_final_val.parquet').assign(seed=sd, rank=1)[cols] for sd in (42, 43, 44)])
        T = pd.concat([pd.read_parquet(td / f'{tag}_s{sd}_final_test.parquet').assign(seed=sd, rank=1)[cols] for sd in (42, 43, 44)])
        for m in ('xgb', 'lgbm'):
            for sfx, sel in (('', [42]), ('_seeds4344', [43, 44])):
                V[(V.step <= 9) & V.seed.isin(sel)].to_parquet(OUT / f'{tag}_{m}_h{h}{sfx}_val.parquet', index=False)
                T[(T.step <= 9) & T.seed.isin(sel)].to_parquet(td / f'{tag}_{m}_h{h}{sfx}_test.parquet', index=False)
        print(tag, len(V), len(T))


# ---------------------------------------------------------------- утечки
def leak():
    rng = np.random.default_rng(0)
    A, harj = GH.load_pit(FEAT)
    res = []
    for h in (22, 5):
        L0 = rvi_leg(A, h)
        for cut in ('2018-06-29', '2020-03-02', '2021-06-30'):
            c = pd.Timestamp(cut)
            B = A.copy().sort_values(['ticker', 'date'])
            # будущее к дате c: всё после c, плюс цели строк, чей конец цели > c (последние h строк тикера до c включительно)
            fut = B.date > c
            B.loc[fut, ['close_RVI', 'rv_target_mean_h22', 'rv_target_mean_h5']] *= rng.uniform(0.2, 5, size=(fut.sum(), 3))
            tail = B[B.date <= c].groupby('ticker').tail(h).index
            B.loc[tail, f'rv_target_mean_h{h}'] *= rng.uniform(0.2, 5, size=len(tail))
            L1 = rvi_leg(B, h)
            m = L0.merge(L1, on=['date', 'ticker'], suffixes=('', '_c'))
            m = m[m.date <= c]
            assert np.allclose(m.leg.fillna(-1), m.leg_c.fillna(-1), rtol=0, atol=0), (h, cut)
            # подложенная утечка: k без эмбарго (сдвиг 0: цели последних h строк ещё не закончились) — должна ловиться.
            # Сдвиг h-1 (одна незаконченная цель) медиана по 252 точкам часто не замечает — поэтому подкладывается сдвиг 0.
            P0, P1 = rvi_leg(A, h, lag=0), rvi_leg(B, h, lag=0)
            pm = P0.merge(P1, on=['date', 'ticker'], suffixes=('', '_c')); pm = pm[pm.date <= c]
            caught = not np.allclose(pm.leg.fillna(-1), pm.leg_c.fillna(-1), rtol=0, atol=0)
            assert caught, ('подложенная утечка не поймана', h, cut)
            res.append(f'h={h}, срез {cut}: нога RVI до среза не меняется при порче будущего (RVI, цели после среза и цели, не закончившиеся к срезу): OK; '
                       f'подложенная утечка (k без эмбарго, сдвиг 0) поймана: OK')
    # rho ноги (в): оценка val шага s не меняется при порче r1/остатков шагов >= s; rho_val[s] == rho_test[s-1]
    q = pd.read_csv(OUT / 'D25_1_rho.csv')
    for (h, sd), g in q.groupby(['h', 'seed']):
        g = g.sort_values('step')
        assert np.allclose(g.rho_val.values[1:], g.rho_test.values[:-1]), (h, sd)
    res.append('(в) rho для val шага s совпадает с оценкой по шагам <= s-1 (OLS без шага s и дальше), все h и seed: OK')
    hv1 = D19.r1_val('top5'); x = pd.read_parquet(R / 'D16_5_h22_xgb_h22_val.parquet')
    assert (x.year == 2015 + x.step).all()
    res.append('r̂₁ и нога Д16.5 (D25_1vx): val — год шага моделью шага (year == 2015+step): OK')
    open(OUT / 'leak_d25_1.txt', 'w').write('\n'.join(res) + '\n'); print('\n'.join(res))


# ---------------------------------------------------------------- оценка
INC = {5: 0.155332001485612, 22: 0.140402478566046}  # 2021 текущего лучшего кругов 1-2 (F2m_h5, D12_h22)


def evaluate():
    import final3 as F3
    rows = []
    for d in sorted((OUT / 'gibrid').iterdir()):
        if not d.is_dir() or not d.name.startswith('D25_1'):
            continue
        b = d.name; h = int(b.split('_h')[1].split('_')[0])
        D = F3.designs(dict(gib_dir=d, gboost=b, h=h))
        if D is None:
            continue
        S = F3.summarize(dict(), D)
        for tag, x in (('лучший', S.iloc[0]), ('V0_XGB', S[S.design.str.contains(r'\|V0_XGB\|')].iloc[0])):
            rows.append(dict(boost=b, h=h, kind=tag, design=x.design, n_failed=x.n_failed, failed=x.failed, mean_1720=x.mean_1720,
                             q2021=x.q2021, mean_1721=x.mean_1721, har_1720=x.har_1720, pass_2021=bool(x.q2021 <= INC[h]),
                             **{f'q_{y}': x[f'q_{y}'] for y in range(2017, 2022)}))
    E = pd.DataFrame(rows); E.to_csv(OUT / 'D25_1_eval.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(E[['boost', 'kind', 'design', 'n_failed', 'failed', 'mean_1720', 'q2021', 'pass_2021']].round(4).to_string())


ETAL = {22: (0.2357, 0.1453), 5: (0.2410, 0.1727)}  # HAR-J с поправкой sm (PR #102): 2017-2020 / 2021


def compare():
    """Вариант против той же конструкции без рычага, по seed 42/43/44 (seed — вариант hyb₁ в ноге Д19; у ноги (а) seed нет).
    Принятие: среднее по трём seed лучше на >= 1% И лучше на каждом seed (2017-2020), 2021 <= текущего лучшего кругов 1-2."""
    E = pd.read_csv(OUT / 'D25_1_eval.csv')
    B = pd.read_csv(R / 'D19' / 'eval_D19.csv')
    f = lambda x: f'{x:.4f}'.replace('.', ',')
    out = []
    for h in (22, 5):
        for kind in ('V0_XGB', 'лучший'):
            x = E[(E.boost == f'D25_1a_h{h}') & (E.kind == kind)].iloc[0]
            d = x.mean_1720 / ETAL[h][0] - 1
            out.append(dict(var=f'D25_1a_h{h} ({"(А) самост." if kind == "V0_XGB" else "(Б) гибрид с HAR-J"})', h=h, design=x.design, m1720=f(x.mean_1720),
                            q2021=f(x.q2021), seeds='нет seed', delta=f'{100 * d:+.1f}% к эталону HAR-J sm ({f(ETAL[h][0])})', crit=f'{x.n_failed} ({x.failed})',
                            ok=bool(d <= -0.01 and x.q2021 <= INC[h])))
        pairs = [(f'D25_1v_h{h}', f'D19_1_h{h}')] + ([('D25_1vx_h22', 'D19_1x_h22')] if h == 22 else [])
        for mine, base in pairs:
            for kind in ('V0_XGB', 'лучший'):
                m = [E[(E.boost == f'{mine}_s{s}') & (E.kind == kind)].iloc[0] for s in (42, 43, 44)]
                b = [B[(B.boost == f'{base}_s{s}') & (B.kind == kind)].iloc[0] for s in (42, 43, 44)]
                dl = [mm.mean_1720 / bb.mean_1720 - 1 for mm, bb in zip(m, b)]
                d = np.mean([mm.mean_1720 for mm in m]) / np.mean([bb.mean_1720 for bb in b]) - 1
                x = E[(E.boost == mine) & (E.kind == kind)].iloc[0]
                ok = bool(d <= -0.01 and all(v < 0 for v in dl) and all(mm.q2021 <= INC[h] for mm in m))
                out.append(dict(var=f'{mine} ({"(А) одиночная нога" if kind == "V0_XGB" else "(Б) гибрид"})', h=h, design=x.design, m1720=' / '.join(f(mm.mean_1720) for mm in m),
                                q2021=' / '.join(f(mm.q2021) for mm in m), seeds=' / '.join(f'{100 * v:+.1f}%' for v in dl),
                                delta=f'{100 * d:+.1f}% к {base} ({" / ".join(f(bb.mean_1720) for bb in b)})', crit=' / '.join(f'{mm.n_failed} ({mm.failed})' for mm in m), ok=ok))
    C = pd.DataFrame(out); C.to_csv(OUT / 'D25_1_compare.csv', index=False)
    md = ['| вариант | h | дизайн (основной) | 2017–2020 (s42/43/44) | 2021 (s42/43/44) | Δ по seed | Δ среднего | нарушено c1–c4 | итог |', '|' + '---|' * 9]
    for r in out:
        r = {**r, 'seeds': r['seeds'].replace('.', ','), 'delta': r['delta'].replace('.', ',', 1)}
        md.append(f"| {r['var']} | {r['h']} | {r['design'].replace('|', '/')} | {r['m1720']} | {r['q2021']} | {r['seeds']} | {r['delta']} | {r['crit'].replace('nan', '—')} | {'принят' if r['ok'] else 'нет'} |")
    open(OUT / 'D25_1_table.md', 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))


YAML_B = """boost: D25_1b_h5
h: 5
semeystvo: "объединитель Д22 (слои 3–4) + нога RVI (Д25.1)"
rychag: Д25.1(б)
opisanie: "объединитель Д22 h=5 (слои 1–4, как d22_full) с компонентой x10 = ln(RVI²·k_i) в группе с суммой весов 1; k_i — медиана RV_i,5/RVI² по 252 строкам тикера с эмбарго 5 дней; одиночная модель в обоих слотах, rv_pred без s²"
regen: data/regen/D25
legs: [single, seeds3]
odinochnaya: false
chistyy: false
test_dir: data/regen/test_locked/D25_1b_h5
osnova: D22_h5
n_variantov: 6
"""


def write_yaml(names):
    d = R / 'final3' / 'kandidaty'
    for b in names:
        if b == 'D25_1b_h5':
            (d / f'{b}.yaml').write_text(YAML_B); print('yaml:', d / f'{b}.yaml'); continue
        h = int(b.split('_h')[1].split('_')[0])
        txt = (f'boost: {b}\nh: {h}\nsemeystvo: "нога RVI (Д25.1)"\nrychag: Д25.1\nopisanie: "нога RVI: y = RVI²·k_i, k_i — медиана RV_i,h/RVI² по 252 дням с эмбарго h"\n'
               f'regen: data/regen/D25\nlegs: [single]\nodinochnaya: false\nchistyy: false\ntest_dir: data/regen/test_locked/D25_1\nn_variantov: 4\n')
        (d / f'{b}.yaml').write_text(txt); print('yaml:', d / f'{b}.yaml')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true'); ap.add_argument('--d22', action='store_true'); ap.add_argument('--leak', action='store_true')
    ap.add_argument('--eval', action='store_true'); ap.add_argument('--yaml', nargs='*'); ap.add_argument('--seeds', type=int, nargs='*', default=[42, 43, 44])
    ap.add_argument('--hs', type=int, nargs='*', default=[22, 5]); ap.add_argument('--export-b', action='store_true')
    ap.add_argument('--leak-b', action='store_true')
    a = ap.parse_args()
    if a.build:
        build()
    if a.d22:
        d22(a.seeds, a.hs)
    if a.export_b:
        export_b()
    if a.leak:
        leak()
    if a.leak_b:
        leak_b()
    if a.eval:
        evaluate(); compare()
    if a.yaml:
        write_yaml(a.yaml)


if __name__ == '__main__':
    main()
