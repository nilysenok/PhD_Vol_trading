"""Гибрид V1-V7 на прогнозах бустинга + пространство дизайнов (слепой тест: метрики только на разработочном наборе val 2016-2021).
python3 -W ignore scripts/analysis/gibrid_v17.py --boost P1|B --out data/regen [--regen <каталог файлов бустинга>] [--h 22] [--v7-lag h|1]
    [--devlog docs/otchety/dev_log.md] [--etalon same|ord] [--confirm --finalists auto|"дизайн1,дизайн2"]
    [--blend rv log]
Дизайн = бустинг-нога (single | top5 | seeds3) | выбор схемы (step | minimax | g3) | поправка (bez | B-ord | B-rob | B-sm | B-smrob) | усадка L0/L0.25/L0.5
[| log - смешивание в логах, Д8.1]; V0 = HAR-J / XGB / LGB / HAR-CJ по отдельности (с каждой поправкой). g3 = веса ~ 1/val-QLIKE трёх компонент (Г3).
Набор схем (Д8, не больше 15) - gibrid_compute.py; V1-V7 - gibrid_schemes.py; Д8.2-Д8.7 - gibrid_d8.py, gibrid_harcj.py.
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
import gibrid_v2sum as V2
import gibrid_harcj as HC
from gibrid_compute import compute
from gibrid_io import load_boost, MAIN_REGEN

DEV = 6  # разработочный набор: шаги 1-6 (val 2016-2021); подтверждающий: шаги 7-9 (val 2022-2024), скрыт без --confirm
LO = 2  # --dev-eval oos (по умолчанию): метрика вне выборки, шаг 1 исключён -> шаги 2-6 (val 2017-2021); --dev-eval in: LO = 1


def dev(df):
    return df[(df.step >= LO) & (df.step <= DEV)]
CR = ('c1', 'c2', 'c3', 'c4')
LAMS = (0.0, 0.25, 0.5)


def hide(df, confirm):
    """Метрики подтверждающего набора (q_*, ratio_har) обнуляются, если не --confirm. Схемы/веса по шагам остаются (нужны для теста)."""
    if confirm:
        return df
    df = df.copy()
    df.loc[df.step > DEV, [c for c in df.columns if c.startswith(('q_', 'qi_')) or c == 'ratio_har']] = np.nan
    return df


def run_h(h, A, harj, a, out, regen):
    hv, ht, _ = GH.har_forecasts(h, A, harj)
    if a.boost == 'Bv2':
        ht = ht.iloc[:0]  # тест B v2 закрыт: тестовые прогнозы не строятся
    ext = dict(hcj=HC.forecasts(h, A), state=HC.state(A))  # Д8.6 HAR-CJ и признаки состояния для гейта Д8.7
    lag = h if a.v7_lag == 'h' else int(a.v7_lag)
    rows, outs, chs, blog, ref = [], {}, {}, [], []

    def add(ch, ov, ot, name):
        ch.insert(0, 'design', name)
        fl = D.flags(dev(ch))  # критерии - только на разработочном наборе (шаги LO..6)
        ch['ratio_har'] = ch.q_hyb / ch.q_har
        for k in CR + ('worst_ratio', 'worst_ratio_single', 'max_w_jump', 'ml_switches'):
            ch[k + '_h'] = fl[k]
        rows.append(ch); outs[name] = (ov, ot); chs[name] = ch

    for var in ['single'] + (['top5', 'seeds3'] if a.boost in ('B', 'Bv2') else []):
        try:
            bx, bl = load_boost(a.boost, h, regen, var)
        except FileNotFoundError as e:
            print('нет файла, вариант', var, 'пропущен:', e, flush=True); continue
        for bm in a.blend:  # Д8.1: rv - арифметическое смешивание по RV (как в коде), log - геометрическое; имя дизайна с суффиксом |log
            S.MODE, sfx = bm, '' if bm == 'rv' else '|' + bm
            etalon = None  # c1-c3 - одиночные модели с Б0.1 ord (--etalon ord) или с той же оценкой, что у дизайна (--etalon same)
            for code in ('ord', 'rob', 'sm', 'smrob', 'bez'):
                lg = []
                res = compute(hv, ht, h, bx, bl, code, lag, lg, ext)
                if code == 'ord':
                    etalon = [{k: r['base'][k] for k in ('q_har', 'q_xgb_single', 'q_lgb_single')} for r in res]
                elif a.etalon == 'ord' or code == 'bez':
                    for r, e in zip(res, etalon):
                        r['base'].update(e)  # q_har_raw остаётся без поправки
                if var == 'single' and bm == a.blend[0]:
                    blog += lg
                lab = 'bez' if code == 'bez' else 'B-' + code
                for lam in LAMS:
                    rs = D.shrink(res, lam)
                    for sel in ('step', 'minimax', 'g3'):
                        add(*D.select(rs, sel, h, a.dev_eval), f'{var}|{sel}|{lab}|L{lam:g}{sfx}')
                if bm == a.blend[0]:  # V0 не зависит от смешивания - один раз
                    for comp in ('har', 'xgb', 'lgb', 'hcj'):
                        add(*D.v0(res, comp, h), f'{var}|V0_{comp.upper()}|{lab}|L0')
        S.MODE = a.blend[0]
        if var == 'single':  # справка (разработочный набор): in-sample Б0.1 и V7 W20 с lag=1 (исходная реализация); шаги, где V7 W20 - лучшая из V1-V7
            c1, c0, cl = (D.select(compute(hv, ht, h, bx, bl, b, l, [], ext), 'step', h, a.dev_eval)[0] for b, l in (('ord_in', lag), ('bez', lag), ('bez', 1)))
            m = lambda c, col: dev(c)[col].mean()
            old = ['q_V1_Adaptive', 'q_V2_MultiVal', 'q_V3_Regime', 'q_V4_Shrinkage', 'q_V5_Combined', 'q_V6_TrimmedW', 'q_V7_Rolling_W20']
            v7s = lambda c: str([s for s, k in zip(c.step, c[old].idxmin(axis=1)) if k == 'q_V7_Rolling_W20' and LO <= s <= DEV])
            ref.append(dict(h=h, hyb_B01_insample=m(c1, 'q_hyb'), har_B01_insample=m(c1, 'q_har'), hyb_bez_lag_h=m(c0, 'q_hyb'),
                            V7_lag_h=m(c0, 'q_V7_Rolling_W20'), V7_lag1_orig=m(cl, 'q_V7_Rolling_W20'), hyb_bez_lag1=m(cl, 'q_hyb'),
                            V7_steps_lag_h=v7s(c0), V7_steps_lag1=v7s(cl)))
    if not rows:
        print('h', h, 'нет данных бустинга, пропуск'); return None
    R = pd.concat(rows, ignore_index=True)
    sp = HS.run(A, harj, (h,))
    if a.boost != 'Bv2':
        pd.DataFrame([GH.check_saved(h, ht)]).to_csv(out / f'hyb_{a.boost}_har_check_h{h}.csv', index=False)
    hide(R, a.confirm).to_csv(out / f'designs_val_{a.boost}_h{h}.csv', index=False)
    pd.DataFrame(blog).to_csv(out / f'bias_factors_{a.boost}_h{h}.csv', index=False)
    pd.DataFrame(ref).to_csv(out / f'ref_{a.boost}_h{h}.csv', index=False)
    hide(sp, a.confirm).to_csv(out / f'har_specs_val_{a.boost}_h{h}.csv', index=False)
    code = 'single|step|bez|L0' + ('' if 'rv' in a.blend else '|' + a.blend[0])
    hide(R[R.design == code], a.confirm).drop(columns=['design']).to_csv(out / f'hyb_{a.boost}_h{h}_choice.csv', index=False)
    for nm, tag in ((code, f'hyb_{a.boost}_h{h}'),):
        outs[nm][0].to_parquet(out / f'{tag}_val.parquet', index=False); outs[nm][1].to_parquet(out / f'{tag}_test.parquet', index=False)
    # правило выбора: прошедшие все критерии -> мин. средний val-QLIKE; иначе мин. нарушенных, затем мин. QLIKE; всё на 2016-2021
    Rd = dev(R)
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
    return dict(h=h, Rd=Rd, F=F, win=win, ref=ref, sp=dev(sp), chs=chs, outs=outs, auto=auto, blog=blog)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boost', required=True, choices=['P1', 'B', 'Bv2'])
    ap.add_argument('--features', default=None, help='Bv2: матрица признаков для HAR-J/HAR-CJ (по умолчанию <общий data/regen>/features_pit_v2.parquet)')
    ap.add_argument('--out', default=None, help='по умолчанию <общий data/regen>/gibrid_d8/<boost> (правило владельца 25.10: данные - только в общем data/)')
    ap.add_argument('--regen', default=str(MAIN_REGEN))
    ap.add_argument('--h', type=int, nargs='*', default=[1, 5, 22], help='горизонты (по одному: --h 22)')
    ap.add_argument('--v7-lag', default='h', help="'h' (по умолчанию) = без заглядывания в цель; 1 = как в коде")
    ap.add_argument('--blend', nargs='+', default=['rv', 'log'], choices=['rv', 'log'],
                    help='Д8.1: rv - смешивание по RV после индивидуальных поправок (как в коде), log - в логах; оба - измерение дизайна')
    ap.add_argument('--dev-eval', default='oos', choices=['oos', 'in'],
                    help='oos: метрика разработочного набора вне выборки (параметры шага t-1 -> val шага t, шаги 2-6); in - справка (шаги 1-6)')
    ap.add_argument('--etalon', default='same', choices=['ord', 'same'], help='эталон c1-c3: Б0.1 ord для всех дизайнов / та же оценка, что у дизайна')
    ap.add_argument('--devlog', default='docs/otchety/dev_log.md')
    ap.add_argument('--confirm', action='store_true', help='подтверждающий набор 2022-2024: только для --finalists, один раз')
    ap.add_argument('--finalists', default='', help='auto или имена дизайнов через запятую, напр. "single|step|B-ord|L0,single|V0_HAR|B-ord|L0"')
    a = ap.parse_args()
    global LO
    LO = 2 if a.dev_eval == 'oos' else 1
    out, regen = Path(a.out or MAIN_REGEN / 'gibrid_d8' / a.boost), Path(a.regen)
    out.mkdir(parents=True, exist_ok=True)
    A, harj = GH.load_pit(a.features or MAIN_REGEN / 'features_pit_v2.parquet') if a.boost == 'Bv2' else GH.load_all()
    A = HC.add_cj(A)  # Д8.6: компоненты HAR-CJ и признаки состояния (Д8.7)
    P = [p for p in (run_h(h, A, harj, a, out, regen) for h in a.h) if p]
    if not P:
        return
    cnt = RP.counter(a.devlog, sum(len(p['F']) for p in P), max(len(p['F']) for p in P))
    L = RP.lines(P, a.boost, a.v7_lag, cnt, a.etalon, a.blend, a.dev_eval)
    if a.boost == 'Bv2':
        L += V2.v2_summary(P)
    if a.confirm:  # один раз; критерии 1-4 на шагах 7-9 (c4 внутри 7-9). Финалист не прошёл -> следующий по правилу (М4), без перебора заново
        L += ['', 'ПОДТВЕРЖДАЮЩИЙ НАБОР (val 2022-2024): финалисты в порядке правила, затем следующие по рейтингу до первого прошедшего',
              '| h | дизайн | c1 | c2 | c3 | c4 | средн. q | итог |', '|---|---|---|---|---|---|---|---|']
        for p in P:
            rank = p['F'].design.tolist()
            fin = sorted(p['auto'] if a.finalists == 'auto' else [x for x in a.finalists.split(',') if x], key=rank.index)
            for nm in fin + [d for d in rank if d not in fin][:25]:
                g = p['chs'][nm]; fl = D.flags(g[g.step > DEV]); ok = all(fl[k] for k in CR)
                L.append(f'| {p["h"]} | {nm.replace("|", "/")} | ' + ' | '.join('+' if fl[k] else '-' for k in CR) + f' | {fl["mean_q"]:.3f} | {"ПРОШЁЛ - принят" if ok else "не прошёл"} |')
                if ok:
                    pd.DataFrame([dict(h=p['h'], design=nm, mean_q_confirm=fl['mean_q'])]).to_csv(out / f'final_confirmed_{a.boost}_h{p["h"]}.csv', index=False)
                    for part, df in zip(('val', 'test'), p['outs'][nm]):
                        df.to_parquet(out / f'hyb_{a.boost}_confirmed_h{p["h"]}_{part}.parquet', index=False)
                    break
            else:
                L.append(f'| {p["h"]} | - | | | | | | никто из проверенных не прошёл |')
    D.devlog(a.devlog, '\n'.join(L))
    print(pd.concat([p['win'] for p in P])[['h', 'design', 'failed', 'mean_q']].to_string())


if __name__ == '__main__':
    main()
