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
    if a.boost not in ('P1', 'B'):
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

    for var in a.legs or (['single'] + (['top5', 'seeds3'] if a.boost != 'P1' else [])):
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
    if a.boost in ('P1', 'B'):
        pd.DataFrame([GH.check_saved(h, ht)]).to_csv(out / f'hyb_{a.boost}_har_check_h{h}.csv', index=False)
    hide(R, False).to_csv(out / f'designs_val_{a.boost}_h{h}.csv', index=False)  # метрики 7-9 - только у проверенных (confirm_*.csv)
    pd.DataFrame(blog).to_csv(out / f'bias_factors_{a.boost}_h{h}.csv', index=False)
    pd.DataFrame(ref).to_csv(out / f'ref_{a.boost}_h{h}.csv', index=False)
    hide(sp, False).to_csv(out / f'har_specs_val_{a.boost}_h{h}.csv', index=False)
    code = 'single|step|bez|L0' + ('' if 'rv' in a.blend else '|' + a.blend[0])
    hide(R[R.design == code], False).drop(columns=['design']).to_csv(out / f'hyb_{a.boost}_h{h}_choice.csv', index=False)
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


def finalist_list(spec, p):
    """'auto' или 'h1:дизайн,h5:дизайн,...' (разделитель полей дизайна | или /); порядок - по разработочному рейтингу."""
    rank = p['F'].design.tolist()
    if spec == 'auto':
        fin = p['auto']
    else:
        fin = [x.split(':', 1)[1].replace('/', '|') for x in spec.split(',') if x.startswith(f'h{p["h"]}:')]
    return sorted(fin, key=rank.index)


def build_test(h, A, harj, a, regen):
    """Тестовые прогнозы одного дизайна (шаги 1-9, тест 2017-2025) из test_locked - БЕЗ МЕТРИК: QLIKE/RMSE/MAE на тесте здесь не считаются.
    Столбцы: date, ticker, year, step, rv_actual, rv_pred (дизайн), har_raw (HAR-J как в диссертации), har_corr (HAR-J с Б0.1 той же оценки;
    для дизайна без поправки - ord), har_ord; design, scheme_test (схема шага)."""
    nm = a.build_test
    if ':' in nm:
        nm = dict(x.split(':', 1) for x in nm.split(','))[f'h{h}']
    nm = nm.replace('/', '|')
    var, sel, lab, lam, *bl = nm.split('|')
    S.MODE = bl[0] if bl else 'rv'
    code = 'bez' if lab == 'bez' else lab[2:]
    hv, ht, _ = GH.har_forecasts(h, A, harj)
    ext = dict(hcj=HC.forecasts(h, A), state=HC.state(A))
    bx, bb = load_boost(a.boost, h, regen, var, test=True)
    lag = h if a.v7_lag == 'h' else int(a.v7_lag)
    res = compute(hv, ht, h, bx, bb, code, lag, [], ext)
    rs = D.shrink(res, float(lam[1:]))
    ch, ov, ot = D.v0(rs, sel[3:].lower(), h) if sel.startswith('V0_') else D.select(rs, sel, h, a.dev_eval)
    key = ['date', 'ticker', 'step']
    rord = compute(hv, ht, h, bx, bb, 'ord', lag, [], ext)
    hc = pd.concat([r['ts'][key + ['har', 'xgb', 'lgb']] for r in (res if code != 'bez' else rord)]).rename(
        columns={'har': 'har_corr', 'xgb': 'xgb_corr', 'lgb': 'lgb_corr'})
    for nmc, df in (('xgb_raw', bx['test']), ('lgb_raw', bb['test'])):  # компоненты ноги без поправки (для табл. 3.4/3.6/3.7)
        hc = hc.merge(df[key + ['rv_pred']].rename(columns={'rv_pred': nmc}), on=key, how='left')
    ho = pd.concat([r['ts'][key + ['har']] for r in rord]).rename(columns={'har': 'har_ord'})
    raw = ht[key + ['rv_pred']].rename(columns={'rv_pred': 'har_raw'})
    o = ot.merge(raw, on=key, how='left').merge(hc, on=key, how='left').merge(ho, on=key, how='left')
    sch = dict(zip(ch.step, ch['scheme_test'] if 'scheme_test' in ch else ch['scheme']))
    o = o.assign(design=nm, h=h, scheme_test=o.step.map(sch))
    assert o[['rv_pred', 'har_raw', 'har_corr']].notna().all().all() and len(o) == len(ot)
    d = Path(a.test_out or MAIN_REGEN / 'test_locked' / 'final')
    d.mkdir(parents=True, exist_ok=True)
    o.to_parquet(d / f'final_h{h}_test.parquet', index=False)
    print(f'h={h}: {nm} -> {d / f"final_h{h}_test.parquet"} ({len(o)} строк, шаги {sorted(o.step.unique())}); метрики не считались', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boost', required=True, help='P1 | B | Bv2 | Bv2b | L2_h22 | L2_h5 (все, кроме P1 и B, — матрица features_pit_v2, тест закрыт)')
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
    ap.add_argument('--legs', nargs='*', default=None, help='ноги бустинга: single top5 seeds3 wrap (wrap - обёртка уровня 2, h=5)')
    ap.add_argument('--build-test', default='', help='"h1:дизайн,h5:дизайн" или имя: собрать тестовые прогнозы (test_locked, без метрик)')
    ap.add_argument('--test-out', default=None, help='каталог (по умолчанию <общий data/regen>/test_locked/final/)')
    ap.add_argument('--confirm-next', type=int, default=25, help='сколько следующих по рейтингу проверять, если не прошёл ни один финалист (второй круг: 0)')
    ap.add_argument('--finalists', default='', help='auto или имена дизайнов через запятую, напр. "single|step|B-ord|L0,single|V0_HAR|B-ord|L0"')
    a = ap.parse_args()
    global LO
    LO = 2 if a.dev_eval == 'oos' else 1
    out, regen = Path(a.out or MAIN_REGEN / 'gibrid_d8' / a.boost), Path(a.regen)
    out.mkdir(parents=True, exist_ok=True)
    A, harj = GH.load_pit(a.features or MAIN_REGEN / 'features_pit_v2.parquet') if a.boost not in ('P1', 'B') else GH.load_all()
    A = HC.add_cj(A)  # Д8.6: компоненты HAR-CJ и признаки состояния (Д8.7)
    if a.build_test:
        for h in a.h:
            build_test(h, A, harj, a, regen)
        return
    P = [p for p in (run_h(h, A, harj, a, out, regen) for h in a.h) if p]
    if not P:
        return
    cnt = (RP.counter(a.devlog, sum(len(p['F']) for p in P), max(len(p['F']) for p in P)) if not a.confirm else
           'Подтверждающий прогон: дизайны не перебирались заново, разработочный рейтинг тот же (счётчик дизайнов не увеличивается)')
    L = RP.lines(P, a.boost, a.v7_lag, cnt, a.etalon, a.blend, a.dev_eval)
    if a.boost not in ('P1', 'B'):
        L += V2.v2_summary(P)
    if a.confirm:  # один раз; критерии 1-4 на шагах 7-9 (c4 внутри 7-9). М4: финальный = первый по правилу из прошедших финалистов;
        # никто из финалистов -> следующие по разработочному рейтингу (до 25), стоп на первом прошедшем; никто -> первый финалист с пометкой
        L += ['', 'ПОДТВЕРЖДАЮЩИЙ НАБОР (val 2022-2024, шаги 7-9, вне выборки): финалисты в порядке правила; если никто не прошёл - следующие по рейтингу до первого прошедшего (до 25)',
              '| h | ранг разраб. | дизайн | роль | c1 | c2 | c3 | c4 | 2022 | 2023 | 2024 | средн. q | итог |', '|' + '---|' * 13]
        for p in P:
            rank = p['F'].design.tolist()
            fin = finalist_list(a.finalists, p)
            rows, chosen, mark = [], None, ''

            def chk(nm, role):
                g = p['chs'][nm]; g7 = g[g.step > DEV]; fl = D.flags(g7); ok = all(fl[k] for k in CR)
                rows.append(dict(h=p['h'], dev_rank=rank.index(nm) + 1, design=nm, role=role, **{k: fl[k] for k in CR}, ok=ok, mean_q_confirm=fl['mean_q'],
                                 **{f'q_{int(y)}': q for y, q in zip(g7.val_year, g7.q_hyb)}, worst_ratio=fl['worst_ratio'],
                                 worst_ratio_single=fl['worst_ratio_single'], max_w_jump=fl['max_w_jump'],
                                 **{f'ref_{c}_{int(y)}': v for c in ('q_har', 'q_xgb_single', 'q_lgb_single') for y, v in zip(g7.val_year, g7[c])}))
                L.append(f'| {p["h"]} | {rank.index(nm) + 1} | {nm.replace("|", "/")} | {role} | ' + ' | '.join('+' if fl[k] else '-' for k in CR) + ' | '
                         + ' | '.join(f'{q:.3f}' for q in g7.q_hyb) + f' | {fl["mean_q"]:.3f} | {"прошёл" if ok else "не прошёл"} |')
                return ok
            oks = [nm for nm in fin if chk(nm, 'финалист')]
            if oks:
                chosen = oks[0]
            else:
                for nm in [d for d in rank if d not in fin][:a.confirm_next]:
                    if chk(nm, 'следующий по рейтингу'):
                        chosen = nm; break
            if chosen is None:
                chosen, mark = fin[0], 'не прошёл подтверждающий набор'
            L.append(f'| {p["h"]} | {rank.index(chosen) + 1} | **финальный: {chosen.replace("|", "/")}** | | | | | | | | | | {mark or "прошёл"} |')
            pd.DataFrame(rows).to_csv(out / f'confirm_{a.boost}_h{p["h"]}.csv', index=False)
            pd.DataFrame([dict(h=p['h'], design=chosen, mark=mark, dev_rank=rank.index(chosen) + 1)]).to_csv(out / f'final_confirmed_{a.boost}_h{p["h"]}.csv', index=False)
    D.devlog(a.devlog, '\n'.join(L))
    print(pd.concat([p['win'] for p in P])[['h', 'design', 'failed', 'mean_q']].to_string())


if __name__ == '__main__':
    main()
