"""Сводка B v2 для dev_log (только разработочный набор, шаги 2-6; тест закрыт; подтверждающий набор не открывается):
(1) val-QLIKE XGB/LGBM по ногам (без поправки и с Б0.1 ord) против трёх строк HAR-J и множители exp(s2/2) четырёх оценок s2;
(2) ворота бустинга (план, уточнение до конца B): h=5 - лучший бустинг с поправкой не хуже HAR-J с поправкой, h=22 - не хуже более
чем на 5%, и финальным на h не выбран V0 = HAR-J (поправка ord у обеих сторон; лучший - по ногам single/top5/seeds3 и XGB/LGBM);
(3) выбор по правилу и невыполненные критерии; (4) финалисты auto (лучшие в разных семействах, до 5)."""
import pandas as pd

GATE = {5: 1.0, 22: 1.05}


def v2_summary(P):
    L = ['', 'B v2 (features_pit_v2): сводка валидации (шаги 2-6, val 2017-2021; тест закрыт, подтверждающий набор не открывался)']
    for p in P:
        Rd, h = p['Rd'], p['h']
        ys = sorted(Rd.val_year.unique())
        q = lambda d: Rd[Rd.design == d].sort_values('step').q_hyb.values
        legs = [v for v in ('single', 'top5', 'seeds3') if (Rd.design == f'{v}|V0_XGB|bez|L0').any()]
        a = Rd[Rd.design == 'single|V0_HAR|B-ord|L0'].sort_values('step').q_har_raw.values
        rows = [('HAR-J (а) как в диссертации', a), ('HAR-J (б) с поправкой ord - эталон', q('single|V0_HAR|B-ord|L0')),
                ('HAR-J (в) лучшая спецификация шага', p['sp'].sort_values('step').q_chosen.values),
                ('HAR-CJ с поправкой ord', q('single|V0_HCJ|B-ord|L0'))]
        boost = {}
        for v in legs:
            for c, nm in (('XGB', 'XGBoost'), ('LGB', 'LightGBM')):
                rows.append((f'{nm} {v} без поправки', q(f'{v}|V0_{c}|bez|L0')))
                rows.append((f'{nm} {v} с поправкой ord', q(f'{v}|V0_{c}|B-ord|L0')))
                boost[f'{nm} {v}'] = rows[-1][1].mean()
        L += ['', f'h={h}: val-QLIKE', '| модель | ' + ' | '.join(map(str, ys)) + ' | средн. |', '|' + '---|' * (2 + len(ys))]
        L += [f'| {n} | ' + ' | '.join(f'{x:.3f}' for x in v) + f' | {v.mean():.3f} |' for n, v in rows]
        bf = pd.DataFrame(p['blog'])
        bf = bf[(bf.step >= 2) & (bf.step <= 6)]
        L += ['', f'h={h}: множители F = exp(s2/2) (одна точка, оценка шага; среднее [мин-макс] по шагам 2-6)', '| оценка | HAR-J | XGB | LGBM | HAR-CJ |', '|---|---|---|---|---|']
        for code, g in bf.groupby('code', sort=False):
            f = {c: gc.F_own for c, gc in g.groupby('comp')}
            L.append(f'| {code} | ' + ' | '.join(f'{f[c].mean():.2f} [{f[c].min():.2f}-{f[c].max():.2f}]' for c in ('har', 'xgb', 'lgb', 'hcj')) + ' |')
        har, (bn, bq) = rows[1][1].mean(), min(boost.items(), key=lambda kv: kv[1])
        v0 = p['win'].family.iloc[0] == 'V0_HAR'
        thr = GATE.get(h)
        verdict = ('пройдены' if bq <= thr * har and not v0 else 'не пройдены') if thr else 'правила ворот для h=1 нет (справка)'
        L.append(f'h={h}: ворота бустинга - лучший бустинг {bn} {bq:.3f} против HAR-J (б) {har:.3f} (отношение {bq / har:.3f}'
                 + (f', порог {thr:.2f}' if thr else '') + f'); финальный V0=HAR-J: {"да" if v0 else "нет"} -> {verdict}')
        w = p['win'].iloc[0]
        L.append(f'h={h}: выбор по правилу - {w.design.replace("|", "/")}, не выполнены: {w.failed or "-"}, средн. q {w.mean_q:.3f}; '
                 f'финалисты (разные семейства): ' + '; '.join(d.replace('|', '/') for d in p['auto']))
    return L
