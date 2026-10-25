"""Текст итерации dev_log: только разработочный набор (val-годы 2016-2021, шаги 1-6). Метрики 2022-2024 здесь не выводятся
(их печатает только --confirm --finalists, один раз). Полные таблицы - ranking_*/designs_val_* по h."""
import re
from pathlib import Path

CR = ('c1', 'c2', 'c3', 'c4')
TOP = 10


def counter(devlog, n_pairs, n_designs):
    p = Path(devlog)
    prev = sum(int(x) for x in re.findall(r'Перебрано дизайн×h на разработочном наборе: (\d+)', p.read_text())) if p.exists() else 0
    return f'Перебрано дизайн×h на разработочном наборе: {n_pairs} (до {n_designs} дизайнов на h); накопленно с прошлых итераций: {prev + n_pairs}'


SCH = ('V1_Adaptive', 'V2_MultiVal', 'V3_Regime', 'V4_Shrinkage', 'V6_TrimmedW', 'V7_HalfLife', 'V10_MinQ', 'V11_Gap', 'Med3', 'Gate_L2', 'G3_InvQ')


def blend_effect(P, blends):
    """Д8.1: средний val-QLIKE 2016-2021 каждой схемы при смешивании rv и log (поправка ord, без усадки) и лучший дизайн каждого смешивания."""
    if len(blends) < 2:
        return []
    L = ['', 'Д8.1 смешивание rv (по RV после индивидуальных поправок) против log (в логах): средний val-QLIKE 2016-2021, поправка ord, L0',
         '| h | смешивание | ' + ' | '.join(s.split('_')[0] for s in SCH) + ' | выбор step | лучший по правилу |', '|' + '---|' * (4 + len(SCH))]
    for p in P:
        for bm in ('rv', 'log'):
            sfx = '' if bm == 'rv' else '|log'
            g = p['Rd'][p['Rd'].design == 'single|step|B-ord|L0' + sfx]
            F = p['F'][(p['F'].design.str.endswith('|log') == (bm == 'log')) & ~p['F'].family.str.startswith('V0')]
            b = F.iloc[0]
            L.append(f'| {p["h"]} | {bm} | ' + ' | '.join(f'{g["q_" + s].mean():.4f}' for s in SCH) +
                     f' | {g.q_hyb.mean():.4f} | {b.design.replace("|", "/")} ({b.failed or "-"}; {b.mean_q:.4f}) |')
    return L


def lines(P, boost, lag, cnt, etalon, blends=('rv',)):
    ys = sorted(P[0]['Rd'].val_year.unique())
    L = [f'Гибрид V0-V7 + Д8, пространство дизайнов на {boost} (gibrid_v17.py; метрики только разработочный набор 2016-2021; V7 lag={lag}; эталон c1-c3: одиночные с Б0.1 {"ord" if etalon == "ord" else "той же оценки"}; single = одна точка бустинга, не V0; смешивание {"/".join(blends)})',
         cnt + '. Подтверждающий набор (val 2022-2024) скрыт. Правило выбора: прошли все критерии -> мин. средний val-QLIKE; иначе мин. нарушенных, затем мин. QLIKE.']
    L += blend_effect(P, blends)
    for p in P:
        h, Rd = p['h'], p['Rd']
        L += ['', f'h={h}: топ-{TOP} по правилу (дизайн = нога|выбор|поправка|усадка)',
              '| # | дизайн | не выполнены | ' + ' | '.join(map(str, ys)) + ' | средн. q |', '|' + '---|' * (4 + len(ys))]
        for i, r in enumerate(p['F'].head(TOP).itertuples(), 1):
            q = Rd[Rd.design == r.design].q_hyb.values
            L.append(f'| {i} | {r.design.replace("|", "/")} | {r.failed or "-"} | ' + ' | '.join(f'{x:.3f}' for x in q) + f' | {r.mean_q:.3f} |')
        L.append(f'финалисты auto (лучшие в разных семействах): ' + '; '.join(d.replace('|', '/') for d in p['auto']))
        ob = Rd[Rd.design == 'single|step|B-ord|L0']
        L += ['', f'h={h}: HAR-J, val-QLIKE (а) как в диссертации (расширяющееся, без поправки); (б) с Б0.1 ord - эталон критериев и DM; (в) лучшая спецификация шага (этап 2б), только устойчивость',
              '| строка | ' + ' | '.join(map(str, ys)) + ' | средн. |', '|' + '---|' * (2 + len(ys))]
        for nm, v in (('(а)', ob.q_har_raw), ('(б)', ob.q_har), ('(в)', p['sp'].q_chosen)):
            L.append(f'| {nm} | ' + ' | '.join(f'{x:.3f}' for x in v) + f' | {v.mean():.3f} |')
        r = p['ref'][0]
        L.append(f'Справка (single/step/bez, 2016-2021): V7 lag=h {r["V7_lag_h"]:.3f}; V7 lag=1 (исходная, с заглядыванием на h-1 дней) {r["V7_lag1_orig"]:.3f}; шаги V7 {r["V7_steps_lag_h"]} / {r["V7_steps_lag1"]}; Б0.1 ord in-sample: гибрид {r["hyb_B01_insample"]:.3f}, HAR-J {r["har_B01_insample"]:.3f}')
    L += ['', 'Выбор по правилу:', '| h | дизайн | не выполнены | средн. q |', '|---|---|---|---|']
    L += [f'| {p["h"]} | {p["win"].design.iloc[0].replace("|", "/")} | {p["win"].failed.iloc[0] or "-"} | {p["win"].mean_q.iloc[0]:.3f} |' for p in P]
    return L
