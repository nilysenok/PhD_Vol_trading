"""Текст итерации dev_log: только разработочный набор (val-годы 2016-2021, шаги 1-6). Метрики 2022-2024 здесь не выводятся
(их печатает только --confirm --finalists, один раз)."""
import re
from pathlib import Path

CR = ('c1', 'c2', 'c3', 'c4')


def counter(devlog, n_pairs, n_designs):
    p = Path(devlog)
    prev = sum(int(x) for x in re.findall(r'Перебрано дизайн×h: (\d+)', p.read_text())) if p.exists() else 0
    return f'Перебрано дизайн×h на разработочном наборе: {n_pairs} ({n_designs} дизайнов x 3 h); накопленно с прошлых итераций: {prev + n_pairs}'


def lines(Rd, win, ref, sp, boost, lag, cnt, ys):
    L = [f'Гибрид V1-V7 + V0, дизайны на {boost} (gibrid_v17.py; метрики только разработочный набор 2016-2021; V7 lag={lag}; Б0.1 только oos; c1-c3 относительно одиночных с Б0.1oos; single = одна точка бустинга, не V0)',
         cnt + '. Подтверждающий набор (val 2022-2024) скрыт; по П1 он был виден до введения правила разработка/подтверждение, поэтому П1 не является чистой проверкой.',
         '| дизайн | h | ' + ' | '.join(str(y) for y in ys) + ' | средн. q | худш. отн. к HAR | c1 | c2 | c3 | c4 |', '|' + '---|' * (8 + len(ys))]
    for (d, h), g in Rd.groupby(['design', 'h'], sort=False):
        f = g.iloc[0]
        L.append(f'| {d.replace("|", "/")} | {h} | ' + ' | '.join(f'{r.q_hyb:.3f} ({r.ratio_har:.2f})' for r in g.itertuples())
                 + f' | {g.q_hyb.mean():.3f} | {f.worst_ratio_h:.2f} | ' + ' | '.join('+' if f[k + '_h'] else '-' for k in CR) + ' |')
    L += ['', 'Выбор по правилу (прошли все -> мин. средний val-QLIKE; иначе мин. нарушенных, затем мин. QLIKE; на 2016-2021):', '| h | дизайн | не выполнены | средн. q |', '|---|---|---|---|']
    L += [f'| {r.h} | {r.design.replace("|", "/")} | {r.failed or "-"} | {r.mean_q:.3f} |' for r in win.itertuples()]
    L += ['', 'Справка (single/step, средний val-QLIKE 2016-2021):', '| h | гибрид Б0.1 in-sample | HAR Б0.1 in-sample | V7 lag=h | V7 lag=1 (исходная, с заглядыванием на h-1 дней) | шаги V7 lag=h | шаги V7 lag=1 |', '|---|---|---|---|---|---|---|']
    L += [f'| {r["h"]} | {r["hyb_B01_insample"]:.3f} | {r["har_B01_insample"]:.3f} | {r["V7_lag_h"]:.3f} | {r["V7_lag1_orig"]:.3f} | {r["V7_steps_lag_h"]} | {r["V7_steps_lag1"]} |' for r in ref]
    ob = Rd[Rd.design == 'single|step|B0.1oos']
    L += ['', 'HAR-J, val-QLIKE: (а) расширяющееся без поправки; (б) то же с Б0.1oos (эталон критериев и DM); (в) лучшая спецификация шага из списка этапа 2б (только строка устойчивости)',
          '| h | строка | ' + ' | '.join(map(str, ys)) + ' | средн. |', '|' + '---|' * (3 + len(ys))]
    for h in (1, 5, 22):
        for nm, v in (('(а)', ob[ob.h == h].q_har_raw), ('(б)', ob[ob.h == h].q_har), ('(в)', sp[sp.h == h].q_chosen)):
            L.append(f'| {h} | {nm} | ' + ' | '.join(f'{x:.3f}' for x in v) + f' | {v.mean():.3f} |')
    return L
