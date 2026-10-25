#!/usr/bin/env python3
"""Таблицы для текста из CSV результатов: docs/tables/ (';', десятичная запятая, UTF-8 с BOM, под таблицей - строка-источник).
Только разработочный набор: val 2017-2021 (шаги 2-6, OOS); отбор третьего круга - 2017-2020. Подтверждающий набор и тест не читаются.
Перезапускаемый: что появилось в data/regen/ - подхватывается, чего нет - строка «ждёт данных». python3 scripts/analysis/tables_text.py"""
import sys, re
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tables_text_lib import R, OTCH, YEARS, fmt, pct, write_table
import tables_text_rychagi as RY

Y = [str(y) for y in YEARS]
CODE16 = {1: 'sm', 5: 'sm', 22: 'smrob'}  # s² HAR-J с поправкой по правилу (Д16, final2)
DM_Y = ['DM t ' + y for y in Y]


def rd(p):
    p = Path(p); return pd.read_csv(p) if p.exists() else None


def mean(q, ys=YEARS):
    v = [q[y] for y in ys if q.get(y) == q.get(y) and q.get(y) is not None]
    return sum(v) / len(v) if len(v) == len(ys) else None


def qs(r, pre='q_'):  # строка рейтинга/таблицы -> {год: QLIKE}
    return {y: float(r[f'{pre}{y}']) for y in YEARS if f'{pre}{y}' in r and r[f'{pre}{y}'] == r[f'{pre}{y}']}


def nn_rows(prefix, h, code):  # GRU/LSTM: single|V0_XGB|B-<s²>|L0 из gibrid-каталога, шаги 2-6
    d = rd(R / f'{prefix}_gibrid' / f'designs_val_{prefix}_h{h}.csv')
    if d is None: return None
    x = d[(d.design == f'single|V0_XGB|B-{code}|L0') & d.step.between(2, 6)].set_index('val_year').q_hyb
    return {int(y): float(v) for y, v in x.items()}


def table3():
    head = ['h', 'Группа', 'Модель', 'Дизайн / примечание'] + Y + ['Среднее 2017–2020', 'Среднее 2017–2021', 'К HAR-J с поправкой, %', 'Лет лучше HAR-J (из 5)', 'Нарушено c1–c4 (дизайн)', 'Финалист (круг 2 — предварительный список до подтверждающего, круг 3 — final3)']
    out, srcs, wait = [], set(), []
    cmp_ = rd(R / 'GARCH_sv_compare_dev.csv')
    for h in (1, 5, 22):
        rk, k3 = None, None
        for d in ('final3', 'final3_dry'):
            if (R / d / f'ranking_h{h}.csv').exists(): rk, k3 = rd(R / d / f'ranking_h{h}.csv'), d; break
        harj = rk[rk.cand == 'HARJ_corr'].iloc[0] if rk is not None and (rk.cand == 'HARJ_corr').any() else None
        ref = qs(harj) if harj is not None else None
        code = harj.design.split('|')[2].replace('B-', '') if harj is not None else CODE16[h]
        if ref is None and cmp_ is not None:
            c = cmp_[(cmp_.h == h) & cmp_.line.str.startswith('HAR-J с поправкой')].iloc[0]; ref = {y: float(c[str(y)]) for y in YEARS}
        def add(grp, model, note, q, failed='', fin=''):
            if q is None: out.append([h, grp, model, note] + [''] * 10 + [fin]); return
            m20, m21 = mean(q, YEARS[:4]), mean(q)
            better = sum(q[y] < ref[y] for y in YEARS) if ref and len(q) == 5 and model != 'HAR-J с поправкой' else None
            out.append([h, grp, model, note] + [q.get(y) for y in YEARS] + [m20, m21, pct(m21, mean(ref)) if ref else None, better, failed, fin])
        if harj is not None:
            add('эконометрика', 'HAR-J без поправки', 'столбец raw_* рейтинга', qs(harj, 'raw_'))
            add('эконометрика', 'HAR-J с поправкой', harj.design + ' (эталон таблицы)', qs(harj))
        if cmp_ is not None:
            for lab, grp, pre in (('GJR-GARCH', 'GARCH справочно', 'GJR-GARCH без поправки'), ('GARCH(1,1)', 'GARCH справочно', 'GARCH(1,1)'), ('GRU', 'нейросеть справочно', 'справочный GRU')):
                c = cmp_[(cmp_.h == h) & cmp_.line.str.startswith(pre)]
                if len(c): c = c.iloc[0]; add(grp, lab, c.line, {y: float(c[str(y)]) for y in YEARS}); srcs.add('GARCH_sv_compare_dev.csv')
        lstm = nn_rows('LSTM_svoos', h, code)
        if lstm is None:
            has = (R / f'LSTM_svoos_h{h}_val.parquet').exists()
            add('нейросеть справочно', 'LSTM', 'прогнозы есть, поправка Б0.1 не оценена' if has else 'нет', None); wait.append(f'LSTM h={h}')
        else: add('нейросеть справочно', 'LSTM', f'LSTM_svoos, B-{code}', lstm); srcs.add('LSTM_svoos_gibrid')
        v0 = R / 'final2/gibrid' / ('D12_h1_V0' if h == 1 else 'Bv3') / (f'designs_val_D12_h1_V0_h1.csv' if h == 1 else f'designs_val_Bv3_h{h}.csv')
        d = rd(v0)
        if d is not None:
            for lab, sl in (('LightGBM', 'LGB'), ('XGBoost', 'XGB')):
                x = d[(d.design == f'single|V0_{sl}|B-{code}|L0') & d.step.between(2, 6)].set_index('val_year').q_hyb
                add('чистый бустинг (V0)', lab, f'{"D12_h1_V0" if h == 1 else "Bv3"} (точки B v2b), single, B-{code}', {int(y): float(v) for y, v in x.items()}); srcs.add(v0.parent.name + '/designs_val')
        fp = rd(R / 'final2/finalists_preliminary.csv'); fin = set(fp[(fp.h == h) & (fp.order > 0)].cand) if fp is not None else set()
        if rk is not None:
            srcs.add(f'{k3}/ranking_h{h}.csv')
            for _, r in rk[rk.cand != 'HARJ_corr'].sort_values(['n_failed', 'mean_1720']).iterrows():
                f3 = (str(r.get('finalist')) not in ('0', '0.0', 'nan')) if k3 == 'final3' else r.cand in fin
                add(f'гибрид, круг {int(r.krug)}', r.cand, f'{r.fam}: {r.design}', qs(r), r.failed if isinstance(r.failed, str) else '', 'да' if f3 else '')
        if k3 != 'final3':  # предварительные кандидаты третьего круга (до сборки final3)
            a = rd(R / 'D18/gibrid_representatives.csv'); b = rd(R / 'D19/eval_D19.csv'); c16 = rd(R / f'D16/d16_eval_h{h}.csv')
            for _, r in (a[a.h == h].iterrows() if a is not None else []):
                add('гибрид, круг 3 (предв.)', r.boost, f'{r.design} (D18/gibrid_representatives)', {y: float(r[str(y)]) for y in YEARS}, r.failed if isinstance(r.failed, str) else '')
            for _, r in (b[(b.h == h) & (b.kind == 'лучший') & ~b.boost.str.contains(r'_s4\d')].iterrows() if b is not None else []):
                add('гибрид, круг 3 (предв.)', r.boost, f'{r.design} (D19/eval_D19)', qs(r), r.failed if isinstance(r.failed, str) else '')
            for _, r in (c16[(c16.code == CODE16[h]) & c16.model.str.startswith('Д16.1')].iterrows() if c16 is not None else []):
                add('гибрид, круг 3 (предв.)', 'Д16.1', r.model + ' (D16/d16_eval)', {y: float(r[f'q{y}']) for y in YEARS})
    if not (R / 'final3/ranking_h1.csv').exists(): wait.append('финалисты круга 3 (final3/ranking_h*.csv); пока — сухой прогон на кандидатах круга 2 и предварительные Д16/Д18/Д19')
    n = write_table('table_3_sravnenie_dev.csv', head, out, 'data/regen/: ' + ', '.join(sorted(srcs)) + '; docs/otchety/2026-10-25_final2.md, 2026-10-26_garch_sv.md; val 2017–2021, OOS (шаги 2–6), QLIKE; отбор круга 3 — 2017–2020',
                    'эталон строки «к HAR-J» — HAR-J с поправкой того же горизонта (отрицательное = лучше). Без DM (DM против HAR-J с поправкой — в table_raskladka_harj_harx.csv). Подтверждающий набор и тест не использованы.')
    return n, wait


def raskladka():
    head = ['h', 'Звено', 'Модель'] + Y + ['Среднее 2017–2020', 'Среднее 2017–2021', 'К HAR-J с поправкой, %', 'DM: против строки'] + DM_Y
    out, wait = [], []
    for h in (1, 5, 22):
        e, dm = rd(R / f'D16/d16_eval_h{h}.csv'), rd(R / f'D16/d16_dm_h{h}.csv')
        if e is None: wait.append(f'D16 h={h}'); continue
        e = e[e.code == CODE16[h]]; ref = mean({y: float(e[e.model == 'HAR-J с поправкой'].iloc[0][f'q{y}']) for y in YEARS})
        for _, r in e[~e.model.str.contains('HAR-X0')].iterrows():
            q = {y: float(r[f'q{y}']) for y in YEARS}; link = r.model.split(' ')[0]
            pairs = dm[(dm.a == r.model) & (dm.code == CODE16[h])] if dm is not None else []
            pairs = [None] if len(pairs) == 0 else [p for _, p in pairs.iterrows()]
            for i, p in enumerate(pairs):
                out.append([h, 'HAR-J' if r.model.startswith('HAR-J') else ('HAR-X' if r.model.startswith('HAR-X') else 'гибрид'), r.model] + ([q[y] for y in YEARS] + [mean(q, YEARS[:4]), mean(q), pct(mean(q), ref)] if i == 0 else [''] * 8)
                           + ([p.b] + [fmt(float(p[f't{y}']), 2) for y in YEARS] if p is not None else [''] * 6))
    n = write_table('table_raskladka_harj_harx.csv', head, out, 'data/regen/D16/d16_eval_h{1,5,22}.csv, d16_dm_h*.csv; docs/otchety/2026-10-26_d16.md; s²: sm / sm / smrob (h = 1 / 5 / 22); val 2017–2021, OOS',
                    'DM t = потери строки минус потери строки «против»: t < 0 — строка лучше. Цепочка: HAR-J без поправки → HAR-J с поправкой → HAR-X с поправкой → гибрид; строки «Д16.1» — гибрид на HAR-X. На h=22 «гибрид» — справка (финал — HAR-J).')
    return n, wait


def rho():
    head = ['h', 'Вариант', 'Год оценки (val)', 'ρ (OLS без константы, окно шага t−1)', 'ρ (импульсный отклик HAR, Д19.2)', 'κ (Д19.4x)', 'b_d (HAR)', 'b_w', 'b_m', 'n истории', 'ρ в (0; 1)']
    out, wait, used = [], [], []
    for h in (5, 22):
        d = rd(R / f'D19/rho_h{h}.csv')
        if d is None: wait.append(f'D19/rho_h{h}.csv'); continue
        used.append(f'rho_h{h}.csv')
        for seed, g in d.groupby('seed', sort=False):
            g = g[g.step.between(2, 6)]
            lab = 'нога r̂₁ ' + ('top5' if seed == 'top5' else 'seed ' + seed.strip('s'))
            for _, r in g.iterrows(): out.append([h, f'Д19.1 (ρ OLS) / Д19.2 (импульсный отклик), {lab}', 2015 + int(r.step), r.rho1_val, r.rho2, '', r.b_d, r.b_w, r.b_m, int(r.n_hist), bool(0 < r.rho1_val < 1)])
            out.append([h, f'Д19.1, {lab}', 'среднее / min / max / СКО', f'{fmt(g.rho1_val.mean())} / {fmt(g.rho1_val.min())} / {fmt(g.rho1_val.max())} / {fmt(g.rho1_val.std())}', f'{fmt(g.rho2.mean())} / {fmt(g.rho2.min())} / {fmt(g.rho2.max())}', '', '', '', '', '', bool(g.rho1_val.between(0, 1, inclusive='neither').all())])
        for pre, lab in (('rho4_h', 'Д19.4'), ('rho4x_h', 'Д19.4x')):
            f = R / f'D19/{pre}{h}.csv'; z = rd(f)
            if z is None: continue
            used.append(f.name); z = z[z.step.between(2, 6)]; col = 'rho_val' if 'rho_val' in z else 'rho4_val'
            for _, r in z.iterrows(): out.append([h, lab, 2015 + int(r.step), r[col], '', r['kappa_val'] if 'kappa_val' in z else '', '', '', '', '', bool(0 < r[col] < 1)])
    if not out: wait.append('D19')
    n = write_table('table_rho_po_shagam.csv', head, out, 'data/regen/D19/' + ', '.join(used) + '; отчёт docs/otchety/2026-10-26_d19.md ' + ('(есть)' if (OTCH / '2026-10-26_d19.md').exists() else '(ещё не опубликован)'),
                    'год оценки = 2015 + шаг (шаги 2–6 = 2017–2021); ρ шага применён к val-году шага (оценка на окне до t−1). Шаги 1 и 7–10 не показаны. Данные Д19 могут обновляться — таблица перезапускаемая.')
    return n, wait


def d22():
    head = ['Вариант', 'h'] + Y + ['Среднее 2017–2020', 'Среднее 2017–2021', 'К полному Д22, %', 'К HAR-J с поправкой, %', 'Статус']
    D = R / 'D22'; rep = OTCH / '2026-10-26_d22.md'
    cand = [p for pat in ('*abl*.csv', '*ablyac*.csv', 'eval*.csv', 'qlike_by_year*.csv') for p in sorted(D.glob(pat))] if D.exists() else []
    layers = ['полный Д22', 'минус слой 1 (HAR-X⁺ → HAR-J)', 'минус слой 2 (hyb₁ из Д8.12 на HAR-X⁺)', 'минус слой 3 (ridge по x1–x7)', 'минус слой 4 (поправка s², защита Д14, винзоризация)']
    out, wait = [], []
    if cand:
        a = pd.read_csv(cand[0]); cols = {c: c.replace('q_', '').replace('q', '') for c in a.columns}; a = a.rename(columns=cols)
        if all(y in a.columns for y in Y):
            for _, r in a.iterrows():
                q = {y: float(r[str(y)]) for y in YEARS}; out.append([r.get('name', r.get('model', r.get('boost', ''))), r.get('h', ''), *[q[y] for y in YEARS], mean(q, YEARS[:4]), mean(q), '', '', f'из {cand[0].name}'])
    if not out:
        has = sorted(p.name for p in D.glob('HXP_*_val.parquet')) if D.exists() else []
        for h in (1, 5, 22):
            for l in layers: out.append([l, h] + [''] * 9 + ['ждёт данных (абляций Д22 в data/regen/D22 пока нет)' + ('; слой 1: ' + ', '.join(has) if has and l == layers[1] else '')])
        wait.append('абляции Д22 (data/regen/D22/*abl*.csv, eval*.csv; отчёт ' + rep.name + ')')
    n = write_table('table_d22_ablyacii.csv', head, out, (f'data/regen/D22/{cand[0].name}' if cand and not wait else 'заготовка: данных абляций Д22 нет') + '; отчёт docs/otchety/2026-10-26_d22.md ' + ('(есть)' if rep.exists() else '(ещё не опубликован)'),
                    'Абляции Д22: минус каждый слой (задание владельца 26.10, 01:35). Метрика — OOS QLIKE, val 2017–2021. Перезапустить после появления файлов абляций.')
    return n, wait


if __name__ == '__main__':
    res = {'table_3_sravnenie_dev.csv': table3(), 'table_d22_ablyacii.csv': d22(), 'table_raskladka_harj_harx.csv': raskladka(), 'table_rho_po_shagam.csv': rho(), 'table_otvergnutye_rychagi.csv': RY.build()}
    for k, (n, w) in res.items(): print(f'{k}: {n} строк' + (f'; ждёт: {"; ".join(w)}' if w else ''))
