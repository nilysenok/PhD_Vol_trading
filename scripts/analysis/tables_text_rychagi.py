"""Таблица отвергнутых рычагов всех кругов (table_otvergnutye_rychagi.csv). Три источника, все только разработочный набор:
1) авто из markdown-таблиц журналов и отчётов (колонка «решение/вердикт/итог» начинается с «отброшен», «нет», «не проходит» ...);
2) ручной ввод scripts/analysis/tables_text_rychagi_vvod.csv - там, где отчёт даёт вердикт прозой, а не таблицей (Д14, Д16.2, Д16.8, Д12 на h=1, Д8 круга 1 ...);
3) авто из данных D19/eval_D19.csv (Д19 - до выхода отчёта).
Новые отчёты Д17-Д22 (docs/otchety/*d17*, *d18*, *d19*, *d22*) подхватываются автоматически. Строки с годами 2022+ и словом «подтвержд» отбрасываются
(защита подтверждающего набора); dev_log.md (общий) не разбирается: в нём есть строки подтверждающего набора."""
import csv, re
from pathlib import Path
import pandas as pd
from tables_text_lib import R, OTCH, write_table

VVOD = Path(__file__).with_name('tables_text_rychagi_vvod.csv')
FILES = [('dev_log_L2.md', '1–2 (уровень 2, Д1–Д11)'), ('2026-10-25_d12.md', '2 (Д12)'), ('2026-10-25_d13.md', '2 (Д13)'), ('2026-10-25_d15.md', '2 (Д15)'), ('2026-10-26_d16.md', '3 (Д16)')]
NEW = ('*d17*.md', '*d18*.md', '*d19*.md', '*d22*.md', '*d20*.md', '*d21*.md')  # отчёты и журналы рычагов третьего круга, когда появятся
REJ = re.compile(r'^(отброш|отверг|нет\b|не прошёл|не проход|не принят|не берёт|не взят|снят)', re.I)
CONF = re.compile(r'\b202[2-5]\b|подтвержд', re.I)
SKIP_H = ('№', 'время', 'дата', 'дата-время')


def cells(line): return [c.strip() for c in line.strip().strip('|').split('|')]


def md_tables(path):
    """[(заголовок, [строки])] из markdown-таблиц файла"""
    out, cur = [], None
    for ln in Path(path).read_text(encoding='utf-8').splitlines():
        if ln.startswith('|'):
            c = cells(ln)
            if cur is None: cur = [c, []]; continue
            if all(re.fullmatch(r':?-{2,}:?', x) for x in c if x): continue
            cur[1].append(c)
        elif cur: out.append(cur); cur = None
    if cur: out.append(cur)
    return out


def clean(s): return re.sub(r'\s+', ' ', s.replace('**', '').replace('`', '')).strip()


def from_table(path, krug):
    rows = []
    for head, body in md_tables(path):
        hl = [h.lower() for h in head]
        ver = [i for i, h in enumerate(hl) if re.search(r'решение|вердикт|итог$', h)]
        if not ver: continue
        lev = next((i for i, h in enumerate(hl) if 'рычаг' in h), next((i for i, h in enumerate(hl) if re.search(r'блок|что|вариант|кандидат|д16', h)), 0))
        var = next((i for i, h in enumerate(hl) if 'вариант' in h and i != lev), None)
        hi = next((i for i, h in enumerate(hl) if h == 'h'), None)
        crit = [i for i, h in enumerate(hl) if 'c1' in h or 'критер' in h]
        last = ''
        for r in body:
            r = r + [''] * (len(head) - len(r))
            if r[lev]: last = clean(r[lev])
            if CONF.search(' '.join(r)): continue
            start = 0
            for k, v in enumerate(ver):
                if not REJ.match(clean(r[v]).lower()): start = v + 1; continue
                nums = '; '.join(f'{clean(head[i])}: {clean(r[i])}' for i in range(start if len(ver) > 1 else 0, v) if i not in (lev, var, hi) and re.search(r'\d', r[i]) and hl[i] not in SKIP_H)
                hh = clean(r[hi]) if hi is not None else (m.group(1) if (m := re.search(r'h=(\d+)', ' '.join(head[start:v]))) else '')
                why = clean(r[v]) + ('; критерии c1–c4: ' + ', '.join(clean(r[i]) for i in crit if r[i]) if crit and any(r[i] for i in crit) else '')
                rows.append([krug, last if lev != var else clean(r[lev]), clean(r[var]) if var is not None else '', hh, nums, why, 'авто из таблицы отчёта', path.name])
                start = v + 1
    return rows


def from_d19():
    e = R / 'D19/eval_D19.csv'
    if not e.exists(): return []
    d = pd.read_csv(e); nm = {'1': 'Д19.1 ρ·r̂₁ (OLS)', '2': 'Д19.2 ρ из импульсного отклика HAR', '3': 'Д19.3 классификатор направления', '4': 'Д19.4 Д18.1 × Д19.1'}
    out = []
    for _, r in d[(d.kind == 'лучший') & ~d.boost.str.contains(r'_s4\d')].iterrows():
        if r.beats_1720: continue
        out.append(['3 (Д19)', nm.get(r.boost.split('_')[1][0], r.boost), f'{r.boost}, дизайн {r.design}', int(r.h),
                    f'2017–2020: {r.mean_1720:.4f} против {r.inc_1720:.4f} у действующего {r.inc}; 2021: {r.q2021:.4f} против {r.inc_2021:.4f}'.replace('.', ','),
                    f'не лучше действующего дизайна на 2017–2020; нарушено: {r.failed if isinstance(r.failed, str) else "—"} (предварительно, отчёт Д19 не опубликован)', 'авто из данных', 'data/regen/D19/eval_D19.csv'])
    return out


def build():
    rows, parsed = [], []
    with open(VVOD, encoding='utf-8-sig') as f:
        for r in csv.reader(f, delimiter=';'):
            if r and r[0] != 'Круг': rows.append(r[:6] + ['ручной ввод из отчёта', r[6]])
    files = [(OTCH / n, k) for n, k in FILES] + [(p, '3 (Д17–Д22)') for pat in NEW for p in sorted(OTCH.glob(pat))]
    for p, k in files:
        if p.exists(): rows += from_table(p, k); parsed.append(p.name)
    rows += from_d19()
    seen, uniq = set(), []
    for r in rows:
        key = (r[1], r[2], str(r[3]), r[4][:60])
        if key not in seen: seen.add(key); uniq.append(r)
    head = ['Круг', 'Рычаг', 'Вариант', 'h', 'Валидационные числа (разработочный набор, 2017–2021 если не указано)', 'Причина отклонения', 'Происхождение строки', 'Источник']
    n = write_table('table_otvergnutye_rychagi.csv', head, uniq, 'журналы и отчёты docs/otchety/: ' + ', '.join(parsed) + '; ручной ввод scripts/analysis/tables_text_rychagi_vvod.csv; data/regen/D19/eval_D19.csv',
                    'метрики — OOS QLIKE (Б0.1), разработочный набор; протокол метрики у каждого круга свой (см. отчёт-источник). Принятые рычаги (V9, Д8.12, Д7, linear_tree, mono −1, Huber на h=5) не включены.')
    wait = ['отчёты рычагов, которых ещё нет: ' + ', '.join(p for p in NEW[:4] if not any(OTCH.glob(p)))] if not all(any(OTCH.glob(p)) for p in NEW[:4]) else []
    return n, wait
