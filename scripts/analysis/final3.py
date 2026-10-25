#!/usr/bin/env python3
"""Итоговая сборка финалистов третьего круга (сводный блок владельца 26.10, 00:55; уточнение дирижёра о «видевших подтверждающий»).
Фазы (порядок — final3.sh):
  pool    — пул кандидатов всех кругов по h -> <OUT>/pool.csv, pool.md (ничего не считает);
  gibrid  — гибрид gibrid_v17.py (--dev-eval oos, шаги 7-9 скрыты) для кандидатов третьего круга без designs_val;
  select  — правило третьего круга: рейтинг по OOS QLIKE val 2017-2020 (шаги 2-5), критерии c1-c4 на val 2017-2021 (шаги 2-6, OOS;
            val-год 2016 = шаг 1 даёт s² для шага 2), эталон — модели с той же s², V0 наравне, h=1 — tie-break владельца;
            отложенный 2021 (шаг 6) для кандидатов третьего круга; три финалиста на h из разных семейств; --zapis -> dev_log.md ДО открытия;
  confirm — ЕДИНСТВЕННОЕ (третье и последнее) открытие подтверждающего набора (val 2022-2024, шаги 7-9) для финалистов:
            gibrid_v17.py --confirm --confirm-next 0; М4 — первый прошедший; h=22 никто -> HAR-J с поправкой; h=1/5 никто -> первый финалист
            с пометкой. Требует --confirm, записи финалистов в dev_log (finalists_logged.json) и отсутствия замка CONFIRM_OPENED.json;
  build   — тестовые прогнозы финального дизайна шагов 1-10 БЕЗ МЕТРИК -> test_locked/final3/final_h{h}_test.parquet (столбцы final_test.py);
            σ_pred(h=1) -> data/regen/sigma_pred_h1_final3.parquet, если финал h=1 не D12_h1_A_d812 top5/V0_XGB/B-sm/L0;
  report  — docs/otchety/FINAL_DESIGN_chernovik.md (перезапись). FINAL_DESIGN.md не создаётся.
--dry: всё в data/regen/final3_dry/ и test_locked/final3_dry/, dev_log и черновик FINAL не трогаются; confirm в --dry запрещён.
--only-krug2: пул без кандидатов третьего круга (сухой прогон на кандидатах второго круга).

Кандидат третьего круга входит в пул, если есть OOS-прогнозы (val шагов 1-9) и yaml-описание. Формат yaml (ключи; обязательны boost и h):
  boost: D17_1_h22_G22      # префикс файлов {boost}_{xgb|lgbm}_h{h}[_top5|_seeds4344|_wrap2]_val.parquet
  h: 22                     # или список [5, 22]
  semeystvo: GLM с офсетом  # семейство для правила «три финалиста из разных семейств» (по умолчанию — rychag, затем boost)
  rychag: Д17.1             # номер рычага
  opisanie: ...             # одна строка для FINAL
  regen: data/regen         # каталог val-файлов (по умолчанию общий data/regen; относительный путь — от корня PhD_Vol_trading)
  legs: [single, seeds3]    # по умолчанию — все ноги, для которых есть оба файла
  odinochnaya: false        # true — одиночная модель без бустинга: в пул НЕ входит (решение владельца 01:12: HAR-X — только нога гибрида)
  chistyy: false            # true — чистый бустинг (V0 владельца; tie-break h=1, цель 3)
  osnova: F2m_h5            # кандидат второго круга, от которого вариант отличается только рычагом (по умолчанию — по имени)
  test_dir: ...             # по умолчанию data/regen/test_locked/<boost>/
  step10_dir / step10_lock: # val 2025 / тест 2026 шага 10 в формате final2_step10.py ({boost}_{m}_h{h}_s10_{val,test}.parquet)
  features_har: ...         # матрица для HAR-J/HAR-CJ гибрида (по умолчанию features_pit_v3.parquet)
  n_variantov: 0            # число вариантов бустинга, перебранных агентом (счётчик FINAL)
  harx: false               # нога гибрида на HAR-X (по умолчанию — по имени: harx / окончание X)
  trees: ...                # CSV деревьев (model, step, best_iteration) для h=22, если не {boost}_selected_h22.csv
  god2021: ...              # как агент использовал 2021 (например: «только отложенный, одна проверка»)
Где ищутся yaml: data/regen/final3/kandidaty/, data/regen/ (верхний уровень), data/regen/{D14,D16,D17,D18-D22,HARX}*/ (до трёх уровней).
python3 scripts/analysis/final3.py pool|gibrid|select|confirm|build|report [--dry] [--only-krug2] [--zapis] [--confirm] [--workers 6]"""
import argparse, hashlib, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent.parent
ROOT = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading')
R = ROOT / 'data' / 'regen'
FEAT = R / 'features_pit_v3.parquet'
CR = ('c1', 'c2', 'c3', 'c4')
RANK_STEPS, HOLD, CRIT = (2, 3, 4, 5), 6, (2, 3, 4, 5, 6)  # 2017-2020 / 2021 / 2017-2021
LEGS2 = ['single', 'top5', 'seeds3', 'wrap2']
SFX = {'single': '', 'top5': '_top5', 'seeds3': '_seeds4344', 'wrap2': '_wrap2'}
H1_FINAL2 = ('D12_h1_A_d812', 'top5|V0_XGB|B-sm|L0')
R2_TOTAL = (18631, 271)  # дизайн×h гибрида и варианты бустинга за круги 1-2 (2026-10-25_final2.md, шаг 7)

# финалисты второго круга (dev_log № 48) и результат единственного открытия 26.10, 00:27 (2026-10-25_final2.md)
R2 = [
    dict(boost='D12_h1_A_d812', h=1, fam='Д8.12: прогноз HAR-J как признак', chistyy=False,
         opisanie='XGBoost/LightGBM на ln RV + Д8.12 (ln HAR-J с поправкой как признак), точки ЛГК B v2b, features_pit_v3',
         seen={'top5|V0_XGB|B-sm|L0': 'прошёл'}),
    dict(boost='D12_h1_B_b0.25', h=1, fam='V9: остатки HAR-J', chistyy=False, opisanie='V9 с усадкой базы β = 0,25, точки B v2b, v3',
         seen={'top5|V0_XGB|B-sm|L0': 'не прошёл'}),
    dict(boost='D12_h1_V0', h=1, fam='чистый бустинг', chistyy=True, opisanie='чистый бустинг (V0), точки ЛГК B v2b на v3',
         seen={'top5|V0_XGB|B-sm|L0': 'прошёл'}),
    dict(boost='F2m_h5', h=5, fam='V9 на HAR-J', chistyy=False, opisanie='V9 + Д8.12 + Д7 + monotone −1, optuna-30 на v3',
         seen={'wrap2|V0_LGB|B-smrob|L0': 'не прошёл'}),
    dict(boost='D12_h5', h=5, fam='V9 на HAR-J', chistyy=False, opisanie='V9 + Д8.12 + Д7 + Huber, optuna-30 на v3',
         seen={'wrap2|V0_XGB|B-sm|L0': 'прошёл'}),
    dict(boost='L2v3_h5', h=5, fam='V9 на HAR-J', chistyy=False, opisanie='V9 + Д8.12 + Д7, optuna-30 на v3',
         seen={'single|V0_XGB|B-smrob|L0': 'прошёл'}),
    dict(boost='D12_h22', h=22, fam='V9 на HAR-J', chistyy=False, opisanie='V9 + linear_tree λ = 1 (LGBM), optuna-30 на v3',
         seen={'seeds3|V0_XGB|B-smrob|L0': 'не прошёл', 'single|minimax|B-smrob|L0.25|log': 'не прошёл', 'single|step|B-smrob|L0.5': 'не прошёл'}),
]
HARJ_HOST = {1: 'D12_h1_A_d812', 5: 'D12_h5', 22: 'D12_h22'}  # HAR-J с поправкой берётся из гибрида этого кандидата (HAR-J общий)
R3_PREFIX = ('D14', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'HARX')  # решение 01:35: + Д18-Д22
F = lambda x, n=4: '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{n}f}'.replace('.', ',')
P = lambda x: '—' if x is None or not np.isfinite(x) else f'{100 * x:+.1f}%'.replace('.', ',')


def now():
    return subprocess.run(['date', '+%Y-%m-%d %H:%M'], capture_output=True, text=True).stdout.strip()


def out_dir(a):
    return R / ('final3_dry' if a.dry else 'final3')


def lock_dir(a):
    return R / 'test_locked' / ('final3_dry' if a.dry else 'final3')


# ---------------------------------------------------------------- пул
def scan_val():
    """-> {(boost, h): {'regen': dir, 'legs': set, 'odin': bool}} по файлам val кандидатов третьего круга (информационно)."""
    two = re.compile(r'^(?P<b>.+?)_(?P<m>xgb|lgbm)_h(?P<h>1|5|22)(?:_(?P<leg>top5|seeds4344|wrap2))?_val\.parquet$')
    one = re.compile(r'^(?P<b>.+?)_h(?P<h>1|5|22)_val\.parquet$')
    dirs = [R] + [d for p in R3_PREFIX for d in R.glob(p + '*') if d.is_dir()]
    sub = []
    for d in dirs[1:]:
        sub += [x for x in d.rglob('*') if x.is_dir() and len(x.relative_to(d).parts) <= 2]
    found, seen2 = {}, {}
    for d in dirs + sub:
        for f in d.glob('*_val.parquet'):
            if '_allpts_' in f.name:
                continue
            m = two.match(f.name)
            if m and m['b'].startswith(R3_PREFIX):
                k = (m['b'], int(m['h']), str(d))
                seen2.setdefault(k, set()).add((m['m'], {'top5': 'top5', 'seeds4344': 'seeds3', 'wrap2': 'wrap2'}.get(m['leg'], 'single')))
                continue
            m = one.match(f.name)
            if m and m['b'].startswith(R3_PREFIX) and not two.match(f.name):
                found[(m['b'], int(m['h']))] = dict(regen=d, legs={'single'}, odin=True)
    for (b, h, d), s in seen2.items():
        legs = {lg for (mm, lg) in s if ('xgb', lg) in s and ('lgbm', lg) in s}
        if legs:
            found[(b, h)] = dict(regen=Path(d), legs=legs, odin=False)
    return found


def scan_yaml():
    import yaml
    places = [R / 'final3' / 'kandidaty', R]
    out = []
    files = [f for pl in places if pl.exists() for f in list(pl.glob('*.yaml')) + list(pl.glob('*.yml'))]
    for p in R3_PREFIX:
        for d in R.glob(p + '*'):
            if d.is_dir():
                files += [f for f in d.rglob('*.y*ml') if len(f.relative_to(d).parts) <= 3]
    for f in sorted(set(files)):
        try:
            y = yaml.safe_load(f.read_text())
        except Exception as e:  # битый yaml — в пул не входит, видно в таблице
            out.append(dict(_file=f, _err=str(e)))
            continue
        if isinstance(y, dict) and 'boost' in y and 'h' in y:
            y['_file'] = f
            out.append(y)
    return out


def rpath(x, default):
    if not x:
        return Path(default)
    x = Path(x)
    return x if x.is_absolute() else ROOT / x


def osnova_of(boost, y):
    if y.get('osnova'):
        return y['osnova']
    names = sorted({c['boost'] for c in R2}, key=len, reverse=True)
    return next((n for n in names if n in boost), None)


def has_test(c):
    td = c['test_dir']
    if c['odin']:
        return (td / f"{c['gboost']}_h{c['h']}_test.parquet").exists()
    legs = c['legs'] or ['single']
    return all((td / f"{c['gboost']}_{m}_h{c['h']}{SFX[lg] if lg != 'wrap2' else '_wrap2'}_test.parquet").exists()
               for m in ('xgb', 'lgbm') for lg in legs if lg in ('single', 'top5', 'seeds3', 'wrap2'))


def has_s10(c):
    if c['odin']:
        return False
    return all((c['step10_dir'] / f"{c['gboost']}_{m}_h{c['h']}_s10_val.parquet").exists() and
               (c['step10_lock'] / f"{c['gboost']}_{m}_h{c['h']}_s10_test.parquet").exists() for m in ('xgb', 'lgbm'))


def pool(a):
    O = out_dir(a)
    C = []
    for r in R2:
        c = dict(r, krug=2, gboost=r['boost'], regen=R, legs=LEGS2, odin=False, test_dir=R / 'test_locked' / r['boost'],
                 step10_dir=R / 'final2' / 'step10', step10_lock=R / 'test_locked' / 'final2' / 'step10', wrap_dir=R / 'final2',
                 gib_dir=R / 'final2' / 'gibrid' / r['boost'], features=FEAT, rychag='круг 2', yaml='—', osnova=None, n_variantov=0)
        C.append(c)
    for h, host in HARJ_HOST.items():
        hc = next(c for c in C if c['boost'] == host)
        C.append(dict(hc, boost='HARJ_corr', fam='HAR-J с поправкой', chistyy=False, krug=1, rychag='фолбэк владельца (h=22)',
                      opisanie='HAR-J (features_pit_v3) с поправкой Б0.1, s² по правилу; хост гибрида ' + host,
                      seen={'single|V0_HAR|B-sm|L0': 'не прошёл (круг 1: single/V0_HAR/B-sm на v2)'}, harj=True))
    used = set()
    for h in (1, 5, 22):  # HAR-X (Д16): решение владельца 01:12 — только нога гибрида (V9/Д8.12/Д17.1 на HAR-X), не V0-кандидат; строка — в раскладке
        if (R / f'HARX_h{h}_val.parquet').exists():
            C.append(dict(boost='HARX', gboost='HARX', h=h, krug=3, fam='HAR-X с поправкой', regen=R, legs=['single'], odin=True, yaml='models/designs/harx_h%d.yaml' % h,
                          rychag='Д16', dopusk=False, prichina='HAR-X — только нога гибрида, не V0-кандидат (решение владельца 01:12); строка — в раскладке', seen={}))
            used.add(('HARX', h))
    found = scan_val() if not a.only_krug2 else {}
    ys = scan_yaml() if not a.only_krug2 else []
    for y in ys:
        if '_err' in y:
            C.append(dict(boost=y['_file'].name, h=0, krug=3, fam='—', yaml=str(y['_file']), dopusk=False, prichina='yaml не читается: ' + y['_err']))
            continue
        for h in (y['h'] if isinstance(y['h'], list) else [y['h']]):
            h = int(h)
            b = str(y['boost'])
            if (b, h) in used:
                continue
            fv = found.get((b, h), {})
            regen = rpath(y.get('regen'), fv.get('regen', R))
            odin = bool(y.get('odinochnaya', fv.get('odin', False)))
            if odin:  # одиночная модель без бустинга (HAR-X и т. п.) — не кандидат (решение владельца 01:12)
                C.append(dict(boost=b, gboost=b, h=h, krug=3, fam='—', yaml=str(y['_file']), legs=['single'], dopusk=False, seen={},
                              prichina='одиночная модель без бустинга — не V0-кандидат (решение владельца 01:12)'))
                used.add((b, h))
                continue
            legs = y.get('legs') or sorted(fv.get('legs', set()), key=LEGS2.index) or None
            if legs is None:  # файлы не нашлись сканом — проверяем прямо в regen
                legs = [lg for lg in LEGS2 if all((regen / f'{b}_{m}_h{h}{SFX[lg]}_val.parquet').exists() for m in ('xgb', 'lgbm'))] \
                    if not odin else (['single'] if (regen / f'{b}_h{h}_val.parquet').exists() else [])
            osn = osnova_of(b, y)
            c = dict(boost=b, gboost=b, h=h, krug=3, fam=str(y.get('semeystvo') or y.get('rychag') or b), chistyy=bool(y.get('chistyy', False)),
                     opisanie=str(y.get('opisanie', '')), rychag=str(y.get('rychag', '')), regen=regen, legs=legs, odin=odin,
                     test_dir=rpath(y.get('test_dir'), R / 'test_locked' / b), step10_dir=rpath(y.get('step10_dir'), R / 'final3' / 'step10'),
                     step10_lock=rpath(y.get('step10_lock'), R / 'test_locked' / 'final3' / 'step10'), wrap_dir=rpath(y.get('wrap_dir'), R / 'final3'),
                     gib_dir=O / 'gibrid' / b, features=rpath(y.get('features_har'), FEAT), yaml=str(y['_file']), osnova=osn, seen={},
                     n_variantov=int(y.get('n_variantov', 0) or 0), trees=y.get('trees'), harx_flag=y.get('harx'),
                     god2021=y.get('god2021'))
            C.append(c)
            used.add((b, h))
    for (b, h), fv in sorted(found.items()):  # OOS есть, yaml нет — в пул не входят (видно в таблице)
        if (b, h) not in used:
            C.append(dict(boost=b, gboost=b, h=h, krug=3, fam='—', regen=fv['regen'], legs=sorted(fv['legs'], key=LEGS2.index), odin=fv['odin'],
                          yaml='—', dopusk=False, prichina='одиночная модель — не V0-кандидат (решение 01:12)' if fv['odin'] else 'нет yaml', seen={},
                          osnova=osnova_of(b, {})))
    # допуск в пул и «видел подтверждающий»
    r2 = {c['boost']: c for c in C if c.get('krug') == 2}
    for c in C:
        c.setdefault('dopusk', True); c.setdefault('prichina', '')
        if c['dopusk'] and not c['legs']:
            c['dopusk'], c['prichina'] = False, 'нет OOS-прогнозов (val шагов 1-9)'
        if 'test_dir' in c and c['dopusk']:
            c['test'] = has_test(c)
            c['s10'] = has_s10(c)
        st = sorted(set(c.get('seen', {}).values()))
        if st:
            c['videl'] = 'не прошёл' if any(s.startswith('не прошёл') for s in st) else 'прошёл'
        elif c.get('osnova') in r2 and r2[c['osnova']]['seen']:
            ob = r2[c['osnova']]
            c['videl'] = ('не прошёл' if any(v.startswith('не') for v in ob['seen'].values()) else 'прошёл') + f' (основа {c["osnova"]})'
        else:
            c['videl'] = 'нет'
    pd.DataFrame([{k: (str(v) if isinstance(v, (Path, dict, list, set)) else v) for k, v in c.items()} for c in C]).to_csv(O / 'pool.csv', index=False) \
        if O.mkdir(parents=True, exist_ok=True) is None else None
    return C


def pool_md(C):
    L = ['| h | кандидат | круг | семейство | рычаг | ноги | yaml | тест 1–9 | шаг 10 | видел подтверждающий | в пуле |', '|' + '---|' * 11]
    for c in sorted(C, key=lambda c: (c['h'], c['krug'], c['boost'])):
        L.append(f"| {c['h']} | {c['boost']} | {c['krug']} | {c.get('fam', '—')} | {c.get('rychag', '')} | {'/'.join(c.get('legs') or []) or '—'} | "
                 f"{'да' if c.get('yaml', '—') != '—' or c['krug'] < 3 else 'нет'} | {'да' if c.get('test') else 'нет'} | {'да' if c.get('s10') else 'нет'} | "
                 f"{c.get('videl', 'нет')} | {'да' if c['dopusk'] else 'нет: ' + c['prichina']} |")
    return L


# ---------------------------------------------------------------- гибрид кандидатов третьего круга (разработочный набор)
def env_of(c):
    e = dict(os.environ)
    if c.get('test_dir') and c['test_dir'] != R / 'test_locked' / c['gboost']:
        e['GIBRID_TEST_DIR'] = str(c['test_dir'])
    return e


def gibrid_cmd(c, out, devlog, extra=()):
    return [sys.executable, '-W', 'ignore', str(HERE / 'gibrid_v17.py'), '--boost', c['gboost'], '--h', str(c['h']), '--features', str(c['features']),
            '--regen', str(c['regen']), '--legs', *c['legs'], '--out', str(out), '--devlog', str(devlog), *extra]


def run_all(jobs, workers):
    def one(j):
        cmd, env, log = j
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, 'w') as f:
            rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO)).returncode
        print(f'{"OK" if rc == 0 else "ОШИБКА " + str(rc)} {log.name} {now()}', flush=True)
        return rc
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, jobs))


def gibrid(a, C):
    O = out_dir(a)
    jobs = []
    for c in C:
        if c['krug'] == 3 and c['dopusk']:
            dv = c['gib_dir'] / f"designs_val_{c['gboost']}_h{c['h']}.csv"
            if dv.exists() and not a.force:
                continue
            jobs.append((gibrid_cmd(c, c['gib_dir'], O / f"gibrid_devlog_{c['gboost']}_h{c['h']}.md"), env_of(c), O / 'logs' / f"gibrid_{c['gboost']}_h{c['h']}.log"))
    print(f'гибрид третьего круга: {len(jobs)} кандидатов', flush=True)
    rc = run_all(jobs, a.workers) if jobs else []
    if any(rc):
        print('ВНИМАНИЕ: часть гибридов упала — эти кандидаты выпадут из пула (см. logs/)', flush=True)


# ---------------------------------------------------------------- правило выбора
def designs(c):
    p = c['gib_dir'] / f"designs_val_{c['gboost']}_h{c['h']}.csv"
    if not p.exists():
        return None
    cols = ['design', 'step', 'val_year', 'q_hyb', 'q_har', 'q_har_raw', 'q_xgb_single', 'q_lgb_single', 'scheme'] + [k + '_h' for k in CR]
    D = pd.read_csv(p, usecols=cols)
    assert D.loc[D.step > 6, 'q_hyb'].isna().all(), f'{p}: в designs_val есть метрики шагов 7-9'
    if c.get('harj'):
        D = D[D.design.str.match(r'^single\|V0_HAR\|B-[a-z]+\|L0$')]
    else:
        D = D[~D.design.str.contains(r'\|V0_HAR\|')]
    return D[D.step.isin(CRIT)]


def summarize(c, D):
    q = D.pivot_table(index='design', columns='step', values='q_hyb', sort=False)
    qh = D.pivot_table(index='design', columns='step', values='q_har', sort=False)
    fl = D.groupby('design', sort=False)[[k + '_h' for k in CR]].first()
    S = pd.DataFrame({'design': q.index, 'mean_1720': q[list(RANK_STEPS)].mean(axis=1).values, 'q2021': q[HOLD].values,
                      'mean_1721': q[list(CRIT)].mean(axis=1).values, 'har_1720': qh[list(RANK_STEPS)].mean(axis=1).values,
                      'har2021': qh[HOLD].values, 'har_1721': qh[list(CRIT)].mean(axis=1).values})
    for k in CR:
        S[k] = fl.loc[S.design, k + '_h'].astype(bool).values
    qr = D.pivot_table(index='design', columns='step', values='q_har_raw', sort=False)
    for s in CRIT:
        S[f'q_{2015 + s}'] = q[s].values
        S[f'har_{2015 + s}'] = qh[s].values
        S[f'raw_{2015 + s}'] = qr[s].values
    S['n_failed'] = (~S[list(CR)]).sum(axis=1)
    S['failed'] = S.apply(lambda r: ','.join(k for k in CR if not r[k]), axis=1)
    return S.sort_values(['n_failed', 'mean_1720'], kind='stable').reset_index(drop=True)


def order(E, h):
    E = E.sort_values(['n_failed', 'mean_1720'], kind='stable').reset_index(drop=True)
    # tie-break: гибрид ставится перед лучшей одиночной моделью, если отстаёт от неё <= 1% по 2017-2020 и нарушает не больше критериев.
    # h=1 — решение владельца 25.10 20:45 (одиночная = чистый бустинг); h=22 — ночное умолчание владельца 26.10 01:45 (одиночная = чистый бустинг
    # или HAR-J с поправкой)
    single = E.chistyy.astype(bool) | ((E.cand == 'HARJ_corr') & (h == 22))
    if h in (1, 22) and single.any():
        v0 = E[single].iloc[0]
        hy = E[~single]
        ok = hy[(hy.n_failed <= v0.n_failed) & (hy.mean_1720 <= v0.mean_1720 * 1.01)]
        idx = list(ok.index) + [v0.name] + [i for i in E.index if i not in set(ok.index) and i != v0.name]
        E = E.loc[idx].reset_index(drop=True)
    return E


def harx_dev(h, code):
    """HAR-X с поправкой на разработочном наборе (только для раскладки и таблицы h=22): OOS QLIKE val-годов шагов 2-6, s² той же оценки
    (gibrid_bias.s2_est) по остаткам HAR-X шагов <= t-1, F = exp(s²/2); строки rv_actual > 1e-12. Шаги 7-9 не читаются. -> {год: QLIKE}."""
    import gibrid_bias as BI
    f = R / f'HARX_h{h}_val.parquet'
    if not f.exists():
        return None
    d = pd.read_parquet(f, columns=['step', 'rv_actual', 'rv_pred'])
    d = d[(d.step <= HOLD) & (d.rv_actual > 1e-12)]
    E = {'har': {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in d.groupby('step')}}
    out = {}
    for st in CRIT:
        g = d[d.step == st]
        fac = 1.0 if code in ('bez', None) else float(np.exp(BI.s2_est(code, E, 'har', st - 1) / 2))
        y, pr = g.rv_actual.values, g.rv_pred.values * fac
        out[2015 + st] = float(np.mean(y / pr - np.log(y / pr) - 1))
    return out


def select(a, C):
    O = out_dir(a)
    O.mkdir(parents=True, exist_ok=True)
    allE, FIN, INC, md = [], [], [], []
    for h in (1, 5, 22):
        rows = []
        for c in [c for c in C if c['h'] == h and c['dopusk']]:
            D = designs(c)
            if D is None or D.empty:
                c['dopusk'], c['prichina'] = False, 'нет designs_val (гибрид не посчитан/упал)'
                continue
            S = summarize(c, D)
            S.to_csv(O / f"designs_rank_{c['boost']}_h{h}.csv", index=False)
            b = S.iloc[0]
            seen = c.get('seen', {})
            rows.append(dict(h=h, cand=c['boost'], gboost=c['gboost'], krug=c['krug'], fam=c['fam'], chistyy=c.get('chistyy', False),
                             design=b.design, **{k: b[k] for k in S.columns if k != 'design'}, videl=c['videl'],
                             videl_design=seen.get(b.design, ''), test=c.get('test', False), s10=c.get('s10', False)))
        if not rows:
            continue
        E = order(pd.DataFrame(rows), h)
        inc = E[E.krug <= 2].iloc[0]
        E['inc_q2021'] = inc.q2021
        E['ok2021'] = np.where(E.krug == 3, E.q2021 <= inc.q2021, np.nan)
        reason = []
        for r in E.itertuples():
            why = []
            if r.videl.startswith('не прошёл'):
                why.append('видел подтверждающий и не прошёл (правило дирижёра, п. 1)')
            if r.krug == 3 and not r.ok2021:
                why.append(f'отложенный 2021: {F(r.q2021)} > {F(inc.q2021)} у текущего лучшего')
            if not r.test:
                why.append('нет тестовых прогнозов шагов 1-9')
            reason.append('; '.join(why))
        E['ne_dopushchen'] = reason
        fams, fin = set(), []
        for i, r in E.iterrows():
            if not r.ne_dopushchen and r.fam not in fams and len(fin) < 3:
                fin.append(i); fams.add(r.fam)
        dup = []
        for i, r in E.iterrows():  # меньше трёх семейств — добор следующими допущенными (↺, объявлено до запуска)
            if len(fin) + len(dup) >= 3:
                break
            if not r.ne_dopushchen and i not in fin:
                dup.append(i)
        E['finalist'] = 0
        for k, i in enumerate(sorted(fin + dup), 1):
            E.loc[i, 'finalist'] = k
        E['fin_note'] = ''
        E.loc[dup, 'fin_note'] = 'семейство повторяется: добор (↺)'
        for i in E.index[E.finalist > 0]:
            r = E.loc[i]
            if r.videl.startswith('прошёл'):
                E.loc[i, 'fin_note'] = (E.loc[i, 'fin_note'] + '; ' if E.loc[i, 'fin_note'] else '') + \
                    'уже прошёл подтверждающий во втором круге — повторная проверка неинформативна' + ('' if r.videl_design else ' (во втором круге проверялся другой дизайн кандидата)')
        E['rank'] = range(1, len(E) + 1)
        E.to_csv(O / f'ranking_h{h}.csv', index=False)
        allE.append(E)
        INC.append(dict(h=h, cand=inc.cand, design=inc.design, q2021=inc.q2021, mean_1720=inc.mean_1720))
        f = E[E.finalist > 0].sort_values('finalist')
        FIN.append(f)
        md += ['', f'### h={h}: рейтинг (OOS; рейтинг — val 2017–2020, критерии — 2017–2021; текущий лучший кругов 1–2 — {inc.cand} {inc.design.replace("|", "/")}, 2021: {F(inc.q2021)})',
               '| № | кандидат | круг | семейство | лучший дизайн | нарушено | 2017–2020 | 2021 | 2017–2021 | HAR-J (та же s²) 2017–2020 | видел подтв. | не допущен | финалист |',
               '|' + '---|' * 13]
        for r in E.itertuples():
            md.append(f'| {r.rank} | {r.cand} | {r.krug} | {r.fam} | {r.design.replace("|", "/")} | {r.failed or "—"} | {F(r.mean_1720)} | {F(r.q2021)} | '
                      f'{F(r.mean_1721)} | {F(r.har_1720)} | {r.videl} | {r.ne_dopushchen or "—"} | {r.finalist or "—"}{" (" + r.fin_note + ")" if r.fin_note else ""} |')
    A = pd.concat(allE, ignore_index=True)
    FN = pd.concat(FIN, ignore_index=True)
    FN.to_csv(O / 'finalists.csv', index=False)
    pd.DataFrame(INC).to_csv(O / 'incumbent.csv', index=False)
    (O / 'select.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md), flush=True)
    if a.zapis and not a.dry:
        import gibrid_designs as GD
        t = now()
        L = [f'Третий круг: финалисты зафиксированы ДО открытия подтверждающего набора ({t}; final3.py select). Правило: рейтинг OOS val 2017-2020, '
             f'критерии c1-c4 на 2017-2021, отложенный 2021 для кандидатов третьего круга, три семейства, правило дирижёра о видевших подтверждающий',
             '| h | № | кандидат | дизайн | семейство | нарушено | 2017–2020 | 2021 | пометка |', '|---|---|---|---|---|---|---|---|---|']
        L += [f'| {r.h} | {r.finalist} | {r.cand} | {r.design.replace("|", "/")} | {r.fam} | {r.failed or "—"} | {F(r.mean_1720)} | {F(r.q2021)} | {r.fin_note or "—"} |'
              for r in FN.itertuples()]
        L += [f'Перебрано дизайн×h на разработочном наборе: 0 (сводка готовых гибридов; дизайны кандидатов третьего круга — в data/regen/final3/gibrid_devlog_*.md)']
        GD.devlog(REPO / 'docs' / 'otchety' / 'dev_log.md', '\n'.join(L))
        (O / 'finalists_logged.json').write_text(json.dumps(dict(time=t, finalists=FN[['h', 'finalist', 'cand', 'design']].to_dict('records')), ensure_ascii=False, indent=1))
        print(f'финалисты записаны в docs/otchety/dev_log.md ({t})', flush=True)
    return A, FN


# ---------------------------------------------------------------- подтверждающий набор (один раз)
def no_finalist_rows(O, have):
    """h без финалистов (все исключены правилом): h=22 — фолбэк владельца HAR-J с поправкой (s² по правилу), подтверждающий не открывается;
    h=1/5 — первый по рейтингу с пометкой."""
    out = []
    for h in (1, 5, 22):
        f = O / f'ranking_h{h}.csv'
        if h in have or not f.exists():
            continue
        E = pd.read_csv(f)
        if h == 22:
            r = E[E.cand == 'HARJ_corr'].iloc[0]
            out.append(dict(h=h, cand='HARJ_corr', design=r.design, mark='финалистов нет (все исключены правилом) — фолбэк владельца: HAR-J с поправкой; подтверждающий на h=22 не открывался', note=''))
        else:
            r = E.iloc[0]
            out.append(dict(h=h, cand=r.cand, design=r.design, mark='финалистов нет — первый по рейтингу с пометкой «подтверждающий не проверялся»', note=''))
    return out


def confirm(a, C):
    O = out_dir(a)
    assert a.confirm and not a.dry, 'подтверждающий открывается только с --confirm и без --dry'
    lock, logged = O / 'CONFIRM_OPENED.json', O / 'finalists_logged.json'
    assert logged.exists(), 'финалисты не записаны в dev_log (select --zapis) — открытие запрещено'
    assert not lock.exists(), f'подтверждающий третьего круга уже открыт ({lock}) — второе открытие запрещено'
    FN = pd.read_csv(O / 'finalists.csv')
    lg = json.loads(logged.read_text())
    assert sorted((d['h'], d['cand'], d['design']) for d in lg['finalists']) == sorted(zip(FN.h, FN.cand, FN.design)), 'finalists.csv не совпадает с записью в dev_log'
    lock.write_text(json.dumps(dict(time=now(), finalists=lg['finalists']), ensure_ascii=False, indent=1))
    byc = {(c['boost'], c['h']): c for c in C}
    groups = {}
    for r in FN.itertuples():
        c = byc[(r.cand, r.h)]
        groups.setdefault((c['gboost'], r.h), (c, []))[1].append(r.design)
    jobs = []
    for (gb, h), (c, ds) in groups.items():
        out = O / 'confirm' / f'{gb}_h{h}'
        fl = ','.join(f'h{h}:{d}' for d in ds)
        jobs.append((gibrid_cmd(c, out, O / f'gibrid_devlog_confirm_{gb}_h{h}.md', ('--confirm', '--confirm-next', '0', '--finalists', fl)), env_of(c),
                     O / 'logs' / f'confirm_{gb}_h{h}.log'))
    rc = run_all(jobs, a.workers)
    assert not any(rc), 'часть подтверждающих прогонов упала — см. logs/ (замок уже стоит: повтор — только решением дирижёра)'
    rows = []
    for r in FN.itertuples():
        c = byc[(r.cand, r.h)]
        cf = pd.read_csv(O / 'confirm' / f"{c['gboost']}_h{r.h}" / f"confirm_{c['gboost']}_h{r.h}.csv")
        x = cf[cf.design == r.design].iloc[0]
        rows.append(dict(h=r.h, finalist=r.finalist, cand=r.cand, design=r.design, fam=r.fam, fin_note=r.fin_note if isinstance(r.fin_note, str) else '',
                         ok=bool(x.ok), **{k: bool(x[k]) for k in CR}, q_2022=x.get('q_2022'), q_2023=x.get('q_2023'), q_2024=x.get('q_2024'),
                         mean_confirm=x.mean_q_confirm, **{k: x[k] for k in x.index if str(k).startswith('ref_')}))
    CF = pd.DataFrame(rows, columns=['h', 'finalist', 'cand', 'design', 'fam', 'fin_note', 'ok'] if not rows else None)
    CF.to_csv(O / 'confirm_finalists.csv', index=False)
    A = pd.concat([pd.read_csv(O / f'ranking_h{h}.csv') for h in (1, 5, 22) if (O / f'ranking_h{h}.csv').exists()])
    ch = []
    for h, g in CF.groupby('h'):
        g = g.sort_values('finalist')
        ok = g[g.ok]
        if len(ok):
            r = ok.iloc[0]
            ch.append(dict(h=h, cand=r.cand, design=r.design, mark='прошёл (М4: первый прошедший)', note=r.fin_note))
        elif h == 22:
            hj = A[(A.h == 22) & (A.cand == 'HARJ_corr')].iloc[0]
            ch.append(dict(h=h, cand='HARJ_corr', design=hj.design, mark='никто из финалистов не прошёл — фолбэк владельца: HAR-J с поправкой', note=''))
        else:
            r = g.iloc[0]
            ch.append(dict(h=h, cand=r.cand, design=r.design, mark='никто не прошёл — первый финалист с пометкой «не прошёл подтверждающий набор»', note=r.fin_note))
    ch += no_finalist_rows(O, set(CF.h) if len(CF) else set())
    pd.DataFrame(ch).sort_values('h').to_csv(O / 'final_choice.csv', index=False)
    print(CF.to_string(index=False) if len(CF) else 'финалистов нет ни на одном h')
    print(pd.DataFrame(ch).to_string(index=False), flush=True)


# ---------------------------------------------------------------- тестовые прогнозы и σ_pred
def finals(a):
    O = out_dir(a)
    if (O / 'final_choice.csv').exists() and not a.dry:
        return pd.read_csv(O / 'final_choice.csv')
    FN = pd.read_csv(O / 'finalists.csv')  # сухой прогон: первый финалист как заглушка
    f = FN[FN.finalist == 1][['h', 'cand', 'design']].assign(mark='сухой прогон: первый финалист (подтверждающий не открывался)', note='')
    return pd.concat([f, pd.DataFrame(no_finalist_rows(O, set(f.h)))], ignore_index=True).sort_values('h').reset_index(drop=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def add_harx(p, r):
    """Столбец harx_corr — HAR-X с поправкой той же оценки s², что у финального дизайна (bez -> ord), тест HARX — оценка шага k.
    ТОЛЬКО для таблицы раскладки приложения (решение владельца 01:12); эталон DM/MCS/H1 — har_corr. Без метрик. -> '' или текст ошибки."""
    T = pd.read_parquet(p)
    K = ['date', 'ticker', 'step']
    code = r.design.split('|')[2].replace('B-', '').replace('bez', 'ord')
    xf = R / 'test_locked' / 'HARX' / f'HARX_h{r.h}_test.parquet'
    if not xf.exists():
        T['harx_corr'] = np.nan
        T.to_parquet(p, index=False)
        return f'нет {xf.name}'
    X = pd.read_parquet(xf, columns=K + [f'rv_pred_{code}']).rename(columns={f'rv_pred_{code}': 'harx_corr'})
    T = T.drop(columns=[c for c in ('harx_corr',) if c in T]).merge(X, on=K, how='left')
    T.to_parquet(p, index=False)
    n = int(T.harx_corr.isna().sum())
    return f'harx_corr: {n} пропусков' if n else ''


def build(a, C):
    O, Lk = out_dir(a), lock_dir(a)
    Lk.mkdir(parents=True, exist_ok=True)
    byc = {(c['boost'], c['h']): c for c in C}
    info = []
    for r in finals(a).itertuples():
        c = byc[(r.cand, r.h)]
        leg, sel, lab, lam = r.design.split('|')[:4]
        s10 = c.get('s10') and sel.startswith('V0_') and sel != 'V0_HCJ' and lam == 'L0'
        cmd = [sys.executable, '-W', 'ignore', str(HERE / 'final2_build.py'), '--boost', c['gboost'], '--h', str(r.h), '--design', r.design,
               '--out', str(Lk), '--regen', str(c['regen']), '--features', str(c['features']), '--step10-dir', str(c['step10_dir']),
               '--step10-lock', str(c['step10_lock']), '--wrap-dir', str(c['wrap_dir'])] + ([] if s10 else ['--no-step10'])
        log = O / 'logs' / f'build_h{r.h}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, 'w') as f:
            rc = subprocess.run(cmd, env=env_of(c), stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO)).returncode
        p = Lk / f'final_h{r.h}_test.parquet'
        ok = rc == 0 and p.exists()
        harx_err = add_harx(p, r) if ok else 'сборка упала'
        info.append(dict(h=r.h, cand=r.cand, design=r.design, step10=bool(s10), rc=rc, harx_err=harx_err, path=str(p), sha256=sha(p) if ok else '',
                         rows=len(pd.read_parquet(p, columns=['step'])) if ok else 0))
        print(f'h={r.h}: сборка {"OK" if ok else "ОШИБКА"} ({r.cand} {r.design}; шаг 10: {"да" if s10 else "нет"}) {now()}', flush=True)
    I = pd.DataFrame(info)
    sig = dict(h1_changed=False)
    h1 = I[I.h == 1]
    if len(h1) and (h1.cand.iloc[0], h1.design.iloc[0]) != H1_FINAL2 and h1.rc.iloc[0] == 0:
        T = pd.read_parquet(h1.path.iloc[0], columns=['date', 'ticker', 'step', 'rv_pred'])
        sp = T.assign(sigma_pred_h1=np.sqrt(T.rv_pred.values).astype('float32')).sort_values(['date', 'ticker'])
        sp_path = (O if a.dry else R) / 'sigma_pred_h1_final3.parquet'
        sp[['date', 'ticker', 'sigma_pred_h1']].to_parquet(sp_path, index=False)
        sp.groupby('step').date.agg(['min', 'max', 'size']).to_csv(sp_path.with_name('sigma_pred_h1_final3_steps.csv'))
        sig = dict(h1_changed=True, path=str(sp_path), sha256=sha(sp_path), rows=len(sp), first=str(sp.date.min().date()), last=str(sp.date.max().date()))
    I.to_csv(O / 'build.csv', index=False)
    (O / 'sigma_pred.json').write_text(json.dumps(sig, ensure_ascii=False, indent=1))
    return I


# ---------------------------------------------------------------- черновик FINAL
def versions():
    v = [f'Python {sys.version.split()[0]}']
    for m in ('lightgbm', 'xgboost', 'numpy', 'pandas', 'optuna', 'scipy', 'sklearn'):
        try:
            v.append(f'{"scikit-learn" if m == "sklearn" else m} {__import__(m).__version__}')
        except Exception:
            v.append(f'{m} —')
    return ', '.join(v)


def count_designs(O):
    rx = re.compile(r'Перебрано дизайн×h на разработочном наборе: (\d+)')
    files = [f for f in O.glob('gibrid_devlog_*.md') if '_confirm_' not in f.name]
    for p in R3_PREFIX:
        for d in R.glob(p + '*'):
            if d.is_dir():
                files += [f for f in d.rglob('gibrid_devlog*.md') if 'confirm' not in f.name]
    per = {str(f.relative_to(R)): sum(int(x) for x in rx.findall(f.read_text())) for f in files}
    return per


def d166():
    pats = ['D16*/**/*16_6*.csv', 'D16*/**/*raskladka*.csv', 'D16*/*16_6*.csv', 'final3/*16_6*.csv']
    for p in pats:
        fs = sorted(R.glob(p))
        if fs:
            return fs[0]
    return None


def md_table(df):
    cols = list(df.columns)
    return ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)] + \
           ['| ' + ' | '.join(F(v) if isinstance(v, float) else str(v) for v in r) + ' |' for r in df.itertuples(index=False)]


def trees_h22(c):
    """Число деревьев (best_iteration) финалиста h=22 по шагам 1-9 и шагу 10: {regen}/{boost}_selected_h22.csv (только столбцы model, step, seed,
    best_iteration — qlike_val не читается), шаг 10 — {step10_dir}/{boost}_h22_s10_trees.csv; yaml-ключ trees — свой CSV (model, step, best_iteration)."""
    out = []
    if c.get('regen') is None:
        return 'нет данных'
    p = Path(c['trees']) if c.get('trees') else Path(c['regen']) / f"{c['gboost']}_selected_h22.csv"
    if p.exists():
        cols = [x for x in ('model', 'step', 'seed', 'best_iteration') if x in pd.read_csv(p, nrows=0).columns]
        t = pd.read_csv(p, usecols=cols)
        if 'seed' in t:
            t = t[t.seed == 42]
        for m, g in t.groupby('model'):
            out.append(f'{m}: ' + ' '.join(str(int(x)) for x in g.sort_values('step').best_iteration))
    q = Path(c['step10_dir']) / f"{c['gboost']}_h22_s10_trees.csv" if c.get('step10_dir') else None
    if q is not None and q.exists():
        t = pd.read_csv(q, usecols=['model', 'seed', 'best_iteration'])
        out.append('шаг 10 (s42/s43/s44): ' + '; '.join(f'{m} ' + '/'.join(str(int(x)) for x in g.sort_values('seed').best_iteration)
                                                     for m, g in t.groupby('model')))
    return '; '.join(out) if out else 'нет данных (V0_HAR — деревьев нет, или файл деревьев кандидата не выложен)'


def uses_harx(c):
    v = c.get('harx_flag')
    if v is not None:
        return bool(v)
    b = str(c.get('boost', ''))
    return 'harx' in b.lower() or b.endswith('X') or 'HAR-X' in str(c.get('opisanie', '')) + str(c.get('fam', ''))


def extra_fields(a, O, Lk, A, FN, FC, CF, byc):
    """Обязательные поля ночного режима владельца (01:45/01:50)."""
    L = ['', '## 6а. Обязательные поля (ночной режим владельца 01:45/01:50)', '',
         '- **s² с шага t−1:** на валидации поправка шага t берёт оценку s² шага t−1 (OOS; шаг 1 — F = 1, в метрику не входит); на тесте шага k — оценка шага k.',
         '- **Шаг 10:** обучение ≤ 2024, ранняя остановка и s² по val 2025, прогноз 01–02.2026 (наличие по h — раздел 6).',
         '- **Три seed:** 42 / 43 / 44 (нога seeds3 — среднее трёх; single/top5 — seed 42; приём рычагов третьего круга — по 3 seed).']
    code = [HERE / f for f in ('final3.py', 'final3.sh', 'gibrid_v17.py', 'gibrid_io.py', 'gibrid_designs.py', 'gibrid_compute.py', 'gibrid_bias.py',
                               'gibrid_har.py', 'final2_build.py', 'final2_step10.py', 'final_test.py', 'ftp_tables.py', 'ftp_stats.py', 'harx.py')]
    L += ['- **SHA-256 кода:**'] + [f'  - `scripts/analysis/{f.name}` — `{sha(f)}`' if f.exists() else f'  - `scripts/analysis/{f.name}` — нет файла' for f in code]
    dat = [R / f'HARX_h{h}_val.parquet' for h in (1, 5, 22)] + [R / 'test_locked' / 'HARX' / f'HARX_h{h}_test.parquet' for h in (1, 5, 22)]
    L += ['- **SHA-256 данных HAR-X** (матрицы признаков — раздел 6; время файла по часам Мака):'] + \
         [f'  - `{f.relative_to(R)}` ({time.strftime("%d.%m %H:%M", time.localtime(f.stat().st_mtime))}) — `{sha(f)}`' for f in dat if f.exists()]
    L += ['', '- **Версия HAR-X у каждого финалиста:**', '', '| h | финалист | HAR-X в дизайне | файл HAR-X (время) | SHA-256 |', '|---|---|---|---|---|']
    for r in FN.sort_values(['h', 'finalist']).itertuples():
        c = byc[(r.cand, r.h)]
        f = R / f'HARX_h{r.h}_val.parquet'
        if uses_harx(c) and f.exists():
            L.append(f'| {r.h} | {r.cand} | да (нога гибрида) | `{f.name}` ({time.strftime("%d.%m %H:%M", time.localtime(f.stat().st_mtime))}) | `{sha(f)}` |')
        else:
            L.append(f'| {r.h} | {r.cand} | нет (нога — HAR-J) | — | — |')
    L += ['', '- **V9 против V0 на h=1** (OOS QLIKE, разработочный набор; подтверждающий второго круга — `2026-10-25_final2.md`):', '',
          '| кандидат | лучший дизайн | 2017–2020 | 2021 | 2017–2021 | подтв. круг 2 (среднее) | к V0 (2017–2021) |', '|---|---|---|---|---|---|---|']
    E1 = A[A.h == 1].set_index('cand')
    conf2 = {'D12_h1_A_d812': 0.4711, 'D12_h1_B_b0.25': 0.4718, 'D12_h1_V0': 0.4600}
    v0q = E1.loc['D12_h1_V0', 'mean_1721'] if 'D12_h1_V0' in E1.index else np.nan
    for cand, nm in (('D12_h1_B_b0.25', 'V9, β = 0,25'), ('D12_h1_A_d812', 'Д8.12'), ('D12_h1_V0', 'V0, чистый бустинг')):
        if cand in E1.index:
            e = E1.loc[cand]
            L.append(f'| {cand} ({nm}) | {e.design.replace("|", "/")} | {F(e.mean_1720)} | {F(e.q2021)} | {F(e.mean_1721)} | {F(conf2[cand])} | {P(e.mean_1721 / v0q - 1)} |')
    L += ['', '  V9 с β = 0,5 / 0,75 / 1 — +1,0 / +2,2 / +4,3% к V0 (второй круг, отброшено). На подтверждающем втором круге V9 β = 0,25 не прошёл (c2), V0 прошёл.']
    L += ['', '- **Число деревьев на h=22 у финалистов** (best_iteration по шагам 1–9, seed 42; у XGBoost отсчёт от 0):']
    fin22 = set(FN[FN.h == 22].cand)
    rows22 = list(FN[FN.h == 22].itertuples()) + [r for r in FC[FC.h == 22].itertuples() if r.cand not in fin22]
    if not rows22:
        L.append('  - финалистов на h=22 нет')
    for r in rows22:
        tr = 'HAR-J с поправкой — деревьев нет' if '|V0_HAR|' in r.design else trees_h22(byc[(r.cand, 22)])
        L.append(f'  - {r.cand} ({r.design.replace("|", "/")}): {tr}')
    L += ['', '- **2021 использован один раз** — как отложенный год третьего круга: в сборке рейтинг считается только по шагам 2–5 (2017–2020); 2021 (шаг 6) '
          'входит лишь в проверку шага 6 для кандидатов третьего круга и в критерии c1–c4 на 2017–2021 (как в кругах 1–2). Для кандидатов кругов 1–2 2021 '
          'отложенным не был (входил в их разработочный набор). Отметки агентов третьего круга о 2021 (yaml-ключ `god2021`):']
    k3 = [c for c in byc.values() if c.get('krug') == 3 and c.get('dopusk')]
    L += [f'  - {c["boost"]} h={c["h"]}: {c.get("god2021") or "не указано"}' for c in k3] or ['  - кандидатов третьего круга в пуле нет']
    return L


def report(a, C):
    O, Lk = out_dir(a), lock_dir(a)
    A = pd.concat([pd.read_csv(O / f'ranking_h{h}.csv') for h in (1, 5, 22) if (O / f'ranking_h{h}.csv').exists()], ignore_index=True)
    FN = pd.read_csv(O / 'finalists.csv')
    FC = finals(a)
    CF = pd.read_csv(O / 'confirm_finalists.csv') if (O / 'confirm_finalists.csv').exists() and not a.dry else None
    B = pd.read_csv(O / 'build.csv') if (O / 'build.csv').exists() else None
    sig = json.loads((O / 'sigma_pred.json').read_text()) if (O / 'sigma_pred.json').exists() else {}
    lg = json.loads((O / 'finalists_logged.json').read_text()) if (O / 'finalists_logged.json').exists() else {}
    lk = json.loads((O / 'CONFIRM_OPENED.json').read_text()) if (O / 'CONFIRM_OPENED.json').exists() else {}
    byc = {(c['boost'], c['h']): c for c in C}
    t = now()
    L = [f'# FINAL_DESIGN — модели главы 3, третий круг (черновик на «да» владельца, {t}{"; СУХОЙ ПРОГОН" if a.dry else ""})', '',
         '> Имя `FINAL_DESIGN_chernovik.md` — чтобы не снять замок теста. `FINAL_DESIGN.md` не создаётся: после «да» дирижёр переименует файл и один раз '
         'запустит `python3 scripts/analysis/final_test.py --scope test --src data/regen/test_locked/final3`. Метрик на тестовых годах (2017–2026) нет. '
         'Черновик второго круга заменён этим (FINAL второго круга не фиксировался — решение владельца 00:55). Сборка — `scripts/analysis/final3.py`, '
         'отчёт — `docs/otchety/2026-10-26_final3.md`.', '',
         '**Подтверждающий набор использовался трижды, не более трёх кандидатов на горизонт за раз.**', '']
    # 1. финальный дизайн
    L += ['## 1. Финальный дизайн по горизонтам', '',
          '| h | кандидат | семейство | дизайн (нога / схема / s² / усадка) | описание | разраб. 2017–2020 | отложенный 2021 | разраб. 2017–2021 | подтв. 2022 / 2023 / 2024 (средн.) | HAR-J с поправкой (та же s²): 2017–2021 / подтв. | итог |',
          '|' + '---|' * 11]
    for r in FC.itertuples():
        e = A[(A.h == r.h) & (A.cand == r.cand)].iloc[0] if len(A[(A.h == r.h) & (A.cand == r.cand)]) else None
        S = pd.read_csv(O / f'designs_rank_{r.cand}_h{r.h}.csv')
        s = S[S.design == r.design].iloc[0]
        c = byc[(r.cand, r.h)]
        cf = CF[(CF.h == r.h) & (CF.cand == r.cand) & (CF.design == r.design)] if CF is not None else pd.DataFrame()
        cq = (f'{F(cf.q_2022.iloc[0])} / {F(cf.q_2023.iloc[0])} / {F(cf.q_2024.iloc[0])} ({F(cf.mean_confirm.iloc[0])})' if len(cf) else 'не открывался' if CF is None else 'не проверялся (фолбэк)')
        hc = (np.nanmean([cf[f'ref_q_har_{y}'].iloc[0] for y in (2022, 2023, 2024)]) if len(cf) and 'ref_q_har_2022' in cf else np.nan)
        L.append(f'| {r.h} | {r.cand} | {c["fam"]} | {r.design.replace("|", " / ")} | {c.get("opisanie", "")} | {F(s.mean_1720)} | {F(s.q2021)} | {F(s.mean_1721)} | {cq} | '
                 f'{F(s.har_1721)} / {F(hc)} | {r.mark}{"; " + r.note if isinstance(r.note, str) and r.note else ""} |')
    # 2. финалисты и подтверждающий
    L += ['', f'## 2. Финалисты третьего круга (записаны в `dev_log.md` {lg.get("time", "—")}, до открытия) и подтверждающий набор (открыт {lk.get("time", "—")})', '',
          '| h | № | кандидат | дизайн | семейство | нарушено (2017–2021) | 2017–2020 | 2021 | c1 | c2 | c3 | c4 | 2022 | 2023 | 2024 | средн. | итог | пометка |', '|' + '---|' * 18]
    for r in FN.sort_values(['h', 'finalist']).itertuples():
        cf = CF[(CF.h == r.h) & (CF.cand == r.cand)] if CF is not None else pd.DataFrame()
        cc = (' | '.join('+' if cf[k].iloc[0] else '−' for k in CR) + f' | {F(cf.q_2022.iloc[0])} | {F(cf.q_2023.iloc[0])} | {F(cf.q_2024.iloc[0])} | {F(cf.mean_confirm.iloc[0])} | '
              f'{"прошёл" if cf.ok.iloc[0] else "не прошёл"}') if len(cf) else ' | '.join(['—'] * 8) + ' | не открывался'
        L.append(f'| {r.h} | {r.finalist} | {r.cand} | {r.design.replace("|", "/")} | {r.fam} | {r.failed if isinstance(r.failed, str) and r.failed else "—"} | '
                 f'{F(r.mean_1720)} | {F(r.q2021)} | {cc} | {r.fin_note if isinstance(r.fin_note, str) and r.fin_note else "—"} |')
    # 3. правила (объявлены до запуска)
    L += ['', '## 3. Правило выбора третьего круга (объявлено до запуска, `2026-10-26_final3.md`)', '',
          '1. Пул: финалисты второго круга (D12_h1_A_d812, D12_h1_B_b0.25, D12_h1_V0, F2m_h5, D12_h5, L2v3_h5, D12_h22), HAR-J с поправкой (фолбэк h=22) '
          'и кандидаты третьего круга с OOS-прогнозами и yaml к запуску сборки (HAR-X — только как нога гибрида: одиночные модели без бустинга в пул не входят, решение 01:12). '
          'Кандидат представлен своим лучшим дизайном по правилу.',
          '2. Рейтинг: число нарушенных критериев c1–c4 (val 2017–2021, OOS: параметры и s² шага t−1 -> val шага t; val 2016 = шаг 1 даёт s² для шага 2; '
          'эталон — модели с той же s²; V0 наравне, сам с собой не сравнивается), затем средний OOS QLIKE val 2017–2020 (шаги 2–5). '
          'Tie-break: гибрид ставится перед лучшей одиночной моделью, если отстаёт от неё не больше чем на 1% по 2017–2020 и нарушает не больше критериев; '
          'h=1 — решение владельца 25.10 (одиночная — чистый бустинг), h=22 — ночное умолчание владельца 26.10 01:45 (одиночная — чистый бустинг или HAR-J с поправкой).',
          '3. Отложенный 2021 (шаг 6) — только для кандидатов третьего круга: «текущий лучший по правилу» = первый в рейтинге п. 2 среди кандидатов кругов 1–2 '
          '(финалисты второго круга и HAR-J с поправкой). Кандидат третьего круга проходит шаг 6, если его OOS QLIKE val 2021 ≤ OOS QLIKE val 2021 текущего лучшего '
          '(без допуска). Не прошедший в финалисты не берётся.',
          '4. Правило дирижёра о видевших подтверждающий (до открытия): (1) дизайн, не прошедший подтверждающий в кругах 1–2, и его варианты, отличающиеся только '
          'рычагом третьего круга, в финалисты не берутся (кандидат исключается целиком; вариант определяется ключом `osnova` yaml или именем основы в имени файла); '
          '(2) прошедший может быть финалистом с пометкой «уже прошёл подтверждающий во втором круге — повторная проверка неинформативна»; (3) новые — без ограничений.',
          '5. Три финалиста на h: по порядку рейтинга, первый допущенный из каждого семейства; если допущенных семейств меньше трёх — добор следующими допущенными '
          '(пометка ↺). Кандидат без тестовых прогнозов шагов 1–9 финалистом быть не может.',
          '6. Подтверждающий (val 2022–2024, шаги 7–9) — одно открытие, `gibrid_v17.py --confirm --confirm-next 0`; критерии c1–c4 на шагах 7–9; М4 — первый прошедший '
          'по порядку финалистов; h=22 — если никто, HAR-J с поправкой (s² по правилу); h=1/5 — если никто, первый финалист с пометкой.', '']
    # 4. пул
    L += ['## 4. Пул кандидатов (все круги)', ''] + pool_md(C)
    L += ['', '### Рейтинг по h (разработочный набор)', ''] + (O / 'select.md').read_text().splitlines()
    # 5. четыре цели
    L += ['', '## 5. Четыре цели владельца (21:40) — разработочный (2017–2021, OOS) и подтверждающий наборы, финальный дизайн', '',
          '| h | набор | (1) гибрид лучше HAR-J с поправкой | (2) гибрид не хуже ни одной модели | (3) чистый бустинг не хуже HAR-J с поправкой | (4) лучше GRU |', '|---|---|---|---|---|---|']
    for r in FC.itertuples():
        S = pd.read_csv(O / f'designs_rank_{r.cand}_h{r.h}.csv')
        s = S[S.design == r.design].iloc[0]
        code = r.design.split('|')[2]
        yrs = [s[f'q_{y}'] for y in range(2017, 2022)]
        c = byc[(r.cand, r.h)]
        D = designs(c)
        hy = D[D.design == r.design].sort_values('step')
        better = int((hy.q_hyb.values < hy.q_har.values).sum())
        g1 = 'финал — сам HAR-J' if r.cand == 'HARJ_corr' else f'{"да" if better == 5 else "нет"}: {F(s.mean_1721)} против {F(s.har_1721)}, {P(s.mean_1721 / s.har_1721 - 1)}, {better} из 5 лет'
        # (2): лучшие одиночные модели (V0_XGB/LGB/HCJ и HAR-J) с той же s² среди всех кандидатов пула h
        best = None
        for cc in [x for x in C if x['h'] == r.h and x['dopusk']]:
            Dx = designs(cc)
            if Dx is None:
                continue
            Dx = Dx[Dx.design.str.contains(r'\|V0_') & Dx.design.str.contains(r'\|' + re.escape(code) + r'\|')]
            if (cc['boost'], r.design) == (r.cand, r.design):
                Dx = Dx[Dx.design != r.design]
            m = Dx.groupby('design').q_hyb.mean()
            if len(m) and (best is None or m.min() < best[1]):
                best = (f'{cc["boost"]}: {m.idxmin().replace("|", "/")}', m.min())
        hj = s.har_1721
        if best is None or hj < best[1]:
            best = ('HAR-J с поправкой', hj) if r.cand != 'HARJ_corr' else best
        g2 = f'{"да" if s.mean_1721 <= best[1] else "нет"}: лучшая одиночная — {best[0]} {F(best[1])}, финал {P(s.mean_1721 / best[1] - 1)}' if best else '—'
        pure = A[(A.h == r.h) & (A.chistyy)]
        g3 = (f'{"да" if pure.mean_1721.iloc[0] <= pure.har_1721.iloc[0] else "нет"}: {pure.cand.iloc[0]} {F(pure.mean_1721.iloc[0])} против {F(pure.har_1721.iloc[0])}, '
              f'{P(pure.mean_1721.iloc[0] / pure.har_1721.iloc[0] - 1)}') if len(pure) else 'чистого бустинга в пуле нет (второй круг: h=5 Д15 −6,0%, h=22 +10,6%)'
        g4 = 'да (справочный GRU, решение дирижёра 00:05; финал третьего круга против GRU не перепроверялся)'
        L.append(f'| {r.h} | разработочный | {g1} | {g2} | {g3} | {g4} |')
        cf = CF[(CF.h == r.h) & (CF.cand == r.cand) & (CF.design == r.design)] if CF is not None else pd.DataFrame()
        if len(cf):
            q = [cf[f'q_{y}'].iloc[0] for y in (2022, 2023, 2024)]
            hq = [cf[f'ref_q_har_{y}'].iloc[0] for y in (2022, 2023, 2024)]
            xq = [min(cf[f'ref_q_xgb_single_{y}'].iloc[0], cf[f'ref_q_lgb_single_{y}'].iloc[0]) for y in (2022, 2023, 2024)]
            nb = sum(a_ < b_ for a_, b_ in zip(q, hq))
            others = CF[(CF.h == r.h) & ~((CF.cand == r.cand) & (CF.design == r.design))]
            ob = f'; другие финалисты: {", ".join(f"{o.cand} {F(o.mean_confirm)}" for o in others.itertuples())}' if len(others) else ''
            pc = CF[(CF.h == r.h) & CF.cand.map(lambda x: bool(byc.get((x, r.h), {}).get('chistyy')))]
            L.append(f'| {r.h} | подтверждающий | {"финал — сам HAR-J" if r.cand == "HARJ_corr" else ("да" if nb == 3 else "нет") + f": {F(np.mean(q))} против {F(np.mean(hq))}, {P(np.mean(q) / np.mean(hq) - 1)}, {nb} из 3 лет"} | '
                     f'одиночные своей ноги (лучшая по году): {F(np.mean(xq))}, финал {P(np.mean(q) / np.mean(xq) - 1)}{ob} | '
                     f'{(pc.cand.iloc[0] + " " + F(pc.mean_confirm.iloc[0])) if len(pc) else "не открывался (чистый бустинг не финалист)"} | GRU на подтверждающем нет |')
        else:
            L.append(f'| {r.h} | подтверждающий | {"не открывался" if CF is None else "финал не проверялся (фолбэк)"} | — | — | — |')
    # 6. поля дизайна
    mats = [FEAT, R / 'features_pit_v3.config.json', R / 'features_pit_v3_d7_h5.parquet', R / 'features_pit_v3_d7_h22.parquet', R / 'features_pit_v3_rrv.parquet']
    mats += sorted({c['features'] for c in C if c.get('features') and c['features'] not in mats})
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=str(REPO)).stdout.strip()
    branch = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True, text=True, cwd=str(REPO)).stdout.strip()
    L += ['', '## 6. Поля дизайна', '',
          '- Схема и поправка: дизайн = нога бустинга | выбор схемы (V0 — одна компонента) | поправка Б0.1 (F = exp(s²/2), одна оценка s² для всех моделей шага) | усадка. '
          'На валидации — вне выборки (оценка шага t−1); на тесте шага k — оценка шага k по val-остаткам до val-года шага k включительно.',
          '- **s² на шаге 1:** на тесте шага 1 — по остаткам val-года 2016 (sm/smrob — «сколько есть»: только 2016, до трёх лет к шагу 3); на валидации шага 1 F = 1, '
          'шаг 1 в OOS-метрику не входит.',
          '- **Шаг 10:** обучение на годах ≤ 2024, ранняя остановка и s² по val 2025 (sm/smrob — 2023–2025), прогноз 01–02.2026; точка — точка шага 6.',
          '- Seed 42 (seeds3 — 42/43/44); ранняя остановка по val-году шага, patience 30.']
    for r in FC.itertuples():
        c = byc[(r.cand, r.h)]
        bf = c['gib_dir'] / f"bias_factors_{c['gboost']}_h{r.h}.csv"
        code = r.design.split('|')[2].replace('B-', '')
        comp = {'V0_XGB': 'xgb', 'V0_LGB': 'lgb', 'V0_HAR': 'har', 'V0_HCJ': 'hcj'}.get(r.design.split('|')[1], 'har')
        fs = ''
        if bf.exists() and code != 'bez':
            b = pd.read_csv(bf)
            b = b[(b.code == code) & (b.comp == comp)].drop_duplicates('step').sort_values('step')
            fs = ' / '.join(F(x, 3) for x in b.F_own)
        f10 = Lk / f'final_h{r.h}_step10_factors.csv'
        f10s = F(float(pd.read_csv(f10)[f'F_{comp}'].iloc[0]), 3) if f10.exists() and comp in ('har', 'xgb', 'lgb') else 'нет'
        bb = B[B.h == r.h] if B is not None else pd.DataFrame()
        L.append(f'- h={r.h}: множители F ({code}, компонента {comp}) на тесте шагов 1–9: {fs or "—"}; шаг 10: {f10s}. Тестовые прогнозы: '
                 f'`{bb.path.iloc[0] if len(bb) else "—"}` ({bb.rows.iloc[0] if len(bb) else 0} строк, шаг 10: {"да" if len(bb) and bb.step10.iloc[0] else "нет"}), '
                 f'SHA-256 `{bb.sha256.iloc[0] if len(bb) else "—"}`.')
    L.append('- σ_pred(h=1) для главы 4: ' + (f'финал h=1 изменился — `{sig["path"]}` ({sig["rows"]} строк, {sig["first"]}–{sig["last"]}; sigma_pred_h1 = √rv_pred финального теста), '
                                             f'SHA-256 `{sig["sha256"]}`; главу 4 перевести на этот файл.' if sig.get('h1_changed') else
                                             'финал h=1 не изменился (D12_h1_A_d812 top5/V0_XGB/B-sm/L0) — остаётся `data/regen/sigma_pred_h1_final2.parquet`.'))
    L += ['- **SHA-256 матриц:**'] + [f'  - `{m.name}` — `{sha(m)}`' for m in mats if Path(m).exists()]
    L += [f'- **Код:** ветка `{branch}`, коммит `{commit}` (база `kanon-v22-pereschet`). **Версии пакетов:** {versions()}.']
    L += extra_fields(a, O, Lk, A, FN, FC, CF, byc)
    # 7. честность выбора
    per = count_designs(O)
    n3 = sum(per.values())
    nv = sum(c.get('n_variantov', 0) for c in C if c.get('krug') == 3)
    L += ['', '## 7. Честность выбора', '',
          f'- Число дизайнов на разработочном наборе за все круги: круги 1–2 — {R2_TOTAL[0]} дизайн×h гибрида и {R2_TOTAL[1]} вариантов бустинга; '
          f'третий круг — {n3} дизайн×h гибрида по журналам ({len(per)} журналов) и {nv} вариантов бустинга по yaml кандидатов; '
          f'**итого {R2_TOTAL[0] + n3} дизайн×h и {R2_TOTAL[1] + nv} вариантов бустинга.**']
    L += [f'  - `{k}`: {v}' for k, v in sorted(per.items())]
    L += ['- **Подтверждающий набор использовался трижды, не более трёх кандидатов на горизонт за раз.** Все случаи просмотра:',
          f'  - **Итого: {3 if lk else 2} открытия подтверждающего набора и 5 случайных (частичных) просмотров** — круг 1: 1 открытие + 5 частичных просмотров; '
          f'круг 2: 1 открытие; круг 3: {"1 открытие" if lk else "не открывался"}.',
          '  1. Первый круг (25.10, 19:55): одно открытие — 30 проверенных дизайнов (финалисты и следующие по рейтингу на h=22); до него 5 случаев частичного '
          'просмотра (таблица — `2026-10-25_final_assembly.md`).',
          '  2. Второй круг (26.10, 00:27): одно открытие, 3 финалиста на h (9 дизайнов), список зафиксирован до открытия (`dev_log.md` № 48).',
          f'  3. Третий круг ({lk.get("time", "не открывался")}): одно открытие, {len(FN)} финалистов (≤ 3 на h), список в `dev_log.md` ({lg.get("time", "—")}) до открытия.',
          '  - Проверка сборки: во всех designs_val кандидатов пула метрики шагов 7–9 пусты (assert в final3.py); правило читает только шаги 2–6.',
          '- Цели (21:40) и правило третьего круга (00:55) объявлены по ходу разработки; уточнение дирижёра о видевших подтверждающий — до открытия.']
    # 8. отброшенное
    L += ['', '## 8. Отброшено (разработочный набор, OOS)', '', '| h | кандидат | круг | лучший дизайн | нарушено | 2017–2020 | 2021 | причина |', '|---|---|---|---|---|---|---|---|']
    fin_set = set(zip(FC.h, FC.cand))
    for r in A.itertuples():
        if (r.h, r.cand) in fin_set:
            continue
        why = r.ne_dopushchen if isinstance(r.ne_dopushchen, str) and r.ne_dopushchen else ('финалист, не выбран по М4' if r.finalist else 'ниже в рейтинге / семейство уже представлено')
        L.append(f'| {r.h} | {r.cand} | {r.krug} | {r.design.replace("|", "/")} | {r.failed if isinstance(r.failed, str) and r.failed else "—"} | {F(r.mean_1720)} | {F(r.q2021)} | {why} |')
    for c in C:
        if not c['dopusk']:
            L.append(f'| {c["h"]} | {c["boost"]} | {c["krug"]} | — | — | — | — | не в пуле: {c["prichina"]} |')
    L += ['- Круги 1–2 (с числами — `2026-10-25_final2.md`): уровень 2 (gamma, цель-отношение на V9, узкое пространство, extra_trees, Д10), Д11.1–11.5, Д12.1–12.4, '
          'Д13 (кроме монотонности h=5), F2_h5 (с Huber, 0,2073), чистый V0 Bv3 (h=5 0,2207, h=22 0,2163), CatBoost (отменён владельцем).']
    # 9. отступления
    dev = ['Рейтинг третьего круга — по 2017–2020, критерии — по 2017–2021 (OOS): val 2016 в метрику не входит (шаг 1 даёт s² для шага 2).',
           'Кандидаты второго круга не пересчитывались: их designs_val (final2/gibrid) переранжированы по правилу третьего круга; у D12_h22 лучший дизайн мог смениться.',
           'HAR-X без бустинга (Д16.4) в пул не входит: решение владельца 01:12 (HAR-X — только нога гибрида) отменяет «Д16.4 — кандидат» из блока 00:55.',
           'HAR-J с поправкой исключён из финалистов по правилу дирижёра (не прошёл подтверждающий в первом круге на v2), но остаётся фолбэком h=22.']
    if B is not None and (~B.step10.astype(bool)).any():
        dev.append('Шаг 10 не собран для: ' + ', '.join(f'h={r.h} {r.cand}' for r in B[~B.step10.astype(bool)].itertuples()) + ' (нет файлов шага 10 или дизайн не V0/L0).')
    if (FN.fin_note.fillna('').str.contains('добор')).any():
        dev.append('Меньше трёх семейств на h — финалисты добраны следующими допущенными (см. таблицу 2).')
    dev += ['Из второго круга сохраняются: топ-5 вместо топ-10 в обёртке; чистый V0 на h=5/22 на точках v2b; точка бустинга выбирается на годе оценки (одна на все тикеры).']
    L += ['', '## 9. Отступления (↺)', ''] + [f'{i}. {x}' for i, x in enumerate(dev, 1)]
    # 10. раскладка (приложение) и h=22
    L += ['', '## 10. Эталон и раскладка (решения владельца 01:05 → 01:12)', '',
          '- Эталон DM, MCS и H1 на всех h — **HAR-J с поправкой** (как в диссертации; столбец `har_corr`, s² та же, что у финального дизайна). HAR-X — только нога '
          'гибрида (V9 / Д8.12 / Д17.1 на основе HAR-X), в V0-кандидаты и основную таблицу сравнения не входит, в MCS не включается. Столбец `harx_corr` '
          'тестовых файлов (HAR-X с поправкой той же s²) — только для таблицы раскладки приложения (`table_3_raskladka_test.csv`).']
    yrs = list(range(2017, 2022))
    L += ['', '### Раскладка HAR-J → HAR-J с поправкой → HAR-X с поправкой → финал (разработочный набор, OOS QLIKE)', '',
          '| h | строка | ' + ' | '.join(map(str, yrs)) + ' | 2017–2020 | 2017–2021 | к предыдущей строке (2017–2021) |', '|' + '---|' * (len(yrs) + 5)]
    for r in FC.itertuples():
        Sf = pd.read_csv(O / f'designs_rank_{r.cand}_h{r.h}.csv'); sf = Sf[Sf.design == r.design].iloc[0]
        Sj = pd.read_csv(O / f'designs_rank_HARJ_corr_h{r.h}.csv'); sj = Sj.iloc[0]
        code = r.design.split('|')[2].replace('B-', '').replace('bez', 'ord')
        hj = Sj[Sj.design == f'single|V0_HAR|B-{code}|L0']
        sj = hj.iloc[0] if len(hj) else sj
        hx = harx_dev(r.h, code)
        lines = [('HAR-J как в диссертации', [sf[f'raw_{y}'] for y in yrs]), (f'HAR-J с поправкой ({sj.design.split("|")[2]})', [sj[f'q_{y}'] for y in yrs])]
        if hx is not None:
            lines.append((f'HAR-X с поправкой (B-{code}; приложение)', [hx[y] for y in yrs]))
        lines.append((f'финал: {r.cand} {r.design.replace("|", "/")}', [sf[f'q_{y}'] for y in yrs]))
        prev = None
        for nm, v in lines:
            m = float(np.mean(v))
            L.append(f'| {r.h} | {nm} | ' + ' | '.join(F(x) for x in v) + f' | {F(float(np.mean(v[:4])))} | {F(m)} | {P(m / prev - 1) if prev else "—"} |')
            prev = m
    if (R / 'HARX_h22_val.parquet').exists() and (O / 'designs_rank_HARJ_corr_h22.csv').exists():
        Sj = pd.read_csv(O / 'designs_rank_HARJ_corr_h22.csv')
        L += ['', '### h=22: эконометрическая нога гибрида — HAR-J, не HAR-X (решение владельца 01:05, оставлено 01:12)', '',
              'Причина — проигрыш HAR-X в 2020 году. OOS QLIKE с поправкой по годам (разработочный набор):', '',
              '| s² | модель | ' + ' | '.join(map(str, yrs)) + ' | 2017–2021 |', '|' + '---|' * (len(yrs) + 3)]
        for code in ('sm', 'smrob'):
            g = Sj[Sj.design == f'single|V0_HAR|B-{code}|L0']
            hx = harx_dev(22, code)
            for nm, v in (('HAR-J с поправкой', [g[f'q_{y}'].iloc[0] for y in yrs] if len(g) else None), ('HAR-X с поправкой', [hx[y] for y in yrs])):
                if v is not None:
                    L.append(f'| {code} | {nm} | ' + ' | '.join(F(x) for x in v) + f' | {F(float(np.mean(v)))} |')
    # Д16.6
    p = d166()
    L += ['', '### Д16.6 (таблица агента B: HAR-J → HAR-J с поправкой → HAR-X → гибрид с DM на шаге)', '']
    if p is not None:
        L += [f'Источник — `{p.relative_to(R)}`.', ''] + md_table(pd.read_csv(p))
    else:
        L += ['Агент B таблицу Д16.6 к сборке не выложил (искались `data/regen/D16*/**/*16_6*.csv`, `*raskladka*.csv`, `final3/*16_6*.csv`).']
    L += ['', '## 11. Что сделает тест после «да»',
          '`python3 scripts/analysis/final_test.py --scope test --src data/regen/test_locked/final3`: QLIKE/RMSE/MAE по h и годам 2017–2026; DM тремя способами '
          'против HAR-J как в диссертации и HAR-J с поправкой (эталон H1 — HAR-J с поправкой, основной DM); MCS (без HAR-X); тесты охвата; Минцер–Зарновиц; доля тикеров; '
          'подпериоды; терцили; приложение — раскладка HAR-J → HAR-J с поправкой → HAR-X с поправкой → финал (`table_3_raskladka_test.csv`). Тест — один раз.']
    dst = (O / 'FINAL_DESIGN_chernovik_dry.md') if a.dry else (REPO / 'docs' / 'otchety' / 'FINAL_DESIGN_chernovik.md')
    dst.write_text('\n'.join(L) + '\n')
    print(f'черновик FINAL -> {dst}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['pool', 'gibrid', 'select', 'confirm', 'build', 'report'])
    ap.add_argument('--dry', action='store_true'); ap.add_argument('--only-krug2', action='store_true')
    ap.add_argument('--zapis', action='store_true', help='select: записать финалистов в docs/otchety/dev_log.md (только без --dry)')
    ap.add_argument('--confirm', action='store_true', help='confirm: открыть подтверждающий (один раз)')
    ap.add_argument('--force', action='store_true', help='gibrid: пересчитать и при наличии designs_val')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    C = pool(a)
    if a.phase == 'pool':
        L = pool_md(C)
        (out_dir(a) / 'pool.md').write_text('\n'.join(L) + '\n')
        print('\n'.join(L))
    elif a.phase == 'gibrid':
        gibrid(a, C)
    elif a.phase == 'select':
        select(a, C)
    elif a.phase == 'confirm':
        confirm(a, C)
    elif a.phase == 'build':
        build(a, C)
    else:
        select_needed = not (out_dir(a) / 'ranking_h1.csv').exists()
        if select_needed:
            select(a, C)
        for c in C:  # dopusk мог измениться в select (нет designs_val) — пересчёт тем же кодом
            pass
        report(a, C)


if __name__ == '__main__':
    main()
