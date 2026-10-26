"""Д24b: общие функции эконометрических пунктов Д24.4/24.5/24.7/24.9/24.11 (свой модуль; общие файлы проекта не меняются).
Схема шага s (как gibrid_har.har_forecasts): val-год vy = 2015+s, тест-год ty = 2016+s, модель шага обучается на годах < vy и прогнозирует
val vy (вне выборки) и тест ty (только сборка, метрики теста не считаются). ЭМБАРГО (как Д18): строка t обучения берётся, только если
дата конца цели t+h раньше 1 янв. vy. Поправка Б0.1: F = exp(s²/2), s² (sm / smrob) по val-остаткам ln, на val шага t - оценка шага t-1.
Оценка: OOS QLIKE val 2017-2021 (шаги 2-6), 2017-2020 - ранжирование, 2021 - отложенный; подтверждающий (шаги 7-9) - прогнозы строятся,
метрики НЕ считаются. Гибрид - gibrid_v17.py (--dev-eval oos, шаги 7-9 скрыты), представитель - правило третьего круга (final3.summarize)."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gibrid_har as GH
from gibrid_io import load_boost

ROOT = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading')
R = ROOT / 'data' / 'regen'
OUT = R / 'D24b'
TL = R / 'test_locked'
FEAT = R / 'features_pit_v3.parquet'
REPO = HERE.parent.parent
K = ['date', 'ticker', 'step']
COLS = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
STEPS = [(s, 2016 + s, 2015 + s) for s in range(1, 10)]  # (шаг, тест-год, val-год), шаги 1-9
# текущий лучший кругов 1-2 (final3_dry/incumbent.csv): (кандидат, дизайн, 2021, 2017-2020)
INC = {1: ('D12_h1_B_b0.25', 'top5|V0_XGB|B-sm|L0', 0.2395531296561522, 0.28917618310624094),
       5: ('F2m_h5', 'wrap2|V0_LGB|B-sm|L0', 0.155332001485612, 0.2122287402429689),
       22: ('D12_h22', 'seeds3|V0_XGB|B-sm|L0', 0.140402478566046, 0.2222331795994113)}


def qlike(y, p):
    p = np.clip(p, 1e-12, None)
    r = y / p
    return float(np.mean(r - np.log(r) - 1))


# ---------------------------------------------------------------- данные
def load():
    """Матрица features_pit_v3 (как гибрид), сортировка (ticker, date); _f{j} = RV дня t+j, _fd{j} - его дата (j = 1..22)."""
    A, harj = GH.load_pit(FEAT)
    A = A.sort_values(['ticker', 'date']).reset_index(drop=True)
    g = A.groupby('ticker', sort=False)
    new = {}
    for j in range(1, 23):
        new[f'_f{j}'] = g['rv'].shift(-j)
        new[f'_fd{j}'] = g['date'].shift(-j)
    A = pd.concat([A, pd.DataFrame(new, index=A.index)], axis=1)
    assert np.allclose(A['_f1'].fillna(-1), A['rv_target_h1'].fillna(-1))
    return A, harj


def target(A, h):
    return A['rv_target_h1'] if h == 1 else A[f'rv_target_mean_h{h}']


def target_end(A, h):
    return A[f'_fd{h}']


def train_mask(A, h, vy, y=None):
    """Обучение шага: годы < vy, цель есть, эмбарго - дата конца цели < 1 янв. vy."""
    y = target(A, h) if y is None else y
    return ((A.year < vy) & y.notna() & (target_end(A, h) < pd.Timestamp(f'{vy}-01-01'))).values


def harj_X(A, harj):
    return GH.clean(np.log(A[harj].values.clip(1e-10)))


def frame(A, sel, s, yr, y, p):
    return pd.DataFrame({'date': A.date.values[sel], 'ticker': A.ticker.values[sel], 'year': yr, 'step': s,
                         'rv_actual': y.values[sel], 'rv_pred': np.clip(p, 1e-10, None)})


def ols(X, y):
    Xc = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return b


def ols_pred(b, X):
    return b[0] + X @ b[1:]


def run_steps(A, h, fit_pred, y=None):
    """Общий контур: fit_pred(s, vy, tr_mask) -> функция pred(mask) -> ln-прогноз на строках mask (и dict коэффициентов).
    Строки val/тест - год шага с непустой целью (как gibrid_har.har_forecasts). Возвращает (V, T, info)."""
    y = target(A, h) if y is None else y
    V, T, info = [], [], []
    for s, ty, vy in STEPS:
        tr = train_mask(A, h, vy, y)
        pred, co = fit_pred(s, vy, tr)
        info.append(dict(step=s, val_year=vy, n_train=int(tr.sum()), **co))
        for yr, lst in ((vy, V), (ty, T)):
            sel = ((A.year == yr) & y.notna()).values
            lst.append(frame(A, sel, s, yr, y, np.exp(pred(sel))))
    return pd.concat(V, ignore_index=True), pd.concat(T, ignore_index=True), pd.DataFrame(info)


def har_ref(A, harj, h):
    """HAR-J (канон гибрида, без эмбарго) - эталон; gibrid_v17 пересчитывает его сам."""
    hv, ht, _ = GH.har_forecasts(h, A.sort_values('date').reset_index(drop=True), harj)
    return hv, ht


# ---------------------------------------------------------------- поправка и метрики
def s2_tab(df):
    E = {s: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for s, g in df[df.rv_actual > 1e-12].groupby('step')}
    res = {}
    for code in ('sm', 'smrob'):
        for s in E:
            e = np.concatenate([E[j] for j in range(max(1, s - 2), s + 1) if j in E])
            res[(code, s)] = float((1.4826 * np.median(np.abs(e - np.median(e)))) ** 2) if code == 'smrob' else float(np.var(e))
    return res


def correct(val, test, code):
    """OOS: val шага t * exp(s2_{t-1}/2) (шаг 1: F = 1); тест шага t * exp(s2_t/2). code = 'bez' - без поправки."""
    v = val.copy()
    t = test.copy() if test is not None else None
    if code == 'bez':
        return v, t
    S = s2_tab(val[val.step <= 9])
    v['rv_pred'] = v.rv_pred * v.step.map(lambda s: np.exp(S[(code, s - 1)] / 2) if s > 1 else 1.0)
    if t is not None:
        t['rv_pred'] = t.rv_pred * t.step.map(lambda s: np.exp(S[(code, s)] / 2))
    return v, t


def by_year(v):
    """OOS QLIKE по годам 2017-2021 (шаги 2-6), среднее 2017-2020; шаги 7-9 не считаются (подтверждающий закрыт)."""
    d = v[(v.step >= 2) & (v.step <= 6) & (v.rv_actual > 1e-12)]
    q = {str(int(s) + 2015): qlike(g.rv_actual.values, g.rv_pred.values) for s, g in d.groupby('step')}
    q['2017-2020'] = float(np.mean([q[str(y)] for y in range(2017, 2021)]))
    return q


def standalone_rows(name, h, v, t=None, codes=('sm', 'smrob')):
    return [dict(name=name, h=h, s2=c, **by_year(correct(v, t, c)[0])) for c in codes]


# ---------------------------------------------------------------- бустинг-ноги финалистов (сырые, до поправки)
def leg(h, seed='main', test=True):
    """Нога бустинга для второго слота гибрида: h=1 - D12_h1_A_d812 XGB (main = top5, как финалист; s42/s43/s44 - одиночные точки);
    h=5 - D12_h5 XGB (main = wrap2; s42/43/44 - одиночные точки); h=22 - Д16.5 single XGB (main = s42; s43/s44 - seeds4344).
    Тест main - xgb_raw файлов сборки final2 (h=1/5) и test_locked/D16_5_h22 (h=22): только сборка, без метрик."""
    boost = {1: 'D12_h1_A_d812', 5: 'D12_h5', 22: 'D16_5_h22'}[h]
    main_var = {1: 'top5', 5: 'wrap2', 22: 'single'}[h]
    if seed == 'main':
        v = load_boost(boost, h, R, main_var)[0]['val'][K + ['rv_pred']]
    elif seed == 's42':
        v = pd.read_parquet(R / f'{boost}_xgb_h{h}_val.parquet')[K + ['rv_pred']]
    else:
        v = pd.read_parquet(R / f'{boost}_xgb_h{h}_seeds4344_val.parquet')
        v = v[v.seed == int(seed[1:])][K + ['rv_pred']]
    t = None
    if test and seed == 'main':
        if h in (1, 5):
            t = pd.read_parquet(TL / 'final2' / f'final_h{h}_test.parquet', columns=K + ['xgb_raw', 'design'])
            assert t.design.iloc[0].startswith(main_var + '|V0_XGB')
            t = t[t.step <= 9].rename(columns={'xgb_raw': 'rv_pred'})[K + ['rv_pred']]
        else:
            t = load_boost(boost, h, R, 'single', test=True)[0]['test']
            t = t[t.step <= 9][K + ['rv_pred']]
    return v.reset_index(drop=True), (t.reset_index(drop=True) if t is not None else None)


def on_frame(base, other):
    """Прогноз other на строках base (date, ticker, step); строки без прогноза - NaN."""
    x = base[K].merge(other[K + ['rv_pred']], on=K, how='left', validate='one_to_one')
    return base[COLS[:-1]].assign(rv_pred=x.rv_pred.values)


# ---------------------------------------------------------------- кандидаты гибрида
CAND = OUT / 'cand'


def save_cand(boost, h, vx, vl, tx=None, tl=None, test_dir=None):
    """Файлы кандидата в формате gibrid_v17/final3: val - data/regen/D24b/cand/{boost}_{xgb|lgbm}_h{h}_val.parquet (шаги 1-9),
    тест - test_locked/<test_dir>/{boost}_{xgb|lgbm}_h{h}_test.parquet (шаги 1-9, без метрик)."""
    CAND.mkdir(parents=True, exist_ok=True)
    for m, v in (('xgb', vx), ('lgbm', vl)):
        d = v.dropna(subset=['rv_pred', 'rv_actual'])[COLS].reset_index(drop=True)
        d.to_parquet(CAND / f'{boost}_{m}_h{h}_val.parquet', index=False)
    if tx is not None:
        td = TL / (test_dir or boost)
        td.mkdir(parents=True, exist_ok=True)
        for m, t in (('xgb', tx), ('lgbm', tl)):
            d = t.dropna(subset=['rv_pred'])[COLS].reset_index(drop=True)
            d.to_parquet(td / f'{boost}_{m}_h{h}_test.parquet', index=False)


def gibrid_cmd(boost, h, regen=CAND):
    out = OUT / 'gibrid' / boost
    return ([sys.executable, '-W', 'ignore', str(HERE / 'gibrid_v17.py'), '--boost', boost, '--h', str(h), '--features', str(FEAT),
             '--regen', str(regen), '--legs', 'single', '--out', str(out), '--devlog', str(out / 'devlog.md')], out)


def run_gibrid(jobs, workers=2):
    """jobs: [(boost, h)] -> gibrid_v17 (OOS, шаги 7-9 скрыты), лог - <out>/run.log."""
    def one(j):
        cmd, out = gibrid_cmd(*j)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / 'run.log', 'w') as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO), env=dict(os.environ, PYTHONWARNINGS='ignore')).returncode
        print(('OK ' if rc == 0 else f'ОШИБКА {rc} ') + j[0], flush=True)
        return rc
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, jobs))


def summary(boost, h):
    """Все дизайны кандидата по правилу третьего круга (final3.designs/summarize; V0_HAR исключён)."""
    import final3 as F3
    D = F3.designs(dict(gib_dir=OUT / 'gibrid' / boost, gboost=boost, h=h))
    return None if D is None else F3.summarize(dict(), D)


NONM = r'\|V0_(LGB|HCJ)\|'  # дизайны без новой ноги (XGB-слот): «кандидат ничего не добавляет»


def evaluate(boost, h, seeds=('s42', 's43', 's44'), uses_slot=r'\|V0_(LGB|HCJ)\|'):
    """Представитель (лучший по правилу) основного варианта; тот же дизайн на seed-вариантах ({boost}_{seed});
    проверки: (1) представитель использует новую ногу (не V0_LGB/V0_HCJ), (2) 2017-2020 >= 1% лучше HAR-J с поправкой той же s²,
    (3) лучше HAR-J на каждом seed, (4) 2021 <= текущего лучшего кругов 1-2."""
    S = summary(boost, h)
    if S is None:
        return None
    x = S.iloc[0]
    row = dict(boost=boost, h=h, design=x.design, n_failed=int(x.n_failed), failed=x.failed or '—', mean_1720=x.mean_1720, q2021=x.q2021,
               har_1720=x.har_1720, har2021=x.har2021, **{f'q_{y}': x[f'q_{y}'] for y in range(2017, 2022)})
    v0l = S[S.design.str.contains(r'\|V0_LGB\|')].iloc[0]
    row.update(v0lgb_design=v0l.design, v0lgb_1720=v0l.mean_1720, v0lgb_2021=v0l.q2021, v0lgb_nf=int(v0l.n_failed))
    row['gain_vs_har'] = x.mean_1720 / x.har_1720 - 1
    sd = []
    for s in seeds:
        Ss = summary(f'{boost}_{s}', h)
        if Ss is None:
            sd.append((s, np.nan, np.nan, np.nan)); continue
        g = Ss[Ss.design == x.design]
        sd.append((s, float(g.mean_1720.iloc[0]), float(g.har_1720.iloc[0]), float(g.q2021.iloc[0])) if len(g) else (s, np.nan, np.nan, np.nan))
    for s, q, hq, q21 in sd:
        row[f'{s}_1720'], row[f'{s}_har'], row[f'{s}_2021'] = q, hq, q21
    seedq = [q for _, q, _, _ in sd if np.isfinite(q)]
    row['seed_mean_1720'] = float(np.mean(seedq)) if seedq else np.nan
    row['seed_mean_har'] = float(np.mean([hq for _, _, hq, _ in sd if np.isfinite(hq)])) if seedq else np.nan
    row['p_novaya_noga'] = not bool(pd.Series([x.design]).str.contains(uses_slot).iloc[0])
    row['p_1pct'] = bool(x.mean_1720 <= 0.99 * x.har_1720) and (not seedq or row['seed_mean_1720'] <= 0.99 * row['seed_mean_har'])
    row['p_seeds'] = bool(len(seedq) == len(seeds) and all(q < hq for _, q, hq, _ in sd)) if seeds else True
    row['p_2021'] = bool(x.q2021 <= INC[h][2])
    row['inc_2021'], row['inc_1720'] = INC[h][2], INC[h][3]
    row['prohodit'] = row['p_novaya_noga'] and row['p_1pct'] and row['p_seeds'] and row['p_2021']
    return row


def write_yaml(boost, h, fam, rychag, opisanie, n_var, test_dir=None, extra=''):
    d = R / 'final3' / 'kandidaty'
    txt = (f'boost: {boost}\nh: {h}\nsemeystvo: "{fam}"\nrychag: {rychag}\nopisanie: "{opisanie}"\nregen: data/regen/D24b/cand\nlegs: [single]\n'
           f'odinochnaya: false\nchistyy: false\ntest_dir: data/regen/test_locked/{test_dir or boost}\nn_variantov: {n_var}\n'
           f'god2021: "только отложенный, одна проверка (Д24b)"\n{extra}')
    (d / f'{boost}.yaml').write_text(txt)
    print('yaml:', d / f'{boost}.yaml', flush=True)


F = lambda x, n=4: '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{n}f}'.replace('.', ',')


# ---------------------------------------------------------------- детектор «будущее не влияет»
def perturb(A, cut, seed=0, cols=None):
    """Все числовые входы строк с датой >= cut умножаются на случайные множители (0,2..5); RV дня и цели пересчитываются из
    испорченного RV (цели строк до cut, заходящие за cut, тоже меняются). Модель шага с эмбарго обязана дать те же коэффициенты."""
    rng = np.random.default_rng(seed)
    B = A.copy()
    fut = (B.date >= pd.Timestamp(cut)).values
    k = rng.uniform(0.2, 5, fut.sum())
    skip = {'year', 'step'} | {c for c in B.columns if c.startswith('_f')} | {c for c in B.columns if 'target' in c}
    num = cols or [c for c in B.columns if c not in skip and pd.api.types.is_numeric_dtype(B[c])]
    B.loc[fut, num] = B.loc[fut, num].values * k[:, None]
    g = B.groupby('ticker', sort=False)
    for j in range(1, 23):
        B[f'_f{j}'] = g['rv'].shift(-j)
    B['rv_target_h1'] = B['_f1']
    for h in (5, 22):
        B[f'rv_target_mean_h{h}'] = B[[f'_f{j}' for j in range(1, h + 1)]].mean(axis=1, skipna=False)
        B[f'rv_target_h{h}'] = B[f'rv_target_mean_h{h}']
    return B
