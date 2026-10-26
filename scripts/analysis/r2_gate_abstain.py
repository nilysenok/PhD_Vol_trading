#!/usr/bin/env python3
"""R2, агент G: воздержание (кластер 16), усадка к эталону (кластер 18), ограниченная коррекция «знак × δ» (кластер 20).
Протокол v4 docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md (раздел 6, раздел 7 строка G); параметры — models/designs/r2/r2_G_abstain.yaml.

Входы (через r2_core.load_series по хэшу дизайна-кандидата): нога L (ln ŷ_L) и гибриды-компоненты seed × lib (роли B, M, E).
Коррекция компоненты c_j = ln ŷ_j − ln ŷ_L; коррекция ансамбля c̄ = ln mean_j ŷ_j − ln ŷ_L (как r2_ens.uncert_shrink).

Кластер 16 — ŷ = нога · exp(g_t · c), g_t ∈ [0, 1] по данным ≤ t:
  conf_seed    g = 1{SD_j(c_j) < τ·|c̄|}                                  τ — сетка, выбор вложенно
  regime       g = P(спокойно на горизонте h) — привратник кластера 13 (r2_ens._gate на матрице r2_ens.prep; обучение — шаг k с
               эмбарго h, r2_core.train_mask; метка — ln y_h ниже медианы тикера на обучении); по желанию изотонная калибровка
  shock_fresh  g = 0 в первые k торговых дней после дня s со скачком RV рынка (среднее rv_d d0 по всем тикерам) выше q-квантиля
               прошлых W дней (день s — первый); k, q — сетка
  halt         g = 0, если окно цели [d+1, d+h] пересекает остановку торгов 28.02–23.03.2022 — свойство модели, умножается на g
               любого ворот. В отборе (даты < 2022) ничего не меняет; готово для сборки A (halt_g)
  soft         g = 1/(1 + Var_j(c_j)/τ) — r2_ens.uncert_shrink агента E (импорт, не копия; τ — сетка E)
Кластер 18 — ln ŷ = ln HAR + λ·(ln ŷ_канд − ln HAR), HAR — эталон В2 (har_corr), кандидат — гибрид-ансамбль; λ ∈ {0,25 … 1}.
Кластер 20 — ln ŷ = ln база + δ·s_t, s = sign(c̄) (znak) или 2p − 1, p — доля компонент с c_j > 0 (dolya, аналог Д19.3); δ ∈ {0,1; 0,2; 0,3}.

Выбор τ, k, q, λ, δ — вложенно в складке Y по обучающим годам T \\ {Y} (протокол v4 § 2): максимум средней по годам панельной t против
эталона (r2_gates агента O, если в нём есть panel_t; до того — ftp_stats.dm_main: d_t — средняя по тикерам разность QLIKE
эталон − кандидат, NW лаг h − 1); равенство — средний QLIKE. Отложенный год в выбор не входит. Параметры по шагам —
<кэш дизайна>/params.csv (строки modul = G). Прогнозы — только val 2017–2021 (шаги 2–6).

Обвязка: проверка сдвига п. 2.6 (r2_core.shift_check — каждый выходной ряд) и детектор «будущее не влияет»: при фиксированных
параметрах возмущение всех входов после среза (ряды, эталон, матрица d0 — r2_core.perturb_frame) не меняет g_t и ŷ на строках ≤ среза;
подлог «вход d := d+1» пойман; граница года — эмбарго привратника regime (без эмбарго — протекает: положительный контроль).
Вложенный выбор LOYO по построению смотрит на другие годы T — это гиперпараметр внутри складки (протокол v3 § 2.1), не g_t.

python3 scripts/analysis/r2_gate_abstain.py selftest
python3 scripts/analysis/r2_gate_abstain.py dym                        # Мак: 3 тикера × шаги 1–3 × seed 42 × h 1, 5, 22
python3 scripts/analysis/r2_gate_abstain.py run --h 5 [--dizayn <yaml|hash>] [--noga <ряд>] [--korr <ряд> …] [--root <кэш>]"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '1')
import argparse, io, json, subprocess, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import r2_core as C  # noqa: E402  ядро агента S

CFG = yaml.safe_load((REPO / 'models' / 'designs' / 'r2' / 'r2_G_abstain.yaml').read_text())
R = Path(os.environ['R2_DATA']) if os.environ.get('R2_DATA') else C.REGEN
OUT = R / 'R2' / 'G'
MAXY = int(CFG['max_year'])
KEY = ['date', 'ticker', 'step']
T_YEARS = C.DEV_YEARS
MODUL = 'G'


def now():
    return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S'], capture_output=True, text=True).stdout.strip()


def qlike(y, p):
    r = y / np.clip(p, 1e-12, None)
    return float(np.mean(r - np.log(r) - 1))


# ================================================================ панельная t (критерий вложенного выбора)
PANEL_T_SRC = None


def panel_t(df, h, ref='ref'):
    """Панельная t кандидата против эталона: r2_gates.panel_t агента O (если есть), иначе ftp_stats.dm_main (тот же код, что в тесте).
    df: date, rv_actual, ref, rv_pred. Плюс — кандидат лучше."""
    global PANEL_T_SRC
    try:
        import r2_gates as OG  # появится у агента O
        if hasattr(OG, 'panel_t'):
            PANEL_T_SRC = 'r2_gates.panel_t'
            return float(OG.panel_t(df, ref=ref, h=h))
    except ImportError:
        pass
    import ftp_stats as FS
    PANEL_T_SRC = 'ftp_stats.dm_main (заглушка до r2_gates)'
    return float(FS.dm_main(df, ref, h)['t'])


# ================================================================ входы
def _val_only(d, name):
    d = d.copy()
    d['date'] = pd.to_datetime(d['date'])
    d['year'] = d['date'].dt.year.astype(int)
    d['ticker'] = d['ticker'].astype(str)
    assert d['year'].max() <= MAXY, f'{name}: год > {MAXY}'
    assert (d['step'] == d['year'] - 2015).all(), f'{name}: не val-формат'
    return d


def load_etalon(h, tickers=None, steps=None):
    """Эталон В2 (ВЫР-1): rv_actual, har_raw, har_corr; только шаги ≤ 6 (фильтр при чтении) и годы ≤ 2021 (assert)."""
    p = R / CFG['etalon']['path'].format(h=h)
    e = _val_only(pd.read_parquet(p, filters=[('step', '<=', MAXY - 2015)]), str(p))
    if tickers:
        e = e[e.ticker.isin(tickers)]
    if steps:
        e = e[e.step.isin(steps)]
    return e


def design_of(dizayn, root=None):
    """yaml / dict / хэш -> полный дизайн (для write_series). Хэш — design.yaml каталога кэша."""
    if isinstance(dizayn, str) and len(dizayn) >= 16 and all(ch in '0123456789abcdef' for ch in dizayn):
        return yaml.safe_load((Path(root or C.CACHE_ROOT) / dizayn[:16] / 'design.yaml').read_text())
    return C.validate(dizayn)


def load_inputs(dizayn, h, noga, korr, seeds=None, steps=None, tickers=None, root=None, noga_hash=None, modul=None):
    """-> (leg DataFrame KEY+year+rv_pred, [компоненты], meta [(ряд, seed)]) через r2_core.load_series."""
    sha = dizayn if isinstance(dizayn, str) and len(dizayn) >= 16 else C.design_hash(dizayn)
    L = C.load_series(noga_hash or sha, ryad=noga, h=h, steps=steps, root=root)
    assert len(L), f'нет ноги {noga} h={h}'
    L = _val_only(L, noga)
    if L['seed'].nunique() > 1:
        L = L[L['seed'] == L['seed'].iloc[0]]
    B = C.load_series(sha, ryad=list(korr), h=h, seed=seeds, steps=steps, modul=modul, root=root)
    assert len(B), f'нет гибридов {korr} h={h}'
    B = _val_only(B, 'korr')
    if tickers:
        L, B = L[L.ticker.isin(tickers)], B[B.ticker.isin(tickers)]
    comps, meta = [], []
    for (ry, sd), g in B.groupby(['ryad', 'seed'], sort=True):
        comps.append(g[KEY + ['rv_pred']]); meta.append((ry, str(sd)))
    return L[KEY + ['year', 'rv_pred']], comps, meta


def assemble(leg, comps, meta, eta, steps_out=None, how='mean'):
    """Рамка F: общие строки ноги, всех компонент и эталона; ln-массивы."""
    b = leg.rename(columns={'rv_pred': 'leg'})
    for i, d in enumerate(comps):
        b = b.merge(d[KEY + ['rv_pred']].rename(columns={'rv_pred': f'c{i}'}), on=KEY)
    cols = ['rv_actual'] + [c for c in ('har_raw', 'har_corr') if c in eta.columns]
    b = b.merge(eta[KEY + cols], on=KEY)
    if steps_out:
        b = b[b.step.isin(steps_out)]
    b = b.sort_values(KEY).reset_index(drop=True)
    ln = lambda x: np.log(np.clip(np.asarray(x, float), 1e-12, None))
    Lg = ln(b['leg'])
    Pj = np.column_stack([b[f'c{i}'].values for i in range(len(comps))])
    Cj = ln(Pj) - Lg[:, None]
    cbar = ln(np.median(Pj, 1) if how == 'median' else Pj.mean(1)) - Lg
    seeds = sorted({s for _, s in meta})
    cs = {s: ln(Pj[:, [i for i, m in enumerate(meta) if m[1] == s]].mean(1)) - Lg for s in seeds}
    return dict(base=b, leg=Lg, Pj=Pj, Cj=Cj, cbar=cbar, cs=cs, meta=meta, sd=Cj.std(1), var=Cj.var(1),
                ppos=(Cj > 0).mean(1), how=how)


# ================================================================ кластер 16: ворота
def g_conf(F, tau):
    return (F['sd'] < tau * np.abs(F['cbar'])).astype(float)


def g_soft(F, tau):
    """r2_ens.uncert_shrink (агент E): w = 1/(1 + Var_seed/τ), Var — по ln компонент (ddof = 0). Вызов функции E, w — её выход."""
    import r2_ens as E
    b = F['base']
    mk = lambda p: pd.DataFrame({'date': b.date, 'ticker': b.ticker, 'year': b.year, 'step': b.step, 'rv_actual': b.rv_actual,
                                 'rv_pred': p})
    dfs = [mk(F['Pj'][:, i]) for i in range(F['Pj'].shape[1])]
    _, w = E.uncert_shrink(dfs, mk(np.exp(F['leg'])), float(tau), how=F['how'])
    w = np.asarray(w, float)
    assert len(w) == len(b), 'uncert_shrink: другие строки'
    return w


def market_rv(A):
    """RV рынка дня d: среднее rv_d (RV дня d, d0) по всем тикерам строки d. -> Series по дате."""
    return A.groupby(pd.to_datetime(A['date']))['rv_d'].mean().sort_index()


def shock_flags(m, k, q, W, mera='uroven'):
    """День s — скачок, если x_s > q-квантиля x за W дней до s (без s). g_t = 0, если скачок был в один из k последних дней (включая t)."""
    x = np.log(m.clip(lower=1e-12))
    if mera == 'prirost':
        x = x.diff()
    thr = x.shift(1).rolling(W, min_periods=max(20, W // 2)).quantile(q)
    shock = (x > thr).astype(float)
    fresh = shock.rolling(int(k), min_periods=1).max()
    return 1.0 - fresh  # g по дате


def g_shock(F, gser):
    return pd.to_datetime(F['base']['date']).map(gser).fillna(1.0).values.astype(float)


def halt_g(dates, tend, h, start=None, end=None, zapas=None):
    """g = 0 для окна цели (d, tend] ∩ [start, end] ≠ ∅ ⟺ d < end и tend ≥ start. tend неизвестен — d + h рабочих дней + запас."""
    hc = CFG['k16']['halt']
    start = pd.Timestamp(start or hc['start']); end = pd.Timestamp(end or hc['end'])
    zapas = int(hc['zapas_dney'] if zapas is None else zapas)
    d = pd.to_datetime(pd.Series(np.asarray(dates)))
    te = pd.to_datetime(pd.Series(np.asarray(tend) if tend is not None else [pd.NaT] * len(d)))
    fb = d + pd.offsets.BDay(int(h)) + pd.Timedelta(days=zapas)
    te = te.where(te.notna(), fb)
    hit = (d < end) & (te >= start)
    return np.where(hit.values, 0.0, 1.0)


def halt_for(F, A, h):
    b = F['base']
    t = A[['date', 'ticker', f'_tend{h}']].copy()
    t['date'] = pd.to_datetime(t['date'])
    te = b[['date', 'ticker']].merge(t, on=['date', 'ticker'], how='left')[f'_tend{h}'].values
    return halt_g(b['date'].values, te, h)


# --- regime: привратник кластера 13 (r2_ens._gate / prep / _z), P(спокойно)
REG_TERMS = ['ln_rvi2', 'ln_rvm_imoex', 'lhar_1', 'lhar_5', 'lhar_22', 'pc1']


def regime_model_P(M, h, k, gate='logit', seed=42, emb=True):
    """Привратник шага k: обучение — r2_core.train_mask (годы < vy, эмбарго h); -> P(спокойно) на всех строках M."""
    import r2_ens as E
    A = M['A']; vy = 2015 + k; t = f'rv_target_h{h}'
    lny = np.log(A[t].values + 1e-10)
    tr = C.train_mask(A, h, vy, emb=emb) & A[t].notna().values
    G = np.column_stack([M['H'], E._z(M, REG_TERMS, tr)])
    med = pd.Series(lny[tr]).groupby(A.ticker.values[tr]).median()
    lab = (lny - A.ticker.map(med).values >= 0).astype(int)  # 0 — спокойно (ниже медианы тикера)
    clf = E._gate(gate, seed).fit(G[tr], lab[tr])
    return clf.predict_proba(G)[:, 0], lab


def regime_P(M, h, steps, gate='logit', kalibr='none', seed=42):
    """-> DataFrame(date, ticker, step, g): P(спокойно) модели шага k на строках val-года шага; изотонная — по OOS шагов < k (конец цели < vy)."""
    from sklearn.isotonic import IsotonicRegression
    A = M['A']; t = f'rv_target_h{h}'
    out, past = [], []
    for k in sorted(steps):
        vy = 2015 + k
        P, lab = regime_model_P(M, h, k, gate, seed)
        va = (A.year.values == vy)
        Pc = P.copy()
        if kalibr == 'isotonic' and past:
            pp = pd.concat(past, ignore_index=True)
            pp = pp[pp.fd < np.datetime64(f'{vy}-01-01')]
            if len(pp) > 50:
                Pc = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(pp.p.values, (pp.lab.values == 0) * 1.0).predict(P)
        has = va & A[t].notna().values
        past.append(pd.DataFrame({'p': P[has], 'lab': lab[has], 'fd': pd.to_datetime(A[f'_tend{h}']).values[has]}))
        out.append(pd.DataFrame({'date': A.date.values[va], 'ticker': A.ticker.values[va].astype(str), 'step': k, 'g': Pc[va]}))
    return pd.concat(out, ignore_index=True)


def g_regime(F, RP):
    b = F['base'][KEY]
    return b.merge(RP, on=KEY, how='left')['g'].fillna(1.0).values.astype(float)


# ================================================================ варианты (семейства) и вложенный выбор
def families(F, A, M, h, which=('k16', 'k18', 'k20'), steps_all=None, fixed=False):
    """-> {семейство: [(tag, параметры, fn(c) -> ln ŷ, g или None)]}. fixed — по одному значению сетки (детектор)."""
    c16, c18, c20 = CFG['k16'], CFG['k18'], CFG['k20']
    pick = (lambda xs: list(xs)[:1]) if fixed else list
    hg = halt_for(F, A, h)  # halt — свойство модели: умножается на g любого ворот
    leg = F['leg']
    fam = {}
    if 'k16' in which:
        fam['G16_conf_seed'] = [(f'tau={t}', dict(tau=t), None, g_conf(F, t) * hg) for t in pick(c16['conf_seed']['tau'])]
        import r2_ens as E
        fam['G16_soft'] = [(f'tau={t}', dict(tau=t), None, g_soft(F, t) * hg) for t in pick(E.TAUS)]
        mrv = market_rv(A)
        fam['G16_shock_fresh'] = [(f'k={k}__q={q}', dict(k=k, q=q),
                                   None, g_shock(F, shock_flags(mrv, k, q, int(c16['shock_fresh']['okno']), c16['shock_fresh']['mera'])) * hg)
                                  for k in pick(c16['shock_fresh']['k']) for q in pick(c16['shock_fresh']['q'])]
        RP = regime_P(M, h, steps_all or sorted(F['base'].step.unique()), c16['regime']['gate'], c16['regime']['kalibr'])
        fam['G16_regime'] = [('P_spokoyno', dict(gate=c16['regime']['gate'], kalibr=c16['regime']['kalibr']), None, g_regime(F, RP) * hg)]
        fam['G16_halt'] = [('halt', dict(start=c16['halt']['start'], end=c16['halt']['end']), None, hg)]
    if 'k18' in which:
        har = np.log(np.clip(F['base'][c18['har']].values, 1e-12, None))
        fam['G18_k_etalonu'] = [(f'lam={l}', dict(lam=l), (lambda c, l=l: har + l * (leg + c - har)), None) for l in pick(c18['lam'])]
    if 'k20' in which:
        base = leg if c20['baza'] == 'noga' else np.log(np.clip(F['base'][CFG['etalon']['ref']].values, 1e-12, None))
        for vid in c20['vid']:
            sfun = (lambda c: np.sign(c)) if vid == 'znak' else (lambda c: 2 * F['ppos'] - 1)
            fam[f'G20_{vid}'] = [(f'delta={dl}', dict(vid=vid, delta=dl), (lambda c, dl=dl, sf=sfun: base + dl * sf(c)), None)
                                 for dl in pick(c20['delta'])]
    return fam


def ln_pred(var, F, c):
    tag, par, fn, g = var
    return F['leg'] + g * c if fn is None else fn(c)


def year_stats(F, lnp, h, ref):
    """{год: (t, QLIKE)} по годам T, присутствующим в рамке."""
    b = F['base']; out = {}
    for y in sorted(set(b.year.unique()) & set(T_YEARS)):
        m = (b.year.values == y) & (b.rv_actual.values > 1e-12)
        df = pd.DataFrame({'date': b.date.values[m], 'rv_actual': b.rv_actual.values[m], 'ref': b[ref].values[m],
                           'rv_pred': np.exp(lnp[m])})
        out[int(y)] = (panel_t(df, h), qlike(df.rv_actual.values, df.rv_pred.values))
    return out


def nested(F, variants, h, ref, rezhim='loyo'):
    """Вложенный выбор: для отложенного года Y — вариант с max средней t на обучающих годах (loyo: T \\ {Y}; proshlye: годы < Y),
    равенство — меньший средний QLIKE. -> (индекс варианта по строкам, таблица параметров)."""
    eps = float(CFG['vybor']['eps_t'])
    st = [year_stats(F, ln_pred(v, F, F['cbar']), h, ref) for v in variants]
    yrs = sorted(st[0])
    b = F['base']; choice = np.full(len(b), -1); rows = []
    for Y in yrs:
        tr = [y for y in yrs if y != Y] if rezhim == 'loyo' else [y for y in yrs if y < Y]
        if not tr:
            j, tm, qm = 0, np.nan, np.nan
        else:
            sc = [(np.mean([s[y][0] for y in tr]), np.mean([s[y][1] for y in tr])) for s in st]
            tmax = max(t for t, _ in sc)
            cand = [i for i, (t, _) in enumerate(sc) if t >= tmax - eps]
            j = min(cand, key=lambda i: sc[i][1]); tm, qm = sc[j]
        choice[b.year.values == Y] = j
        g = variants[j][3]
        mY = b.year.values == Y
        rows.append(dict(god=Y, step=Y - 2015, skladka=Y, rezhim=rezhim, obuch_gody=';'.join(map(str, tr)), variant=variants[j][0],
                         parametry=json.dumps(variants[j][1], ensure_ascii=False), t_obuch=tm, q_obuch=qm,
                         t_otlozh=st[j][Y][0], q_otlozh=st[j][Y][1],
                         g_sred=float(np.mean(g[mY])) if g is not None else np.nan,
                         dolya_g0=float(np.mean(g[mY] == 0)) if g is not None else np.nan,
                         **{f't_{v[0]}': st[i][Y][0] for i, v in enumerate(variants)}))
    return choice, pd.DataFrame(rows)


def series_out(F, variants, choice, c):
    lnp = np.full(len(F['base']), np.nan)
    for j in np.unique(choice):
        m = choice == j
        lnp[m] = ln_pred(variants[j], F, c)[m]
    b = F['base']
    return pd.DataFrame({'date': b.date.values, 'ticker': b.ticker.values, 'year': b.year.values, 'step': b.step.values,
                         'rv_actual': b.rv_actual.values, 'rv_pred': np.exp(lnp)})


def write_params(cdir, P):
    """<кэш>/params.csv: свои строки (modul = G, те же ряды) заменяются, чужие сохраняются; запись атомарная."""
    f = Path(cdir) / 'params.csv'
    P = P.assign(modul=MODUL)
    if f.exists():
        old = pd.read_csv(f)
        if 'modul' in old and 'ryad' in old:
            old = old[~((old.modul == MODUL) & old.ryad.isin(P.ryad.unique()) & old.h.isin(P.h.unique()))]
        P = pd.concat([old, P], ignore_index=True)
    buf = io.StringIO(); P.to_csv(buf, index=False)
    C._atomic_write(f, buf.getvalue().encode('utf-8'))
    return f


def run_h(design, h, F, A, M, steps_all, root=None, block_file=None, which=('k16', 'k18', 'k20'), write=True):
    """Все семейства на h: вложенный выбор, ряды (ансамбль и по seed), сдвиг, params.csv. -> (сводка, params)."""
    ref = CFG['etalon']['ref']; rezhim = CFG['vybor']['rezhim']
    fam = families(F, A, M, h, which, steps_all)
    full = design_of(design, root)
    seeds = sorted(F['cs'])
    S, Pall = [], []
    for name, variants in fam.items():
        choice, P = nested(F, variants, h, ref, rezhim)
        P = P.assign(ryad=name, h=h, panel_t=PANEL_T_SRC)
        V = series_out(F, variants, choice, F['cbar'])
        sh = C.shift_check(V, A, h, name, design=full, block_file=block_file)
        outs = {}
        if len(seeds) == 1:
            outs[int(seeds[0]) if seeds[0].isdigit() else None] = V
        else:
            outs[None] = V
            for s in seeds:
                outs[int(s)] = series_out(F, variants, choice, F['cs'][s])
        if write:
            for sd, df in outs.items():
                C.write_series(full, name, df, h, sd, module=MODUL, root=root,
                               note=f'G: {name}; выбор {rezhim} по {PANEL_T_SRC}')
        q = {int(y): qlike(g.rv_actual.values, g.rv_pred.values) for y, g in V[V.rv_actual > 1e-12].groupby('year')}
        S.append(dict(h=h, ryad=name, variantov=len(variants), argmax=sh.get('argmax'), sdvig_ok=sh.get('ok'),
                      **{f'q{y}': v for y, v in q.items()}, t_otlozh_sred=float(P.t_otlozh.mean()), vybrano=';'.join(P.variant)))
        Pall.append(P)
    P = pd.concat(Pall, ignore_index=True)
    if write:
        write_params(C.cache_dir(full, root=root), P)
    return pd.DataFrame(S), P


# ================================================================ детектор «будущее не влияет»
def _perturb_series(df, cut, rng):
    d = df.copy(); f = pd.to_datetime(d.date).values > np.datetime64(cut)
    d.loc[f, 'rv_pred'] = d.loc[f, 'rv_pred'].values * rng.uniform(0.5, 2.0, int(f.sum()))
    return d


def _perturb_eta(e, A, h, cut, rng):
    e = e.copy(); cutd = np.datetime64(cut)
    t = A[['date', 'ticker', f'_tend{h}']].copy(); t['date'] = pd.to_datetime(t.date)
    te = e[['date', 'ticker']].merge(t, on=['date', 'ticker'], how='left')[f'_tend{h}'].values
    fut = pd.to_datetime(e.date).values > cutd
    cross = fut | ~(pd.to_datetime(te) <= pd.Timestamp(cut))  # цель: окно [d+1, d+h] заходит за срез
    e.loc[cross, 'rv_actual'] = e.loc[cross, 'rv_actual'].values * rng.uniform(0.5, 2.0, int(cross.sum()))
    for c in [c for c in ('har_raw', 'har_corr') if c in e]:  # прогноз эталона дня d известен в d
        e.loc[fut, c] = e.loc[fut, c].values * rng.uniform(0.5, 2.0, int(fut.sum()))
    return e


def _plant_series(df):
    """Подлог: значение строки d := значение строки d+1 того же тикера и шага (по дате)."""
    d = df.sort_values(['ticker', 'step', 'date']).copy()
    d['rv_pred'] = d.groupby(['ticker', 'step'])['rv_pred'].shift(-1).fillna(d['rv_pred'])
    return d.sort_index()


def leak(h, leg, comps, meta, eta, A, harj, steps_all, steps_out, cuts=None, seed=20261026):
    """Δ ln ŷ (и Δ g) на строках ≤ среза у шагов k ≤ год среза − 2015 при фиксированных параметрах; подлог d+1 пойман."""
    import r2_ens as E
    cuts = cuts or CFG['leak']['cuts_mid']

    def build(Ax, legx, compsx, etax):
        F = assemble(legx, compsx, meta, etax, steps_out, CFG['vhod']['agg'])
        M = E.prep(Ax, harj)
        fam = families(F, Ax, M, h, steps_all=steps_all, fixed=True)
        o = {}
        for nm, vs in fam.items():
            b = F['base'][KEY].copy(); b['rv_pred'] = np.exp(ln_pred(vs[0], F, F['cbar'])); o[nm] = b
            if vs[0][3] is not None:
                bg = F['base'][KEY].copy(); bg['rv_pred'] = np.exp(vs[0][3]); o[nm + '#g'] = bg  # g ≥ 0: exp(g) > 0
        return o

    def pl(Ax):
        return C.plant_frame(Ax), _plant_series(leg), [_plant_series(c) for c in comps], eta

    base = build(A, leg, comps, eta)
    P0 = build(*pl(A))
    rows = []
    for i, cut in enumerate(cuts):
        cut = pd.Timestamp(cut); rng = np.random.default_rng(seed + i); kmax = cut.year - 2015
        Ap = C.perturb_frame(A, cut, seed + i)
        legp, compsp = _perturb_series(leg, cut, rng), [_perturb_series(c, cut, rng) for c in comps]
        etap = _perturb_eta(eta, A, h, cut, rng)
        pert = build(Ap, legp, compsp, etap)
        P1 = build(C.plant_frame(Ap), _plant_series(legp), [_plant_series(c) for c in compsp], etap)
        for r in base:
            past, fut, n = C._diff(base[r], pert[r], cut, kmax)
            pp, _, _ = C._diff(P0[r], P1[r], cut, kmax)
            # ряд g: главное — Δ = 0 до среза (g_t только из прошлого); g бинарный или ≡ 1 (halt в val) может не сдвинуться
            # после среза и на одной строке подлога — подлог ловится на ряде ŷ того же семейства
            gsr = r.endswith('#g')
            rows.append(dict(h=h, ryad=r, srez=str(cut.date()), kmax=kmax, strok_do=n, max_dln_do=past, max_dln_posle=fut,
                             podlog_max_dln_do=pp, ok=bool(past == 0.0 and n > 0 and (gsr or (fut > 0 and pp > 0)))))
    # граница года: эмбарго привратника regime (P модели шага vy на всех строках ≤ 31.12 предыдущего года)
    for ye in CFG['leak']['cuts_year_end']:
        cut = pd.Timestamp(ye); k = cut.year + 1 - 2015
        Ap = C.perturb_frame(A, cut, seed + 99)
        for emb in (True, False):
            p0, _ = regime_model_P(E.prep(A, harj), h, k, emb=emb)
            p1, _ = regime_model_P(E.prep(Ap, harj), h, k, emb=emb)
            m = (pd.to_datetime(A.date) <= cut).values
            d = float(np.abs(p0[m] - p1[m]).max())
            rows.append(dict(h=h, ryad=f'G16_regime#emb={emb}', srez=str(cut.date()), kmax=k, strok_do=int(m.sum()), max_dln_do=d,
                             max_dln_posle=np.nan, podlog_max_dln_do=np.nan,
                             ok=bool(d == 0.0) if emb else bool(d > 0)))  # без эмбарго должно протекать (положительный контроль)
    return pd.DataFrame(rows)


# ================================================================ самопроверки
def selftest():
    rows = []
    ok = lambda nm, v: rows.append(dict(test=nm, ok=bool(v)))
    # halt: синтетические даты (данные ≥ 2022 не читаются)
    d = pd.to_datetime(['2022-02-24', '2022-02-25', '2022-03-23', '2022-03-24', '2021-12-30', '2022-01-28'])
    te = pd.to_datetime(['2022-02-25', '2022-03-24', '2022-03-24', '2022-03-25', '2022-02-01', pd.NaT])
    g = halt_g(d, te, 5)
    ok('halt: окно до остановки — g = 1', g[0] == 1)
    ok('halt: окно через остановку — g = 0', g[1] == 0)
    ok('halt: d = последний день остановки — окно после, g = 1', g[2] == 1)
    ok('halt: после остановки — g = 1', g[3] == 1)
    ok('halt: val 2021 — g = 1', g[4] == 1)
    ok('halt: нет tend — d + 5 рабочих + запас (28.01 → 07.02) — g = 1', g[5] == 1)
    ok('halt: нет tend, h = 22 от 28.01 — g = 0', halt_g(d[5:], [pd.NaT], 22)[0] == 0)
    # shock: k дней, день скачка первый
    idx = pd.bdate_range('2019-01-01', periods=400)
    m = pd.Series(1.0, index=idx); m.iloc[300] = 10.0
    gs = shock_flags(m * np.exp(np.random.default_rng(0).normal(0, 0.01, 400)), 5, 0.99, 250)
    ok('shock: g = 0 ровно k = 5 дней с дня скачка', (gs.iloc[300:305] == 0).all() and gs.iloc[305] == 1 and gs.iloc[299] == 1)
    gs2 = shock_flags(m.shift(-1).fillna(1.0), 5, 0.99, 250)
    ok('shock: только прошлое (скачок завтра не виден сегодня)', gs2.iloc[298] == 1 and gs2.iloc[299] == 0)
    # conf_seed / soft на синтетике
    rng = np.random.default_rng(1)
    n = 200; b = pd.DataFrame({'date': pd.bdate_range('2017-01-02', periods=n), 'ticker': 'X', 'step': 2, 'year': 2017})
    leg = b.assign(rv_pred=np.exp(rng.normal(0, 0.3, n)))
    comps = [b[KEY].assign(rv_pred=leg.rv_pred * np.exp(0.1 + rng.normal(0, s, n))) for s in (0.01, 0.02, 0.2)]
    eta = b.assign(rv_actual=leg.rv_pred * np.exp(rng.normal(0, 0.5, n)), har_raw=leg.rv_pred, har_corr=leg.rv_pred * 1.05)
    F = assemble(leg, comps, [('a', '42'), ('b', '42'), ('a', '43')], eta)
    ok('conf_seed: τ → ∞ — g = 1', (g_conf(F, 1e9) == 1).all())
    ok('conf_seed: τ = 0 — g = 0', (g_conf(F, 0.0) == 0).all())
    w = g_soft(F, 0.01)
    ok('soft = r2_ens.uncert_shrink: 1/(1+Var/τ)', np.allclose(w, 1 / (1 + F['var'] / 0.01)))
    ok('кластер 16: g = 0 — нога, g = 1 — гибрид', np.allclose(F['leg'] + 0 * F['cbar'], F['leg'])
       and np.allclose(F['leg'] + F['cbar'], np.log(F['Pj'].mean(1))))
    ok('по seed: две компоненты seed 42 — среднее', np.allclose(F['cs']['42'], np.log(F['Pj'][:, :2].mean(1)) - F['leg']))
    ok('кластер 20 dolya: p ∈ [0, 1]', ((F['ppos'] >= 0) & (F['ppos'] <= 1)).all())
    # вложенный выбор: отложенный год не влияет на свой выбор
    b2 = pd.concat([b.assign(date=pd.bdate_range(f'{y}-01-03', periods=n), year=y, step=y - 2015) for y in (2017, 2018, 2019)])
    leg2 = b2.assign(rv_pred=np.exp(rng.normal(0, 0.3, len(b2))))
    act = leg2.rv_pred.values * np.exp(rng.normal(0, 0.4, len(b2)))
    comps2 = [b2[KEY].assign(rv_pred=leg2.rv_pred.values * np.exp(rng.normal(0.05, 0.1, len(b2))))]
    eta2 = b2.assign(rv_actual=act, har_raw=leg2.rv_pred.values, har_corr=leg2.rv_pred.values)
    F2 = assemble(leg2, comps2, [('a', '42')], eta2)
    vs = [(f'lam={l}', dict(lam=l), (lambda c, l=l: F2['leg'] + l * c), None) for l in (0.0, 0.5, 1.0)]
    ch, P = nested(F2, vs, 5, 'har_corr')
    eta3 = eta2.copy(); m19 = eta3.year.values == 2019
    eta3.loc[m19, 'rv_actual'] = eta3.loc[m19, 'rv_actual'] * np.exp(rng.normal(0, 1, int(m19.sum())))
    ch3, P3 = nested(assemble(leg2, comps2, [('a', '42')], eta3), vs, 5, 'har_corr')
    ok('LOYO: изменение 2019 не меняет выбор складки 2019', P.variant.iloc[2] == P3.variant.iloc[2])
    ok('LOYO: обучающие годы складки не содержат отложенный', all(str(r.god) not in r.obuch_gody.split(';') for r in P.itertuples()))
    S = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    S.to_csv(OUT / 'selftest_G.csv', index=False)
    print(S.to_string(index=False))
    print('selftest:', 'ВСЕ OK' if S.ok.all() else 'ЕСТЬ ОШИБКИ')
    return bool(S.ok.all())


# ================================================================ дым
def dym():
    import r2_ens as E
    t0 = time.time(); dc = CFG['dym']
    tick, steps, seed = list(dc['tickers']), list(dc['steps']), int(dc['seed'])
    steps_out = [s for s in steps if s in CFG['steps_out']]
    D = OUT / 'dym'; root = D / 'cache'; D.mkdir(parents=True, exist_ok=True)
    blk = D / 'blok_sdvig_dym.csv'
    A, harj = C.load_d0()  # все тикеры: RV рынка и обучение привратника; ряды — 3 тикера
    C.check_d0(A)
    assert A.date.dt.year.max() <= MAXY
    M = E.prep(A, harj)
    src = C.REGEN / 'R2' / 'cache'
    S, P, LK = [], [], []
    for h in dc['hs']:
        try:
            design = C.validate({'h': h, 'boost': {'role': 'residual'}})
        except C.DesignError:
            design = C.validate({'h': h})
        rd = lambda p: _val_only(pd.read_parquet(src / p), p)
        legsrc = rd(dc['noga'].format(h=h))
        C.write_series(design, 'noga_dym', legsrc[legsrc.step.isin(steps) & legsrc.ticker.isin(tick)], h, None, module='L', root=root)
        for p in dc['korr'][h]:
            x = rd(p); lib = 'xgb' if 'xgb' in p else 'lgbm'
            C.write_series(design, f'gib_{lib}', x[x.step.isin(steps) & x.ticker.isin(tick)], h, seed, module='B', root=root)
        sha = C.design_hash(design)
        leg, comps, meta = load_inputs(sha, h, 'noga_dym', ['gib_xgb', 'gib_lgbm'], seeds=[seed], steps=steps, tickers=tick, root=root)
        eta = load_etalon(h, tick, steps)
        F = assemble(leg, comps, meta, eta, steps_out, CFG['vhod']['agg'])
        s, p = run_h(design, h, F, A, M, steps, root=root, block_file=blk)
        S.append(s); P.append(p)
        LK.append(leak(h, leg, comps, meta, eta, A, harj, steps, steps_out))
        Lr = C.load_series(sha, h=h, modul=MODUL, root=root)
        assert set(Lr.ryad.unique()) == set(s.ryad), 'load_series: не все ряды G'
        C.log(f'h={h}: {len(s)} рядов G, сдвиг ок {int(s.sdvig_ok.sum())}/{len(s)}')
    S, P, LK = pd.concat(S), pd.concat(P), pd.concat(LK)
    S.to_csv(D / 'svodka_dym.csv', index=False); P.to_csv(D / 'params_dym.csv', index=False); LK.to_csv(D / 'leak_dym.csv', index=False)
    st = selftest()
    ok = bool(S.sdvig_ok.all() and LK.ok.all() and st)
    print(S.round(4).to_string(index=False))
    print(LK[~LK.ok].to_string(index=False) if (~LK.ok).any() else f'детектор: {len(LK)} строк, все OK')
    print(f'дым G: {"ВСЕ OK" if ok else "ЕСТЬ ОШИБКИ"} за {time.time() - t0:.0f} с; панельная t — {PANEL_T_SRC}')
    return ok


def run(a):
    """Боевой счёт (фаза E): дизайн-кандидат → ряды G в его каталоге кэша. Входы — из yaml (vhod) или аргументов."""
    import r2_ens as E
    v = CFG['vhod']
    dizayn = a.dizayn or v['dizayn']
    noga = a.noga or v['noga']['ryad']; korr = a.korr or v['korr']['ryad']
    assert dizayn and noga and korr, 'нужны vhod.dizayn, vhod.noga.ryad, vhod.korr.ryad'
    if isinstance(dizayn, str) and Path(dizayn).exists():
        dizayn = C.validate(dizayn)
    A, harj = C.load_d0(); C.check_d0(A)
    M = E.prep(A, harj)
    steps = list(range(1, MAXY - 2015 + 1))
    sha = dizayn if isinstance(dizayn, str) else C.design_hash(dizayn)
    for h in a.h:
        leg, comps, meta = load_inputs(sha, h, noga, korr, seeds=v['korr']['seeds'], steps=steps, root=a.root,
                                       noga_hash=v['noga'].get('hash'), modul=v['korr'].get('modul'))
        F = assemble(leg, comps, meta, load_etalon(h, steps=steps), CFG['steps_out'], v['agg'])
        s, _ = run_h(dizayn, h, F, A, M, steps, root=a.root)
        print(s.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['selftest', 'dym', 'run'])
    ap.add_argument('--h', type=int, nargs='+', default=[1, 5, 22])
    ap.add_argument('--dizayn'); ap.add_argument('--noga'); ap.add_argument('--korr', nargs='+'); ap.add_argument('--root')
    a = ap.parse_args()
    if a.cmd == 'selftest':
        sys.exit(0 if selftest() else 1)
    if a.cmd == 'dym':
        sys.exit(0 if dym() else 1)
    run(a)


if __name__ == '__main__':
    main()
