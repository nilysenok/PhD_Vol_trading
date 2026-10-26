#!/usr/bin/env python3
"""R2, агент G: воздержание (кластер 16), усадка к эталону (кластер 18), ограниченная коррекция «знак × δ» (кластер 20).
Протокол v4 docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md (раздел 6, раздел 7 строка G); параметры — models/designs/r2/r2_G_abstain.yaml.

Входы (через r2_core.load_series по хэшу дизайна-кандидата): нога L (ln ŷ_L) и гибриды-компоненты seed × lib (роли B, M, E).
Коррекция компоненты c_j = ln ŷ_j − ln ŷ_L; коррекция ансамбля c̄ = ln mean_j ŷ_j − ln ŷ_L (как r2_ens.uncert_shrink).

Кластер 16 — ŷ = нога · exp(g_t · c), g_t ∈ [0, 1] по данным ≤ t:
  conf_seed    g = 1{SD_j(c_j) < τ·|c̄|}                                  τ — сетка, выбор вложенно
  regime       g = p0 = P(спокойно на горизонте h) из r2_ens.moe_proba агента E (импорт — один источник вероятностей привратника
               кластера 13: обучение шага с эмбарго h, метка — ln y_h ниже медианы тикера); по желанию изотонная калибровка E
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
r2_core.params_path(дизайн, 'G') = <кэш дизайна>/params_G.csv (PR #201; читать — r2_core.load_params). Прогнозы — только val 2017–2021 (шаги 2–6).

Обвязка: проверка сдвига п. 2.6 (r2_core.shift_check — каждый выходной ряд) и детектор «будущее не влияет»: при фиксированных
параметрах возмущение всех входов после среза (ряды, эталон, матрица d0 — r2_core.perturb_frame) не меняет g_t и ŷ на строках ≤ среза;
подлог «вход d := d+1» пойман; граница года — эмбарго привратника regime (без эмбарго — протекает: положительный контроль).
Вложенный выбор LOYO по построению смотрит на другие годы T — это гиперпараметр внутри складки (протокол v3 § 2.1), не g_t.

Боевой счёт — контракт драйвера S (PR #184): ключи schema_v2 k16.* / k18.* / k20.*, r2_core.run_phase (module = G, DONE_G_h{h}_s{seed});
входы — ряды цепочки в каталоге дизайна (нога L + предыдущий модуль chain_order); по ряду на точку сетки и seed (варианты для O).

python3 scripts/analysis/r2_gate_abstain.py selftest
python3 scripts/analysis/r2_gate_abstain.py dym        # Мак: 3 тикера × шаги 1–3 × seed 42 × h 1, 5, 22; + контракт run_batch (9 дизайнов v2 × h)
python3 scripts/analysis/r2_gate_abstain.py run --designs <yaml> --h 5 --seeds 42 43 44 --steps 1 2 3 4 5 6 --workers 10 [--phase E]
python3 scripts/analysis/r2_gate_abstain.py run --dizayn <yaml> --h 5      # один дизайн (seed — по рядам каталога)
python3 scripts/analysis/r2_gate_abstain.py leak [--designs <yaml>] --h 5"""
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


# --- regime: P(спокойно) = p0 из r2_ens.moe_proba (агент E); слагаемые привратника — рыночные S3
REG_TERMS = ['ln_rvi2', 'ln_rvm_imoex', 'lhar_1', 'lhar_5', 'lhar_22', 'pc1']


def regime_P(M, h, steps, gate='logit', kalibr='none', seed=42, plant=False):
    """-> DataFrame(date, ticker, step, g): g = p0 = P(спокойно) из r2_ens.moe_proba агента E (импорт — один источник вероятностей
    привратника кластера 13: обучение шага с эмбарго h, калибровка по OOS шагов < k). plant — подлог E «обучение видит val-год»."""
    import r2_ens as E
    P = E.moe_proba(M, h, REG_TERMS, mode='k2', gate=gate, calib=kalibr, seed=seed, steps=tuple(sorted(steps)), plant=plant)
    P = P.assign(date=pd.to_datetime(P['date']), ticker=P['ticker'].astype(str))
    return P[KEY].assign(g=P['p0'].values.astype(float))


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


def write_params(design, P, root=None):
    """Свой файл параметров r2_core.params_path(дизайн, 'G') = <кэш>/<hash16>/params_G.csv (общий params.csv не пишется, PR #201):
    строки тех же рядов и h заменяются, остальные сохраняются; запись атомарная. Читать — r2_core.load_params(хэш, modul='G')."""
    f = C.params_path(design, MODUL, root=root)
    P = P.drop(columns=['modul'], errors='ignore')
    if f.exists():
        old = pd.read_csv(f)
        if 'ryad' in old and 'h' in old:
            old = old[~(old.ryad.isin(P.ryad.unique()) & old.h.isin(P.h.unique()))]
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
        write_params(full, P, root=root)
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
    # граница года: эмбарго привратника regime (r2_ens.moe_proba). Возмущаются только цели, чьё окно заходит за 31.12 (и все цели
    # после); признаки не трогаются. P модели шага vy = год + 1 не меняется; с подлогом E «обучение видит val-год» — меняется.
    M0 = E.prep(A, harj)
    for ye in CFG['leak']['cuts_year_end']:
        cut = pd.Timestamp(ye); k = cut.year + 1 - 2015
        rng = np.random.default_rng(seed + 99)
        Ap = A.copy(); tc = f'rv_target_h{h}'
        cross = ((pd.to_datetime(Ap.date) > cut) | ~(pd.to_datetime(Ap[f'_tend{h}']) <= cut)).values
        Ap.loc[cross, tc] = Ap.loc[cross, tc].values * rng.uniform(0.5, 2.0, int(cross.sum()))
        Mp = E.prep(Ap, harj)
        for plant in (False, True):
            p0, p1 = regime_P(M0, h, [k], plant=plant), regime_P(Mp, h, [k], plant=plant)
            d = float(np.abs(p0.g.values - p1.g.values).max())
            rows.append(dict(h=h, ryad=f'G16_regime#emb={not plant}', srez=str(cut.date()), kmax=k, strok_do=len(p0), max_dln_do=d,
                             max_dln_posle=np.nan, podlog_max_dln_do=np.nan,
                             ok=bool(d > 0) if plant else bool(d == 0.0)))  # без эмбарго должно протекать (положительный контроль)
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
    # E8: нули сетки (ключи схемы обходятся: K задаётся явно)
    K0 = dict(gate='shock_fresh', tau=[1.0], k=[0, 3], q=[0.9], measure='uroven', regime_calib='none', lam_to_ref='on', lam=[0, 1.0],
              sign_delta='off', delta=[0.1], base='noga', ref='noga')
    Ad = pd.DataFrame({'date': F2['base'].date, 'ticker': 'X', 'rv_d': F2['Pj'][:, 0], '_tend5': pd.NaT})
    V0 = {t: fn for t, _, fn, _ in design_variants(K0, F2, Ad, None, 5, [2, 3, 4])}
    ok('E8: k = 0 — один вариант без q', sum(t.startswith('shock_fresh__k=0') for t in V0) == 2)
    ok('E8: k = 0, λ = 1 — гибрид', np.allclose(V0['shock_fresh__k=0__lam=1.0_k_noge'](F2['cbar']), F2['leg'] + F2['cbar']))
    ok('E8: λ = 0 к ноге — нога', np.allclose(V0['shock_fresh__k=0__lam=0_k_noge'](F2['cbar']), F2['leg']))
    K1 = {**K0, 'ref': 'etalon'}
    V1 = {t: fn for t, _, fn, _ in design_variants(K1, F2, Ad, None, 5, [2, 3, 4])}
    ok('кластер 18: λ = 0 к эталону — эталон', np.allclose(V1['shock_fresh__k=0__lam=0'](F2['cbar']), np.log(F2['base'].har_corr.values)))
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
    D = OUT / 'dym'; root = D / 'cache_p'; D.mkdir(parents=True, exist_ok=True)  # cache_p — кэш дыма после PR #201 (params_G.csv)
    blk = D / 'blok_sdvig_dym.csv'
    A, harj = C.load_d0()  # все тикеры: RV рынка и обучение привратника; ряды — 3 тикера
    C.check_d0(A)
    assert A.date.dt.year.max() <= MAXY
    M = E.prep(A, harj)
    src = C.REGEN / 'R2' / 'cache'
    S, P, LK = [], [], []
    for h in dc['hs']:
        try:
            design = C.validate({'h': h, 'boost': {'role': 'residual', 'early_stop': 'qlike_total'}})
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


# ================================================================ боевой счёт по контракту драйвера S (PR #184, docs/server_r2.md § 10)
# Ключи дизайна — schema_v2: k16.gate (off|conf_seed|regime|shock_fresh|halt|soft), k16.tau/k/q (сетки), k16.measure, k16.regime_calib;
# k18.lam_to_ref (off|on), k18.lam (сетка); k20.sign_delta (off|znak|dolya), k20.delta (сетка), k20.base (noga|etalon).
# Цепочка L → … → E → G пишет в один каталог дизайна: G читает ногу (modul L) и ряды предыдущего модуля цепочки по r2_core.load_series,
# пишет по ряду на каждую точку сетки (варианты для r2_gates.ocenka: вложенный выбор по t делает O) и на каждый seed; params.csv —
# справочно собственный вложенный выбор G (тем же правилом). Порядок: k20 (коррекция := δ·s) → k16 (g_t·коррекция, halt) → k18 (усадка).
SDRV = yaml.safe_load((REPO / 'models' / 'designs' / 'r2' / 'r2_S_driver.yaml').read_text()) \
    if (REPO / 'models' / 'designs' / 'r2' / 'r2_S_driver.yaml').exists() else {}
CHAIN = list(SDRV.get('chain_order') or ['L', 'P', 'R', 'B', 'BG', 'M', 'L3', 'E', 'G'])
LIB_RE = r'(?:^|_)(xgb|lgbm|xgboost|lightgbm)(?=_|$)'


def _ls(v):
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _onoff(v):
    return {False: 'off', True: 'on'}.get(v, v) if isinstance(v, bool) else str(v)


def _opt(full, dotted, default):
    """Ключ, которого может не быть в схеме (k18.ref — ❓ S, schema): нет — значение по умолчанию из yaml G."""
    try:
        return str(_onoff(C.get(full, dotted)))
    except (KeyError, TypeError):
        return str(default)


def g_keys(full):
    """Ключи G дизайна (schema_v2) -> dict; None, если все группы выключены."""
    g = lambda k: C.get(full, k)
    K = dict(gate=_onoff(g('k16.gate')), tau=_ls(g('k16.tau')), k=_ls(g('k16.k')), q=_ls(g('k16.q')), measure=str(g('k16.measure')),
             regime_calib=str(g('k16.regime_calib')), lam_to_ref=_onoff(g('k18.lam_to_ref')), lam=_ls(g('k18.lam')),
             sign_delta=_onoff(g('k20.sign_delta')), delta=_ls(g('k20.delta')), base=str(g('k20.base')),
             ref=_opt(full, 'k18.ref', CFG['k18'].get('ref', 'etalon')))
    for nm, v in (('k16.k', K['k']), ('k18.lam', K['lam'])):  # 0 допускается (E8): k = 0 — без воздержания, λ = 0 — ref
        if any(float(x) < 0 for x in v):
            raise C.DesignError(f'{nm}: отрицательные значения {v}')
    if K['gate'] == 'off' and K['lam_to_ref'] == 'off' and K['sign_delta'] == 'off':
        return None
    return K


def design_variants(K, F, A, M, h, steps):
    """-> [(tag, параметры, fn(c) -> ln ŷ, g | None)] — декартово произведение сеток включённых групп."""
    import itertools
    import r2_ens as E
    hg = halt_for(F, A, h)
    leg = F['leg']
    p16 = [('', {}, np.ones(len(leg)))]
    if K['gate'] == 'conf_seed':
        p16 = [(f'tau={t}', dict(tau=t), g_conf(F, float(t))) for t in K['tau']]
    elif K['gate'] == 'soft':
        p16 = [(f'tau={t}', dict(tau=t), g_soft(F, t)) for t in E.TAUS]
    elif K['gate'] == 'shock_fresh':
        mrv = market_rv(A)
        # k = 0 — без воздержания (E8, владелец 20:20): g ≡ 1, один вариант без q
        p16 = [('k=0', dict(k=0), np.ones(len(leg)))] if any(float(k) == 0 for k in K['k']) else []
        p16 += [(f'k={k}__q={q}', dict(k=k, q=q), g_shock(F, shock_flags(mrv, k, q, int(CFG['k16']['shock_fresh']['okno']), K['measure'])))
                for k in K['k'] if float(k) > 0 for q in K['q']]
    elif K['gate'] == 'regime':
        RP = regime_P(M, h, steps, CFG['k16']['regime']['gate'], K['regime_calib'])
        p16 = [('P_spokoyno', dict(regime_calib=K['regime_calib']), g_regime(F, RP))]
    p16 = [(t, p, g * hg) for t, p, g in p16]  # halt — свойство модели G (в val g_halt ≡ 1)
    p20 = [('', {}, None)]
    if K['sign_delta'] != 'off':
        base = leg if K['base'] == 'noga' else np.log(np.clip(F['base'][CFG['etalon']['ref']].values, 1e-12, None))
        p20 = [(f'{K["sign_delta"]}_delta={d}', dict(sign_delta=K['sign_delta'], delta=d, base=K['base']), (base, float(d)))
               for d in K['delta']]
    p18 = [('', {}, None)]
    if K['lam_to_ref'] == 'on':
        # ref = etalon — эталон В2 (кластер 18); ref = noga — нога (E8: ŷ = нога + λ·(кандидат − нога)); λ = 0 — сам ref
        har = leg if K['ref'] == 'noga' else np.log(np.clip(F['base'][CFG['k18']['har']].values, 1e-12, None))
        p18 = [(f'lam={l}' + ('_k_noge' if K['ref'] == 'noga' else ''), dict(lam=l, ref=K['ref']), (har, float(l))) for l in K['lam']]
    out = []
    for (t16, a16, g), (t20, a20, b20), (t18, a18, b18) in itertools.product(p16, p20, p18):
        def fn(c, g=g, b20=b20, b18=b18):
            if b20 is not None:
                s = np.sign(c) if K['sign_delta'] == 'znak' else 2 * F['ppos'] - 1
                c = b20[0] + b20[1] * s - leg
            lnp = leg + g * c
            if b18 is not None:
                lnp = b18[0] + b18[1] * (lnp - b18[0])
            return lnp
        tag = '__'.join(x for x in (K['gate'] if K['gate'] != 'off' else '', t16, t20, t18) if x) or 'base'
        out.append((tag, {**a16, **a20, **a18}, fn, g))
    return out


def chain_inputs(full, h, steps, root=None):
    """Ряды цепочки в каталоге дизайна: нога (modul L) и ряды предыдущего модуля (последний перед G в chain_order).
    -> (leg, {вариант предыдущего модуля: (компоненты, meta)}, модуль)."""
    import re
    sha = C.design_hash(full)
    S = C.load_series(sha, h=h, steps=steps, root=root)
    if S.empty:
        raise FileNotFoundError(f'G: в каталоге {sha[:16]} нет рядов h={h}')
    mods = [m for m in CHAIN if m in set(S.modul) and m not in ('G', 'L')]
    if not mods:
        raise FileNotFoundError(f'G: нет рядов предыдущего модуля цепочки (есть {sorted(set(S.modul))}) — коррекции нет')
    prev = mods[-1]
    Lr = S[S.modul == 'L']
    want = CFG['vhod']['noga'].get('ryad')
    names = sorted(Lr.ryad.unique())
    if not names:
        raise FileNotFoundError(f'G: нет ноги (modul L) в каталоге {sha[:16]}')
    P = S[S.modul == prev]
    groups = {}
    for (ry, sd), g in P.groupby(['ryad', 'seed'], sort=True):
        pv = re.sub(LIB_RE, '', ry) or ry
        groups.setdefault(pv, ([], [], []))
        groups[pv][0].append(_val_only(g, ry)[KEY + ['rv_pred']]); groups[pv][1].append((ry, str(sd))); groups[pv][2].append(ry)

    def leg_of(pv):
        """Нога варианта: одна нога — она; сетка ноги (leg.enet и т. п.: ряды leg__<вариант>, у B — B_…__<вариант>[__cal], как R в #199) —
        нога, чей <вариант> стоит в имени ряда отдельным сегментом «__…__» (самое длинное совпадение)."""
        if want:
            return want
        if len(names) == 1:
            return names[0]
        key = lambda n: n.split('__', 1)[1] if n.startswith('leg__') else n
        hit = [n for n in names if re.search(r'(^|__)' + re.escape(key(n)) + r'(__|$)', pv)]
        if not hit:
            raise ValueError(f'G: ног несколько {names}, вариант {pv} не сопоставлен — задайте vhod.noga.ryad')
        return max(hit, key=lambda n: len(key(n)))

    def leg_df(n):
        L = Lr[Lr.ryad == n]
        return _val_only(L[L.seed == sorted(L.seed.unique())[0]], n)[KEY + ['year', 'rv_pred']]

    # L3 после B: ряды L3 — калиброванный гибрид. Нога переводится в тот же масштаб: ln ноги += ln L3 − ln B (поправка калибровки
    # строки, среднее по seed), тогда коррекция = ln B − ln ноги (сырая поправка бустинга), а g = 0 даёт ногу с той же калибровкой.
    Bm = S[S.modul == 'B'] if prev == 'L3' else None
    out = {}
    for pv, (comps, meta, rys) in groups.items():
        L = leg_df(leg_of(pv))
        if Bm is not None and len(Bm):
            sh = []
            for c, ry, (_, sd) in zip(comps, rys, meta):
                b = Bm[(Bm.ryad == re.sub(r'__cal$', '', ry)) & (Bm.seed.astype(str) == sd)]
                if len(b):
                    m = c.merge(_val_only(b, ry)[KEY + ['rv_pred']], on=KEY, suffixes=('', '_b'))
                    sh.append(m[KEY].assign(d=np.log(m.rv_pred.values) - np.log(m.rv_pred_b.values)))
            if sh:
                d = pd.concat(sh).groupby(KEY, as_index=False)['d'].mean()
                L = L.merge(d, on=KEY)
                L = L.assign(rv_pred=L.rv_pred.values * np.exp(L.d.values)).drop(columns='d')
        out[pv] = (L, comps, meta)
    return out, prev


def fn_G(design, seed, steps):
    """Функция фазы для r2_core.run_phase: -> {ряд: DataFrame}. Данные — r2_core.SHARED: A, harj, G_M (r2_ens.prep), G_root."""
    full = C.validate(design); h = int(full['h'])
    K = g_keys(full)
    A, M, root = C.SHARED['A'], C.SHARED['G_M'], C.SHARED.get('G_root')
    tick = C.SHARED.get('G_tickers')
    groups, prev = chain_inputs(full, h, steps, root)
    eta = load_etalon(h, tick, steps)
    out, prows = {}, []
    for pv, (leg, comps, meta) in groups.items():
        F = assemble(leg, comps, meta, eta, list(steps), CFG['vhod']['agg'])
        c = F['cs'].get(str(seed), F['cbar'])
        var = design_variants(K, F, A, M, h, list(steps))
        for tag, par, fn, g in var:
            nm = f'G_{tag}' if len(groups) == 1 else f'G_{pv}__{tag}'
            b = F['base']
            out[nm] = pd.DataFrame({'date': b.date.values, 'ticker': b.ticker.values, 'year': b.year.values, 'step': b.step.values,
                                    'rv_actual': b.rv_actual.values, 'rv_pred': np.exp(fn(c))})
        if str(seed) == min(F['cs'], key=lambda s: (len(s), s)) or str(seed) not in F['cs']:  # params.csv — один раз на дизайн
            vs = [(t, p, (lambda cc, fn=fn: fn(cc)), g) for t, p, fn, g in var]
            ch, P = nested(F, vs, h, CFG['etalon']['ref'], CFG['vybor']['rezhim'])
            prows.append(P.assign(ryad=(f'G_*' if len(groups) == 1 else f'G_{pv}__*'), h=h, vhod_modul=prev, panel_t=PANEL_T_SRC))
    if prows:
        write_params(full, pd.concat(prows, ignore_index=True), root=root)
    return out


def load_designs(p):
    Y = yaml.safe_load(Path(p).read_text())
    return Y['designs'] if isinstance(Y, dict) and 'designs' in Y else (Y if isinstance(Y, list) else [Y])


def run_batch(designs, h, seeds, steps, workers=None, phase='E', root=None, check_gate=True, block_file=None, tickers=None):
    """Контракт драйвера: дизайны (список dict / yaml) -> r2_core.run_phase (ворота, кэш, запись, сдвиг, DONE_G_h{h}_s{seed}).
    Дизайны без включённых ключей 16/18/20 пропускаются с записью в журнал."""
    import r2_ens as E
    A, harj = C.load_d0(); C.check_d0(A)
    assert A.date.dt.year.max() <= MAXY
    C.share(A=A, harj=harj, G_M=E.prep(A, harj), G_root=root, G_tickers=tickers)
    ok = []
    for d in (load_designs(designs) if isinstance(designs, (str, Path)) else designs):
        full = C.validate({**d, 'h': h})
        if g_keys(full) is None:
            C.log(f'G: не применим (k16/k18/k20 выключены): {full.get("design")}')
            continue
        ok.append(full)
    return C.run_phase(phase, h, ok, fn_G, seeds=tuple(seeds), steps=tuple(steps), workers=workers, module=MODUL, root=root,
                       block_file=block_file, check_gate=check_gate)


def seeds_in_dir(full, h, root=None):
    S = C.read_index(root)
    S = S[(S['hash'].astype(str) == C.design_hash(full)[:16]) & (S['modul'] != MODUL) & (S['modul'] != 'L')]
    s = sorted({int(x) for x in S['seed'].astype(str) if x.isdigit()})
    return s or [42]


def leak_cmd(h_list, designs=None):
    """Детектор по контракту (`leak --designs --h`): входы дыма, все семейства при фиксированных параметрах (ряды G не зависят от
    дизайна-носителя: g_t считается одинаково). Код возврата ≠ 0 — провал."""
    import r2_ens as E
    dc = CFG['dym']; tick, steps = list(dc['tickers']), list(dc['steps'])
    steps_out = [s for s in steps if s in CFG['steps_out']]
    A, harj = C.load_d0(); C.check_d0(A)
    src = C.REGEN / 'R2' / 'cache'
    LK = []
    for h in h_list:
        rd = lambda p: _val_only(pd.read_parquet(src / p), p)
        L = rd(dc['noga'].format(h=h)); L = L[L.step.isin(steps) & L.ticker.isin(tick)]
        comps = [rd(p) for p in dc['korr'][h]]
        comps = [c[c.step.isin(steps) & c.ticker.isin(tick)][KEY + ['rv_pred']] for c in comps]
        meta = [('xgb' if 'xgb' in p else 'lgbm', str(dc['seed'])) for p in dc['korr'][h]]
        LK.append(leak(h, L[KEY + ['year', 'rv_pred']], comps, meta, load_etalon(h, tick, steps), A, harj, steps, steps_out))
    LK = pd.concat(LK)
    OUT.mkdir(parents=True, exist_ok=True)
    LK.to_csv(OUT / 'leak_G.csv', index=False)
    print(LK[~LK.ok].to_string(index=False) if (~LK.ok).any() else f'детектор G: {len(LK)} строк, все OK')
    return bool(LK.ok.all())


def dym_batch(A_done=None):
    """Дым контракта: дизайны schema_v2 (по одному на ворота/группу) × h; входы цепочки (L — нога, B — гибриды xgb/lgbm) пишутся
    в кэш дыма под хэш каждого дизайна; run_batch с 3 воркерами, без ворот; проверки — ряды G по вариантам, DONE, params.csv, повтор."""
    dc = CFG['dym']; tick, steps, seed = list(dc['tickers']), list(dc['steps']), int(dc['seed'])
    # свой кэш на версию кода (SHA модуля): новый код — первый прогон заново (DONE от прежней версии не мешает), повтор — пропуск
    D = OUT / 'dym_batch'; root = D / f'cache_{C.sha_file(Path(__file__))[:8]}'; D.mkdir(parents=True, exist_ok=True)
    src = C.REGEN / 'R2' / 'cache'
    G_DES = [{'k16': {'gate': 'conf_seed'}}, {'k16': {'gate': 'regime'}}, {'k16': {'gate': 'shock_fresh', 'k': [3, 5]}},
             {'k16': {'gate': 'soft'}}, {'k16': {'gate': 'halt'}}, {'k18': {'lam_to_ref': 'on'}},
             {'k20': {'sign_delta': 'znak'}}, {'k20': {'sign_delta': 'dolya', 'base': 'etalon'}},
             {'k16': {'gate': 'conf_seed', 'tau': [0.5, 1.0]}, 'k18': {'lam_to_ref': 'on', 'lam': [0.5, 1.0]}}]
    rows = []
    for h in dc['hs']:
        des = []
        for i, gk in enumerate(G_DES):
            d = C.validate({'design': f'dymG_{i}', 'h': h, 'boost': {'role': 'residual', 'early_stop': 'qlike_total'}, **gk})
            rd = lambda p: _val_only(pd.read_parquet(src / p), p)
            L = rd(dc['noga'].format(h=h))
            C.write_series(d, 'noga_dym', L[L.step.isin(steps) & L.ticker.isin(tick)], h, None, module='L', root=root)
            for p in dc['korr'][h]:
                x = rd(p); lib = 'xgb' if 'xgb' in p else 'lgbm'
                C.write_series(d, f'gib_{lib}', x[x.step.isin(steps) & x.ticker.isin(tick)], h, seed, module='B', root=root)
            des.append(d)
        des.append(C.validate({'design': 'dymG_bez', 'h': h}))  # без ключей G — пропуск
        f = D / f'designs_h{h}.yaml'
        f.write_text(yaml.safe_dump(dict(designs=des), allow_unicode=True, sort_keys=False))
        pre = {C.design_hash(d)[:16] for d in des[:-1] if C.done_path(d, MODUL, h, seed, root).exists()}  # уже посчитано этой версией
        T = run_batch(str(f), h, [seed], steps, workers=3, root=root, check_gate=False, block_file=D / 'blok_sdvig.csv', tickers=tick)
        sdv = lambda s16: True if s16 in pre else bool(len(T) and (T.hash == s16).any() and T[T.hash == s16].sdvig_ok.all())
        T2 = run_batch(str(f), h, [seed], steps, workers=3, root=root, check_gate=False, block_file=D / 'blok_sdvig.csv', tickers=tick)
        for i, d in enumerate(des[:-1]):
            sha = C.design_hash(d)
            Ls = C.load_series(sha, h=h, modul=MODUL, root=root)
            K = g_keys(C.validate(d))
            nexp = (len(K['tau']) if K['gate'] == 'conf_seed' else len(__import__('r2_ens').TAUS) if K['gate'] == 'soft'
                    else len(K['k']) * len(K['q']) if K['gate'] == 'shock_fresh' else 1) \
                * (len(K['delta']) if K['sign_delta'] != 'off' else 1) * (len(K['lam']) if K['lam_to_ref'] == 'on' else 1)
            pc = C.params_path(d, MODUL, root=root)
            lp = C.load_params(sha, modul=MODUL, root=root)
            stary = (C.cache_dir(d, root=root) / 'params.csv').exists()  # общий params.csv не пишется (PR #201)
            rows.append(dict(h=h, design=d['design'], hash=sha[:16], ryadov=Ls.ryad.nunique(), ozhidalos=nexp,
                             shagi=','.join(map(str, sorted(Ls.step.unique()))) if len(Ls) else '',
                             done=C.done_path(d, MODUL, h, seed, root).exists(), params=bool(pc.exists() and len(lp) and not stary),
                             sdvig_ok=sdv(sha[:16]),
                             ok=bool(Ls.ryad.nunique() == nexp and C.done_path(d, MODUL, h, seed, root).exists()
                                     and pc.exists() and len(lp) and not stary and sdv(sha[:16]))))
        rows.append(dict(h=h, design='povtor', ryadov=len(T2), ozhidalos=0, ok=bool(len(T2) == 0)))  # готовое пропущено (DONE_G)
        # E5: нога с сеткой leg.enet (ряды leg__<вариант>), B_rf50__<вариант>, L3 B_rf50__<вариант>__cal (= B × 1,05) — как в кэше уровня E
        de = C.validate({'design': 'dymG_enet', 'h': h, 'leg': {'type': 'HARX'}, 'boost': {'role': 'residual', 'early_stop': 'qlike_total'},
                         'k16': {'gate': 'shock_fresh', 'k': [3], 'q': [0.9]}})
        rd = lambda p: _val_only(pd.read_parquet(src / p), p)
        L0 = rd(dc['noga'].format(h=h)); L0 = L0[L0.step.isin(steps) & L0.ticker.isin(tick)]
        B0 = rd(dc['korr'][h][0]); B0 = B0[B0.step.isin(steps) & B0.ticker.isin(tick)]
        vv = ['leg.enet.alpha=0__leg.enet.lambda=0.1', 'leg.enet.alpha=1__leg.enet.lambda=1']
        for j, v in enumerate(vv):
            C.write_series(de, f'leg__{v}', L0.assign(rv_pred=L0.rv_pred * (1 + 0.1 * j)), h, None, module='L', root=root)
            C.write_series(de, f'B_rf50__{v}', B0.assign(rv_pred=B0.rv_pred * (1 + 0.1 * j)), h, seed, module='B', root=root)
            C.write_series(de, f'B_rf50__{v}__cal', B0.assign(rv_pred=B0.rv_pred * (1 + 0.1 * j) * 1.05), h, seed, module='L3', root=root)
        fe = D / f'designs_enet_h{h}.yaml'
        fe.write_text(yaml.safe_dump(dict(designs=[de]), allow_unicode=True, sort_keys=False))
        preE = C.done_path(de, MODUL, h, seed, root).exists()
        Te = run_batch(str(fe), h, [seed], steps, workers=1, root=root, check_gate=False, block_file=D / 'blok_sdvig.csv', tickers=tick)
        Ge = C.load_series(C.design_hash(de), h=h, modul=MODUL, root=root)
        okk = Ge.ryad.nunique() == 2
        for j, v in enumerate(vv):  # g = 1 — L3; g = 0 — нога × 1,05 (калибровка L3 перенесена на ногу); варианты не перепутаны
            gg = Ge[Ge.ryad == f'G_B_rf50__{v}__cal__shock_fresh__k=3__q=0.9'].merge(L0[KEY + ['rv_pred']], on=KEY, suffixes=('', '_L'))
            gg = gg.merge(B0[KEY + ['rv_pred']], on=KEY, suffixes=('', '_B'))
            r_cal = gg.rv_pred / (gg.rv_pred_B * (1 + 0.1 * j) * 1.05)
            r_leg = gg.rv_pred / (gg.rv_pred_L * (1 + 0.1 * j) * 1.05)
            okk &= bool(len(gg) and (np.isclose(r_cal, 1) | np.isclose(r_leg, 1)).all() and np.isclose(r_leg, 1).any())
        rows.append(dict(h=h, design='dymG_enet', ryadov=Ge.ryad.nunique(), ozhidalos=2, ok=bool(okk and (preE or (len(Te) and Te.sdvig_ok.all())))))
        rows.append(dict(h=h, design='dymG_bez', ryadov=0, ozhidalos=0,
                         ok=not C.done_path(des[-1], MODUL, h, seed, root).exists()))
    R = pd.DataFrame(rows)
    R.to_csv(D / 'svodka_batch.csv', index=False)
    print(R.to_string(index=False))
    return bool(R.ok.all())


# ================================================================ поручение владельца 21:00: E8 поверх E1 / E1b (h = 22), вне драйвера
E8_RYADY = {'E1': ('3e8cf0c43a642d7a', 'M_E1'), 'E1b': ('c6d88471794d13c3', 'M_E1b')}
E8_LAM = (0.0, 0.25, 0.5, 0.75, 1.0)
E8_K = (0, 3, 5, 10)
E8_Q = 0.95  # q порога скачка shock_fresh — середина сетки k16.q (владелец задал только k)


def _e8_row(nm, r, s2=None):
    m = r['moshchnost']
    return dict(variant=nm, t_mean=r['t_mean'], skladok_t_ge_1=r['n_t_ge'], hit=r['hit'], tikery=f"{r['n_tikerov_vyigrysh']}/{r['n_tikerov']}",
                konc=r['konc'], c3=r['c3_vse'], vyigrysh_pct=r['vyigrysh_pct'], P=m['P_p_lt_005'], t_ozhid=m['t_ozhid'],
                vhod=r['vhod']['proshyol'], net=r['vhod']['net'],
                s2_passed=(s2 or {}).get('passed'), s2_d=(s2 or {}).get('d_mean'), s2_thr=(s2 or {}).get('thr'),
                folds=';'.join(f'{int(f.Y)}:{f.variant}:{f.t:.2f}:{"c3" if f.c3 else "-"}' for f in r['folds'].itertuples()))


# h = 5 — решение владельца 21:05: E1 и альтернатива E2 (forward HAR: hyb₁ вместо RV_d, агент M, PR #220)
E8_RYADY_H = {22: E8_RYADY, 5: {'E1': ('855ccbb0a71ff5f6', 'M_E1'), 'E2': ('140243b0a0942055', 'M_E2')}}


def e8_e1(h=22, write=True, ryady=None, order=None):
    """order: 'lam_k' (21:00, h = 22) — сетка λ, затем k поверх лучшего λ; 'k_lam' (21:05, h = 5) — сначала воздержание
    (halt, k ∈ {3, 5, 10} при λ = 1), затем λ поверх лучшего k."""
    """ŷ = нога + λ·(E1 − нога) в ln (нога — эталон В2 ETA_b_d0_s2emb), λ ∈ E8_LAM; затем воздержание shock_fresh k ∈ E8_K поверх
    лучшего λ (правило выбора O: r2_gates.vybor; если вход не прошёл никто — максимум t̄). Оценка — r2_gates.ocenka (панельная t,
    вход кластера 17, c3, мощность; вложенный итог — ocenka по всем вариантам: λ складки Y — по остальным 4); 2σ — r2_gates.sigma2_F
    по 10 seed парно с F_h(d0) s2emb. Метрики — на среднем по 10 seed. Ряды (по seed) — в кэш дизайна E1 / E1b, modul = G."""
    import r2_driver as DR
    import r2_gates as RG
    A, harj = C.load_d0(); C.check_d0(A)
    assert A.date.dt.year.max() <= MAXY
    E = DR.etalon(h)
    assert E.date.max() < pd.Timestamp('2022-01-01')
    Fe = DR.F_emb(h)
    seeds = list(SDRV['E']['seeds_final'])
    mrv = market_rv(A)
    gk = {k: (shock_flags(mrv, k, E8_Q, int(CFG['k16']['shock_fresh']['okno']), 'uroven') if k > 0 else None) for k in E8_K}
    OUTd = OUT / 'E8_E1'; OUTd.mkdir(parents=True, exist_ok=True)
    tabs, sv = [], []
    ryady = ryady or E8_RYADY_H[h]
    order = order or ('k_lam' if h == 5 else 'lam_k')
    for nm, (hsh, ryad) in ryady.items():
        full = design_of(hsh)
        S = C.load_series(hsh, ryad=ryad, h=h, steps=list(range(1, 7)))
        S = _val_only(S, ryad)
        Lg = E[KEY + ['year', 'rv_actual', 'rv_pred']].rename(columns={'rv_pred': 'leg'})
        per = {}
        for s in seeds:
            x = S[S.seed.astype(str) == str(s)][KEY + ['rv_pred']].merge(Lg, on=KEY)
            per[s] = x.sort_values(KEY).reset_index(drop=True)
        base = per[seeds[0]][KEY + ['year', 'rv_actual']]
        for s in seeds:
            assert (per[s][KEY].values == base[KEY].values).all()
        lnL = np.log(per[seeds[0]].leg.values)
        lnC = {s: np.log(per[s].rv_pred.values) for s in seeds}
        lnCm = np.log(np.mean([per[s].rv_pred.values for s in seeds], axis=0))  # ансамбль 10 seed (как final_series драйвера)
        dates = pd.to_datetime(base.date)
        G = {k: (np.ones(len(base)) if gk[k] is None else dates.map(gk[k]).fillna(1.0).values) for k in E8_K}
        tend = base[['date', 'ticker']].merge(A[['date', 'ticker', f'_tend{h}']], on=['date', 'ticker'], how='left')[f'_tend{h}'].values
        hg = halt_g(base.date.values, tend, h)  # в val ≡ 1 (проверяется ниже)
        assert (hg == 1).all(), 'halt изменил строки val 2017–2021'
        Lv = per[seeds[0]].leg.values
        mk = lambda ln: base.assign(rv_pred=Lv * np.exp(ln - lnL))  # от ноги: λ = 0 или g = 0 — ровно нога (без шума exp(ln x) ≠ x)
        f = lambda lam, k, c: lnL + G[k] * hg * lam * (c - lnL)  # k = 0 — без воздержания (g = halt ≡ 1 в val)

        def block(vars_, tag):
            """vars_ — {имя: (λ, k)} -> строки (по варианту и вложенный итог) и словарь рядов."""
            R, rows = {}, []
            for v, (lam, k) in vars_.items():
                V = mk(f(lam, k, lnCm))
                r = RG.ocenka({v: V}, E, h)
                s2 = RG.sigma2_F({s: mk(f(lam, k, lnC[s])) for s in seeds}, h, seeds=seeds, F=Fe) if lam > 0 else \
                    RG.sigma2_F(mk(lnL), h, seeds=seeds, F=Fe)
                R[v] = r
                rows.append(dict(ryad=nm, blok=tag, lam=lam, k=k, **_e8_row(v, r, s2)))
            rn = RG.ocenka({v: mk(f(lam, k, lnCm)) for v, (lam, k) in vars_.items()}, E, h)  # вложенный: вариант складки — по 4 остальным
            pick = {int(fo.Y): fo.variant for fo in rn['folds'].itertuples()}
            yrs = base.year.values
            s2n = RG.sigma2_F({s: mk(np.select([yrs == y for y in pick], [f(*vars_[pick[y]], lnC[s]) for y in pick], lnL)) for s in seeds},
                              h, seeds=seeds, F=Fe)
            rows.append(dict(ryad=nm, blok=tag, lam=np.nan, k=np.nan, **_e8_row('вложенный', rn, s2n)))
            return R, rows, pick

        def luchshiy(R, vs, slozh):
            b, _ = RG.vybor(R, slozh)
            return (b, True) if b is not None else (max(R, key=lambda v: R[v]['t_mean']), False)

        if order == 'lam_k':
            lv = {f'lam={l}': (l, 0) for l in E8_LAM}
            R1, rows1, pick1 = block(lv, 'lam')
            best, vh = luchshiy(R1, lv, {v: 0 if l in (0.0, 1.0) else 1 for v, (l, _) in lv.items()})
            lam_b = lv[best][0]
            kv = {f'lam={lam_b}__k={k}': (lam_b, k) for k in E8_K}
            R2, rows2, pick2 = block(kv, f'k (λ = {lam_b})')
            sv.append(dict(ryad=nm, poryadok=order, luchshiy_1=best, vhod_1=vh, vlozh_1=pick1, vlozh_2=pick2))
        else:  # k_lam: (а) воздержание при λ = 1: halt (k = 0, g = halt ≡ 1 в val) и k ∈ {3, 5, 10}; (б) λ поверх лучшего k
            lv = {('halt' if k == 0 else f'k={k}'): (1.0, k) for k in E8_K}
            R1, rows1, pick1 = block(lv, 'k (λ = 1)')
            best, vh = luchshiy(R1, lv, {v: (0 if k == 0 else 1) for v, (_, k) in lv.items()})
            k_b = lv[best][1]
            kv = {f'{best}__lam={l}': (l, k_b) for l in E8_LAM}
            R2, rows2, pick2 = block(kv, f'λ ({best})')
            sv.append(dict(ryad=nm, poryadok=order, luchshiy_1=best, vhod_1=vh, vlozh_1=pick1, vlozh_2=pick2,
                           halt_max_dln=float(np.abs(f(1.0, 0, lnCm) - lnCm).max())))  # halt в val ничего не меняет: 0
        tabs += rows1 + rows2
        if write:  # ряды по seed: λ-сетка и k поверх лучшего λ; сдвиг — каждый ряд (ансамбль)
            for v, (lam, k) in {**lv, **kv}.items():
                name = f'G_E8_{nm}__{v}'
                sc = C.shift_check(mk(f(lam, k, lnCm)), A, h, name, full)
                sv.append(dict(ryad=name, sdvig_ok=sc['ok'], argmax=sc['argmax'], okonchatelno=sc.get('okonchatelno')))
                for s in seeds:
                    C.write_series(full, name, mk(f(lam, k, lnC[s])), h, s, module=MODUL, note=f'G E8 (владелец 21:00): {v}, нога ETA_b_d0_s2emb')
            P = pd.DataFrame([r for r in rows1 + rows2]).assign(h=h)
            write_params(full, P.assign(ryad=f'G_E8_{nm}__*'))
    Tb = pd.DataFrame(tabs)
    Tb.to_csv(OUTd / f'tablica_E8_h{h}.csv', index=False)
    Sv = pd.DataFrame(sv)
    Sv.to_csv(OUTd / f'svodka_E8_h{h}.csv', index=False)
    return Tb, Sv


def e8_leak(h=22):
    """Детектор на входах поручения: нога — эталон В2, компонента — E1 seed 42; все семейства при фиксированных параметрах."""
    import r2_driver as DR
    A, harj = C.load_d0(); C.check_d0(A)
    E = DR.etalon(h)
    steps = list(range(1, 7))
    leg = E[KEY + ['year', 'rv_pred']]
    hs_, ry_ = E8_RYADY_H[h]['E1']
    S = _val_only(C.load_series(hs_, ryad=ry_, h=h, seed=42, steps=steps), 'E1')
    eta = load_etalon(h, steps=steps)
    LK = leak(h, leg, [S[KEY + ['rv_pred']]], [('E1', '42')], eta, A, harj, steps, [2, 3, 4, 5, 6])
    LK.to_csv(OUT / 'E8_E1' / f'leak_E8_h{h}.csv', index=False)
    return LK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['selftest', 'dym', 'run', 'leak', 'e8-e1'])
    ap.add_argument('--designs', help='yaml со списком designs (контракт драйвера S)')
    ap.add_argument('--dizayn', help='один дизайн (прежняя строка драйвера: per_design)')
    ap.add_argument('--h', type=int, nargs='+', default=[1, 5, 22])
    ap.add_argument('--seeds', type=int, nargs='+')
    ap.add_argument('--steps', type=int, nargs='+', default=list(range(1, MAXY - 2015 + 1)))
    ap.add_argument('--workers', type=int)
    ap.add_argument('--phase', default='E')
    ap.add_argument('--root')
    a = ap.parse_args()
    if a.cmd == 'selftest':
        sys.exit(0 if selftest() else 1)
    if a.cmd == 'dym':
        sys.exit(0 if (dym() and dym_batch()) else 1)
    if a.cmd == 'leak':
        sys.exit(0 if leak_cmd(a.h, a.designs) else 1)
    if a.cmd == 'e8-e1':
        pd.set_option('display.width', 250)
        okv = True
        for h in a.h:  # владелец 21:05: сначала h = 5, затем h = 22
            Tb, Sv = e8_e1(h)
            print(Tb.drop(columns=['folds']).round(3).to_string(index=False))
            print(Sv.to_string(index=False))
            LK = e8_leak(h)
            print(LK[~LK.ok].to_string(index=False) if (~LK.ok).any() else f'детектор E8 h={h}: {len(LK)} строк, все OK')
            okv &= bool(LK.ok.all())
        sys.exit(0 if okv else 1)
    src = a.designs or a.dizayn
    assert src, 'run: нужен --designs <yaml> (или --dizayn <yaml>)'
    for h in a.h:
        seeds = a.seeds or seeds_in_dir(C.validate({**load_designs(src)[0], 'h': h}), h, a.root)
        T = run_batch(src, h, seeds, a.steps, a.workers, a.phase, a.root)
        print(T.to_string(index=False))
        if len(T) and ((T.ryad == 'ОШИБКА').any() or (T.sdvig_ok == False).any()):  # noqa: E712
            sys.exit(1)


if __name__ == '__main__':
    main()
