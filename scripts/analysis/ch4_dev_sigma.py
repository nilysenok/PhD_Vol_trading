#!/usr/bin/env python3
"""Шаг 0 раунда стратегий на разработке 2018–2021 (постановка владельца 26.10 22:20,
docs/zadaniya/2026-10-26_razrabotka_strategiy_v2.md, п. 0): σ_pred разработки из файлов открытия 21:36 и проверка сдвига.

Тест (годы ≥ 2022) закрыт. Правило доступа:
- из test_locked/ читаются ТОЛЬКО 4 файла R2/final/{final,etalon}_d0_s2emb_h{1,5}_test.parquet; до чтения их SHA-256 сверяются
  с models/designs/r2/r2_A_sborka.yaml: sobrano (иначе СТОП);
- каждый файл с данными (эти 4, features_pit_v3, прежние σ разработки) читается функцией read_dev: фильтр pyarrow date ≤ 2021-12-31
  при чтении, затем — до любых других операций — маска года ≤ 2021 и assert max(date) ≤ 2021-12-31; печати, describe и метрик до
  фильтра нет;
- пишется только data/regen/ch4_dev/.

σ_pred — кодом главы 3 / главы 4 (vpred, PR #124), функции импортируются из scripts/analysis/h2_d0_ch4.py (сам скрипт не
запускается — его build_series читает тест без фильтра):
  гибрид    — √rv_pred  final_d0_s2emb_h{1,5}  (F_h(d0) с s² сборки 21:35)             → sigma_hyb_h{1,5}_dev.parquet;
  HAR-J (б) — √har_corr etalon_d0_s2emb_h{1,5} (HAR-J с поправкой s² t−1 с эмбарго)   → sigma_harj_b_h{1,5}_dev.parquet;
  наивная   — √(среднее RV дней d−4…d) из features_pit_v3.rv (H2 d0, FINAL_DESIGN_strategies п. 11; прежняя √rv_w = RV дней
              d−5…d−1 отставала на день — здесь только контроль)                       → sigma_naive_dev.parquet.
Столбцы выхода: date, ticker, sigma_pred_h1 | sigma_pred_h5 (float32, сортировка date, ticker) — формат fastbt.s9.SOURCES.

Проверки (до любого счёта стратегий):
- сдвиг: r2_loyo.sdvig — Спирмен ln σ²_d с ln RV_{d+k}, k = −2…+1, медиана по тикерам, решение r2_loyo.sdvig_ok, K = {0},
  симметричное ε = 0,01; рядом профиль уровня и приращения r2_features.shift_table. Модельный ряд решает уровень, оконное среднее
  (наивная) — приращение (как h2_d0_ch4.shift_rows и R2/O blok_sdvig). Периоды 2017–2021 и 2018–2021, ряд проходит, если проходит в обоих;
- контроли: каждый ряд со сдвигом −1 (σ_d := σ_{d−1}) и с утечкой +1 (σ_d := σ_{d+1}); у наивной ещё прежняя √rv_w и утечка RV_{d+1}
  внутри окна — все обязаны блокироваться; справочно (в решение не входят) — та же проверка прежних σ разработки (prezhniy:);
- детектор «будущее не влияет»: устройство рядов открытия (ряд = сырой × F_sm, одна F_sm на шаг), пересборка наивной из RV с
  возмущением RV после среза, признаки главы 4 (ранг по прошлому окну W126/252/504, σ / медиана σ за 22 дня) с возмущением σ после среза;
- покрытие: первая/последняя дата каждого ряда по тикеру, история до 2018 для окна 504;
- сверка с прежними σ разработки (fastbt.s9 dev: final3_full, h5 final2, HAR-J final2, наивная √rv_w): корреляция по лагам −2…+2.

Ряд со сдвигом — СТОП (код выхода 2), в расчёты не допускается.

python3 scripts/analysis/ch4_dev_sigma.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import h2_d0_ch4 as H  # noqa: E402  только функции (sigma_ch3, naive_from_rv, plant_next, perturb); main не вызывается

MAIN = H.MAIN
REGEN = MAIN / 'data' / 'regen'
R2F = REGEN / 'test_locked' / 'R2' / 'final'
OUT = REGEN / 'ch4_dev'
SPEC = ROOT / 'models' / 'designs' / 'r2' / 'r2_A_sborka.yaml'
LAST = pd.Timestamp('2021-12-31')
EPS = 0.01
SEED = 20261026
CUTS = ('2017-12-29', '2018-06-29', '2019-12-31', '2020-03-02', '2020-12-31', '2021-06-30')
WINDOWS = (126, 252, 504)

# метка: (файл R2/final, столбец ряда, сырой столбец, h, выход, столбец выхода)
INPUTS = {
    'hyb_h1': ('final_d0_s2emb_h1_test.parquet', 'rv_pred', 'leg_raw', 1, 'sigma_hyb_h1_dev.parquet', 'sigma_pred_h1'),
    'hyb_h5': ('final_d0_s2emb_h5_test.parquet', 'rv_pred', 'leg_raw', 5, 'sigma_hyb_h5_dev.parquet', 'sigma_pred_h5'),
    'harj_b_h1': ('etalon_d0_s2emb_h1_test.parquet', 'har_corr', 'har_raw', 1, 'sigma_harj_b_h1_dev.parquet', 'sigma_pred_h1'),
    'harj_b_h5': ('etalon_d0_s2emb_h5_test.parquet', 'har_corr', 'har_raw', 5, 'sigma_harj_b_h5_dev.parquet', 'sigma_pred_h5'),
}
NAIVE_OUT = 'sigma_naive_dev.parquet'
# прежние σ разработки главы 4 (fastbt.s9 SOURCES, scope dev; FINAL_DESIGN_strategies п. 1–2, 9)
OLD = {
    'hyb_h1': (REGEN / 'sigma_pred_h1_final3_full.parquet', 'sigma_pred_h1'),
    'hyb_h5': (REGEN / 'sigma_pred_h5_final2.parquet', 'sigma_pred_h5'),
    'harj_b_h1': (REGEN / 'sigma_pred_h1_harj_final2.parquet', 'sigma_pred_h1'),
    'naive': (MAIN / 'data' / 'fastbt' / 's9' / 'sigma_naive_rvw.parquet', 'sigma_pred_h1'),
}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def now() -> str:
    return time.strftime('%F %T')


def read_dev(path, cols) -> pd.DataFrame:
    """Чтение с жёстким фильтром года ≤ 2021: фильтр pyarrow при чтении, затем маска и assert — до любых других операций."""
    d = pd.read_parquet(path, columns=cols, filters=[('date', '<=', LAST)])
    d = d[pd.to_datetime(d['date']).dt.year <= 2021].reset_index(drop=True)
    assert len(d) == 0 or pd.to_datetime(d['date']).max() <= LAST, f'{path}: дата > 2021-12-31 после фильтра'
    d['date'] = pd.to_datetime(d['date'])
    return d


def check_dev(df, what):
    mx = pd.to_datetime(df['date']).max()
    assert mx <= LAST, f'{what}: max date {mx} > 2021-12-31'


# ======================= 1. выборка σ =======================

def verify_inputs():
    spec = yaml.safe_load(SPEC.read_text())
    sob = spec['sobrano']
    got = {}
    for lab, (fn, *_r) in INPUTS.items():
        want = sob[f'R2/final/{fn}']
        g = sha(R2F / fn)
        if g != want:
            raise SystemExit(f'СТОП: SHA {fn} {g[:16]}… ≠ sobrano {want[:16]}…')
        got[fn] = g
    return got, sha(SPEC)


def build_model_series():
    raw, ser = {}, {}
    for lab, (fn, col, rawc, h, outn, oc) in INPUTS.items():
        d = read_dev(R2F / fn, ['date', 'ticker', 'step', col, rawc, 'F_sm'])   # фильтр ≤ 2021 — первой операцией
        raw[lab] = d
        s = H.sigma_ch3(d, col, oc)
        check_dev(s, lab)
        assert not s[oc].isna().any(), f'{lab}: NaN в σ'
        ser[lab] = (s, oc)
    return raw, ser


def rv_frame_dev():
    f = read_dev(REGEN / 'features_pit_v3.parquet', ['date', 'ticker', 'rv', 'rv_w'])
    return f.sort_values(['ticker', 'date']).reset_index(drop=True)


def naive_old(f):
    """Прежняя наивная разработки (fastbt.s9.sources): √rv_w, строка дня t = RV дней t−5…t−1 — только контроль."""
    nv = f[(f['date'] >= '2017-01-03') & (f['date'] <= LAST)][['date', 'ticker']].copy()
    nv['sigma_pred_h1'] = np.sqrt(f.loc[nv.index, 'rv_w'].astype(float)).astype('float32')
    return nv.dropna().sort_values(['date', 'ticker']).reset_index(drop=True)


# ======================= 2. проверка сдвига =======================

def shift_ts(df, col, k):
    """σ_d := σ_{d+k} по строкам тикера (k = −1 — сдвиг назад/отставание, +1 — утечка)."""
    d = df.sort_values(['ticker', 'date']).copy()
    d[col] = d.groupby('ticker')[col].shift(-k)
    return d.dropna(subset=[col]).sort_values(['date', 'ticker']).reset_index(drop=True)


def shift_row(O, RF, rv, name, df, col, tip, rol, start):
    d = df[(df['date'] >= start) & (df['date'] <= LAST)]
    pr = d[['date', 'ticker']].assign(rv_pred=d[col].astype(float) ** 2)
    s = O.sdvig(pr, rv, K=(0,))
    F = d[['date', 'ticker']].assign(ln_sigma2=np.log(d[col].astype(float) ** 2))
    prof = RF.shift_table(F, {'B': rv}, name).iloc[0]
    lev_ok = s['ok']
    dam = None if prof['d_argmax'] is None or prof['d_argmax'] != prof['d_argmax'] else int(prof['d_argmax'])
    if tip == 'model':
        ok = lev_ok
        why = ('по уровню' + (f'; {s["pometka"]}' if s['pometka'] else '')) if lev_ok else f'уровень: {s["prichina"]}'
    elif dam is None:
        ok, why = False, 'приращение: нет пар'
    elif dam > 0:
        ok, why = False, f'заглядывание (+1) по приращению (уровень — окно); уровень argmax {s["argmax"]:+d}'
    elif O.sdvig_ok(dam, (0,))[0]:
        ok, why = True, (f'по приращению (уровень — окно): argmax приращения 0; уровень argmax {s["argmax"]:+d}'
                         + ('' if lev_ok else f' ({s["prichina"]})'))
    else:
        ok, why = False, f'отстаёт (по приращению, уровень — окно): argmax приращения {dam:+d}; уровень argmax {s["argmax"]:+d}'
    return dict(ryad=name, rol=rol, tip=tip, period=f'{start[:4]}–2021', n_tikerov=s['n_tikerov'], n_strok=s['n_strok'],
                **{f'corr_k{k:+d}': s[f'corr_s{k:+d}'] for k in (-2, -1, 0, 1)}, argmax_uroven=s['argmax'], uroven_ok=lev_ok,
                **{f'rho_prir_k{k:+d}': prof[f'drho_{k}'] for k in (-2, -1, 0, 1)}, argmax_prirascheniya=dam,
                ok=bool(ok), reshenie=why, eps=EPS, vremya=now())


def shift_check(ser, nv, f):
    import r2_loyo as O
    import r2_features as RF
    assert abs(float(O.CFG['sdvig']['eps']) - EPS) < 1e-12
    rv = O.rv_daily()                       # features_pit_v3.rv, фильтр date ≤ 2021-12-31 при чтении + assert (r2_loyo)
    rv['date'] = pd.to_datetime(rv['date'])
    check_dev(rv, 'rv_daily')
    series = {}
    for lab, (s, oc) in ser.items():
        series[lab] = (s, oc, 'model', 'ряд разработки')
    series['naive'] = (nv, 'sigma_pred_h1', 'okno', 'ряд разработки')
    for lab, (s, oc, tip, _) in list(series.items()):
        series[f'kontrol:{lab}_sdvig-1'] = (shift_ts(s, oc, -1), oc, tip, 'контроль: σ_d := σ_{d−1} (сдвиг −1)')
        series[f'kontrol:{lab}_utechka+1'] = (shift_ts(s, oc, +1), oc, tip, 'контроль: σ_d := σ_{d+1} (утечка +1)')
    planted = H.naive_from_rv(f.assign(rv=f.groupby('ticker')['rv'].shift(-1)), cut=LAST)
    series['kontrol:naive_utechka_RV_d+1'] = (planted, 'sigma_pred_h1', 'okno', 'контроль: RV_{d+1} в окне (утечка +1)')
    series['kontrol:naive_prezhnyaya_rv_w'] = (naive_old(f), 'sigma_pred_h1', 'okno',
                                               'контроль: прежняя √rv_w (RV d−5…d−1) разработки')
    for lab, (p, oc) in OLD.items():            # справочно: прежние σ разработки главы 4 (в решение не входят)
        series[f'prezhniy:{lab}'] = (read_dev(p, ['date', 'ticker', oc]), oc, 'okno' if lab == 'naive' else 'model',
                                     f'справочно: прежний ряд разработки {p.name}')
    rows = []
    for name, (df, c, tip, rol) in series.items():
        for start in ('2017-01-01', '2018-01-01'):
            rows.append(shift_row(O, RF, rv, name, df, c, tip, rol, start))
            print(f'  сдвиг {name} {start[:4]}: argmax ур. {rows[-1]["argmax_uroven"]}, прир. '
                  f'{rows[-1]["argmax_prirascheniya"]}, ok {rows[-1]["ok"]}', flush=True)
    return pd.DataFrame(rows)


# ======================= 3. детекторы «будущее не влияет» =======================

def structure_rows(raw):
    rows = []
    for lab, (fn, col, rawc, *_r) in INPUTS.items():
        d = raw[lab]
        rel = float(np.max(np.abs(d[col] - d[rawc] * d['F_sm']) / d[col]))
        nf = d.groupby('step')['F_sm'].nunique()
        rows.append(dict(detektor='устройство ряда открытия: ряд = сырой × F_sm, одна F_sm на шаг', ryad=lab, srez='—',
                         strok_do=len(d), max_d_do=rel, max_d_posle=np.nan, podlog_max_d_do=float(nf.max()),
                         ok=bool(rel < 1e-12 and nf.max() == 1),
                         primechanie=f'{fn}:{col}; max отн. |{col} − {rawc}·F_sm| = {rel:.2e}; F_sm на шаг ≤ {int(nf.max())}; '
                                     f'шаги {sorted(d["step"].unique().tolist())}'))
    return rows


def detector_naive(f):
    """RV после среза × U(0,5; 2): наивная ≤ среза не меняется, после — меняется; подложенная утечка RV_{d+1} видна ≤ среза."""
    base = H.naive_from_rv(f, cut=LAST)
    planted = H.naive_from_rv(f.assign(rv=f.groupby('ticker')['rv'].shift(-1).fillna(f['rv'])), cut=LAST)
    rows = []
    for i, cut in enumerate(CUTS):
        c = pd.Timestamp(cut)
        rng = np.random.default_rng(SEED + i)
        fut = (f['date'] > c).values
        f2 = f.copy()
        f2.loc[fut, 'rv'] = f2.loc[fut, 'rv'].values * rng.uniform(0.5, 2.0, int(fut.sum()))
        pert = H.naive_from_rv(f2, cut=LAST)
        plantp = H.naive_from_rv(f2.assign(rv=f2.groupby('ticker')['rv'].shift(-1).fillna(f2['rv'])), cut=LAST)

        def dd(a, b):
            m = a.merge(b, on=['date', 'ticker'], suffixes=('', '_p'))
            x = np.abs(np.log(m['sigma_pred_h1'].astype(float)) - np.log(m['sigma_pred_h1_p'].astype(float)))
            pre = (m['date'] <= c).values
            return float(x[pre].max()), float(x[~pre].max()), int(pre.sum())
        pre, post, n = dd(base, pert)
        ppre, _, _ = dd(planted, plantp)
        rows.append(dict(detektor='наивная: пересборка из RV, RV после среза × U(0,5; 2)', ryad='naive', srez=cut, strok_do=n,
                         max_d_do=pre, max_d_posle=post, podlog_max_d_do=ppre, ok=bool(pre == 0.0 and post > 0 and ppre > 0),
                         primechanie='max |Δ ln σ|'))
    return rows


def features_ch4(s, W_list=WINDOWS):
    """Признаки главы 4 по строкам тикера: ранг по прошлому окну (fastbt.s9_stats.past_rank, minp) и σ / медиана σ за 22 прошлых дня."""
    from fastbt.s9_stats import minp, past_rank
    out = {}
    for W in W_list:
        out[f'rank_W{W}'] = past_rank(s.astype(np.float64).copy(), W, minp(W))
    # медиана σ дней t−22…t−1 (сдвиг на 1, как s9_stats.past_median; там minp ≥ 50 > 22, поэтому min_periods = 22)
    med = pd.Series(s.astype(np.float64)).shift(1).rolling(22, min_periods=22).median().values
    out['sigma_med22'] = s / med
    return out


def detector_ch4(ser, nv):
    """σ после среза × U(0,5; 2): признаки главы 4 на строках ≤ среза не меняются, после — меняются; подложенная утечка σ_d := σ_{d+1}
    видна на строках ≤ среза."""
    allser = dict(ser)
    allser['naive'] = (nv, 'sigma_pred_h1')
    rows = []
    for lab, (s, oc) in allser.items():
        base = s.sort_values(['ticker', 'date']).reset_index(drop=True)
        pl = H.plant_next(base, oc).sort_values(['ticker', 'date']).reset_index(drop=True)
        for i, cut in enumerate(CUTS):
            c = pd.Timestamp(cut)
            p = H.perturb(base, oc, cut, SEED + i).sort_values(['ticker', 'date']).reset_index(drop=True)
            plp = H.plant_next(p, oc).sort_values(['ticker', 'date']).reset_index(drop=True)
            acc = {}
            for tk in base['ticker'].unique():
                m = (base['ticker'] == tk).values
                dts = base.loc[m, 'date'].values
                fb, fp = features_ch4(base.loc[m, oc].values), features_ch4(p.loc[m, oc].values)
                fl, flp = features_ch4(pl.loc[m, oc].values), features_ch4(plp.loc[m, oc].values)
                pre = dts <= np.datetime64(c)
                for k in fb:
                    a = acc.setdefault(k, [0.0, 0.0, 0.0, 0])
                    d1 = np.abs(fb[k] - fp[k])
                    d2 = np.abs(fl[k] - flp[k])
                    a[0] = max(a[0], float(np.nanmax(d1[pre])) if pre.any() else 0.0)
                    a[1] = max(a[1], float(np.nanmax(d1[~pre])) if (~pre).any() else 0.0)
                    a[2] = max(a[2], float(np.nanmax(d2[pre])) if pre.any() else 0.0)
                    a[3] += int(pre.sum())
            for k, (dpre, dpost, dpl, n) in acc.items():
                rows.append(dict(detektor=f'вход главы 4: {k} (σ после среза × U(0,5; 2))', ryad=lab, srez=cut, strok_do=n,
                                 max_d_do=dpre, max_d_posle=dpost, podlog_max_d_do=dpl,
                                 ok=bool(dpre == 0.0 and dpost > 0 and dpl > 0),
                                 primechanie='подлог виден' if dpl > 0 else 'подлог ≤ среза не виден'))
    return rows


# ======================= 4. покрытие =======================

def coverage(ser, nv, f):
    from fastbt.s9_stats import minp
    allser = dict(ser)
    allser['naive'] = (nv, 'sigma_pred_h1')
    rows = []
    for lab, (s, oc) in allser.items():
        for tk, g in s.sort_values('date').groupby('ticker'):
            dts = g['date'].reset_index(drop=True)
            n17 = int((dts < '2018-01-01').sum())
            r = dict(ryad=lab, ticker=tk, pervaya=dts.iloc[0].date(), poslednyaya=dts.iloc[-1].date(), n_strok=len(dts),
                     n_do_2018=n17, n_2018_2021=int((dts >= '2018-01-01').sum()), nan=int(g[oc].isna().sum()))
            for W in WINDOWS:
                mp = minp(W)
                r[f'W{W}_minp'] = mp
                r[f'W{W}_pervyy_rang'] = dts.iloc[mp].date() if len(dts) > mp else None
                r[f'W{W}_polnoe_okno'] = dts.iloc[W].date() if len(dts) > W else None
                r[f'W{W}_okno_na_1y_bar_2018'] = min(n17, W)
            rows.append(r)
    rvf = f.dropna(subset=['rv']).groupby('ticker')['date'].agg(['min', 'count']).reset_index()
    for _, x in rvf.iterrows():
        n17 = int(((f['ticker'] == x['ticker']) & f['rv'].notna() & (f['date'] < '2018-01-01')).sum())
        rows.append(dict(ryad='RV features_pit_v3 (история для наивной)', ticker=x['ticker'], pervaya=x['min'].date(),
                         poslednyaya=f.loc[f['ticker'] == x['ticker'], 'date'].max().date(), n_strok=int(x['count']),
                         n_do_2018=n17))
    return pd.DataFrame(rows)


# ======================= 5. сверка с прежними σ разработки =======================

def sverka(ser, nv):
    allser = dict(ser)
    allser['naive'] = (nv, 'sigma_pred_h1')
    rows = []
    for lab, (p, oc) in OLD.items():
        old = read_dev(p, ['date', 'ticker', oc])      # фильтр ≤ 2021 — первой операцией
        new, nc = allser[lab]
        per = {L: dict(pe=[], sp=[], dln=[], eq=[], n=0) for L in (-2, -1, 0, 1, 2)}
        for tk, g in new.sort_values('date').groupby('ticker'):
            g = g.reset_index(drop=True)
            ln_new = np.log(g[nc].astype(float).values)
            o = old[old['ticker'] == tk].set_index('date')[oc]
            pos = pd.Series(g.index.values, index=g['date']).reindex(o.index).values
            m = ~np.isnan(pos)
            pos = pos[m].astype(int)
            ln_old = np.log(o.values[m].astype(float))
            for L in per:
                j = pos + L                         # прежний σ дня d против нового σ строки d+L тикера
                ok = (j >= 0) & (j < len(g))
                a, b = ln_old[ok], ln_new[j[ok]]
                if len(a) > 10:
                    per[L]['pe'].append(np.corrcoef(a, b)[0, 1])
                    per[L]['sp'].append(pd.Series(a).corr(pd.Series(b), method='spearman'))
                    per[L]['dln'].append(np.mean(a - b))
                    per[L]['eq'].append(np.mean(np.abs(a - b) < 1e-6))
                    per[L]['n'] += len(a)
        best = max(per, key=lambda L: np.median(per[L]['pe']))
        for L, v in per.items():
            rows.append(dict(ryad=lab, prezhniy=str(p.relative_to(MAIN)), prezhniy_sha=sha(p), lag_L=L,
                             smysl='прежний σ_d против нового σ_{d+L}', n_par=v['n'], n_tikerov=len(v['pe']),
                             pearson_ln_med=float(np.median(v['pe'])), spearman_med=float(np.median(v['sp'])),
                             sredniy_dln_old_minus_new=float(np.mean(v['dln'])), dolya_sovpadeniy=float(np.mean(v['eq'])),
                             luchshiy_lag=best))
    return pd.DataFrame(rows)


# ======================= main =======================

def main():
    t0 = now()
    OUT.mkdir(parents=True, exist_ok=True)
    got, spec_sha = verify_inputs()                 # SHA до чтения
    raw, ser = build_model_series()
    f = rv_frame_dev()
    nv = H.naive_from_rv(f, cut=LAST)
    check_dev(nv, 'naive')
    print(f'выборка ≤ 2021: ' + ', '.join(f'{k} {len(s)} строк' for k, (s, _) in ser.items()) + f', naive {len(nv)}', flush=True)

    sh = shift_check(ser, nv, f)
    sh.to_csv(OUT / 'proverka_sdvig.csv', index=False)
    det = pd.DataFrame(structure_rows(raw) + detector_naive(f) + detector_ch4(ser, nv))
    det.to_csv(OUT / 'proverka_detektor.csv', index=False)
    cov = coverage(ser, nv, f)
    cov.to_csv(OUT / 'pokrytie.csv', index=False)
    sv = sverka(ser, nv)
    sv.to_csv(OUT / 'sverka_prezhnie.csv', index=False)

    kontr = sh['ryad'].str.startswith('kontrol:')
    work = ~kontr & ~sh['ryad'].str.startswith('prezhniy:')
    ok_by = sh[work].groupby('ryad')['ok'].all().to_dict()
    ctrl_block = bool((~sh.loc[kontr, 'ok']).all())
    det_ok = bool(det['ok'].all())
    bad = [k for k, v in ok_by.items() if not v]

    # выход: только ряды, прошедшие проверку сдвига
    written = {}
    for lab, (s, oc) in ser.items():
        if ok_by.get(lab):
            p = OUT / INPUTS[lab][4]
            check_dev(s, lab)
            s.to_parquet(p, index=False)
            written[lab] = p
    if ok_by.get('naive'):
        p = OUT / NAIVE_OUT
        nv.to_parquet(p, index=False)
        written['naive'] = p

    lines = [f'# ch4_dev_sigma.py, {now()} (старт {t0}); входы — до чтения сверены с r2_A_sborka.yaml: sobrano',
             f'{spec_sha}  models/designs/r2/r2_A_sborka.yaml  (спецификация)',
             f'{sha(HERE / "ch4_dev_sigma.py")}  scripts/analysis/ch4_dev_sigma.py  (код)',
             f'{sha(HERE / "h2_d0_ch4.py")}  scripts/analysis/h2_d0_ch4.py  (импорт функций)']
    lines += [f'{v}  test_locked/R2/final/{k}  (вход, прочитано ≤ 2021)' for k, v in got.items()]
    lines.append(f'{sha(REGEN / "features_pit_v3.parquet")}  data/regen/features_pit_v3.parquet  (вход RV, прочитано ≤ 2021)')
    lines += [f'{sha(p)}  data/regen/ch4_dev/{p.name}  (выход)' for p in written.values()]
    for n in ('proverka_sdvig.csv', 'proverka_detektor.csv', 'pokrytie.csv', 'sverka_prezhnie.csv'):
        lines.append(f'{sha(OUT / n)}  data/regen/ch4_dev/{n}  (проверка)')
    (OUT / 'SHA256.txt').write_text('\n'.join(lines) + '\n')
    res = dict(time=now(), sdvig_ok=ok_by, kontroli_blok=ctrl_block, detektor_ok=det_ok,
               detektor_ne_ok=det.loc[~det['ok'], ['detektor', 'ryad', 'srez']].astype(str).values.tolist(),
               vyhod={k: dict(file=str(p), sha256=sha(p)) for k, p in written.items()},
               rows={k: len(s) for k, (s, _) in ser.items()} | {'naive': len(nv)})
    (OUT / 'ch4_dev_sigma_run.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if bad:
        print(f'СТОП: ряды со сдвигом {bad} — в расчёты не допускаются (не записаны)')
        return 2
    if not ctrl_block or not det_ok:
        print('СТОП: контроли не блокируются или детектор не пройден')
        return 3
    print('ПРОВЕРКА СДВИГА И ДЕТЕКТОР: OK')
    return 0


if __name__ == '__main__':
    os.environ.setdefault('NUMBA_CACHE_DIR', str(MAIN / 'data' / 'fastbt' / 'numba_cache'))
    sys.exit(main())
