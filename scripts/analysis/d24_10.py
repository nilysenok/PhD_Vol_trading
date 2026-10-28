#!/usr/bin/env python3
"""Д24.10 (26.10, ночь): история 10-минутных баров 17 тикеров до 2014-06 и обучение «с начала истории ISS».
Бары — крейт moex_iss (scripts/analysis/d24_bars.sh, команда hist-bars): TQBR с 2011-12-01 по 2014-05-31 (до 2013-09 ISS отдаёт под TQBR
историю EQBR), для VTBR и PHOR до 2013-02-28 — режим EQNL; плюс бары ISS 2014-06-02…2014-06-06 из data/iss_check (до первого дня хранимых свечей).
Правило отрезка (ночное умолчание): берётся непрерывный отрезок, примыкающий к 2014-06-09, без разрывов дольше 30 календарных дней
(OGKB: торгов нет с 28.05.2013 по 08.06.2014 — история не берётся; AFLT: разрыв 07.05–31.10.2013 — с 31.10.2013).
Стадии:
  rv      — RV/BV/J/RSV± окна [09:50; 18:50) без баров карантина (функция пайплайна detect_quarantine_flags на ряду отрезка, формула —
            порт rv.rs::daily из d121_rv28) для дней < 2014-06-09 + хранимый rv_1850_noquar с 2014-06-09 -> data/regen/D24/rv_1850_noquar_hist.parquet;
            свечи для калькулятора собственных признаков: отрезок без карантина + хранимые candles_10m -> data/regen/D24/candles_hist/;
  matrix  — та же функция m0_pit_lib.assemble(fill='asof') на расширенных ключах (дни калькулятора до 2014-06-09 + ключи v3); внешние ряды —
            as-of по общему календарю (с 26.08.2014; раньше — NaN, как в v3). Строки годов >= 2016 (val/test всех шагов) берутся из v3 побайтно,
            пересобраны только строки обучающих окон (годы <= 2015) -> data/regen/features_pit_v3h.parquet (+ _d7_h5 для h=5);
  leak    — «будущее не влияет»: свечи и RV после t -> шум, пересборка собственных признаков, строки <= t побайтно (+ самопроверка);
  boost / gibrid / eval / yaml — как d24_8.py, кандидаты D24_10_h1 (Д8.12, точки B v2b) и D24_10_h5 (D12_h5, точки D12_h5).
python3 scripts/analysis/d24_10.py --stage rv|matrix|leak|boost|gibrid|eval|yaml [--h 1 5] [--workers 2]"""
import os, sys, json, time, argparse, shutil
from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')
R = DATA / 'regen'
FEATH = R / 'features_pit_v3h.parquet'
if len(sys.argv) > 2 and sys.argv[2] in ('boost', 'gibrid', 'eval', 'score', 'yaml'):
    os.environ['L2_FEAT'] = str(FEATH)  # матрица обучения D24.10 (load_base / блок Д7 — {FEAT.stem}_d7_h5)
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd

D24 = R / 'D24'
HIST = DATA / 'iss_hist' / 'bars_10m_full'
ISS = DATA / 'iss_check' / 'bars_10m_full'
CAND = D24 / 'candles_hist'
RVH = D24 / 'rv_1850_noquar_hist.parquet'
FEAT3 = R / 'features_pit_v3.parquet'
T17 = ["SBER", "VTBR", "MOEX", "LKOH", "TATN", "NVTK", "ALRS", "MTLR", "HYDR", "IRAO", "OGKB", "RTKM", "MTSS", "MGNT", "AFLT", "PHOR", "LSRG"]
FIRST = pd.Timestamp('2014-06-09')  # первый день хранимых свечей и матрицы v3
SPLICE = 2016  # строки годов >= SPLICE — из v3 без изменений (val-год шага 1 = 2016)
GAP = 30
STORED = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/moex_discovery/data/dataset_final/01_stocks/candles_10m')
TAG = 'D24_10'


def log(*a):
    print(time.strftime('%T'), *a, flush=True)


# ---------------------------------------------------------------- бары истории и RV
def hist_bars(t):
    """Отрезок истории тикера до 2014-06-09 (timestamp, open, high, low, close, volume, value), правило разрыва GAP."""
    parts = [pd.read_parquet(HIST / f'{t}.parquet')]
    if (HIST / f'{t}_EQNL.parquet').exists():
        e = pd.read_parquet(HIST / f'{t}_EQNL.parquet')
        parts.insert(0, e[e.begin < parts[0].begin.min()])
    i = pd.read_parquet(ISS / f'{t}.parquet')
    parts.append(i[(i.begin > parts[-1].begin.max()) & (i.begin < FIRST)])
    b = pd.concat(parts, ignore_index=True).rename(columns={'begin': 'timestamp'})
    b['timestamp'] = pd.to_datetime(b['timestamp'])
    b = b.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    days = pd.Series(sorted(b.timestamp.dt.normalize().unique()))
    gaps = pd.concat([days, pd.Series([FIRST])], ignore_index=True).diff().dt.days
    cut = days[gaps.iloc[:-1][gaps.iloc[:-1] > GAP].index].max() if (gaps > GAP).any() else days.iloc[0]
    if gaps.iloc[-1] > GAP:  # разрыв перед самим 2014-06-09 — история не берётся
        return b.iloc[:0], dict(ticker=t, first_iss=str(days.iloc[0].date()), last_iss=str(days.iloc[-1].date()), used_from=None, days=0)
    b = b[b.timestamp >= cut].reset_index(drop=True)
    return b, dict(ticker=t, first_iss=str(days.iloc[0].date()), last_iss=str(days.iloc[-1].date()), used_from=str(pd.Timestamp(cut).date()),
                   days=int(b.timestamp.dt.normalize().nunique()))


def stage_rv(a):
    import d121_rv28 as RV
    import m0_pit_lib as M
    CAND.mkdir(parents=True, exist_ok=True)
    old = pd.read_parquet(M.RV1850)
    old['date'] = pd.to_datetime(old['date'])
    rows, info = [], []
    for t in T17:
        b, inf = hist_bars(t)
        st = pd.read_parquet(STORED / f'{t}.parquet')
        st['timestamp'] = pd.to_datetime(st['timestamp'])
        if len(b):
            q = RV.quarantine_ts(b[['timestamp', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True))
            ts = b.timestamp.values.astype('datetime64[ms]').astype(np.int64)
            isq = np.isin(ts, np.fromiter(q, np.int64, len(q))) if q else np.zeros(len(b), bool)
            w = RV.in_window(b.timestamp).to_numpy()
            d = RV.daily(b[w & ~isq])
            nq = pd.Series(isq[w]).groupby(b.timestamp[w].dt.normalize().values).sum()
            d['n_quar'] = nq.reindex(d.index).fillna(0).astype(int)
            d = d.reset_index().rename(columns={'index': 'date'})
            d.insert(1, 'ticker', t)
            rows.append(d[d.date < FIRST])
            clean = b[~isq][['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            inf.update(quar_bars=int(isq.sum()), quar_share=round(float(isq.mean()), 5), rv_days=int((d.date < FIRST).sum()))
        else:
            clean = st.iloc[:0]
            inf.update(quar_bars=0, quar_share=0.0, rv_days=0)
        c = pd.concat([clean[clean.timestamp < st.timestamp.min()], st[['timestamp', 'open', 'high', 'low', 'close', 'volume']]], ignore_index=True)
        c['volume'] = c['volume'].astype('int64')
        c.to_parquet(CAND / f'{t}.parquet', index=False)
        info.append(inf)
        log(inf)
    new = pd.concat(rows, ignore_index=True)
    keep = [c for c in old.columns]
    new = new[[c for c in keep if c in new.columns]]
    out = pd.concat([new, old[old.date >= FIRST]], ignore_index=True).sort_values(['ticker', 'date']).reset_index(drop=True)
    out.to_parquet(RVH, index=False)
    I = pd.DataFrame(info)
    I.to_csv(D24 / 'hist_tickers.csv', index=False)
    log('записан', RVH, len(out), 'строк; новых до 2014-06-09:', len(new))
    print(I.to_string())


# ---------------------------------------------------------------- матрица
def build_ext(cands_dir=CAND, rvh=None, log_=print):
    """Расширенная матрица той же функцией m0_pit_lib.assemble (fill='asof'); ключи до 2014-06-09 — дни калькулятора собственных признаков."""
    import m0_pit_lib as M
    from m0_common import calendar, load_prepared
    M.CANDLES = cands_dir  # калькулятор собственных признаков читает свечи отрезка истории + хранимые
    r18 = pd.read_parquet(RVH) if rvh is None else rvh.copy()
    r18['date'] = pd.to_datetime(r18['date'])
    prep = load_prepared()
    prep['date'] = pd.to_datetime(prep['date'])
    own, inf = M.own_features(sorted(prep.ticker.unique()), r18)
    log_(f'собственные (для ключей): {inf}')
    first = prep.groupby('ticker').date.min()
    extra = own[own.date < own.ticker.map(first)][['date', 'ticker']]
    pre = pd.DataFrame({c: pd.Series(dtype=prep[c].dtype) for c in prep.columns}).reindex(range(len(extra)))
    pre['date'], pre['ticker'] = extra.date.values, extra.ticker.values
    prep2 = pd.concat([pre, prep], ignore_index=True)
    out, v2, NEWF, DAYS, lost, pre_nan = M.assemble(calendar(), prep2, r18, fill='asof', log=log_)
    return out, NEWF, len(extra)


def splice(ext, v3, key=('date', 'ticker')):
    """Годы < SPLICE — из пересборки, годы >= SPLICE — строки v3 как есть; столбцы и порядок — как в v3."""
    e = ext[pd.to_datetime(ext.date).dt.year < SPLICE][list(v3.columns)]
    v = v3[pd.to_datetime(v3.date).dt.year >= SPLICE]
    return pd.concat([e, v], ignore_index=True)


def diff_table(ext, v3, cols):
    """Сверка пересборки с v3 на общих ключах по годам: доля строк, где хоть один признак не совпал побайтно."""
    m = v3[['date', 'ticker']].merge(ext[['date', 'ticker']].assign(_e=np.arange(len(ext))), on=['date', 'ticker'], how='left')
    assert m._e.notna().all()
    E = ext.iloc[m._e.astype(int).values][cols].reset_index(drop=True)
    V = v3[cols].reset_index(drop=True)
    a, b = E.values.astype('float64'), V.values.astype('float64')
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    yr = pd.to_datetime(v3.date).dt.year.values
    rows = []
    for y in sorted(set(yr)):
        s = same[yr == y]
        bad = [cols[j] for j in np.where(~s.all(0))[0]]
        rows.append(dict(year=y, rows=int(s.shape[0]), rows_differ=int((~s.all(1)).sum()), cols_differ=len(bad), examples=','.join(bad[:8])))
    return pd.DataFrame(rows)


def stage_matrix(a):
    import d7_block as D7B
    v3 = pd.read_parquet(FEAT3)
    v3['date'] = pd.to_datetime(v3['date'])
    ext, NEWF, n_extra = build_ext()
    ext['date'] = pd.to_datetime(ext['date'])
    cfg = json.load(open(R / 'features_pit_v3.config.json'))
    tgt = ['rv_target_h1', 'rv_target_h5', 'rv_target_h22', 'rv_target_mean_h1', 'rv_target_mean_h5', 'rv_target_mean_h22']
    dt = diff_table(ext, v3, cfg['feature_cols'] + tgt)
    dt.to_csv(D24 / 'v3h_vs_v3.csv', index=False)
    print('пересборка против v3 на общих ключах:\n', dt.to_string())
    M = splice(ext, v3)
    M = M.sort_values(['date', 'ticker'], kind='stable').reset_index(drop=True)
    M.to_parquet(FEATH, index=False)
    shutil.copy(R / 'features_pit_v3.config.json', FEATH.with_suffix('.config.json'))
    yv = M[M.date.dt.year >= SPLICE].sort_values(['date', 'ticker']).reset_index(drop=True)
    y3 = v3[v3.date.dt.year >= SPLICE].sort_values(['date', 'ticker']).reset_index(drop=True)
    eq = all(yv[c].equals(y3[c]) for c in v3.columns)
    log(f'матрица {FEATH.name}: строк {len(M)} (новых до 2014-06-09: {n_extra}, v3: {len(v3)}); строки >= {SPLICE} равны v3: {eq}')
    assert eq
    # блок Д7 (h=5): та же функция на расширенной матрице; годы >= SPLICE — из features_pit_v3_d7_h5 как есть
    d7v3 = pd.read_parquet(R / 'features_pit_v3_d7_h5.parquet')
    d7v3['date'] = pd.to_datetime(d7v3['date'])
    ext_full = pd.concat([ext[ext.date.dt.year < SPLICE][list(v3.columns)], v3[v3.date.dt.year >= SPLICE]], ignore_index=True)
    d7e = D7B.assemble(ext_full.sort_values(['date', 'ticker'], kind='stable').reset_index(drop=True), 5)
    d7e['date'] = pd.to_datetime(d7e['date'])
    dd = diff_table(d7e, d7v3, [c for c in d7v3.columns if c.startswith('d7_')])
    dd.to_csv(D24 / 'v3h_d7_vs_v3_d7.csv', index=False)
    print('Д7 пересборка против v3_d7:\n', dd.to_string())
    M7 = splice(d7e, d7v3).sort_values(['date', 'ticker'], kind='stable').reset_index(drop=True)
    M7.to_parquet(R / 'features_pit_v3h_d7_h5.parquet', index=False)
    z = M7[M7.date.dt.year >= SPLICE].sort_values(['date', 'ticker']).reset_index(drop=True)
    z3 = d7v3[d7v3.date.dt.year >= SPLICE].sort_values(['date', 'ticker']).reset_index(drop=True)
    eq7 = all(z[c].equals(z3[c]) for c in d7v3.columns)
    log(f'features_pit_v3h_d7_h5: строк {len(M7)}; строки >= {SPLICE} равны v3_d7: {eq7}')
    assert eq7
    for p in (FEATH, R / 'features_pit_v3h_d7_h5.parquet'):
        log(p.name, sha(p))


def sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


# ---------------------------------------------------------------- детектор
def stage_leak(a):
    """Свечи (OHLCV, begin > t) и RV (date > t) -> шум; собственные признаки пересобираются тем же калькулятором; строки <= t побайтно.
    Самопроверка: rv_d = RV следующей строки тикера (утечка) обязана дать расхождение до t."""
    import m0_pit_lib as M
    rng = np.random.default_rng(20261026)
    r18 = pd.read_parquet(RVH)
    r18['date'] = pd.to_datetime(r18['date'])
    tick = T17
    M.CANDLES = CAND
    base, _ = M.own_features(tick, r18)
    res = []
    tmp = D24 / 'candles_leak'
    for t in ('2013-03-31', '2014-03-31'):
        t = pd.Timestamp(t)
        tmp.mkdir(exist_ok=True)
        for tk in tick:
            c = pd.read_parquet(CAND / f'{tk}.parquet')
            f = c.timestamp > t + pd.Timedelta(hours=23, minutes=59)
            for col in ('open', 'high', 'low', 'close'):
                c.loc[f, col] = c.loc[f, col].values * rng.uniform(0.9, 1.1, f.sum())
            c.loc[f, 'volume'] = (c.loc[f, 'volume'].values * rng.uniform(0.2, 5, f.sum())).astype('int64')
            c.to_parquet(tmp / f'{tk}.parquet', index=False)
        B = r18.copy()
        f = B.date > t
        for col in ('rv', 'bv', 'j', 'rsv_pos', 'rsv_neg'):
            B.loc[f, col] = B.loc[f, col].values * rng.uniform(0.2, 5, f.sum())
        M.CANDLES = tmp
        G, _ = M.own_features(tick, B)
        M.CANDLES = CAND
        own = [c for c in M.OWN if c in base]
        for leak in (False, True):
            X, Y = base.copy(), G.copy()
            if leak:
                for Z in (X, Y):
                    Z.sort_values(['ticker', 'date'], inplace=True)
                    Z['rv_d'] = Z.groupby('ticker')['rv'].shift(-1)
            k = X[['date', 'ticker']].merge(Y[['date', 'ticker']].assign(_i=np.arange(len(Y))), on=['date', 'ticker'])
            past = (k.date <= t).values
            Xa = X.set_index(['date', 'ticker']).loc[list(zip(k.date[past], k.ticker[past]))]
            Ya = Y.iloc[k._i[past].values].set_index(['date', 'ticker'])
            for c in own:
                x, y = Xa[c].values.astype('float64'), Ya[c].values.astype('float64')
                res.append(dict(t=str(t.date()), leak=leak, feature=c, rows=int(past.sum()),
                                mismatch=int((~((x == y) | (np.isnan(x) & np.isnan(y)))).sum())))
    shutil.rmtree(tmp, ignore_errors=True)
    Lk = pd.DataFrame(res)
    Lk.to_csv(D24 / 'leak_hist_own.csv', index=False)
    ok = Lk[~Lk.leak].mismatch.sum() == 0
    caught = (Lk[Lk.leak & (Lk.feature == 'rv_d')].groupby('t').mismatch.sum() > 0).all()
    log(f'детектор истории: признаков {Lk.feature.nunique()}, расхождений до t {int(Lk[~Lk.leak].mismatch.sum())} -> {"PASS" if ok else "FAIL"}; '
        f'самопроверка rv_d = RV(t+1): {"PASS" if caught else "FAIL"}')
    assert ok and caught


# ---------------------------------------------------------------- бустинг и оценка (общие функции d24_8)
def stage_boost(a):
    import l2_lib as L
    import d24_8 as E
    assert L.FEAT == FEATH, L.FEAT
    E.FEAT3 = FEATH  # матрица обучения h=1; h=5 — {FEAT.stem}_d7_h5 = features_pit_v3h_d7_h5
    for h in a.h:
        t0 = time.time()
        A, cols, cfg = E.matrix(h, ext=False)
        log(f'h={h}: матрица {L.FEAT.name}, признаков {len(cols)}, строк {len(A)}, годы {int(A.year.min())}-{int(A.year.max())}')
        res = E.train(h, A, cols, cfg, a.workers)
        boost = f'{TAG}_h{h}'
        E.write_legs(boost, h, res, R / 'test_locked' / boost)
        pd.DataFrame([dict(model=r[0], step=r[1], point=r[2], seed=r[3], rank=r[4], trees=r[5]) for r in res]).to_csv(D24 / f'{boost}_trees.csv', index=False)
        log(f'h={h}: обучено {len(res)} моделей за {time.time() - t0:.0f} с')


def stage_gibrid(a):
    import d24_8 as E
    for h in a.h:
        E.run_gibrid(f'{TAG}_h{h}', h)


def stage_eval(a):
    import d24_8 as E
    E.stage_eval(a, tag=TAG, label='Д24.10')


YAML = {1: dict(semeystvo='Д8.12: прогноз HAR-J как признак', osnova='D12_h1_A_d812',
                opisanie='Д8.12 (точки ЛГК B v2b) с обучающими окнами от начала 10-мин истории ISS (2011-12 / 2012-10 / 2013-02..04 по тикерам); val/test-годы без изменений'),
        5: dict(semeystvo='V9 на HAR-J', osnova='D12_h5',
                opisanie='D12_h5 (V9 + Д8.12 + Д7 + Huber, точки D12_h5) с обучающими окнами от начала 10-мин истории ISS; val/test-годы без изменений; нога wrap2 без 1a')}


def stage_yaml(a):
    import d24_8 as E
    Ev = pd.read_csv(D24 / f'eval_{TAG}.csv')
    for h in a.h:
        e = Ev[(Ev.h == h) & (Ev.kind == 'лучший')].iloc[0]
        if e.proshel:
            E.write_yaml(f'{TAG}_h{h}', h, 'Д24.10', YAML[h], E.LEGS[h], f'data/regen/test_locked/{TAG}_h{h}', 1)
        else:
            log(f'h={h}: не прошёл — yaml не пишется')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['rv', 'matrix', 'leak', 'boost', 'gibrid', 'eval', 'score', 'yaml'])
    ap.add_argument('--h', type=int, nargs='+', default=[1, 5])
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    D24.mkdir(parents=True, exist_ok=True)
    globals()[f'stage_{"eval" if a.stage == "score" else a.stage}'](a)


if __name__ == '__main__':
    main()
