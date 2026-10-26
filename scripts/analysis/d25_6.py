#!/usr/bin/env python3
"""Д25.6 (26.10, ночь): наличие 1-минутных свечей ISS (interval=1) для 17 тикеров и оценка объёма выгрузки.

Только GET на iss.moex.com, последовательно, пауза PAUSE между запросами.
  python3 scripts/analysis/d25_6.py borders   -> candleborders interval 1 и 10 по 17 тикерам
  python3 scripts/analysis/d25_6.py sample    -> выборочные дни: число 1-мин баров, страниц, время ответа
Выход: data/regen/D25/iss_1min_borders.csv, iss_1min_sample.csv (абсолютный путь общего data/).
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')
OUT = DATA / 'regen' / 'D25'
TICKERS = ['SBER', 'LKOH', 'TATN', 'NVTK', 'MTLR', 'HYDR', 'IRAO', 'MTSS', 'MGNT', 'LSRG',
           'RTKM', 'VTBR', 'PHOR', 'ALRS', 'MOEX', 'AFLT', 'OGKB']
BASE = 'https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{t}'
PAUSE = 0.5
# выборка: ранний обучающий год, середина, конец обучения шага 6, ликвидные и неликвидные бумаги
SAMPLE = [('SBER', '2014-06-10'), ('SBER', '2018-03-15'), ('LSRG', '2015-03-02'), ('OGKB', '2016-06-01'),
          ('MOEX', '2020-09-15'), ('AFLT', '2014-11-12'), ('PHOR', '2017-05-17'), ('MTLR', '2019-12-03')]


def get(url):
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=30) as r:
        d = json.loads(r.read().decode('utf-8'))
    time.sleep(PAUSE)
    return d, time.time() - t0


def borders():
    rows = []
    for t in TICKERS:
        d, _ = get(BASE.format(t=t) + '/candleborders.json?iss.meta=off')
        b = d['borders']
        for r in b['data']:
            rec = dict(zip(b['columns'], r))
            if rec['interval'] in (1, 10):
                rows.append({'ticker': t, 'interval': rec['interval'], 'begin': rec['begin'], 'end': rec['end']})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / 'iss_1min_borders.csv', index=False)
    print(df.to_string(index=False))


def sample():
    rows = []
    for t, day in SAMPLE:
        bars, pages, sec, start = [], 0, 0.0, 0
        while True:
            url = BASE.format(t=t) + f'/candles.json?interval=1&from={day}&till={day}&start={start}&iss.meta=off'
            d, dt = get(url)
            c = d['candles']
            bars += [dict(zip(c['columns'], r)) for r in c['data']]
            pages += 1
            sec += dt
            if len(c['data']) < 500:
                break
            start += 500
        ins = [b for b in bars if '09:50:00' <= b['begin'][11:19] < '18:50:00']
        rows.append({'ticker': t, 'day': day, 'bars_1m': len(bars), 'bars_1m_in_window': len(ins),
                     'first': bars[0]['begin'] if bars else None, 'last': bars[-1]['begin'] if bars else None,
                     'pages': pages, 'sec_with_pause': round(sec, 2)})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'iss_1min_sample.csv', index=False)
    # оценка объёма: 17 тикеров x обучающие дни 2014-06..2020-12 (шаг 6 обучается до 2020 включительно)
    days = 17 * 1650
    bpd = df['bars_1m'].mean()
    pages = days * max(1.0, bpd / 500.0)
    per_page = df['sec_with_pause'].sum() / df['pages'].sum()
    print(f'оценка: {days} тикер-дней, ~{bpd:.0f} баров/день, ~{pages:.0f} страниц по 500 '
          f'(запрос с from..till на месяц), {per_page:.2f} с/страницу с паузой -> '
          f'{pages * per_page / 3600:.1f} ч последовательно, {pages * per_page / 3600 / 4:.1f} ч в 4 потока')


BARS = OUT / 'bars1m'
YEARS = list(range(2014, 2025))  # 2014-05 … 2024-12: обучение шагов 1-9 (годы < val-года шага); 2014-05 — разгон окон
COLS = ['begin', 'open', 'high', 'low', 'close', 'volume', 'value']
DL_PAUSE = float(os.environ.get('D25_PAUSE', '0.2'))  # пауза после каждого GET в каждом из 4 потоков (у крейта moex_iss — 0,1 с при 4 потоках); с 05:23 — 0,1


def _get_retry(url, tries=6):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                d = json.loads(r.read().decode('utf-8'))
            time.sleep(DL_PAUSE)
            return d
        except Exception as e:  # обрыв/429/5xx: пауза с ростом, затем повтор
            time.sleep(min(60, 2 ** k * 2))
            last = e
    raise RuntimeError(f'GET не удался после {tries} попыток: {url}: {last}')


def _chunk(t, y):
    """Один (тикер, год): страницы по 500 подряд; файл .csv пишется целиком и помечается .done (докачка — по отсутствию .done)."""
    f, done = BARS / f'{t}_{y}.csv', BARS / f'{t}_{y}.done'
    if done.exists():
        return t, y, -1, 0
    fr = f'{y}-05-01' if y == 2014 else f'{y}-01-01'
    rows, start, pages = [], 0, 0
    while True:
        d = _get_retry(BASE.format(t=t) + f'/candles.json?interval=1&from={fr}&till={y}-12-31&start={start}&iss.meta=off')
        c = d['candles']
        idx = {k: i for i, k in enumerate(c['columns'])}
        rows += [[r[idx[k]] for k in COLS] for r in c['data']]
        pages += 1
        if len(c['data']) < 500:
            break
        start += 500
    pd.DataFrame(rows, columns=COLS).to_csv(f, index=False)
    done.write_text(json.dumps({'rows': len(rows), 'pages': pages, 'time': time.strftime('%F %T')}))
    return t, y, len(rows), pages


def download():
    """4 потока в одном процессе, только GET; порядок — по годам (сначала обучающие 2014-2020)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    BARS.mkdir(parents=True, exist_ok=True)
    jobs = [(t, y) for y in YEARS for t in TICKERS]
    t0, n = time.time(), 0
    with ThreadPoolExecutor(4) as ex:
        fs = [ex.submit(_chunk, t, y) for t, y in jobs]
        for f in as_completed(fs):
            t, y, k, p = f.result()
            n += 1
            print(f'{time.strftime("%T")} {t} {y}: баров {k}, страниц {p} ({n}/{len(jobs)}, {time.time() - t0:.0f} с)', flush=True)


def check():
    """Полнота: у каждого торгового дня v3 (тикер, дата; RV 10 мин есть) должны быть 1-мин бары в окне [09:50; 18:50); страница < 500 — конец."""
    v3 = pd.read_parquet(DATA / 'regen' / 'features_pit_v3.parquet', columns=['date', 'ticker', 'n_bars'])
    v3['date'] = pd.to_datetime(v3['date'])
    rows = []
    for t in TICKERS:
        fs = sorted(BARS.glob(f'{t}_*.csv'))
        if not fs:
            continue
        b = pd.concat([pd.read_csv(f, usecols=['begin']) for f in fs])
        ts = pd.to_datetime(b['begin'])
        hm = ts.dt.strftime('%H:%M')
        cnt = ts[(hm >= '09:50') & (hm < '18:50')].dt.normalize().value_counts()
        yrs = [int(f.stem.split('_')[1]) for f in fs if (BARS / f'{f.stem}.done').exists()]
        x = v3[(v3.ticker == t) & v3.date.dt.year.isin(yrs)]
        n1 = x['date'].map(cnt).fillna(0)
        rows.append(dict(ticker=t, years=f'{min(yrs)}-{max(yrs)}' if yrs else '', days_v3=len(x), days_no_1m=int((n1 == 0).sum()),
                         days_lt30=int((n1 < 30).sum()), median_bars=float(n1.median()) if len(x) else np.nan,
                         dup_ts=int(b['begin'].duplicated().sum())))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'bars1m_completeness.csv', index=False)
    print(df.to_string(index=False))


# ---------------------------------------------------------------- RV 5 мин с подвыборкой (крейт) и цель
CRATE_BIN = Path('/private/tmp/claude-501/-Users-nilysenok-Desktop-------/3152041c-56a6-4891-987d-fc18e18dc186/scratchpad/target/release/rv5sub_daily')
RV5 = OUT / 'rv5sub_daily.csv'
T5 = OUT / 'rv5sub_targets.parquet'
HS = (1, 5, 22)


def rv():
    import subprocess
    subprocess.run([str(CRATE_BIN), str(BARS), str(RV5)], check=True)
    d = pd.read_csv(RV5, parse_dates=['date'])
    print(d.groupby('ticker').agg(days=('rv', 'size'), first=('date', 'min'), last=('date', 'max'), med_bars=('n_bars', 'median')).to_string())


def leak():
    """Детектор «будущее не влияет»: (1) крейт со срезом — бары с begin >= среза зашумлены, RV дней < среза обязана совпасть побайтно;
    (2) цель: RV5 дней >= T зашумлена — цель строк, у которых окно [d+1, d+h] целиком раньше T, обязана совпасть;
    самопроверка — цель со сдвигом окна на 1 день вперёд (подложенная утечка) обязана разойтись."""
    import subprocess
    base = pd.read_csv(RV5, parse_dates=['date'])
    res = []
    for cut in ('2015-06-30', '2017-03-15', '2019-06-28', '2020-11-02'):
        if base.date.max() < pd.Timestamp(cut):
            continue
        p = OUT / f'_leak_rv5_{cut}.csv'
        subprocess.run([str(CRATE_BIN), str(BARS), str(p), cut], check=True, capture_output=True)
        g = pd.read_csv(p, parse_dates=['date'])
        m = base.merge(g, on=['ticker', 'date'], suffixes=('', '_n'))
        pre, post = m[m.date < cut], m[m.date >= cut]
        res.append(dict(test='крейт: RV до среза', cut=cut, rows=len(pre), mismatch=int((pre.rv != pre.rv_n).sum()),
                        changed_after=int((post.rv != post.rv_n).sum())))
        p.unlink()
    v3 = _v3()
    rng = np.random.default_rng(20261026)
    for T in ('2015-06-30', '2016-06-30', '2019-06-28', '2023-06-30'):
        T = pd.Timestamp(T)
        if base.date.max() < T + pd.Timedelta(days=60):  # срез вне выгруженного периода — шум ничего не меняет, проверка пуста
            continue
        B = base.copy()
        mk = B.date >= T
        B.loc[mk, 'rv'] = B.loc[mk, 'rv'].values * rng.uniform(0.2, 5, mk.sum())
        for lk in (False, True):
            a0, a1 = targets(v3, base, leak=lk), targets(v3, B, leak=lk)
            for h in HS:
                c = f't5_h{h}'
                # строка d «до T»: окно цели [d+1, d+h] (торговые строки тикера) кончается раньше T
                end = v3.groupby('ticker')['date'].shift(-h)
                ok = (end < T).values
                x, y = a0[c].values[ok], a1[c].values[ok]
                res.append(dict(test=f'цель h={h}' + (' [самопроверка: окно +1 день]' if lk else ''), cut=str(T.date()), rows=int(ok.sum()),
                                mismatch=int((~((x == y) | (np.isnan(x) & np.isnan(y)))).sum()), changed_after=int((a0[c].values[~ok] != a1[c].values[~ok]).sum())))
    R_ = pd.DataFrame(res)
    R_.to_csv(OUT / 'leak_rv5sub.csv', index=False)
    print(R_.to_string(index=False))
    okk = (R_[~R_.test.str.contains('самопроверка')].mismatch == 0).all() and (R_[R_.test.str.contains('самопроверка')].mismatch > 0).all()
    print('ДЕТЕКТОР:', 'PASS' if okk else 'FAIL')
    assert okk


def _v3():
    v3 = pd.read_parquet(DATA / 'regen' / 'features_pit_v3.parquet', columns=['date', 'ticker', 'rv', 'rv_target_h1', 'rv_target_mean_h5', 'rv_target_mean_h22'])
    v3['date'] = pd.to_datetime(v3['date'])
    return v3.sort_values(['ticker', 'date'], kind='stable').reset_index(drop=True)


def targets(v3, d5, leak=False):
    """Цель как у v3 (проверено: rv_target_h1 = rv следующей строки тикера; rv_target_mean_h{h} = среднее rv строк d+1..d+h, min h),
    с подстановкой RV5 вместо rv. Строки v3 без RV5 (нет 1-мин баров) -> NaN. leak=True — окно сдвинуто на 1 день вперёд (самопроверка)."""
    x = v3[['date', 'ticker']].merge(d5[['date', 'ticker', 'rv']].rename(columns={'rv': 'rv5'}), on=['date', 'ticker'], how='left')
    g = x.groupby('ticker', sort=False)['rv5']
    out = x[['date', 'ticker']].copy()
    # сумма явных сдвигов (не rolling: у скользящей суммы pandas округление ранних строк зависит от поздних значений — ложная «утечка» ~1e-20)
    for h in HS:
        acc = sum(g.shift(-k - int(leak)) for k in range(1, h + 1))
        out[f't5_h{h}'] = acc / h
    return out


def target():
    import hashlib
    v3 = _v3()
    d5 = pd.read_csv(RV5, parse_dates=['date'])
    # проверка формулы на 10-мин rv: та же функция должна воспроизвести цели v3 побайтно
    chk = targets(v3, v3[['date', 'ticker', 'rv']])
    for h, c in ((1, 'rv_target_h1'), (5, 'rv_target_mean_h5'), (22, 'rv_target_mean_h22')):
        dd = np.nanmax(np.abs(chk[f't5_h{h}'].values - v3[c].values))
        print(f'формула цели h={h} на 10-мин rv: max |Δ| = {dd:.2e}')
        assert dd < 1e-15
    T = targets(v3, d5)
    m = v3[['date', 'ticker', 'rv']].merge(d5[['date', 'ticker', 'rv']].rename(columns={'rv': 'rv5'}), on=['date', 'ticker'], how='left')
    r = np.log(m.rv5 / m.rv)
    print(f'ln(RV5sub/RV10) по строкам v3: медиана {np.nanmedian(r):.4f}, среднее {np.nanmean(r):.4f}, СКО {np.nanstd(r):.4f}; '
          f'корр ln: {np.corrcoef(np.log(m.rv[m.rv5.notna()]), np.log(m.rv5[m.rv5.notna()]))[0, 1]:.4f}; без RV5 строк {int(m.rv5.isna().sum())}')
    for lag in (-1, 1):  # день строки: корреляция с соседними днями ниже, чем с тем же днём
        s = m.groupby('ticker')['rv'].shift(lag)
        ok = m.rv5.notna() & s.notna()
        print(f'  корр ln RV5(d) с ln RV10(d{lag:+d}): {np.corrcoef(np.log(s[ok]), np.log(m.rv5[ok]))[0, 1]:.4f}')
    T.to_parquet(T5, index=False)
    hs = hashlib.sha256(open(T5, 'rb').read()).hexdigest()
    (OUT / 'rv5sub_targets.sha256').write_text(f'{hs}  {T5}\n')
    print('записан', T5, 'SHA-256', hs, '; доля NaN:', T[[f't5_h{h}' for h in HS]].isna().mean().round(4).to_dict())


# ---------------------------------------------------------------- обучение на цели RV5 (оценка и ранняя остановка — по 10-мин цели)
VAR = {'raw': 'D25_6', 'sc': 'D25_6sc'}  # raw — ln RV5sub как есть; sc — + c_тикер = среднее (ln RV10 − ln RV5) по строкам обучения шага
FALLBACK = {}


def patch_target(h, var):
    """Подмена ytr_ln в l2_lib.step_data (её же вызывает make_ref): строки обучения получают ln цели RV5 (окно [d+1, d+h], как v3);
    строки без RV5 — 10-мин цель (счётчик в FALLBACK). Винзоризация 1/99 по тикеру (cfg wins) — по новой цели. val/test, ES, оценка — 10 мин."""
    import l2_lib as L
    t5 = pd.read_parquet(T5)[['date', 'ticker', f't5_h{h}']]
    sd0 = L.step_data

    def sd(A, cols, hh, step, model, cfg, base_tag, **kw):
        D = sd0(A, cols, hh, step, model, cfg, base_tag, **kw)
        tp = D['parts']['tr']
        k = tp[['date', 'ticker']].assign(date=pd.to_datetime(tp['date']).values).merge(t5, on=['date', 'ticker'], how='left')
        y5 = k[f't5_h{h}'].values
        y10 = D['y']['tr']
        miss = np.isnan(y5) | (y5 <= 0)
        ln5 = np.log(np.where(miss, y10, y5) + 1e-10)
        if var == 'sc':
            dlt = pd.Series(np.log(y10 + 1e-10) - ln5).where(~miss).groupby(tp['ticker'].values).transform('mean').fillna(0).values
            ln5 = ln5 + dlt
        if cfg.get('wins'):
            q = tp.assign(_y=ln5).groupby('ticker')['_y'].quantile([.01, .99]).unstack()
            ln5 = np.clip(ln5, tp['ticker'].map(q[0.01]).values, tp['ticker'].map(q[0.99]).values)
        D['ytr_ln'] = ln5
        FALLBACK[(model, step)] = (int(miss.sum()), len(miss))
        return D
    L.step_data = sd


def boost(h, var, workers, steps=range(1, 10), part=None):
    """part — обучение части шагов (1-6, пока докачиваются годы 2021+); итог пишется, когда собраны все части (pickle в m6/)."""
    import pickle
    import d25_7 as J
    J.TAG = VAR[var]
    J.OUTD = OUT / 'm6'
    J.OUTD.mkdir(parents=True, exist_ok=True)
    patch_target(h, var)
    boost_ = f'{J.TAG}_h{h}'
    t0 = time.time()
    steps = list(steps)
    if h in (1, 5):
        if h == 1:
            J.matrix1(jx=False)
            res = J.train1(workers, steps=steps)
        else:
            T = J.t5()
            T.D24 = J.OUTD
            A, cols, cfg = T.matrix(5, ext=False)
            res = T.train(5, A, cols, cfg, workers, legs=('single', 'seeds3'), steps=steps)
        if part:
            pickle.dump(res, open(J.OUTD / f'res_{boost_}_{part}.pkl', 'wb'))
            got = sorted(J.OUTD.glob(f'res_{boost_}_*.pkl'))
            res = [r for f in got for r in pickle.load(open(f, 'rb'))]
            if len({r[1] for r in res}) < 9:
                print(time.strftime('%T'), f'{boost_}: часть {part} ({len(steps)} шагов) готова за {time.time() - t0:.0f} с, ждём остальные', flush=True)
                return
        if h == 1:
            J.write1(boost_, res, J.R / 'test_locked' / boost_)
        else:
            T.write_legs(boost_, 5, res, J.R / 'test_locked' / boost_)
    else:  # нога Д16.5 h=22 (XGB, точки D16_5_h22) -> LGB-слот D19_1x / D19_4y, как Д25.7
        import l2_lib as L
        from tune_L2 import SPACEB
        A, FC = L.load_base()
        _, cols, _, _ = L.build_features(A, FC, 22, {})
        S = pd.read_csv(J.R / 'D16_5_h22_selected_h22.csv')
        S = S[S.model == 'xgb'].set_index('step')
        P = {s: {k: (int(S.loc[s, k]) if sc in ('int', 'logint') else float(S.loc[s, k])) for k, _, _, sc in SPACEB['xgb']} for s in range(1, 10)}
        J.G.update(A=A, cols=cols, P=P)
        import multiprocessing as mp
        with mp.get_context('fork').Pool(workers) as pool:
            res = pool.map(J._job22, [(s, sd, False) for s in steps for sd in (42, 43, 44)], chunksize=1)
        if part:
            pickle.dump(res, open(J.OUTD / f'res_{boost_}_{part}.pkl', 'wb'))
            res = [r for f in sorted(J.OUTD.glob(f'res_{boost_}_*.pkl')) for r in pickle.load(open(f, 'rb'))]
            if len({r[0] for r in res}) < 9:
                print(time.strftime('%T'), f'{boost_}: часть {part} готова за {time.time() - t0:.0f} с, ждём остальные', flush=True)
                return
        lock = J.R / 'test_locked' / f'{J.TAG}_d165_h22'
        lock.mkdir(parents=True, exist_ok=True)
        for sfx, f in (('', lambda r: r[1] == 42), ('_seeds4344', lambda r: r[1] in (43, 44))):
            pd.concat([r[3] for r in res if f(r)], ignore_index=True).to_parquet(J.OUTD / f'{J.TAG}_d165_xgb_h22{sfx}_val.parquet', index=False)
            pd.concat([r[4] for r in res if f(r)], ignore_index=True).to_parquet(lock / f'{J.TAG}_d165_xgb_h22{sfx}_test.parquet', index=False)
    print(time.strftime('%T'), f'{boost_}: обучено {len(res)} за {time.time() - t0:.0f} с', flush=True)


def post(h, var):
    """Гибрид (gibrid_v17.py) и правило третьего круга — кодом Д25.7 с другим TAG (база — та же конструкция на 10-мин цели)."""
    from types import SimpleNamespace
    import d25_7 as J
    J.TAG = VAR[var]
    J.OUTD = OUT / 'm6'
    a = SimpleNamespace(h=[h], v0=False, workers=1, harj=False)
    if h == 22:
        J.stage_d19(a)
        J.stage_gib22(a)
        J.stage_score22(a)
    else:
        J.stage_gibrid(a)
        J.stage_eval(a)


def smoke(h):
    """Проверка подмены без обучения: шаг 2, XGB — доля строк обучения с RV5 и сверка ytr_ln с ln t5 (вне винзоризации)."""
    import l2_lib as L
    patch_target(h, 'raw')
    A, FC = L.load_base()
    _, cols, _, _ = L.build_features(A, FC, h, {})
    for s in (1, 6, 9):
        D = L.step_data(A, cols, h, s, 'xgb', {}, 'Bv2b')
        print(f'h={h} шаг {s}: строк обучения {len(D["ytr_ln"])}, без RV5 (10-мин цель) {FALLBACK[("xgb", s)][0]}; '
              f'среднее ln y10 {np.log(D["y"]["tr"] + 1e-10).mean():.4f}, ln цели {D["ytr_ln"].mean():.4f}', flush=True)


if __name__ == '__main__':
    import numpy as np
    if sys.argv[1] == 'post':
        os.environ.setdefault('L2_FEAT', str(DATA / 'regen' / 'features_pit_v3.parquet'))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        post(int(sys.argv[2]), sys.argv[3])
        sys.exit(0)
    if sys.argv[1] == 'smoke':
        os.environ.setdefault('L2_FEAT', str(DATA / 'regen' / 'features_pit_v3.parquet'))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        smoke(int(sys.argv[2]))
        sys.exit(0)
    if sys.argv[1] == 'boost':
        h_, var_, w_ = int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
        part_ = sys.argv[5] if len(sys.argv) > 5 else None  # '1-6' | '7-9'
        st_ = range(int(part_.split('-')[0]), int(part_.split('-')[1]) + 1) if part_ else range(1, 10)
        if h_ == 1:
            import d22_h1  # noqa: F401 — L2_FEAT = features_pit_v3_rrv до импорта l2_lib (как D22_h1)
        elif h_ == 22:
            os.environ['L2_FEAT'] = str(DATA / 'regen' / 'features_pit_v3_d165.parquet')
        else:
            os.environ.setdefault('L2_FEAT', str(DATA / 'regen' / 'features_pit_v3.parquet'))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        boost(h_, var_, w_, st_, part_)
        sys.exit(0)
    {'borders': borders, 'sample': sample, 'download': download, 'check': check, 'rv': rv, 'leak': leak, 'target': target}[sys.argv[1]]()
else:
    import numpy as np
