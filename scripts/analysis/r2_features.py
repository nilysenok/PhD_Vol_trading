#!/usr/bin/env python3
"""R2, агент F (26.10.2026): группы признаков кластера 5 (G1–G10) и история с начала 10-мин баров ISS (кластер 1, history_start = ISS_start).
Протокол — docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md (v3), схема — models/designs/r2/schema_v1.yaml (ключ features).

Информационное множество d0: в строке (date = d, ticker) — только данные по день d включительно, RV_d до 18:50 (бары begin < 18:50).
Никакого shift(1) у величин дня d: RV_d/w/m, J, BV и т. п. пересчитаны из дневной RV (data/regen/D24/rv_1850_noquar_hist.parquet,
= столбец rv матрицы v3/v3h), а не взяты из features_pit_v3 (там rv_d = rv.shift(1), jump_lag1 = jump.shift(1)).

Ключи строк — features_pit_v3h (Д24.10: 2011-12-08 … 2026-02, 17 тикеров; строки годов ≥ 2016 побайтно равны v3). Для
history_start = 2014 берутся строки с date ≥ 2014-06-09 (ключи v3) — значения признаков на них те же.

Группы (столбцы g{k}_*; полный перечень и описание — models/designs/r2/r2_F_features.yaml):
  G1  внешние as-of: ln RVI² (срез 18:50), ln RV_m IMOEX, ln RV_m рынка (17 тикеров, d0), PC1–PC3 глобальной волатильности (Д7.2, PCA 1 января)
  G2  календарь окна [t+1, t+h] для h = 1/5/22: будних дней, число заседаний ЦБ / экспираций MOEX / FOMC в окне, дней до ближайшего —
      только события с known_from ≤ t (как Д22); дивидендов нет (файлы пусты); до 2014-01-01 календаря нет -> NaN
  G3  форма дня: ночной гэп ln(open_d / close_{d−1}) и ln(гэп² + 1e-6) (как D13), доля RV последнего часа (крейт), RRV PK/GK d/w/m (Д17.2)
  G4  скачки: ln BV_d, J_d/RV_d, число дней с J > 0 за 5/22, дней с последнего J > 0, сумма J за 22 (Д25.7, но окно включает день d);
      то же для «крупного» скачка J/RV ≥ 0,25 (ночное умолчание)
  G5  кросс-секция (крейт moex_iss, бинарник r2_xs): средняя парная реализованная корреляция 17 тикеров d/w/m, СКО ln RV по тикерам
      (день и неделя), ln RV тикера минус среднее по тикерам, ln RV лидеров SBER, LKOH, GAZP дня d (для цели d+1 — лаг 1)
  G6  RV USDRUB (USD000UTSTOM, CETS) и ближнего BR d/w/m со срезом 18:40 (бары begin < 18:30+10 мин, т. е. закрытые к 18:40), день d
  G7  ликвидность: ln объёма, ln оборота окна, оборот к среднему за 22 дня, Amihud d/w/m (крейт); числа сделок и капитализации нет
  G8  история ошибок: ē_5, ē_10 по тикеру и рынку и длина серии ошибок одного знака для OOS-прогноза HAR-J h=1 на d0 (пул, OLS, переоценка
      1 января по строкам, чья цель раньше этой даты); e(s) = ln RV(s+1) − ŷ(s) известна в s+1, ē_m(t) = среднее e(t−1..t−m);
      плюс готовые ē Д23 (финалист D12_h1, с 2016)
  G9  срочная структура: ln RV_d/RV_w, ln RV_w/RV_m, ln RVI²/RV_m, ln RV_m/RV_m(−22) (все — d0)
  G10 сезонность: день недели, месяц, день месяца, будних дней до конца месяца, последний месяц квартала

Стадии:
  bars   — сборка крейта, начала рядов (xs_starts.csv), прогон r2_xs (полный и со срезами для детектора), выгрузка GAZP (если нет)
  daily  — дневные входы (RRV с ISS_start, ночной гэп, RV USDRUB/BR со срезом 18:40, RV GAZP) + сверки с готовыми рядами
  build  — G1…G10 -> data/regen/R2/feat/G{k}.parquet, объединённый data/regen/R2/features/features_r2_v1.parquet, INDEX_F.csv, yaml
  check  — детектор «будущее не влияет» с подложенной утечкой (каждая группа) и проверка сдвига (Спирмен с ln RV_{d+s}, s = −2…+1,
           2017–2021, медиана по тикерам; argmax должен быть s = 0)
  nobs   — число обучающих наблюдений по шагам 1–10 для history_start = 2014 и ISS_start (только ключи date/ticker, значения не читаются)
Ничего за годы ≥ 2022 не оценивается: проверка сдвига и детекторы — только на датах ≤ 2021-12-31 (RV ≥ 2022 в проверки не входит).
python3 scripts/analysis/r2_features.py --stage bars|daily|build|check|nobs|all [--bin путь к r2_xs]"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import d73_features as D73  # единственное определение PIT-версии (known_from/known_to)
MAIN = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading')
DATA = MAIN / 'data'
REG = DATA / 'regen'
OUT = REG / 'R2' / 'feat'
WORK = OUT / '_work'
# v2 (26.10, 16:30): без 3 столбцов, заблокированных O (PR #167); v3 (16:45): G1 ln RV_m IMOEX и выбор строк внешних рядов — как harx
# (флаг imoex_low_bars вместо low_bars_dup0, строки с NaN close_IMOEX не выбрасываются). v1, v2 не меняются.
COMB = REG / 'R2' / 'features' / 'features_r2_v3.parquet'
DOCS = ROOT / 'docs' / 'otchety' / 'r2_F'
YAML = ROOT / 'models' / 'designs' / 'r2' / 'r2_F_features.yaml'
V3H = REG / 'features_pit_v3h.parquet'
RVH = REG / 'D24' / 'rv_1850_noquar_hist.parquet'
HIST_TK = REG / 'D24' / 'hist_tickers.csv'
CAND = REG / 'D24' / 'candles_hist'
RRV14 = DATA / 'iss_check' / 'rrv_1850.parquet'
BARS_H = DATA / 'iss_hist' / 'bars_10m'          # основная сессия, история до 2014-05-31 (+ {T}_EQNL)
BARS_M = DATA / 'iss_check' / 'bars_10m'         # основная сессия, с 2014-06-02 (вход d23_corr и D13)
EXT = DATA / 'iss_ext' / 'bars_10m_full'
BR_ROLL = REG / 'D24' / 'rv_ext_daily_br_roll.csv'
EXT_DAILY = REG / 'D24' / 'rv_ext_daily.parquet'
LEAD_DIR = DATA / 'iss_lead'
EBAR = REG / 'D23_ebar.parquet'
D23RAW = REG / 'D23' / 'd23_corr_raw.parquet'
D13I = REG / 'D13' / 'intraday1850.parquet'
CALD = DATA / 'calendar'
T17 = ["SBER", "VTBR", "MOEX", "LKOH", "TATN", "NVTK", "ALRS", "MTLR", "HYDR", "IRAO", "OGKB", "RTKM", "MTSS", "MGNT", "AFLT", "PHOR", "LSRG"]
FIRST14 = pd.Timestamp('2014-06-09')
LAST_DEV = pd.Timestamp('2021-12-31')            # проверки — только до этой даты
CUTS = ('2016-06-30', '2019-06-30', '2021-06-30')
CAL_TYPES = ('cbr', 'expirations', 'fomc')
GROUPS = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10']
EPS = 1e-10


def log(*a):
    print(time.strftime('%T'), *a, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def lnc(x):
    return np.log(np.clip(np.asarray(x, dtype='float64'), EPS, None))


# ================================================================ bars: крейт и GAZP
def xs_bin(a):
    if a.bin:
        return Path(a.bin)
    td = Path(os.environ.get('CARGO_TARGET_DIR', ROOT / 'moex_iss' / 'target'))
    return td / 'release' / 'r2_xs'


def stage_bars(a):
    WORK.mkdir(parents=True, exist_ok=True)
    subprocess.run(['cargo', 'build', '--release', '-j', '4', '--bin', 'r2_xs', '--bin', 'moex_iss',
                    '--manifest-path', str(ROOT / 'moex_iss' / 'Cargo.toml')], check=True)
    H = pd.read_csv(HIST_TK)
    st = H[['ticker', 'used_from']].rename(columns={'used_from': 'from'})
    st['from'] = st['from'].fillna('2014-06-02')  # OGKB: истории нет (Д24.10) — ряд с начала основной выгрузки
    st.to_csv(WORK / 'xs_starts.csv', index=False)
    b = xs_bin(a)
    for cut in ['-'] + list(CUTS):
        out = WORK / ('xs_full.parquet' if cut == '-' else f'xs_cut_{cut}.parquet')
        subprocess.run([str(b), str(out), cut, str(WORK / 'xs_starts.csv'), str(BARS_H), str(BARS_M)], check=True)
    if not (LEAD_DIR / 'bars_10m_full' / 'GAZP.parquet').exists():  # GET iss.moex.com, 4 потока, 100 мс (download.rs)
        mi = b.parent / 'moex_iss'
        subprocess.run([str(mi), '--out', str(LEAD_DIR), 'bars', '--from', '2011-12-01', '--till', '2026-02-27', '--tickers', 'GAZP'], check=True)


# ================================================================ daily: дневные входы
def read_bars(p):
    b = pd.read_parquet(p).rename(columns={'begin': 'timestamp'})
    b['timestamp'] = pd.to_datetime(b['timestamp'])
    return b.sort_values('timestamp').drop_duplicates('timestamp', keep='last').reset_index(drop=True)


def mins(ts):
    return ts.dt.hour * 60 + ts.dt.minute


def rv_daily(bars, lo, hi):
    """Порт moex_iss rv.rs::daily (d121_rv28.daily, побитово): окно begin ∈ [lo; hi) минут, первая доходность дня — от последнего бара окна
    прошлого дня."""
    import d121_rv28 as RV
    m = mins(bars.timestamp)
    w = bars[(m >= lo) & (m < hi)].reset_index(drop=True)
    return RV.daily(w)


def merged_main(t, start):
    """Бары основной сессии тикера: история iss_hist (EQNL, затем TQBR) + iss_check; поздний источник важнее; с дня start."""
    parts = []
    for d in (BARS_H, BARS_M):
        for nm in (f'{t}_EQNL.parquet', f'{t}.parquet'):
            if (d / nm).exists():
                parts.append(read_bars(d / nm))
    b = pd.concat(parts, ignore_index=True).drop_duplicates('timestamp', keep='last').sort_values('timestamp').reset_index(drop=True)
    return b[b.timestamp >= pd.Timestamp(start)].reset_index(drop=True)


def gap_one(b):
    """Как d13_lib.intraday_one: open первого бара окна [09:50; 18:50) дня d к close последнего бара прошлого дня; |гэп| ≥ 0,4 -> NaN."""
    d = b.timestamp.dt.normalize()
    last_close = b.groupby(d)['close'].last()
    m = mins(b.timestamp)
    w = b[(m >= 9 * 60 + 50) & (m < 18 * 60 + 50)]
    op = w.groupby(w.timestamp.dt.normalize())['open'].first()
    prev = last_close.shift(1).reindex(op.index)
    g = np.log(op / prev)
    return g.where(g.abs() < 0.4)


def rrv_one(b):
    """Д17.2 (moex_iss range.rs::daily): Σ (ln H/L)²/(4 ln 2) и Σ [0,5 (ln H/L)² − (2 ln 2 − 1)(ln C/O)²] по барам окна [09:50; 18:50)."""
    m = mins(b.timestamp)
    w = b[(m >= 9 * 60 + 50) & (m < 18 * 60 + 50)]
    w = w[(w.open > 0) & (w.high > 0) & (w.low > 0) & (w.close > 0) & (w.high >= w.low)]
    hl, co = np.log(w.high / w.low), np.log(w.close / w.open)
    d = w.timestamp.dt.normalize()
    pk = (hl ** 2 / (4 * np.log(2))).groupby(d).sum()
    gk = (0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2).groupby(d).sum()
    return pd.DataFrame({'rrv_pk': pk, 'rrv_gk': gk})


def stage_daily(a):
    WORK.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    st = pd.read_csv(WORK / 'xs_starts.csv').set_index('ticker')['from']
    chk = []
    # 1) крейт против d23_corr (те же определения, вход d23 — iss_check/bars_10m с 2014-06-02)
    x = pd.read_parquet(WORK / 'xs_full.parquet'); x['date'] = pd.to_datetime(x.date)
    r = pd.read_parquet(D23RAW); r['date'] = pd.to_datetime(r.date)
    m = r.merge(x, on=['date', 'ticker'], suffixes=('_d23', ''))
    m = m[m.date >= pd.Timestamp('2014-07-15')]  # окно corr_m (22 дня) уже не задевает историю до 2014-06-02
    for c in ['corr_d', 'corr_w', 'corr_m', 'disp_lnrv', 'amihud', 'lasthour']:
        chk.append(dict(sverka='r2_xs vs d23_corr_raw (>= 2014-07-15)', col=c, n=len(m),
                        max_abs=float(np.nanmax(np.abs(m[c] - m[c + '_d23']))), nan_mismatch=int((m[c].isna() != m[c + '_d23'].isna()).sum())))
    # 2) RRV: история — свечи D24.10 без карантина (data/regen/D24/candles_hist, до 2014-06-09) + rrv_1850 (крейт) с 2014-06-09
    rows = []
    for t in T17:
        c = pd.read_parquet(CAND / f'{t}.parquet'); c['timestamp'] = pd.to_datetime(c['timestamp'])
        c = c[c.timestamp < FIRST14]
        if len(c):
            q = rrv_one(c); q.index.name = 'date'
            rows.append(q.reset_index().assign(ticker=t))
    rr = pd.concat(rows, ignore_index=True)
    r14 = pd.read_parquet(RRV14); r14['date'] = pd.to_datetime(r14.date)
    # сверка формулы: тот же порт на барах iss_check (без исключения карантина) против rrv_1850 в днях, где число баров совпало
    for t in ('SBER', 'LKOH', 'AFLT'):
        b = read_bars(DATA / 'iss_check' / 'bars_10m_full' / f'{t}.parquet')
        q = rrv_one(b[b.timestamp >= FIRST14]); q.index.name = 'date'
        mm = q.reset_index().merge(r14[r14.ticker == t], on='date', suffixes=('', '_c'))
        mw = mins(b.timestamp); nb = b[(mw >= 590) & (mw < 1130)].groupby(b.timestamp.dt.normalize()).size()
        mm = mm[mm.n_bars.values == nb.reindex(mm.date).values]
        chk.append(dict(sverka=f'RRV-порт vs rrv_1850 ({t}, дни без карантина)', col='rrv_pk,rrv_gk', n=len(mm),
                        max_abs=float(max(np.abs(mm.rrv_pk - mm.rrv_pk_c).max(), np.abs(mm.rrv_gk - mm.rrv_gk_c).max())), nan_mismatch=0))
    rr = pd.concat([rr, r14[['date', 'ticker', 'rrv_pk', 'rrv_gk']]], ignore_index=True).sort_values(['ticker', 'date'])
    rr.to_parquet(WORK / 'rrv_daily.parquet', index=False)
    # 3) ночной гэп по барам основной сессии (история + iss_check), сверка с D13 intraday1850
    gp = []
    for t in T17:
        g = gap_one(merged_main(t, st[t])); g.index.name = 'date'
        gp.append(g.rename('gap_on').reset_index().assign(ticker=t))
    gp = pd.concat(gp, ignore_index=True)
    gp.to_parquet(WORK / 'gap_daily.parquet', index=False)
    d13 = pd.read_parquet(D13I, columns=['date', 'ticker', 'gap_on']); d13['date'] = pd.to_datetime(d13.date)
    mm = d13.merge(gp, on=['date', 'ticker'], suffixes=('_d13', ''))
    mm = mm[mm.date > pd.Timestamp('2014-06-02')]
    chk.append(dict(sverka='гэп vs D13 intraday1850 (> 2014-06-02)', col='gap_on', n=len(mm),
                    max_abs=float(np.nanmax(np.abs(mm.gap_on - mm.gap_on_d13))), nan_mismatch=int((mm.gap_on.isna() != mm.gap_on_d13.isna()).sum())))
    # 4) RV USDRUB и ближнего BR: порт rv.rs проверяется повтором rv_ext_daily (окна Д24.8), затем срез 18:40
    roll = pd.read_csv(BR_ROLL); roll['date'] = pd.to_datetime(roll.date)
    usd = read_bars(EXT / 'USD000UTSTOM.parquet')
    ext0 = pd.read_parquet(EXT_DAILY); ext0['date'] = pd.to_datetime(ext0.date)

    def ext_series(usd_hi, br_hi):
        u = rv_daily(usd, 9 * 60 + 50, usd_hi)[['rv', 'n_bars']]; u.index.name = 'date'
        u = u.reset_index().assign(ser='USD')
        br = []
        for sec, g in roll.groupby('secid'):
            p = EXT / f'{sec}.parquet'
            if not p.exists():
                continue
            d = rv_daily(read_bars(p), 10 * 60, br_hi)[['rv', 'n_bars']]; d.index.name = 'date'
            d = d.reset_index()
            br.append(d[d.date.isin(g.date)])
        return pd.concat([u, pd.concat(br, ignore_index=True).assign(ser='BR')], ignore_index=True)

    e0 = ext_series(18 * 60 + 50, 18 * 60 + 45)
    for s, tk in (('USD', 'USDRUB'), ('BR', 'BRENT')):
        mm = e0[e0.ser == s].merge(ext0[ext0.ticker == tk], on='date', suffixes=('', '_d24'))
        chk.append(dict(sverka=f'порт RV {s} (окно Д24.8) vs rv_ext_daily', col='rv', n=len(mm), max_abs=float(np.abs(mm.rv - mm.rv_d24).max()),
                        nan_mismatch=int(len(e0[e0.ser == s]) - len(mm))))
    e40 = ext_series(18 * 60 + 40, 18 * 60 + 40)  # begin < 18:40: последний бар 18:30–18:40
    e40 = e40[e40.n_bars >= 30]  # день ряда — при ≥ 30 доходностях окна (Д24.8)
    e40.to_parquet(WORK / 'ext1840_daily.parquet', index=False)
    # 5) GAZP — лидер вне 17 тикеров: RV окна [09:50; 18:50) без карантина
    gz = LEAD_DIR / 'bars_10m_full' / 'GAZP.parquet'
    if gz.exists():
        g = rv_daily(read_bars(gz), 9 * 60 + 50, 18 * 60 + 50)[['rv', 'n_bars']]; g.index.name = 'date'
        g = g.reset_index()
        g = g[g.n_bars >= 30]
        g.to_parquet(WORK / 'lead_gazp_daily.parquet', index=False)
        chk.append(dict(sverka='GAZP: дней RV (≥ 30 доходностей)', col='rv', n=len(g), max_abs=np.nan, nan_mismatch=0,
                        note=f'{g.date.min().date()} … {g.date.max().date()}'))
    C = pd.DataFrame(chk)
    C.to_csv(DOCS / 'sverki_daily.csv', index=False)
    print(C.to_string())


# ================================================================ входы и построение групп
def load_inputs():
    K = pd.read_parquet(V3H, columns=['date', 'ticker', 'volume', 'close_RVI', 'rv_IMOEX', 'imoex_low_bars', 'close_IMOEX',
                                      'sp500', 'eem', 'dxy', 'gold', 'brent', 'wti', 'vix'])
    K['date'] = pd.to_datetime(K.date)
    K = K.sort_values(['date', 'ticker']).reset_index(drop=True)
    I = {}
    I['KEYS'] = K[['date', 'ticker']].copy()
    I['V'] = K[['date', 'ticker', 'volume']].copy()
    I['IDX'] = K.drop(columns=['ticker', 'volume']).drop_duplicates('date').reset_index(drop=True)
    B = pd.read_parquet(RVH); B['date'] = pd.to_datetime(B.date)
    I['B'] = I['KEYS'].merge(B[['date', 'ticker', 'rv', 'bv', 'j']], on=['date', 'ticker'], how='left')
    x = pd.read_parquet(WORK / 'xs_full.parquet'); x['date'] = pd.to_datetime(x.date)
    I['XS'] = x
    for k, f in (('RRV', 'rrv_daily'), ('GAP', 'gap_daily'), ('EXT', 'ext1840_daily'), ('LEAD', 'lead_gazp_daily')):
        p = WORK / f'{f}.parquet'
        I[k] = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=['date', 'rv'])
        I[k]['date'] = pd.to_datetime(I[k]['date'])
    e = pd.read_parquet(EBAR); e['date'] = pd.to_datetime(e.date)
    I['EBAR'] = e
    ev = {}
    for ty in CAL_TYPES:
        d = pd.read_csv(CALD / f'{ty}.csv')
        ev[ty] = pd.DataFrame({'date': pd.to_datetime(d['date']), 'known_from': pd.to_datetime(d['known_from']),
                               'known_to': D73.known_to_col(d)})
    I['EV'] = ev
    return I


def on_keys(I, df, cols, how='left'):
    """Перенос столбцов df (date, ticker, cols) на ключи."""
    k = I['KEYS'].merge(df[['date', 'ticker'] + cols], on=['date', 'ticker'], how=how)
    assert len(k) == len(I['KEYS'])
    return k


def by_date_on_keys(I, s, name):
    """Ряд по датам -> столбец на ключах."""
    return I['KEYS'].assign(**{name: I['KEYS'].date.map(s).values})


def roll_t(df, col, w, minp, key='ticker'):
    return df.groupby(key, sort=False)[col].transform(lambda s: s.rolling(w, min_periods=minp).mean())


def har0(I):
    """RV_d, RV_w, RV_m в версии d0 (окно кончается днём d, min_periods = 1, как add_har_lags в v3) и J_w, J_m."""
    B = I['B'].sort_values(['ticker', 'date']).reset_index(drop=True)
    B['rvw'] = roll_t(B, 'rv', 5, 1)
    B['rvm'] = roll_t(B, 'rv', 22, 1)
    B['jw'] = roll_t(B, 'j', 5, 1)
    B['jm'] = roll_t(B, 'j', 22, 1)
    return B


def _fill_rows(IDX):
    """Строки-заполнители матрицы (как harx._by_date: close_IMOEX равен значению первой строки) — не реальные дни внешних рядов."""
    # v3: ровно как harx (без «| isna()»: строки с NaN остаются, rolling их пропускает, но они занимают место в окне 22 строк)
    return (IDX['close_IMOEX'] == IDX['close_IMOEX'].iloc[0]).values


def g1(I):
    x = I['IDX'].sort_values('date').set_index('date')
    v = x[~_fill_rows(x.reset_index())].copy()
    # v3: как harx.t_rvi / t_rvm_imoex — reindex(method='ffill') по тем же датам, т. е. без протяжки пропусков
    rvi = np.log(v['close_RVI'].clip(lower=1e-6) ** 2).reindex(x.index, method='ffill')
    # дни с неполными барами IMOEX (n_bars < 30, флаг imoex_low_bars матрицы) — NaN и протяжка ≤ 2 дней, как harx.t_rvm_imoex
    rim = v['rv_IMOEX'].where(v['imoex_low_bars'] != 1).ffill(limit=2)
    rvm_im = np.log((rim ** 2).rolling(22, min_periods=13).mean().clip(lower=EPS)).reindex(x.index, method='ffill')
    # PC1–PC3 (Д7.2 на трёх компонентах): 5-дневная реализованная вол. глобальных активов (Yahoo, в матрице лаг доступности 1) и ln VIX,
    # стандартизация и веса оцениваются 1 января года по строкам строго раньше
    GL = ['sp500', 'eem', 'dxy', 'gold', 'brent', 'wti']
    r = {g: np.log(v[g].clip(lower=EPS)).diff() for g in GL}
    X = pd.DataFrame({g: np.log(np.sqrt((r[g] ** 2).rolling(5, min_periods=4).mean()) + 1e-4) for g in GL})
    X['lvix'] = np.log(v['vix'].clip(lower=EPS))
    pcs = pd.DataFrame(np.nan, index=v.index, columns=['pc1', 'pc2', 'pc3'])
    for y in sorted(set(v.index.year)):
        fit = X[X.index < pd.Timestamp(y, 1, 1)].dropna()
        if len(fit) < 252:
            continue
        mu, sd = fit.mean(), fit.std().replace(0, np.nan)
        W = np.linalg.svd(((fit - mu) / sd).values, full_matrices=False)[2][:3]
        W = W * np.sign(W.sum(axis=1, keepdims=True) + 1e-300)
        m = (v.index.year == y) & X.notna().all(axis=1).values
        pcs.loc[m, :] = ((X[m] - mu) / sd).values @ W.T
    pcs = pcs.reindex(x.index).ffill()
    B = har0(I)
    mkt = np.log(B.groupby('date')['rvm'].mean().clip(lower=EPS))
    out = I['KEYS'].copy()
    out['g1_ln_rvi2'] = out.date.map(rvi).values
    out['g1_ln_rvm_imoex'] = out.date.map(rvm_im).values
    out['g1_ln_rvm_mkt'] = out.date.map(mkt).values
    for c in pcs:
        out[f'g1_{c}'] = out.date.map(pcs[c]).values
    return out


def cal_window(dates, a, b, ev, leak=False):
    """Как d22_harxp.cal_window: окно горизонта по будним дням, события с known_from ≤ t (leak=True — без этого условия)."""
    ud = np.array(sorted(pd.unique(pd.to_datetime(dates))), dtype='datetime64[D]')
    stt = np.busday_offset(ud, 0, roll='backward')
    lo = (np.busday_offset(stt, a - 1) if a > 1 else ud) + np.timedelta64(1, 'D')
    hi = np.busday_offset(stt, b)
    res = {'bd': ((hi - lo) / np.timedelta64(1, 'D') + 1).astype(float)}
    for ty, E in ev.items():
        d = E['date'].values.astype('datetime64[D]'); kf = E['known_from'].values.astype('datetime64[D]')
        kt = E['known_to'].values.astype('datetime64[D]')
        n, dte = np.zeros(len(ud)), np.full(len(ud), 90.0)
        for i, t in enumerate(ud):
            k = d[(D73.izvestna(kf, kt, t) | leak) & (d > t)]
            n[i] = ((k >= lo[i]) & (k <= hi[i])).sum()
            if len(k):
                dte[i] = min(90.0, float((k.min() - t) / np.timedelta64(1, 'D')))
        res[f'n_{ty}'], res[f'dte_{ty}'] = n, dte
    return pd.DataFrame(res, index=pd.DatetimeIndex(ud))


def g2(I):
    out = I['KEYS'].copy()
    leak = I.get('_cal_leak', False)
    for h in (1, 5, 22):
        G = cal_window(out.date, 1, h, I['EV'], leak)
        out[f'g2_bd_h{h}'] = out.date.map(G['bd']).values
        for ty in CAL_TYPES:
            out[f'g2_n_{ty}_h{h}'] = out.date.map(G[f'n_{ty}']).values
            if h == 1:
                out[f'g2_dte_{ty}'] = out.date.map(G[f'dte_{ty}']).values
    cols = [c for c in out if c.startswith('g2_')]
    out.loc[out.date < pd.Timestamp('2014-01-01'), cols] = np.nan  # календарь собран с 2014 года
    return out


def g3(I):
    gp = I['GAP'].copy()
    out = on_keys(I, gp, ['gap_on'])
    out = out.rename(columns={'gap_on': 'g3_gap_on'})
    out['g3_ln_gap2'] = np.log(out['g3_gap_on'] ** 2 + 1e-6)
    xs = on_keys(I, I['XS'], ['lasthour'])
    out['g3_lasthour'] = xs['lasthour'].values
    # RRV PK/GK d/w/m: ln среднего за 1/5/22 дня ряда тикера, окно кончается днём d (Д17.2), протяжка по строкам тикера
    R = I['RRV'].sort_values(['ticker', 'date']).reset_index(drop=True)
    f = R[['date', 'ticker']].copy()
    for m in ('pk', 'gk'):
        for w, n in (('d', 1), ('w', 5), ('m', 22)):
            mu = roll_t(R, f'rrv_{m}', n, 1)
            f[f'g3_rrv_{m}_{w}'] = np.log(mu.where(mu > 0))
    rc = [c for c in f if c.startswith('g3_rrv')]
    k = on_keys(I, f, rc).sort_values(['ticker', 'date'])
    k[rc] = k.groupby('ticker', sort=False)[rc].ffill()
    k = k.sort_index()
    for c in rc:
        out[c] = k[c].values
    return out


def g4(I):
    B = I['B'].sort_values(['ticker', 'date']).reset_index(drop=True)
    g = B.groupby('ticker', sort=False)
    out = B[['date', 'ticker']].copy()
    out['g4_ln_bv'] = lnc(B.bv)
    out['g4_jshare'] = (B.j / B.rv.where(B.rv > 0)).values
    pos = g.cumcount().astype('float64')
    for nm, ind in (('jx', (B.j > 0)), ('jbig', (B.j >= 0.25 * B.rv) & (B.j > 0))):
        ind = ind.astype('float64').where(B.j.notna())
        out[f'g4_{nm}_n5'] = ind.groupby(B.ticker).transform(lambda s: s.rolling(5, min_periods=3).sum())
        out[f'g4_{nm}_n22'] = ind.groupby(B.ticker).transform(lambda s: s.rolling(22, min_periods=13).sum())
        last = pos.where(ind == 1).groupby(B.ticker).ffill()
        out[f'g4_{nm}_since'] = pos - last
    out['g4_jx_sum22'] = B.j.groupby(B.ticker).transform(lambda s: s.rolling(22, min_periods=13).sum())
    out['g4_ln_jx_sum22'] = lnc(out['g4_jx_sum22'] + 1e-8)
    return I['KEYS'].merge(out, on=['date', 'ticker'], how='left')


def g5(I):
    X = I['XS']
    out = on_keys(I, X, ['corr_d', 'corr_w', 'corr_m', 'disp_lnrv'])
    out = out.rename(columns={c: f'g5_{c}' for c in ['corr_d', 'corr_w', 'corr_m', 'disp_lnrv']})
    dd = X.drop_duplicates('date').set_index('date').sort_index()['disp_lnrv']
    out['g5_disp_lnrv_w'] = out.date.map(dd.rolling(5, min_periods=3).mean()).values
    B = I['B']
    lr = pd.Series(lnc(B.rv), index=B.index)
    out['g5_lnrv_rel'] = (lr - lr.groupby(B.date).transform('mean')).values  # ключи B = ключи KEYS в том же порядке
    for t in ('SBER', 'LKOH'):
        s = B[B.ticker == t].set_index('date')['rv']
        out[f'g5_ln_lead_{t}'] = np.log(out.date.map(s).astype('float64').clip(lower=1e-12)).values
    L = I['LEAD']
    s = L.set_index('date')['rv'] if len(L) else pd.Series(dtype='float64')
    out['g5_ln_lead_GAZP'] = np.log(out.date.map(s).astype('float64').clip(lower=1e-12)).values
    return out


def g6(I):
    E = I['EXT']
    out = I['KEYS'].copy()
    ud = pd.DataFrame({'date': np.sort(out.date.unique())})
    for s, nm in (('USD', 'usd'), ('BR', 'br')):
        e = E[E.ser == s].sort_values('date').reset_index(drop=True)
        e['d'] = e.rv
        e['w'] = e.rv.rolling(5, min_periods=3).mean()
        e['m'] = e.rv.rolling(22, min_periods=13).mean()
        e['sday'] = e.date
        m = pd.merge_asof(ud, e[['date', 'sday', 'd', 'w', 'm']], on='date', direction='backward')
        stale = (m.date - m.sday).dt.days > 10  # Д24.8: последний день ряда старше 10 календарных дней -> NaN
        for c in ('d', 'w', 'm'):
            m.loc[stale, c] = np.nan
            s = pd.Series(np.log(m[c].where(m[c] > 0)).values, index=m.date)
            out[f'g6_{nm}_ln_rv_{c}'] = out.date.map(s).values
    return out


def g7(I):
    out = on_keys(I, I['V'], ['volume'])
    out['g7_ln_volume'] = np.log1p(out.pop('volume').astype('float64'))
    X = I['XS'].sort_values(['ticker', 'date']).reset_index(drop=True)
    X['v22'] = roll_t(X, 'value', 22, 13)
    X['a5'] = roll_t(X, 'amihud', 5, 3)
    X['a22'] = roll_t(X, 'amihud', 22, 13)
    k = on_keys(I, X, ['value', 'v22', 'amihud', 'a5', 'a22'])
    out['g7_ln_value'] = lnc(k.value + 1)
    out['g7_ln_value_rel22'] = np.log(k.value.where(k.value > 0) / k.v22.where(k.v22 > 0)).values
    out['g7_amihud'] = k.amihud.values
    for c, nm in (('amihud', 'd'), ('a5', 'w'), ('a22', 'm')):
        out[f'g7_ln_amihud_{nm}'] = np.log(k[c].where(k[c] > 0)).values
    return out


def g8(I):
    B = har0(I)
    g = B.groupby('ticker', sort=False)
    X = np.column_stack([lnc(B[c]) for c in ('rv', 'rvw', 'rvm', 'j', 'jw', 'jm')])
    y = np.log(g['rv'].shift(-1).astype('float64').clip(lower=EPS)).values
    tdate = g['date'].shift(-1)
    yh = np.full(len(B), np.nan)
    years = B.date.dt.year.values
    coefs = []
    for Y in range(int(years.min()) + 1, int(years.max()) + 1):
        tr = (tdate < pd.Timestamp(Y, 1, 1)).values & np.isfinite(y) & np.isfinite(X).all(axis=1)
        if tr.sum() < 1000:
            continue
        A = np.column_stack([np.ones(tr.sum()), X[tr]])
        beta = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        te = years == Y
        yh[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ beta
        coefs.append(dict(year=Y, n=int(tr.sum()), **{f'b{i}': float(b) for i, b in enumerate(beta)}))
    B['e'] = y - yh  # ошибка прогноза строки s; известна в день s+1
    B['e1'] = g['e'].shift(1)  # в строке t — e(t−1): к 18:50 дня t известна
    out = B[['date', 'ticker']].copy()
    for m in (5, 10):
        out[f'g8_ebar{m}'] = B.groupby('ticker', sort=False)['e1'].transform(lambda s: s.rolling(m, min_periods=m).mean())
    em = B.groupby('date')['e'].mean().sort_index()  # рыночная ошибка дня s (среднее по тикерам)
    em1 = em.shift(1)
    for m in (5, 10):
        out[f'g8_ebar{m}_mkt'] = B.date.map(em1.rolling(m, min_periods=m).mean()).values
    # длина серии ошибок одного знака: e(t−1), e(t−2), … со знаком e(t−1)
    sg = np.sign(B['e1'])
    brk = (sg != g['e1'].shift(1).pipe(np.sign)) | sg.isna()
    grp = brk.groupby(B.ticker).cumsum()
    run = sg.groupby([B.ticker, grp]).cumcount() + 1
    out['g8_run_sign'] = (run * sg).where(sg.notna())
    out = I['KEYS'].merge(out, on=['date', 'ticker'], how='left')
    E = I['EBAR'][['date', 'ticker', 'ebar5', 'ebar10', 'ebar5_mkt', 'ebar10_mkt']]
    k = on_keys(I, E, ['ebar5', 'ebar10', 'ebar5_mkt', 'ebar10_mkt'])
    for c in ('ebar5', 'ebar10', 'ebar5_mkt', 'ebar10_mkt'):
        out[f'g8_d23_{c}'] = k[c].values
    I['_g8_coefs'] = pd.DataFrame(coefs)
    return out


def g9(I):
    B = har0(I)
    g = B.groupby('ticker', sort=False)
    out = B[['date', 'ticker']].copy()
    out['g9_ln_d_w'] = lnc(B.rv) - lnc(B.rvw)
    out['g9_ln_w_m'] = lnc(B.rvw) - lnc(B.rvm)
    out['g9_ln_m_m22'] = lnc(B.rvm) - lnc(g['rvm'].shift(22))
    x = I['IDX'].sort_values('date').set_index('date')
    v = x[~_fill_rows(x.reset_index())]
    rvi_d = ((v['close_RVI'] / 100) ** 2 / 252).reindex(x.index).ffill()  # RVI (% годовых) -> дневная дисперсия
    out['g9_ln_rvi2_m'] = lnc(B.date.map(rvi_d).astype('float64')) - lnc(B.rvm)
    out.loc[~np.isfinite(out['g9_ln_m_m22']), 'g9_ln_m_m22'] = np.nan
    return I['KEYS'].merge(out, on=['date', 'ticker'], how='left')


def g10(I):
    out = I['KEYS'].copy()
    d = out.date
    out['g10_dow'] = d.dt.dayofweek.astype('float64')
    out['g10_month'] = d.dt.month.astype('float64')
    out['g10_dom'] = d.dt.day.astype('float64')
    me = (d + pd.offsets.MonthEnd(0)).values.astype('datetime64[D]')
    out['g10_bd_to_month_end'] = np.busday_count(d.values.astype('datetime64[D]') + 1, me + 1).astype('float64')
    out['g10_qend_month'] = d.dt.month.isin([3, 6, 9, 12]).astype('float64')
    return out


BUILD = {'G1': g1, 'G2': g2, 'G3': g3, 'G4': g4, 'G5': g5, 'G6': g6, 'G7': g7, 'G8': g8, 'G9': g9, 'G10': g10}
# какие входы читает группа (для детектора)
USES = {'G1': ['IDX', 'B'], 'G2': ['EV'], 'G3': ['GAP', 'XS', 'RRV'], 'G4': ['B'], 'G5': ['XS', 'B', 'LEAD'], 'G6': ['EXT'],
        'G7': ['V', 'XS'], 'G8': ['B', 'EBAR'], 'G9': ['B', 'IDX'], 'G10': []}
DESC = {
    'G1': 'внешние as-of: RVI (18:50), RV_m IMOEX, RV_m рынка d0, PC1–PC3 глобальной вол. (Yahoo — лаг доступности 1 в матрице)',
    'G2': 'календарь окна [t+1, t+h] h=1/5/22: будни, число событий ЦБ/экспираций/FOMC, дней до ближайшего; known_from ≤ t; до 2014 — NaN',
    'G3': 'ночной гэп, доля RV последнего часа (крейт), RRV PK/GK d/w/m (Д17.2) — день d',
    'G4': 'скачки: ln BV, J/RV, число дней J>0 и J/RV≥0,25 за 5/22, дней с последнего, сумма J за 22 — окна включают день d',
    'G5': 'кросс-секция (крейт r2_xs): корреляция 17 тикеров d/w/m, СКО ln RV (день, неделя), ln RV к среднему, ln RV SBER/LKOH/GAZP дня d',
    'G6': 'RV USDRUB (CETS) и ближнего BR d/w/m, срез 18:40, день d (as-of, протяжка не дольше 10 дней)',
    'G7': 'ликвидность: ln объёма, ln оборота, оборот к среднему 22, Amihud d/w/m (крейт)',
    'G8': 'ē_5/ē_10 (тикер, рынок), серия знака — OOS HAR-J h=1 d0 (переоценка 1 января); ē Д23 (финалист, с 2016)',
    'G9': 'ln RV_d/RV_w, ln RV_w/RV_m, ln RVI²/RV_m, ln RV_m/RV_m(−22) — d0',
    'G10': 'день недели, месяц, день месяца, будних дней до конца месяца, последний месяц квартала',
}


#: v2: столбцы, заблокированные O по п. 2.6 (PR #167). Ошибки в коде нет (stage razbor), максимум на +1 — свойство ряда; столбцы убраны.
DROP = {'G4': ['g4_jbig_n22'], 'G7': ['g7_ln_amihud_w', 'g7_ln_amihud_m']}


def build_all(I, groups=GROUPS, drop=True):
    out = {gk: BUILD[gk](I) for gk in groups}
    if drop:
        for gk, cols in DROP.items():
            if gk in out:
                out[gk] = out[gk].drop(columns=cols)
    return out


def stage_build(a):
    OUT.mkdir(parents=True, exist_ok=True); COMB.parent.mkdir(parents=True, exist_ok=True); DOCS.mkdir(parents=True, exist_ok=True)
    I = load_inputs()
    F = build_all(I)
    rows, comb = [], I['KEYS'].copy()
    for gk, df in F.items():
        assert len(df) == len(I['KEYS']) and (df.date.values == I['KEYS'].date.values).all() and (df.ticker.values == I['KEYS'].ticker.values).all()
        cols = [c for c in df if c not in ('date', 'ticker')]
        df = df[['date', 'ticker'] + cols].copy()
        for c in cols:
            df[c] = df[c].astype('float64').replace([np.inf, -np.inf], np.nan)
            comb[c] = df[c].values
        p = OUT / f'{gk}.parquet'
        df.to_parquet(p, index=False)
        dev = df[(df.date >= pd.Timestamp('2017-01-01')) & (df.date <= LAST_DEV)]
        pre = df[df.date < FIRST14]
        rows.append(dict(group=gk, file=str(p.relative_to(MAIN)), sha256=sha(p), rows=len(df), cols=len(cols),
                         date_min=str(df.date.min().date()), date_max=str(df.date.max().date()),
                         nan_share_2017_2021=round(float(dev[cols].isna().mean().mean()), 4),
                         nan_share_pre_2014_06_09=round(float(pre[cols].isna().mean().mean()), 4) if len(pre) else np.nan,
                         columns=' '.join(cols), opisanie=DESC[gk]))
    comb.to_parquet(COMB, index=False)
    rows.append(dict(group='ALL', file=str(COMB.relative_to(MAIN)), sha256=sha(COMB), rows=len(comb), cols=comb.shape[1] - 2,
                     date_min=str(comb.date.min().date()), date_max=str(comb.date.max().date()), columns='', opisanie='все группы G1–G10 на ключах v3h'))
    Path(str(COMB) + '.sha256').write_text(f'{sha(COMB)}  {COMB.name}\n')
    X = pd.DataFrame(rows)
    X.to_csv(OUT / 'INDEX_F.csv', index=False)
    X.drop(columns=['columns']).to_csv(DOCS / 'INDEX_F.csv', index=False)
    if '_g8_coefs' in I:
        I['_g8_coefs'].to_csv(DOCS / 'g8_harj_coefs.csv', index=False)
    # yaml группы -> столбцы (для r2_core: features: [G0, G1, …])
    L = ['# R2, агент F: группы признаков кластера 5 (schema_v1: features). Строится scripts/analysis/r2_features.py --stage build.',
         '# Информационное множество d0: строка (d, ticker) — данные по день d включительно; RV_d до 18:50; без shift(1).',
         '# Ключи — features_pit_v3h (ISS_start); history_start = 2014 -> строки date >= 2014-06-09.',
         'agent: F', 'version: features_r2_v3', 'changed_v3: "G1 (ln_rvm_imoex: флаг imoex_low_bars; строки внешних рядов как harx), G9 ln_rvi2_m — как R1"', 'removed_v2: [' + ', '.join(sum(DROP.values(), [])) + ']  # блок O (PR #167), разбор — stage razbor', f'combined: {{file: {COMB.relative_to(MAIN)}, sha256: {sha(COMB)}}}',
         'key: [date, ticker]', 'history_start: {"2014": "date >= 2014-06-09 (ключи v3)", ISS_start: "все строки (v3h, с 2011-12-08)"}',
         'groups:', '  G0: {file: data/regen/features_pit_v3.parquet, note: "база; в версии d0 — у агента S/L (rv_d, rv_w, rv_m, jump_lag* там сдвинуты на 1)"}']
    for r in rows[:-1]:
        L.append(f"  {r['group']}:")
        L.append(f"    file: {r['file']}")
        L.append(f"    sha256: {r['sha256']}")
        L.append(f"    note: \"{r['opisanie']}\"")
        L.append(f"    columns: [{', '.join(r['columns'].split())}]")
    YAML.parent.mkdir(parents=True, exist_ok=True)
    YAML.write_text('\n'.join(L) + '\n')
    print(X.drop(columns=['columns', 'opisanie']).to_string())


# ================================================================ check: детектор и проверка сдвига
def perturb(I, t, rng):
    """Числовые входы с датой > t -> бутстрап значений до t × {0,5; 1; 3}; события календаря, объявленные после t, — в случайные даты > t."""
    t = pd.Timestamp(t)
    J = dict(I)
    for k in ('B', 'V', 'IDX', 'XS', 'RRV', 'GAP', 'EXT', 'LEAD', 'EBAR'):
        d = I[k].copy()
        if not len(d):
            continue
        f = (d.date > t).values
        for c in d.columns:
            if c in ('date', 'ticker', 'ser', 'sday') or not np.issubdtype(d[c].dtype, np.number):
                continue
            pool = d.loc[~f, c].dropna().values.astype('float64')
            if len(pool) and f.any():
                d[c] = d[c].astype('float64')
                d.loc[f, c] = rng.choice(pool, f.sum()) * rng.choice([0.5, 1.0, 3.0], f.sum())
        J[k] = d
    ev = {}
    for ty, E in I['EV'].items():
        E = E.copy()
        f = (E.known_from > t).values  # объявлены после t: дата события — случайная после t, known_from не меняется (> t)
        E.loc[f, 'date'] = t + pd.to_timedelta(rng.integers(1, 400, f.sum()), unit='D')
        ev[ty] = E
    J['EV'] = ev
    return J


def leakify(I, cal):
    """Подложенная утечка: значения дня d+1 попадают в строку d (дата каждой записи входа сдвигается на торговый день назад);
    календарь — без условия known_from ≤ t."""
    prev = pd.Series(cal[:-1], index=cal[1:])
    J = dict(I)
    for k in ('B', 'V', 'IDX', 'XS', 'RRV', 'GAP', 'EXT', 'LEAD', 'EBAR'):
        d = I[k].copy()
        if not len(d):
            continue
        d['date'] = d['date'].map(prev)
        d = d.dropna(subset=['date'])
        if k in ('B', 'V'):  # таблицы на ключах: вернуть к ключам
            d = I['KEYS'].merge(d, on=['date', 'ticker'], how='left')
        if k == 'IDX':
            d = I['IDX'][['date']].merge(d, on='date', how='left')
        J[k] = d
    J['_cal_leak'] = True
    return J


def diff_le(A, Bx, t):
    m = (A.date <= pd.Timestamp(t)).values
    cols = [c for c in A if c not in ('date', 'ticker')]
    a, b = A.loc[m, cols].to_numpy('float64'), Bx.loc[m, cols].to_numpy('float64')
    nanmis = int((np.isnan(a) != np.isnan(b)).sum())
    with np.errstate(invalid='ignore'):
        d = np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b))
    d = np.nan_to_num(d, nan=1.0)
    return float(d.max()) if d.size else 0.0, nanmis, int((d > 0).sum())


def shift_table(F, I, gk):
    """Спирмен признака строки d с ln RV_{d+s}, s ∈ {−2, −1, 0, +1}, только 2017–2021, по тикерам, медиана; RV ≥ 2022 не участвует."""
    B = I['B'][['date', 'ticker', 'rv']].copy()
    B = B[B.date <= LAST_DEV].sort_values(['ticker', 'date'])
    B['lnrv'] = lnc(B.rv)
    for s in (-2, -1, 0, 1):
        B[f'y{s}'] = B.groupby('ticker')['lnrv'].shift(-s)
    M = F.merge(B.drop(columns=['rv', 'lnrv']), on=['date', 'ticker'])
    cols = [c for c in F if c not in ('date', 'ticker')]
    # приращение признака x_d − x_{d−1} по строкам тикера: для окон w/m (максимум уровня — в середине окна) последний известный день
    # виден по приращению — оно несёт RV дня d
    M = M.sort_values(['ticker', 'date'])
    for c in cols:
        M[f'_d_{c}'] = M.groupby('ticker')[c].diff()
    M = M[(M.date >= pd.Timestamp('2017-01-01')) & (M.date <= LAST_DEV)]

    def prof(c):
        rho = {}
        for s in (-2, -1, 0, 1):
            vals = []
            for t, g in M.groupby('ticker'):
                z = g[[c, f'y{s}']].dropna()
                if len(z) > 100 and z[c].nunique() > 1:
                    vals.append(z[c].rank().corr(z[f'y{s}'].rank()))
            rho[s] = float(np.nanmedian(vals)) if vals else np.nan
        ab = {s: abs(v) for s, v in rho.items() if np.isfinite(v)}
        return rho, (max(ab, key=ab.get) if ab else None)

    rows = []
    for c in cols:
        rho, am = prof(c)
        drho, dam = prof(f'_d_{c}')
        rows.append(dict(group=gk, col=c, **{f'rho_{s}': round(rho[s], 4) for s in rho}, argmax=am,
                         **{f'drho_{s}': round(drho[s], 4) for s in drho}, d_argmax=dam))
    return pd.DataFrame(rows)


def group_forecast_shift(F, I, gk):
    """Проверка п. 2.6 протокола на ряде прогноза группы: ŷ_d = OLS ln RV_{d+1} на признаках группы (обучение — строки 2014-06-09…2016,
    цель ≤ 2016-12-31; NaN -> среднее обучения), затем Спирмен ŷ_d с ln RV_{d+s} на 2017–2021, медиана по тикерам; argmax должен быть 0."""
    B = I['B'][['date', 'ticker', 'rv']].copy()
    B = B[B.date <= LAST_DEV].sort_values(['ticker', 'date'])
    B['lnrv'] = lnc(B.rv)
    g = B.groupby('ticker')
    B['tgt'] = g['lnrv'].shift(-1); B['tdate'] = g['date'].shift(-1)
    for s in (-2, -1, 0, 1):
        B[f'y{s}'] = g['lnrv'].shift(-s)
    M = F.merge(B.drop(columns=['rv']), on=['date', 'ticker'])
    cols = [c for c in F if c not in ('date', 'ticker')]
    tr = (M.date >= FIRST14) & (M.tdate <= pd.Timestamp('2016-12-31')) & M.tgt.notna()
    mu = M.loc[tr, cols].mean()
    keep = [c for c in cols if np.isfinite(mu[c])]
    X = M[keep].fillna(mu[keep]).to_numpy('float64')
    sd = X[tr.values].std(axis=0); sd[sd == 0] = 1
    Z = (X - X[tr.values].mean(axis=0)) / sd
    A = np.column_stack([np.ones(len(Z)), Z])
    beta = np.linalg.lstsq(A[tr.values], M.loc[tr, 'tgt'].values, rcond=None)[0]
    M['yhat'] = A @ beta
    D = M[(M.date >= pd.Timestamp('2017-01-01')) & (M.date <= LAST_DEV)]
    rho = {}
    for s in (-2, -1, 0, 1):
        v = [z['yhat'].rank().corr(z[f'y{s}'].rank()) for _, z in D.groupby('ticker') if z[f'y{s}'].notna().sum() > 100]
        rho[s] = float(np.nanmedian(v))
    return dict(group=gk, n_cols=len(keep), **{f'rho_{s}': round(rho[s], 4) for s in rho}, argmax=max(rho, key=lambda s: abs(rho[s])))


def stage_check(a):
    DOCS.mkdir(parents=True, exist_ok=True)
    I = load_inputs()
    cal = np.sort(I['KEYS'].date.unique())
    rng = np.random.default_rng(20261026)
    base = build_all(I)
    LK = leakify(I, pd.DatetimeIndex(cal))
    leakb = build_all(LK)
    res = []
    for t in CUTS:
        P = perturb(I, t, rng)
        pb = build_all(P)
        pl = build_all(leakify(P, pd.DatetimeIndex(cal)))
        for gk in GROUPS:
            d0, nm0, n0 = diff_le(base[gk], pb[gk], t)
            d1, nm1, n1 = diff_le(leakb[gk], pl[gk], t)
            res.append(dict(group=gk, cut=t, inputs='+'.join(USES[gk]) or '—', clean_max_abs=d0, clean_nan_mismatch=nm0, clean_cells=n0,
                            leak_max_abs=d1, leak_nan_mismatch=nm1, leak_cells=n1,
                            verdict=('ок' if (d0 == 0 and nm0 == 0 and (n1 > 0 or not USES[gk])) else 'ПРОВАЛ')))
        log('детектор, срез', t)
    # крейт: прогон со срезом (бары позже t отброшены) против полного — строки ≤ t побайтно
    full = pd.read_parquet(WORK / 'xs_full.parquet'); full['date'] = pd.to_datetime(full.date)
    for t in CUTS:
        c = pd.read_parquet(WORK / f'xs_cut_{t}.parquet'); c['date'] = pd.to_datetime(c.date)
        f = full[full.date <= pd.Timestamp(t)].reset_index(drop=True)
        c = c[c.date <= pd.Timestamp(t)].reset_index(drop=True)
        same = len(f) == len(c) and all(np.array_equal(f[k].to_numpy(), c[k].to_numpy(), equal_nan=True) for k in f.columns if k not in ('date', 'ticker'))
        res.append(dict(group='крейт r2_xs', cut=t, inputs='бары ISS', clean_max_abs=0.0 if same else np.nan, clean_nan_mismatch=0,
                        clean_cells=0 if same else -1, leak_max_abs=np.nan, leak_nan_mismatch=0, leak_cells=np.nan,
                        verdict='ок (побайтно)' if same else 'ПРОВАЛ'))
    R = pd.DataFrame(res)
    R.to_csv(DOCS / 'leak_detector.csv', index=False)
    print(R.to_string())
    S = pd.concat([shift_table(base[gk], I, gk) for gk in GROUPS], ignore_index=True)
    SL = pd.concat([shift_table(leakb[gk], I, gk) for gk in GROUPS], ignore_index=True)
    S['leak_argmax'] = SL['argmax'].values
    S['leak_d_argmax'] = SL['d_argmax'].values
    S['deterministic'] = S.group.isin(['G2', 'G10'])
    # признак: заглядывания нет, если argmax ≤ 0 (окна w/m и ē дают максимум в середине прошлого окна, s = −2/−1);
    # подложенная утечка сдвигает argmax вправо
    S['leak_shift'] = S['leak_argmax'] - S['argmax']
    mx = S[[f'rho_{s}' for s in (-2, -1, 0, 1)] + [f'drho_{s}' for s in (-2, -1, 0, 1)]].abs().max(axis=1)
    S['status'] = np.select(
        [S.deterministic, mx < 0.05, (S.argmax == 0) & (S.d_argmax <= 0), (S.argmax <= 0) & (S.d_argmax == 0),
         (S.argmax < 0) & (S.d_argmax < 0)],
        ['календарь', 'слабый (|ρ| < 0,05)', 'ок', 'ок (по приращению)', 'прошлое, без заглядывания'], default='ФЛАГ')
    S.to_csv(DOCS / 'shift_check.csv', index=False)
    summ = S.groupby('group', sort=False).agg(cols=('col', 'size'), at0=('argmax', lambda x: int((x == 0).sum())),
                                              at_minus=('argmax', lambda x: int((x < 0).sum())), at_plus=('argmax', lambda x: int((x == 1).sum())),
                                              leak_moved_right=('leak_shift', lambda x: int((x > 0).sum())),
                                              leak_at_plus=('leak_argmax', lambda x: int((x == 1).sum())),
                                              d_at0=('d_argmax', lambda x: int((x == 0).sum())),
                                              d_at_plus=('d_argmax', lambda x: int((x == 1).sum())),
                                              leak_d_at_plus=('leak_d_argmax', lambda x: int((x == 1).sum())))
    summ.to_csv(DOCS / 'shift_check_summary.csv')
    print(summ.to_string())
    print(S[(S.argmax == 1)].to_string())
    print(S.status.value_counts().to_string())
    print(S[S.status == 'ФЛАГ'].to_string())
    # G1 без PC1–PC3: PC строятся по Yahoo с лагом доступности 1 (закрытие США дня d — после 18:50), их последний известный день — d−1
    sub = {'G1_bez_PC': ('G1', ['g1_ln_rvi2', 'g1_ln_rvm_imoex', 'g1_ln_rvm_mkt'])}
    gl = GROUPS + list(sub)
    pick = lambda F, k: F[k] if k in F else F[sub[k][0]][['date', 'ticker'] + sub[k][1]]
    G = pd.DataFrame([group_forecast_shift(pick(base, gk), I, gk) for gk in gl])
    GL = pd.DataFrame([group_forecast_shift(pick(leakb, gk), I, gk) for gk in gl])
    G['leak_argmax'] = GL['argmax'].values
    for s in (-2, -1, 0, 1):
        G[f'leak_rho_{s}'] = GL[f'rho_{s}'].values
    G['deterministic'] = G.group.isin(['G2', 'G10'])
    G['verdict'] = np.where(G.deterministic, 'календарь (не блок)', np.where((G.argmax == 0) & (G.leak_argmax == 1), 'ок', 'БЛОК'))
    G.to_csv(DOCS / 'shift_check_groups.csv', index=False)
    print(G.to_string())


# ================================================================ razbor: 3 столбца, заблокированные O (PR #167)
def stage_razbor(a):
    """1) пересчёт «в лоб» по сырым рядам только с датами ≤ d на 3000 случайных строк 2017–2021; 2) детектор на этих столбцах;
    3) профили сдвига: столбец, подложенная утечка (окно d−w+2…d+1), leakify, разложение Amihud = |r_d| / оборот_d."""
    I = load_inputs()
    rng = np.random.default_rng(20261026)
    cols = sum(DROP.values(), [])
    F = {gk: BUILD[gk](I) for gk in DROP}
    X = pd.concat([F['G4'][['date', 'ticker', 'g4_jbig_n22']], F['G7'][['g7_ln_amihud_w', 'g7_ln_amihud_m']]], axis=1)
    rows = []
    # 1) пересчёт в лоб
    XS = I['XS'].sort_values(['ticker', 'date'])
    B = I['B'].sort_values(['ticker', 'date'])
    dev = X[(X.date >= pd.Timestamp('2017-01-01')) & (X.date <= LAST_DEV)]
    smp = dev.sample(3000, random_state=1)
    err = {c: 0.0 for c in cols}
    for _, r in smp.iterrows():
        xs = XS[(XS.ticker == r.ticker) & (XS.date <= r.date)]['amihud']
        for c, w, mp in (('g7_ln_amihud_w', 5, 3), ('g7_ln_amihud_m', 22, 13)):
            v = xs.iloc[-w:]
            m = v.mean() if v.notna().sum() >= mp else np.nan
            ref = np.log(m) if np.isfinite(m) and m > 0 else np.nan
            got = r[c]
            err[c] = max(err[c], 0.0 if (np.isnan(ref) and np.isnan(got)) else abs(ref - got) if np.isfinite(ref - got) else 1.0)
        b = B[(B.ticker == r.ticker) & (B.date <= r.date)].iloc[-22:]
        ind = ((b.j >= 0.25 * b.rv) & (b.j > 0)).astype(float).where(b.j.notna())
        ref = ind.sum() if ind.notna().sum() >= 13 else np.nan
        got = r['g4_jbig_n22']
        err['g4_jbig_n22'] = max(err['g4_jbig_n22'], 0.0 if (np.isnan(ref) and np.isnan(got)) else abs(ref - got))
    for c in cols:
        rows.append(dict(proverka='пересчёт в лоб по датам ≤ d (3000 строк 2017–2021)', col=c, znachenie=err[c]))
    # 2) детектор на этих столбцах
    cal = pd.DatetimeIndex(np.sort(I['KEYS'].date.unique()))
    LK = {gk: BUILD[gk](leakify(I, cal)) for gk in DROP}
    for t in CUTS:
        P = perturb(I, t, rng)
        PB = {gk: BUILD[gk](P) for gk in DROP}
        PL = {gk: BUILD[gk](leakify(P, cal)) for gk in DROP}
        for gk, cc in DROP.items():
            for c in cc:
                d0 = diff_le(F[gk][['date', 'ticker', c]], PB[gk][['date', 'ticker', c]], t)
                d1 = diff_le(LK[gk][['date', 'ticker', c]], PL[gk][['date', 'ticker', c]], t)
                rows.append(dict(proverka=f'детектор, срез {t}: без утечки max|Δ| / ячеек', col=c, znachenie=d0[0], yacheek=d0[2]))
                rows.append(dict(proverka=f'детектор, срез {t}: подложенная утечка max|Δ| / ячеек', col=c, znachenie=d1[0], yacheek=d1[2]))
    R = pd.DataFrame(rows)
    R.to_csv(DOCS / 'razbor_3stolbca.csv', index=False)
    print(R.to_string())
    # 3) профили сдвига
    P = X.copy()
    g = P.sort_values(['ticker', 'date']).groupby('ticker')
    for c in cols:  # подложенная утечка: окно сдвинуто на день вперёд (d−w+2 … d+1)
        P[f'{c}__utechka_okno+1'] = g[c].shift(-1)
    for c in cols:
        P[f'{c}__leakify'] = pd.concat([LK['G4'], LK['G7']], axis=1)[c].values
    k = on_keys(I, I['XS'], ['amihud', 'value'])
    absr = k.amihud * k.value / 1e9  # |ln C_d/C_{d−1}|
    P['ln_amihud_d'] = np.log(k.amihud.where(k.amihud > 0)).values
    P['ln_abs_r_d'] = np.log(absr.where(absr > 0)).values
    P['minus_ln_oborot_d'] = -np.log(k.value.where(k.value > 0)).values
    S = shift_table(P, I, 'razbor')
    S.to_csv(DOCS / 'razbor_3stolbca_sdvig.csv', index=False)
    print(S[['col', 'rho_-2', 'rho_-1', 'rho_0', 'rho_1', 'argmax', 'drho_-2', 'drho_-1', 'drho_0', 'drho_1', 'd_argmax']].to_string())


# ================================================================ sverka_r1: G1 против слагаемых R1 (harx.TERMS на v3), только ≤ 2021
def stage_sverka_r1(a):
    import harx as HX
    A, harj = HX.load()
    A['date'] = pd.to_datetime(A['date'])
    r1 = A[['date', 'ticker']].copy()
    for t in ('ln_rvi2', 'ln_rvm_imoex', 'pc1'):
        r1[t] = HX.TERMS[t](A).values
    rows = []
    for ver in ('v2', 'v3'):
        p = REG / 'R2' / 'features' / f'features_r2_{ver}.parquet'
        F = pd.read_parquet(p, columns=['date', 'ticker', 'g1_ln_rvi2', 'g1_ln_rvm_imoex', 'g1_pc1'])
        F['date'] = pd.to_datetime(F['date'])
        m = r1.merge(F, on=['date', 'ticker'])
        m = m[m.date <= LAST_DEV]
        for t, c in (('ln_rvi2', 'g1_ln_rvi2'), ('ln_rvm_imoex', 'g1_ln_rvm_imoex'), ('pc1', 'g1_pc1')):
            a_, b_ = m[t].values, m[c].values
            ok = np.isfinite(a_) & np.isfinite(b_)
            d = np.abs(a_[ok] - b_[ok])
            rows.append(dict(versiya=ver, r1=t, f=c, n=int(ok.sum()), median_abs=float(np.median(d)), max_abs=float(d.max()),
                             dolya_1e12=float((d <= 1e-12).mean()), nan_tolko_r1=int((~np.isfinite(a_) & np.isfinite(b_)).sum()),
                             nan_tolko_f=int((np.isfinite(a_) & ~np.isfinite(b_)).sum())))
    R = pd.DataFrame(rows)
    R.to_csv(DOCS / 'sverka_G1_R1.csv', index=False)
    print(R.to_string())


# ================================================================ nobs: ISS_start
def stage_nobs(a):
    DOCS.mkdir(parents=True, exist_ok=True)
    K = pd.read_parquet(V3H, columns=['date', 'ticker']); K['date'] = pd.to_datetime(K.date)
    rows = []
    for hs, KK in (('2014', K[K.date >= FIRST14]), ('ISS_start', K)):
        KK = KK.sort_values(['ticker', 'date']).reset_index(drop=True)
        for h in (1, 5, 22):
            tend = KK.groupby('ticker')['date'].shift(-h)  # дата последнего дня цели (по ключам, значения RV не читаются)
            for k in range(1, 11):
                vy = 2015 + k
                tr = (KK.date.dt.year < vy) & tend.notna() & (tend.dt.year < vy)  # эмбарго h: цель целиком до года валидации
                rows.append(dict(history_start=hs, h=h, step=k, val_year=vy, n_train=int(tr.sum()), n_tickers=int(KK.loc[tr, 'ticker'].nunique()),
                                 first_date=str(KK.loc[tr, 'date'].min().date()) if tr.any() else '',
                                 n_val=int(((KK.date.dt.year == vy) & tend.notna()).sum()) if vy <= 2021 else np.nan))
    N = pd.DataFrame(rows)
    p = N.pivot_table(index=['h', 'step', 'val_year'], columns='history_start', values='n_train').reset_index()
    p['plus'] = p['ISS_start'] - p['2014']
    p['plus_pct'] = (100 * p['plus'] / p['2014']).round(1)
    N.to_csv(DOCS / 'nobs_po_shagam.csv', index=False)
    p.to_csv(DOCS / 'nobs_po_shagam_svodka.csv', index=False)
    print(p.to_string())
    by = K.assign(y=K.date.dt.year).groupby(['y']).size()
    print('строк ключей по годам:', by.to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['bars', 'daily', 'build', 'check', 'nobs', 'razbor', 'sverka_r1', 'all'])
    ap.add_argument('--bin', default=None)
    a = ap.parse_args()
    for s in (['bars', 'daily', 'build', 'check', 'nobs'] if a.stage == 'all' else [a.stage]):
        log('стадия', s)
        globals()[f'stage_{s}'](a)


if __name__ == '__main__':
    main()
