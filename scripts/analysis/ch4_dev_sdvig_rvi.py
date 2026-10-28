#!/usr/bin/env python3
"""S0, дополнение (поручение дирижёра 26.10): проверка сдвига ряда RVI, входа семьи D6 / D6lag1 (fastbt.s9.S9.d_families:
ratio = σ_pred·√252·100 / RVI(lag), ранг по прошлому окну 252; D6 — lag 0, D6lag1 — RVI дня d−1).

Ряд — data/fastbt/s9/rvi_1850.parquet (fastbt.s9.RVI; fastbt.s9.build_sources: close_RVI из features_pit_v3, dropna, первое значение
на дату). close_RVI — закрытие индекса RVI по 10-минутным барам с begin ≤ 18:50 дня d (m0_pit_lib / create_external_final.py,
docs/otchety/2026-10-25_vneshnie_dannye.md): срез основной сессии дня d, вечерняя сессия не входит.
На бары тикера ряд кладётся как fastbt.s9.align: последнее значение с датой ≤ даты бара.

Проверки (только ≤ 2021):
- сдвиг: та же ch4_dev_sigma.shift_row (r2_loyo.sdvig + r2_features.shift_table): Спирмен ln RVI²_d с ln RV_{d+k} тикера, k = −2…+1,
  медиана по тикерам, K = {0}, симметричное ε = 0,01; RVI — котировка, не оконное среднее RV: решает уровень (как у модельных σ),
  приращение — рядом. Периоды 2017–2021 и 2018–2021. Справочно — цель ln RV IMOEX (rv_IMOEX, срез 18:50) и вход D6lag1 (RVI_{d−1})
  с последним известным днём K = {−1};
- контроли −1 (RVI_d := RVI_{d−1}) и +1 (RVI_d := RVI_{d+1}) — обязаны блокироваться при K = {0};
- детектор: ряд rvi_1850 = пересборка из features_pit_v3 (≤ 2021) побитно; признак D6 / D6lag1 (ранг W252 отношения σ hyb_h5 / RVI):
  RVI после среза × U(0,5; 2) — строки ≤ среза не меняются, после — меняются; подлог RVI_d := RVI_{d+1} виден ≤ среза.
Выход — дописывается в data/regen/ch4_dev/proverka_sdvig.csv (cel) и proverka_detektor.csv, SHA — в SHA256.txt.

python3 scripts/analysis/ch4_dev_sdvig_rvi.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch4_dev_sigma as S  # noqa: E402

OUT = S.OUT
RVI = S.MAIN / 'data' / 'fastbt' / 's9' / 'rvi_1850.parquet'
CEL_DAY = 'RV_{d+k} дневная'
CEL_IMOEX = 'RV IMOEX_{d+k} дневная (справочно)'
TAG = 'rvi'


def on_bars(rvi, keys):
    """fastbt.s9.align: на строку (date, ticker) — последнее RVI с датой ≤ date."""
    r = rvi.sort_values('date')
    out = []
    for tk, g in keys.groupby('ticker'):
        m = pd.merge_asof(g.sort_values('date'), r, on='date', direction='backward')
        out.append(m)
    d = pd.concat(out).rename(columns={'close_RVI': 'rvi'})
    return d.dropna(subset=['rvi']).sort_values(['date', 'ticker']).reset_index(drop=True)


def main():
    import r2_loyo as O
    import r2_features as RF
    from fastbt.s9_stats import minp, past_rank
    rvi = S.read_dev(RVI, ['date', 'close_RVI'])                       # фильтр ≤ 2021 — первой операцией
    f = S.read_dev(S.REGEN / 'features_pit_v3.parquet', ['date', 'ticker', 'close_RVI', 'rv_IMOEX', 'close_IMOEX'])
    # пересборка кодом fastbt.s9.build_sources (≤ 2021)
    fb = f[(f['date'] >= '2017-01-03') & (f['date'] <= S.LAST)]
    re_ = fb.dropna(subset=['close_RVI']).groupby('date')['close_RVI'].first().reset_index()
    m = rvi.merge(re_, on='date', how='outer', suffixes=('', '_re'), indicator=True)
    same = bool((m['_merge'] == 'both').all() and np.array_equal(m['close_RVI'].values, m['close_RVI_re'].values))
    det = [dict(detektor='RVI: rvi_1850 = пересборка fastbt.s9.build_sources из features_pit_v3', ryad='rvi_1850', srez='—',
                strok_do=len(rvi), max_d_do=float(np.nanmax(np.abs(m['close_RVI'] - m['close_RVI_re']))), max_d_posle=np.nan,
                podlog_max_d_do=np.nan, ok=same,
                primechanie=f'{rvi.date.min().date()}…{rvi.date.max().date()}, строк {len(rvi)}, побитно {same}')]

    rv = O.rv_daily()
    rv['date'] = pd.to_datetime(rv['date'])
    S.check_dev(rv, 'rv_daily')
    keys = rv[(rv['date'] >= '2017-01-03')][['date', 'ticker']]
    base = on_bars(rvi, keys)
    series = {'rvi_1850': (base, 'ряд RVI (вход D6, D6lag1)'),
              'kontrol:rvi_1850_sdvig-1': (S.shift_ts(base, 'rvi', -1), 'контроль: RVI_d := RVI_{d−1} (сдвиг −1)'),
              'kontrol:rvi_1850_utechka+1': (S.shift_ts(base, 'rvi', +1), 'контроль: RVI_d := RVI_{d+1} (утечка +1)')}
    rows = []
    for name, (d, rol) in series.items():
        for start in ('2017-01-01', '2018-01-01'):
            r = S.shift_row(O, RF, rv, name, d, 'rvi', 'model', rol, start)
            r['cel'] = CEL_DAY
            rows.append(r)
    # справочно: вход D6lag1 = RVI_{d−1}, последний известный день −1
    lag1 = S.shift_ts(base, 'rvi', -1)
    for start in ('2017-01-01', '2018-01-01'):
        d = lag1[lag1['date'] >= start]
        s = O.sdvig(d[['date', 'ticker']].assign(rv_pred=d['rvi'] ** 2), rv, K=(-1,))
        rows.append(dict(ryad='spravka:rvi_1850_lag1 (вход D6lag1)', rol='справочно: K = {−1}', tip='model',
                         period=f'{start[:4]}–2021', n_tikerov=s['n_tikerov'], n_strok=s['n_strok'],
                         **{f'corr_k{k:+d}': s[f'corr_s{k:+d}'] for k in (-2, -1, 0, 1)}, argmax_uroven=s['argmax'],
                         uroven_ok=s['ok'], ok=bool(s['ok']), reshenie=(s['prichina'] or s['pometka'] or 'по уровню, K = {−1}'),
                         eps=S.EPS, vremya=S.now(), cel=CEL_DAY))
    # справочно: цель — RV IMOEX (срез 18:50), один «тикер»
    im = f.dropna(subset=['rv_IMOEX']).groupby('date')['rv_IMOEX'].first().reset_index()
    rvm = im.assign(ticker='IMOEX', rv=im['rv_IMOEX'] ** 2)[['date', 'ticker', 'rv']]
    bm = rvi.merge(rvm[['date']], on='date').assign(ticker='IMOEX').rename(columns={'close_RVI': 'rvi'})
    for name, d in (('spravka:rvi_1850', bm), ('spravka:kontrol:rvi_1850_sdvig-1', S.shift_ts(bm, 'rvi', -1)),
                    ('spravka:kontrol:rvi_1850_utechka+1', S.shift_ts(bm, 'rvi', +1))):
        r = S.shift_row(O, RF, rvm, name, d, 'rvi', 'model', 'справочно: цель RV IMOEX', '2017-01-01')
        r['cel'] = CEL_IMOEX
        rows.append(r)
    # дополнительная проверка момента среза (сила): приращение ln RVI_d с доходностью IMOEX дня d+k (close_IMOEX, срез 18:50).
    # RVI (implied vol) движется вместе с индексом в тот же день (обратная связь), так что |ρ| максимален на k = 0, если RVI и
    # close_IMOEX сняты в один момент дня d; у сдвига −1 максимум переходит на k = −1, у утечки +1 — на k = +1.
    cl = f.dropna(subset=['close_IMOEX']).groupby('date')['close_IMOEX'].first() if 'close_IMOEX' in f else None
    if cl is not None:
        ret = np.log(cl).diff()
        for name, rs in (('rvi_1850', rvi), ('kontrol:rvi_1850_sdvig-1', rvi.assign(close_RVI=rvi['close_RVI'].shift(1))),
                         ('kontrol:rvi_1850_utechka+1', rvi.assign(close_RVI=rvi['close_RVI'].shift(-1)))):
            x = rs.set_index('date')['close_RVI'].sort_index()
            dx = np.log(x).diff()
            for start in ('2017-01-01', '2018-01-01'):
                rho = {}
                for k in (-2, -1, 0, 1):
                    rk = ret.shift(-k).reindex(dx.index)
                    z = pd.DataFrame({'a': dx, 'b': rk})
                    z = z[(z.index >= start) & (z.index <= S.LAST)].dropna()
                    rho[k] = float(z['a'].corr(z['b'], method='spearman'))
                am = max(rho, key=lambda k: abs(rho[k]))
                marg = abs(rho[0]) - max(abs(v) for k, v in rho.items() if k != 0)
                okk = bool(am == 0 and marg > S.EPS)
                why = ('|ρ| максимален на k = 0' if okk else f'|ρ| максимален на k = {am:+d}') + f'; запас k=0 над прочими {marg:+.4f}'
                rows.append(dict(ryad=f'sila:{name}', rol='доп. проверка момента среза: Δ ln RVI_d с доходностью IMOEX_{d+k}',
                                 tip='prirashchenie', period=f'{start[:4]}–2021', n_tikerov=1, n_strok=len(z),
                                 **{f'corr_k{k:+d}': rho[k] for k in rho}, argmax_prirascheniya=am, ok=okk, reshenie=why,
                                 eps=S.EPS, vremya=S.now(), cel='доходность IMOEX_{d+k} (Δ ln RVI, срез 18:50)'))
    for r in rows:
        print(f'  {r["ryad"]} {r["period"]} [{r["cel"][:8]}]: ' + ' '.join(
            f'k{k:+d} {r[f"corr_k{k:+d}"]:.4f}' for k in (-2, -1, 0, 1))
              + f' argmax {r.get("argmax_uroven")} прир. {r.get("argmax_prirascheniya")} ok {r["ok"]}', flush=True)

    # детектор: признак D6 / D6lag1 на σ hyb_h5 (как в fastbt.s9: ratio = S·√252·100 / RVI(lag), ранг W252)
    sig = S.read_dev(OUT / 'sigma_hyb_h5_dev.parquet', ['date', 'ticker', 'sigma_pred_h5'])
    sig = sig.sort_values(['ticker', 'date']).reset_index(drop=True)
    pl_rvi = rvi.sort_values('date').assign(close_RVI=lambda x: x['close_RVI'].shift(-1).fillna(x['close_RVI']))

    def feat(rv_ser, lag):
        a = on_bars(rv_ser, sig[['date', 'ticker']]).sort_values(['ticker', 'date']).reset_index(drop=True)
        mm = sig.merge(a, on=['date', 'ticker'])
        out = {}
        for tk, g in mm.groupby('ticker'):
            R = g['rvi'].values.astype(float)
            if lag:
                R = np.r_[np.full(lag, np.nan), R[:-lag]]
            ratio = g['sigma_pred_h5'].values.astype(float) * np.sqrt(252) * 100 / R
            out[tk] = (g['date'].values, past_rank(np.ascontiguousarray(ratio), 252, minp(252)))
        return out
    for lag, fam in ((0, 'D6'), (1, 'D6lag1')):
        fb0, fl0 = feat(rvi, lag), feat(pl_rvi, lag)
        for i, cut in enumerate(S.CUTS):
            c = np.datetime64(pd.Timestamp(cut))
            rng = np.random.default_rng(S.SEED + 100 + i)
            fut = (rvi['date'] > pd.Timestamp(cut)).values
            rp = rvi.copy()
            rp.loc[fut, 'close_RVI'] = rp.loc[fut, 'close_RVI'].values * rng.uniform(0.5, 2.0, int(fut.sum()))
            plp = rp.sort_values('date').assign(close_RVI=lambda x: x['close_RVI'].shift(-1).fillna(x['close_RVI']))
            fp, flp = feat(rp, lag), feat(plp, lag)
            dpre = dpost = dpl = 0.0
            n = 0
            for tk in fb0:
                dts, b = fb0[tk]
                pre = dts <= c
                d1 = np.abs(b - fp[tk][1])
                d2 = np.abs(fl0[tk][1] - flp[tk][1])
                dpre = max(dpre, float(np.nanmax(d1[pre])) if pre.any() else 0.0)
                dpost = max(dpost, float(np.nanmax(d1[~pre])) if (~pre).any() else 0.0)
                dpl = max(dpl, float(np.nanmax(d2[pre])) if pre.any() else 0.0)
                n += int(pre.sum())
            # у D6lag1 подлог RVI_{d+1} на строке d превращается в RVI_d — информацию дня d, не утечку: его видимость
            # проверяется только у D6 (lag 0)
            ok = dpre == 0.0 and dpost > 0 and (dpl > 0 if lag == 0 else True)
            det.append(dict(detektor=f'RVI: признак {fam} (ранг W252 σ hyb_h5 / RVI), RVI после среза × U(0,5; 2)',
                            ryad=f'rvi_1850 → {fam}', srez=cut, strok_do=n, max_d_do=dpre, max_d_posle=dpost,
                            podlog_max_d_do=dpl, ok=bool(ok),
                            primechanie='подлог RVI_{d+1} виден' if dpl > 0 else 'подлог ≤ среза не виден'))

    new = pd.DataFrame(rows)
    fs = OUT / 'proverka_sdvig.csv'
    old = pd.read_csv(fs)
    old = old[~old['ryad'].str.contains('rvi_1850', regex=False)]
    pd.concat([old, new], ignore_index=True).to_csv(fs, index=False)
    fd = OUT / 'proverka_detektor.csv'
    oldd = pd.read_csv(fd)
    oldd = oldd[~oldd['detektor'].str.startswith('RVI:')]
    detf = pd.DataFrame(det)
    pd.concat([oldd, detf], ignore_index=True).to_csv(fd, index=False)

    lines = [ln for ln in (OUT / 'SHA256.txt').read_text().splitlines() if 'ch4_dev_sdvig_rvi.py' not in ln
             and 'rvi_1850.parquet' not in ln]
    lines = [f'{S.sha(fs)}  data/regen/ch4_dev/proverka_sdvig.csv  (проверка; + цель h=5, + RVI)' if 'proverka_sdvig.csv' in ln
             else f'{S.sha(fd)}  data/regen/ch4_dev/proverka_detektor.csv  (проверка; + RVI)' if 'proverka_detektor.csv' in ln
             else ln for ln in lines]
    lines.append(f'{S.sha(RVI)}  data/fastbt/s9/rvi_1850.parquet  (вход RVI, прочитано ≤ 2021)')
    lines.append(f'{S.sha(HERE / "ch4_dev_sdvig_rvi.py")}  scripts/analysis/ch4_dev_sdvig_rvi.py  (код: сдвиг и детектор RVI)')
    (OUT / 'SHA256.txt').write_text('\n'.join(lines) + '\n')

    dok = bool(detf['ok'].all())
    res = {}
    for nm, pre in (('уровень против RV тикера (правило)', ''), ('сила: Δ ln RVI против доходности IMOEX', 'sila:')):
        okm = bool(new.loc[new['ryad'] == f'{pre}rvi_1850', 'ok'].all())
        ct = new[new['ryad'].str.startswith(f'{pre}kontrol:')]
        res[nm] = (okm, bool((~ct['ok']).all()), ct.loc[ct['ok'], 'ryad'].unique().tolist())
        print(f'RVI, {nm}: ряд {"OK" if okm else "СДВИГ"}; контроли блокируются {res[nm][1]} '
              f'(не заблокированы: {res[nm][2]}); детектор {dok}')
    allok = all(v[0] and v[1] for v in res.values()) and dok
    return 0 if allok else 2


if __name__ == '__main__':
    os.environ.setdefault('NUMBA_CACHE_DIR', str(S.MAIN / 'data' / 'fastbt' / 'numba_cache'))
    sys.exit(main())
