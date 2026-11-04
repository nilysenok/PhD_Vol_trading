# -*- coding: utf-8 -*-
"""Owner item1, dev only. Run BEFORE any round5/6 candidate training."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
REG=ROOT/'data/regen'
TAB=ROOT/'docs/tables/gl3_r5_r6'
LAST=pd.Timestamp('2021-12-31')
READS=[]


def read(path,cols,filters=None):
    path=Path(path);assert 'test_locked' not in path.parts
    d=pd.read_parquet(path,columns=cols,filters=filters or [('date','<=',LAST)])
    d['date']=pd.to_datetime(d.date)
    assert d.date.max()<=LAST
    READS.append(dict(path=str(path.relative_to(ROOT)),columns=cols,n=len(d),last=str(d.date.max()),
                      values_sha256=hashlib.sha256(pd.util.hash_pandas_object(d,index=False).values.tobytes()).hexdigest()))
    return d


def render(d):
    def f(v):
        if isinstance(v,(list,dict)):return str(v)
        if pd.isna(v):return '—'
        if isinstance(v,float):return f'{v:.6g}'
        return str(v)
    return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+
                     ['| '+' | '.join(f(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)])


def main():
    TAB.mkdir(exist_ok=True)
    a=read(REG/'features_pit_v3.parquet',['date','ticker','rv','close_RVI'])
    a=a.sort_values(['ticker','date']).reset_index(drop=True)
    a['RV_d0']=a.rv
    a['RV_w0']=a.groupby('ticker').rv.transform(lambda s:s.rolling(5,min_periods=5).mean())
    a['RV_d_w']=a.RV_d0/a.RV_w0
    rank=a.groupby('ticker').rv.transform(lambda s:s.rolling(23,min_periods=23).apply(lambda z:np.mean(z[:-1]<=z[-1]),raw=True))
    a['RV_tercile']=np.minimum(1+np.floor(rank*3),3)
    assert a.groupby('date').close_RVI.nunique().max()<=1
    rvi=a.groupby('date').close_RVI.first().sort_index()
    median=rvi.expanding(min_periods=22).median().shift(1)
    a['RVI_high']=a.date.map((rvi>median).where(median.notna()))
    rr=rvi.rolling(253,min_periods=253).apply(lambda z:np.mean(z[:-1]<=z[-1]),raw=True)
    a['stress_RVI']=a.date.map(rr>2/3)
    for h in (1,5,22):a[f'tend{h}']=a.groupby('ticker').date.shift(-h)
    regimes=[];stress=[];bounds=[];outside=[];frames={};gates={}
    for h in (1,5,22):
        target=f'rv_target_h{h}' if h==1 else f'rv_target_mean_h{h}'
        # Crop target decoding to dates whose full horizon is inside dev.
        predicates=[]
        for tk,g in a.groupby('ticker'):
            predicates.append([('ticker','==',tk),('date','<=',g.loc[g[f'tend{h}'].notna(),'date'].max())])
        tr=read(REG/'features_pit_v3.parquet',['date','ticker',target],predicates)
        tr=tr.merge(a,on=['date','ticker'],how='left',validate='1:1')
        tr=tr[(tr[target]>1e-12)&tr[f'tend{h}'].notna()]
        d=read(REG/f'gl3_r3/dev/predictions_h{h}.parquet',
               ['date','ticker','year','rv_actual','rv_pred','har_corr'],
               [('date','>=',pd.Timestamp('2017-01-01')),('date','<=',LAST)])
        d=d.merge(a,on=['date','ticker'],how='left',validate='1:1')
        assert d.ticker.nunique()==17 and d.year.between(2017,2021).all()
        assert d[f'tend{h}'].max()<=LAST
        components={'финал':'rv_pred','HAR-J(б)':'har_corr'}
        for name,col in components.items():
            ratio=d.rv_actual/d[col];d['ratio_'+name]=ratio;d['loss_'+name]=ratio-np.log(ratio)-1
        states=[('RV нижний',d.RV_tercile==1),('RV средний',d.RV_tercile==2),('RV верхний',d.RV_tercile==3),
                ('RVI ≤ прошлой медианы',d.RVI_high==False),('RVI > прошлой медианы',d.RVI_high==True)]
        for year in [2017,2018,2019,2020,2021,'2017–2021 пул']:
            ym=np.ones(len(d),dtype=bool) if isinstance(year,str) else d.year.eq(year)
            for state,sm in states:
                g=d[ym&sm]
                for name in components:
                    regimes.append(dict(h=h,year=year,state=state,model=name,n=len(g),mean_RV_over_prediction=float(g['ratio_'+name].mean()),QLIKE=float(g['loss_'+name].mean())))
        gateparts=[]
        for label,mask in [('2020 март–июнь',d.date.between('2020-03-01','2020-06-30')),
                           ('RVI верхний терциль',d.stress_RVI.astype(bool))]:
            g=d[mask]
            for name in components:
                stress.append(dict(h=h,state=label,model=name,n=len(g),mean_RV_over_prediction=float(g['ratio_'+name].mean()),QLIKE=float(g['loss_'+name].mean())))
            rf=float(g['ratio_финал'].mean());rb=float(g['ratio_HAR-J(б)'].mean())
            gateparts.append(rf>1+1e-12 and rf>rb+1e-12)
        gates[h]=dict(run_K9_K10=all(gateparts),stress_conditions=gateparts,
                      rule='Mean RV/p_final>1 and greater than HAR-J(b) on BOTH mandatory dev stress subsets; numerical tolerance1e-12.')
        for year in range(2017,2022):
            cutoff=pd.Timestamp(f'{year}-01-01')
            train=tr[(tr.date<cutoff)&(tr[f'tend{h}']<cutoff)]
            v=train[target];q99=float(v.quantile(.99));maximum=float(v.max())
            g=d[d.year==year].copy()
            mx={c:float(train[c].max()) for c in ['RV_d0','RV_w0','RV_d_w']}
            cases=[('все',np.ones(len(g),dtype=bool))]+[(c+'>max_train',g[c]>mx[c]) for c in mx]
            for name,col in components.items():
                p=g[col]
                bounds.append(dict(h=h,year=year,model=name,n=len(g),train_n=len(train),train_y_q99=q99,train_y_max=maximum,
                                   frac_pred_gt_train_q99=float((p>q99).mean()),frac_pred_near_train_max_1pct=float((abs(p/maximum-1)<=.01).mean()),
                                   frac_pred_gt_train_max=float((p>maximum).mean()),pred_max=float(p.max())))
                for state,mask in cases:
                    z=g[mask]
                    outside.append(dict(h=h,year=year,model=name,state=state,n=len(z),
                                        mean_prediction=float(z[col].mean()),mean_RV=float(z.rv_actual.mean()),
                                        mean_RV_over_prediction=float(z['ratio_'+name].mean()),QLIKE=float(z['loss_'+name].mean()),
                                        frac_pred_gt_train_q99=float((z[col]>q99).mean()) if len(z) else np.nan,
                                        train_RV_d_max=mx['RV_d0'],train_RV_w_max=mx['RV_w0'],train_ratio_max=mx['RV_d_w']))
        frames[h]=d
    for name,data in [('diag_regimes',regimes),('diag_stress',stress),('diag_prediction_bounds',bounds),('diag_outside_training',outside)]:
        pd.DataFrame(data).to_csv(TAB/f'{name}.csv',index=False)
    payload=dict(gates=gates,control_read=False,new_fits=0,reads=READS,
                 near_max_definition='abs(prediction/max_training_target-1)<=.01',
                 regime_definition='RV_d/RV_w is additionally tested as a ratio, as well as RV_d and RV_w individually; d0 windows include known dayd.')
    (TAB/'diagnostics_manifest.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    s=pd.DataFrame(stress)
    gate_table=pd.DataFrame([dict(h=h,**g) for h,g in gates.items()])
    lines=['# Диагностика перед раундами5–6 — только dev2017–2021','',
           'До кандидатов, без обучения и без контроля2022–2026. Среднее RV/p>1 означает среднее занижение уровня. Условие запуска К9/К10 фиксировано до расчёта: на данномh финал должен иметь RV/p>1 и больше HAR-J(б) на обеих обязательных стрессовых выборках. Смешанный результат не объявляется подтверждением.','',
           render(s),'',render(gate_table.drop(columns=['rule'])),'',
           'Границы обучения — расширяющаяся история с2014, цель заканчивается раньше1января года оценки. p99/max вычислены по целям, не прогнозам; «у max» означает±1%. RV_d/RV_w показан и как отношение, и как два отдельных признака. Все окна известны наd до18:50; терцили RV — по предыдущим22 датам, медиана RVI — расширяющаяся, без текущей даты.','',
           '## По годам и режимам','',render(pd.DataFrame(regimes)),'','## Процентили и максимум обучения','',render(pd.DataFrame(bounds)),'',
           '## Выход RV_d, RV_w и их отношения за обучающий диапазон','',render(pd.DataFrame(outside))]
    (ROOT/'docs/otchety/2026-11-04_gl3_r56_diagnostics.md').write_text('\n'.join(lines)+'\n')
    print(s.to_string(index=False));print(json.dumps(gates,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
