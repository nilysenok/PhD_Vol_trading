# -*- coding: utf-8 -*-
"""Dev data and fresh-feature design for rounds5–6. No test reader."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gl3_r56_diagnostics import read,READS,ROOT,REG,TAB,LAST

OUT=REG/'gl3_r5_r6'
YEARS=tuple(range(2017,2022))
MODES=('annual','quarterly','monthly','half1','half2','rolling3')
SECTORS={'LKOH':'OG','NVTK':'OG','TATN':'OG','SBER':'FN','VTBR':'FN','MOEX':'FN','HYDR':'EU','IRAO':'EU','OGKB':'EU','ALRS':'MM','MTLR':'MM','MTSS':'TL','RTKM':'TL','MGNT':'CN','PHOR':'CH','AFLT':'TN','LSRG':'RE'}


def disk_guard(stage):
    free=shutil.disk_usage('/').free
    out=subprocess.run(['df','-h','/'],capture_output=True,text=True,check=True).stdout
    with (TAB/'disk_checks.log').open('a') as f:
        f.write(str(pd.Timestamp.now(tz='UTC'))+' '+stage+'\n'+out)
    if free<20*1024**3:raise RuntimeError('DISK_STOP: less than20GiB free; owner notification required')


def safe_csv(path,columns,step=1):
    meta=pd.read_csv(path,usecols=['step'])
    keep=set(meta.index[meta.step==step])
    return pd.read_csv(path,usecols=columns,skiprows=lambda i:i>0 and i-1 not in keep)


def periods(mode):
    freq='MS' if mode=='monthly' else ('QS' if mode=='quarterly' else 'YS')
    dates=pd.date_range('2016-01-01','2022-01-01',freq=freq)
    return list(zip(dates[:-1],dates[1:]))


def mask_train(a,h,cut,mode,y):
    m=(a.date<cut)&(a[f'_end{h}']<cut)&np.isfinite(y)&(y>1e-12)
    if mode=='rolling3':m&=a.date>=cut-pd.DateOffset(years=3)
    assert not m.any() or a.loc[m,f'_end{h}'].max()<cut
    return m.values


def weights(a,cut,mode):
    if mode in ('half1','half2'):
        half=1 if mode=='half1' else 2
        return np.power(.5,((cut-a.date).dt.days.values/365.25)/half)
    return np.ones(len(a))


class Data:
    def __init__(self):
        OUT.mkdir(exist_ok=True);TAB.mkdir(exist_ok=True)
        cfg=json.loads((REG/'features_pit_v3.config.json').read_text())
        self.columns=list(cfg['feature_cols'])
        a=read(REG/'features_pit_v3.parquet',list(dict.fromkeys(['date','ticker','n_bars','rkurt','close','rv','bv','rsv_pos','rsv_neg','jump','rv_IMOEX']+self.columns)))
        a=a.sort_values(['ticker','date']).reset_index(drop=True);a['year']=a.date.dt.year
        for h in (1,5,22):a[f'_end{h}']=a.groupby('ticker').date.shift(-h)
        rq=read(REG/'L2/rq1850.parquet',['date','ticker','rq'])
        a=a.merge(rq,on=['date','ticker'],how='left',validate='1:1')
        rr=[c for c in pq.read_schema(REG/'features_pit_v3_rrv.parquet').names if c.startswith('rrv_') and 'target' not in c]
        a=a.merge(read(REG/'features_pit_v3_rrv.parquet',['date','ticker']+rr),on=['date','ticker'],how='left',validate='1:1')
        gap=read(REG/'R2/feat/_work/gap_daily.parquet',['date','ticker','gap_on'])
        a=a.merge(gap,on=['date','ticker'],how='left',validate='1:1')
        self.d7=[c for c in pq.read_schema(REG/'features_pit_v3_d7_h5.parquet').names if c.startswith('d7_') and not c.startswith('d7_cal_')]
        d7=read(REG/'features_pit_v3_d7_h5.parquet',['date','ticker']+self.d7)
        a=a.merge(d7,on=['date','ticker'],how='left',validate='1:1')
        h5schema=pq.read_schema(REG/'features_pit_v3_d7_h5.parquet').names
        self.basecols={1:self.columns+rr+['r2_on','cal_bd'],5:[c for c in self.columns if c in h5schema]+self.d7,
                       22:['rv_d','rv_w','rv_m','jump','jump_annualized','jump_lag1','jump_lag5','jump_lag22']}
        a['r2_on']=np.log(a.gap_on.pow(2)+1e-6)
        self.a=a;self.frames={};self.refs={};self.warm={}
        for h in (1,5,22):
            target=f'rv_target_h{h}' if h==1 else f'rv_target_mean_h{h}'
            pred=[]
            for tk,g in a.groupby('ticker'):
                pred.append([('ticker','==',tk),('date','<=',g.loc[g[f'_end{h}'].notna(),'date'].max())])
            z=read(REG/'features_pit_v3.parquet',['date','ticker',target],pred)
            a=a.merge(z.rename(columns={target:f'_y{h}'}),on=['date','ticker'],how='left',validate='1:1')
            ref=read(REG/f'gl3_r3/dev/predictions_h{h}.parquet',['date','ticker','year','rv_actual','rv_pred','har_corr','har_raw'],
                     [('date','>=',pd.Timestamp('2017-01-01')),('date','<=',LAST)])
            index=pd.MultiIndex.from_frame(a[['date','ticker']]).get_indexer(pd.MultiIndex.from_frame(ref[['date','ticker']]))
            assert (index>=0).all()
            np.testing.assert_allclose(a.loc[index,f'_y{h}'].values,ref.rv_actual.values,rtol=1e-9,atol=1e-12)
            self.refs[h]=(ref,index)
        self.a=a
        alt=read(OUT/'grid10_shift_mean.parquet',['date','ticker','rv_alt','rq_alt','bv_alt','rsv_pos_alt','rsv_neg_alt'])
        for measure in ('original','alt'):
            b=a.copy()
            if measure=='alt':
                b=b.merge(alt,on=['date','ticker'],how='left',validate='1:1')
                for col in ('rv','rq','bv','rsv_pos','rsv_neg'):b[col]=b[col+'_alt']
                b['jump']=(b.rv-b.bv).clip(lower=0)
                b['rv_annualized']=np.sqrt(b.rv*252);b['bv_annualized']=np.sqrt(b.bv*252);b['jump_annualized']=np.sqrt(b.jump*252)
                for w in (1,5,22):b[f'jump_lag{w}']=b.groupby('ticker').jump.transform(lambda s:s.shift(1).rolling(w,min_periods=1).mean())
                for h in (1,5,22):
                    b[f'_y{h}']=np.column_stack([b.groupby('ticker').rv.shift(-j) for j in range(1,h+1)]).mean(axis=1)
            group=b.groupby('ticker')
            b['rv_d']=b.rv
            for c,w in [('rv_w',5),('rv_m',22),('rv_2w',10),('rv_q',66)]:b[c]=group.rv.transform(lambda s:s.rolling(w,min_periods=1).mean())
            b['_scale']=group.rv.transform(lambda s:s.rolling(252,min_periods=252).mean())
            b['_RV66']=group.rv.transform(lambda s:s.rolling(66,min_periods=1).mean())
            b['_RV132']=group.rv.transform(lambda s:s.rolling(132,min_periods=1).mean())
            for c in ['rv_d','rv_w','rv_m','rq','close_RVI']:
                b['_ln_'+c]=np.log(b[c].clip(lower=1e-12))
            q=(np.sqrt((b.n_bars*b.rkurt/3).clip(lower=0)).clip(upper=10) if measure=='original' else (np.sqrt(b.rq.clip(lower=0))/b.rv_d.clip(lower=1e-12)).clip(upper=10))
            b['_HARQ']=np.log(b.rv_d.clip(lower=1e-12))*q.groupby(b.ticker).shift(1)
            ret=np.log(b.close.where(b.close>0)).groupby(b.ticker).diff().clip(-.5,.5)
            for w in (1,5,22):b['_LHAR'+str(w)]=ret.groupby(b.ticker).transform(lambda s:s.rolling(w,min_periods=1).mean()).clip(upper=0)
            rvi=b.groupby('date').close_RVI.first().sort_index()
            b['_state']=b.date.map(rvi>rvi.expanding(min_periods=22).median().shift(1)).astype(bool)
            rank=rvi.rolling(253,min_periods=253).apply(lambda z:np.mean(z[:-1]<=z[-1]),raw=True)
            b['_stress']=b.date.map(rank>2/3).astype(bool)
            self.frames[measure]=b.replace([np.inf,-np.inf],np.nan)
        p=REG/'D22_h1_xgb_h1_top5_val.parquet'
        ids=read(p,['date','step','rank','point'],[('date','>=',pd.Timestamp('2016-01-01')),('date','<',pd.Timestamp('2017-01-01'))])
        top=ids.groupby('rank').point.first().sort_index()
        grid=pd.read_csv(REG/'D22_h1_grid.csv');grid=grid[grid.model=='xgb'].set_index('point')
        self.fin1params=[grid.loc[int(point)].to_dict() for point in top.iloc[:5]]
        fields=['model','step','num_leaves','learning_rate','bagging_fraction','feature_fraction','min_data_in_leaf','reg_alpha','reg_lambda','max_depth','subsample','colsample_bytree','min_child_weight']
        p5=safe_csv(REG/'D12_h5_selected_h5.csv',fields)
        self.fin5params={lib:p5[p5.model==lib].iloc[0].to_dict() for lib in ('lgbm','xgb')}
        self.hxp_params=(.001,.5)
        hp=REG/'D22/HXP_coefs.csv'
        if hp.exists():
            params=safe_csv(hp,['tgt','step','alpha','l1_ratio']);x=params[params.tgt.astype(str)=='1']
            if len(x):self.hxp_params=(float(x.iloc[0].alpha),float(x.iloc[0].l1_ratio))

    def provenance(self):
        import hashlib
        def clean(v):
            if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
            if isinstance(v,list):return [clean(x) for x in v]
            if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
            return v
        return clean(dict(fin1params=self.fin1params,fin5params=self.fin5params,hxp_params=list(self.hxp_params),
                          feature_names=self.basecols,selected_numerical_configuration_year=2016,
                          selection_reads=READS,control_read=False))

    def y(self,h,measure):return self.frames[measure][f'_y{h}'].values

    def harx(self,h,measure,normalized=False):
        b=self.frames[measure]
        cols=['rv_d','rv_w','rv_m','jump','jump_annualized','jump_lag1','jump_lag5','jump_lag22']
        x=b[cols].values.astype(float)
        if normalized:x=x/b['_scale'].values[:,None]
        return np.column_stack([np.ones(len(b)),np.log(np.clip(x,1e-12,None))])

    def static_x(self,h,measure,sector=False,normalize=False):
        b=self.frames[measure];cols=list(dict.fromkeys(self.basecols[h]));arr=[];names=[]
        for col in cols:
            if col=='cal_bd':continue
            val=b[col].values.copy()
            if normalize:val=val/b['_scale'].values
            if col in ('rv_d','rv_w','rv_m'):
                val=np.log(np.clip(val,1e-12,None));col='ln_'+col
            arr.append(val);names.append(col)
        if h==1:
            dates=b.date.values.astype('datetime64[D]');bd=np.busday_offset(dates,1,roll='backward')
            arr.append((bd-dates).astype(float));names.append('cal_bd')
        if sector:
            for sec in sorted(set(SECTORS.values())):arr.append(b.ticker.map(SECTORS).eq(sec).values.astype(float));names.append('sector_'+sec)
        x=np.column_stack(arr).astype(np.float32)
        return x,names

    def score(self,h,p):
        ref,idx=self.refs[h];v=p[idx]
        assert np.isfinite(v).all() and (v>0).all()
        r=ref.rv_actual.values/v;l=r-np.log(r)-1
        result={str(y):float(l[ref.year==y].mean()) for y in YEARS}
        result['mean_QLIKE']=float(np.mean(list(result.values())))
        sm=ref.date.between('2020-03-01','2020-06-30').values
        high=self.frames['original'].loc[idx,'_stress'].values.astype(bool)
        result['stress2020']=float(l[sm].mean());result['stressRVI']=float(l[high].mean())
        return result
