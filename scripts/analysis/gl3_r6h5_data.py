# -*- coding: utf-8 -*-
"""Round6 h5 only: filtered inputs, common dissertation frame, metrics."""
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from gl3_r56_diagnostics import read,READS,ROOT,REG
from gl3_r56_data import Data as PriorData,SECTORS

TAB=ROOT/'docs/tables/gl3_r6_h5';OUT=REG/'gl3_r6_h5';YEARS=tuple(range(2017,2022))


def disk(stage):
    TAB.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    out=subprocess.run(['df','-h','/'],capture_output=True,text=True,check=True).stdout
    with (TAB/'disk.log').open('a') as f:f.write(str(pd.Timestamp.now(tz='UTC'))+' '+stage+'\n'+out)
    if shutil.disk_usage('/').free<20*1024**3:raise RuntimeError('DISK_STOP: less than20GiB free')


def load_events():
    rows=[];packed=TAB/'fit_ledger.csv.gz'
    if packed.exists():
        q=pd.read_csv(packed);q['detail']=q.detail.map(json.loads);rows+=q.to_dict('records')
    live=TAB/'fit_events.jsonl'
    if live.exists():rows+=[json.loads(x) for x in live.read_text().splitlines()]
    unique={tuple(r[k] for k in ('time_utc','component','phase','h','cutoff')):r for r in rows}
    return list(unique.values())


class Data(PriorData):
    def __init__(self):
        disk('load h5 DEV only')
        self.columns=json.loads((REG/'features_pit_v3.config.json').read_text())['feature_cols']
        cols=list(dict.fromkeys(['date','ticker','n_bars','rkurt','close','rv','bv','rsv_pos','rsv_neg','jump','rv_IMOEX']+self.columns))
        a=read(REG/'features_pit_v3.parquet',cols).sort_values(['ticker','date']).reset_index(drop=True)
        a['year']=a.date.dt.year;a['_end5']=a.groupby('ticker').date.shift(-5)
        filters=[[('ticker','==',tk),('date','<=',g.loc[g._end5.notna(),'date'].max())] for tk,g in a.groupby('ticker')]
        target=read(REG/'features_pit_v3.parquet',['date','ticker','rv_target_mean_h5'],filters)
        a=a.merge(target.rename(columns={'rv_target_mean_h5':'_y5'}),on=['date','ticker'],how='left',validate='1:1')
        rr=[c for c in pq.read_schema(REG/'features_pit_v3_rrv.parquet').names if c.startswith('rrv_') and 'target' not in c]
        a=a.merge(read(REG/'features_pit_v3_rrv.parquet',['date','ticker']+rr),on=['date','ticker'],how='left',validate='1:1')
        for path,cols in [(REG/'L2/rq1850.parquet',['rq']),(REG/'R2/feat/_work/gap_daily.parquet',['gap_on'])]:
            a=a.merge(read(path,['date','ticker']+cols),on=['date','ticker'],how='left',validate='1:1')
        names=pq.read_schema(REG/'features_pit_v3_d7_h5.parquet').names
        self.d7=[c for c in names if c.startswith('d7_')]
        a=a.merge(read(REG/'features_pit_v3_d7_h5.parquet',['date','ticker']+self.d7),on=['date','ticker'],how='left',validate='1:1')
        self.basecols={5:[c for c in self.columns if c in names]+self.d7}
        a['r2_on']=np.log(a.gap_on.pow(2)+1e-6)
        a['rv_d']=a.rv
        for c,w in [('rv_w',5),('rv_m',22),('_RV66',66),('_RV132',132),('_scale',252)]:
            a[c]=a.groupby('ticker').rv.transform(lambda s:s.rolling(w,min_periods=252 if w==252 else 1).mean())
        for c in ('rv_d','rv_w','rv_m','rq','close_RVI'):a['_ln_'+c]=np.log(a[c].clip(lower=1e-12))
        q=np.sqrt((a.n_bars*a.rkurt/3).clip(lower=0)).clip(upper=10)
        a['_HARQ']=np.log(a.rv_d.clip(lower=1e-12))*q.groupby(a.ticker).shift(1)
        ret=np.log(a.close.where(a.close>0)).groupby(a.ticker).diff().clip(-.5,.5)
        for w in (1,5,22):a['_LHAR'+str(w)]=ret.groupby(a.ticker).transform(lambda s:s.rolling(w,min_periods=1).mean()).clip(upper=0)
        rvi=a.groupby('date').close_RVI.first().sort_index()
        a['_state']=a.date.map(rvi>rvi.expanding(min_periods=22).median().shift(1)).astype(bool)
        rank=rvi.rolling(253,min_periods=253).apply(lambda z:(z[:-1]<=z[-1]).mean(),raw=True)
        a['_stress']=a.date.map(rank>2/3).astype(bool)
        self.a=a.replace([np.inf,-np.inf],np.nan);self.frames={'original':self.a}
        ref=read(REG/'gl3_r3/dev/predictions_h5.parquet',['date','ticker','year','rv_actual','rv_pred','har_corr','har_raw'],[('date','>=',pd.Timestamp('2017-01-01')),('date','<=',pd.Timestamp('2021-12-31'))])
        self.idx=pd.MultiIndex.from_frame(a[['date','ticker']]).get_indexer(pd.MultiIndex.from_frame(ref[['date','ticker']]))
        np.testing.assert_allclose(a.loc[self.idx,'_y5'],ref.rv_actual,rtol=1e-9,atol=1e-12)
        self.ref=ref;self.refs={5:(ref,self.idx)};self.filters=filters;self.hxp_params=(.1,1.)
        self.base=self.score5(self.series(ref,'rv_pred'))

    def series(self,z,col):
        out=np.full(len(self.a),np.nan)
        ix=pd.MultiIndex.from_frame(self.a[['date','ticker']]).get_indexer(pd.MultiIndex.from_frame(z[['date','ticker']]))
        assert (ix>=0).all() and not z.duplicated(['date','ticker']).any()
        out[ix]=z[col].values;return out

    def saved(self,path,col):return self.series(read(path,['date','ticker',col],self.filters),col)

    def old(self,label):
        import hashlib
        key='h5:'+label;path=REG/'gl3_r5_r6'/(hashlib.sha256(key.encode()).hexdigest()[:16]+'.parquet')
        return self.saved(path,'prediction')

    def score5(self,p):
        out=super().score(5,p);r=self.ref.rv_actual.values/p[self.idx]
        m=self.ref.date.between('2020-03-01','2020-06-30').values
        high=self.a.loc[self.idx,'_stress'].values
        out['ratio_stress2020']=float(r[m].mean());out['ratio_stressRVI']=float(r[high].mean())
        return out

    def harx(self,h,measure,normalized=False):
        if not normalized:return super().harx(h,measure,normalized)
        assert h==5 and measure=='original'
        b=self.a;cols=['rv_d','rv_w','rv_m','jump','jump_annualized','jump_lag1','jump_lag5','jump_lag22']
        x=b[cols].values.astype(float).copy()
        for j,c in enumerate(cols):x[:,j]/=np.sqrt(b._scale.values) if c=='jump_annualized' else b._scale.values
        return np.column_stack([np.ones(len(b)),np.log(np.clip(x,1e-12,None))])

    def static_x(self,h,measure,sector=False,normalize=False):
        assert h==5 and measure=='original'
        if not normalize:
            x,names=super().static_x(h,measure,sector,normalize)
            if getattr(self,'cat_ticker',False):
                code=self.a.ticker.map({t:i for i,t in enumerate(sorted(self.a.ticker.unique()))}).values
                x=np.column_stack([x,code]).astype('float32');names+=['cat_ticker']
            return x,names
        x,names=self.raw_features(True)
        for col in ('rv_d','rv_w','rv_m'):
            if col in names:
                j=names.index(col);x[:,j]=np.log(np.clip(x[:,j],1e-12,None));names[j]='ln_'+col
        return x,names

    def raw_features(self,normalized=False):
        names=list(dict.fromkeys(self.basecols[5]));x=self.a[names].values.astype(float)
        if normalized:
            for j,c in enumerate(names):
                if c in ('rv_annualized','bv_annualized','jump_annualized'):x[:,j]/=np.sqrt(self.a._scale.values)
                elif c.startswith(('rv','bv','jump','rsv')) and not any(k in c for k in ('annualized','RVI','IMOEX','MOEX','CNY','RGBI')):x[:,j]/=self.a._scale.values
            sectors=pd.get_dummies(self.a.ticker.map(SECTORS),dtype=float)
            x=np.column_stack([x,sectors.values]);names+=['sector_'+c for c in sectors]
            if getattr(self,'cat_ticker',False):
                x=np.column_stack([x,self.a.ticker.map({t:i for i,t in enumerate(sorted(self.a.ticker.unique()))}).values]);names+=['cat_ticker']
        return x.astype('float32'),names
