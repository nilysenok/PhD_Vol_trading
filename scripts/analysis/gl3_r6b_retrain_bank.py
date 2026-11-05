# -*- coding: utf-8 -*-
"""Г13/Г16: переобучение D12_h5 (lgb+xgb, замороженные параметры warmup 2016, ранняя остановка с purge/embargo 5, годовой refit) — путь FIN_PIT (gl3_r6b_pit).
Варианты: 'a' — как финал (= FIN_PIT); 'b' — цель ln y обрезана 0,5/99,5 по тикеру; 'v' — признаки обрезаны 1/99; 'g16' — признаки финала + характеристики У29."""
import json,warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import HuberRegressor
import gl3_r6b_retrain_base as B
from gl3_r6h5_engine import purged,SPACE

LIBS=('lgb','xgb');SEED=42;HARJC=['rv_d','rv_w','rv_m','jump','jump_annualized','jump_lag1','jump_lag5','jump_lag22']
FZ=json.loads((B.ROOT/'docs/tables/gl3_r5_r6/frozen_warmup_configuration.json').read_text())['fin5params']


def frozen(lib):
    f=FZ['lgbm' if lib=='lgb' else 'xgb'];return {k:(int(f[k]) if sc in ('int','logint') else float(f[k])) for k,lo,hi,sc in SPACE[lib]}


def target_bounds(yy,tick,mask,lo=.5,hi=99.5):
    """Пороги обрезки ln y по тикеру — только по строкам mask (строки обучения текущей отсечки)."""
    s=pd.DataFrame(dict(y=np.asarray(yy)[mask],t=np.asarray(tick)[mask]));q=s.groupby('t').y.quantile([lo/100,hi/100]).unstack()
    return {t:(r.iloc[0],r.iloc[1]) for t,r in q.iterrows()}


def clip_target(yy,tick,bounds,mask):
    """Обрезка только на строках mask (валидация ES остаётся необрезанной)."""
    out=np.asarray(yy,float).copy();tick=np.asarray(tick)
    for t,(l,h) in bounds.items():
        sel=(tick==t)&mask;out[sel]=np.clip(out[sel],l,h)
    return out


def feature_bounds(x,mask,lo=1,hi=99):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');q=np.nanpercentile(np.asarray(x)[mask],[lo,hi],axis=0)
    return np.where(np.isfinite(q[0]),q[0],-np.inf),np.where(np.isfinite(q[1]),q[1],np.inf)


def prepare(variant,x,yy,tick,mask):
    """Вход и цель для обучения на строках mask. Пороги — только по mask; у 'v' те же пороги применяются ко всем строкам, включая прогнозные."""
    if variant=='b':return x,clip_target(yy,tick,target_bounds(yy,tick,mask),mask)
    if variant=='v':
        lo,hi=feature_bounds(x,mask);return np.clip(x,lo,hi).astype(x.dtype),yy
    return x,yy


def offset(a,hx,y,mask,cut,variant,phase):
    fit=HuberRegressor(max_iter=500).fit(hx[mask],y[mask])
    B.record('G1_HuberHAR',cut,mask.sum(),a.loc[mask,'_end5'].max(),phase,dict(variant=variant));return fit.predict(hx)


def design(x,keep,off):return np.nan_to_num(np.column_stack([x[:,keep],off]),nan=0.,posinf=0.,neginf=0.).astype('float32')


def fit_lib(lib,p,xx,target,tr,va,off,nround):
    """Как Engine.g1fit без отбора параметров: huber-цель (lgb) / pseudohuber (xgb); ES 30 на va; -> (прогноз off+f на всех строках, число деревьев)."""
    if lib=='lgb':
        params=dict(objective='huber',metric='huber',num_threads=2,verbosity=-1,seed=SEED,force_col_wise=True,max_bin=255,bagging_freq=1,**p)
        train=lgb.Dataset(xx[tr],label=target[tr],categorical_feature='auto');valid=[];cb=[]
        if va is not None:valid=[lgb.Dataset(xx[va],label=target[va],reference=train,categorical_feature='auto')];cb=[lgb.early_stopping(30,verbose=False)]
        model=lgb.train(params,train,nround or 1000,valid_sets=valid,callbacks=cb)
        return off+model.predict(xx,raw_score=True),int(model.best_iteration or nround or model.current_iteration())
    params=dict(objective='reg:pseudohubererror',eval_metric='mphe',tree_method='hist',max_bin=255,nthread=2,seed=SEED,verbosity=0,**p)
    ev=[(xgb.DMatrix(xx[va],label=target[va]),'past')] if va is not None else []
    model=xgb.train(params,xgb.DMatrix(xx[tr],label=target[tr]),nround or 1000,evals=ev,early_stopping_rounds=30 if va is not None else None,verbose_eval=False)
    trees=int(getattr(model,'best_iteration',model.num_boosted_rounds()-1))+1
    return off+model.predict(xgb.DMatrix(xx),output_margin=True,iteration_range=(0,trees)),trees


def one_year(d,variant,x,names,yy,hx,ex,year):
    a=d.a;cut=pd.Timestamp(f'{year}-01-01');tr,va,full=purged(a,cut,year-1);tk=a.ticker.values
    pred=((a.date>=cut)&(a.date<pd.Timestamp(f'{year+1}-01-01'))).values
    xi,yi=prepare(variant,x,yy,tk,tr);xo,yo=prepare(variant,x,yy,tk,full)
    inner=offset(a,hx,yi,tr,cut,variant,'inner');outer=offset(a,hx,yo,full,cut,variant,'refit');outs=[]
    for lib in LIBS:
        drop=set(ex[(ex.h==5)&(ex.model==('lgbm' if lib=='lgb' else 'xgb'))].feature) if year>2016 else set()
        keep=[j for j,n in enumerate(names) if n not in drop];p=frozen(lib);det=dict(variant=variant,lib=lib,year=year,params=p,n_features=len(keep)+1)
        _,trees=fit_lib(lib,p,design(xi,keep,inner),yi-inner,tr,va,inner,None)
        B.record(f'D12_{variant}_{lib}',cut,tr.sum(),a.loc[tr,'_end5'].max(),'inner_ES',{**det,'trees':trees})
        pp,_=fit_lib(lib,p,design(xo,keep,outer),yo-outer,full,None,outer,trees)
        B.record(f'D12_{variant}_{lib}',cut,full.sum(),a.loc[full,'_end5'].max(),'refit',{**det,'trees':trees})
        outs.append(np.exp(np.clip(pp[pred],-60,60)))
    return pred,np.log(np.mean(outs,axis=0))


def build(d,variant):
    """Лог-прогноз 2016–2021 варианта; кэш bank_<variant>.parquet; ledger сбрасывается после записи кэша."""
    path=B.OUT/f'bank_{variant}.parquet'
    if path.exists():return d.series(pd.read_parquet(path),'raw')
    a=d.a
    if variant=='g16':
        import gl3_r6b_retrain_feat as F
        x,names=F.extended_features(d)
    else:x,names=d.raw_features()
    yy=np.log(d.y(5,'original')+1e-10);hx=np.nan_to_num(np.log(a[HARJC].values.clip(1e-10)));ex=pd.read_csv(B.ROOT/'data/regen/Bv2b_excluded_features.csv');raw=np.full(len(a),np.nan)
    for year in range(2016,2022):
        B.disk(f'D12 {variant} {year}');pred,r=one_year(d,variant,x,names,yy,hx,ex,year);raw[pred]=r
    z=a[['date','ticker']].copy();z['raw']=raw.astype('float32');z=z[z.date>='2016-01-01'];B.OUT.mkdir(parents=True,exist_ok=True);z.to_parquet(path,index=False)
    B.flush();return d.series(z,'raw')
