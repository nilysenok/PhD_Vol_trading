# -*- coding: utf-8 -*-
"""Раунд 6б, h=5, волна 2 (Г13, Г16–Г18): общие части — пути, ledger, диск, загрузка DEV, s² sm, ES-бустинг группы/панели. DEV 2017–2021, контроль не читается."""
import os
import sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import json,shutil,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
sys.path.insert(0,str(Path(__file__).resolve().parent))
import gl3_r6h5_data as D
from gl3_r56_diagnostics import read,REG,ROOT
import gl3_r6b_g8 as G8

OUT=REG/'gl3_r6b_h5/retrain';TAB=ROOT/'docs/tables/gl3_r6b_h5/retrain';EVENTS=TAB/'fit_events.jsonl';YEARS=G8.YEARS
PENDING=[]


def disk(stage):
    TAB.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    out=subprocess.run(['df','-h','/'],capture_output=True,text=True,check=True).stdout
    with (TAB/'disk.log').open('a') as f:f.write(str(pd.Timestamp.now(tz='UTC'))+' '+stage+'\n'+out)
    if shutil.disk_usage('/').free<20*1024**3:raise RuntimeError('DISK_STOP: less than 20GiB free')


def record(kind,cut,n,maxend,phase,detail=None):
    """У12: событие обучения. max_target_end < cutoff < 2022-01-01. Пишется в файл только через flush() после сохранения результата стадии."""
    assert pd.Timestamp(maxend)<pd.Timestamp(cut)<pd.Timestamp('2022-01-01'),(kind,str(cut),str(maxend))
    PENDING.append(dict(component=kind,h=5,cutoff=str(cut),n=int(n),max_target_end=str(maxend),phase=phase,detail=detail or {},time_utc=pd.Timestamp.now(tz='UTC').isoformat()))


def flush():
    TAB.mkdir(parents=True,exist_ok=True)
    with EVENTS.open('a') as f:
        for r in PENDING:f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n')
    n=len(PENDING);PENDING.clear();return n


def load():
    D.disk=disk;d=D.Data();h=read(REG/'gl3_r3/dev/predictions_h5.parquet',['date','ticker','har_corr'])
    return d,d.series(h,'har_corr')


def f32(p):return np.asarray(p).astype('float32').astype('float64')


def scalar_level(a,y,raw):
    """Уровень exp(raw + s²/2); s² — дисперсия лог-ошибки по закрытым окнам трёх прошлых лет (конец цели < cut), как Engine.calibrate 'scalar', годовой график."""
    out=np.full(len(a),np.nan);y=np.asarray(y,float)
    for cut in pd.date_range('2016-01-01','2021-01-01',freq='YS'):
        va=((a.date>=cut)&(a.date<cut+pd.DateOffset(years=1))).values
        tr=((a.date<cut)&(a._end5<cut)&(a.date>=cut-pd.DateOffset(years=3))).values&np.isfinite(raw)&np.isfinite(y)&(y>1e-12)
        err=np.log(y[tr])-raw[tr];w=np.ones(int(tr.sum()))
        v=np.full(int(va.sum()),np.average((err-np.average(err,weights=w))**2,weights=w)) if len(err)>=50 else np.zeros(int(va.sum()))
        out[va]=np.exp(np.clip(raw[va]+v/2,-60,60))
    return out


def es_boost(x,y,tr,va,full,a,cut,kind,detail):
    """Параметры Г8а/Г8б (G8.PARAMS), ранняя остановка 30 на (tr,va) с purge/embargo 5 (У10), refit на full; возвращает модель и прогноз ES-модели на va."""
    P=G8.PARAMS
    first=lgb.train(P,lgb.Dataset(x[tr],label=y[tr]),500,valid_sets=[lgb.Dataset(x[va],label=y[va])],callbacks=[lgb.early_stopping(30,verbose=False)])
    trees=max(1,first.best_iteration);det={**detail,'trees':trees,'params':{k:P[k] for k in G8.LOGKEYS}}
    record(kind,cut,tr.sum(),a.loc[tr,'_end5'].max(),'inner_ES',det)
    model=lgb.train(P,lgb.Dataset(x[full],label=y[full]),trees)
    record(kind,cut,full.sum(),a.loc[full,'_end5'].max(),'refit',det)
    return model,first.predict(x[va],num_iteration=trees)


def save_pred(d,name,p):
    z=d.ref[['date','ticker']].copy();z['prediction']=np.asarray(p)[d.idx].astype('float32');OUT.mkdir(parents=True,exist_ok=True)
    z.to_parquet(OUT/f'{name}_dev.parquet',index=False)


def load_pred(d,name):
    return d.series(pd.read_parquet(OUT/f'{name}_dev.parquet'),'prediction')
