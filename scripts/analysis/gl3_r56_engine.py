# -*- coding: utf-8 -*-
"""Fresh estimators, chronological refits, OOS variance calibration. DEV only."""
import os
import fcntl
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='2'

import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import ElasticNet,HuberRegressor

from gl3_r56_data import Data,OUT,TAB,ROOT,periods,mask_train,weights,disk_guard

EVENTS=TAB/'fit_events.jsonl'


def stable(spec):return json.dumps(spec,sort_keys=True,separators=(',',':'))
def level(z):return np.exp(np.clip(z,-60,60))


def qlike_grad(pred,ds,c=1.):
    base=ds.get_init_score()
    f=base+c*(pred-base)
    u=np.exp(ds.get_label()-f)
    w=ds.get_weight()
    if w is None:w=1.
    return w*c*(1-u),w*c*c*u


def record(kind,h,cut,mode,measure,seed,n,maxend,phase='fit',detail=None):
    assert pd.Timestamp(maxend)<pd.Timestamp(cut)
    r=dict(component=kind,h=h,cutoff=str(cut),mode=mode,measure=measure,seed=seed,n=int(n),
           max_target_end=str(maxend),phase=phase,detail=detail or {},time_utc=pd.Timestamp.now(tz='UTC').isoformat())
    with EVENTS.open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')


class Engine:
    def __init__(self,data):
        self.data=data;self.har={};self.har_oos={};self.har_calib={};self.hxp={};self.pc={};self.rawcache={};self.predcache={}
        self.started=(TAB/'diagnostics_manifest.json').stat().st_mtime;self.deadline=self.started+3*86400
        self.contract='e7e7c92e0035'  # Numerical training contract; storage-only migration retains predictions.
        (OUT/'cache').mkdir(exist_ok=True)

    def context(self,h,mode,measure,cut,normalize=False):
        disk_guard('fit '+str((h,mode,measure,str(cut))))
        b=self.data.frames[measure];y=self.data.y(h,measure).copy()
        if normalize:y=y/b['_scale'].values
        tr=mask_train(b,h,cut,mode,y)
        if normalize:tr&=np.isfinite(b['_scale'].values)&(b['_scale'].values>1e-12)
        assert tr.sum()>=100
        w=weights(b,cut,mode)
        return b,y,tr,w

    def har_current(self,h,mode,measure,cut,kind='HAR',normalize=False):
        key=(h,mode,measure,str(cut),kind,normalize)
        b,y,tr,w=self.context(h,mode,measure,cut,normalize)
        hb=np.nan_to_num(self.data.harx(h,measure,normalize),nan=0.,posinf=0.,neginf=0.)
        if kind=='HARQ':
            z=b._HARQ.values
            if normalize:z=z/b['_scale'].values
            hb=np.column_stack([hb,np.nan_to_num(z)])
        elif kind=='HARCJ':
            cc=b.bv.clip(lower=1e-12).values
            if normalize:cc=cc/b['_scale'].values
            cm=pd.Series(cc).groupby(b.ticker).transform(lambda s:s.rolling(22,min_periods=1).mean()).values
            cw=pd.Series(cc).groupby(b.ticker).transform(lambda s:s.rolling(5,min_periods=1).mean()).values
            hb=np.column_stack([np.ones(len(b)),np.log(np.clip(cc,1e-12,None)),np.log(np.clip(cw,1e-12,None)),np.log(np.clip(cm,1e-12,None)),hb[:,4:]])
            hb=np.nan_to_num(hb)
        if key not in self.har:
            xx=hb[tr];yy=np.log(y[tr]);ww=w[tr]
            if kind=='HUBER':
                model=HuberRegressor(max_iter=500).fit(xx[:,1:],yy,sample_weight=ww)
                coef=np.r_[model.intercept_,model.coef_]
            else:coef=np.linalg.lstsq(xx*np.sqrt(ww[:,None]),yy*np.sqrt(ww),rcond=None)[0]
            self.har[key]=coef
            record(kind,h,cut,mode,measure,42,int(tr.sum()),b.loc[tr,f'_end{h}'].max(),detail={'normalize':normalize})
        return hb@self.har[key]

    def har_series(self,h,mode,measure,kind='HAR',normalize=False):
        key=(h,mode,measure,kind,normalize)
        if key in self.har_oos:return self.har_oos[key]
        b=self.data.frames[measure];pred=np.full(len(b),np.nan)
        for cut,end in periods(mode):
            m=(b.date>=cut)&(b.date<end)
            val=self.har_current(h,mode,measure,cut,kind,normalize)
            if normalize:val=val+np.log(b['_scale'].values)
            pred[m]=val[m]
        self.har_oos[key]=pred
        return pred

    def calibrate(self,h,mode,measure,raw,bias='scalar'):
        b=self.data.frames[measure];y=self.data.y(h,measure)
        p=np.full(len(b),np.nan);vv=np.zeros(len(b))
        z=b[['_ln_close_RVI','_ln_rq','_ln_rv_w']].values
        for cut,end in periods(mode):
            va=(b.date>=cut)&(b.date<end)
            tr=(b.date<cut)&(b[f'_end{h}']<cut)&(b.date>=cut-pd.DateOffset(years=3))&np.isfinite(raw)&np.isfinite(y)&(y>1e-12)
            err=np.log(y[tr])-raw[tr]
            w=weights(b,cut,mode)[tr]
            if len(err)<50 or bias=='none':v=np.zeros(int(va.sum()))
            elif bias=='conditional':
                zz=np.column_stack([np.ones(int(tr.sum())),z[tr]])
                good=np.isfinite(zz).all(axis=1)
                yy=(err-np.average(err,weights=w))**2
                if good.sum()<50:v=np.full(int(va.sum()),np.average(yy,weights=w))
                else:
                    coef=np.linalg.lstsq(zz[good]*np.sqrt(w[good,None]),yy[good]*np.sqrt(w[good]),rcond=None)[0]
                    zv=np.column_stack([np.ones(int(va.sum())),z[va]])
                    zv=np.nan_to_num(zv,nan=0.)
                    v=np.maximum(0,zv@coef)
                    record('conditional_variance',h,cut,mode,measure,42,int(good.sum()),b.loc[tr,f'_end{h}'].max(),detail={'coef':coef.tolist()})
            else:
                v=np.full(int(va.sum()),np.average((err-np.average(err,weights=w))**2,weights=w))
            vv[va]=v;p[va]=level(raw[va]+v/2)
        return p,vv

    def har_b(self,h,mode,measure):
        key=(h,mode,measure)
        if key not in self.har_calib:self.har_calib[key]=self.calibrate(h,mode,measure,self.har_series(h,mode,measure))
        return self.har_calib[key][0]

    def hxp_current(self,h,mode,measure,cut):
        key=(h,mode,measure,str(cut))
        b,y,tr,w=self.context(h,mode,measure,cut)
        base=self.har_current(h,mode,measure,cut)
        hb=np.nan_to_num(self.data.harx(h,measure))
        market=b.groupby('date').rv_m.mean()
        imoex=b.groupby('date').rv_IMOEX.first().sort_index()
        im=np.log((imoex**2).rolling(22,min_periods=13).mean().clip(lower=1e-12))
        pcfields=[c for c in ['yahoo_sp500_vol22','yahoo_eem_vol22','yahoo_gold_vol22','yahoo_wti_vol22','yahoo_brent_vol22','yahoo_dxy_vol22','vix'] if c in b]
        daily=b.groupby('date')[pcfields].first().sort_index()
        if key not in self.pc:
            past=daily.loc[daily.index.isin(b.loc[tr,'date'])].values
            pm=np.nan_to_num(np.nanmean(past,axis=0));ps=np.nan_to_num(np.nanstd(past,axis=0));ps[ps<1e-8]=1.
            zz=np.nan_to_num((past-pm)/ps)
            load=np.linalg.svd(zz,full_matrices=False)[2][0]
            if load[np.argmax(np.abs(load))]<0:load=-load
            self.pc[key]=(pm,ps,load)
        pm,ps,load=self.pc[key];pc=pd.Series(np.nan_to_num((daily.values-pm)/ps)@load,index=daily.index)
        extras=np.column_stack([2*b._ln_close_RVI.values,b.date.map(im).values,np.log(b.date.map(market).clip(lower=1e-12)),
             b._LHAR1,b._LHAR5,b._LHAR22,b.date.map(pc),b._HARQ,np.log(b.rsv_pos.clip(lower=1e-12)),np.log(b.rsv_neg.clip(lower=1e-12)),
             b.r2_on]+[b[c].values for c in b if c.startswith('rrv_') and 'target' not in c])
        extras=np.nan_to_num(extras,nan=0.,posinf=0.,neginf=0.)
        if h==22:extras=np.column_stack([extras,np.log(b._RV66.clip(lower=1e-12)),np.log(b._RV132.clip(lower=1e-12))])
        if key not in self.hxp:
            mean=extras[tr].mean(axis=0);sd=extras[tr].std(axis=0);sd[sd<1e-8]=1.
            zz=(extras-mean)/sd
            projection=np.linalg.lstsq(hb[tr]*np.sqrt(w[tr,None]),zz[tr]*np.sqrt(w[tr,None]),rcond=None)[0]
            residual=zz[tr]-hb[tr]@projection
            alpha,l1=self.data.hxp_params
            model=ElasticNet(alpha=max(alpha,1e-8),l1_ratio=l1,max_iter=3000,tol=1e-5,fit_intercept=False).fit(residual,np.log(y[tr])-base[tr],sample_weight=w[tr])
            self.hxp[key]=(mean,sd,projection,model)
            record('HAR-X+',h,cut,mode,measure,42,int(tr.sum()),b.loc[tr,f'_end{h}'].max(),detail={'alpha':alpha,'l1':l1})
        mean,sd,projection,model=self.hxp[key]
        return base+model.predict((extras-mean)/sd-hb@projection)

    def features(self,h,mode,measure,cut,normalize=False,sector=False):
        b=self.data.frames[measure]
        x,names=self.data.static_x(h,measure,sector,normalize)
        har=self.har_current(h,mode,measure,cut,normalize=normalize)
        bank=self.har_series(h,mode,measure,normalize=normalize)
        key=(h,mode,measure,normalize)
        if key not in self.har_calib:self.har_calib[key]=self.calibrate(h,mode,measure,bank)
        bias=self.har_calib[key][1]
        m=(b.date>=cut)&(b.date<(cut+pd.Timedelta(days=32)))
        correction=float(bias[np.flatnonzero(m)[0]]) if m.any() else 0.
        base=har+correction/2
        feature=self.hxp_current(h,mode,measure,cut) if h==1 else (self.har_current(h,mode,measure,cut,kind='HUBER') if h==5 else har)
        x=np.column_stack([x,feature,base]).astype(np.float32)
        names+=['har_feature','ln_HAR_b']
        return x,names,base

    def boost(self,h,mode,measure,cut,seed,lib,x,y,tr,w,style='a',objective='mse',base=None,c=1.,warm=None,huber=False,tag=''):
        if time.time()>self.deadline:raise TimeoutError('Three-day round budget exhausted; no false closure')
        b=self.data.frames[measure];dates=sorted(b.loc[tr,'date'].unique());edge=dates[max(1,len(dates)-126)]
        ti=tr&(b.date<edge).values&(b[f'_end{h}']<edge).values;vi=tr&(b.date>=edge).values
        assert ti.sum()>50 and vi.sum()>20
        xx=np.array(x,copy=True)
        if style=='linear':
            med=np.nanmedian(xx[tr],axis=0);med=np.nan_to_num(med)
            ii,jj=np.where(~np.isfinite(xx));xx[ii,jj]=med[jj]
        mono=np.zeros(xx.shape[1],dtype=int)
        if style=='mono':
            for name in ('ln_rv_d','ln_rv_w','ln_rv_m'):
                if name in self.current_names:mono[self.current_names.index(name)]=1
            mono[self.current_names.index('ln_HAR_b')]=-1
        before=time.time()
        if lib=='lgb':
            p=dict(learning_rate=.03,num_leaves=31,min_data_in_leaf=100,lambda_l2=10.,feature_fraction=.8,
                   bagging_fraction=.8,bagging_freq=1,num_threads=2,verbosity=-1,seed=seed,force_col_wise=True)
            if warm:
                for dest,source in [('num_leaves','num_leaves'),('learning_rate','learning_rate'),('min_data_in_leaf','min_data_in_leaf'),('bagging_fraction','bagging_fraction'),('feature_fraction','feature_fraction'),('lambda_l1','reg_alpha'),('lambda_l2','reg_lambda')]:
                    if source in warm and np.isfinite(warm[source]):p[dest]=int(warm[source]) if dest in ('num_leaves','min_data_in_leaf') else float(warm[source])
            p.update(objective='huber' if huber else 'regression',metric='huber' if huber else 'l2')
            if style=='mono':p['monotone_constraints']=mono.tolist()
            if style=='linear':p.update(linear_tree=True,linear_lambda=1.)
            if objective=='qlike':p.update(objective=lambda pred,ds:qlike_grad(pred,ds,c),metric='None')
            def dataset(mask):return lgb.Dataset(xx[mask],label=y[mask],weight=w[mask],init_score=base[mask] if objective=='qlike' else None,free_raw_data=True,params={'linear_tree':style=='linear'})
            ds=dataset(ti);dv=dataset(vi)
            metric=None
            if objective=='qlike':
                def metric(pred,ds):
                    off=ds.get_init_score();f=off+c*(pred-off);r=np.exp(ds.get_label()-f)
                    return 'QLIKE',float(np.mean(r-np.log(r)-1)),False
            first=lgb.train(p,ds,300,valid_sets=[dv],callbacks=[lgb.early_stopping(20,verbose=False)],feval=metric)
            trees=max(1,first.best_iteration)
            record(tag+lib,h,cut,mode,measure,seed,int(ti.sum()),b.loc[ti,f'_end{h}'].max(),phase='inner_ES',detail={'trees':trees,'style':style,'objective':objective})
            model=lgb.train(p,dataset(tr),trees)
            pred=model.predict(xx,raw_score=True)
        else:
            p=dict(tree_method='hist',max_depth=5,eta=.05,min_child_weight=20,reg_lambda=10.,subsample=.8,colsample_bytree=.8,nthread=2,seed=seed,verbosity=0,
                   objective='reg:pseudohubererror' if huber else 'reg:squarederror',eval_metric='mphe' if huber else 'rmse')
            if warm:
                for dest,source in [('max_depth','max_depth'),('eta','learning_rate'),('min_child_weight','min_child_weight'),('reg_lambda','reg_lambda'),('alpha','reg_alpha'),('subsample','subsample'),('colsample_bytree','colsample_bytree')]:
                    if source in warm and np.isfinite(warm[source]):p[dest]=int(warm[source]) if dest=='max_depth' else float(warm[source])
            if style=='mono':p['monotone_constraints']='('+','.join(map(str,mono))+')'
            ds=xgb.DMatrix(xx[ti],label=y[ti],weight=w[ti]);dv=xgb.DMatrix(xx[vi],label=y[vi])
            first=xgb.train(p,ds,300,evals=[(dv,'past')],early_stopping_rounds=20,verbose_eval=False)
            trees=int(first.best_iteration)+1
            record(tag+lib,h,cut,mode,measure,seed,int(ti.sum()),b.loc[ti,f'_end{h}'].max(),phase='inner_ES',detail={'trees':trees,'style':style})
            model=xgb.train(p,xgb.DMatrix(xx[tr],label=y[tr],weight=w[tr]),trees)
            pred=model.predict(xgb.DMatrix(xx),output_margin=True)
        record(tag+lib,h,cut,mode,measure,seed,int(tr.sum()),b.loc[tr,f'_end{h}'].max(),phase='refit',detail={'trees':trees,'style':style,'seconds':time.time()-before,'objective':objective})
        return pred.astype(float)

    def raw(self,h,spec,mode='annual',measure='original',seed=42):
        lockkey=hashlib.sha256(stable([h,spec,mode,measure,seed]).encode()).hexdigest()[:18]
        with (OUT/'cache'/f'{lockkey}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            return self._raw(h,spec,mode,measure,seed)

    def _raw(self,h,spec,mode='annual',measure='original',seed=42):
        spec={k:v for k,v in spec.items() if k in ('core','lib','style','c','normalize','k9','k10')}
        core=spec['core'];c=float(spec.get('c',1.))
        if core=='K9' and c!=1:
            parent={**spec,'c':1.};r=self.raw(h,parent,mode,measure,seed)
            hb=np.log(self.har_b(h,mode,measure))
            return hb+c*(r-hb)
        key=stable([self.contract,h,spec,mode,measure,seed]);digest=hashlib.sha256(key.encode()).hexdigest()[:18]
        if key in self.rawcache:return self.rawcache[key]
        path=OUT/'cache'/f'{digest}.parquet'
        if path.exists():
            r=pd.read_parquet(path,columns=['raw'])['raw'].values;self.rawcache[key]=r;return r
        if core in ('STACK','MOE','DIST','MLP'):
            from gl3_r56_reserve import reserve_raw
            r=reserve_raw(self,h,spec,mode,measure,seed)
        elif core in ('HAR','HARQ','HARCJ'):
            r=self.har_series(h,mode,measure,kind=core)
        elif core=='HXP':
            b=self.data.frames[measure];r=np.full(len(b),np.nan)
            for cut,end in periods(mode):
                m=(b.date>=cut)&(b.date<end);v=self.hxp_current(h,mode,measure,cut);r[m]=v[m]
        elif core=='FIN' and h==22:r=self.har_series(h,mode,measure)
        else:
            b=self.data.frames[measure];r=np.full(len(b),np.nan)
            partial=path.with_name(path.stem+'_partial.parquet')
            if partial.exists():r=pd.read_parquet(partial,columns=['raw'])['raw'].values
            for cut,end in periods(mode):
                va=((b.date>=cut)&(b.date<end)).values
                if np.isfinite(r[va]).all():continue
                normalized=core=='PANEL'
                _,y,tr,w=self.context(h,mode,measure,cut,normalized)
                x,names,base=self.features(h,mode,measure,cut,normalized,normalized)
                self.current_names=names
                logy=np.log(np.clip(y,1e-12,None))
                if core=='FIN':
                    keep=[i for i,n in enumerate(names) if n!='ln_HAR_b']
                    x=x[:,keep]
                    if h==1:
                        yy=logy.copy();vals=pd.Series(yy[tr]).groupby(b.loc[tr,'ticker'].values).quantile([.01,.99]).unstack()
                        yy[tr]=np.clip(yy[tr],b.loc[tr,'ticker'].map(vals[.01]),b.loc[tr,'ticker'].map(vals[.99]))
                        pp=[self.boost(h,mode,measure,cut,seed,'xgb',x,yy,tr,w,warm=p,tag='FIN_') for p in self.data.fin1params]
                    else:
                        off=self.har_current(h,mode,measure,cut,kind='HUBER')
                        pp=[off+self.boost(h,mode,measure,cut,seed,lib,x,logy-off,tr,w,warm=self.data.fin5params['lgbm' if lib=='lgb' else 'xgb'],huber=True,tag='FIN_') for lib in ('lgb','xgb')]
                    pred=np.log(np.mean([level(z) for z in pp],axis=0))
                else:
                    lib=spec.get('lib','lgb');style=spec.get('style','a')
                    if core=='K10':
                        g=self.boost(h,mode,measure,cut,seed,'lgb',x,logy,tr,w,style,objective='qlike',base=base,c=c,tag=core+'_')
                        pred=base+c*g
                    elif core in ('K9','PANEL'):
                        pred=base+self.boost(h,mode,measure,cut,seed,lib,x,logy-base,tr,w,style,tag=core+'_')
                    else:raise ValueError(core)
                if normalized:pred+=np.log(b['_scale'].values)
                r[va]=pred[va]
                checkpoint=self.data.a[['date','ticker']].copy();checkpoint['raw']=r.astype(np.float32)
                temporary=partial.with_suffix('.tmp');checkpoint.to_parquet(temporary,index=False);temporary.replace(partial)
        r=r.astype(np.float32)
        z=self.data.a[['date','ticker']].copy();z['raw']=r;z.to_parquet(path,index=False)
        partial=path.with_name(path.stem+'_partial.parquet')
        if partial.exists():partial.unlink()
        self.rawcache[key]=r
        return r

    def predict(self,h,recipe,seed=42):
        mode=recipe.get('mode','annual');measure=recipe.get('measure','original')
        bias=recipe.get('bias','none' if recipe['core']=='K10' else 'scalar')
        key=stable([h,recipe,seed,self.contract])
        if key in self.predcache:return self.predcache[key]
        if recipe.get('bagging',False):
            raw=np.mean([self.raw(h,recipe,mode,measure,s) for s in range(42,52)],axis=0)
        else:raw=self.raw(h,recipe,mode,measure,seed)
        p,v=self.calibrate(h,mode,measure,raw,bias)
        p=p.astype(np.float32)
        self.predcache[key]=p
        return p
