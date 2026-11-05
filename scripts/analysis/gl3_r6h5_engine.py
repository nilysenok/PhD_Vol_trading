# -*- coding: utf-8 -*-
"""h5-only fresh estimators and purged Optuna30, never a control reader."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import json,time,hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.linear_model import HuberRegressor
import gl3_r56_engine as B
from gl3_r6h5_data import Data,OUT,TAB,ROOT,disk

B.disk_guard=disk;B.OUT=OUT;B.TAB=TAB;B.EVENTS=TAB/'fit_events.jsonl'
_record_base=B.record
def _record_h5(*args,**kwargs):
    kwargs['detail']={**(kwargs.get('detail') or {}),'round_contract':'h5-clean-contract-v4'}
    return _record_base(*args,**kwargs)
B.record=_record_h5
START=TAB/'start.json'
if not START.exists():START.write_text(json.dumps({'started':time.time(),'control_read':False}))
# Prior engine only uses this timestamp for budget initialization; override to2days below.
if not (TAB/'diagnostics_manifest.json').exists():(TAB/'diagnostics_manifest.json').write_text('{"control_read":false}')
SPACE={'xgb':[('max_depth',3,12,'int'),('learning_rate',.03,.2,'log'),('subsample',.5,1,'float'),('colsample_bytree',.4,1,'float'),('min_child_weight',1,100,'logint'),('reg_alpha',1e-3,10,'log'),('reg_lambda',1e-3,10,'log')],
       'lgb':[('num_leaves',15,255,'int'),('learning_rate',.03,.2,'log'),('bagging_fraction',.5,1,'float'),('feature_fraction',.4,1,'float'),('min_data_in_leaf',5,200,'logint'),('reg_alpha',1e-3,10,'log'),('reg_lambda',1e-3,10,'log')]}


def gap_boundary(a,cut,gap=5):
    dates=np.sort(a.date.unique());pos=np.searchsorted(dates,np.datetime64(cut),'left')
    assert pos>gap
    return pd.Timestamp(dates[pos-gap])


def purged(a,cut,valyear=None):
    y=a._y5.values;edge=gap_boundary(a,cut)
    full=(a.date<cut)&(a._end5<edge)&np.isfinite(y)&(y>1e-12)
    if valyear is None:return full.values
    vstart=pd.Timestamp(f'{valyear}-01-01');v_edge=gap_boundary(a,vstart)
    tr=full&(a.date<vstart)&(a._end5<v_edge)
    va=full&(a.date>=vstart)
    assert a.loc[tr,'_end5'].max()<v_edge and a.loc[va,'_end5'].max()<edge
    return tr.values,va.values,full.values


class Engine(B.Engine):
    def __init__(self,d):
        super().__init__(d);self.started=json.loads(START.read_text())['started'];self.deadline=self.started+2*86400
        self.contract='r6h5-v2-norm';self.banks={};self.selected={}

    def features(self,h,mode,measure,cut,normalize=False,sector=False):
        assert h==5
        x,names,base=super().features(h,mode,measure,cut,normalize,sector)
        if normalize:x[:,-2]-=np.log(self.data.a._scale.values)
        return x,names,base

    def context(self,h,mode,measure,cut,normalize=False):
        assert h==5;disk(f'fresh h5 {mode} {cut}')
        return super().context(h,mode,measure,cut,normalize)

    def panel(self,seed=42,ticker=False):
        old=getattr(self.data,'cat_ticker',False);contract=self.contract
        try:
            self.data.cat_ticker=ticker
            if ticker:self.contract=contract+'-ticker'
            return self.raw(5,{'core':'PANEL'},seed=seed)
        finally:self.data.cat_ticker=old;self.contract=contract

    def boost(self,h,mode,measure,cut,seed,lib,x,y,tr,w,style='a',objective='mse',base=None,c=1.,warm=None,huber=False,tag=''):
        if 'cat_ticker' not in self.current_names:
            return super().boost(h,mode,measure,cut,seed,lib,x,y,tr,w,style,objective,base,c,warm,huber,tag)
        assert lib=='lgb' and objective=='mse'
        a=self.data.a;dates=sorted(a.loc[tr,'date'].unique());edge=dates[max(1,len(dates)-126)]
        ti=tr&(a.date<edge).values&(a._end5<edge).values;vi=tr&(a.date>=edge).values
        cats=[self.current_names.index('cat_ticker')]
        def dataset(m):return lgb.Dataset(x[m],label=y[m],weight=w[m],categorical_feature=cats)
        params=dict(objective='regression',metric='l2',num_leaves=31,learning_rate=.03,min_data_in_leaf=100,lambda_l2=10.,bagging_fraction=.8,bagging_freq=1,feature_fraction=.8,num_threads=2,seed=seed,verbosity=-1,force_col_wise=True)
        first=lgb.train(params,dataset(ti),300,valid_sets=[dataset(vi)],callbacks=[lgb.early_stopping(20,verbose=False)])
        trees=max(1,first.best_iteration)
        B.record('G7c_'+tag+'lgb',5,cut,mode,measure,seed,int(ti.sum()),a.loc[ti,'_end5'].max(),phase='inner_ES',detail={'trees':trees,'ticker_category':True})
        model=lgb.train(params,dataset(tr),trees)
        B.record('G7c_'+tag+'lgb',5,cut,mode,measure,seed,int(tr.sum()),a.loc[tr,'_end5'].max(),phase='refit',detail={'trees':trees,'ticker_category':True})
        return model.predict(x,raw_score=True)

    def scalar(self,raw,bias='scalar',mode='annual'):
        return self.calibrate(5,mode,'original',raw,bias)[0].astype('float32')

    def bank(self,variant,seed=42):
        path=OUT/f'G1v3_{variant}_seed{seed}.parquet'
        if path.exists():return pd.read_parquet(path).raw.values
        d=self.data;a=d.a;n=len(a);raw=np.full(n,np.nan);x,names=d.raw_features()
        for year in range(2016,2022):
            disk(f'G1 {variant} {seed} {year}')
            if time.time()>self.deadline:raise TimeoutError('two-day budget exhausted; not a no-winner closure')
            cut=pd.Timestamp(f'{year}-01-01');tr,va,full=purged(a,cut,year-1)
            pred=(a.date>=cut)&(a.date<pd.Timestamp(f'{year+1}-01-01'))
            yy=np.log(d.y(5,'original')+1e-10)
            def offset(mask):
                hc=['rv_d','rv_w','rv_m','jump','jump_annualized','jump_lag1','jump_lag5','jump_lag22'];hx=np.nan_to_num(np.log(d.a[hc].values.clip(1e-10)));fit=HuberRegressor(max_iter=500).fit(hx[mask],yy[mask]);out=fit.predict(hx)
                B.record('G1_HuberHAR',5,cut,'annual','original',seed,int(mask.sum()),a.loc[mask,'_end5'].max(),detail={'variant':variant,'train_boundary':'inner' if np.array_equal(mask,tr) else 'refit'})
                return out
            inner=offset(tr);outer=offset(full)
            out=[]
            for lib in ('lgb','xgb'):
                base_variant=variant.removesuffix('_ticker');categorical=variant.endswith('_ticker')
                configpath=TAB/f'G1v3_{variant}_{lib}_{year}.json'
                packed=TAB/'selected_parameters.json'
                if not configpath.exists() and packed.exists():
                    store=json.loads(packed.read_text())
                    if configpath.name in store:configpath.write_text(json.dumps(store[configpath.name]))
                if not configpath.exists() and base_variant=='final_fixed':
                    frozen=json.loads((ROOT/'docs/tables/gl3_r5_r6/frozen_warmup_configuration.json').read_text())['fin5params']['lgbm' if lib=='lgb' else 'xgb']
                    p={k:(int(frozen[k]) if scale in ('int','logint') else float(frozen[k])) for k,lo,hi,scale in SPACE[lib]}
                    _,trees,_=self.g1fit(lib,p,x,yy,inner,tr,va,42,cut,variant,categorical=categorical)
                    configpath.write_text(json.dumps(dict(params=p,trees=trees,source='shared warmup2016 D12 parameters, no ticker tuning')))
                if not configpath.exists() and categorical:
                    previous=TAB/f'G1v3_{base_variant}_{lib}_{year}.json'
                    assert previous.exists(),previous
                    configpath.write_text(previous.read_text())
                if not configpath.exists():
                    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42));optuna.logging.set_verbosity(optuna.logging.WARNING)
                    best=np.inf;records=[]
                    for _ in range(30):
                        trial=study.ask();p={}
                        for k,lo,hi,scale in SPACE[lib]:p[k]=trial.suggest_int(k,lo,hi,log=scale=='logint') if scale in ('int','logint') else trial.suggest_float(k,lo,hi,log=scale=='log')
                        if variant=='regularized':
                            p['reg_lambda']*=4
                            if lib=='lgb':p.update(min_data_in_leaf=p['min_data_in_leaf']*4,max_depth=4,num_leaves=min(16,p['num_leaves']))
                            else:p.update(min_child_weight=p['min_child_weight']*4,max_depth=min(4,p['max_depth']))
                        pv,trees,pruned=self.g1fit(lib,p,x,yy,inner,tr,va,42,cut,variant,trial=trial.number,best=best)
                        ratio=d.y(5,'original')[va]/np.exp(np.clip(pv[va],-60,60));q=float(np.mean(ratio-np.log(ratio)-1))
                        if pruned:study.tell(trial,state=optuna.trial.TrialState.PRUNED)
                        else:study.tell(trial,q);best=min(best,q)
                        records.append(dict(year=year,variant=variant,lib=lib,trial=trial.number,params=p,trees=trees,qlike=q,pruned=pruned))
                        with (TAB/'optuna_trials.jsonl').open('a') as f:f.write(json.dumps(records[-1])+'\n')
                    winner=min([r for r in records if not r['pruned']],key=lambda r:r['qlike']);configpath.write_text(json.dumps(winner,indent=2))
                win=json.loads(configpath.read_text())
                pp,_,_=self.g1fit(lib,win['params'],x,yy,outer,full,None,seed,cut,variant,nround=win['trees'],categorical=categorical)
                out.append(np.exp(np.clip(pp[pred],-60,60)))
            raw[pred]=np.log(np.mean(out,axis=0))
        z=a[['date','ticker']].copy();z['raw']=raw.astype('float32');z.to_parquet(path,index=False)
        return z.raw.values

    def g1fit(self,lib,p,x,y,off,tr,va,seed,cut,variant,trial=None,best=np.inf,nround=None,categorical=False):
        actual=self.data.y(5,'original');a=self.data.a;names=self.data.raw_features()[1];ex=pd.read_csv(ROOT/'data/regen/Bv2b_excluded_features.csv');drop=set(ex[(ex.h==5)&(ex.model==('lgbm' if lib=='lgb' else 'xgb'))].feature) if cut.year>2016 else set();keep=[j for j,n in enumerate(names) if n not in drop];xx=np.nan_to_num(np.column_stack([x[:,keep],off]),nan=0.,posinf=0.,neginf=0.).astype('float32');target=y-off;pruned=[]
        if categorical:xx=np.column_stack([xx,self.data.a.ticker.map({t:i for i,t in enumerate(sorted(self.data.a.ticker.unique()))}).values]).astype('float32')
        cat=[xx.shape[1]-1] if categorical else []
        xkw={'feature_types':['q']*(xx.shape[1]-1)+['c'],'enable_categorical':True} if categorical else {}
        if lib=='lgb':
            params=dict(objective='huber',metric='huber',num_threads=2,verbosity=-1,seed=seed,force_col_wise=True,max_bin=255,bagging_freq=1,**p)
            train=lgb.Dataset(xx[tr],label=target[tr],categorical_feature=cat or 'auto');callbacks=[];valid=[]
            if va is not None:
                valid=[lgb.Dataset(xx[va],label=target[va],reference=train,categorical_feature=cat or 'auto')];callbacks=[lgb.early_stopping(30,verbose=False)]
                def prune(env):
                    if env.iteration==99 and np.isfinite(best):
                        z=off[va]+env.model.predict(xx[va],raw_score=True);r=self.data.y(5,'original')[va]/np.exp(np.clip(z,-60,60));q=np.mean(r-np.log(r)-1)
                        if q>1.2*best:pruned.append(True);raise lgb.callback.EarlyStopException(env.iteration,env.evaluation_result_list)
                callbacks.insert(0,prune)
            model=lgb.train(params,train,nround or 1000,valid_sets=valid,callbacks=callbacks)
            trees=int(model.best_iteration or nround or model.current_iteration());pp=off+model.predict(xx,raw_score=True)
        else:
            params=dict(objective='reg:pseudohubererror',eval_metric='mphe',tree_method='hist',max_bin=255,nthread=2,seed=seed,verbosity=0,**p)
            callbacks=[]
            if va is not None and np.isfinite(best):
                class Prune(xgb.callback.TrainingCallback):
                    def after_iteration(self,model,epoch,evals_log):
                        if epoch==99:
                            z=off[va]+model.predict(xgb.DMatrix(xx[va],**xkw),output_margin=True);r=actual[va]/np.exp(np.clip(z,-60,60))
                            if np.mean(r-np.log(r)-1)>1.2*best:pruned.append(True);return True
                        return False
                callbacks=[Prune()]
            model=xgb.train(params,xgb.DMatrix(xx[tr],label=target[tr],**xkw),nround or 1000,evals=[(xgb.DMatrix(xx[va],label=target[va],**xkw),'past')] if va is not None else [],early_stopping_rounds=30 if va is not None else None,verbose_eval=False,callbacks=callbacks)
            trees=int(getattr(model,'best_iteration',model.num_boosted_rounds()-1))+1;pp=off+model.predict(xgb.DMatrix(xx,**xkw),output_margin=True,iteration_range=(0,trees))
        B.record('G1_'+lib,5,cut,'annual','original',seed,int(tr.sum()),a.loc[tr,'_end5'].max(),phase='trial_ES' if va is not None else 'refit',detail={'numerical_contract':'D12_clean_v3','categorical_ticker':categorical,'variant':variant,'trial':trial,'trees':trees,'params':p,'pruned':bool(pruned)})
        return pp,trees,bool(pruned)
