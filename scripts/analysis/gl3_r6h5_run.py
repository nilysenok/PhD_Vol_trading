# -*- coding: utf-8 -*-
"""One h5 round: closed six families, strict mean-improvement admission."""
import json,hashlib,traceback
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import Ridge
from gl3_r6h5_data import Data,TAB,OUT,REG,ROOT,YEARS,READS,disk
from gl3_r6h5_engine import Engine,B
from gl3_r56_data import periods,mask_train


NUMERICAL_CONTRACT='h5-clean-contract-v4'

def canonical(z):return json.dumps(z,sort_keys=True,separators=(',',':'))


class Study:
    def __init__(self):
        self.d=Data();self.e=Engine(self.d);self.rows=[];self.rec={};self.pcache={};self.rawcache={};self.scorecache={};self.status='running';self.nomination=None
        d=self.d;a=d.a
        self.inputs={k:d.saved(REG/'gl3_r3/dev/predictions_h5.parquet',c) for k,c in [('FIN','rv_pred'),('HAR','har_corr')]}
        self.rawinputs={k:np.log(d.saved(REG/'gl3_r3/dev/predictions_h5.parquet',c)) for k,c in [('FIN','raw'),('HAR','har_raw')]}
        self.inputs['HXP']=d.saved(REG/'D22/HXP_5_val.parquet','rv_pred_sm');self.rawinputs['HXP']=np.log(d.saved(REG/'D22/HXP_5_val.parquet','rv_pred'))
        self.inputs['K9']=d.old('K9_lgb_a');self.inputs['K10']=d.old('K10');self.rawinputs['K10']=np.log(self.inputs['K10'])
        self.inputs['CJ']=d.saved(REG/'vyr/ref_harcj_d0/REF_HARCJ_d0_h5_val.parquet','rv_pred')
        self.inputs['ITER']=d.saved(REG/'D26/x6_h5_s42_val.parquet','rv_pred')
        # Warmup for K9 is reconstructed from the stored c=.75 raw forecast, without fitting another hybrid.
        rb=d.saved(REG/'gl3_r5_r6/winner_h5_dev.parquet','raw_log_prediction');hb=np.log(self.e.har_b(5,'annual','original'))
        self.rawinputs['K9']=hb+(rb-hb)/.75
        warm=a.year.eq(2016).values
        self.inputs['K9'][warm]=np.exp(self.rawinputs['K9'][warm])
        test=self.e.scalar(self.rawinputs['K9']);good=np.isfinite(self.inputs['K9'])&~warm
        assert np.max(np.abs(np.log(test[good]/self.inputs['K9'][good])))<5e-6
        # Only the missing2016 K10 warmup is replayed; saved2017–2021 expert forecasts stay fixed.
        previous=B.periods
        try:
            B.periods=lambda mode:[(pd.Timestamp('2016-01-01'),pd.Timestamp('2017-01-01'))]
            prior=list(d.basecols[5]);d.basecols[5]=json.loads((ROOT/'docs/tables/gl3_r5_r6/frozen_warmup_configuration.json').read_text())['feature_names']['5']
            w=Engine(d);w.contract='r6h5-K10warm-fixed';p=w.predict(5,{'core':'K10','lib':'lgb','style':'a','c':1.,'bias':'none'})
            self.inputs['K10'][warm]=p[warm];self.rawinputs['K10'][warm]=np.log(p[warm])
        finally:B.periods=previous;d.basecols[5]=prior
        for label,p in self.inputs.items():
            assert np.isfinite(p[d.idx]).all(),label
        # Correct the recursive h1-path mean only from past h5 residuals, no future realized path.
        self.inputs['ITER']=self.e.scalar(np.log(self.inputs['ITER']))
        (TAB/'reads.json').write_text(json.dumps(READS,ensure_ascii=False,indent=2))

    def save(self):
        pd.DataFrame(self.rows).to_csv(TAB/'dev_candidates.csv',index=False)
        (TAB/'selection.json').write_text(json.dumps(dict(status=self.status,recipes=self.rec,winner=self.nomination,control_read=False),ensure_ascii=False,indent=2))

    def predict(self,spec):
        key=canonical([NUMERICAL_CONTRACT,spec])
        if key in self.pcache:return self.pcache[key],self.rawcache[key]
        path=OUT/(hashlib.sha256(key.encode()).hexdigest()[:18]+'.parquet')
        if path.exists():
            z=pd.read_parquet(path);p=z.prediction.values;raw=z.raw.values
        else:
            p,raw=self.compute(spec);p=p.astype('float32');raw=raw.astype('float32')
            z=self.d.a[['date','ticker']].copy();z['prediction']=p;z['raw']=raw;z.to_parquet(path,index=False)
        self.pcache[key]=p;self.rawcache[key]=raw;return p,raw

    def compute(self,s):
        disk('candidate '+canonical(s));d=self.d;a=d.a;core=s['core']
        if core=='G1':
            raw=self.e.bank(s['variant'],s.get('seed',42));return self.e.scalar(raw),raw
        if core=='G4':
            raw=self.e.panel(seed=s.get('seed',42),ticker=s.get('ticker',False));return self.e.scalar(raw),raw
        if core in ('G2','G3') and s.get('cat_inputs',False):
            seed=s.get('seed',42);fin,rf=self.predict({'core':'G1','variant':'final_fixed_ticker','seed':seed})
            oldnames=d.basecols[5];oldcat=getattr(d,'cat_ticker',False);contract=self.e.contract
            try:
                d.basecols[5]=json.loads((ROOT/'docs/tables/gl3_r5_r6/frozen_warmup_configuration.json').read_text())['feature_names']['5']
                d.cat_ticker=True;self.e.contract=contract+'-K9ticker'
                rk=self.e.raw(5,{'core':'K9','lib':'lgb','style':'a','c':1.},seed=seed);k9=self.e.scalar(rk)
            finally:d.basecols[5]=oldnames;d.cat_ticker=oldcat;self.e.contract=contract
            oldp=self.inputs;oldr=self.rawinputs
            try:
                self.inputs={**oldp,'FIN':fin,'K9':k9};self.rawinputs={**oldr,'FIN':rf,'K9':rk}
                return self.compute({k:v for k,v in s.items() if k!='cat_inputs'})
            finally:self.inputs=oldp;self.rawinputs=oldr
        if core=='G2':
            expert=s['expert'];ep=self.inputs[expert];fin=self.inputs['FIN'];rawfin=self.rawinputs['FIN'];rawep=self.rawinputs[expert]
            xx=np.column_stack([np.ones(len(a)),a._ln_close_RVI,a._ln_rv_w]);w=a._state.values.astype(float)
            if s['gate']=='logistic':
                w=np.full(len(a),np.nan)
                for cut,end in periods('annual'):
                    tr=mask_train(a,5,cut,'annual',d.y(5,'original'))&np.isfinite(ep)&np.isfinite(fin)&np.isfinite(xx).all(axis=1)
                    dates=np.sort(a.loc[tr,'date'].unique())
                    if len(dates)>252:tr&=(a.date>=pd.Timestamp(dates[-252])).values
                    va=((a.date>=cut)&(a.date<end)).values
                    if tr.sum()<100:w[va]=a.loc[va,'_state'].values;continue
                    mean=xx[tr,1:].mean(0);sd=xx[tr,1:].std(0);sd[sd<1e-8]=1.;zz=xx.copy();zz[:,1:]=np.nan_to_num((xx[:,1:]-mean)/sd)
                    y=d.y(5,'original')[tr];f=fin[tr];e=ep[tr]
                    def fg(beta):
                        g=expit(zz[tr]@beta);p=f+g*(e-f);r=y/p
                        loss=np.mean(r-np.log(r)-1)+.5e-4*np.sum(beta[1:]**2)
                        grad=zz[tr].T@((1-r)*(e-f)/p*g*(1-g))/len(y)+np.r_[0.,1e-4*beta[1:]]
                        return loss,grad
                    fit=minimize(fg,np.zeros(3),method='BFGS',jac=True,options={'maxiter':200,'gtol':1e-8})
                    w[va]=expit(zz[va]@fit.x)
                    B.record('G2_logistic',5,cut,'annual','original',42,int(tr.sum()),a.loc[tr,'_end5'].max(),detail={'expert':expert,'coef':fit.x.tolist(),'optimizer_success':bool(fit.success)})
            p=(1-w)*fin+w*ep;raw=np.log((1-w)*np.exp(rawfin)+w*np.exp(rawep));return p,raw
        if core=='G3':
            keys=['FIN','HAR','CJ','HXP','K9','ITER'];x=np.column_stack([np.log(self.inputs[k]) for k in keys]);raw=np.full(len(a),np.nan)
            for cut,end in periods(s['mode']):
                tr=mask_train(a,5,cut,s['mode'],d.y(5,'original'))&np.isfinite(x).all(axis=1)
                va=((a.date>=cut)&(a.date<end)).values
                if tr.sum()<100:raw[va]=np.log(self.inputs['FIN'][va]);continue
                m=Ridge(alpha=1.,positive=True,max_iter=3000,tol=1e-8).fit(x[tr],np.log(d.y(5,'original')[tr]));ok=va&np.isfinite(x).all(axis=1);raw[ok]=m.predict(x[ok])
                B.record('G3_positive_ridge',5,cut,s['mode'],'original',42,int(tr.sum()),a.loc[tr,'_end5'].max(),detail={'weights':m.coef_.tolist(),'intercept':float(m.intercept_)})
            return self.e.scalar(raw,mode=s['mode']),raw
        if core=='G5':
            parent=s['parent']
            if parent['core'] in ('G2','G3') and not parent.get('cat_inputs',False):
                # The fixed forecast inputs and convex/logistic estimator are deterministic.
                return self.predict(parent)
            parts=[self.predict({**parent,'seed':seed}) for seed in range(42,52)]
            raw=np.mean([z[1] for z in parts],axis=0);return self.e.scalar(raw),raw
        if core=='G6':
            _,raw=self.predict(s['parent']);return self.e.scalar(raw,bias='conditional'),raw
        if core=='FIXED_FIN':return self.inputs['FIN'],self.rawinputs['FIN']
        if core=='G7':
            parent,raw=self.predict(s['parent']);out=np.full(len(a),np.nan)
            for cut,end in periods('annual'):
                va=((a.date>=cut)&(a.date<end)).values
                tr=mask_train(a,5,cut,'annual',d.y(5,'original'))&(a.date>=cut-pd.DateOffset(years=3)).values
                tr&=np.isfinite(raw if s['kind']=='s2' else parent)
                if tr.sum()<30:out[va]=np.exp(raw[va]) if s['kind']=='s2' else parent[va];continue
                values=np.log(d.y(5,'original')[tr])-raw[tr] if s['kind']=='s2' else d.y(5,'original')[tr]/parent[tr]
                pooled=float(np.var(values)) if s['kind']=='s2' else float(np.mean(values))
                frame=pd.DataFrame({'ticker':a.loc[tr,'ticker'].values,'v':values})
                per=frame.groupby('ticker').v.agg(['count','var','mean'])
                if s['kind']=='s2':
                    # ddof0 matches exp(s²/2) in the existing scalar correction.
                    local=frame.groupby('ticker').v.var(ddof=0);factor=np.exp(a.loc[va,'ticker'].map(local).fillna(pooled).values/2)
                    out[va]=np.exp(raw[va])*factor
                else:
                    local=per['mean'].where(per['count']>=30,pooled)
                    factor=s['w']*a.loc[va,'ticker'].map(local).fillna(pooled).values+(1-s['w'])*pooled
                    out[va]=parent[va]*factor
                B.record('G7_'+s['kind'],5,cut,'annual','original',42,int(tr.sum()),a.loc[tr,'_end5'].max(),detail={'w':s.get('w'),'pooled':pooled,'n_tickers':len(per),'per_ticker':{k:float(v) for k,v in local.items()}})
            return out,raw
        raise ValueError(core)

    def add(self,label,recipes,eligible=True):
        if 'core' in recipes:recipes={str(y):recipes for y in list(YEARS)+['all']}
        p=np.full(len(self.d.a),np.nan)
        for year in YEARS:
            z,_=self.predict(recipes[str(year)]);m=self.d.a.year.eq(year).values;p[m]=z[m]
        score=self.d.score5(p);base=self.d.base
        n=sum(score[str(y)]<base[str(y)]-1e-12 for y in YEARS);stress=all(score[k]<=base[k]+1e-12 for k in ('stress2020','stressRVI'));mean=score['mean_QLIKE']<base['mean_QLIKE']-1e-12
        row=dict(candidate=label,eligible=eligible,**score,better_years=n,stress_pass=bool(stress),better_mean=bool(mean),admitted=bool(eligible and n>=3 and stress and mean))
        self.rows.append(row);self.rec[label]=recipes
        q=self.d.ref[['date','ticker']].copy();q['prediction']=p[self.d.idx].astype('float32');q.to_parquet(OUT/f'{label}_dev.parquet',index=False)
        self.save();print(json.dumps(row),flush=True);return label

    def best(self,keys,held=None):
        columns=[str(y) for y in YEARS if y!=held]
        def metric(k):
            cache=(k,held)
            if cache not in self.scorecache:
                p,_=self.predict(self.rec[k]['all' if held is None else str(held)]);q=self.d.score5(p)
                self.scorecache[cache]=np.mean([q[c] for c in columns])
            return self.scorecache[cache],k
        return min(keys,key=metric)

    def choose(self,keys,wrap=None):
        out={}
        for y in list(YEARS)+['all']:
            k=self.best(keys,None if y=='all' else y);p=self.rec[k][str(y)]
            out[str(y)]=wrap(p) if wrap else p
        return out

    def run(self):
        keys=[self.add('G1_'+v,dict(core='G1',variant=v)) for v in ('standard','regularized')]
        for gate in ('hard','logistic'):
            # Choose the stress expert on the other4years; the expert series themselves are frozen prior-round inputs.
            recipes={}
            for year in list(YEARS)+['all']:
                cols=[str(y) for y in YEARS if y!=year]
                expert=min(('K9','K10','HAR','HXP'),key=lambda k:np.mean([self.d.score5(self.inputs[k])[c] for c in cols]))
                recipes[str(year)]=dict(core='G2',gate=gate,expert=expert)
            keys.append(self.add('G2_'+gate,recipes))
        for mode in ('annual','quarterly'):keys.append(self.add('G3_'+mode,dict(core='G3',mode=mode)))
        keys.append(self.add('G4',dict(core='G4')))
        keys.append(self.add('G5',self.choose(keys,lambda p:dict(core='G5',parent=p))))
        keys.append(self.add('G6',self.choose(keys,lambda p:dict(core='G6',parent=p))))
        for base in ('FIN','BEST'):
            parents={str(y):{'core':'FIXED_FIN'} for y in list(YEARS)+['all']} if base=='FIN' else self.choose(keys)
            self.add('G7a_'+base,{y:{'core':'G7','kind':'s2','parent':r} for y,r in parents.items()})
            grid=[]
            for w in (.25,.5,.75):
                grid.append(self.add('G7b_'+base+'_w'+str(w),{y:{'core':'G7','kind':'qlike','w':w,'parent':r} for y,r in parents.items()},eligible=False))
            self.add('G7b_'+base+'_LOYO',self.choose(grid))
            def categorical(r):
                r=dict(r);c=r['core']
                if c=='FIXED_FIN':return {'core':'G1','variant':'final_fixed_ticker'}
                if c=='G1':r['variant']=r['variant']+'_ticker'
                elif c=='G4':r['ticker']=True
                elif c in ('G2','G3'):r['cat_inputs']=True
                elif c in ('G5','G6'):r['parent']=categorical(r['parent'])
                else:raise ValueError(c)
                return r
            self.add('G7c_'+base,{y:categorical(r) for y,r in parents.items()})
        passed=[r for r in self.rows if r['admitted']]
        self.nomination=min(passed,key=lambda r:(r['mean_QLIKE'],r['candidate']))['candidate'] if passed else None
        self.status='dev_complete_no_control_opening';self.save()


if __name__=='__main__':
    s=Study()
    try:s.run()
    except BaseException:
        s.status='incomplete';s.save();(TAB/'failure.txt').write_text(traceback.format_exc());raise
