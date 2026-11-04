# -*- coding: utf-8 -*-
"""The closed reserve models. Called only if no round5 candidate passes."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import Ridge
from gl3_r56_data import periods,mask_train,weights


def mlp(e,h,mode,measure,cut,seed,x,y,tr,w):
    import torch
    from gl3_r56_engine import record
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed)
    b=e.data.frames[measure];dates=sorted(b.loc[tr,'date'].unique());edge=dates[max(1,len(dates)-126)]
    ti=tr&(b.date<edge).values&(b[f'_end{h}']<edge).values;vi=tr&(b.date>=edge).values
    mean=np.nanmean(x[ti],axis=0);sd=np.nanstd(x[ti],axis=0);sd[~np.isfinite(sd)|(sd<1e-6)]=1.
    xx=np.nan_to_num((x-mean)/sd,nan=0.,posinf=0.,neginf=0.).astype('float32')
    xx=np.clip(xx,-20,20);yy=y.astype('float32');ww=w.astype('float32')
    def fresh():
        torch.manual_seed(seed)
        return torch.nn.Sequential(torch.nn.Linear(x.shape[1],32),torch.nn.ReLU(),torch.nn.Linear(32,16),torch.nn.ReLU(),torch.nn.Linear(16,1))
    def epoch(model,opt,indices,ep):
        rng=np.random.RandomState(seed+ep);indices=rng.permutation(indices)
        model.train()
        for ids in np.array_split(indices,max(1,int(np.ceil(len(indices)/512)))):
            tx=torch.from_numpy(xx[ids]);ty=torch.from_numpy(yy[ids]);tw=torch.from_numpy(ww[ids])
            opt.zero_grad();loss=(tw*(model(tx).ravel()-ty)**2).mean();loss.backward();opt.step()
    model=fresh();opt=torch.optim.Adam(model.parameters(),lr=.001)
    best=np.inf;wait=0;epochs=1
    for ep in range(100):
        epoch(model,opt,np.flatnonzero(ti),ep)
        with torch.no_grad():v=float(((model(torch.from_numpy(xx[vi])).ravel()-torch.from_numpy(yy[vi]))**2).mean())
        if v<best-1e-6:best=v;epochs=ep+1;wait=0
        else:wait+=1
        if wait>=10:break
    record('R5_MLP',h,cut,mode,measure,seed,int(ti.sum()),b.loc[ti,f'_end{h}'].max(),phase='inner_ES',detail={'epochs':epochs})
    # Standardization is then refitted on the full past training window.
    mean=np.nanmean(x[tr],axis=0);sd=np.nanstd(x[tr],axis=0);sd[~np.isfinite(sd)|(sd<1e-6)]=1.
    xx=np.clip(np.nan_to_num((x-mean)/sd,nan=0.,posinf=0.,neginf=0.),-20,20).astype('float32')
    model=fresh();opt=torch.optim.Adam(model.parameters(),lr=.001)
    for ep in range(epochs):epoch(model,opt,np.flatnonzero(tr),ep)
    with torch.no_grad():pred=model(torch.from_numpy(xx)).numpy().ravel().astype(float)
    record('R5_MLP',h,cut,mode,measure,seed,int(tr.sum()),b.loc[tr,f'_end{h}'].max(),phase='refit',detail={'epochs':epochs})
    return pred


def reserve_raw(e,h,spec,mode,measure,seed):
    from gl3_r56_engine import record,level
    b=e.data.frames[measure];n=len(b);r=np.full(n,np.nan);core=spec['core']
    if core=='STACK':
        bases=[{'core':c} for c in ('FIN','HAR','HXP','HARQ','HARCJ')]
        bases+=spec.get('k9',[])
        if spec.get('k10'):bases+=[spec['k10']]
        # R5 has no standalone admission and is only evaluated here, five seeds.
        mlps=[e.raw(h,{'core':'MLP'},mode,measure,s) for s in range(42,47)]
        mlp_log=np.mean(mlps,axis=0);mlp_p,_=e.calibrate(h,mode,measure,mlp_log)
        xx=np.column_stack([np.log(e.predict(h,{**z,'mode':mode,'measure':measure})) for z in bases]+[np.log(mlp_p)])
        for cut,end in periods(mode):
            y=e.data.y(h,measure);tr=mask_train(b,h,cut,mode,y)&np.isfinite(xx).all(axis=1)
            va=((b.date>=cut)&(b.date<end)).values
            if tr.sum()<100:r[va]=xx[va,0];continue
            model=Ridge(alpha=1.,positive=True,fit_intercept=True,max_iter=2000,tol=1e-8).fit(xx[tr],np.log(y[tr]),sample_weight=weights(b,cut,mode)[tr])
            r[va]=model.predict(xx[va])
            record('R1_positive_ridge',h,cut,mode,measure,seed,int(tr.sum()),b.loc[tr,f'_end{h}'].max(),detail={'weights':model.coef_.tolist(),'intercept':float(model.intercept_)})
        return r
    # OOS log-mean residuals for the variance head, available only after maturity.
    mean_oof=np.full(n,np.nan)
    for cut,end in periods(mode):
        va=((b.date>=cut)&(b.date<end)).values
        _,y,tr,w=e.context(h,mode,measure,cut)
        x,names,base=e.features(h,mode,measure,cut);e.current_names=names
        logy=np.log(np.clip(y,1e-12,None));target=logy-base
        if core=='MLP':pred=base+mlp(e,h,mode,measure,cut,seed,x,target,tr,w)
        elif core=='MOE':
            gatex=np.column_stack([np.ones(n),b._ln_close_RVI,b._ln_rv_w]);good=np.isfinite(gatex).all(axis=1)
            gm=np.nanmean(gatex[tr,1:],axis=0);gs=np.nanstd(gatex[tr,1:],axis=0);gs[gs<1e-8]=1.
            gatex[:,1:]=np.nan_to_num((gatex[:,1:]-gm)/gs)
            resp=b._state.values.astype(float)*.9+.05;beta=np.array([0.,1.,0.])
            for it in range(3):
                calm=e.boost(h,mode,measure,cut,seed,'lgb',x,target,tr,w*(1-resp),tag=f'R3_calm_EM{it}_')
                stress=e.boost(h,mode,measure,cut,seed,'lgb',x,target,tr,w*resp,tag=f'R3_stress_EM{it}_')
                v=max(float(np.average((target[tr]-(calm[tr]*(1-resp[tr])+stress[tr]*resp[tr]))**2,weights=w[tr])),1e-4)
                posterior=expit(gatex@beta+((target-calm)**2-(target-stress)**2)/(2*v))
                posterior=np.clip(posterior,.01,.99)
                def fg(z):
                    q=expit(gatex[tr]@z);t=posterior[tr];ww=w[tr]
                    loss=np.sum(ww*(np.logaddexp(0,gatex[tr]@z)-t*(gatex[tr]@z)))/ww.sum()+.5e-3*np.sum(z[1:]**2)
                    grad=gatex[tr].T@(ww*(q-t))/ww.sum()+np.r_[0.,1e-3*z[1:]]
                    return loss,grad
                fit=minimize(fg,beta,jac=True,method='BFGS',options={'maxiter':100});beta=fit.x
                resp=np.nan_to_num(posterior,nan=.5)
                record('R3_logistic_gate',h,cut,mode,measure,seed,int(tr.sum()),b.loc[tr,f'_end{h}'].max(),detail={'iteration':it,'coef':beta.tolist()})
            g=expit(gatex@beta);pred=base+calm*(1-g)+stress*g
        elif core=='DIST':
            mu=base+e.boost(h,mode,measure,cut,seed,'lgb',x,target,tr,w,tag='R4_mean_')
            vt=tr&np.isfinite(mean_oof)
            if vt.sum()<300:
                # Warmup variance uses a chronologically held-out past tail, not in-sample errors.
                dates=sorted(b.loc[tr,'date'].unique());edge=dates[max(1,len(dates)-126)]
                train=tr&(b[f'_end{h}']<edge).values;vt=tr&(b.date>=edge).values
                inner=base+e.boost(h,mode,measure,cut,seed,'lgb',x,target,train,w,tag='R4_warmmean_')
                vv=(logy-inner)**2
            else:vv=(logy-mean_oof)**2
            v=e.boost(h,mode,measure,cut,seed,'lgb',x,vv,vt,w,tag='R4_variance_')
            pred=mu+np.maximum(v,0)/2;mean_oof[va]=mu[va]
        else:raise ValueError(core)
        r[va]=pred[va]
    return r
