# -*- coding: utf-8 -*-
"""Г17 (У30): бустинг на остатке HAR-J(б) по группам k-means; Г18 (У31): потикерный HAR-J со сжатием к общему + панельный бустинг на остатке. Параметры и признаки — Г8."""
import json
import numpy as np
import pandas as pd
import gl3_r6b_g8 as G8
import gl3_r6b_retrain_base as B
import gl3_r6b_retrain_feat as F

LAMS=(.25,.5,.75)


def top25_x(d):
    feats=json.loads((G8.TAB/'features_top25.json').read_text())['features'];x,names=d.static_x(5,'original');return x[:,[names.index(f) for f in feats]]


def harj_x(a):return np.nan_to_num(np.log(a[G8.HARJ].values.astype(float).clip(1e-10)),nan=0.,posinf=0.,neginf=0.)


def shrunk_coefs(X,ly,tick,mask,lam):
    """OLS по mask: общий β_pool и потикерный β_i; β*_i = λ·β_i + (1−λ)·β_pool. mask — только строки с концом цели < cut."""
    Z=np.column_stack([np.ones(len(X)),X]);tick=np.asarray(tick);pool=np.linalg.lstsq(Z[mask],ly[mask],rcond=None)[0];out={}
    for t in np.unique(tick):
        m=mask&(tick==t);out[t]=lam*np.linalg.lstsq(Z[m],ly[m],rcond=None)[0]+(1-lam)*pool if m.any() else pool
    return pool,out


def ln_forecast(X,tick,coefs):
    Z=np.column_stack([np.ones(len(X)),X]);tick=np.asarray(tick);ln=np.full(len(X),np.nan)
    for t,b in coefs.items():
        m=tick==t
        with np.errstate(all='ignore'):ln[m]=Z[m]@b  # Accelerate (macOS) ставит ложные флаги FP в matmul; конечность проверяется ниже
    assert np.isfinite(ln).all(),'прогноз HAR-J не конечен'
    return ln


def g18_bases(a,X,lam,steps=range(2016,2022)):
    """Как G8.harj_bases, но коэффициенты — β* (У31); s² sm — общий, по ошибкам трёх прошлых val-лет с концом цели < vy. -> {vy: (ln-смещение, n, max конец цели)}."""
    y5=a._y5.values;ly=np.log(y5+1e-10);errs={};out={};first=min(steps)
    for vy in steps:
        tp=G8.harj_trmask(a,vy,True);_,co=shrunk_coefs(X,ly,a.ticker.values,tp,lam);ln=ln_forecast(X,a.ticker.values,co)
        sel=(a.year.values==vy)&np.isfinite(y5)&(y5>1e-12);e=ly-np.log(np.clip(np.exp(ln),1e-10,None))
        errs[vy]=pd.DataFrame(dict(e=e[sel],te=a._end5.values[sel]));P=pd.concat([errs[k] for k in range(max(first,vy-3),vy)]) if vy>first else None
        if P is not None:P=P[P.te<np.datetime64(f'{vy}-01-01')]
        s2=float(np.var(P.e.values)) if P is not None and len(P)>=30 else 0.
        out[vy]=(ln+.5*s2,int(tp.sum()),a.loc[tp,'_end5'].max())
    return out


def g17_year(a,x,y,hc,Y,labels):
    cut,full,tr,va=G8.year_masks(a,y,Y);pr=(a.year==Y).values&np.isfinite(hc);out=np.full(len(a),np.nan);k=labels.nunique()
    for g in sorted(labels.unique()):
        tks=list(labels.index[labels==g]);t=a.ticker.isin(tks).values
        m,fv=B.es_boost(x,y,tr&t,va&t,full&t,a,cut,f'G17_k{k}',dict(year=Y,group=int(g),tickers=tks))
        s2=float(np.var(y[va&t]-fv));sel=pr&t;out[sel]=hc[sel]*np.exp(m.predict(x[sel])+s2/2)
    return out


def run_g17(d,chars,bases,x):
    a=d.a;rows=[];res={k:np.full(len(a),np.nan) for k in (3,4)}
    for Y in B.YEARS:
        hc=np.exp(bases[Y][0]);y=G8.residual(a,hc);m=(a.year==Y).values
        for k in (3,4):
            lab=F.cluster_table(chars,f'{Y}-01-01',k);rows+=[dict(year=Y,k=k,ticker=t,group=int(g)) for t,g in lab.items()]
            B.disk(f'G17 k{k} {Y}');res[k][m]=g17_year(a,x,y,hc,Y,lab)[m]
    return res,pd.DataFrame(rows)


def run_g18(d,x):
    a=d.a;X=harj_x(a);res={}
    for lam in LAMS:
        bases=g18_bases(a,X,lam);p=np.full(len(a),np.nan)
        for vy,(b,n,mx) in bases.items():B.record('G18_HARJ_shrunk',f'{vy}-01-01',n,mx,'fit',dict(lam=lam,regressors=G8.HARJ))
        for Y in B.YEARS:
            hc=np.exp(bases[Y][0]);y=G8.residual(a,hc);cut,full,tr,va=G8.year_masks(a,y,Y);pr=(a.year==Y).values&np.isfinite(hc)
            B.disk(f'G18 {lam} {Y}');m,fv=B.es_boost(x,y,tr,va,full,a,cut,'G18_panel',dict(year=Y,lam=lam))
            s2=float(np.var(y[va]-fv));p[pr]=hc[pr]*np.exp(m.predict(x[pr])+s2/2)
        res[lam]=p
    return res
