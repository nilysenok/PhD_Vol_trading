# -*- coding: utf-8 -*-
"""Раунд 6б, h=5, Г8 (У19–У20): LightGBM на остатке HAR-J(б) по тикеру (Г8а) и панельный + поправка c_i (Г8б). DEV 2017–2021."""
import os
import sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import json,math,shutil,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import HuberRegressor
sys.path.insert(0,str(Path(__file__).resolve().parent))
import gl3_r6h5_data as D
from gl3_r6h5_engine import gap_boundary,purged
from gl3_r56_diagnostics import read,REG,ROOT,LAST
from gl3_r56_data import mask_train

OUT=REG/'gl3_r6b_h5/g8';TAB=ROOT/'docs/tables/gl3_r6b_h5/g8';EVENTS=TAB/'fit_events.jsonl';YEARS=tuple(range(2017,2022));LAMS=(126,504)
PARAMS=dict(objective='regression',metric='l2',max_depth=3,num_leaves=7,min_data_in_leaf=100,lambda_l2=10.,learning_rate=.03,feature_fraction=.8,
            bagging_fraction=.8,bagging_freq=1,num_threads=2,seed=42,verbosity=-1,force_col_wise=True)
LOGKEYS=('max_depth','num_leaves','min_data_in_leaf','lambda_l2','learning_rate','feature_fraction','bagging_fraction')


def disk(stage):
    TAB.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    out=subprocess.run(['df','-h','/'],capture_output=True,text=True,check=True).stdout
    with (TAB/'disk.log').open('a') as f:f.write(str(pd.Timestamp.now(tz='UTC'))+' '+stage+'\n'+out)
    if shutil.disk_usage('/').free<20*1024**3:raise RuntimeError('DISK_STOP: less than 20GiB free')


def record(kind,cut,n,maxend,phase,detail=None):
    assert pd.Timestamp(maxend)<pd.Timestamp(cut)<pd.Timestamp('2022-01-01')
    r=dict(component=kind,h=5,cutoff=str(cut),n=int(n),max_target_end=str(maxend),phase=phase,detail=detail or {},time_utc=pd.Timestamp.now(tz='UTC').isoformat())
    with EVENTS.open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')


def split(a,full,nval=126):
    """Внутренняя валидация — последние nval дат окна; purge по концу цели и embargo 5 дат до её начала (У10)."""
    dates=np.sort(a.loc[full,'date'].unique());vs=pd.Timestamp(dates[-nval]);edge=gap_boundary(a,vs)
    tr=full&(a.date<vs).values&(a._end5<edge).values;va=full&(a.date>=vs).values
    assert a.loc[tr,'_end5'].max()<edge<=vs
    return tr,va


def fit(x,y,tr,va,full,a,cut,kind,detail):
    """ES на (tr,va), затем refit на full с найденным числом деревьев; возвращает модель и прогноз ES-модели на va."""
    first=lgb.train(PARAMS,lgb.Dataset(x[tr],label=y[tr]),500,valid_sets=[lgb.Dataset(x[va],label=y[va])],callbacks=[lgb.early_stopping(30,verbose=False)])
    trees=max(1,first.best_iteration);det={**detail,'trees':trees,'params':{k:PARAMS[k] for k in LOGKEYS}}
    record(kind,cut,tr.sum(),a.loc[tr,'_end5'].max(),'inner_ES',det)
    model=lgb.train(PARAMS,lgb.Dataset(x[full],label=y[full]),trees)
    record(kind,cut,full.sum(),a.loc[full,'_end5'].max(),'refit',det)
    return model,first.predict(x[va],num_iteration=trees)


def shrink(tick,err,lam):
    """c_i = n_i/(n_i+λ)·m̄_i по логам ошибок внутренней валидации тикера (У20)."""
    g=pd.Series(np.asarray(err,float)).groupby(np.asarray(tick))
    n=g.size().astype(float);return n/(n+lam)*g.mean()


def residual(a,hc):return np.log(np.clip(a._y5.values,1e-12,None))-np.log(hc)
HARJ=['rv_d','rv_w','rv_m','jump','jump_annualized','jump_lag1','jump_lag5','jump_lag22']


def harj_trmask(a,vy,emb=True):
    """Обучение HAR-J на отсечке 1 января vy: год строки < vy, цель есть; при emb конец цели тоже < vy (как r2_nogi.trmask)."""
    m=(a.year.values<vy)&np.isfinite(a._y5.values)
    return m&(a._end5.dt.year.values<vy) if emb else m


def harj_bases(a,X,steps=range(2016,2022),emb=True):
    """HAR-J(б) PIT: ln-OLS на отсечке vy по всем строкам с концом цели < vy; s² sm = var ошибок трёх прошлых val-годов (шаг 1 — без поправки),
    эмбарго пула (конец цели < vy), κ = 0,5 (r2_calib). -> {vy: (ln-смещение на всех строках, n обучения, max конец цели)}."""
    from sklearn.linear_model import LinearRegression
    y5=a._y5.values;ly=np.log(y5+1e-10);errs={};out={};first=min(steps)
    for vy in steps:
        tp=harj_trmask(a,vy,emb);ln=LinearRegression().fit(X[tp],ly[tp]).predict(X)
        sel=(a.year.values==vy)&np.isfinite(y5)&(y5>1e-12);e=ly-np.log(np.clip(np.exp(ln),1e-10,None))
        errs[vy]=pd.DataFrame(dict(e=e[sel],te=a._end5.values[sel]));pool=[errs[k] for k in range(max(first,vy-3),vy)]
        P=pd.concat(pool) if pool else None
        if P is not None and emb:P=P[P.te<np.datetime64(f'{vy}-01-01')]
        s2=float(np.var(P.e.values)) if P is not None and len(P)>=30 else 0.
        out[vy]=(ln+.5*s2,int(tp.sum()),a.loc[tp,'_end5'].max())
    return out


def year_masks(a,y,Y):
    cut=pd.Timestamp(f'{Y}-01-01');full=purged(a,cut)&np.isfinite(y);tr,va=split(a,full);return cut,full,tr,va


def g8a_year(a,x,y,hc,Y):
    cut,full,tr,va=year_masks(a,y,Y);pr=(a.year==Y).values&np.isfinite(hc);out=np.full(len(a),np.nan)
    for tk in sorted(a.ticker.unique()):
        t=(a.ticker==tk).values;m,fv=fit(x,y,tr&t,va&t,full&t,a,cut,'G8a_v2',dict(ticker=tk,year=Y))
        s2=float(np.var(y[va&t]-fv));sel=pr&t;out[sel]=hc[sel]*np.exp(m.predict(x[sel])+s2/2)
    return out


def g8b_year(a,x,y,hc,Y):
    cut,full,tr,va=year_masks(a,y,Y);pr=(a.year==Y).values&np.isfinite(hc)
    m,fv=fit(x,y,tr,va,full,a,cut,'G8b_v2',dict(year=Y));err=y[va]-fv;f=m.predict(x[pr]);out={}
    for lam in LAMS:
        c=shrink(a.ticker.values[va],err,lam);cv=a.ticker[va].map(c).values;s2=float(np.var(err-cv))
        o=np.full(len(a),np.nan);o[pr]=hc[pr]*np.exp(f+a.ticker[pr].map(c).values+s2/2);out[lam]=o
    return out


def top25(d,hc):
    """25 лучших признаков по gain панельного LightGBM с параметрами финала на строках с концом цели < 2016-01-01 (У19); пишется до dev-счёта."""
    path=TAB/'features_top25.json'
    if path.exists():return json.loads(path.read_text())['features']
    a=d.a;x,names=d.static_x(5,'original');cut=pd.Timestamp('2016-01-01');y0=a._y5.values;tr=mask_train(a,5,cut,'annual',y0)
    hb=np.nan_to_num(d.harx(5,'original'),nan=0.,posinf=0.,neginf=0.);ly=np.log(np.clip(y0,1e-12,None))
    hub=HuberRegressor(max_iter=500).fit(hb[tr][:,1:],ly[tr]);yy=ly-(hb[:,1:]@hub.coef_+hub.intercept_)
    fz=json.loads((ROOT/'docs/tables/gl3_r5_r6/frozen_warmup_configuration.json').read_text())['fin5params']['lgbm']
    p=dict(PARAMS,objective='huber',metric='huber',num_leaves=int(fz['num_leaves']),learning_rate=fz['learning_rate'],min_data_in_leaf=int(fz['min_data_in_leaf']),
           bagging_fraction=fz['bagging_fraction'],feature_fraction=fz['feature_fraction'],lambda_l1=fz['reg_alpha'],lambda_l2=fz['reg_lambda'],max_depth=-1)
    dates=sorted(a.loc[tr,'date'].unique());edge=dates[max(1,len(dates)-126)]
    ti=tr&(a.date<edge).values&(a._end5<edge).values;vi=tr&(a.date>=edge).values
    first=lgb.train(p,lgb.Dataset(x[ti],label=yy[ti]),500,valid_sets=[lgb.Dataset(x[vi],label=yy[vi])],callbacks=[lgb.early_stopping(20,verbose=False)])
    trees=max(1,first.best_iteration);model=lgb.train(p,lgb.Dataset(x[tr],label=yy[tr]),trees)
    record('G8_importance',cut,tr.sum(),a.loc[tr,'_end5'].max(),'refit',dict(trees=trees,frozen=fz))
    gain=pd.Series(model.feature_importance('gain'),index=names).sort_values(ascending=False,kind='stable');top=list(gain.index[:25])
    path.write_text(json.dumps(dict(features=top,gain={k:float(gain[k]) for k in top},n_candidates=len(names),written_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        note='har_feature не входит: это выход модели, не столбец данных'),ensure_ascii=False,indent=1))
    return top


def load():
    D.disk=disk;d=D.Data()
    h=read(REG/'gl3_r3/dev/predictions_h5.parquet',['date','ticker','har_corr'])
    return d,d.series(h,'har_corr')


def loyo(scores,preds,a):
    """У4: прогноз года Y — точка с наименьшим средним QLIKE по остальным четырём годам."""
    out=np.full(len(a),np.nan);pick={}
    for Y in YEARS:
        best=min(preds,key=lambda k:np.mean([scores[k][str(y)] for y in YEARS if y!=Y]));pick[Y]=best
        m=(a.year==Y).values;out[m]=preds[best][m]
    return out,pick


def ticker_q(d,p):
    r=d.ref.rv_actual.values/p[d.idx];l=r-np.log(r)-1
    return pd.Series(l).groupby([d.ref.ticker.values,d.ref.year.values]).mean().groupby(level=0).mean()


def binom_p(k,n=17):return sum(math.comb(n,j) for j in range(k,n+1))/2**n


def row(d,name,eligible,p,tb):
    s=d.score5(p);base=d.base;nb=int((ticker_q(d,p)<tb-1e-12).sum());better=sum(s[str(y)]<base[str(y)]-1e-12 for y in YEARS)
    sp=all(s[c]<=base[c]+1e-12 for c in ('stress2020','stressRVI'));bm=bool(s['mean_QLIKE']<base['mean_QLIKE']-1e-12)
    return dict(candidate=name,eligible=eligible,**{k:s[k] for k in [*map(str,YEARS),'mean_QLIKE','stress2020','stressRVI','ratio_stress2020','ratio_stressRVI']},
                better_years=int(better),stress_pass=bool(sp),better_mean=bm,n_better_tickers=nb,binom_p=binom_p(nb),admitted=bool(eligible and better>=3 and sp and bm and nb>=10))


def harj_check(a,arch,record_=False):
    """Смещение HAR-J(б) PIT по отсечкам 2016–2021; гейт: на val-году = архивный har_corr (≤ 1e-8 отн.), иначе остановка."""
    X=np.nan_to_num(np.log(a[HARJ].values.astype(float).clip(1e-10)),nan=0.,posinf=0.,neginf=0.);bases=harj_bases(a,X)
    for vy,(b,n,mx) in bases.items():
        m=(a.year.values==vy)&np.isfinite(arch);rel=float(np.max(np.abs(np.exp(b[m])/arch[m]-1)));assert rel<=1e-8,(vy,rel)
        if record_:record('HARJ_b',f'{vy}-01-01',n,mx,'fit',dict(max_rel_vs_har_corr=rel,regressors=HARJ))
    return bases


def run():
    assert not (TAB/'g8_candidates.csv').exists(),'уже посчитано; повторный запуск запрещён (У12)'
    disk('load');d,arch=load();a=d.a;feats=top25(d,arch);disk('G8 fits');x,names=d.static_x(5,'original');x=x[:,[names.index(f) for f in feats]]
    bases=harj_check(a,arch,record_=True);pa=np.full(len(a),np.nan);pb={l:np.full(len(a),np.nan) for l in LAMS};t0=pd.Timestamp.now()
    for Y in YEARS:
        hc=np.exp(bases[Y][0]);y=residual(a,hc);disk(f'G8 {Y}');m=(a.year==Y).values;pa[m]=g8a_year(a,x,y,hc,Y)[m]
        for l,o in g8b_year(a,x,y,hc,Y).items():pb[l][m]=o[m]
    f32=lambda p:p.astype('float32').astype('float64')
    preds={'G8a':f32(pa),**{f'G8b_lam{l}':f32(pb[l]) for l in LAMS}};tb=ticker_q(d,d.series(d.ref,'rv_pred'));rows=[]
    scores={k:d.score5(v) for k,v in preds.items()};lo,pick=loyo({k:scores[k] for k in preds if k!='G8a'},{k:v for k,v in preds.items() if k!='G8a'},a)
    preds['G8b_LOYO']=f32(lo)
    for k,v in preds.items():
        z=d.ref[['date','ticker']].copy();z['prediction']=v[d.idx].astype('float32');z.to_parquet(OUT/f'{k}_dev.parquet',index=False)
        rows.append(row(d,{'G8a':'Г8а','G8b_LOYO':'Г8б_LOYO'}.get(k,k.replace('G8b_lam','Г8б_λ')),k in ('G8a','G8b_LOYO'),v,tb))
    pd.DataFrame(rows).to_csv(TAB/'g8_candidates.csv',index=False)
    ev=[json.loads(s) for s in EVENTS.read_text().splitlines()]
    cnt=lambda c:sum(e['component']==c and e['phase']=='refit' for e in ev)
    (TAB/'attempt_counts.json').write_text(json.dumps(dict(candidate_rows=len(rows),grid_points=len(LAMS),loyo_pick={str(k):v for k,v in pick.items()},
        fits_G8a_v2_refit=cnt('G8a_v2'),fits_G8b_v2_refit=cnt('G8b_v2'),fits_HARJ_b=sum(e['component']=='HARJ_b' for e in ev),events=len(ev),
        seconds=(pd.Timestamp.now()-t0).total_seconds(),v1_okno2016=json.loads((TAB/'attempt_counts_v1_okno2016.json').read_text()))))


def audit():
    """G2–G5: независимый пересчёт из сохранённых parquet и проверка ledger."""
    d,arch=load();ev=[json.loads(s) for s in EVENTS.read_text().splitlines()];dev=pd.read_csv(TAB/'g8_candidates.csv');assert len(dev)==4
    for e in ev:assert e['h']==5 and pd.Timestamp(e['max_target_end'])<pd.Timestamp(e['cutoff'])<pd.Timestamp('2022-01-01')
    harj_check(d.a,arch);assert sum(e['component']=='HARJ_b' for e in ev)==6 and (TAB/'g8_candidates_v1_okno2016.csv').exists()
    for suf,ga_name,gb_name in (('_v2','G8a_v2','G8b_v2'),('','G8a','G8b')):
        ga=[e for e in ev if e['component']==ga_name];keys={(e['detail']['ticker'],e['detail']['year']) for e in ga if e['phase']=='refit'}
        assert len(keys)==85 and sum(e['phase']=='inner_ES' for e in ga)==85
        for e in ga+[e for e in ev if e['component']==gb_name]:
            assert all(e['detail']['params'][k]==PARAMS[k] for k in LOGKEYS) and (e['detail']['params']['max_depth'],e['detail']['params']['min_data_in_leaf'],e['detail']['params']['lambda_l2'])==(3,100,10.)
    first=min(e['time_utc'] for e in ev if e['component'] in ('G8a','G8a_v2','G8b','G8b_v2'));assert json.loads((TAB/'features_top25.json').read_text())['written_utc']<first
    for r in dev.to_dict('records'):
        name={'Г8а':'G8a','Г8б_LOYO':'G8b_LOYO'}.get(r['candidate'],r['candidate'].replace('Г8б_λ','G8b_lam'));p=pd.read_parquet(OUT/f'{name}_dev.parquet')
        assert p.prediction.dtype==np.dtype('float32') and p.date.max()<pd.Timestamp('2022-01-01')
        z=d.ref[['date','ticker','year','rv_actual']].merge(p,on=['date','ticker'],validate='1:1');assert len(z)==len(d.ref)==len(p) and np.isfinite(z.prediction).all() and (z.prediction>0).all()
        ratio=z.rv_actual/z.prediction.astype(float);loss=ratio-np.log(ratio)-1;years=loss.groupby(z.year).mean()
        for y,q in years.items():np.testing.assert_allclose(q,r[str(y)],rtol=0,atol=1e-11)
        np.testing.assert_allclose(years.mean(),r['mean_QLIKE'],rtol=0,atol=1e-11)
    print('AUDIT OK',len(ev),'events',len(keys),'ticker-years')


if __name__=='__main__':
    {'run':run,'audit':audit}[sys.argv[1]]()
