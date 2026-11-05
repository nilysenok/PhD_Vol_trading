# -*- coding: utf-8 -*-
"""Г13/Г16/Г17/Г18 (раунд 6б, h=5, волна 2): синтетика + подложенные ошибки. Без сети, без данных ≥ 2022."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
import gl3_r6b_retrain_base as B
import gl3_r6b_retrain_bank as K
import gl3_r6b_retrain_feat as F
import gl3_r6b_retrain_grp as Grp
import gl3_r6b_g8 as G8


def frame(seed=0,start='2014-06-02',end='2017-12-29',tickers=('AAA','BBB','CCC')):
    rng=np.random.default_rng(seed);dates=pd.bdate_range(start,end);rows=[]
    for tk in tickers:
        z=pd.DataFrame(dict(date=dates,ticker=tk));z['_end5']=z.date.shift(-5);rows.append(z)
    a=pd.concat(rows,ignore_index=True);a['year']=a.date.dt.year
    a['_y5']=np.where(a._end5.notna(),np.exp(rng.normal(-8,.3,len(a))),np.nan)
    for c in K.HARJC:a[c]=np.exp(rng.normal(-8,.3,len(a)))
    return a


def test_target_bounds_known_answer_and_train_rows_only():
    y=np.r_[np.arange(1001.),np.arange(1001.)*2];tk=np.r_[['A']*1001,['B']*1001];m=np.ones(2002,bool)
    b=K.target_bounds(y,tk,m);assert b['A']==pytest.approx((5.,995.)) and b['B']==pytest.approx((10.,1990.))
    m2=m.copy();m2[1000]=False;y2=y.copy();y2[1000]=1e9  # будущая строка вне mask не влияет на пороги
    assert K.target_bounds(y2,tk,m2)['A']==K.target_bounds(y,tk,m2)['A']


def test_mutation_bounds_over_all_rows_is_caught():
    a=frame();cut=pd.Timestamp('2017-01-01');yy=np.log(a._y5.fillna(1e-4).values);mask=((a.date<cut)&(a._end5<cut)).values
    yy2=yy.copy();yy2[(a._end5>=cut).values]=50.  # строки с концом цели ≥ cut — выброс
    ok=K.prepare('b',None,yy2,a.ticker.values,mask)[1];ref=K.prepare('b',None,yy,a.ticker.values,mask)[1]
    np.testing.assert_array_equal(ok[mask],ref[mask])
    bad=K.prepare('b',None,yy2,a.ticker.values,np.ones(len(a),bool))[1]  # мутация: пороги по всем строкам
    assert not np.array_equal(bad[mask],ref[mask])


def test_feature_clip_thresholds_from_train_rows_and_applied_to_all():
    rng=np.random.default_rng(1);x=rng.normal(size=(1000,3)).astype('float32');mask=np.arange(1000)<800;x[900,0]=1e6
    xc=K.prepare('v',x,np.zeros(1000),np.array(['A']*1000),mask)[0];lo,hi=K.feature_bounds(x,mask)
    assert float(xc[900,0])==pytest.approx(float(hi[0])) and xc[:800].max()<=hi.max()+1e-6 and (xc[:, 0].astype(float)<=float(hi[0])+1e-6).all()
    xb=K.prepare('v',x,np.zeros(1000),np.array(['A']*1000),np.ones(1000,bool))[0];assert xb[900,0]!=xc[900,0]  # мутация: по всем строкам


def test_clip_target_only_on_mask_rows_validation_untouched():
    y=np.arange(100.);tk=np.array(['A']*100);m=np.arange(100)<80;out=K.clip_target(y,tk,K.target_bounds(y,tk,m),m)
    assert out[80:].tolist()==y[80:].tolist() and out[:80].max()<79


@pytest.fixture
def spy(monkeypatch):
    calls=[];orig=K.prepare
    def wrap(variant,x,yy,tick,mask):
        r=orig(variant,x,yy,tick,mask);calls.append((variant,mask.copy(),r));return r
    monkeypatch.setattr(K,'prepare',wrap);return calls


@pytest.mark.parametrize('variant',['a','b','v'])
def test_one_year_fits_only_on_closed_rows(variant,spy):
    a=frame();rng=np.random.default_rng(3);x=rng.normal(size=(len(a),5)).astype('float32');names=[f'f{i}' for i in range(5)]
    yy=np.log(a._y5.values+1e-10);hx=np.nan_to_num(np.log(a[K.HARJC].values.clip(1e-10)));ex=pd.DataFrame(dict(h=[5],model=['lgbm'],feature=['f4']))
    cut=pd.Timestamp('2017-01-01');pred,raw=K.one_year(SimpleNamespace(a=a),variant,x,names,yy,hx,ex,2017)
    assert len(spy)==2 and np.isfinite(raw).all() and pred.sum()==len(raw)
    for _,mask,_r in spy:assert a.loc[mask,'_end5'].max()<cut and a.loc[mask,'date'].max()<cut
    bad=((a.date<cut)|(a.date<'2017-01-04')).values;assert not a.loc[bad,'_end5'].max()<cut  # мутация: строки с концом цели ≥ cut не прошли бы проверку
    ev=[e for e in B.PENDING if e['component'].startswith(f'D12_{variant}_')];assert len(ev)==4 and all(e['max_target_end']<e['cutoff'] for e in ev);B.PENDING.clear()


def test_record_rejects_target_end_not_before_cutoff():
    with pytest.raises(AssertionError):B.record('X','2017-01-01',5,'2017-01-01','refit')
    with pytest.raises(AssertionError):B.record('X','2022-01-01',5,'2021-12-30','refit')
    B.PENDING.clear()


def test_scalar_level_uses_closed_windows_only():
    a=frame(start='2014-06-02',end='2017-12-29',tickers=('AAA','BBB'));y=a._y5.values;rng=np.random.default_rng(5);raw=np.log(np.nan_to_num(y,nan=1e-4))+rng.normal(0,.2,len(a))
    raw=np.where(a.date>='2016-01-01',raw,np.nan).astype('float32');base=B.scalar_level(a,y,raw);cut=pd.Timestamp('2017-01-01')
    y2=y.copy();y2[(a._end5>=cut).values&(a.date<cut).values]*=50  # цели, закрывающиеся после cut
    np.testing.assert_array_equal(B.scalar_level(a,y2,raw)[(a.year==2017).values],base[(a.year==2017).values])
    a_bad=a.copy();a_bad['_end5']=a_bad.date  # мутация: фильтр по концу цели снят
    assert not np.allclose(B.scalar_level(a_bad,y2,raw)[(a.year==2017).values],base[(a.year==2017).values])
    m=(a.year==2017).values;e=np.log(y)[(a.date<cut).values&(a._end5<cut).values&(a.date>='2014-01-01').values&np.isfinite(raw)]
    assert np.isfinite(base[m]).all() and e.size>=50


def chars_input(n=420,seed=0):
    rng=np.random.default_rng(seed);dates=pd.bdate_range('2020-01-01',periods=n);rm=rng.normal(0,.01,n);mc=pd.Series(1000*np.exp(np.cumsum(rm)),index=dates);rows=[]
    for tk,b in (('AAA',2.),('BBB',.5)):
        c=100*np.exp(np.cumsum(b*rm));rows.append(pd.DataFrame(dict(date=dates,ticker=tk,close=c,mclose=mc.values,value=1e6,amihud=2e-4,jump=np.where(np.arange(n)%2==0,1e-5,0.),rv=1e-4)))
    return pd.concat(rows,ignore_index=True)


def test_characteristics_known_answer():
    f=chars_input();c=F.characteristics(f);last=f.index[f.date==f.date.max()]
    assert c.loc[last,'ch_beta252'].tolist()==pytest.approx([2.,.5],abs=1e-6) and c.loc[last,'ch_lnturn60'].abs().max()<1e-12
    assert c.loc[last,'ch_amihud63'].tolist()==pytest.approx([2e-4,2e-4]) and c.loc[last,'ch_lnrv252'].tolist()==pytest.approx([np.log(1e-4)]*2)
    assert c.loc[last,'ch_jumpshare63'].tolist()==pytest.approx([.5,.5],abs=.02)


def test_characteristics_no_future_and_mutation_d_plus_1_caught():
    f=chars_input();c0=F.characteristics(f);t0=f.date.unique()[300];f2=f.copy();late=(f2.date>t0)
    for col in ('close','mclose','value','amihud','jump','rv'):f2.loc[late,col]=f2.loc[late,col]*3+1  # меняем всё после t0
    c2=F.characteristics(f2);pre=(f.date<=t0).values
    pd.testing.assert_frame_equal(c0[pre],c2[pre])
    def shifted(g):  # мутация: данные дня d+1
        g=g.copy();g['value']=g.groupby('ticker').value.shift(-1);g['amihud']=g.groupby('ticker').amihud.shift(-1);return g
    m0=F.characteristics(shifted(f));g2=f.copy();g2.loc[g2.date==pd.Timestamp(t0)+pd.offsets.BDay(1),['value','amihud']]=[9e9,.5]
    m2=F.characteristics(shifted(g2));at=(f.date==t0).values
    assert not np.allclose(m0[at][['ch_lnturn60','ch_amihud63']],m2[at][['ch_lnturn60','ch_amihud63']])  # утечка видна — тест инвариантности её ловит


def chars_clusters(swap):
    dates=pd.bdate_range('2019-01-01','2022-02-28');rows=[]
    for i in range(6):
        base=0. if i<3 else 10.
        for dt in dates:
            sw=swap and dt>=pd.Timestamp('2021-01-01') and i in (2,3);b=10. if i==2 and sw else (0. if i==3 and sw else base)
            rows.append(dict(date=dt,ticker=f'T{i}',**{c:b+.01*i for c in F.NUM}))
    return pd.DataFrame(rows)


def test_clusters_use_only_past_window_and_mutation_caught():
    ch=chars_clusters(True);lab=F.cluster_table(ch,'2021-01-01',2)
    assert len({lab['T0'],lab['T1'],lab['T2']})==1 and len({lab['T3'],lab['T4'],lab['T5']})==1 and lab['T0']!=lab['T3']
    pd.testing.assert_series_equal(lab,F.cluster_table(chars_clusters(False),'2021-01-01',2))  # будущее не меняет кластеры
    bad=F.cluster_table(ch,'2022-01-01',2);assert bad['T2']==bad['T4'] and lab['T2']==lab['T0']  # мутация: окно с текущим годом


def shrink_input(seed=0):
    a=frame(seed=seed,start='2015-01-01',end='2017-06-30',tickers=('AAA','BBB'));rng=np.random.default_rng(seed)
    X=rng.normal(size=(len(a),3));ly=X@[.5,.2,.1]+np.where(a.ticker=='AAA',1.,-1.)*X[:,0]*.3+rng.normal(0,.1,len(a));return a,X,ly


def test_shrunk_coefs_limits_and_closed_rows():
    a,X,ly=shrink_input();cut=pd.Timestamp('2017-01-01');mask=((a.date<cut)&(a._end5<cut)).values;tk=a.ticker.values;Z=np.column_stack([np.ones(len(a)),X])
    pool,c1=Grp.shrunk_coefs(X,ly,tk,mask,1.);_,c0=Grp.shrunk_coefs(X,ly,tk,mask,0.)
    own=np.linalg.lstsq(Z[mask&(tk=='AAA')],ly[mask&(tk=='AAA')],rcond=None)[0]
    np.testing.assert_allclose(c1['AAA'],own,atol=1e-12);np.testing.assert_allclose(c0['AAA'],pool,atol=1e-12);np.testing.assert_allclose(c0['BBB'],pool,atol=1e-12)
    assert not np.allclose(c1['AAA'],c1['BBB'])  # λ=1: ровно потикерный; λ=0: общий (мутация λ↔1−λ различима)
    ly2=ly.copy();ly2[(a._end5>=cut).values]+=100
    np.testing.assert_allclose(Grp.shrunk_coefs(X,ly2,tk,mask,.5)[1]['AAA'],Grp.shrunk_coefs(X,ly,tk,mask,.5)[1]['AAA'],atol=1e-12)
    bad=Grp.shrunk_coefs(X,ly2,tk,np.ones(len(a),bool),.5)[1]['AAA'];assert not np.allclose(bad,Grp.shrunk_coefs(X,ly,tk,mask,.5)[1]['AAA'])  # мутация: строки с концом ≥ cut


def test_g18_lambda_zero_equals_pooled_harj_b_and_mutation_forbidden_rows():
    a=frame(start='2014-06-02',end='2018-12-31',tickers=('AAA','BBB','CCC'));X=Grp.harj_x(a)
    ref=G8.harj_bases(a,X,steps=range(2016,2019));g0=Grp.g18_bases(a,X,0.,steps=range(2016,2019))
    for vy in ref:np.testing.assert_allclose(g0[vy][0],ref[vy][0],atol=1e-9)
    g1=Grp.g18_bases(a,X,1.,steps=range(2016,2019));assert not np.allclose(g1[2018][0],g0[2018][0])
    a2=a.copy();late=(a2.year>=2018);a2.loc[late,'_y5']*=50  # цели 2018 не должны влиять на отсечку 2018
    g1b=Grp.g18_bases(a2,X,1.,steps=range(2016,2019));m=(a.year==2018).values
    np.testing.assert_allclose(g1b[2018][0][m],g1[2018][0][m],atol=1e-12)
