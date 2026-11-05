# -*- coding: utf-8 -*-
"""Г8 (раунд 6б, h=5): синтетика + подложенные ошибки. Без сети, без данных ≥ 2022."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
import gl3_r6b_g8 as G


def frame(seed=0):
    rng=np.random.default_rng(seed);dates=pd.bdate_range('2015-06-01','2017-12-29');rows=[]
    for tk in ('AAA','BBB','CCC'):
        z=pd.DataFrame(dict(date=dates,ticker=tk));z['_end5']=z.date.shift(-5);rows.append(z)
    a=pd.concat(rows,ignore_index=True);a['year']=a.date.dt.year
    a['_y5']=np.where(a._end5.notna(),np.exp(rng.normal(-8,.4,len(a))),np.nan)
    hc=np.exp(rng.normal(-8,.2,len(a)));hc[(a.date<'2016-01-01').values]=np.nan
    x=rng.normal(size=(len(a),4)).astype('float32');x[:,0]+=np.log(a._y5.fillna(1).values/hc)  # сигнал в признаке
    return a,x,hc


def test_split_purge_and_embargo():
    a,x,hc=frame();y=G.residual(a,hc);cut,full,tr,va=G.year_masks(a,y,2017)
    vs=a.loc[va,'date'].min();edge=G.gap_boundary(a,vs)
    assert a.loc[tr,'_end5'].max()<edge<=vs and va.sum()>0 and a.loc[full,'_end5'].max()<cut
    assert a.loc[va,'date'].nunique()==126


def test_mutation_end_inside_validation_is_caught():
    a,x,hc=frame();y=G.residual(a,hc);cut,full,tr,va=G.year_masks(a,y,2017)
    vs=a.loc[va,'date'].min();bad=full&(a.date<vs).values  # без purge/embargo
    assert a.loc[bad,'_end5'].max()>=vs and a.loc[tr,'_end5'].max()<vs


def test_shrink_known_answer_and_mutations():
    tick=np.array(['A']*126+['B']*126);err=np.r_[np.ones(126),-np.ones(126)]
    c=G.shrink(tick,err,126);assert c['A']==pytest.approx(.5) and c['B']==pytest.approx(-.5)
    assert G.shrink(tick,err,504)['A']==pytest.approx(126/630)
    assert G.shrink(tick,err,0)['A']==pytest.approx(1.)  # мутация λ=0 отличается от заданной
    swapped=G.shrink(np.where(tick=='A','B','A'),err,126);assert swapped['A']==pytest.approx(-.5)  # перепутанные тикеры


def test_g8b_ticker_shuffle_and_lambda_zero_change_forecast(monkeypatch,tmp_path):
    monkeypatch.setattr(G,'EVENTS',tmp_path/'e.jsonl');a,x,hc=frame();y=G.residual(a,hc);y[(a.ticker=='AAA').values]+=.8
    ok=G.g8b_year(a,x,y,hc,2017)[126];m=(a.year==2017).values
    orig=G.shrink
    def shuffled(t,e,lam):
        c=orig(t,e,lam);return pd.Series(c.values[::-1],index=c.index)
    monkeypatch.setattr(G,'shrink',shuffled);bad=G.g8b_year(a,x,y,hc,2017)[126]
    assert not np.allclose(ok[m],bad[m])
    monkeypatch.setattr(G,'shrink',lambda t,e,lam:orig(t,e,0));zero=G.g8b_year(a,x,y,hc,2017)[126]
    assert not np.allclose(ok[m],zero[m])


def test_g8a_per_ticker_fits_params_and_forecast(monkeypatch,tmp_path):
    monkeypatch.setattr(G,'EVENTS',tmp_path/'e.jsonl');a,x,hc=frame();y=G.residual(a,hc);out=G.g8a_year(a,x,y,hc,2017);m=(a.year==2017).values
    assert np.isfinite(out[m]).all() and (out[m]>0).all() and np.isnan(out[~m]).all()
    ev=[json.loads(s) for s in (tmp_path/'e.jsonl').read_text().splitlines()]
    assert len(ev)==6 and {e['detail']['ticker'] for e in ev}=={'AAA','BBB','CCC'}
    for e in ev:
        p=e['detail']['params'];assert (p['max_depth'],p['min_data_in_leaf'],p['lambda_l2'])==(3,100,10.) and e['h']==5
        assert pd.Timestamp(e['max_target_end'])<pd.Timestamp(e['cutoff'])<pd.Timestamp('2022-01-01')


def test_harj_bases_pit_and_mutation_future_target_caught():
    a,x,hc=frame();X=x[:,1:3].astype(float);ok=G.harj_bases(a,X,steps=range(2016,2018))
    for vy in (2016,2017):assert a.loc[G.harj_trmask(a,vy),'_end5'].max()<pd.Timestamp(f'{vy}-01-01')
    # цели строк с концом >= 2017-01-01 не должны влиять на смещение отсечки 2017
    a2=a.copy();late=(a2._end5>=pd.Timestamp('2017-01-01')).values;a2.loc[late,'_y5']*=5
    same=G.harj_bases(a2,X,steps=range(2016,2018));assert np.allclose(ok[2017][0],same[2017][0])
    bad=G.harj_bases(a,X,steps=range(2016,2018),emb=False);bad2=G.harj_bases(a2,X,steps=range(2016,2018),emb=False)
    assert a.loc[G.harj_trmask(a,2017,False),'_end5'].max()>=pd.Timestamp('2017-01-01')  # мутация: конец цели >= cut в обучении
    assert not np.allclose(bad[2017][0],bad2[2017][0])  # и она меняет смещение, то есть ловится


def test_record_rejects_end_not_before_cut(monkeypatch,tmp_path):
    monkeypatch.setattr(G,'EVENTS',tmp_path/'e.jsonl')
    with pytest.raises(AssertionError):G.record('x','2017-01-01',10,'2017-01-01','refit')
    with pytest.raises(AssertionError):G.record('x','2022-01-01',10,'2021-12-01','refit')


def test_loyo_excludes_own_year_and_binom():
    a=pd.DataFrame(dict(year=np.repeat(range(2017,2022),2)))
    s={'p':{str(y):v for y,v in zip(range(2017,2022),[.1,.2,.2,.2,.2])},'q':{str(y):v for y,v in zip(range(2017,2022),[.9,.15,.15,.15,.15])}}
    p={'p':np.zeros(10),'q':np.ones(10)};out,pick=G.loyo(s,p,a)
    assert pick[2017]=='q' and pick[2018]=='p'  # 2017: p по остальным=.2 > q=.15; 2018: p=.175 < q=.3375
    assert out[:2].tolist()==[1,1] and out[2:4].tolist()==[0,0]
    assert G.binom_p(0)==1 and G.binom_p(17)==pytest.approx(1/2**17) and G.binom_p(10)==pytest.approx(41226/131072)  # 65536 − C(17,9) = 41226
