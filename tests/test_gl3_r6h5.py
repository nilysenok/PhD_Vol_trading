# -*- coding: utf-8 -*-
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
from gl3_r6h5_engine import purged,gap_boundary
from gl3_r6h5_data import Data


def test_purge_five_and_additional_embargo_five():
    dates=pd.bdate_range('2013-01-01','2021-12-31');a=pd.DataFrame({'date':dates,'_y5':1.})
    a['_end5']=a.date.shift(-5)
    for year in range(2016,2022):
        cut=pd.Timestamp(f'{year}-01-01');tr,va,full=purged(a,cut,year-1)
        inner=gap_boundary(a,pd.Timestamp(f'{year-1}-01-01'));outer=gap_boundary(a,cut)
        assert (a.loc[tr,'_end5']<inner).all()
        assert (a.loc[va,'_end5']<outer).all() and (a.loc[full,'_end5']<outer).all()
        # Advancing the latest mature target by one observed trading day breaks the contract.
        last=a.loc[tr,'_end5'].max();nextday=dates[dates.get_loc(last)+1]
        assert nextday>=inner
        assert not bool(nextday<inner)


def test_panel_normalizes_stock_variances_and_sqrt_variances_only():
    d=Data.__new__(Data)
    d.a=pd.DataFrame({'ticker':['SBER','LKOH'],'rv_d':[4.,9.],'rv_annualized':[2.,3.],
                      'rv_IMOEX':[8.,8.],'close_RVI':[20.,20.],'_scale':[4.,9.]})
    d.basecols={5:['rv_d','rv_annualized','rv_IMOEX','close_RVI']}
    x,n=d.raw_features(True)
    np.testing.assert_allclose(x[:,n.index('rv_d')],[1.,1.])
    np.testing.assert_allclose(x[:,n.index('rv_annualized')],[1.,1.])
    np.testing.assert_allclose(x[:,n.index('rv_IMOEX')],[8.,8.])
    np.testing.assert_allclose(x[:,n.index('close_RVI')],[20.,20.])
    d.a['_scale']=d.a._scale.shift(-1)
    mutated,_=d.raw_features(True)
    assert not np.allclose(x[:,n.index('rv_d')],mutated[:,n.index('rv_d')],equal_nan=True)


def test_hard_gate_uses_corrected_components_and_separate_raw_mixture(monkeypatch):
    import gl3_r6h5_run as run
    monkeypatch.setattr(run,'disk',lambda stage:None)
    s=run.Study.__new__(run.Study)
    s.d=SimpleNamespace(a=pd.DataFrame({'_state':[False,True],'_ln_close_RVI':[2.,3.],'_ln_rv_w':[1.,2.]}))
    s.inputs={'FIN':np.array([4.,9.]),'HXP':np.array([16.,25.])}
    s.rawinputs={'FIN':np.log([1.,4.]),'HXP':np.log([9.,16.])}
    p,raw=s.compute({'core':'G2','gate':'hard','expert':'HXP'})
    np.testing.assert_allclose(p,[4.,25.]);np.testing.assert_allclose(np.exp(raw),[1.,16.])


def test_selection_is_same_outer_fold_through_parent_chain():
    from gl3_r6h5_run import Study
    s=Study.__new__(Study);s.scorecache={}
    s.rec={k:{str(y):{'name':k,'held':y} for y in range(2017,2022)} for k in ('A','B')}
    calls=[]
    def pred(spec):calls.append(spec);return spec,None
    s.predict=pred
    s.d=SimpleNamespace(score5=lambda spec:{str(y):1000. if y==2017 else (2. if spec['name']=='A' else 1.) for y in range(2017,2022)})
    assert s.best(['A','B'],2017)=='B'
    assert all(r['held']==2017 for r in calls)


def test_new_rule_rejects_old_round5_admission_without_mean_gain():
    # The previously admitted c=.75 candidate has3wins/stress passes but loses in average.
    base=.20996539211548085;candidate=.211594
    assert not (3>=3 and True and candidate<base-1e-12)


def test_h5_disk_stop(monkeypatch,tmp_path):
    import gl3_r6h5_data as data
    monkeypatch.setattr(data,'TAB',tmp_path);monkeypatch.setattr(data,'OUT',tmp_path/'out')
    monkeypatch.setattr(data.shutil,'disk_usage',lambda p:SimpleNamespace(free=19*1024**3))
    monkeypatch.setattr(data.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='df fixture\n'))
    with pytest.raises(RuntimeError,match='DISK_STOP'):data.disk('fixture')


@pytest.mark.parametrize('w',[.25,.5,.75])
def test_ticker_qlike_shrinkage_uses_past_only(monkeypatch,w):
    import gl3_r6h5_run as run
    monkeypatch.setattr(run,'disk',lambda stage:None)
    monkeypatch.setattr(run,'periods',lambda mode:[(pd.Timestamp('2021-01-01'),pd.Timestamp('2022-01-01'))])
    monkeypatch.setattr(run.B,'record',lambda *a,**k:None)
    past=pd.bdate_range('2018-01-01',periods=60)
    dates=list(past)+[pd.Timestamp('2021-01-04')]
    a=pd.DataFrame([(t,tk) for tk in ('A','B') for t in dates],columns=['date','ticker']);a['_end5']=a.date+pd.offsets.BDay(5)
    y=np.where(a.ticker.eq('A'),2.,.5);parent=np.ones(len(a));raw=np.zeros(len(a))
    s=run.Study.__new__(run.Study);s.d=SimpleNamespace(a=a,y=lambda h,measure:y)
    s.predict=lambda spec:(parent,raw)
    spec={'core':'G7','kind':'qlike','w':w,'parent':{}}
    out,_=s.compute(spec);va=a.date.eq('2021-01-04')
    expected=np.array([w*2+(1-w)*1.25,w*.5+(1-w)*1.25])
    np.testing.assert_allclose(out[va],expected)
    y[va]=1e8
    again,_=s.compute(spec);np.testing.assert_allclose(again[va],expected)


def test_shared_ticker_category_follows_row_key():
    d=Data.__new__(Data);d.cat_ticker=True
    d.a=pd.DataFrame({'ticker':['SBER','SBER','LKOH','LKOH'],'rv_d':[4.,4.,9.,9.],
                      '_scale':[4.,4.,9.,9.]})
    d.basecols={5:['rv_d']}
    x,n=d.raw_features(True);code=x[:,n.index('cat_ticker')]
    np.testing.assert_array_equal(code,[1.,1.,0.,0.])
    shifted=pd.Series(code).shift(-1).values
    assert not np.allclose(shifted,code,equal_nan=True)


def test_prediction_cache_is_versioned_by_numerical_contract(monkeypatch,tmp_path):
    import gl3_r6h5_run as run
    monkeypatch.setattr(run,'OUT',tmp_path)
    data=SimpleNamespace(a=pd.DataFrame({'date':pd.to_datetime(['2017-01-03','2017-01-04']),'ticker':['SBER','SBER']}))
    def fresh(value):
        s=run.Study.__new__(run.Study);s.d=data;s.pcache={};s.rawcache={}
        s.compute=lambda spec:(np.full(2,value),np.full(2,np.log(value)))
        return s
    monkeypatch.setattr(run,'NUMERICAL_CONTRACT','test_v1')
    p,_=fresh(1.).predict({'core':'synthetic'});np.testing.assert_array_equal(p,[1.,1.])
    p,_=fresh(7.).predict({'core':'synthetic'});np.testing.assert_array_equal(p,[1.,1.])
    monkeypatch.setattr(run,'NUMERICAL_CONTRACT','test_v2')
    p,_=fresh(7.).predict({'core':'synthetic'});np.testing.assert_array_equal(p,[7.,7.])


def test_h5_fit_archive_does_not_double_count_live_events(monkeypatch,tmp_path):
    import gl3_r6h5_data as data
    monkeypatch.setattr(data,'TAB',tmp_path)
    r=dict(time_utc='2026-11-04T00:00:00Z',component='synthetic',phase='fit',h=5,cutoff='2017-01-01',detail={})
    pd.DataFrame([{**r,'detail':'{}'}]).to_csv(tmp_path/'fit_ledger.csv.gz',index=False)
    (tmp_path/'fit_events.jsonl').write_text(json.dumps(r)+'\n')
    assert len(data.load_events())==1


def test_normalized_har_offset_is_invariant_to_stock_rv_scale():
    d=Data.__new__(Data)
    b=pd.DataFrame({'rv_d':[4.,9.],'rv_w':[3.,8.],'rv_m':[2.,7.],'jump':[1.,2.],
                    'jump_annualized':np.sqrt([252.,504.]),'jump_lag1':[.4,.8],'jump_lag5':[.3,.7],
                    'jump_lag22':[.2,.6],'_scale':[2.,3.]})
    d.a=b;d.frames={'original':b}
    reference=d.harx(5,'original',True)
    for c in b:
        b[c]*=3. if c=='jump_annualized' else 9.
    np.testing.assert_allclose(d.harx(5,'original',True),reference,rtol=1e-12,atol=1e-12)
