# -*- coding: utf-8 -*-
"""Behavior and mutation checks for the dev-only rounds5–6 pipeline."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
from gl3_r56_engine import qlike_grad
from gl3_r56_data import periods,mask_train,weights
from gl3_r56_measure import one_day


class Dataset:
    def __init__(self,y,base,w):self.y=y;self.base=base;self.w=w
    def get_label(self):return self.y
    def get_init_score(self):return self.base
    def get_weight(self):return self.w


@pytest.mark.parametrize('c',[.25,.5,.75,1.])
def test_qlike_gradient_hessian_with_initial_score(c):
    y=np.array([-7.,-5.,-9.]);base=np.array([-6.,-6.,-8.]);delta=np.array([.2,-.1,.4]);w=np.array([1.,2.,.5])
    ds=Dataset(y,base,w);g,h=qlike_grad(base+delta,ds,c)
    def loss(d):
        u=y-(base+c*d);return w*(np.exp(u)-u-1)
    eps=1e-4
    np.testing.assert_allclose(g,(loss(delta+eps)-loss(delta-eps))/(2*eps),rtol=1e-6)
    np.testing.assert_allclose(h,(loss(delta+eps)-2*loss(delta)+loss(delta-eps))/eps**2,rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('mode',['annual','quarterly','monthly','half1','half2','rolling3'])
@pytest.mark.parametrize('h',[1,5,22])
def test_refit_maturity_and_advance_day_mutation(mode,h):
    dates=pd.bdate_range('2012-01-01','2021-12-31');a=pd.DataFrame({'date':dates})
    a[f'_end{h}']=a.date.shift(-h);y=np.ones(len(a))
    for cut,end in periods(mode):
        train=mask_train(a,h,cut,mode,y)
        assert (a.loc[train,f'_end{h}']<cut).all()
        assert not (a.loc[train,'date']>=cut).any()
        if mode=='rolling3':assert (a.loc[train,'date']>=cut-pd.DateOffset(years=3)).all()
        # A date advanced to the next refit includes labels not yet known at the true cut.
        future=mask_train(a,h,pd.offsets.BDay().rollforward(cut)+pd.offsets.BDay(1),mode,y)
        assert (a.loc[future,f'_end{h}']>=cut).any()
        ww=weights(a,cut,mode)
        assert np.isfinite(ww).all() and (ww>0).all()


def test_shifted_rv_ignores_post1850_prices_and_overnight():
    ts=pd.date_range('2020-03-10 09:50',periods=600,freq='min')
    x=pd.DataFrame({'begin':ts,'close':100*np.exp(.001*np.sin(np.arange(len(ts))/20))})
    expected=one_day(x)
    z=x.copy();z.loc[z.begin>=pd.Timestamp('2020-03-10 18:50'),'close']*=100
    assert one_day(z)==expected
    assert expected['last_available_minute']==1130
    z=x.copy();z['close']*=7
    np.testing.assert_allclose(one_day(z)['rv_alt'],expected['rv_alt'],rtol=1e-10)


def test_minute_forward_shift_mutation_is_detected():
    ts=pd.date_range('2020-03-10 09:50',periods=541,freq='min')
    x=pd.DataFrame({'begin':ts,'close':100+np.arange(len(ts))*.001})
    # The final available boundary must not inherit the close of the18:50 minute.
    x.loc[x.begin==pd.Timestamp('2020-03-10 18:50'),'close']=10000
    original=one_day(x)['rv_alt']
    x['close']=x.close.shift(-1)
    assert one_day(x)['rv_alt']>original*100


def test_chained_loyo_selection_uses_same_outer_fold_not_diagonal_rows():
    from types import SimpleNamespace
    from gl3_r56_run import Study
    s=Study.__new__(Study);s.foldscore={}
    # Diagonal row summaries would favor A; for the2017 outer fold B is actually better.
    s.rows=[dict(h=1,candidate='A',**{str(y):0. for y in range(2017,2022)}),dict(h=1,candidate='B',**{str(y):10. for y in range(2017,2022)})]
    s.rec={k:{str(y):{'name':k,'outer_held':y} for y in range(2017,2022)} for k in ('h1:A','h1:B')}
    calls=[]
    def predict(h,recipe):calls.append(recipe);return recipe
    def score(h,recipe):
        return {str(y): (1000. if y==2017 else (2. if recipe['name']=='h1:A' else 1.)) for y in range(2017,2022)}
    s.e=SimpleNamespace(predict=predict);s.d=SimpleNamespace(score=score)
    assert s.best(['h1:A','h1:B'],held=2017)=='h1:B'
    assert all(r['outer_held']==2017 for r in calls)


@pytest.mark.parametrize('free,stops',[(19*1024**3,True),(21*1024**3,False)])
def test_disk_stop_at_twenty_gib(monkeypatch,tmp_path,free,stops):
    from types import SimpleNamespace
    import gl3_r56_data as data
    monkeypatch.setattr(data,'TAB',tmp_path)
    monkeypatch.setattr(data.shutil,'disk_usage',lambda p:SimpleNamespace(free=free))
    monkeypatch.setattr(data.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='df fixture\n'))
    if stops:
        with pytest.raises(RuntimeError,match='DISK_STOP'):data.disk_guard('fixture')
    else:data.disk_guard('fixture')


def test_fit_ledger_deduplicates_archive_and_live_log(monkeypatch,tmp_path):
    import json
    import gl3_r56_data as data
    monkeypatch.setattr(data,'TAB',tmp_path)
    a=dict(time_utc='2026-11-04T00:00:00Z',component='synthetic',phase='fit',h=1,cutoff='2017-01-01',detail={})
    b={**a,'time_utc':'2026-11-04T00:01:00Z'}
    pd.DataFrame([{**a,'detail':'{}'}]).to_csv(tmp_path/'fit_ledger.csv.gz',index=False)
    (tmp_path/'fit_events.jsonl').write_text(json.dumps(a)+'\n'+json.dumps(b)+'\n')
    assert len(data.load_fit_events())==2
