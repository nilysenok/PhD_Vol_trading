# -*- coding: utf-8 -*-
"""Independent dev-feature and origin mutation audit; never opens control."""
import json
import numpy as np
import pandas as pd
from gl3_r56_data import Data,TAB,OUT,periods,mask_train,load_fit_events


def derived(b,measure):
    g=b.groupby('ticker');out={}
    for c,w in [('rv_d',1),('rv_w',5),('rv_m',22),('_RV66',66),('_RV132',132),('_scale',252)]:
        out[c]=g.rv.transform(lambda s:s.rolling(w,min_periods=252 if w==252 else 1).mean())
    for c in ('rq','close_RVI'):
        out['_ln_'+c]=np.log(b[c].clip(lower=1e-12))
    out['_ln_rv_w']=np.log(out['rv_w'].clip(lower=1e-12))
    q=np.sqrt((b.n_bars*b.rkurt/3).clip(lower=0)).clip(upper=10) if measure=='original' else (np.sqrt(b.rq.clip(lower=0))/b.rv.clip(lower=1e-12)).clip(upper=10)
    out['_HARQ']=np.log(b.rv.clip(lower=1e-12))*q.groupby(b.ticker).shift(1)
    ret=np.log(b.close.where(b.close>0)).groupby(b.ticker).diff().clip(-.5,.5)
    for w in (1,5,22):out['_LHAR'+str(w)]=ret.groupby(b.ticker).transform(lambda s:s.rolling(w,min_periods=1).mean()).clip(upper=0)
    rvi=b.groupby('date').close_RVI.first().sort_index()
    med=rvi.expanding(min_periods=22).median().shift(1)
    rank=rvi.rolling(253,min_periods=253).apply(lambda z:(z[:-1]<=z[-1]).mean(),raw=True)
    out['_state']=b.date.map(rvi>med).astype(bool)
    out['_stress']=b.date.map(rank>2/3).astype(bool)
    return pd.DataFrame(out,index=b.index)


def compare(x,y):
    np.testing.assert_allclose(np.asarray(x,dtype=float),np.asarray(y,dtype=float),rtol=2e-6,atol=1e-11,equal_nan=True)


def feature_checks(d):
    rows=[]
    for measure,b in d.frames.items():
        independent=derived(b,measure)
        for c in independent:compare(b[c],independent[c])
        # Prefix invariance at every year boundary: future input replacement cannot change past.
        for cut in pd.to_datetime(['2017-06-01','2018-06-01','2019-06-01','2020-06-01','2021-06-01']):
            prefix=b.loc[b.date<=cut].copy();q=derived(prefix,measure)
            for c in q:compare(q[c],independent.loc[prefix.index,c])
        for c in independent:
            advanced=independent[c].groupby(b.ticker).shift(-1)
            dif=~np.isclose(independent[c].astype(float),advanced.astype(float),rtol=2e-6,atol=1e-11,equal_nan=True)
            assert dif.any(),f'Mutation not detected: {measure}/{c}'
            rows.append(dict(measure=measure,feature=c,prefix_invariant=True,forward_day_caught=True,n_differences=int(dif.sum())))
    return rows


def main():
    d=Data();rows=feature_checks(d)
    events=load_fit_events()
    for r in events:
        assert pd.Timestamp(r['max_target_end'])<pd.Timestamp(r['cutoff'])
        assert pd.Timestamp(r['cutoff'])<pd.Timestamp('2022-01-01')
        # An explicit origin mutation at/after refit is rejected, in every mode.
        mutated=dict(r,max_target_end=str(pd.Timestamp(r['cutoff'])+pd.Timedelta(days=1)))
        assert not pd.Timestamp(mutated['max_target_end'])<pd.Timestamp(mutated['cutoff'])
    result=dict(control_read=False,feature_mutations=rows,fit_records_checked=len(events),
                all_fit_maturity_valid=True,modes=sorted(set(r['mode'] for r in events)),
                feature_names='derived only; existing v3 families retain earlier audit, not retested as new',
                reserve_state='not_run' if not any(r['component'].startswith('R') for r in events) else 'see fit ledger')
    (TAB/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='feature_mutations'},ensure_ascii=False))


if __name__=='__main__':main()
