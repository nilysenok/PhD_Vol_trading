# -*- coding: utf-8 -*-
"""K13: exactly ten shifted10-minute RV estimates from existing1m DEV CSVs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/regen/gl3_r5_r6'
TAB=ROOT/'docs/tables/gl3_r5_r6'
LAST=pd.Timestamp('2021-12-31')


def one_day(group):
    ts=pd.to_datetime(group.begin)
    available=(ts.dt.hour*60+ts.dt.minute+1).values
    keep=(available>=591)&(available<=1130)&(group.close.values>0)
    t=available[keep];p=group.close.values[keep]
    if len(p)<3:return None
    values=[]
    for offset in range(10):
        grid=np.arange(590+offset,1131,10)
        index=np.searchsorted(t,grid,side='right')-1
        index=index[index>=0]
        if len(index)<3:return None
        r=np.diff(np.log(p[index]));n=len(r)
        values.append([np.sum(r*r),(n/3)*np.sum(r**4),np.pi/2*np.sum(np.abs(r[1:]*r[:-1])),
                       np.sum(r[r>0]**2),np.sum(r[r<0]**2),n])
    v=np.asarray(values)
    return dict(rv_alt=float(v[:,0].mean()),rq_alt=float(v[:,1].mean()),bv_alt=float(v[:,2].mean()),
                rsv_pos_alt=float(v[:,3].mean()),rsv_neg_alt=float(v[:,4].mean()),
                min_grid_returns=int(v[:,5].min()),minute_bars=int(len(p)),last_available_minute=int(t.max()))


def main():
    OUT.mkdir(exist_ok=True);TAB.mkdir(exist_ok=True)
    target=OUT/'grid10_shift_mean.parquet'
    if target.exists():
        print('K13 measure already prepared; reusing saved dev-only output.');return
    inputs=[];rows=[]
    files=sorted(p for p in (ROOT/'data/regen/D25/bars1m').glob('*.csv') if 2014<=int(p.stem.rsplit('_',1)[1])<=2021)
    assert len(files)==136,'Need17tickers×8years of1min source, no10min substitution'
    for i,path in enumerate(files):
        tk,year=path.stem.rsplit('_',1);year=int(year)
        # Validate identifier column BEFORE decoding any prices.
        ids=pd.to_datetime(pd.read_csv(path,usecols=['begin']).begin)
        assert ids.dt.year.eq(year).all() and ids.max()<pd.Timestamp('2022-01-01')
        b=pd.read_csv(path,usecols=['begin','close']).drop_duplicates('begin',keep='last').sort_values('begin')
        b['date']=pd.to_datetime(b.begin).dt.normalize()
        for date,g in b.groupby('date',sort=True):
            d=one_day(g)
            if d is not None:rows.append(dict(date=date,ticker=tk,**d))
        inputs.append(dict(path=str(path.relative_to(ROOT)),n=len(b),last=str(ids.max()),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if (i+1)%8==0:print(f'K13 {i+1}/{len(files)} yearly CSVs, days={len(rows)}',flush=True)
    result=pd.DataFrame(rows).sort_values(['ticker','date']).reset_index(drop=True)
    assert not result.duplicated(['ticker','date']).any() and result.date.max()<=LAST
    assert result.last_available_minute.max()<=1130
    result.to_parquet(target,index=False)
    (TAB/'measure_manifest.json').write_text(json.dumps(dict(control_read=False,source='existing1m CSVs,2014–2021 only',
         formula='mean of10 sum log-return-squares on clock grids09:50+offset,offset0..9,10min step; prices available by grid boundary; no overnight returns or future backfill',
         n=len(result),tickers=result.ticker.nunique(),first=str(result.date.min()),last=str(result.date.max()),
         output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),inputs=inputs),ensure_ascii=False,indent=2))
    print('K13 prepared',len(result),flush=True)


if __name__=='__main__':main()
