# -*- coding: utf-8 -*-
"""Г16 (У29) — шесть характеристик бумаги по прошлым данным (окна кончаются в d включительно) и Г17 (У30) — k-means по средним характеристик за 252 даты до 1 января.
Источники (DEV до 2022): close, jump, rv, close_IMOEX — features_pit_v3.parquet; оборот `value` и дневной `amihud` — R2/feat/_work/xs_full.parquet."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import gl3_r6b_retrain_base as B
from gl3_r56_data import SECTORS

NUM=['ch_beta252','ch_lnturn60','ch_amihud63','ch_lnrv252','ch_jumpshare63']


def minp(w):return int(np.ceil(.8*w))


def roll(s,tick,w,fn):
    return s.groupby(tick).transform(lambda z:getattr(z.rolling(w,min_periods=minp(w)),fn)())


def characteristics(f):
    """f: строки отсортированы по (ticker,date); столбцы date,ticker,close,mclose (закрытие IMOEX на дату),value,amihud,jump,rv. -> пять числовых характеристик."""
    assert f.reset_index(drop=True).equals(f.sort_values(['ticker','date']).reset_index(drop=True)),'порядок строк'
    tk=f.ticker;g=f.groupby(tk)
    r=g.close.transform(lambda s:np.log(s.where(s>0)).diff())
    mc=f.drop_duplicates('date').set_index('date').mclose.sort_index();rm=f.date.map(np.log(mc.where(mc>0)).diff())
    ok=r.notna()&rm.notna();r=r.where(ok);rm=rm.where(ok)
    mr,mm=roll(r,tk,252,'mean'),roll(rm,tk,252,'mean')
    cov=roll(r*rm,tk,252,'mean')-mr*mm;var=roll(rm*rm,tk,252,'mean')-mm*mm
    out=pd.DataFrame(index=f.index)
    out['ch_beta252']=cov/var.where(var>0)
    out['ch_lnturn60']=np.log(f.value.where(f.value>0)/roll(f.value,tk,60,'median'))
    out['ch_amihud63']=roll(f.amihud,tk,63,'mean')
    out['ch_lnrv252']=np.log(roll(f.rv,tk,252,'mean').where(lambda s:s>0))
    out['ch_jumpshare63']=roll((f.jump>0).astype(float).where(f.jump.notna()),tk,63,'mean')
    return out


def chars_frame(d):
    """Характеристики по строкам d.a (индекс совпадает); sector — статический ключ SECTORS."""
    a=d.a;xs=B.read(B.REG/'R2/feat/_work/xs_full.parquet',['date','ticker','value','amihud'])
    im=B.read(B.REG/'features_pit_v3.parquet',['date','ticker','close_IMOEX'])
    f=a[['date','ticker','close','jump','rv']].merge(xs,on=['date','ticker'],how='left',validate='1:1').merge(im,on=['date','ticker'],how='left',validate='1:1').rename(columns={'close_IMOEX':'mclose'})
    assert len(f)==len(a) and (f.date.values==a.date.values).all()
    f.index=a.index;return pd.concat([a[['date','ticker']],characteristics(f)],axis=1)


def extended_features(d):
    """Признаки финала (raw_features) + пять характеристик + one-hot сектора (У29)."""
    x,names=d.raw_features();c=chars_frame(d);sec=pd.get_dummies(d.a.ticker.map(SECTORS),dtype=float)
    ext=np.column_stack([c[NUM].values,sec.values]).astype('float32')
    return np.column_stack([x,ext]).astype('float32'),names+NUM+['ch_sector_'+s for s in sec]


def cluster_table(chars,cut,k,seed=42):
    """Средние характеристик по последним 252 датам строго до cut, стандартизация по тикерам, k-means (seed 42, n_init 20). Строки с date >= cut не читаются."""
    past=chars[chars.date<pd.Timestamp(cut)].sort_values('date').groupby('ticker').tail(252)
    m=past.groupby('ticker')[NUM].mean();assert np.isfinite(m.values).all(),'нет характеристик до cut'
    z=(m-m.mean())/m.std(ddof=0).where(lambda s:s>0,1.)
    return pd.Series(KMeans(k,random_state=seed,n_init=20).fit_predict(z.values),index=m.index,name=f'k{k}')
