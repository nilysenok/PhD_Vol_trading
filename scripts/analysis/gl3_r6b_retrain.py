# -*- coding: utf-8 -*-
"""Раунд 6б, h=5, волна 2 (Г13, Г16, Г17, Г18): сборка candidates.csv, LOYO, аудит. DEV 2017–2021.
Запуск (PYTHONPATH=scripts/analysis): `bank a|b|v|g16` -> `groups` -> `table` -> `audit`. Каждая стадия пишет кэш; ledger — `fit_events.jsonl`."""
import sys
import json
import numpy as np
import pandas as pd
import gl3_r6b_retrain_base as B
import gl3_r6b_retrain_bank as K
import gl3_r6b_retrain_feat as F
import gl3_r6b_retrain_grp as Grp
import gl3_r6b_g8 as G8

VAR={'a':'G13a','b':'G13b','v':'G13v','g16':'G16'}
RU={'G13a':'Г13а','G13b':'Г13б','G13v':'Г13в','G13_LOYO':'Г13','G16':'Г16','G17_k3':'Г17_k3','G17_k4':'Г17_k4','G17_LOYO':'Г17',
    'G18_lam0.25':'Г18_λ0.25','G18_lam0.5':'Г18_λ0.5','G18_lam0.75':'Г18_λ0.75','G18_LOYO':'Г18'}
ORDER=['G13a','G13b','G13v','G13_LOYO','G16','G17_k3','G17_k4','G17_LOYO','G18_lam0.25','G18_lam0.5','G18_lam0.75','G18_LOYO']
GRID={'G13_LOYO':['G13a','G13b','G13v'],'G17_LOYO':['G17_k3','G17_k4'],'G18_LOYO':['G18_lam0.25','G18_lam0.5','G18_lam0.75']}
FIN_PIT=B.REG/'gl3_r6b_h5/pit/experts/FIN_PIT.parquet'


def stage_bank(v):
    d,_=B.load();raw=K.build(d,v);lev=B.f32(B.scalar_level(d.a,d.y(5,'original'),raw.astype('float32')));B.save_pred(d,VAR[v],lev)
    if v=='a':
        fin=d.series(pd.read_parquet(FIN_PIT),'prediction');m=np.isfinite(fin);diff=float(np.max(np.abs(lev[m]-fin[m])))
        (B.TAB/'g2_fin_pit.json').write_text(json.dumps(dict(max_abs_diff=diff,n_rows=int(m.sum()),fin_pit=str(FIN_PIT.relative_to(B.ROOT)))))
        assert diff<=1e-9,f'Г13а != FIN_PIT: {diff}'
        print('G2 max|Δ| =',diff,'rows',int(m.sum()))


def stage_groups():
    if (B.OUT/'G17_k3_dev.parquet').exists():raise SystemExit('Г17/Г18 уже посчитаны; повторный запуск запрещён (У12)')
    d,arch=B.load();a=d.a;bases=G8.harj_check(a,arch)
    for vy,(b,n,mx) in bases.items():B.record('HARJ_b',f'{vy}-01-01',n,mx,'fit',dict(regressors=G8.HARJ))
    x=Grp.top25_x(d);chars=F.chars_frame(d);r17,cl=Grp.run_g17(d,chars,bases,x);r18=Grp.run_g18(d,x)
    for k,p in r17.items():B.save_pred(d,f'G17_k{k}',B.f32(p))
    for lam,p in r18.items():B.save_pred(d,f'G18_lam{lam}',B.f32(p))
    cl.to_csv(B.TAB/'clusters.csv',index=False);print('events',B.flush())


def stage_table():
    d,_=B.load();a=d.a;tb=G8.ticker_q(d,d.series(d.ref,'rv_pred'));preds={n:B.load_pred(d,n) for n in ORDER if n in VAR.values() or n.startswith(('G17_k','G18_lam'))}
    scores={n:d.score5(p) for n,p in preds.items()};pick={}
    for name,pts in GRID.items():
        lo,pick[name]=G8.loyo({k:scores[k] for k in pts},{k:preds[k] for k in pts},a);preds[name]=B.f32(lo);B.save_pred(d,name,preds[name])
    rows=[G8.row(d,RU[n],n in GRID or n=='G16',preds[n],tb) for n in ORDER]
    pd.DataFrame(rows).to_csv(B.TAB/'candidates.csv',index=False)
    (B.TAB/'selection.json').write_text(json.dumps({RU[k]:{str(y):RU.get(v,v) for y,v in p.items()} for k,p in pick.items()},ensure_ascii=False,indent=1))
    ev=[json.loads(s) for s in B.EVENTS.read_text().splitlines()];refit=lambda c:sum(e['component'].startswith(c) and e['phase']=='refit' for e in ev)
    (B.TAB/'attempt_counts.json').write_text(json.dumps(dict(candidate_rows=len(rows),grid_points=sum(len(v) for v in GRID.values())+1,events=len(ev),
        fits_refit={c:refit(c) for c in ('D12_a','D12_b','D12_v','D12_g16','G17_k3','G17_k4','G18_panel')},loyo_pick={RU[k]:{str(y):v for y,v in p.items()} for k,p in pick.items()})))
    print(pd.DataFrame(rows)[['candidate','eligible','mean_QLIKE','stress2020','stressRVI','better_years','n_better_tickers','admitted']].to_string())


def audit():
    d,_=B.load();ev=[json.loads(s) for s in B.EVENTS.read_text().splitlines()];t=pd.read_csv(B.TAB/'candidates.csv')
    bad=[e for e in ev if not (e['h']==5 and pd.Timestamp(e['max_target_end'])<pd.Timestamp(e['cutoff'])<pd.Timestamp('2022-01-01'))];assert ev and not bad,len(bad)
    cnt=lambda c:sum(e['component']==c and e['phase']=='refit' for e in ev)
    for v in VAR:
        for lib in K.LIBS:assert cnt(f'D12_{v}_{lib}')==6,(v,lib)
    assert cnt('G17_k3')==15 and cnt('G17_k4')==20 and cnt('G18_panel')==15,'число fit Г17/Г18'
    assert list(t.candidate)==[RU[n] for n in ORDER],'G5: строки таблицы'
    cl=pd.read_csv(B.TAB/'clusters.csv');assert len(cl)==2*17*5 and json.loads((B.TAB/'selection.json').read_text())
    fin=d.series(pd.read_parquet(FIN_PIT),'prediction')
    for n in ORDER:
        r=t[t.candidate==RU[n]].iloc[0];p=pd.read_parquet(B.OUT/f'{n}_dev.parquet')
        assert p.prediction.dtype==np.dtype('float32') and p.date.max()<pd.Timestamp('2022-01-01')
        z=d.ref[['date','ticker','year','rv_actual']].merge(p,on=['date','ticker'],validate='1:1');assert len(z)==len(d.ref)==len(p) and np.isfinite(z.prediction).all() and (z.prediction>0).all()
        q=(z.rv_actual/z.prediction.astype(float));loss=(q-np.log(q)-1).groupby(z.year).mean()
        for y,v in loss.items():np.testing.assert_allclose(v,r[str(y)],rtol=0,atol=1e-11)
        np.testing.assert_allclose(loss.mean(),r['mean_QLIKE'],rtol=0,atol=1e-11)
        if n=='G13a':
            m=np.isfinite(fin[d.idx]);diff=np.abs(p.prediction.values.astype(float)[m]-fin[d.idx][m]).max();assert diff<=1e-9,diff
    print('AUDIT OK',len(ev),'events; G13a vs FIN_PIT dev rows max|Δ|',diff)


if __name__=='__main__':
    cmd=sys.argv[1];{'bank':lambda:stage_bank(sys.argv[2]),'groups':stage_groups,'table':stage_table,'audit':audit}[cmd]()
