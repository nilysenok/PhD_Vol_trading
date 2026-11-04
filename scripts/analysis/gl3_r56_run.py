# -*- coding: utf-8 -*-
"""Closed sequential DEV selection. No control reader; resumable raw fits."""
import json
import hashlib
import traceback
import numpy as np
import pandas as pd
from gl3_r56_data import Data,OUT,TAB,YEARS,READS,disk_guard
from gl3_r56_engine import Engine,stable


class Study:
    def __init__(self):
        self.d=Data();self.e=Engine(self.d);self.rows=[];self.pred={};self.rec={};self.choices={}
        self.year=self.d.a.year.values
        self.base={h:self.d.score(h,self.reference(h)) for h in (1,5,22)}
        self.gates=json.loads((TAB/'diagnostics_manifest.json').read_text())['gates']
        self.status='running';self.foldscore={}

    def reference(self,h):
        ref,idx=self.d.refs[h];p=np.full(len(self.d.a),np.nan);p[idx]=ref.rv_pred.values;return p

    def save(self):
        pd.DataFrame(self.rows).to_csv(TAB/'dev_candidates.csv',index=False)
        (TAB/'selection.json').write_text(json.dumps(dict(status=self.status,choices=self.choices,recipes=self.rec,control_opened=False),ensure_ascii=False,indent=2))
        (TAB/'development_reads.json').write_text(json.dumps(READS,ensure_ascii=False,indent=2))

    def add(self,h,label,recipes,stage='5',eligible=True,component=False):
        disk_guard(f'candidate h{h} {label}')
        key=f'h{h}:{label}'
        if 'core' in recipes:recipes={str(y):recipes for y in YEARS}|{'all':recipes}
        pp=np.full(len(self.d.a),np.nan)
        for y in YEARS:
            p=self.e.predict(h,recipes[str(y)]);m=self.year==y;pp[m]=p[m]
        score=self.d.score(h,pp);base=self.base[h]
        ny=sum(score[str(y)]<base[str(y)]-1e-12 for y in YEARS)
        stress=all(score[c]<=base[c]+1e-12 for c in ('stress2020','stressRVI'))
        row=dict(h=h,candidate=label,round=stage,**score,better_years=ny,stress_pass=stress,
                 admitted=bool(eligible and ny>=3 and stress),eligible=eligible,component_only=component)
        self.rows.append(row);self.pred[key]=pp;self.rec[key]=recipes
        # Store original target and predictions solely on DEV, no redundant raw-source copies.
        ref,idx=self.d.refs[h];z=ref[['date','ticker']].copy();z['prediction']=pp[idx].astype(np.float32)
        z.to_parquet(OUT/(hashlib.sha256(key.encode()).hexdigest()[:16]+'.parquet'),index=False)
        self.save();print(json.dumps(row,ensure_ascii=False),flush=True)
        return key

    def best(self,keys,held=None):
        cols=[str(y) for y in YEARS if y!=held]
        def metric(k):
            cache=(k,held)
            if cache not in self.foldscore:
                h=int(k.split(':')[0][1:]);recipe=self.rec[k]['all' if held is None else str(held)]
                # Re-score this outer fold's recipe on its own training years.
                # Using diagonal LOYO rows here would import the held-out year via other folds.
                score=self.d.score(h,self.e.predict(h,recipe))
                self.foldscore[cache]=float(np.mean([score[c] for c in cols]))
            return self.foldscore[cache],k
        return min(keys,key=metric)

    def row(self,k):return next(r for r in self.rows if f"h{r['h']}:{r['candidate']}"==k)

    def choose(self,keys,transform=None):
        out={}
        for y in list(YEARS)+['all']:
            k=self.best(keys,None if y=='all' else y)
            r=dict(self.rec[k][str(y)])
            out[str(y)]=transform(r) if transform else r
        return out

    def r5(self,h):
        candidates=[]
        fin=self.add(h,'FIN_fresh',{'core':'FIN'},eligible=False)
        self.add(h,'HAR_b_fresh',{'core':'HAR'},eligible=False)
        pre=[fin];k9=[];lib9={}
        gate=self.gates.get(str(h),self.gates.get(h,{}))
        run=gate if isinstance(gate,bool) else gate.get('run_K9_K10',gate.get('run_k9_k10',False))
        print('GATE',h,gate,run,flush=True)
        if run:
            for lib in ('lgb','xgb'):
                group=[]
                for style in (('a','mono','linear') if lib=='lgb' else ('a','mono')):
                    k=self.add(h,f'K9_{lib}_{style}',dict(core='K9',lib=lib,style=style,c=1.));group.append(k);k9.append(k);candidates.append(k)
                for c in (.25,.5,.75):
                    rr=self.choose(group,lambda r:{**r,'c':c})
                    k=self.add(h,f'K9_{lib}_shrink{c}',rr);k9.append(k);candidates.append(k)
                lib9[lib]=[k for k in k9 if f'K9_{lib}_' in k]
            k10=self.add(h,'K10',self.choose(k9,lambda r:{**r,'core':'K10','lib':'lgb','bias':'none'}));candidates.append(k10)
            pre+=k9+[k10]
        k11=self.add(h,'K11_FIN',{'core':'FIN','bias':'conditional'});pre.append(k11);candidates.append(k11)
        if k9:
            kk=self.add(h,'K11_K9_K10',self.choose(k9+[k10],lambda r:{**r,'bias':'conditional'}));pre.append(kk);candidates.append(kk)
        k12=[]
        for mode in ('quarterly','monthly','half1','half2','rolling3'):
            k=self.add(h,'K12_'+mode,self.choose(pre,lambda r:{**r,'mode':mode}));k12.append(k);candidates.append(k)
            self.add(h,'HAR_b_adapt_'+mode,{'core':'HAR','mode':mode},eligible=False)
        k13=self.add(h,'K13',self.choose([k for k in pre if k!=fin]+k12,lambda r:{**r,'measure':'alt'}));candidates.append(k13)
        self.add(h,'HAR_b_K13',{year:{'core':'HAR','measure':'alt','mode':r.get('mode','annual')} for year,r in self.rec[k13].items()},eligible=False)
        k14=self.add(h,'K14',self.choose(candidates,lambda r:{**r,'bagging':True}));candidates.append(k14)
        self.choices[f'r5_h{h}']=min(candidates,key=lambda k:(self.row(k)['mean_QLIKE'],k));self.save()
        return candidates,k12,lib9

    def r6(self,h,k12,lib9):
        # Reserve is a single closed extension, invoked only if every r5 horizon failed.
        pool=[]
        panel=self.add(h,'R2',{'core':'PANEL'},stage='6');pool.append(panel)
        moe=self.add(h,'R3',{'core':'MOE'},stage='6');pool.append(moe)
        dist=self.add(h,'R4',{'core':'DIST','bias':'none'},stage='6');pool.append(dist)
        modes=self.choose(k12,lambda r:{'core':'STACK','mode':r.get('mode','annual')})
        # Every stack recipe records its LOYO-selected base K9 specifications.
        for year in modes:
            modes[year]['k9']=[self.rec[self.best(lib9[lib],None if year=='all' else int(year))][year] for lib in sorted(lib9)]
            modes[year]['k10']=next((self.rec[k][year] for k in self.rec if k==f'h{h}:K10'),None)
        stack=self.add(h,'R1',modes,stage='6');pool.append(stack)
        r6=self.add(h,'R6',self.choose(pool,lambda r:{**r,'measure':'alt'}),stage='6');pool.append(r6)
        self.choices[f'r6_h{h}']=min(pool,key=lambda k:(self.row(k)['mean_QLIKE'],k));self.save();return pool

    def run(self):
        r5={};metadata={}
        for h in (1,5,22):
            keys,k12,lib9=self.r5(h);r5[h]=keys;metadata[h]=(k12,lib9)
        reserve=not any(self.row(k)['admitted'] for keys in r5.values() for k in keys)
        for h in (1,5,22):
            keys=r5[h]+(self.r6(h,*metadata[h]) if reserve else [])
            passed=[k for k in keys if self.row(k)['admitted']]
            self.choices[f'winner_h{h}']=min(passed,key=lambda k:(self.row(k)['mean_QLIKE'],k)) if passed else None
        self.status='ready_for_owner_yes' if any(self.choices[f'winner_h{h}'] for h in (1,5,22)) else 'closed_without_opening'
        self.save()


if __name__=='__main__':
    s=Study()
    try:s.run()
    except BaseException:
        s.status='failed_incomplete';s.save();(TAB/'failure.txt').write_text(traceback.format_exc());raise
