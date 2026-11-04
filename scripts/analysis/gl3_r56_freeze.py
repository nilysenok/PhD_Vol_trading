# -*- coding: utf-8 -*-
"""Audit saved DEV predictions and freeze nominations, never open control."""
import json
import hashlib
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from gl3_r56_data import Data,OUT,TAB,ROOT,disk_guard
from gl3_r56_engine import Engine


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    disk_guard('freeze dev nominations')
    s=json.loads((TAB/'selection.json').read_text());assert s['status'] in ('ready_for_owner_yes','closed_without_opening')
    d=Data();e=Engine(d);dev=pd.read_csv(TAB/'dev_candidates.csv');manifest=[];baselines=[];nominations=[]
    for h in (1,5,22):
        ref,idx=d.refs[h];base=np.full(len(d.a),np.nan);base[idx]=ref.rv_pred.values
        base_score=d.score(h,base);baselines.append(dict(h=h,**base_score))
        for row in dev[dev.h==h].to_dict('records'):
            key=f"h{h}:{row['candidate']}";path=OUT/(hashlib.sha256(key.encode()).hexdigest()[:16]+'.parquet')
            p=pd.read_parquet(path)
            assert p.prediction.dtype==np.dtype('float32') and p.date.max()<pd.Timestamp('2022-01-01')
            assert not p.duplicated(['date','ticker']).any()
            assert len(p)==len(ref) and p[['date','ticker']].equals(ref[['date','ticker']].reset_index(drop=True))
            pred=np.full(len(d.a),np.nan);pred[idx]=p.prediction.values;sc=d.score(h,pred)
            for k,v in sc.items():np.testing.assert_allclose(v,row[k],rtol=1e-10,atol=1e-12)
            ny=sum(sc[str(y)]<base_score[str(y)]-1e-12 for y in range(2017,2022))
            stress=all(sc[k]<=base_score[k]+1e-12 for k in ('stress2020','stressRVI'))
            assert ny==row['better_years'] and stress==row['stress_pass']
            assert bool(row['admitted'])==bool(row['eligible'] and ny>=3 and stress)
            manifest.append(dict(h=h,candidate=row['candidate'],path=str(path.relative_to(ROOT)),sha256=sha(path),n=len(p),dtype='float32',last=str(p.date.max()),audit_qlike_and_admission=True))
        winner=s['choices'][f'winner_h{h}']
        if winner:
            row=dev[(dev.h==h)&(dev.candidate==winner.split(':',1)[1])].iloc[0]
            assert row.admitted
            spec=s['recipes'][winner]['all'];raw=e.raw(h,spec,spec.get('mode','annual'),spec.get('measure','original'))
            pp=e.predict(h,spec);np.testing.assert_allclose(d.score(h,pp)['mean_QLIKE'],row.mean_QLIKE,rtol=1e-9)
            z=d.a[['date','ticker']].copy();z['raw_log_prediction']=raw.astype('float32');z['prediction']=pp.astype('float32')
            z=z[(z.date>=pd.Timestamp('2016-01-01'))&np.isfinite(z.raw_log_prediction)]
            path=OUT/f'winner_h{h}_dev.parquet';z.to_parquet(path,index=False)
            nominations.append(dict(h=h,candidate=row.candidate,recipe=spec,dev_QLIKE=float(row.mean_QLIKE),incumbent_QLIKE=base_score['mean_QLIKE'],
                                    gain_to_incumbent_pct=100*(1-row.mean_QLIKE/base_score['mean_QLIKE']),better_years=int(row.better_years),
                                    stress2020=float(row.stress2020),stressRVI=float(row.stressRVI),
                                    incumbent_stress2020=base_score['stress2020'],incumbent_stressRVI=base_score['stressRVI'],
                                    predictions=str(path.relative_to(ROOT)),predictions_sha256=sha(path),
                                    full_recipe_identical_across_outer_folds=all(s['recipes'][winner][str(y)]==spec for y in range(2017,2022))))
        else:nominations.append(dict(h=h,candidate=None,incumbent_QLIKE=base_score['mean_QLIKE'],reason='No admitted candidate'))
    pd.DataFrame(baselines).to_csv(TAB/'incumbent_dev.csv',index=False)
    pd.DataFrame(manifest).to_csv(TAB/'prediction_manifest.csv',index=False)
    frozen=dict(status=s['status'],control_read=False,opening_count_before=2,opening_count_after=2,nominees=nominations,
                reserve6_run=bool((dev['round']==6).any()),
                dev_table_sha256=sha(TAB/'dev_candidates.csv'),selection_sha256=sha(TAB/'selection.json'),
                versions={k:importlib.metadata.version(k) for k in ('numpy','pandas','scipy','scikit-learn','lightgbm','xgboost','torch')})
    (TAB/'frozen_nominations.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2))
    # Old trace is a temporary file from interrupted technical runs, not the final status.
    failure=TAB/'failure.txt'
    if failure.exists():failure.unlink()
    print(json.dumps(frozen,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
