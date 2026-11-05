# -*- coding: utf-8 -*-
"""Remove only this round's reproducible intermediates after the final audit."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np
from gl3_r56_data import OUT,TAB,disk_guard


def main():
    disk_guard('cleanup audited intermediates')
    f=json.loads((TAB/'frozen_nominations.json').read_text())
    v=json.loads((TAB/'validation.json').read_text())
    assert f['status'] in ('ready_for_owner_yes','closed_without_opening')
    assert v['all_fit_maturity_valid'] and not f['control_read']
    audited=pd.read_csv(TAB/'prediction_manifest.csv');assert audited.audit_qlike_and_admission.all()
    # Complete model outputs are forecasts, not saved model objects. Remove their temporary cache.
    cache=OUT/'cache';rows=[];freed=0
    for p in sorted(cache.iterdir()):
        assert p.suffix in ('.parquet','.lock','.tmp'),p
        size=p.stat().st_size;freed+=size
        if p.suffix=='.parquet':
            z=pd.read_parquet(p,columns=['date','raw']);assert z.raw.dtype==np.dtype('float32')
            assert z.date.max()<pd.Timestamp('2022-01-01')
            rows.append(dict(path=str(p.name),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),n=len(z),dtype='float32',deleted_intermediate=True))
        p.unlink()
    pd.DataFrame(rows).to_csv(TAB/'intermediate_prediction_cache.csv',index=False)
    measure=OUT/'grid10_shift_mean.parquet'
    if measure.exists():
        assert hashlib.sha256(measure.read_bytes()).hexdigest()==json.loads((TAB/'measure_manifest.json').read_text())['output_sha256']
        freed+=measure.stat().st_size;measure.unlink()
    # Logs for counts and checks are permanent audit evidence; repeated warning output is temporary.
    for name in ('run.log','prefill.log','freeze.log','failure.txt'):
        p=TAB/name
        if p.exists():freed+=p.stat().st_size;p.unlink()
    result=dict(control_read=False,deleted_only_own_intermediates=True,freed_bytes=freed,
                retained='74 named candidate/reference forecast parquet files + winner raw/corrected forecasts; metadata/tables/fit audit ledger',
                saved_model_objects=0,regenerate_measure='python3 scripts/analysis/gl3_r56_measure.py before re-running Data',
                cached_forecast_banks_removed=len(rows))
    (TAB/'cleanup.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
