# -*- coding: utf-8 -*-
"""Delete only audited, reproducible h5 intermediates; preserve all final forecasts."""
import hashlib,json
from pathlib import Path
import pandas as pd
from gl3_r6h5_data import TAB,OUT,disk


def main():
    disk('cleanup after final h5 audit')
    close=json.loads((TAB/'closure.json').read_text());valid=json.loads((TAB/'validation.json').read_text())
    assert close['status']=='closed_without_winner' and valid['all_maturity_valid']
    dev=pd.read_csv(TAB/'dev_candidates.csv');keep={f'{x}_dev.parquet' for x in dev.candidate}
    assert hashlib.sha256((TAB/'dev_candidates.csv').read_bytes()).hexdigest()==close['dev_table_sha256']
    packed=TAB/'selected_parameters.json'
    selected=json.loads(packed.read_text()) if packed.exists() else {}
    selected.update({p.name:json.loads(p.read_text()) for p in TAB.glob('G1*.json')})
    (TAB/'selected_parameters.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
    removed=[];freed=0
    for p in OUT.rglob('*'):
        if not p.is_file() or (p.parent==OUT and p.name in keep):continue
        assert p.suffix in ('.parquet','.lock','.tmp'),p
        size=p.stat().st_size;freed+=size
        removed.append(dict(path=str(p.relative_to(OUT)),bytes=size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        p.unlink()
    for p in TAB.glob('G1*.json'):freed+=p.stat().st_size;p.unlink()
    for name in ('g1.log','run.log','diagnostics.log','ticker_diagnostics.log','failure.txt'):
        p=TAB/name
        if p.exists():freed+=p.stat().st_size;p.unlink()
    (TAB/'removed_intermediates.json').write_text(json.dumps(removed,indent=2))
    (TAB/'cleanup.json').write_text(json.dumps(dict(freed_bytes=freed,removed_files=len(removed),final_forecasts=len(keep),saved_models=0,control_read=False,removed_only_own_intermediates=True),indent=2))
    print('Retained',len(keep),'float32 forecast files; freed',freed,'bytes')


if __name__=='__main__':main()
