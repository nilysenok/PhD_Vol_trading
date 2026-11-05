# -*- coding: utf-8 -*-
"""Precompute last three fixed bagging seeds on Mac; no new candidates."""
import json
import pandas as pd
from gl3_r56_run import Study
from gl3_r56_data import TAB,disk_guard


def main():
    s=Study();saved=json.loads((TAB/'selection.json').read_text())
    s.rec=saved['recipes'];s.rows=pd.read_csv(TAB/'dev_candidates.csv').to_dict('records')
    keys=[f"h1:{r['candidate']}" for r in s.rows if r['h']==1 and r['eligible'] and r['candidate']!='K14']
    recipe=s.choose(keys)
    unique={json.dumps(r,sort_keys=True):r for r in recipe.values()}
    for seed in (51,50,49):
        for r in unique.values():
            disk_guard(f'prefill existing K14 seed{seed}')
            print('Prefill',seed,r,flush=True)
            s.e.raw(1,r,r.get('mode','annual'),r.get('measure','original'),seed)
    print('Fixed K14 prefill finished',flush=True)


if __name__=='__main__':main()
