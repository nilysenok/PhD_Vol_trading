"""W5-A: вердикт «основание в обеих половинах» (перенос `w5/verdict.py`), выведенный ТОЛЬКО из `W5_DIAG.csv` (критерий объявлен в docstring `x1_w5_diag` до счёта). Только чтение.
Публичное API: `verdict(diag=DIAG) -> (DataFrame по состояниям, список «блок/состояние» прошедших, всего проверено)`. Состояние проходит, если одновременно:
перенос h1->h2 и h2->h1 оба < 0 к контролю и к X1; h1->h2_no2022 < 0 к контролю; порядок групп по параметру одинаков во внутриполовинных оценках h1 и h2.
`main()` печатает таблицу, `прошли: …` и `всего проверено: N`; возвращает 1, если проверено не 24 (18 уровней + 6 доверия), иначе 0."""
from __future__ import annotations

import pandas as pd

from models_v2.s16 import x1_w3_core as C

DIAG = C.ROOT / "docs/otchety/harj_x1/W5_DIAG.csv"
N_STATES = 24


def verdict(diag=DIAG):
    t = pd.read_csv(diag, comment="#")
    b = t[t["block"].isin(["B_level", "B_trust"])]
    out = []
    for (blk, s), g in b.groupby(["block", "state"]):
        v = g.set_index("window")
        c1 = v.loc["h1->h2", "d_state_vs_ctrl_pct"] < 0 and v.loc["h1->h2", "d_state_vs_X1_pct"] < 0
        c2 = v.loc["h2->h1", "d_state_vs_ctrl_pct"] < 0 and v.loc["h2->h1", "d_state_vs_X1_pct"] < 0
        c3 = v.loc["h1->h2_no2022", "d_state_vs_ctrl_pct"] < 0

        def par(w):
            return dict(kv.split(":") for kv in v.loc[w, "group"].split("|"))

        p1, p2 = par("h1->h1_insample"), par("h2->h2_insample")
        keys = sorted(set(p1) & set(p2))
        c4 = [k for k in sorted(keys, key=lambda k: float(p1[k]))] == [k for k in sorted(keys, key=lambda k: float(p2[k]))]
        out.append((blk, s, c1, c2, c3, c4, c1 and c2 and c3 and c4))
    r = pd.DataFrame(out, columns=["block", "state", "h1->h2", "h2->h1", "h1->h2_no2022", "direction", "PASS"])
    return r, ["%s/%s" % (a, s) for a, s, *_, ok in out if ok], len(out)


def main(diag=DIAG) -> int:
    r, passed, total = verdict(diag)
    print(r.to_string(index=False))
    print("прошли:", ", ".join(passed) or "нет")
    print("всего проверено:", total)
    return 0 if total == N_STATES else 1
