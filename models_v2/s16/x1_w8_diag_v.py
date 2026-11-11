"""W8-A: вердикт П1–П5 (`W8_SPEC.md` разд. 1.0), выведенный ТОЛЬКО из `W8_DECOMP.csv` (блоки D_transfer и C_group) и сверенный со строками E_verdict. Только чтение.
Публичное API:
- `verdict(path=DECOMP) -> (DataFrame по признакам: P1…P5, passed, расхождение со строкой E_verdict, список прошедших, всего проверено)`.
  П1: Δ(признак к контролю) < 0 при h1->h2 и h2->h1; П2: то же к X1; П3: к контролю < 0 при h1->h2no22 и h2no22->h1; П4: порядок групп по beta_expost (группы n ≥ 15 в обеих
  половинах h1 и h2no22) одинаков, общих групп ≥ 2; П5: к X1 ≤ 0 при h1->h2no22 и h2no22->h1.
- `main(path=DECOMP) -> int`: печатает `прошли: <признаки>; проверено N`; 0, если проверено 10 и пересчёт совпал со строками E_verdict, иначе 1."""
from __future__ import annotations

import pandas as pd

from models_v2.s16.x1_w8_diag_core import MIN_GROUP, ROOT

DECOMP = ROOT / "docs/otchety/harj_x1/W8_DECOMP.csv"
N_FEATURES = 10


def _read(path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", dtype={"group": str})


def verdict(path=DECOMP):
    t = _read(path)
    d, c, e = t[t["block"] == "D_transfer"], t[t["block"] == "C_group"], t[t["block"] == "E_verdict"].set_index("feature")
    out = []
    for f, g in d.groupby("feature", sort=False):
        v = g.set_index("direction")
        k, x = v["d_feat_vs_ctrl_pct"], v["d_feat_vs_X1_pct"]
        cf = c[(c["feature"] == f) & (c["n"] >= MIN_GROUP) & (c["group"] != "na")]
        o = {h: cf[cf["half"] == h].set_index("group")["beta_expost"] for h in ("h1", "h2no22")}
        common = sorted(set(o["h1"].index) & set(o["h2no22"].index))
        p4 = len(common) >= 2 and sorted(common, key=lambda G: o["h1"][G]) == sorted(common, key=lambda G: o["h2no22"][G])
        p = [k["h1->h2"] < 0 and k["h2->h1"] < 0, x["h1->h2"] < 0 and x["h2->h1"] < 0, k["h1->h2no22"] < 0 and k["h2no22->h1"] < 0, p4,
             x["h1->h2no22"] <= 0 and x["h2no22->h1"] <= 0]
        ok = all(p)
        out.append((f, *[bool(z) for z in p], ok, ok != bool(e.at[f, "passed"]) if f in e.index else True))
    r = pd.DataFrame(out, columns=["feature", "P1", "P2", "P3", "P4", "P5", "PASS", "differs_from_E_verdict"])
    return r, [f for f, *_, ok, _ in out if ok], len(out)


def main(path=DECOMP) -> int:
    r, passed, total = verdict(path)
    print(r.to_string(index=False))
    print("прошли: %s; проверено %d" % (", ".join(passed) or "нет", total))
    return 0 if total == N_FEATURES and not r["differs_from_E_verdict"].any() else 1


if __name__ == "__main__":
    raise SystemExit(main())
