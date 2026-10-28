"""Д17.2: «будущее не влияет» на новых колонках features_pit_v3_rrv.

Ряд RRV (rrv_1850.parquet: rrv_pk, rrv_gk) на датах > t заменяется шумом, новые колонки пересчитываются той же
функцией d172_rrv_features.new_columns. Признаки rrv_{pk,gk}_{d,w,m} на строках date <= t обязаны совпасть побайтно.
Цели: rrv_*_target_h1 / mean_h{h} на строке обязаны совпасть, если последняя строка их окна (t+h по строкам
тикера) имеет дату <= t — цель не смотрит дальше своего горизонта.
Самопроверка: подложенная утечка rrv_pk_d = ln RRV(t+1) ловится.
Сам ряд RRV по дню считается крейтом только из баров этого дня (сумма по барам дня, cargo test range).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
import d172_rrv_features as F  # noqa: E402


def noisy(rrv, t, seed):
    r = rrv.copy()
    m = r.date > pd.Timestamp(t)
    rng = np.random.default_rng(seed)
    for c in ("rrv_pk", "rrv_gk"):
        r.loc[m, c] = r.loc[m, c].to_numpy() * rng.lognormal(0, 1, m.sum()) + 1e-6
    return r


def eq(a, b):
    return a.to_numpy().tobytes() == b.to_numpy().tobytes()


def last_row_date(keys, h):
    o = keys.sort_values(["ticker", "date"])
    d = o.groupby("ticker")["date"].shift(-h)
    return d.reindex(keys.index)


def check(keys, rrv, t, leak=False):
    base = F.new_columns(keys, rrv)
    pert = F.new_columns(keys, noisy(rrv, t, 7))
    if leak:
        for x in (base, pert):
            src = rrv if x is base else noisy(rrv, t, 7)
            s = src.set_index(["ticker", "date"])["rrv_pk"]
            nxt = s.groupby(level=0).shift(-1)
            x["rrv_pk_d"] = np.log(nxt.reindex(pd.MultiIndex.from_frame(keys[["ticker", "date"]])).to_numpy())
    pre = keys.date <= pd.Timestamp(t)
    bad_f = [c for c in F.FEAT if not eq(base.loc[pre, c], pert.loc[pre, c])]
    post_changed = sum(not eq(base.loc[~pre, c], pert.loc[~pre, c]) for c in F.NEW)
    bad_t = []
    for c in F.TGT:
        h = 1 if c.endswith("_h1") else int(c.rsplit("h", 1)[1])
        ok = last_row_date(keys, h) <= pd.Timestamp(t)
        if not eq(base.loc[ok, c], pert.loc[ok, c]):
            bad_t.append(c)
    return int(pre.sum()), bad_f, bad_t, post_changed


def main():
    v3 = pd.read_parquet(F.V3, columns=["date", "ticker"])
    rrv = F.load_rrv()
    fail = False
    for t in ("2019-06-30", "2023-06-30"):
        n, bf, bt, pc = check(v3, rrv, t)
        print(f"t={t}: строк <= t {n}, признаков с расхождением до t {len(bf)} {bf}, целей {len(bt)} {bt}, "
              f"колонок, изменившихся после t {pc}/{len(F.NEW)}")
        fail |= bool(bf or bt) or pc == 0
    n, bf, _, _ = check(v3, rrv, "2021-12-31", leak=True)
    caught = "rrv_pk_d" in bf
    print(f"самопроверка (rrv_pk_d = ln RRV(t+1)): поймана = {caught}")
    fail |= not caught
    print("ТЕСТ Д17.2:", "FAIL" if fail else "PASS")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
