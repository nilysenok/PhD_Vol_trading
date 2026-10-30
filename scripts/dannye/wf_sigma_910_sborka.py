#!/usr/bin/env python3
"""К-6: r2_sborka_test -> final_d0_s2emb_h{1,5} для σ̂ C/D на wf: шаги 1-8 прежний ряд, шаги 9-10 прогон К-5 (без ДСВД, конец 2026-02-03).
scripts/analysis/ не правится; склеенный вход - настоящие файлы теневого корня (allowed() сверяет resolve()). Склейка: шаги <= 8 - основная
копия, 9-10 - К-5; хвост шага 8 (17/85 строк 24-30.12.2024) - из v1: во входе rv = σ̂², после сборки подставляется в итог (rv_pred = σ̂_v1²,
leg_raw = rv_pred / F_sm сборки; G7). F_sm шагов 9-10 во входе - по R1 (gibrid_bias.s2_est) на склеенных val (иначе kontrol_r1 отвергает).
  python3 scripts/dannye/wf_sigma_910_sborka.py --main <основная копия> --k5 <k5/out> --shadow <k6/shadow> --out <k6/out>"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KEYS = ["date", "ticker", "step"]
KONEC = pd.Timestamp("2026-02-03")
POSL_STARYJ = 8
VYR = tuple(f"{n}_d0_h{h}_test.parquet" for n in ("etalon", "final") for h in (1, 5, 22))


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def f_sm_r1(val, k):
    """F_sm шага k по правилу R1 (gibrid_bias.s2_est sm: val-шаги k-2..k, y > 1e-12) = exp(s²/2). val: step, rv_actual, leg_raw."""
    sys.path.insert(0, str(HERE.parent / "analysis"))
    import gibrid_bias as BI
    v = val[val.rv_actual > 1e-12]
    E = {"c": {int(s): np.log(g.rv_actual.values + 1e-10) - np.log(g.leg_raw.values) for s, g in v.groupby("step")}}
    return float(np.exp(BI.s2_est("sm", E, "c", k) / 2))


def sklejka(old, new, v1=None, posl=POSL_STARYJ):
    """old, new - final_d0_h{h}_test; v1 - sigma_hyb_h{h}_dev (σ̂ хвоста шага 8). -> склеенный вход сборки."""
    o, n = old[old.step <= posl], new[new.step > posl]
    assert len(n) and set(n.step) == {9, 10}, "К-5: нужны шаги 9-10"
    ot = o[o.part == "test"]
    Fst = ot.groupby("step").F_sm.first()
    assert v1 is not None, "хвост шага 8 - из v1 (σ̂), v1 обязателен"
    nt = new[(new.part == "test") & (new.step <= posl)]
    gap = nt[~nt.set_index(KEYS).index.isin(ot.set_index(KEYS).index)]
    assert gap.step.isin([posl]).all(), "дыры старого ряда - только хвост шага 8"
    sg = [c for c in v1.columns if c.startswith("sigma_pred_h")][0]
    fill = v1.merge(gap[KEYS], on=["date", "ticker"])  # хвост = ключи дыры, значения - v1; rv = σ̂² (σ̂ = float32(sqrt(rv_pred)))
    assert len(fill) == len(gap), "v1 не содержит весь хвост шага 8"
    fill = fill.assign(year=2016 + posl, part="test", rv_actual=np.nan, design=ot.design.iloc[0], F_sm=fill.step.map(Fst).values,
                       rv_pred=fill[sg].astype("float64") ** 2).drop(columns=sg)
    fill["leg_raw"] = fill.rv_pred / fill.F_sm
    s = pd.concat([o, fill, n], ignore_index=True)
    val = s[s.part == "val"]
    for k in (9, 10):
        m = (s.part == "test") & (s.step == k)
        s.loc[m, "F_sm"] = f_sm_r1(val, k)
        s.loc[m, "rv_pred"] = s.loc[m, "leg_raw"] * s.loc[m, "F_sm"]
    return s.sort_values(["part", "step", "date", "ticker"]).reset_index(drop=True)


def f_s2_embargo(val, cal, h, k, kappa=0.5):
    """Независимый пересчёт F шага k: s² по val-шагам k-2..k (годы < 2016+k), без строк, чья цель [t+1, t+h] заходит в 2016+k."""
    c = cal[["date", "ticker"]].drop_duplicates().sort_values(["ticker", "date"])
    c["tend"] = c.groupby("ticker").date.shift(-h)
    v = val[(val.step.between(max(1, k - 2), k)) & (val.rv_actual > 1e-12)].merge(c, on=["date", "ticker"], how="left")
    v = v[v.tend < pd.Timestamp(f"{2016 + k}-01-01")]
    e = np.log(v.rv_actual.values + 1e-10) - np.log(v.leg_raw.values)
    return float(np.exp(kappa * np.var(e)))


def g1(itog, staryj):
    s = staryj[staryj.step <= POSL_STARYJ]
    m = s.merge(itog, on=KEYS, how="left", suffixes=("_o", "_n"))
    d = {c: float(np.nanmax(np.abs(m[c + "_o"] - m[c + "_n"]))) if m[c + "_n"].notna().all() else float("inf") for c in ("leg_raw", "F_sm", "rv_pred")}
    return dict(n=len(s), **{"max_abs_" + c: v for c, v in d.items()}, ok=all(v == 0 for v in d.values()))


def g2(itog, spl, h):
    f = f_s2_embargo(spl[spl.part == "val"], spl, h, 10)
    got = itog[itog.step == 10].F_sm.unique()
    return dict(F_nezavisimyj=f, F_itog=[float(x) for x in got], ok=len(got) == 1 and abs(got[0] / f - 1) < 1e-9)


def g3(itog, novyj, konec=KONEC):
    t, i = novyj[novyj.part == "test"].set_index(KEYS).index, itog.set_index(KEYS).index
    nm, nl = len(t.difference(i)), len(i.difference(t))
    return dict(posl=str(itog.date.max().date()), propusk=nm, lishnih=nl, ok=itog.date.max() == konec and nm == 0 and nl == 0)


def g4(itog, rv_staryj, rv_novyj, W):
    x = itog[["date", "ticker", "rv_pred"]].assign(sig=lambda d: np.sqrt(d.rv_pred))
    r = {}
    for nm, rv, lo, hi in (("2017_2024", rv_staryj, "2017-01-01", "2024-12-31"), ("2025_2026", rv_novyj, "2025-01-01", "2026-02-03")):
        t = W.sdvig(x, "sig", rv, lo, hi)
        r[nm] = dict(corr={str(k): round(v, 4) for k, v in t["corr"].items()}, argmax=t["argmax"], ok=t["ok"])
    return dict(r, ok=all(v["ok"] for v in r.values()))


def g6(itog, v1, h):
    m = itog.merge(v1.rename(columns={f"sigma_pred_h{h}": "s1"}), on=["date", "ticker"])
    m["d"] = np.abs(0.5 * np.log(m.rv_pred) - np.log(m.s1.astype(float)))
    g = m.groupby("step").d.agg(["median", "max", "count"])
    return {int(k): dict(med=round(float(r["median"]), 6), max=round(float(r["max"]), 4), n=int(r["count"])) for k, r in g.iterrows()}


def hvost8(it, old, v1, h):
    """Хвост шага 8 (строк нет в прежнем final_d0_s2emb) после сборки: rv_pred = σ̂_v1², leg_raw = rv_pred / F_sm сборки (с эмбарго)."""
    sg = v1.set_index(["date", "ticker"])[f"sigma_pred_h{h}"].astype("float64")
    t = (it.step == 8).to_numpy() & ~it.set_index(KEYS).index.isin(old.set_index(KEYS).index)
    it = it.copy()
    it.loc[t, "rv_pred"] = sg.reindex(pd.MultiIndex.from_frame(it.loc[t, ["date", "ticker"]])).to_numpy() ** 2
    it.loc[t, "leg_raw"] = it.loc[t, "rv_pred"] / it.loc[t, "F_sm"]
    return it, int(t.sum())


def g7(it, v1, h):
    m = it[it.step == 8].merge(v1, on=["date", "ticker"], how="left")
    n = int((np.sqrt(m.rv_pred.astype("float64")).astype("float32") == m[f"sigma_pred_h{h}"].astype("float32")).sum())
    return dict(sovpalo=n, iz=len(m), ok=n == len(m))


def postroit(main, shadow, spl):
    """Теневой корень: симлинки на основную копию; vyr - настоящие файлы (склейка), R2/final - пустой."""
    spec = importlib.util.spec_from_file_location("wk5", HERE / "wf_sigma_910_progon.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    main, shadow = Path(main).resolve(), Path(shadow).resolve()
    assert main not in shadow.parents and shadow != main, "теневой корень внутри основной копии запрещён"
    P._mirror(main, shadow, replace=("data",))
    P._mirror(main / "data", shadow / "data", replace=("regen",))
    mr, rg = main / "data" / "regen", shadow / "data" / "regen"
    P._mirror(mr, rg, replace=("test_locked",))
    P._mirror(mr / "test_locked", rg / "test_locked", replace=("vyr", "R2"))
    v = rg / "test_locked" / "vyr"
    v.mkdir(parents=True, exist_ok=True)
    for f in VYR:
        shutil.copy2(mr / "test_locked" / "vyr" / f, v / f)
    for h, s in spl.items():
        s.to_parquet(v / f"final_d0_h{h}_test.parquet", index=False)
    P._mirror(mr / "test_locked" / "R2", rg / "test_locked" / "R2", replace=("final",))
    (rg / "test_locked" / "R2" / "final").mkdir(parents=True, exist_ok=True)
    return shadow


def schet(shadow, out):
    os.environ["FASTBT_MAIN"] = str(shadow)
    sys.path.insert(0, str(HERE.parent / "analysis"))
    import r2_sborka_test as R
    assert str(R.MAIN.resolve()) == str(Path(shadow).resolve()), "FASTBT_MAIN не принят сборкой"
    R.JOURNAL = Path(out) / "dev_log_r2_A.md"
    R.run(R.SPEC)
    return R.OUTDIR


def shapka(vhody, vyhody):
    return {k: _sha(p) for k, p in {**vhody, **vyhody}.items()}


def main():
    ap = argparse.ArgumentParser()
    [ap.add_argument("--" + x, required=True) for x in ("main", "k5", "shadow", "out", "novye")]  # novye - корень с features_pit_v3 без ДСВД
    a = ap.parse_args()
    main_, k5, out = Path(a.main), Path(a.k5), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mr = main_ / "data" / "regen"
    spl, vhody = {}, {}
    for h in (1, 5):
        po, pn = mr / "test_locked" / "vyr" / f"final_d0_h{h}_test.parquet", k5 / "test_locked_vyr" / f"final_d0_h{h}_test.parquet"
        pv = mr / "wf" / "ch4_sigma" / f"sigma_hyb_h{h}_dev.parquet"
        spl[h] = sklejka(pd.read_parquet(po), pd.read_parquet(pn), pd.read_parquet(pv))
        vhody.update({f"staryj_h{h}": po, f"k5_h{h}": pn, f"v1_h{h}": pv})
    sh = postroit(main_, a.shadow, spl)
    vhody.update({f"sklejka_h{h}": sh / "data" / "regen" / "test_locked" / "vyr" / f"final_d0_h{h}_test.parquet" for h in (1, 5)})
    od, rep, fp = schet(sh, out), {}, Path("data/regen/features_pit_v3.parquet")
    sys.path.insert(0, str(HERE))
    import wf_sigma_cd as W
    rv = {k: pd.read_parquet(r / fp, columns=["date", "ticker", "rv"]) for k, r in (("s", main_), ("n", Path(a.novye)))}
    vhody.update(rv_staryj=main_ / fp, rv_novyj=Path(a.novye) / fp,
                 kod_sborki=HERE.parent / "analysis" / "r2_sborka_test.py", spec=HERE.parents[1] / "models" / "designs" / "r2" / "r2_A_sborka.yaml")
    vyhody = {f"sborka_syraya_h{h}": od / f"final_d0_s2emb_h{h}_test.parquet" for h in (1, 5)}
    for h in (1, 5):
        f = od / f"final_d0_s2emb_h{h}_test.parquet"
        old = pd.read_parquet(mr / "test_locked" / "R2" / "final" / f.name)
        v1 = pd.read_parquet(mr / "wf" / "ch4_sigma" / f"sigma_hyb_h{h}_dev.parquet")
        it, nt = hvost8(pd.read_parquet(f), old, v1, h)  # хвост шага 8 подставлен после сборки
        it.to_parquet(out / f.name, index=False)
        vyhody[f"vyhod_h{h}"] = out / f.name
        nov = pd.read_parquet(k5 / "test_locked_vyr" / f"final_d0_h{h}_test.parquet")
        rep[f"h{h}"] = dict(G1=g1(it, old), G2=g2(it, spl[h], h), G3=g3(it, nov), G4=g4(it, rv["s"], rv["n"], W), G6=g6(it, v1, h),
                            G7=g7(it, v1, h), hvost_strok=nt)
    (out / "sborka_shapka.json").write_text(json.dumps(shapka(vhody, vyhody), indent=1))
    (out / "gates_k6.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
