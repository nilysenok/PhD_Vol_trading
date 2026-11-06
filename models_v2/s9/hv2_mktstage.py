"""Ж-214, лист 4, стадия `mkt` (V10, ступень Л8; У29 GATES_V2). Минимальный набор E1 (Mb), E2 (semicov), E3 (idio), E4 (relrv) → V13 (лучшая из E2–E4 против Mb) → при «продолжать»: E5 (dmkt), E6 (mktsemi),
E7 (Mb + информативные блоки), E8 (A0 + они же, без Mb), E9 (окно corr/beta 63); E10 — резерв. «Информативен сверх Mb»: Δcrit_wf <= −0,0005, ΔQLIKE < 0 (вся история и без 24.02.2022), лет лучше >= 6 из 10."""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd

from models_v2.s9 import hv2_a0 as A0M, hv2_dec as DC, hv2_decstage as DS, hv2_eval as E, hv2_harq as HQ, hv2_ledger as L, hv2_mkt as MK, hv2_msfeat as MS, hv2_paths as P, hv2_world as W

DIR, SUB = "MKT", "market_commonality"
CTRL = ("A0", "XJ", "LF", "Mb")
CFG = {"E1": ("mb", True, (), 22), "E2": ("mb_semicov", True, ("semicov",), 22), "E3": ("mb_idio", True, ("idio",), 22), "E4": ("mb_relrv", True, ("relrv",), 22),
       "E5": ("mb_dmkt", True, ("dmkt",), 22), "E6": ("mb_mktsemi", True, ("mktsemi",), 22), "E9": ("mb_win63", True, (), 63)}
BLK = {"E2": "semicov", "E3": "idio", "E4": "relrv", "E5": "dmkt", "E6": "mktsemi"}
MIN_SET = ("E1", "E2", "E3", "E4")


def mods() -> tuple:
    """Д5: ключ кэша = код + M10 + components + файл баров IMOEX (дневной ряд индекса строится из него)."""
    return (MK, MS, DC, HQ, A0M, W, sys.modules[__name__], types.SimpleNamespace(__file__=str(P.IMOEX)))


def compute(dw, use_mb: bool, blocks: tuple, win: int, name: str) -> dict:
    s2, B, names, fb = MK.run(dw, use_mb, blocks, win)
    return {name: s2, name + "__fb": fb.astype(float), **{"%s__%s" % (name, n): B[:, k] for k, n in enumerate(names)}}


def trunc(dw, use_mb: bool, blocks: tuple, win: int) -> float:
    """Главный тест обрезки: σ̂² всех дат ≤ d из исходного и испорченного (всё после d, включая IMOEX и синхронные величины) мира."""
    return max(DC.trunc_diff(lambda x: MK.run(x, use_mb, blocks, win)[0], dw, int(dw.el[k])) for k in E.TRUNC_AT)


def stage_mkt(_a=None) -> int:
    ctx = E.context()
    dw = MS.build_world(ctx["w"], pd.read_parquet(P.M10), pd.read_parquet(P.IMOEX))
    rows, ser = [], {}

    def go(name: str, kind: str, use_mb: bool, blocks: tuple, win: int) -> dict:
        key = L.register("market", {"name": name, "kind": kind, "use_mb": use_mb, "blocks": list(blocks), "win": win, "controls": ["Mb (E1)", "A0", "XJ", "LF"],
                                    "spec": "log-OLS A0-спецификации + %s%s" % ("Mb (imoex_d/w/m, corr/beta окна %d)" % win if use_mb else "без Mb", " + блоки " + ",".join(blocks) if blocks else "")})
        ser[name] = E.cached(DIR, name, mods(), lambda: compute(dw, use_mb, blocks, win, name), ctx)
        if name == "E1":
            ctx["refs"]["Mb"] = ser[name][name]
        r = E.row(ctx, name, ser[name][name], {"config": name, "kind": kind, "blocks": ",".join(blocks), "win": win, "use_mb": use_mb, "key": key, "trunc_max_diff": trunc(dw, use_mb, blocks, win),
                                               "n_fallback": int((ser[name][name + "__fb"] > 0).sum())}, CTRL)
        r["informative_vs_Mb"] = bool(blocks) and use_mb and MK.informative(r)
        rows.append(r)
        return r

    for n in MIN_SET:
        go(n, *CFG[n])
    v13 = E.v13([r for r in rows if r["config"] != "E1"], ("Mb",), len(rows), L.LIMITS["market"])
    ran = list(MIN_SET)
    if v13["continue"]:
        ran += run_rest(go, rows)
    E.decide("V10", {**v13, "ran": ran, "unused_trials": L.LIMITS["market"] - len(rows), "informative_blocks": [r["config"] for r in rows if r["informative_vs_Mb"]]})
    write(ctx, rows, ser, v13)
    return 0 if all(r["trunc_max_diff"] == 0 for r in rows) else 1


def run_rest(go, rows: list) -> list:
    ran = []
    for n in ("E5", "E6"):
        go(n, *CFG[n])
        ran.append(n)
    inf = tuple(BLK[r["config"]] for r in rows if r["config"] in BLK and r["informative_vs_Mb"])
    if inf:
        go("E7", "mb_all_new", True, inf, 22)
        go("E8", "a0_new_only", False, inf, 22)
        ran += ["E7", "E8"]
    go("E9", *CFG["E9"])
    return ran + ["E9"]


def write(ctx: dict, rows: list, ser: dict, v13: dict) -> None:
    refs, df = ctx["refs"], ctx["df"]
    ctl = [E.row(ctx, k, refs[k], {"kind": "контроль"}, CTRL) for k in ("A0", "XJ", "LF")]
    P.write_csv(pd.DataFrame(ctl + rows), "MARKET_COMMONALITY.csv", "V10; Δ — строка минус контроль (минус — строка лучше), Mb = E1; V13: лучшая из E2–E4 = %s; потолок: выбор по crit_wf на всей истории" % v13["best"], SUB)
    cols = ["config", "blocks", "win", "d_crit_Mb", "d_qlike_Mb", "d_qlike_no_240222_Mb", "years_better_Mb", "p_dm_Mb", "informative_vs_Mb", "d_crit_A0", "d_qlike_no_240222_A0", "years_better_A0"]
    P.write_csv(pd.DataFrame(rows)[cols], "MKT_BLOCKS_INFORMATIVE.csv", "что дают показатели СВЕРХ существующих признаков IMOEX (Mb): Δ к Mb и критерий информативности V10", SUB)
    P.write_csv(DS.by_year(df, {"A0": refs["A0"], "XJ": refs["XJ"], **{r["config"]: ser[r["config"]][r["config"]] for r in rows}}), "MKT_BY_YEAR.csv", "QLIKE по годам цели; Δ к A0 и XJ", SUB)
    yr = pd.DatetimeIndex(df["d"]).year.to_numpy()
    co = [{"config": r["config"], "year": int(Y), **{c.split("__", 1)[1]: float(ser[r["config"]][c][np.flatnonzero(yr == Y)[-1]]) for c in ser[r["config"]] if "__" in c and not c.endswith("__fb")}}
          for r in rows for Y in sorted(set(yr))]
    P.write_csv(pd.DataFrame(co), "MKT_COEFS.csv", "коэффициенты log-OLS на последнюю дату сетки года", SUB)
    print(pd.DataFrame(rows)[["config", "qlike_all", "crit_wf", "d_crit_Mb", "d_crit_A0", "d_crit_XJ", "years_better_Mb", "informative_vs_Mb", "n_fallback", "trunc_max_diff"]].round(5).to_string(index=False), flush=True)
