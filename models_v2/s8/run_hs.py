"""Ж-212 SBER h=1: стадии по одной. `python3 -W ignore -m models_v2.s8.run_hs --stage ctrl|table|diag|q8` (этап 0 и A; B–G — после коммита гейтов)."""
from __future__ import annotations

import argparse

import pandas as pd

from models_v2.s8 import hs_ctrl, hs_diag, hs_diag8, hs_pred, hs_table
from models_v2.s8 import hs_paths as P


def st_table(_a=None) -> int:
    o = hs_table.build()
    hs_table.save(o)
    print("ERROR_STRUCTURE_DAILY.parquet: %d строк, %d столбцов; NaN mkt_rv: %d" % (len(o), o.shape[1], int(o["mkt_rv"].isna().sum())), flush=True)
    return 0


def _load() -> pd.DataFrame:
    return pd.read_parquet(P.OUT / "ERROR_STRUCTURE_DAILY.parquet")


def st_diag(_a=None) -> int:
    o = _load()
    P.write_csv(hs_diag.when_table(o), "Q1_Q2_WHEN.csv", "A: вопросы 1–2; dq = QLIKE(HAR) − QLIKE(XJ), плюс — бустинг лучше; факт-условия ex post (диагностика)")
    P.write_csv(hs_diag.slice_table(o), "Q4_Q6_SLICES.csv", "A: вопросы 4–6; срезы по прошлому (PIT-терцили), годам, режимам; mean_r = среднее RV/σ̂²")
    t = hs_pred.table(o)
    P.write_csv(t, "Q3_PREDICTORS.csv", "A: вопрос 3; OLS dq~z с HAC, Холм по признакам; robust = Холм<0,10, знак в 2 из 3 периодов, |t|>2 без 24.02.2022 и на подрезанном dq")
    print("Q3: признаков %d, robust %d" % (len(t), int(t["robust"].sum())), flush=True)
    print(t.head(8)[["feature", "t_nw", "p_holm", "t_nw_ex_240222", "robust"]].round(3).to_string(index=False), flush=True)
    return 0


def st_q8(_a=None) -> int:
    o = _load()
    P.write_csv(hs_diag8.q7_table(o), "Q7_DELTA_INFO.csv", "A: вопрос 7; MZ-регрессия ошибки ML на δ = ln HAR − ln ML, HAC; persistence = dq на past_dqX_m22 (закрытые)")
    P.write_csv(hs_diag8.beta_profile(o), "Q8_BETA_PROFILE.csv", "A: вопрос 8; профиль QLIKE и sd ln-ошибки по β в σ̂²=ML·exp(β·δ) (в выборке, потолок)")
    t, y = hs_diag8.q8_tables(o)
    P.write_csv(t, "Q8_ENCOMPASSING.csv", "A: вопрос 8; σ̂²=ML·exp(Xγ) QLIKE-согласованно; in_sample — потолок; delta_* — OOS по прошлому (минус — лучше базы), DM Ньюи–Уэст")
    P.write_csv(y, "Q8_OOS_BY_YEAR.csv", "A: вопрос 8; OOS по прошлому, QLIKE по году цели; delta = QLIKE(новая) − QLIKE(база)")
    print(t[["spec", "base", "p_joint_in_sample", "gain_in_sample_ceiling", "delta_all", "p_all", "delta_no_240222", "p_no_240222"]].round(5).to_string(index=False), flush=True)
    return 0


def st_ledger(_a=None) -> int:
    from models_v2.s8 import hs_ledger
    hs_ledger.init()
    ck = hs_ledger.write_check(hs_ledger.load())
    print("ledger: 521 строка побайтно; used", ck["used"], "total", ck["used_total"], "N_final", ck["N_final"], flush=True)
    return 0


def _lazy(mod: str, fn: str):
    def run(a=None) -> int:
        import importlib
        return getattr(importlib.import_module("models_v2.s8." + mod), fn)(a)
    return run


STAGES = {"ctrl": hs_ctrl.stage, "table": st_table, "diag": st_diag, "q8": st_q8, "ledger": st_ledger,
          "base": _lazy("hs_bstage", "stage_base"), "b": _lazy("hs_bstage", "stage_b"), "c": _lazy("hs_cstage", "stage_c"),
          "d": _lazy("hs_dstage", "stage_d"), "e": _lazy("hs_estage", "stage_e"), "f": _lazy("hs_fstage", "stage_f"),
          "g": _lazy("hs_gstage", "stage_g"), "final": _lazy("hs_final", "stage_final")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    return STAGES[ap.parse_args().stage]()


if __name__ == "__main__":
    raise SystemExit(main())
