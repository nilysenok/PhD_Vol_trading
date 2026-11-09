"""V6-0, G2: PRIOR_SBER — все прежние гибриды с сохранёнными прогнозами SBER по датам: источник и SHA, число дат, QLIKE на своих и общих датах, сверка с отчётом программы (≤ 1e-4) и класс сопоставимости
с доказательством. Ряды без сохранённых прогнозов — строки «нет прогнозов» без оценки. Только чтение."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_class as CL, v6_0_io as IO, v6_0_load as LD, v6_0_q as Q

TOL_REP = 1e-4
CTRL_CMP = ["K1", "K2", "K3", "M", "K8", "D2"]
OTC = IO.ROOT / "docs/otchety"


def dup_of(w: pd.DataFrame, s: str) -> str:
    """Имя замороженного контроля, с которым ряд совпадает побитно (на конечных значениях), иначе ''."""
    x = w[s].to_numpy(float)
    for c in CTRL_CMP:
        if np.array_equal(x, w[c].to_numpy(float), equal_nan=True):
            return c
    return ""


def _rep_table(path, key: str, sel: dict, label_col: str, qcol: str = "qlike") -> dict:
    t = pd.read_csv(path, comment="#")
    for c, v in sel.items():
        t = t[t[c] == v]
    out = {}
    for lab, q in zip(t[label_col].astype(str), t[qcol]):
        out[lab.split(":")[-1].strip()] = float(q)
    return out


def reported() -> dict:
    """{программа/семейство: {id: QLIKE всей истории из отчёта программы}} + источник."""
    f211 = OTC / "hybrid_final_v1/FORECAST_COMPARISONS.csv"
    f212 = OTC / "hybrid_struct_v1/FORECAST_COMPARISONS.csv"
    f214 = OTC / "harj_hybrid_v2/statistical_validation/FORECAST_COMPARISONS.csv"
    r = {"J211": (_rep_table(f211, "", {"period": "all"}, "model"), f211), "J212": (_rep_table(f212, "", {"period": "all"}, "model"), f212),
         "V2": (_rep_table(f214, "", {"period": "all"}, "model"), f214)}
    for prog, name in (("V4", "harj_hybrid_v4"), ("V5", "harj_hybrid_v5")):
        p = OTC / name / "FIN_SBER.csv"
        r[prog] = (_rep_table(p, "", {"window": "all"}, "series"), p)
    return {k: (v[0], str(v[1].relative_to(IO.ROOT))) for k, v in r.items()}


def _rep_for(rep: dict, program: str, family: str, sid: str) -> tuple:
    key = {"J210": "J211", "J211": "J211", "J212": "J212", "V2": "V2", "V4": "V4", "V5": "V5"}.get(program)
    if key is None:
        return float("nan"), ""
    tab, src = rep[key]
    return (tab[sid], src) if sid in tab else (float("nan"), "нет строки в " + src)


def no_forecast_rows() -> pd.DataFrame:
    """Программы без сохранённых по датам прогнозов SBER (или с частью конфигураций без ряда): строки без оценки."""
    reg = pd.read_csv(OTC / "harj_hybrid_v2/ALL_TRIALS.csv")
    cnt = reg.groupby("program").size().to_dict()
    rows = [dict(series="V3 (Ж-221)", program="V3", class_="нет прогнозов", reason="V3 не исполнялся: «Исполнение не начиналось … CHECKPOINT V3 не создавался» (Ж-221); Ж-222 п. 1 «V3 не выполнялся»",
                 evidence="docs/resheniya/resheniya.md, Ж-221 и Ж-222")]
    for prog in ("Ж-204", "Ж-206", "Ж-210", "Ж-211", "Ж-212", "Ж-214"):
        if prog in cnt:
            rows.append(dict(series="%s: конфигурации реестра без ряда по датам" % prog, program=prog, class_="нет прогнозов",
                             reason="в ALL_TRIALS (harj_hybrid_v2) %d конфигураций программы; ряды по датам сохранены только для перечисленных выше (Optuna-пробы и абляции без отдельного ряда)" % cnt[prog],
                             evidence="docs/otchety/harj_hybrid_v2/ALL_TRIALS.csv, сборка по полю program"))
    return pd.DataFrame(rows)


def prior_table(w: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    rep = reported()
    rows = []
    for _, m in meta.iterrows():
        s = m["series"]
        x = w[s].to_numpy(float)
        ok = np.isfinite(x) & (x > 0)
        eq = dup_of(w, s)
        cls, why, ev = CL.classify(s, m["program"], m["family"], eq)
        rq, rsrc = _rep_for(rep, m["program"], m["family"], s.split(":", 1)[-1])
        q = Q.ql(w["rv"].to_numpy()[ok], x[ok]).mean() if ok.any() else float("nan")
        rows.append(dict(series=s, program=m["program"], family=m["family"], source=m["source"], sha256=m["sha256"], n_dates=int(ok.sum()), first_date=w.index[ok].min().date(), last_date=w.index[ok].max().date(),
                         qlike_own=float(q), reported_qlike=rq, reported_source=rsrc, diff_reported=abs(q - rq) if np.isfinite(rq) else float("nan"), rv_maxdiff=m["rv_maxdiff"], tplus_mismatch=m["tplus_mismatch"],
                         equal_to=eq, class_=cls, reason=why, evidence=ev))
    t = pd.DataFrame(rows)
    pit = [s for s, c in zip(t["series"], t["class_"]) if c == CL.PIT]
    ok = np.isfinite(w[pit].to_numpy(float)).all(axis=1) & (w[pit].to_numpy(float) > 0).all(axis=1)
    t["qlike_common_pit"] = [float(Q.ql(w["rv"].to_numpy()[ok], w[s].to_numpy()[ok]).mean()) for s in t["series"]]
    t["n_common_pit"] = int(ok.sum())
    return pd.concat([t, no_forecast_rows()], ignore_index=True).rename(columns={"class_": "class"})


def mismatch_report(t: pd.DataFrame) -> pd.DataFrame:
    """Строки, где QLIKE пересчёта расходится с отчётом программы > 1e-4 (причина — в колонке reported_source / отдельной записи отчёта)."""
    x = t[t["diff_reported"] > TOL_REP]
    return x[["series", "qlike_own", "reported_qlike", "diff_reported", "reported_source"]]
