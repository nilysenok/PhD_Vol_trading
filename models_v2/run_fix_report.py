"""Ж-200: итоговые таблицы по строкам И1…И6 (после run_fix.py --stage a и b). Запуск: python3 models_v2/run_fix_report.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models_v2 import prov  # noqa: E402
from models_v2.s4 import prov4, report  # noqa: E402

OUT, TAB = prov4.OUT, prov4.TAB
REFS = ("xgb_mse_nohar_mono_b", "b2_m_xgb_b__daily", "b1_xgb_c100__daily")
IMOEX = prov.ROOT / "data/iss_check/index_10m/IMOEX.parquet"
RESTAMP = ("fix_i6_choice.csv", "fix_i2_choice_hard.csv")   # пишет run_fix.py без sha IMOEX; здесь правится только строка шапки


def imoex_sha() -> str:
    """sha256[:16] IMOEX, если в таблицах есть строки с IMOEX (счёт этапа c выполнен), иначе пусто."""
    return prov.sha_file(IMOEX)[:16] if (OUT / "fix_forecasts_c.parquet").exists() else ""


def restamp(path, imx: str) -> None:
    """В первой строке шапки «imoex=нет» -> «imoex=<sha>»; строки данных не трогаются."""
    head, sep, rest = Path(path).read_text(encoding="utf-8").partition("\n")
    assert head.startswith("#") and "imoex=нет" in head, path
    Path(path).write_text(head.replace("imoex=нет", "imoex=" + imx, 1) + sep + rest, encoding="utf-8")


def load():
    parts = [pd.read_parquet(OUT / ("fix_forecasts_%s.parquet" % s)) for s in "ac" if (OUT / ("fix_forecasts_%s.parquet" % s)).exists()]
    leak = pd.concat([pd.read_csv(TAB / ("fix_leak_%s.csv" % s), comment="#") for s in "ac" if (TAB / ("fix_leak_%s.csv" % s)).exists()])
    p1 = pd.read_parquet(OUT / "forecasts_p1.parquet")
    refs = pd.concat([pd.read_parquet(OUT / "forecasts_p2.parquet"), pd.read_parquet(OUT / "forecasts_pb.parquet")])
    fc = pd.concat(parts + [p1[p1.model == report.BASE], refs[refs.model.isin(REFS)]], ignore_index=True)
    for c in ("d", "target_date"):
        fc[c] = fc[c].astype("datetime64[ns]")
    n = fc.groupby("model").size()
    assert n.nunique() == 1 and n.iloc[0] == 2271 and (fc.sigma2 > 0).all(), n.value_counts()
    return fc.sort_values(["model", "d"]).reset_index(drop=True), leak.reset_index(drop=True)


def main():
    fc, leak = load()
    rvi = pd.read_parquet(prov.RVI)
    mt = report.metrics_all(fc, rvi)
    over = pd.concat([pd.read_csv(TAB / ("fix_overfit_%s.csv" % s), comment="#") for s in "ac"])
    cr = report.criterion(fc, leak, over)
    q = mt[mt.slice == "all"].set_index("model")["qlike"]
    sl = mt.pivot(index="model", columns="slice", values="qlike")
    mr = mt[mt.slice == "all"].set_index("model")["mean_r"]
    cr["qlike_y2022"] = [sl.loc[m + "_b", "y2022"] for m in cr.model]
    cr["qlike_mar_jun_2020"] = [sl.loc[m + "_b", "mar_jun_2020"] for m in cr.model]
    cr["rv_over_sigma2"] = [mr[m + "_b"] for m in cr.model]
    cr["qlike_all_b"] = [q[m + "_b"] for m in cr.model]
    cr["best_all_period_ex_post"] = cr["qlike_all_b"] == cr["qlike_all_b"].min()   # выбор на оценочном периоде
    imx = imoex_sha()
    prov4.write_csv(mt, TAB / "fix_metrics.csv", "П4: срезы по дате цели; mean_r = среднее RV/σ̂²", imx)
    prov4.write_csv(cr, TAB / "fix_criterion.csv", "Ж-202: исправлен = Р (блок/обучение <= 1,3) и К2 (2017–2021 и 2023–2025 <= HAR-J(б)); К1 (full <= 0,95·work) справочно", imx)
    for name in RESTAMP:
        if imx and (TAB / name).exists() and "imoex=нет" in (TAB / name).open(encoding="utf-8").readline():
            restamp(TAB / name, imx)
    chosen = {r.model + "_b": "прошла Р+К2" for r in cr.itertuples() if r.fixed}
    best = cr.sort_values("qlike_all_b").iloc[0].model + "_b"
    chosen.setdefault(best, "лучшая по QLIKE(б) за весь период (выбор на оценочном периоде)")
    dmt = report.dm_table(fc, chosen)
    prov4.write_csv(dmt, TAB / "fix_dm.csv", "DM: d = QLIKE(HAR-J(б)) − QLIKE(модель); плюс = модель лучше; Ньюи–Уэст лаг 5, Холм внутри выборки", imx)
    pd.set_option("display.width", 250)
    print(cr.to_string(), "\n", dmt.to_string())
    print("метрики: строк", len(mt), "; qlike all:\n", q.sort_values().to_string())
    assert np.isfinite(cr.qlike_work).all()


if __name__ == "__main__":
    main()
