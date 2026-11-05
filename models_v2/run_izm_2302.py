"""izmeneniya_2302.csv: ключевые строки, QLIKE по срезам и флаги критериев до/после исключения 23.02.2026."""
import sys
from pathlib import Path

ROOT = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading")
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from models_v2 import metrics, prov

OUT = ROOT / "data/regen/v2_sber_h1"
ARH = OUT / "arhiv_do_2302"
TAB = ROOT / "docs/tables/v2_sber_h1"
FILES = ["forecasts_p1", "forecasts_p2", "forecasts_pb", "forecasts_p3", "fix_forecasts_c"]
KEY = [("HAR-J (а)", "harj_a"), ("HAR-J (б)", "harj_b"), ("HAR (б)", "har_b"), ("лучшая этапа 2", "xgb_mse_nohar_mono_b"),
       ("лучшая Б", "b2_m_xgb_b__monthly"), ("Г5 stress h20 (б)", "g5_stress_h20"), ("fx_b2_g132_b", "fx_b2_g132_b"),
       ("fx_b1hard_g132_b", "fx_b1hard_g132_b")]
SL = {"all": lambda y: y > 0, "y2026": lambda y: y == 2026, "2017-2021": lambda y: (y >= 2017) & (y <= 2021),
      "y2022": lambda y: y == 2022, "2023-2025": lambda y: (y >= 2023) & (y <= 2025)}


def load(base):
    fc = pd.concat([pd.read_parquet(base / (f + ".parquet")) for f in FILES], ignore_index=True)
    return fc


def qlikes(fc, model):
    g = fc[fc.model == model]
    if g.empty:
        return None
    y = pd.to_datetime(g["target_date"]).dt.year.to_numpy()
    q = metrics.qlike(g["rv"].to_numpy(), g["sigma2"].to_numpy())
    return {k: float(np.mean(q[f(y)])) for k, f in SL.items()}, len(g)


def flags(tdir, model):
    out = {}
    cr = tdir / "fix_criterion.csv"
    if model.startswith("fx_"):
        t = pd.read_csv(cr, comment="#").set_index("model")
        r = t.loc[model[:-2]]
        out = {"GB": "", "K1": bool(r.K1), "R": bool(r.R), "K2": bool(r.K2), "исправлен": bool(r.fixed)}
    elif model.startswith("b2_") or model.startswith("b1_"):
        t = pd.read_csv(tdir / "pb_verdict.csv", comment="#").set_index("model")
        r = t.loc[model]
        out = {"GB_закрывает_2022": bool(r.closes2022), "GB_сильный": bool(r.closes_and_strong), "GB_слабый": bool(r.closes_and_weak)}
    return out


def main():
    A, B = load(ARH), load(OUT)
    rows = []
    for label, m in KEY:
        qa, qb = qlikes(A, m), qlikes(B, m)
        if qa is None or qb is None:
            print("нет модели", m, "до" if qa is None else "после")
            continue
        for k in SL:
            rows.append({"строка": label, "модель": m, "показатель": "QLIKE_" + k, "до": qa[0][k], "после": qb[0][k],
                         "разность": qb[0][k] - qa[0][k]})
        rows.append({"строка": label, "модель": m, "показатель": "n_прогнозов", "до": qa[1], "после": qb[1], "разность": qb[1] - qa[1]})
        fa, fb = flags(ARH / "tables", m), flags(TAB, m)
        for k in fa:
            rows.append({"строка": label, "модель": m, "показатель": "флаг_" + k, "до": fa[k], "после": fb[k],
                         "разность": "изменился" if fa[k] != fb[k] else "без изменений"})
    df = pd.DataFrame(rows)
    p = TAB / "izmeneniya_2302.csv"
    with open(p, "w", encoding="utf-8") as f:
        f.write("# ↺ пересчёт без 23.02.2026 (nonworking_2025); до = data/regen/v2_sber_h1/arhiv_do_2302, после = текущие; "
                "bars=%s imoex=%s; разработка, до аудита; QLIKE (б) по дате цели\n" % (prov.sha_file(prov.BARS)[:16], prov.sha_file(ROOT / "data/iss_check/index_10m/IMOEX.parquet")[:16]))
        df.to_csv(f, index=False, lineterminator="\n", float_format="%.6f")
    pd.set_option("display.width", 220)
    print(df.to_string())


main()
