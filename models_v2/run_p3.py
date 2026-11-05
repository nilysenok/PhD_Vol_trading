"""Этап 3 (Ж-197 п. 3): Г0, Г1 (= строки Б-1 из forecasts_pb), Г2, Г3, Г4, Г5 поверх прогнозов этапов 1–2 и Б.

Запуск: python3 models_v2/run_p3.py   (минуты, 1 процесс). Бустинг не переобучается: Г2–Г5 — только по прогнозам.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models_v2 import metrics, prov  # noqa: E402
from models_v2.s3 import chain, leak3, prov3  # noqa: E402

OUT = prov.ROOT / "data/regen/v2_sber_h1"
TAB = prov.ROOT / "docs/tables/v2_sber_h1"
BEST2 = "xgb_mse_nohar_mono_b"
G1_SRC = {"g1_c050": "b1_xgb_c050_b__daily", "g1_c075": "b1_xgb_c075_b__daily", "g1_c100": "b1_xgb_c100_b__daily",
          "g1_c100_nobias": "b1_xgb_c100_nobias_b__daily"}
NOTE = "выбор на оценочном периоде (QLIKE по всей выборке 2017–02.2026)"


def series(df: pd.DataFrame, name: str, ref: pd.DataFrame) -> pd.DataFrame:
    f = df[df.model == name].sort_values("d").reset_index(drop=True)
    for c in ("d", "target_date"):
        f[c] = f[c].astype("datetime64[ns]")
    assert len(f) == len(ref) and (f["d"].values == ref["d"].values).all() and (f["target_date"].values == ref["target_date"].values).all()
    assert (f["rv"].values == ref["rv"].values).all() and (f["sigma2"] > 0).all() and np.isfinite(f["sigma2"]).all(), name
    return f


def qall(df, names) -> dict:
    return {n: float(metrics.qlike(df[df.model == n]["rv"], df[df.model == n]["sigma2"]).mean()) for n in names}


def g0_g2(p2: pd.DataFrame) -> tuple:
    names = sorted(m for m in p2.model.unique() if m.endswith("_b"))
    assert len(names) == 16, len(names)
    q = qall(p2, names)
    g0 = min(q, key=q.get)
    if g0 != BEST2:
        raise SystemExit("Г0 не совпал: %s вместо %s — стоп и доклад" % (g0, BEST2))
    har_b = [n for n in names if "_nohar_" not in n]
    assert len(har_b) == 8, har_b
    return g0, min(har_b, key=q.get), q


def inputs(p1, p2, pb_):
    ref = p1[p1.model == "harj_b"].sort_values("d").reset_index(drop=True)
    for c in ("d", "target_date"):
        ref[c] = ref[c].astype("datetime64[ns]")
    g0, g2, q16 = g0_g2(p2)
    ln = {"h": np.log(ref["sigma2"].values), "b": np.log(series(p2, g0, ref)["sigma2"].values),
          "g2": np.log(series(p2, g2, ref)["sigma2"].values)}
    for g, src in G1_SRC.items():
        ln[g] = np.log(series(pb_, src, ref)["sigma2"].values)
    return ref, ln, g0, g2, q16


def regime_high(ref: pd.DataFrame) -> np.ndarray:
    rvi = pd.read_parquet(prov.RVI)
    rvi = rvi[rvi["date"] <= pd.Timestamp("2021-12-31")]  # RVI >= 2022 закрыт (Ж-197, ❓ владельцу): после 2021 условие RVI ложно
    return (metrics.rvi_regime(ref["d"], rvi) == "high").to_numpy(float)


def choice_table(a, ln, g0, g2, q16, aux, base_pool) -> pd.DataFrame:
    rows = [{"step": "Г0 (16 строк (б) этапа 2)", "model": n, "qlike_all": q, "chosen": n == g0} for n, q in q16.items()]
    cand = (("Г4: спокойный эксперт", "C", list(chain.G1) + ["g2", "g3"]), ("Г5: основа", "base", base_pool))
    for step, key, names in cand:
        rows += [{"step": step, "model": n, "qlike_all": chain.qmean(a["rv"], ln[n]), "chosen": n == aux["choice"][key]} for n in names]
    rows.append({"step": "Г2 (8 строк (б) с HAR-признаком) -> выбрана", "model": g2, "qlike_all": q16[g2], "chosen": True})
    return pd.DataFrame(rows)


def main():
    p1, p2 = (pd.read_parquet(OUT / ("forecasts_%s.parquet" % s)) for s in ("p1", "p2"))
    pb_ = pd.read_parquet(OUT / "forecasts_pb.parquet")
    ref, ln, g0, g2, q16 = inputs(p1, p2, pb_)
    daily = pd.read_parquet(OUT / "daily_p1.parquet")
    rv_w = daily["rv_w"].reindex(ref["d"]).to_numpy()
    assert np.isfinite(rv_w).all()
    rvh = regime_high(ref)
    a = chain.prep(ref["d"].values, ref["target_date"].values, ref["rv"].values, rv_w, rvh)
    sig, aux = chain.build(a, ln)
    ch = aux["choice"]
    lw = dict(ln, g3=np.log(sig["g3"]), g4_hard=np.log(sig["g4_hard"]), g4_logit=np.log(sig["g4_logit"]))
    out = {g: np.exp(ln[g]) for g in G1_SRC} | {"g2": np.exp(ln["g2"])} | sig
    fc = pd.concat([pd.DataFrame({"d": ref["d"], "target_date": ref["target_date"], "model": n, "sigma2": s, "rv": ref["rv"]})
                    for n, s in out.items()])
    fc.to_parquet(OUT / "forecasts_p3.parquet")
    pool = list(chain.G1) + ["g2", "g3", "g4_hard", "g4_logit"]
    ex = " | Г0=%s g2=%s C=%s base=%s | %s" % (g0, g2, ch["C"], ch["base"], NOTE)
    prov3.write_csv(choice_table(a, lw, g0, g2, q16, aux, pool), TAB / "p3_choice.csv", ex)
    prov3.write_csv(pd.DataFrame({"d": ref["d"], "w": aux["w"], "a": aux["a"], "b": aux["b"]}), TAB / "p3_w.csv", "ряд w_d Г3 и (a, b) Г4 (logit)")
    year = ref["d"].dt.year
    prov3.write_csv(pd.DataFrame({"year": year, "stress": a["st"], "rvi_high": rvh}).groupby("year")
                    .agg(share_stress=("stress", "mean"), share_rvi_high=("rvi_high", "mean"), n=("stress", "size")).reset_index(),
                    TAB / "p3_stress.csv", "доля стрессовых дней Г4 по году d; после 2021 RVI не используется")
    pd.DataFrame({"d": ref["d"], "stress": a["st"]}).to_parquet(OUT / "stress_p3.parquet")
    lk = pd.read_csv(TAB / "pb_leak.csv", comment="#")
    leak = pd.concat([leak3.g1_rows(lk, {g: s.replace("__daily", "") for g, s in G1_SRC.items()}), leak3.chain_rows(a, ln, ch)])
    prov3.write_csv(leak, TAB / "p3_leak.csv", "(i)–(iii) по допускам GATES; Г1 — из pb_leak.csv (monthly), Г3–Г5 — ежедневный счёт" + ex)
    print("готово:", ex, "моделей", fc["model"].nunique(), "дат", len(ref))


if __name__ == "__main__":
    main()
