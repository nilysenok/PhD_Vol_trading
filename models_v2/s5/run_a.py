"""Волна A Ж-204: python3 models_v2/s5/run_a.py --stage a1|a2|pred|manifest|a5|a8|a9|a10|a11|a12|a13|registry|readme|check-readme|size|all
Модели не обучаются; a2 — воспроизведение 3 дат (не trial). Один процесс."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models_v2 import metrics  # noqa: E402
from models_v2.s5 import checks, common, cum, frame, groups, hyp, lossck, manifest, registry, regimes, reprod, report, stats  # noqa: E402

R = common.REP
TOL = 1e-12


def load_pred() -> pd.DataFrame:
    return pd.read_parquet(R / "PREDICTIONS.parquet")


def stage_a1() -> dict:
    c = common.controls()
    q = {"b2": float(metrics.qlike(c.rv, c.s2_b2).mean()), "harj": float(metrics.qlike(c.rv, c.s2_harj).mean()), "n": len(c)}
    fm = common.read_comment_csv(common.TAB / "fix_metrics.csv")
    pm = common.read_comment_csv(common.TAB / "p1_metrics.csv")
    ref_b = float(fm[(fm.model == common.B2) & (fm.slice == "all")].qlike.iloc[0])
    ref_h = float(pm[(pm.model == common.HJ) & (pm.slice == "all")].qlike.iloc[0])
    q.update(ref_b2_table=ref_b, ref_harj_table=ref_h, diff_b2=abs(q["b2"] - ref_b), diff_harj=abs(q["harj"] - ref_h))
    print("A1:", q, "округл.: %.4f %.4f" % (q["b2"], q["harj"]))
    if q["diff_b2"] > TOL or q["diff_harj"] > TOL or round(q["b2"], 4) != 0.2205 or round(q["harj"], 4) != 0.2337 or q["n"] != common.N_GRID:
        print("СТОП: контроли не совпали")
        sys.exit(1)
    return q


def stage_a2() -> dict:
    out = reprod.run()
    print("A2:", json.dumps(out, ensure_ascii=False))
    if not out["reproduced"]:
        print("СТОП: воспроизведение не совпало")
        sys.exit(1)
    return out


def stage_manifest(q: dict) -> None:
    rp = R / "REPRODUCTION_A2.json"
    manifest.write(R / "BASELINE_MANIFEST.json", json.loads(rp.read_text()) if rp.exists() else None, q)


def stage_a5(df) -> None:
    t = lossck.eps_check(df, frame.daily_p1())
    common.write_csv(t, R / "EPS_CHECK.csv", "правило эпсилон (диагностика, PIT)")
    print(t.T.to_string())


def stage_a8(df) -> None:
    d = stats.diagnostics(df)
    common.write_csv(d, R / "ERROR_DIAGNOSTICS.csv", "срезы по дате цели; Δ = QLIKE Б-2 − HAR-J(б)")
    common.write_csv(stats.annual(d), R / "ANNUAL_MODEL_COMPARISON.csv", "по годам цели; 2026 неполный (янв–фев)")
    common.write_csv(stats.rolling(df), R / "ROLLING_252.csv", "скользящие 252 даты цели")


def stage_a9(df) -> None:
    t = cum.table(df)
    common.write_csv(t, R / "CUMULATIVE_LOSS_DIFF.csv", "Σ(qlike_b2 − qlike_harj)")
    cum.plot(t, R / "cum_loss_diff.png")


def stage_a10(df) -> None:
    dl = frame.daily_p1()
    fact = groups.table(df, groups.fact_bucket(df["rv"]), "fact", "объяснительно: квантили всей выборки (знают будущее), не PIT")
    pit = groups.table(df, groups.pit_bucket(df, dl), "pit", "PIT: пороги по 252 предыдущим RV_d")
    common.write_csv(fact, R / "RV_GROUPS_FACT.csv", "5 групп по факту RV цели")
    common.write_csv(pit, R / "RV_GROUPS_PIT.csv", "5 групп по RV_d, известной на d")
    ev = pd.concat([groups.events(df, dl, k) for k in ("jump", "rv_change")])
    common.write_csv(ev, R / "JUMP_EVENTS.csv", "траектории ±5 дней вокруг событий (offset — по дате прогноза)")


def stage_a11(df) -> None:
    common.write_csv(regimes.table(df), R / "REGIME_DIAGNOSTICS.csv", "режимы по прошлому (252 строки), стресс Г4 из пилота")


def stage_a12(df) -> None:
    _, npairs = frame.features_b2()
    ov, ch = hyp.load_inputs()
    yt = hyp.years_table(df, npairs, ov, ch)
    top = hyp.top_days(df)
    common.write_csv(yt, R / "HYPOTHESIS_YEARS.csv", "годовые величины для гипотез; es_over_train — справка (ж)")
    common.write_csv(top, R / "TOP20_DAYS.csv", "топ-20 дней 2017–2021 по |разности потерь| (диагностика)")
    common.write_csv(hyp.top_days(df, until=2026), R / "TOP20_DAYS_ALL.csv", "топ-20 дней всей выборки по |разности потерь| (диагностика)")
    common.write_csv(hyp.sensitivity(df), R / "SENSITIVITY_A.csv", "диагностическая чувствительность, не результат (§24.9 блока)")
    h = hyp.hypotheses(df, yt, top)
    common.write_csv(h, R / "HYPOTHESES_A.csv", "правила вывода заданы в GATES_A.md до счёта")
    print(h.to_string())


def stage_a13(df) -> None:
    common.write_csv(lossck.table(df), R / "LOSS_CHECK.csv", "сдвиг масштаба exp(c): %s; обучение: %s" % (lossck.NOTE, lossck.TRAIN_LOSS["objective"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    s = ap.parse_args().stage
    R.mkdir(parents=True, exist_ok=True)
    if s == "size":
        bad = checks.size_problems() + checks.scope_problems()
        print("size/scope:", bad or "OK")
        sys.exit(1 if bad else 0)
    if s == "check-readme":
        ok = report.check_readme()
        print("README авто-блоки совпадают с перегенерацией:", ok)
        sys.exit(0 if ok else 1)
    q = stage_a1()
    if s in ("a1",):
        return
    if s in ("a2", "all"):
        stage_a2()
    if s == "a2":
        return
    if s in ("pred", "all") or not (R / "PREDICTIONS.parquet").exists():
        frame.make()
    df = load_pred()
    stages = {"manifest": lambda: stage_manifest(q), "a5": lambda: stage_a5(df), "a8": lambda: stage_a8(df), "a9": lambda: stage_a9(df),
              "a10": lambda: stage_a10(df), "a11": lambda: stage_a11(df), "a12": lambda: stage_a12(df), "a13": lambda: stage_a13(df),
              "registry": lambda: registry.ensure(frame._key()), "readme": report.write_readme}
    for name, fn in stages.items():
        if s in (name, "all"):
            print("--", name)
            fn()
    print("PREDICTIONS:", df.shape, "np.isfinite rv:", bool(np.isfinite(df.rv).all()))


if __name__ == "__main__":
    main()
