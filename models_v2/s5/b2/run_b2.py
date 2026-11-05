"""Поток B2 (Ж-204, волна B): python3 models_v2/s5/b2/run_b2.py --stage control|info|ablate|window|shap|all   (процессов <= 4)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models_v2.s5 import common, frame  # noqa: E402
from models_v2.s5.b2 import engine, feat, info, shap, tab, world  # noqa: E402

REP, OUT = common.REP, world.OUT
LIM = {"feature_engineering": 22, "rolling_recency": 17}


def write(df: pd.DataFrame, path: Path, note: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# разработка; волна B Ж-204 поток B2; до аудита; оценка 2017–02.2026 (2022+ — просмотр контроля гл. 3 по SBER), историю нельзя "
                "считать независимым OOS; код b2 sha=%s; %s\n" % (world.code_sha(), note))
        df.to_csv(f, index=False, lineterminator="\n")


def stage_control(L) -> None:
    world.compute(L, [world.BASE])
    fc, _ = world.get(L, world.BASE)
    har, b2 = world.controls_frames()
    err = world.check_control(fc, b2)
    q = float(tab.metrics.qlike(fc["rv"], fc["sigma2"]).mean())
    qh = float(tab.metrics.qlike(har["rv"], har["sigma2"]).mean())
    print("G1: max rel %.3e, QLIKE BASE %.12f, HAR-J(б) %.12f" % (err, q, qh))
    if err > 1e-9 or abs(q - 0.220486502764) > 1e-9:
        sys.exit("G1 КРАСНЫЙ: контроль не воспроизведён — стоп и доклад")


def stage_info(L) -> pd.DataFrame:
    F = L["F"]
    base = pd.DataFrame(L["dz0"].X, columns=frame.FEAT, index=pd.DatetimeIndex(F.dates))
    t = info.check(base, L["feats"], feat.GROUPS)
    t.to_csv(OUT / "INFO_CHECK_B2.csv", index=False)
    acc = info.accepted(t)
    (OUT / "accepted.json").write_text(json.dumps({g: list(c) for g, c in acc.items()}, ensure_ascii=False))
    print(t.groupby("group", sort=False)["status"].value_counts().to_string())
    return t


def accepted() -> dict:
    return {g: tuple(c) for g, c in json.loads((OUT / "accepted.json").read_text()).items()}


def build_table(L, cfgs, path: Path, note: str, trials: bool = True) -> pd.DataFrame:
    har, b2 = world.controls_frames()
    stb = world.get(L, world.BASE)[1]
    meta0 = {"crit_loyo_b2": stb["crit_loyo"], "crit_wf_b2": tab.crit_wf(world.get(L, world.BASE)[0])}
    out, tr = [], []
    for c in [world.BASE] + cfgs:
        fc, st = world.get(L, c)
        r = tab.row(c.name, fc, har, b2, st, dict(meta0, trial_id=c.trial_id(), direction=c.direction, crit_loyo=st["crit_loyo"]))
        out.append(r)
        if c is not world.BASE:
            tr.append(tab.trial_row(c, r, st["secs"]))
    df = pd.DataFrame(out)
    write(df, path, note)
    n = tab.register(tr) if trials else 0
    print(df[["model", "qlike_all", "qlike_p2017_2021", "qlike_y2022", "qlike_p2023_2025", "dloss_harj_no_20220224", "crit_loyo", "crit_wf",
              "perspective"]].round(5).to_string(), "\nновых trials:", n)
    return df


def feature_cfgs(acc: dict) -> list:
    return ([engine.Cfg("b2_" + g.lower(), add=cols) for g, cols in acc.items() if cols] +
            [engine.Cfg("b2_" + k, drop=k) for k in engine.DROP])


def stage_ablate(L) -> None:
    acc = accepted()
    first = feature_cfgs(acc)
    world.compute(L, [world.BASE] + first)
    grp = [c for c in first if c.add]
    score = {c.name: world.get(L, c)[1]["crit_loyo"] for c in grp}
    best = sorted(grp, key=lambda c: score[c.name])
    combos = [engine.Cfg("b2_top%d" % k, add=tuple(x for c in best[:k] for x in c.add)) for k in (2, 3) if len(best) >= k]
    world.compute(L, combos)
    build_table(L, first + combos, REP / "FEATURE_ABLATIONS.csv", "12 trials плана; crit_loyo — среднее годовых LOYO; перспективно по GATES_B2 §5")


def stage_window(L) -> None:
    cfgs = engine.window_cfgs()
    world.compute(L, [world.BASE] + cfgs)
    df = build_table(L, cfgs, REP / "WINDOW_RESEARCH.csv", "10 trials плана; rolling/веса: LOYO по ближайшим парам; weekly/monthly/CUSUM: crit_wf")
    av = shap.availability(L["F"], L["grid"], L["dz0"])
    av.to_csv(OUT / "WINDOW_AVAILABILITY.csv", index=False)
    print(av.to_string(), df.shape)


def stage_shap(L) -> None:
    acc = accepted()
    cfgs = [world.BASE] + [c for c in feature_cfgs(acc) if c.add]
    best = sorted(cfgs[1:], key=lambda c: world.get(L, c)[1]["crit_loyo"])
    for k in (2, 3):   # комбинации — как собраны в stage_ablate (кэш прогнозов существует)
        if world._paths("b2_top%d" % k)[0].exists():
            cfgs.append(engine.Cfg("b2_top%d" % k, add=tuple(x for c in best[:k] for x in c.add)))
    out = shap.importance(L, cfgs, frame.FEAT)
    write(out, REP / "FEATURE_IMPORTANCE.csv", "диагностика (TreeSHAP, gain), не доказательство вклада; модель на первой дате сетки каждого года")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("control", "info", "ablate", "window", "shap", "all"), required=True)
    a = ap.parse_args()
    L = world.load()
    st = {"control": stage_control, "info": stage_info, "ablate": stage_ablate, "window": stage_window, "shap": stage_shap}
    for s in (list(st) if a.stage == "all" else [a.stage]):
        print("== stage", s)
        st[s](L)
    n = pd.read_csv(tab.TRIALS_B2).groupby("направление").size().to_dict() if tab.TRIALS_B2.exists() else {}
    (OUT / "CHECKPOINT_B2.json").write_text(json.dumps({"trials_used": n, "limits_first_pass": LIM}, ensure_ascii=False, indent=1))
    assert all(n.get(k, 0) <= v for k, v in LIM.items()), "превышен бюджет первого прохода B2"


if __name__ == "__main__":
    main()
