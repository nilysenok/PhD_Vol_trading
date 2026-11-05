"""Волна C (Ж-204): python3 models_v2/s5/c/run_c.py --stage control|tpe|c1|c2|c3|robust|report|all   (пул <= 4; seed 42)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models_v2 import metrics  # noqa: E402
from models_v2.s5 import common  # noqa: E402
from models_v2.s5.b2 import world as b2w  # noqa: E402
from models_v2.s5.c import ctx_c, engine_c, mix, robust, space, stages, tab_c, tpe  # noqa: E402

HEAD = "# разработка; волна C Ж-204; до аудита; оценка 2017–02.2026 (2022+ — просмотр контроля гл. 3 по SBER, не независимый OOS); код c sha=%s; %s\n"


def write(df, path, note):
    with open(path, "w", encoding="utf-8") as f:
        f.write(HEAD % (ctx_c.code_sha(), note))
        df.to_csv(f, index=False, lineterminator="\n")


def stage_control(C, p):
    ctrl = C["ctrl"]
    qb, qh = (float(metrics.qlike(ctrl["rv"], ctrl[c]).mean()) for c in ("s2_b2", "s2_harj"))
    assert (ctrl["rv"] > 0).all() and len(ctrl) == common.N_GRID
    ok = abs(qb - 0.220486502764) < 1e-12 and abs(qh - 0.233662115694) < 1e-12
    res = {}
    for name in ("f4", "top2"):
        fc = stages.daily(C, p, name, C["i2"])[0]
        ref = pd.read_parquet(b2w.OUT / ("fc_b2_%s.parquet" % ("f4" if name == "f4" else "top2")))
        res[name] = bool(np.array_equal(fc["sigma2"].to_numpy(), ref["sigma2"].to_numpy()))
    print("G1: Б-2 %.12f HAR-J(б) %.12f; побитно к кэшу B2: %s" % (qb, qh, res))
    ctx_c.save_json("control.json", {"b2": qb, "harj": qh, "bitwise": res})
    if not ok or not all(res.values()):
        sys.exit("G1 КРАСНЫЙ: контроль не воспроизведён — стоп и доклад")


def stage_tpe(C, p):
    i2, F = C["i2"], C["w"].F
    done = tpe.run_study(p, "f4", ctx_c.DATA / "tpe_cache.json")
    sc2 = ctx_c.load_json("i2_f4_scores.json", None) or engine_c.scores(p, "f4", i2)
    ctx_c.save_json("i2_f4_scores.json", sc2)
    sc2 = {int(k): v for k, v in sc2.items()}
    cand = {d["k"]: (tpe.hp_of(d["params"]), {int(Y): v for Y, v in d["scores"].items()}) for d in done}
    ch = {}
    for Y in space.ALL_YEARS:
        hp, src, score = tpe.choose(cand, {y: (i2[y], sc2[y]) for y in space.ALL_YEARS}, Y)
        ch[str(Y)] = {"hp": tpe.hp_json(hp), "src": src, "score": score}
    ctx_c.save_json("chosen.json", ch)
    rec = [dict(k=d["k"], J=d["J"], **d["params"], **{"score_%s" % Y: v for Y, v in d["scores"].items()}) for d in done]
    write(pd.DataFrame(rec), common.REP / "C_TPE.csv", "22 trials Optuna TPE seed 42; J — среднее score_Y по Y 2017–2021")
    write(pd.DataFrame([dict(Y=Y, src=v["src"], score=v["score"], **v["hp"]) for Y, v in ch.items()]), common.REP / "C_CHOICE.csv", "HP_Y по score_Y (цели < 01.01.Y)")
    print(pd.DataFrame(rec)[["k", "J"]].round(5).to_string(), {Y: v["src"] for Y, v in ch.items()})


def crit_booster(C, p, b):
    f = ctx_c.DATA / ("crit_%s.json" % b)
    if f.exists():
        return ctx_c.load_json(f.name, 0.0)
    hp = stages.chosen_hp() if b == "opt" else C["i2"]
    c = engine_c.crit(engine_c.scores(p, "f4" if b == "opt" else b, hp))
    ctx_c.save_json(f.name, c)
    return c


def stage_robust(C, p):
    df = pd.DataFrame(ctx_c.load_json("rows.json", {}).values())
    top = tab_c.top3(df)
    if top.empty:
        write(pd.DataFrame(), common.REP / "C_SEEDS.csv", "нет кандидатов perspective")
        write(pd.DataFrame(), common.REP / "C_PLATEAU.csv", "нет кандидатов perspective")
        return
    ids = list(top["row_id"])
    ctx = mix.context(C["w"].F, C["w"].grid, C["ctrl"])
    sd = robust.seed_table(p, C["w"], ids, ctx, C["ctrl"], C["i2"], stages.chosen_hp(), lambda p_, w_, n, hp, s: stages.daily(C, p_, n, hp, s))
    bs = sorted({b for r in ids for b in robust.needs(r)})
    crit0 = {b: crit_booster(C, p, b) for b in bs}
    pl = robust.plateau_table(p, C["w"], bs, C["i2"], stages.chosen_hp(), crit0, stages.crit_b2())
    write(sd, common.REP / "C_SEEDS.csv", "5 seeds 42–46 для лучших 3 строк C")
    write(pl, common.REP / "C_PLATEAU.csv", "соседи ±1 шаг, внутренний критерий")
    print(top[["row_id", "rel_gain", "d_harj_no_240222"]], sd.groupby("row_id")[["n_ok", "seed_ok"]].first(), pl.groupby("booster")[["share_better_b2", "plateau_ok"]].first())


def stage_report(C, p):
    rows = ctx_c.load_json("rows.json", {})
    df = pd.DataFrame(rows.values())
    df = tab_c.holm_cols(tab_c.holm_cols(df), tab_c.FAMILY_C, "_ref40")
    write(df, common.REP / "C_ALL.csv", "Холм: N = %d (все trials программы), справочно N = %d; 2017–2021 строк на c1_opt — потолок" % (tab_c.FAMILY, tab_c.FAMILY_C))
    for fn, dirn in (("HAR_BOOSTING_RESULTS.csv", stages.HAR), ("REGIME_MODEL_RESULTS.csv", stages.REG)):
        tab_c.append_table(common.REP / fn, df[df["direction"] == dirn], "row_id")
    counts = tab_c.merge_registry(list(ctx_c.load_json("trials_rows.json", {}).values()))
    tab_c.write_checkpoint(counts)
    print(counts, len(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("control", "tpe", "c1", "c2", "c3", "robust", "report", "all"), required=True)
    a = ap.parse_args()
    C = ctx_c.load()
    seq = {"control": stage_control, "tpe": stage_tpe, "c1": stages.stage_c1, "c2": stages.stage_c2, "c3": stages.stage_c3, "robust": stage_robust,
           "report": stage_report}
    for s in (list(seq) if a.stage == "all" else [a.stage]):
        print("== stage", s, flush=True)
        if s == "report":
            seq[s](C, None)
            continue
        with engine_c.pool(C["w"]) as p:
            seq[s](C, p)
        if s == "c2":
            stages.stage_h2l4(C)


if __name__ == "__main__":
    main()
