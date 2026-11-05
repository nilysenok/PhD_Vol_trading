"""Ж-200/Ж-202: строки И1…И6. Запуск: python3 models_v2/run_fix.py --stage a|c [--imoex путь.parquet] | --check-untouched
Этап a — «И2 (24 точки, до Ж-202)»: I1-base, I1-free, I1+I2, I1+I2+I3(прокси). Этап c — сетка 132 (суффикс _g132), И3 с IMOEX (ISS).
Процессов <= 4."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from models_v2 import har, metrics, prov  # noqa: E402
from models_v2.s2 import features as ft  # noqa: E402
from models_v2.s3 import resid  # noqa: E402
from models_v2.s4 import design, diag, mkt, prov4, rows, select  # noqa: E402
from models_v2.s4.fit import HP  # noqa: E402

YEARS = tuple(range(2017, 2027))
OUT, TAB = prov4.OUT, prov4.TAB
GUARD = OUT / "fix_untouched.json"
IMOEX = prov.ROOT / "data/iss_check/index_10m/IMOEX.parquet"
G = "_g132"


def untouched() -> dict:
    fs = [p for p in (prov.ROOT / "models_v2").rglob("*.py") if "/s4/" not in str(p) and not p.name.startswith("run_")]
    fs += [OUT / ("forecasts_%s.parquet" % n) for n in ("p1", "p2", "pb")] + [OUT / "retrain_p2.parquet"]
    return {str(p.relative_to(prov.ROOT)): prov.sha_file(p) for p in sorted(fs)}


def check_untouched() -> bool:
    now = untouched()
    old = json.loads(GUARD.read_text())
    bad = [k for k in old if old[k] != now.get(k)]
    print("нетронутость:", "OK (%d файлов)" % len(old) if not bad else "ИЗМЕНЕНЫ %s" % bad)
    return not bad


def prepare(imoex_path):
    bars = pd.read_parquet(prov.BARS)
    F = ft.build(bars)
    pan = har.Panel(F.dates, F.seg, F.rv, F.X[:, :4])
    grid = har.eligible(pan)
    papers = mkt.load_papers(prov4.BARS_DIR, prov4.PAPERS, bars)
    imoex = pd.read_parquet(imoex_path) if imoex_path else None
    M, meta = mkt.block(None, F.X[:, [0, 1, 2]], F.seg, F.dates, papers, imoex)
    print("рынок:", meta)
    lb = resid.harj_ab(pan)[1]
    sets = {"base": None, "mkt": design.columns(M, "proxy")}
    if imoex is not None:
        sets["mkx"] = design.columns(M, "imoex")
    DZ = {"%s_%s" % (f, k): design.make(F, f, m, lb if f == "resid" else None) for f in design.FORMS for k, m in sets.items()}
    d, _ = diag.p2_params()
    return F, grid, DZ, HP(d, 50.0, 5.0, 0.03, 0.8), meta


def choose(F, DZ, dzk, hps, name, note):
    """Годовой LOYO-выбор из hps на конфигурации dzk (без весов) -> {год: HP}; таблица fix_<name>.csv."""
    sc, _ = select.run_loyo(F, DZ, {Y: {hp: (dzk, hp, False) for hp in hps} for Y in YEARS}, YEARS)
    ch, out = {}, []
    for Y in YEARS:
        s = sc[sc.Y == Y].set_index("label")
        ch[Y] = select.pick(s["score"].to_dict(), select.order_key)
        out += [{"Y": Y, "depth": h.depth, "min_data": h.min_child, "l2": h.l2, "lr": h.lr, "subsample": h.subsample,
                 "colsample": h.colsample, "max_trees": h.max_trees, "score": r.score, "n_folds": r.n_folds,
                 "chosen": h == ch[Y]} for h, r in s.iterrows()]
    prov4.write_csv(pd.DataFrame(out), TAB / ("fix_%s.csv" % name), "%s: LOYO y∈[2015,Y−1], цели < 01.01.Y; QLIKE(б); точек %d" % (note, len(hps)))
    return ch


def choose_i6(ctx, feat):
    combos = {"pure": [(f, k, w) for f in ("level", "m", "delta") for k in ("base", feat) for w in (False, True)],
              "resid": [("resid", k, w) for k in ("base", feat) for w in (False, True)]}
    cands = {Y: {(fam,) + c: ("%s_%s" % c[:2], ctx.i2[Y], c[2]) for fam, cs in combos.items() for c in cs} for Y in YEARS}
    sc, _ = select.run_loyo(ctx.F, ctx.DZ, cands, YEARS)
    picks, out = {"pure": {}, "resid": {}}, []
    for Y in YEARS:
        for fam in combos:
            s = sc[(sc.Y == Y) & (sc.label.map(lambda x: x[0]) == fam)].set_index("label")["score"].to_dict()
            best = select.pick(s, lambda lab: list(s).index(lab))
            picks[fam][Y] = ("%s_%s" % best[1:3], best[3])
            out += [{"Y": Y, "family": fam, "form": k[1], "feat": k[2], "w": k[3], "score": v, "chosen": k == best} for k, v in s.items()]
    prov4.write_csv(pd.DataFrame(out), TAB / "fix_i6_choice.csv", "И6: LOYO при гиперпараметрах года из И2 (сетка 132); признаки база/рынок(IMOEX+прокси)")
    return picks


def stage_rows(ctx, stage):
    if stage == "a":
        return [rows.Row("fx_i1_base", "old_mono", "level_base"), rows.Row("fx_i1_free", "old_free", "level_base"),
                rows.Row("fx_i12", "i2", "level_base"), rows.Row("fx_i123_proxy", "i2", "level_mkt")]
    ctx.hard = choose(ctx.F, ctx.DZ, "resid_mkx", select.HARD, "i2_choice_hard", "Б-1 жёсткий (глубина 1–2), признаки с IMOEX")
    pk = choose_i6(ctx, "mkx")
    R = [rows.Row("fx_i12" + G, "i2", "level_base"), rows.Row("fx_i123p" + G, "i2", "level_mkt"),
         rows.Row("fx_i123i" + G, "i2", "level_mkx"), rows.Row("fx_i1234w" + G, "i2", "level_mkx", True),
         rows.Row("fx_i5" + G, "i2", "delta_mkx"), rows.Row("fx_b1" + G, "i2", "resid_mkx"),
         rows.Row("fx_b2" + G, "i2", "m_mkx"), rows.Row("fx_b1hard" + G, "hard", "resid_mkx"),
         rows.Row("fx_i6_pure" + G, "i6", pick=pk["pure"]), rows.Row("fx_i6_resid" + G, "i6", pick=pk["resid"])]
    return R


def load_i2(name):
    t = pd.read_csv(TAB / ("fix_%s.csv" % name), comment="#")
    return {int(r.Y): HP(int(r.depth), float(r.min_data), float(r.l2), float(r.lr), float(r.subsample), float(r.colsample), int(r.max_trees))
            for r in t[t.chosen].itertuples()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("a", "c"))
    ap.add_argument("--imoex")
    ap.add_argument("--check-untouched", action="store_true")
    a = ap.parse_args()
    if a.check_untouched:
        sys.exit(0 if check_untouched() else 1)
    t0 = time.time()
    imoex = a.imoex or (str(IMOEX) if a.stage == "c" else None)
    F, grid, DZ, old, meta = prepare(imoex)
    if a.stage == "a":
        i2 = choose(F, DZ, "level_base", select.GRID, "i2_choice", "И2 (24 точки, до Ж-202)")
    else:
        i2 = choose(F, DZ, "level_base", select.GRID132, "i2_choice_g132", "И2 сетка 132 (Ж-202)")
    print("И2 выбрано, %.1f мин" % ((time.time() - t0) / 60), {y: (h.depth, h.min_child, h.l2, h.lr, h.subsample, h.colsample, h.max_trees) for y, h in i2.items()})
    ctx = rows.Ctx(F, grid, DZ, old, i2)
    R = stage_rows(ctx, a.stage)
    jobs = rows.plan(ctx, R)
    print("задач %d, %.1f мин" % (len(jobs), (time.time() - t0) / 60))
    res = rows.run_plan(ctx, jobs)
    p1 = pd.read_parquet(OUT / "forecasts_p1.parquet")
    rm = p1[p1.model == "rv_m"]
    q_m = metrics.summary(rm["rv"], rm["sigma2"])["qlike"]
    fc = [rows.forecasts(ctx, res, r) for r in R]
    lk = pd.DataFrame([rows.leak_row(ctx, res, r, f[f.model.str.endswith("_b")], q_m) for r, f in zip(R, fc)])
    imx = prov.sha_file(imoex)[:16] if imoex else ""
    pd.concat(fc).to_parquet(OUT / ("fix_forecasts_%s.parquet" % a.stage))
    prov4.write_csv(lk, TAB / ("fix_leak_%s.csv" % a.stage), "(б); monthly work/oracle/perm, full; %s" % meta, imx)
    prov4.write_csv(pd.concat([rows.overfit(ctx, res, r) for r in R]), TAB / ("fix_overfit_%s.csv" % a.stage),
                    "MSE ln: обучение (предв. модель) / блок остановки; ratio_mean — среднее отношений по переобучениям", imx)
    print(lk.to_string(), "\n", "%.1f мин" % ((time.time() - t0) / 60))
    check_untouched()


if __name__ == "__main__":
    main()
