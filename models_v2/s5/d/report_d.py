"""D3: реестр и чекпойнт, FINAL_RANKING, накопленные разности (CSV + PNG), MODEL_SPECIFICATIONS.json. Тексты (FINAL.md, README, REPRODUCTION) — doc_d."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models_v2 import metrics  # noqa: E402
from models_v2.s4 import fit  # noqa: E402
from models_v2.s5 import common  # noqa: E402
from models_v2.s5.b2 import world as b2w  # noqa: E402
from models_v2.s5.c import ctx_c, space  # noqa: E402
from models_v2.s5.c import stages as cstages  # noqa: E402
from models_v2.s5.d import ctx_d, d2_d, doc_d, ens, robust_d, stages_d, tab_d  # noqa: E402

REP = common.REP


def ranking(fin: pd.DataFrame, top: dict, st: pd.DataFrame) -> pd.DataFrame:
    """Строки финалистов (и справочной) с статусом по правилу D3; p Холма (N = 106) к Б-2 и HAR-J(б) — DM весь период и бутстрап."""
    def p(r, t, per, base):
        m = st[(st.test == t) & (st.row_id == r) & (st.period == per) & (st.base == base)]
        return float(m["p_holm_N106"].iloc[0]) if len(m) else np.nan
    f = fin.copy()
    f["rel_gain"] = (f["q_int_b2"] - f["q_int"]) / f["q_int_b2"]
    tp = top["top3"]
    def status(r, e):
        if r in tp:
            return "тройка №%d" % (tp.index(r) + 1)
        if e:
            return "отдельно: выбор на оценочном периоде (потолок)"
        return ("не допущен: " + ", ".join(top["excluded"][r])) if top["excluded"].get(r) else "допущен по флагам, вне тройки (порядок D3)"
    f["status"] = [status(r, e) for r, e in zip(f["row_id"], f["eval_selected"])]
    for lab, t, per, base in (("dm_b2", "DM", "весь период", "Б-2"), ("dm_harj", "DM", "весь период", "HAR-J(б)"), ("boot_b2", "boot", "весь период", "Б-2"),
                              ("boot_harj", "boot", "весь период", "HAR-J(б)")):
        f["p_holm_N106_" + lab] = [p(r, t, per, base) for r in f["row_id"]]
    mc = st[(st.test == "MCS") & (st.period == "весь период")].set_index("row_id")["p"]
    f["p_mcs"] = [float(mc.get(r, np.nan)) for r in f["row_id"]]
    return f.sort_values("rel_gain", ascending=False).reset_index(drop=True)


def cum_diff(sig: dict, rv, td, rids) -> pd.DataFrame:
    out = {"target_date": pd.Series(td).dt.strftime("%Y-%m-%d").to_numpy()}
    for bk in ("HARJ", "B2"):
        for r in rids:
            out["cum_%s_vs_%s" % (r, bk)] = np.cumsum(metrics.qlike(rv, sig[r]) - metrics.qlike(rv, sig[bk]))
    return pd.DataFrame(out)


def plot_cum(cd: pd.DataFrame, rids, bk: str, path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    t = pd.to_datetime(cd["target_date"])
    for r in rids:
        ax.plot(t, cd["cum_%s_vs_%s" % (r, bk)], lw=1.1, label=r)
    ax.axhline(0, color="k", lw=0.6)
    for y in range(2018, 2027):
        ax.axvline(pd.Timestamp("%d-01-01" % y), color="grey", lw=0.4, ls=":")
    ax.set_title(title)
    ax.set_ylabel("накопленная Σ(QLIKE конфиг − QLIKE база); вниз — конфиг лучше")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def booster_spec(key: str, C: dict) -> dict:
    nm, hk, lib, win, hl = stages_d.SPEC_M[key]
    hp = C["i2"] if hk == "i2" else cstages.chosen_hp()
    return {"design": nm, "added_columns": list(C["cols"][nm]), "library": lib, "window_pairs": win or "expanding", "half_life_days": hl or None, "seed": 42,
            "hp_by_year_of_forecast_date": {str(Y): space.hp_dict(h) for Y, h in hp.items()}, "hp_source": "И2 по годам (цели < 01.01.Y)" if hk == "i2" else "C1: TPE+И2 по годам (C_CHOICE.csv)"}


def spec_of(rid: str, best: str, C: dict) -> dict:
    keys = robust_d.needs(rid, best)
    d = {"row_id": rid, "target": "y = ln RV_{t+1} − ln RV_m (RV_m — средняя за 22 дня); σ̂² = exp(ln RV_m + f + s²_es/2), s²_es — средний квадрат остатка на блоке ранней остановки",
         "boosters": {k: booster_spec(k, C) for k in keys}, "fit": {"early_stopping_block_pairs": fit.ES_N, "embargo": fit.EMBARGO, "patience": fit.PATIENCE, "max_trees_cap": fit.MAX_TREES,
         "refit": "каждый день сетки; пары с t+1 <= d; признаки на d — данные <= 18:50"}}
    if rid == best:
        d["ensemble"] = {"members": ens.SPEC[rid][0], "rule": ens.SPEC[rid][1], "weights": "симплекс шаг 0,1; потери по закрытым прогнозам (target_date <= d); до 252 закрытых — равные"}
    elif rid.startswith("c2_h3") or rid.startswith("c3_"):
        d["rule"] = "см. models_v2/s5/c/mix.py::build(%r); веса H3 — models_v2/s5/b1/rules.py::h3_weights" % rid
    return d


def features_text() -> dict:
    gl = {ln.split("**")[1].split(" ")[0]: ln for ln in (REP / "GATES_B2.md").read_text(encoding="utf-8").splitlines() if ln.startswith("**F")}
    man = json.loads((REP / "BASELINE_MANIFEST.json").read_text(encoding="utf-8"))
    return {"base_b2": man["b2"]["features"], "group_definitions": {k: v for k, v in gl.items() if k in ("F3", "F4")}}


def specs(C, best, top) -> dict:
    sha = {"code_c": ctx_c.code_sha(), "code_d": ctx_d.code_sha(), "code_b2": b2w.code_sha(), "world_key_b2": C["L"]["key"],
           "fix_forecasts_c.parquet": common.sha_file(common.OUT / "fix_forecasts_c.parquet"), "forecasts_p1.parquet": common.sha_file(common.OUT / "forecasts_p1.parquet"),
           "vcs": "рабочая копия без коммита результатов D; коммитит дирижёр"}
    return {"status": "разработка, до аудита; оценка 2017–02.2026, не независимый OOS", "features": features_text(), "sha": sha,
            "variants": [spec_of(r, best, C) for r in top["top3"]], "note_if_empty": "тройка пуста: Б-2 остаётся контролем" if not top["top3"] else ""}


def run(C) -> None:
    best = ctx_d.load_json("best.json", {})["best"]
    top, flags = ctx_d.load_json("top3.json", {}), ctx_d.load_json("flags.json", {})
    counts = tab_d.merge_registry(list(ctx_d.load_json("trials_rows.json", {}).values()))
    tab_d.write_checkpoint(counts)
    fin = pd.read_csv(ctx_d.DATA / "fin_table.csv", comment="#")
    st = pd.read_csv(REP / "STATISTICAL_TESTS.csv", comment="#")
    rk = ranking(fin, top, st)
    ctx_d.write(rk, REP / "FINAL_RANKING.csv", "статус по правилу D3 (GATES_D §4): perspective, seed-ok, plateau-ok, MCS (p >= 0,10); строки с выбором на оценочном периоде — отдельно")
    ctx, sig = d2_d.sigmas(C, best)
    rids = robust_d.finalists(best)
    cd = cum_diff(sig, ctx["rv"], C["ctrl"]["target_date"], rids)
    ctx_d.write(cd, REP / "D_CUMDIFF.csv", "накопленная разность ежедневных QLIKE-потерь (конфиг − база) по датам цели")
    for bk, nm in (("HARJ", "HAR-J(б)"), ("B2", "Б-2")):
        plot_cum(cd, rids, bk, REP / ("D_cum_vs_%s.png" % bk.lower()), "SBER h=1: накопленная разность QLIKE к %s (финалисты D)" % nm)
    (REP / "MODEL_SPECIFICATIONS.json").write_text(json.dumps(specs(C, best, top), ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    doc_d.write_all(C, best, top, flags, rk, st, sig, ctx, counts)
    print("G10:", counts, "| тройка:", top["top3"])
