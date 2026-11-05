"""Прогнозы всех строк B1 на сетке 2271 дат: из потоков, правила H3 / R3 / R4, эксперты R2. -> {row_id: (fc, extra)}, фолды LOYO."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5 import common
from models_v2.s5.b1 import configs, designs, evalrows, rules, streams


def check_ctrl(w, fc: pd.DataFrame, ctrl: pd.DataFrame, col: str, tol: float = 1e-9) -> float:
    """Сетка и RV совпадают с Б-2 / HAR-J(б); σ̂² пересчёта к parquet: макс. относительная разность (> tol — стоп)."""
    assert (fc["d"].to_numpy("datetime64[ns]") == ctrl["d"].to_numpy("datetime64[ns]")).all(), "сетка дат"
    assert np.array_equal(fc["rv"].to_numpy(), ctrl["rv"].to_numpy()), "RV"
    ref = ctrl[col].to_numpy(float)
    rel = float(np.max(np.abs(fc["sigma2"].to_numpy() / ref - 1)))
    if rel > tol:
        raise SystemExit("стоп: пересчёт %s отличается от parquet: %.3e" % (col, rel))
    return rel


def pilot_b1(ctrl: pd.DataFrame) -> pd.DataFrame:
    c = pd.read_parquet(common.OUT / "fix_forecasts_c.parquet")
    return c[c.model == "fx_b1_g132_b"].reset_index(drop=True)


def stream_rows(w, specs, ctrl, procs) -> tuple:
    out, folds, raws = {}, {}, {}
    for name in configs.STREAMS_BLOCK:
        raws[name] = streams.raw_stream(w, specs, name, procs)
        folds[name] = streams.folds(w, specs, name, procs)
    for r in configs.ROWS:
        if r["kind"] == "stream":
            s = specs[r["stream"]]
            out[r["row_id"]] = streams.fc_stream(w, s, raws[r["stream"]], r["c"]) + (s.cs.index(r["c"]),)
    return out, folds, raws


def group_s2(w, ex, spec, s2_b2: np.ndarray) -> tuple:
    """σ̂² эксперта группы строки d; нет эксперта (пар < N_MIN) или группы — Б-2. -> (σ̂², доля резерва)."""
    g = spec.grp[ex["row"].to_numpy()]
    out, fb = s2_b2.copy(), np.ones(len(g), bool)
    for k in range(spec.ng):
        e = streams.expert_s2(w, spec, ex, k)
        use = (g == k) & np.isfinite(e)
        out[use], fb[use] = e[use], False
    return out, float(fb.mean())


def r2_rows(w, specs, ctrl, procs) -> dict:
    out = {}
    for rid, nm in (("reg_r2a", "reg_r2a"), ("reg_r2b", "reg_r2b")):
        ex = streams.raw_experts(w, specs, nm, procs)
        s2, fb = group_s2(w, ex, specs[nm], ctrl["s2_b2"].to_numpy(float))
        extra = {"fallback_share": fb}
        extra.update({"n_pairs_min_g%d" % k: int(ex["n%d" % k].min()) for k in range(specs[nm].ng)})
        out[rid] = (streams.frame_of(w, ex["row"].to_numpy(), s2), extra)
    return out


def r3_row(w, specs, ctrl, procs) -> tuple:
    sp = specs["reg_r2a"]
    ex = streams.raw_experts(w, specs, "reg_r2a", procs)
    b2 = ctrl["s2_b2"].to_numpy(float)
    lo, hi = streams.expert_s2(w, sp, ex, 0), streams.expert_s2(w, sp, ex, 1)
    lo, hi = np.where(np.isfinite(lo), lo, b2), np.where(np.isfinite(hi), hi, b2)
    X, qw = w.F.X, designs.regime_cols(w.F)["qw"]
    Z = np.column_stack([qw / 4, X[:, 0] - X[:, 2], X[:, 1] - X[:, 2]])
    p = rules.r3_prob(Z, np.where(np.isnan(sp.grp), np.nan, (sp.grp == 1).astype(float)), w.grid)
    s2 = np.where(np.isfinite(p), (1 - np.nan_to_num(p)) * lo + np.nan_to_num(p) * hi, b2)
    return streams.frame_of(w, w.grid, s2), {"fallback_share": float(np.isnan(p).mean()), "p_mean": float(np.nanmean(p))}


def h3_rows(w, specs, ctrl) -> dict:
    rv, h, b = (ctrl[c].to_numpy(float) for c in ("rv", "s2_harj", "s2_b2"))
    d, td = ctrl["d"].to_numpy("datetime64[ns]"), ctrl["target_date"].to_numpy("datetime64[ns]")
    grp = (designs.regime_cols(w.F)["qw"][w.grid] >= 4).astype(float)
    ws = {"har_h3_w05": np.full(len(rv), 0.5), "har_h3_exp": rules.h3_weights("exp", rv, h, b, d, td),
          "har_h3_roll252": rules.h3_weights("roll", rv, h, b, d, td), "har_h3_reg": rules.h3_weights("reg", rv, h, b, d, td, grp)}
    return {k: (streams.frame_of(w, w.grid, rules.mix(v, h, b)), {"w_mean": float(v.mean()), "w_last": float(v[-1])}) for k, v in ws.items()}


def r4_rows(w, ctrl, alt_h2: np.ndarray) -> dict:
    rv, h, b = (ctrl[c].to_numpy(float) for c in ("rv", "s2_harj", "s2_b2"))
    qw = designs.regime_cols(w.F)["qw"][w.grid]
    ty = ctrl["target_date"].dt.year.to_numpy()
    out = {}
    for rid, alt in (("reg_r4a", b), ("reg_r4b", alt_h2)):
        th = rules.theta_by_year(qw, ty, h, alt, rv)
        out[rid] = (streams.frame_of(w, w.grid, rules.switch(qw, ty, th, h, alt)), {"theta_" + str(y): v for y, v in th.items()})
    st = pd.read_parquet(common.OUT / "stress_p3.parquet").set_index("d")["stress"].reindex(ctrl["d"]).to_numpy(bool)
    out["reg_r4c"] = (streams.frame_of(w, w.grid, np.where(st, b, h)), {"stress_share": float(st.mean())})
    return out


def best_h2(folds: dict, rows: dict) -> str:
    """Лучшая H2-строка по Q_loyo среди кандидатов (до просмотра таблиц описания)."""
    sc = {}
    for rid in configs.H2_CAND:
        r = configs.BY_ID[rid]
        f = folds[r["stream"]]
        sc[rid] = evalrows.q_loyo(f[f.ci == rows[rid][2]])
    return min(sc, key=lambda k: sc[k])
