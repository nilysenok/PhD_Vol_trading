"""P17, G16: 20 гибридов (A1…D2) с обучением на 17 (У-19…У-21). Формулы У-11 — прежние (`hp_hyb_*`); меняются только состав обучения (`hp_p17.train_exclude`) и выбор года (`hp_p17.sel_mask`, У-20):
κ у B, (глубина, c, λ_m, λ_u) у C, (a, b) у D1, w у D2 — по закрытым прогнозам года Y−1 по строкам 17; индивидуальные части (α/γ, Ḡ, β) — из прошлого самой бумаги; прогноз — для всех 28."""
from __future__ import annotations

import csv
import hashlib
import json
import time

import numpy as np
import pandas as pd

from models_v2.s10 import hp_engine as E, hp_hyb_a4 as A4, hp_hyb_b as B, hp_hyb_c as CC, hp_hyb_core as C, hp_hyb_d as D, hp_hyb_out as O, hp_hyb_run as HR, hp_p17 as X, hp_paths as P

PROGRAM = "Ж-220-p17"


def _years(panel: pd.DataFrame) -> tuple:
    yr = pd.DatetimeIndex(panel["date"]).year.to_numpy()
    return yr, sorted(set(yr[yr >= E.FIRST_YEAR]))


def forecast_b(panel: pd.DataFrame, f_a1: np.ndarray, kind: str, kappas=B.KAPPAS) -> tuple:
    """B1 / B2: как `hp_hyb_b.forecast_b`, κ — по 17."""
    ahat, n = B.alpha_b1(panel, f_a1) if kind == "b1" else (None, None)
    fk = {k: f_a1 * np.exp(B.adj_b1(ahat, n, k) if kind == "b1" else B.adj_b2(panel, f_a1, k)) for k in kappas}
    sel, (yr, years) = X.sel_mask(panel["ticker"].to_numpy()), _years(panel)
    out, ch = np.full(len(panel), np.nan), []
    for Y in years:
        s = B.select_kappa(panel[sel], {k: f[sel] for k, f in fk.items()}, Y)
        ch.append({"year": int(Y), **s})
        out[yr == Y] = fk[s["kappa"]][yr == Y]
    return out, pd.DataFrame(ch)


def finalize_c(panel, hm, hu, runs_m, runs_u, g_m_zero: bool = False, cs=E.CS, lams=E.LAMS) -> tuple:
    """C1 / C2: как `hp_hyb_c.finalize_c`, параметры года — по 17."""
    sel, (yr, years) = X.sel_mask(panel["ticker"].to_numpy()), _years(panel)
    sub = lambda r: None if r is None else {dp: df[sel] for dp, df in r.items()}
    out, ch = np.full(len(panel), np.nan), []
    for Y in years:
        s = CC.select_c(panel[sel], hm[sel], hu[sel], sub(runs_m), sub(runs_u), Y, cs, lams, g_m_zero)
        ch.append({"year": int(Y), **s})
        m = yr == Y
        gm = np.zeros(len(panel)) if g_m_zero else runs_m[s["depth"]]["g"].to_numpy()
        out[m] = CC.forecast_c(hm, hu, gm, runs_u[s["depth"]]["g"].to_numpy(), s["c"], s["lam_m"], s["lam_u"])[m]
    return out, pd.DataFrame(ch)


def finalize_d1(panel, k1, g, c, gb) -> tuple:
    """D1: как `hp_hyb_d.finalize_d1`, (a, b) — по 17."""
    sel, (yr, years) = X.sel_mask(panel["ticker"].to_numpy()), _years(panel)
    out, ch = np.full(len(panel), np.nan), []
    for Y in years:
        s = D.select_ab(panel[sel], k1[sel], g[sel], c[sel], gb[sel], Y)
        ch.append({"year": int(Y), **s})
        m = yr == Y
        out[m] = C.apply_lam(k1, g, c, D.lam_d1(s["a"], s["b"], gb))[m]
    return out, pd.DataFrame(ch)


def finalize_d2(panel, k1, g, c, Z, gn) -> tuple:
    """D2: как `hp_hyb_d.finalize_d2`; логистика w обучается только на закрытых парах 17 (У-20)."""
    sel, (yr, years) = X.sel_mask(panel["ticker"].to_numpy()), _years(panel)
    out, ch = np.full(len(panel), np.nan), []
    for Y in years:
        tr = D.train_mask(panel, Y) & np.isfinite(Z).all(axis=1) & np.isfinite(gn) & sel
        if tr.sum() == 0:
            lam, row = np.ones(len(panel)), dict(source="default", n_train=0)
        else:
            w = D.fit_w(Z[tr], (gn[tr] > 0).astype(int))
            lam = D.lam_d2(Z, w)
            row = dict(source="closed", n_train=int(tr.sum()), intercept=w["intercept"], **{"w_" + k: float(v) for k, v in zip(D.ZCOLS, w["coef"])})
        ch.append({"year": int(Y), **row})
        out[yr == Y] = C.apply_lam(k1, g, c, lam)[yr == Y]
    return out, pd.DataFrame(ch)


class Run17(HR.Run):
    """Гибриды на 17: те же конфигурации и порядок, что `hp_hyb_run.Run`; проходы — `hp_p17.cached_runs`, выбор — по 17."""

    def __init__(self, module: str, nthread: int, log, panel: pd.DataFrame | None = None, hp: dict | None = None):
        super().__init__(module, nthread, log, panel=pd.read_parquet(X.ENGINE_P17).reset_index(drop=True) if panel is None else panel, hp=hp)
        self.sha = P.sha16(X.ENGINE_P17) if X.ENGINE_P17.exists() and panel is None else "synthetic"

    def _runs(self, name, panel, feats, base, loss="qlike"):
        r = X.cached_runs(self.module, name, panel, feats, base, self.sha, loss, self.hp, E.DEPTHS, self.nthread, self.log)
        self.runs[name] = r
        return r

    def cfg_a(self, name: str) -> tuple:
        panel, feats = self.panel, {"A3": C.ZX}.get(name, C.Z28)
        if name == "A4":
            a4 = A4.build_cached()
            panel, feats = self.panel.merge(a4, on=C.KEY, how="left"), C.Z28 + A4.cols_of(a4)
        runs = self._runs(name, panel, feats, self.k1, HR.LOSS[name])
        fc, ch = X.finalize(panel, self.k1, runs, HR.LOSS[name])
        if name == "A1":
            self.a1 = {k: C.align(self.panel, fc, k) for k in ("g", "depth", "c", "lam")}
            self.a1["runs"] = runs
        return C.align(self.panel, fc, "sigma2"), ch, [name]

    def cfg_b(self, name: str) -> tuple:
        s, ch = forecast_b(self.panel, self.sig["A1"], name.lower())
        return s, ch, ["A1"]

    def cfg_c(self, name: str) -> tuple:
        parts = HR.parts_c(self.panel)
        m = self.panel[C.KEY].merge(parts, left_on=C.KEY, right_on=["ticker", "date"], how="left")
        hm, hu, b2 = m["hm"].to_numpy(float), m["hu"].to_numpy(float), m["b2"].to_numpy(float)
        tm, tu = CC.targets(self.panel, b2)
        r_u = self._runs("cu", self.panel.assign(rv_tplus=tu), C.Z28, hu)
        r_m = self._runs("cm", self.panel.assign(rv_tplus=tm), C.Z28, hm) if name == "C1" else None
        s, ch = finalize_c(self.panel, hm, hu, r_m, r_u, g_m_zero=name == "C2")
        return s, ch, ["cu"] + (["cm"] if name == "C1" else [])

    def cfg_d(self, name: str) -> tuple:
        a = self.a1
        gn, gb = self._gain()
        if name == "D1":
            s, ch = finalize_d1(self.panel, self.k1, a["g"], a["c"], gb)
            return s, ch, ["A1"]
        depths = sorted({int(d) for d in np.unique(a["depth"][np.isfinite(a["depth"])])})
        bags = [X.cached_runs(self.module, "bag%d" % sd, self.panel, C.Z28, self.k1, self.sha, "qlike", {**(self.hp or {}), "seed": sd}, tuple(depths), self.nthread, self.log) for sd in C.SEEDS_BAG]
        for i, r in enumerate(bags):
            self.runs["bag%d" % C.SEEDS_BAG[i]] = r
        gbags = [C.pick_by_depth(r, "g", a["depth"]) for r in bags]
        Z = D.zmatrix(self.panel, gb, D.dispersion(a["g"], gbags), D.sign_share(self.panel, a["g"]))
        s, ch = finalize_d2(self.panel, self.k1, a["g"], a["c"], Z, gn)
        return s, ch, ["A1"] + ["bag%d" % x for x in C.SEEDS_BAG]

    def write(self) -> None:
        k = self.panel[HR.OUT_COLS].copy()
        for n, s in self.sig.items():
            k["sigma2_" + n] = s
        if self.a1:
            for key in ("g", "depth", "c", "lam"):
                k["a1_" + key] = self.a1[key]
        k[pd.DatetimeIndex(k["date"]).year >= E.FIRST_YEAR].reset_index(drop=True).to_parquet(P.OUT / ("hyb_%s_p17.parquet" % self.module))

    def one(self, name: str) -> None:
        """Счёт конфигурации, запись `hyb_<модуль>_p17.parquet`, строка реестра `Ж-220-p17:*` (used не меняется, У-21)."""
        t0 = time.time()
        used = self.calc(name)
        s = self.sig[name]
        fit = sum(float(df.groupby("retrain")["fit_sec"].first().sum()) for u in used for df in self.runs[u].values())
        self.time[name] = dict(config=name, wall_sec=round(time.time() - t0, 1), fit_sec_runs=round(fit, 1), runs=",".join(used))
        self.write()
        added = register(self.module, name, HR.ARCH[name], {"loss": HR.LOSS.get(name, "qlike"), "years": "2016-2026", "select": "closed Y-1 по 17 (У-20)", "n_rows": int(np.isfinite(s).sum())})
        self.log("%s %s p17: строк %d, строка реестра %s, %.0f с" % (self.module, name, np.isfinite(s).sum(), "добавлена" if added else "уже была", time.time() - t0))


def register(module: str, name: str, arch: str, config: dict) -> bool:
    """Строка реестра `Ж-220-p17:<модуль>:<имя>:<хэш>` (пересчёт объявленной конфигурации на 17, У-21); повтор не дублируется; `used` в CHECKPOINT не меняется."""
    with O.lock():
        pre = "%s:%s:%s:" % (PROGRAM, module, name)
        rows = O.REG.read_text(encoding="utf-8").splitlines()
        if any(r.startswith(pre) for r in rows):
            return False
        cfg = {"name": name, "module": module, "arch": arch, "kind": "panel_hybrid_p17", "trial": True, "recompute_of": "Ж-217:%s:%s" % (module, name), **config}
        js = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
        with open(O.REG, "a", newline="", encoding="utf-8") as f:
            if not O.REG.read_bytes().endswith(b"\n"):
                f.write("\n")
            csv.writer(f, lineterminator="\n").writerow([pre + hashlib.sha256(js.encode()).hexdigest()[:12], PROGRAM, "harj_panel_hybrid", arch, js, "trial"])
        return True
