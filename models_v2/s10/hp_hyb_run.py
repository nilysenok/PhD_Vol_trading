"""P2, G9/G10: стадия `hyb` — 10 конфигураций гибрида (A1–A4, B1–B2, C1–C2, D1–D2) одного модуля (xgb | lgbm) на движке P1b. Контракт — GATES_PANEL R6–R8, У-6…У-11.
Порядок: A1, B1, C1, D1, A2, A3, A4, B2, C2, D2. После каждой конфигурации: запись `hyb_<module>.parquet`, строка реестра и used. Проходы walk-forward кэшируются (`hp_hyb_core.cached_runs`).
Выбор финалистов, статистика, LOTO — не здесь (лист P3)."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from models_v2.s10 import hp_ctrl as CT, hp_engine as E, hp_hyb_a4 as A4, hp_hyb_b as B, hp_hyb_c as CC, hp_hyb_core as C, hp_hyb_d as D, hp_hyb_out as O, hp_paths as P

ORDER = ["A1", "B1", "C1", "D1", "A2", "A3", "A4", "B2", "C2", "D2"]
ARCH = {"A1": "panel_global", "A2": "panel_global_l2log", "A3": "panel_global_X_only", "A4": "panel_global_plus_K2_features", "B1": "global_individual_alpha", "B2": "global_individual_alpha_gamma",
        "C1": "market_idio", "C2": "market_idio_gm0", "D1": "confidence_gated", "D2": "confidence_gated_reliability_index"}
LOSS = {"A1": "qlike", "A2": "l2log", "A3": "qlike", "A4": "qlike"}
OUT_COLS = ["ticker", "date", "t_plus", "group", "gap", "rv_tplus", "sigma2_k1"]


def parts_c(panel: pd.DataFrame) -> pd.DataFrame:
    """(β², Ĥ^m, Ĥ^u) бумаг на датах панели (кэш `hyb_c_parts.parquet`)."""
    path = P.OUT / "hyb_c_parts.parquet"
    if path.exists():
        return pd.read_parquet(path)
    imo = pd.read_parquet(P.OUT / "daily_IMOEX.parquet")
    k1_imo = CT.k1_ext({"IMOEX": imo})["IMOEX"]
    out = []
    for t, g in panel.groupby("ticker"):
        d = pd.read_parquet(P.OUT / ("daily_%s.parquet" % t))
        out.append(CC.k7_parts(d, imo, g["date"].to_numpy(), k1_imo).assign(ticker=t))
    df = pd.concat(out, ignore_index=True)
    df.to_parquet(path)
    return df


class Run:
    def __init__(self, module: str, nthread: int, log, panel: pd.DataFrame | None = None, hp: dict | None = None):
        """panel / hp заданы только в тестах (синтетика, малые HP); боевой счёт — панель движка P1b после сверки хэшей входов."""
        self.module, self.nthread, self.log, self.hp = module, nthread, log, hp
        if panel is None:
            bad = O.check_hashes()
            if bad:
                raise SystemExit("стоп: входы не совпали с P0/P1b: %s" % bad)
            panel = pd.read_parquet(CT.ENGINE_PANEL).reset_index(drop=True)
        self.sha = P.sha16(CT.ENGINE_PANEL) if (P.OUT / "engine_panel.parquet").exists() else "synthetic"
        self.panel = panel
        self.k1 = self.panel["sigma2_k1"].to_numpy(float)
        self.sig, self.choice, self.time, self.a1, self.runs = {}, {}, {}, None, {}

    def _runs(self, name, panel, feats, base, loss="qlike"):
        r = C.cached_runs(self.module, name, panel, feats, base, self.sha, loss, self.hp, E.DEPTHS, self.nthread, self.log)
        self.runs[name] = r
        return r

    def cfg_a(self, name: str) -> tuple:
        panel, feats = self.panel, {"A3": C.ZX}.get(name, C.Z28)
        if name == "A4":
            a4 = A4.build_cached()
            panel, feats = self.panel.merge(a4, on=C.KEY, how="left"), C.Z28 + A4.cols_of(a4)
        runs = self._runs(name, panel, feats, self.k1, LOSS[name])
        fc, ch = E.finalize(panel, self.k1, runs, LOSS[name])
        if name == "A1":
            self.a1 = {k: C.align(self.panel, fc, k) for k in ("g", "depth", "c", "lam")}
            self.a1["runs"] = runs
        return C.align(self.panel, fc, "sigma2"), ch, [name]

    def cfg_b(self, name: str) -> tuple:
        s, ch = B.forecast_b(self.panel, self.sig["A1"], name.lower())
        return s, ch, ["A1"]

    def cfg_c(self, name: str) -> tuple:
        parts = parts_c(self.panel)
        m = self.panel[C.KEY].merge(parts, left_on=C.KEY, right_on=["ticker", "date"], how="left")
        hm, hu, b2 = m["hm"].to_numpy(float), m["hu"].to_numpy(float), m["b2"].to_numpy(float)
        tm, tu = CC.targets(self.panel, b2)
        r_u = self._runs("cu", self.panel.assign(rv_tplus=tu), C.Z28, hu)
        r_m = self._runs("cm", self.panel.assign(rv_tplus=tm), C.Z28, hm) if name == "C1" else None
        s, ch = CC.finalize_c(self.panel, hm, hu, r_m, r_u, g_m_zero=name == "C2")
        return s, ch, ["cu"] + (["cm"] if name == "C1" else [])

    def _gain(self):
        a = self.a1
        gn = D.gain_a1(self.panel, self.k1, a["g"], a["c"])
        return gn, D.gbar(self.panel, gn)

    def cfg_d(self, name: str) -> tuple:
        a = self.a1
        gn, gb = self._gain()
        if name == "D1":
            s, ch = D.finalize_d1(self.panel, self.k1, a["g"], a["c"], gb)
            return s, ch, ["A1"]
        depths = sorted({int(d) for d in np.unique(a["depth"][np.isfinite(a["depth"])])})
        bags = D.bag_runs(self.module, self.panel, C.Z28, self.k1, self.sha, depths, self.nthread, self.log, self.hp)
        for i, r in enumerate(bags):
            self.runs["bag%d" % C.SEEDS_BAG[i]] = r
        gbags = [C.pick_by_depth(r, "g", a["depth"]) for r in bags]
        Z = D.zmatrix(self.panel, gb, D.dispersion(a["g"], gbags), D.sign_share(self.panel, a["g"]))
        s, ch = D.finalize_d2(self.panel, self.k1, a["g"], a["c"], Z, gn)
        return s, ch, ["A1"] + ["bag%d" % x for x in C.SEEDS_BAG]

    def calc(self, name: str) -> list:
        """Счёт конфигурации (проходы — через кэш); результат в self.sig / self.choice; -> имена проходов, на которые она опирается."""
        s, ch, used = {"A": self.cfg_a, "B": self.cfg_b, "C": self.cfg_c, "D": self.cfg_d}[name[0]](name)
        self.sig[name], self.choice[name] = s, ch
        return used

    def one(self, name: str) -> None:
        t0 = time.time()
        used = self.calc(name)
        s = self.sig[name]
        fit = sum(float(df.groupby("retrain")["fit_sec"].first().sum()) for u in used for df in self.runs[u].values())
        self.time[name] = dict(config=name, wall_sec=round(time.time() - t0, 1), fit_sec_runs=round(fit, 1), runs=",".join(used))
        self.write()
        added = O.register(self.module, name, ARCH[name], {"loss": LOSS.get(name, "qlike"), "years": "2016-2026", "select": "closed Y-1 (У-7)", "n_rows": int(np.isfinite(s).sum())})
        self.log("%s %s: строк %d, строка реестра %s, %.0f с" % (self.module, name, np.isfinite(s).sum(), "добавлена" if added else "уже была", time.time() - t0))

    def write(self) -> None:
        k = self.panel[OUT_COLS].copy()
        for n, s in self.sig.items():
            k["sigma2_" + n] = s
        if self.a1:
            for key in ("g", "depth", "c", "lam"):
                k["a1_" + key] = self.a1[key]
        k[pd.DatetimeIndex(k["date"]).year >= E.FIRST_YEAR].reset_index(drop=True).to_parquet(P.OUT / ("hyb_%s.parquet" % self.module))

    def tables(self) -> None:
        ch = pd.concat([c.assign(config=n) for n, c in self.choice.items()], ignore_index=True)
        O.merge_csv(ch, "P2_CHOICE.csv", "выбор параметров по годам (У-7: закрытые прогнозы Y−1); пусто в столбце = параметр не применяется к конфигурации", self.module)
        O.merge_csv(pd.DataFrame(self.time.values()), "P2_TIME.csv", "время счёта конфигураций, с (fit_sec_runs — чистое обучение проходов, на которые опирается конфигурация; общие проходы A1/cu не пересчитываются)", self.module)
        k = pd.read_parquet(P.OUT / ("hyb_%s.parquet" % self.module)).merge(pd.read_parquet(P.OUT / "k4567.parquet")[C.KEY + ["sigma2_k4", "sigma2_k5", "sigma2_k6", "sigma2_k7"]], on=C.KEY, how="left")
        ctl = "k4" if self.module == "xgb" else "k5"
        t = O.describe(k, ["k1", ctl, "k6", "k7"] + list(self.sig))
        P.write_csv(t, "P2_%s.csv" % self.module.upper(), "%s: метрики гибридов и контролей K1, %s, K6, K7 на общих датах бумаги, цели с 2017-01-03; группы — среднее по бумагам; ОПИСАТЕЛЬНО, без выбора (R9 — лист P3)" % (self.module, ctl.upper()))


def compute(module: str, nthread: int, log=print) -> None:
    r = Run(module, nthread, log)
    for name in ORDER:
        r.one(name)
    r.tables()


def check(module: str) -> list:
    """Ошибки выхода P2 модуля (пусто = ок): входы, 10 колонок, положительность, λ = 0 ≡ K1 побитно, реестр и used."""
    bad = O.check_hashes()
    k = pd.read_parquet(P.OUT / ("hyb_%s.parquet" % module))
    bad += ["дубли (бумага, дата)"] if k.duplicated(C.KEY).any() else []
    for n in ORDER:
        v = k.get("sigma2_" + n)
        ok = v is not None and v.notna().sum() > 0 and bool((v.dropna() > 0).all() and np.isfinite(v.dropna()).all())
        print("%s: значений %d" % (n, 0 if v is None else v.notna().sum()))
        bad += [] if ok else ["%s: нет значений или не конечны / ≤ 0" % n]
    z = k[k["a1_lam"] == 0]
    print("A1 при λ = 0: строк %d, побитно K1: %s" % (len(z), bool((z["sigma2_A1"] == z["sigma2_k1"]).all())))
    bad += [] if (z["sigma2_A1"] == z["sigma2_k1"]).all() else ["A1 при λ = 0 не равна K1 побитно"]
    reg = [r for r in O.REG.read_text(encoding="utf-8").splitlines() if r.startswith("%s:%s:" % (O.PROGRAM, module))]
    used = json.loads(O.CP.read_text())["used"]
    print("строк реестра модуля %d, used %d" % (len(reg), used))
    bad += [] if len(reg) == 10 and used >= (10 if module == "xgb" else 20) else ["реестр/used: %d строк модуля, used %d" % (len(reg), used)]
    return bad + [f for f in ("P2_CHOICE.csv", "P2_TIME.csv", "P2_%s.csv" % module.upper()) if not (P.REP / f).exists()]


def stage(module: str, check_only: bool, nthread: int = 5) -> int:
    if not check_only:
        compute(module, nthread, lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
    bad = check(module)
    print("\n".join(bad))
    print("ИТОГ:", "OK" if not bad else "FAIL")
    return 1 if bad else 0
