"""V6 лист Z2 (Ж-230), G60: реестр рядов сравнения SBER V1–V6 (h = 1) и данные. Только СОХРАНЁННЫЕ прогнозы (обучения нет): контроли и прежние ряды — `v6_0_load` (общие 2271 даты), конфигурации V6 — `v6_fin_data`.
Правила объявлены до счёта (docs/zadaniya/2026-11-09_harj_v6_Z2.md, п. «Правила»): (а) финалист, который программа сама объявила (источник — строкой в SPEC); (б) минимум QLIKE 2017–2021 среди PIT-сопоставимых гибридов пула программы
(без точных дублей контролей K1/K2/K3/K8/M и без A0g — диагностический контроль Ж-230); класс «потолок» в выбор лучшего не входит. Класс V6: «PIT-сопоставим» (18 конфигураций объявлены до счёта, параметры по закрытым парам),
но выбор лучшей после просмотра — потолок (колонка (б)); это толкование записано в отступлениях."""
from __future__ import annotations

import functools
import hashlib

import numpy as np
import pandas as pd

from models_v2.s11 import v4_io as IO4
from models_v2.s13 import v6_0_io as IO, v6_0_load as LD, v6_0_q as Q, v6_fin_data as FD

PIT, CEIL, CTL = "PIT-сопоставим", "потолок", "контроль"
CTRL = ["K1", "K2", "K3", "M", "K8"]
V6 = ["V6:" + c for c in FD.CFG if c != "A0g"]
Q_WIN = ("all", "2017_2021", "2022_2025")
REPORTS = "docs/otchety/"
# строка таблицы: имя, программа, N Холма программы (источник), финалисты (а) [(ряд, источник строкой)], пул (б): префиксы и исключения
SPEC = [
    dict(row="V1 (Ж-211)", prog="V1", n=521, n_src="hybrid_final_v1/FINAL_HYBRID_CONFIG.json N_final", pool=("J210a:", "J211", "J212"), excl=(),
         a=[("J210a:A1_roll_126", "hybrid_final_v1/FINAL_HYBRID_CONFIG.json finalists.H1; HYBRID_HAR_BOOSTING_FULL_RESEARCH_FINAL.md п. 1 (лучший по crit_wf)")]),
    dict(row="V1 (Ж-212)", prog="V1", n=541, n_src="hybrid_struct_v1/FINAL_STRUCT_CONFIG.json N_final", pool=("J210a:", "J211", "J212"), excl=(),
         a=[("J212B:B_xgb_Bf", "hybrid_struct_v1/FINAL_STRUCT_CONFIG.json decisions.har_features_min.best")]),
    dict(row="V2", prog="V2", n=585, n_src="harj_hybrid_v2/FINAL_V2_CONFIG.json N_holm", pool=("V2:",), excl=(),
         a=[("V2:E8", "harj_hybrid_v2/FINAL_V2_CONFIG.json top3[0] (ранг по шести правилам У31)")]),
    dict(row="V3", prog="V3", n=634, n_src="harj_panel_hybrid/CHECKPOINT.json N_declared_before_stats_p17_loto16", pool=("PANxgb:", "PANlgbm:", "PANk"), excl=(),
         a=[("PANlgbm:D2", "harj_panel_hybrid/FINAL_PANEL_CONFIG.json finalists.primary = lgbm_D2")]),
    dict(row="V4", prog="V4", n=694, n_src="harj_hybrid_v4/CHECKPOINT.json N_declared_before_stats", pool=("V4:",), excl=("V4:K8",),
         a=[("V4:I2", "HARJ_HYBRID_V4_FINAL.md п. 1(а): лучшая по crit_panel — I2")]),
    dict(row="V5", prog="V5", n=704, n_src="harj_hybrid_v5/CHECKPOINT.json N_declared_before_stats", pool=("V5:",), excl=(),
         a=[("V5:SP1", "HARJ_HYBRID_V5_FINAL.md п. 1: SP1 0,26018 — лучшая по crit_panel")]),
    dict(row="V6", prog="V6", n=722, n_src="harj_hybrid_v6/CHECKPOINT.json ключ 0 N", pool=("V6:",), excl=("V6:A0g",),
         a=[("V6:A2", "HARJ_HYBRID_SBER_V6_FINAL.md п. 1: лучшая по QLIKE 2017–2021 — A2"), ("V6:E2", "HARJ_HYBRID_SBER_V6_FINAL.md п. 1: лучшая по всей истории — E2")]),
]
DUP = ("K1", "K2", "K3", "K8", "M")          # точные дубли контролей в пул (б) не входят
CTRL_FILES ={"K1": ("H", "SBER"), "K2": ("K2", "SBER"), "K3": ("K2", "SBER")}


def sha_arr(x) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x, float)).tobytes()).hexdigest()


@functools.lru_cache(maxsize=1)
def prior() -> pd.DataFrame:
    return IO.read_csv("PRIOR_SBER.csv").set_index("series", drop=False)


@functools.lru_cache(maxsize=1)
def universe() -> dict:
    """-> dict(rv, tp, f): общие 2271 даты d; f: имя -> прогноз (контроли, все прежние ряды, 'V6:<cfg>')."""
    wide, _ = LD.load()
    d = FD.load()
    assert d["w"].index.equals(wide.index) and np.array_equal(d["w"]["rv"].to_numpy(), wide["rv"].to_numpy()), "V6 и лист 0: разные даты или цель"
    f = {c: wide[c].to_numpy(float) for c in wide.columns if c not in ("t_plus", "rv")}
    f.update({"V6:" + c: d["fs"][c] for c in FD.CFG})
    return dict(rv=wide["rv"].to_numpy(float), tp=wide["t_plus"].to_numpy(), f=f, dates=wide.index)


def klass(s: str) -> tuple:
    """-> (класс, причина) ряда: контроли — «контроль» (замороженные, оговорка K2/K3), V6 — толкование выше, прежние — PRIOR_SBER.csv листа 0."""
    if s in CTRL or s == "D2":
        return CTL, "замороженный контроль листа 0 (оговорка: дизайн K2/K3 выбирался на SBER 2017–2021)"
    if s.startswith("V6:"):
        return PIT, "18 конфигураций объявлены до счёта; параметры по закрытым парам; лучшая из них выбирается после просмотра (колонка (б)) — потолок"
    r = prior().loc[s]
    return r["class"], str(r["reason"])


def q_mean(u: dict, s: str, w: str) -> float:
    m = Q.wmask(u["tp"], w)
    return float(Q.ql(u["rv"][m], u["f"][s][m]).mean())


def pool_of(spec: dict, u: dict) -> list:
    out = []
    for s in u["f"]:
        if not s.startswith(spec["pool"]) or s in spec["excl"]:
            continue
        if not s.startswith("V6:") and (s not in prior().index or prior().loc[s]["equal_to"] in DUP):
            continue
        out.append(s)
    return out


def pick_b(spec: dict, u: dict) -> tuple:
    """-> (ряд (б) | None, ряд-потолок-минимум | None): (б) — минимум QLIKE 2017–2021 среди PIT пула; потолок — то же среди класса «потолок»."""
    pool = pool_of(spec, u)
    out = []
    for cls in (PIT, CEIL):
        c = [s for s in pool if klass(s)[0] == cls]
        out.append(min(c, key=lambda s: q_mean(u, s, "2017_2021")) if c else None)
    return tuple(out)


def src_file(s: str) -> tuple:
    """-> (путь от корня, sha256 файла) источника ряда."""
    if s in CTRL_FILES:
        p = IO4.path_of(*CTRL_FILES[s])
    elif s in ("M", "K8"):
        p = IO4.path_of("K2", "SBER")
    elif s.startswith("V6:"):
        p = IO.OUT / (s[3:] + ".parquet")
    else:
        p = IO.ROOT / prior().loc[s]["source"]
    return str(p.relative_to(IO.ROOT)), IO.sha_file(p)


def build_registry(u: dict) -> pd.DataFrame:
    rows = []
    def add(s, row, prog, role, src):
        c, why = klass(s)
        f, sha = src_file(s)
        rows.append(dict(series=s, row=row, program=prog, role=role, declared_by=src, file=f, sha256_file=sha, sha256_forecast=sha_arr(u["f"][s]), cls=c, reason=why, n_dates=int(np.isfinite(u["f"][s]).sum()),
                         first_d=str(u["dates"].min().date()), last_d=str(u["dates"].max().date())))
    for s in CTRL:
        add(s, "все", "общие", "контроль", "K1 = HAR-J PIT, K2 = XGBoost, K3 = LightGBM; M = √(K2·K3); K8 = √(K1·M)")
    for sp in SPEC:
        for s, src in sp["a"]:
            add(s, sp["row"], sp["prog"], "а", src)
        b, ceil = pick_b(sp, u)
        if b:
            add(b, sp["row"], sp["prog"], "б", "минимум QLIKE 2017–2021 среди PIT-сопоставимых гибридов пула; выбор после просмотра, потолок")
        if ceil:
            add(ceil, sp["row"], sp["prog"], "потолок_мин", "минимум QLIKE 2017–2021 среди рядов класса «потолок» пула; в выбор лучшего не входит")
    return pd.DataFrame(rows)


def candidates(reg: pd.DataFrame) -> list:
    """Ряды основного множества (K1, K2, K3, M, K8 + гибриды (а) и (б)): только класс PIT/контроль; потолки не входят."""
    ok = reg["cls"].isin([PIT, CTL]) & reg["role"].isin(["контроль", "а", "б"])
    return list(dict.fromkeys(reg.loc[ok, "series"]))
