"""P3, G11: данные для статистики и выбор финалистов / тройки (R9, У-12; Ж-219). Выбор — ПЕРВЫМ, до любого p; p-значений здесь нет.
crit_panel = среднее по бумагам M16 `crit_wf` (d_comp.crit_wf) на ОБЩИХ датах (цель t⁺ в 2017–2021) всех 20 конфигураций и K1, K4, K5, K6, K7. Статус: DEVELOPMENT, потолок."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from models_v2.s6.xgb.d_comp import crit_wf
from models_v2.s10 import hp_paths as P

KEY = ["ticker", "date"]
NAMES = ["A1", "A2", "A3", "A4", "B1", "B2", "C1", "C2", "D1", "D2"]
POS = {"A": ["A1", "A2", "A3", "A4"], "B": ["B1", "B2"], "C": ["C1", "C2"], "D": ["D1", "D2"]}
CTRL = ["k1", "k4", "k5", "k6", "k7"]
CRIT_YEARS = (2017, 2021)      # окно выбора (R9); подмена в мутации — выбор по 2022+
MARGIN = 0.0005                # R9 / У-12
STAT = "исследовательский, HISTORICAL WALK-FORWARD, не независимый OOS; выбор — внутренняя валидация M16 × 2017–2021, цифры — потолок"
STATS_DIR = P.REP / "statistics"
HASHES = {"hyb_xgb.parquet": "c864e31cfe764676", "hyb_lgbm.parquet": "c5d7cbcb3a4c4948"}       # входы P3 (задание); engine/k4567/panel_base — hp_hyb_out.HASHES


def model_cols() -> list:
    return CTRL + ["%s_%s" % (m, n) for m in ("xgb", "lgbm") for n in NAMES]


def load() -> pd.DataFrame:
    """Одна таблица (бумага, дата): rv_tplus, gap, group и σ² всех рядов (`k1`, `k4`…, `xgb_A1`…). Только чтение входов (хэши сверяет вызывающий)."""
    k = pd.read_parquet(P.OUT / "k4567.parquet")
    k = k[KEY + ["t_plus", "group", "gap", "rv_tplus"] + ["sigma2_" + c for c in CTRL]].rename(columns={"sigma2_" + c: c for c in CTRL})
    for m in ("xgb", "lgbm"):
        h = pd.read_parquet(P.OUT / ("hyb_%s.parquet" % m))
        h = h[KEY + ["sigma2_" + n for n in NAMES]].rename(columns={"sigma2_" + n: "%s_%s" % (m, n) for n in NAMES})
        k = k.merge(h, on=KEY, how="left")
    return k


def common(k: pd.DataFrame, cols: list, first: str = P.FIRST_TARGET) -> pd.DataFrame:
    """Строки, где все ряды cols конечны и > 0 и RV > 0, цели с first."""
    ok = (k["t_plus"] >= pd.Timestamp(first)) & np.isfinite(k["rv_tplus"]) & (k["rv_tplus"] > 0)
    for c in cols:
        ok &= np.isfinite(k[c]) & (k[c] > 0)
    return k[ok].reset_index(drop=True)


def crit_window(k: pd.DataFrame) -> pd.DataFrame:
    """Строки окна выбора M16 × CRIT_YEARS по году цели (R9)."""
    y = k["t_plus"].dt.year
    return k[(k["group"] == "M16") & (y >= CRIT_YEARS[0]) & (y <= CRIT_YEARS[1])]


def crit_panel(k: pd.DataFrame, cols: list) -> pd.DataFrame:
    """-> (ряд, crit_panel, число бумаг, число дат): среднее по бумагам M16 crit_wf на общих датах окна; бумага с < 2 годами цели не входит."""
    w = crit_window(common(k, cols))
    rows = []
    for c in cols:
        per = []
        for _, g in w.groupby("ticker"):
            if g["t_plus"].dt.year.nunique() >= 2:
                r = g["rv_tplus"].to_numpy(float) / g[c].to_numpy(float)
                per.append(crit_wf(r - np.log(r) - 1.0, g["t_plus"]))
        rows.append(dict(series=c, crit_panel=float(np.mean(per)) if per else float("nan"), n_tickers=len(per), n_rows=len(w)))
    return pd.DataFrame(rows)


def select(sc: pd.DataFrame) -> dict:
    """Финалисты модулей, позиции A–D, тройка и условие У-16 по таблице crit_panel (У-12)."""
    v = sc.set_index("series")["crit_panel"]
    k1 = float(v["k1"])
    fin = {m: min(["%s_%s" % (m, n) for n in NAMES], key=lambda s: v[s]) for m in ("xgb", "lgbm")}
    ctl = {"xgb": "k4", "lgbm": "k5"}
    beats = {s: bool(v[s] <= v[ctl[s.split("_")[0]]] - MARGIN) for s in v.index if "_" in s}
    pos = {}
    for p, ns in POS.items():
        best = min(["%s_%s" % (m, n) for m in ("xgb", "lgbm") for n in ns], key=lambda s: v[s])
        pos[p] = dict(best=best, crit=float(v[best]), passes=bool(v[best] < k1 - MARGIN))
    triple = [p for p in sorted(pos, key=lambda p: pos[p]["crit"]) if pos[p]["passes"]][:3]
    u16 = [s for s in v.index if "_" in s and v[s] <= k1 - MARGIN and beats[s]]
    return dict(finalists=fin, positions=pos, triple=triple, triple_cfgs=[pos[p]["best"] for p in triple], beats_k45=beats, u16_configs=u16, u16=bool(u16), crit_k1=k1)


def loto_list(sel: dict) -> list:
    """У-13: A1-XGB, A1-LGBM и конфигурации тройки (её модуль); без дублей, ≤ 5 (порядок: A1, затем тройка по crit_panel)."""
    out = []
    for s in ["xgb_A1", "lgbm_A1"] + sel["triple_cfgs"]:
        if s not in out:
            out.append(s)
    return out[:5]


def declare(sel: dict, n_loto: int) -> int:
    """У-14: N = 585 + 20 + n_LOTO в CHECKPOINT.json ДО любого p; повторное объявление другим N — отказ."""
    n = 585 + 20 + n_loto
    cp = json.loads((P.REP / "CHECKPOINT.json").read_text())
    if cp.get("N_declared_before_stats") not in (None, n):
        raise SystemExit("N уже объявлено: %s" % cp["N_declared_before_stats"])
    cp.update(N_declared_before_stats=n, loto_declared=loto_list(sel), declared_at=time.strftime("%Y-%m-%d %H:%M"), leaf="P3")
    (P.REP / "CHECKPOINT.json").write_text(json.dumps(cp, ensure_ascii=False, indent=1))
    return n


def run(log=print) -> dict:
    """Счёт SELECTION.csv и объявление N; возвращает sel."""
    k = load()
    sc = crit_panel(k, model_cols())
    sel = select(sc)
    sc["finalist"] = sc["series"].isin(sel["finalists"].values())
    sc["beats_own_ctrl_by_margin"] = sc["series"].map(sel["beats_k45"])
    sc["triple_position"] = sc["series"].isin(sel["triple_cfgs"])
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    P.write_csv(sc, "statistics/SELECTION.csv", "crit_panel = среднее по M16 crit_wf, общие даты, цели 2017–2021 (R9, У-12); %s" % STAT)
    n = declare(sel, len(loto_list(sel)))
    (STATS_DIR / "SELECTION.json").write_text(json.dumps({**sel, "N_declared": n, "loto": loto_list(sel)}, ensure_ascii=False, indent=1))
    log("финалисты %s; тройка %s; У-16: %s; LOTO %s; N=%d" % (sel["finalists"], sel["triple_cfgs"], sel["u16"], loto_list(sel), n))
    return sel
