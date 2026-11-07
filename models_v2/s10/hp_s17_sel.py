"""P3-17, G17 (Ж-220, У-12/У-22): выбор на результатах с обучением на 17 — ПЕРВЫМ, до любого p. Функции выбора — `hp_stats_sel` (crit_panel, select, loto_list), без копирования; здесь — только данные P17 и объявление N.
crit_panel — M16 × цели 2017–2021 (E11 и SBER в критерий не входят); N = 608 + 20 + n_LOTO объявляется в `CHECKPOINT.json` (`N_declared_before_stats_p17`) до любого p. Статус: DEVELOPMENT, потолок."""
from __future__ import annotations

import json
import time

import pandas as pd

from models_v2.s10 import hp_paths as P, hp_stats_sel as S

KEY = S.KEY
D17 = P.REP / "statistics_p17"
HASHES17 = {"engine_panel_p17.parquet": "64efbc21ae4ee182", "hyb_xgb_p17.parquet": "4651022b40f8fb8f", "hyb_lgbm_p17.parquet": "4ae8d821a0a93377"}      # входы листа P17 (G16)
N_PRIOR = 608                  # N объявленный на панели из 28 (У-14) — все просмотренные прогоны
N_P17 = 20                     # пересчёт 20 объявленных конфигураций на 17 (У-21)
STAT = "исследовательский, HISTORICAL WALK-FORWARD, не независимый OOS; обучение на 17 (Ж-220); выбор — внутренняя валидация M16 × 2017–2021, цифры — потолок"


def load17() -> pd.DataFrame:
    """Таблица (бумага, дата): rv_tplus, gap, group, σ² K1, K4, K5 (обучение на 17), K6, K7 (не пересчитывались, У-23) и 20 гибридов `xgb_A1`… с обучением на 17. Только чтение."""
    k = _with_k67(pd.read_parquet(P.OUT / "k45_p17.parquet"))
    k = k.rename(columns={"sigma2_" + c: c for c in S.CTRL})
    for m in ("xgb", "lgbm"):
        h = pd.read_parquet(P.OUT / ("hyb_%s_p17.parquet" % m))
        h = h[KEY + ["sigma2_" + n for n in S.NAMES]].rename(columns={"sigma2_" + n: "%s_%s" % (m, n) for n in S.NAMES})
        k = k.merge(h, on=KEY, how="left")
    return k


def _with_k67(k: pd.DataFrame) -> pd.DataFrame:
    """`k45_p17.parquet` содержит K1, K4, K5; K6, K7 — из `k4567.parquet` (без пересчёта)."""
    old = pd.read_parquet(P.OUT / "k4567.parquet")[KEY + ["sigma2_k6", "sigma2_k7"]]
    k = k.merge(old, on=KEY, how="left")
    return k[KEY + ["t_plus", "group", "gap", "rv_tplus"] + ["sigma2_" + c for c in S.CTRL]]


def check_inputs() -> list:
    return ["%s: SHA %s != %s" % (f, P.sha16(P.OUT / f), h) for f, h in HASHES17.items() if P.sha16(P.OUT / f) != h]


def declare17(sel: dict, n_loto: int) -> int:
    """N = 608 + 20 + n_LOTO в `CHECKPOINT.json` ДО любого p; повторное объявление другим N — отказ. Ключи 28 (`N_declared_before_stats`) не трогаются."""
    n = N_PRIOR + N_P17 + n_loto
    cp = json.loads((P.REP / "CHECKPOINT.json").read_text())
    if cp.get("N_declared_before_stats_p17") not in (None, n):
        raise SystemExit("N уже объявлено: %s" % cp["N_declared_before_stats_p17"])
    cp.update(N_declared_before_stats_p17=n, loto_declared_p17=S.loto_list(sel), declared_at_p17=time.strftime("%Y-%m-%d %H:%M"), leaf="P3-17")
    (P.REP / "CHECKPOINT.json").write_text(json.dumps(cp, ensure_ascii=False, indent=1))
    return n


def run(log=print) -> dict:
    """Счёт `statistics_p17/SELECTION.csv` и объявление N; возвращает sel. p-значений здесь нет."""
    bad = check_inputs()
    if bad:
        raise SystemExit("стоп: входы не совпали: %s" % bad)
    k = load17()
    sc = S.crit_panel(k, S.model_cols())
    sel = S.select(sc)
    sc["finalist"] = sc["series"].isin(sel["finalists"].values())
    sc["beats_own_ctrl_by_margin"] = sc["series"].map(sel["beats_k45"])
    sc["triple_position"] = sc["series"].isin(sel["triple_cfgs"])
    D17.mkdir(parents=True, exist_ok=True)
    P.write_csv(sc, "statistics_p17/SELECTION.csv", "crit_panel = среднее по M16 crit_wf, общие даты, цели 2017–2021 (R9, У-12, У-22); %s" % STAT)
    n = declare17(sel, len(S.loto_list(sel)))
    hashes = {f: P.sha16(P.OUT / f) for f in HASHES17}
    (D17 / "SELECTION.json").write_text(json.dumps({**sel, "N_declared": n, "loto": S.loto_list(sel), "hashes": hashes}, ensure_ascii=False, indent=1))
    log("финалисты %s; тройка %s; У-16: %s; LOTO %s; N=%d" % (sel["finalists"], sel["triple_cfgs"], sel["u16"], S.loto_list(sel), n))
    return sel
