"""V7-0, G6: карта переиспользования абляций п. 12 блока Ж-232. Для каждого пункта — существующий СОХРАНЁННЫЙ ряд V1–V6 (id, программа, QLIKE по окнам 2017–2021 / 2022–2025 / вся история на общих 2271 датах, класс) или «нет, нужен счёт».
QLIKE пересчитывается из файла прогноза (определение `models_v2.metrics.qlike`), класс — из `PRIOR_SBER.csv` (V6 лист 0). Только чтение; ничего не обучается."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s14 import v7_0_ctrl as C, v7_0_io as IO

V4, V6 = "data/regen/harj_hybrid_v4/", "data/regen/harj_hybrid_v6/"
# (пункт п. 12, название, [(ряд, программа, файл или None для контролей)], вывод/комментарий)
ITEMS = [
    (1, "HAR-J", [("K1", "V1–V6 (контроль)", None)], "есть"),
    (2, "XGBoost", [("K2", "V1–V6 (контроль)", None)], "есть"),
    (3, "LightGBM", [("K3", "V1–V6 (контроль)", None)], "есть"),
    (4, "Лучший прежний гибрид", [("V4:NTP2", "V4", V4 + "NTP2.parquet"), ("V6:A2", "V6", V6 + "A2.parquet"), ("J210a:A1_roll_126", "V1 (Ж-210)", "data/regen/xgb_lgbm_full_v1/e/series_a.parquet")], "есть (три ответа: лучший PIT по всей истории, лучший PIT 2017–2021, потолок по всей истории)"),
    (5, "HAR-J + ручные M10-признаки", [("V6:A0", "V6", V6 + "A0.parquet"), ("V6:A0g", "V6", V6 + "A0g.parquet")], "есть: набор S (части дня, семивариации, RQ, форма); A0 — линейный корректор, A0g — GBM на том же S"),
    (6, "HAR-J + TCN", [], "нет, нужен счёт: нейросетевых энкодеров M10 в V1–V6 не было"),
    (7, "HAR-J + CNN", [], "нет, нужен счёт: то же"),
    (8, "HAR-J + Path Signature", [("V6:A1", "V6", V6 + "A1.parquet"), ("V6:A2", "V6", V6 + "A2.parquet"), ("V6:A3", "V6", V6 + "A3.parquet")], "есть для сигнатуры порядка 2 (площади Леви, окна 1 и 5 дней) в линейном (A1, A3) и GBM-корректоре (A2); обучаемый энкодер сигнатуры — нет"),
    (9, "HAR-J + обычный Gating", [("V4:B1", "V4", V4 + "B_B1.parquet"), ("V4:C6", "V4", V4 + "C6.parquet")], "есть частично: динамические веса смеси (B1–B4, B6) и два режима (C5–C10); обучаемый softmax-гейт по состоянию — нет"),
    (10, "HAR-J + Residual MoE", [], "нет, нужен счёт: ближайшие V4:C6 (режимы) и V6:F1 (устойчивая цель по средам) — не смесь экспертов с гейтом на положительной поправке"),
    (11, "HAR-J + M10 Encoder + Residual MoE", [], "нет, нужен счёт: энкодера и MoE в V1–V6 нет"),
]
V6_COMP = [("A0", "S", "LIN"), ("A0g", "S", "GBM"), ("A1", "S + Σ", "LIN"), ("A2", "S + Σ", "GBM"), ("A3", "S + Σ + Σ3", "LIN"), ("C1", "S + Q", "LIN"), ("C2", "S + Q", "GBM"), ("C3", "S + Q + u*", "LIN"), ("C4", "S + Q + u*", "GBM"),
           ("E1", "S + R", "LIN"), ("E2", "S + R", "GBM"), ("E3", "S + R + Σ", "LIN")]


def load_series(name: str, path, w: pd.DataFrame) -> np.ndarray:
    if path is None:
        return w[name].to_numpy(float)
    f = pd.read_parquet(IO.ROOT / path)
    if "ticker" in f.columns:
        f = f[f["ticker"] == "SBER"]
    d = pd.DatetimeIndex(pd.to_datetime(f["d"] if "d" in f.columns else f["date"]))
    col = "F" if "F" in f.columns else name.split(":", 1)[1]
    return pd.Series(f[col].to_numpy(float), index=d).reindex(w.index).to_numpy(float)


def qlike_windows(f: np.ndarray, w: pd.DataFrame) -> dict:
    q = metrics.qlike(w["rv"].to_numpy(float), f)
    return {k: float(q[C.wmask(w["t_plus"], k)].mean()) for k in ("2017_2021", "2022_2025", "all")}


def classes() -> dict:
    p = pd.read_csv(IO.ROOT / "docs/otchety/harj_hybrid_v6/PRIOR_SBER.csv", comment="#").set_index("series")["class"].to_dict()
    return {**p, "K1": "контроль", "K2": "контроль", "K3": "контроль"}


def table() -> pd.DataFrame:
    w, cl = C.aligned(), classes()
    rows = []
    for n, title, series, note in ITEMS:
        if not series:
            rows.append(dict(item=n, title=title, series="", program="", status="нет, нужен счёт", cls="", q_2017_2021=np.nan, q_2022_2025=np.nan, q_all=np.nan, note=note))
        for s, prog, path in series:
            f = load_series(s, path, w)
            q = qlike_windows(f, w)
            cls = cl.get(s, "PIT-сопоставим" if s.startswith("V6:") else "")
            rows.append(dict(item=n, title=title, series=s, program=prog, status="есть", cls=cls, q_2017_2021=q["2017_2021"], q_2022_2025=q["2022_2025"], q_all=q["all"], note=note))
    return pd.DataFrame(rows)


def markdown(t: pd.DataFrame) -> str:
    L = ["# REUSE_MAP — карта переиспользования абляций п. 12 (Ж-232), SBER, h = 1", "", "Статус: " + IO.STATUS + ". QLIKE пересчитан из сохранённых прогнозов на общих 2271 датах (d = 2016-12-30 … 2026-02-26); окна по году цели t⁺.",
         "Класс «потолок» — ряд выбран на SBER (первая цифра — потолок); «PIT-сопоставим» — параметры по закрытым парам, дизайн выбран по панели/предрегистрации (SBER — диагностика); K2/K3 — с оговоркой: " + IO.CAVEAT + ".", "",
         "| № | Пункт | Ряд | Программа | Класс | QLIKE 2017–2021 | QLIKE 2022–2025 | QLIKE вся история | Вывод |", "|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "—" if pd.isna(v) else "%.4f" % v
    for _, r in t.iterrows():
        L.append("| %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (r["item"], r["title"], r["series"] or "—", r["program"] or "—", r["cls"] or "—", f(r["q_2017_2021"]), f(r["q_2022_2025"]), f(r["q_all"]), r["note"]))
    L += ["", "## Состав признаков V6 (A0, A0g, A1, A2, C*, E*) — для сравнения с V7 по составу", "", "S — дневные M10-признаки HAR-J (части дня, семивариации, RQ, форма); Σ — площади Леви (path signature порядка 2) окон 1 и 5 дней; Q — качество измерения; R — шок-восстановление; u* — индекс неустойчивости.", "",
          "| Конфигурация V6 | Признаки | Обучатель |", "|---|---|---|"] + ["| %s | %s | %s |" % c for c in V6_COMP]
    L += ["", "По составу (не по результату): V7-п. 5 (ручные M10-признаки) ≡ набор S в A0/A0g; V7 Path Signature (E3 блока) пересекается с Σ в A1/A2 (отличие — обучаемый энкодер вместо фиксированных площадей); V7 Fixed Statistical Encoder (E4) — набор из S. Эквивалентность и вычёркивание — решение G9 (дирижёр), не этого листа."]
    return "\n".join(L) + "\n"
