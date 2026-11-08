"""V6-0, G5: паспорт M10 SBER по годам 2014…2026 и тождество цели. Источник M10 и определение цели — как V5 G4 (`models_v2/s12/v5_0_bars`: `ticker_bands`, `day_table`; повторно используются, не переписываются):
RV = 1e4·Σ r², r_1 = ln(close_1/open_1), далее close-к-close, бары begin ∈ [10:00; 18:40]. Паспорт: сессии, бары на сессию (мода, доля ≠ моды, смены расписания), дни с пропусками, объём, нулевые доходности, шаг цены,
тождество Σr² = RV канона и = цель K1 (по t⁺). Вычислимость входов направлений A–F — по фактическим колонкам источника баров (begin, open, high, low, close, volume, value)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s10 import hp_paths as P
from models_v2.s12 import v5_0_bars as BA
from models_v2.s9 import hv2_m10 as M

TOL_ID = 1e-10
MIN_RUN = 20


def regime_changes(n: np.ndarray) -> int:
    """Смены расписания: число переходов между устойчивыми режимами n (серии ≥ 20 сессий подряд с одним значением); разовые дыры не считаются."""
    runs, start = [], 0
    for i in range(1, len(n) + 1):
        if i == len(n) or n[i] != n[start]:
            if i - start >= MIN_RUN:
                runs.append(n[start])
            start = i
    return int(sum(1 for a, b in zip(runs[:-1], runs[1:]) if a != b))


def tick(close: np.ndarray) -> float:
    u = np.unique(np.round(close, 6))
    return float(np.min(np.diff(u))) if len(u) > 1 else float("nan")


def identity_max(bars_rv: pd.Series, ref: pd.Series) -> float:
    a, b = bars_rv.to_numpy(float), ref.to_numpy(float)
    return float(np.nanmax(np.abs(a - b) / np.abs(b))) if len(a) else float("nan")


def year_rows(bars: pd.DataFrame, days: pd.DataFrame, k1: pd.DataFrame) -> pd.DataFrame:
    """days — `BA.ticker_bands('SBER')` (канонические дни: n, rv_bars, rv); k1 — кадр (t_plus, rv цели). Строка на год + итог."""
    main = D.main_session(bars)
    r = M.bar_returns(bars)
    yb = main["begin"].dt.year.to_numpy()
    yr = pd.DatetimeIndex(r["date"]).year.to_numpy()
    tg = days["rv_bars"].reindex(pd.DatetimeIndex(k1["t_plus"]))
    tg.index = k1.index
    rows = []
    for y in range(2014, 2027):
        d = days[(days.index.year == y) & days["rv"].notna()]
        mb, rx = main[yb == y], r["r"].to_numpy()[yr == y]
        if len(d) == 0:
            continue
        n = d["n"].to_numpy()
        mode = int(pd.Series(n).mode().iloc[0])
        sel = k1.index.year == y
        v = mb["volume"].to_numpy()
        rows.append(dict(section="year", year=y, n_sessions=len(d), n_bars=len(mb), bars_mode=mode, share_not_mode=float((n != mode).mean()), schedule_changes=regime_changes(n), share_days_with_gaps=float((n < mode).mean()),
                         has_volume=bool(len(v) and np.isfinite(v).all()), share_zero_volume_bars=float((v == 0).mean()), share_zero_returns=float((rx == 0).mean()), price_step=tick(mb["close"].to_numpy(float)),
                         max_rel_vs_daily=identity_max(d["rv_bars"], d["rv"]), n_target_days=int(sel.sum()), max_rel_vs_target=identity_max(tg[sel].dropna(), k1.loc[tg[sel].dropna().index, "rv"]) if sel.any() else float("nan")))
    return pd.DataFrame(rows)


def inputs_rows(bars: pd.DataFrame, yrs: pd.DataFrame) -> pd.DataFrame:
    """Вычислимость входов A–F: «да / нет / с оговоркой» по фактическим колонкам и статистике паспорта."""
    cols = set(bars.columns)
    vol = {"volume", "value"} <= cols and float(yrs["share_zero_volume_bars"].max()) < 0.01
    ohlc = {"open", "high", "low", "close"} <= cols
    gaps = float(yrs["share_days_with_gaps"].mean())
    note_sched = "расписание меняется (%d смен), число баров на сессию непостоянно: путь нормировать по доле сессии" % int(yrs["schedule_changes"].sum())
    spec = {"A": ("Path Signature", "да" if ohlc else "нет", "путь цены по M10 внутри дня (open/close баров); " + note_sched),
            "B": ("Time-Reversal", "да" if ohlc else "нет", "доходности M10 внутри дня; асимметрия вперёд/назад считается по барам сессии; %s" % note_sched),
            "C": ("Measurement Uncertainty", "с оговоркой" if ohlc else "нет", "только M10: RQ, сетки M20/M30/M60 подвыборкой; тиков, спредов и котировок нет — шум микроструктуры (signature plot) только от 10 минут; в источнике есть колонки %s" % sorted(cols)),
            "D": ("Historical Analog", "да", "дневные признаки RV, J, окна (2014–2026) из канонических дней; аналоги ищутся только по закрытым парам"),
            "E": ("Shock-Recovery", "да" if ohlc and vol else ("с оговоркой" if ohlc else "нет"), "шоки по барам M10; объём %s (доля нулей максимум %.4f)" % ("есть" if "volume" in cols else "отсутствует", float(yrs["share_zero_volume_bars"].max()))),
            "F": ("Distributionally Robust", "да", "использует распределение прошлых потерь/остатков H (RESID_SBER), новых данных не требует")}
    return pd.DataFrame([dict(section="inputs", direction=k, name=v[0], computable=v[1], note=v[2] + "; доля дней с пропусками баров в среднем %.4f" % gaps) for k, v in spec.items()])


def passport(w: pd.DataFrame) -> tuple:
    bars = pd.read_parquet(P.bars_path("SBER"))
    days = BA.ticker_bands("SBER")
    k1 = w[["t_plus", "rv"]]
    yrs = year_rows(bars, days, k1)
    tot = dict(section="total", year=0, n_sessions=int(yrs["n_sessions"].sum()), n_bars=int(yrs["n_bars"].sum()), max_rel_vs_daily=float(yrs["max_rel_vs_daily"].max()), max_rel_vs_target=float(yrs["max_rel_vs_target"].max()),
               n_target_days=int(yrs["n_target_days"].sum()))
    return pd.concat([yrs, pd.DataFrame([tot]), inputs_rows(bars, yrs)], ignore_index=True), dict(max_daily=tot["max_rel_vs_daily"], max_target=tot["max_rel_vs_target"], bars=bars, days=days)
