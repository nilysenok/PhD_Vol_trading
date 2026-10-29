"""A_zashch (Ж-124 п. 1): A (s9), ежегодный top-10 ансамбль (ApproachA → turnover.apply, как s9._job) — защищённая конфигурация,
база содержательных сравнений реверса. A_fix — фиксированная точка этапа 1, только якорь кода (в содержательные таблицы не входит).
Общий помощник листов б, в, е, ж: ряд и позиции берутся тем же путём, что в r1_a_cd.py (кэш s9 → s9.Engine.port(A_row)).

API: ryad_i_pozicii(period, div=None, variant="baza", kesh=None) -> (ряд DataFrame по датам, позиции {(тикер, Sk): {год: np.ndarray}}).
Ряд = выход s9.Engine.port (excess, exposure, …) + колонка `baza` (доля капитала в шорте за ночь × календарные дни / 365, база платы за шорт).
div — метка режима дивидендов ("NET" — без таблицы 2022–2024; иное — с ней): не совпала со средой (FASTBT_Z2_DIV) — отказ.
variant (как строки листа б; позиции всюду A_row, меняется только то, что названо):
  baza; long / short — только положительные / только отрицательные позиции A_row; comm0 — комиссия 0;
  div0 — позиции те же, PnL по ценам без прибавки дивиденда (как inp.div = 0 у A_fix; lr пересобран из цен, сверка вне отсечек);
  fdr_t0 / fdr_t1 — занятость на 18:50 по fdr.noch_iz_pos. В модели s9 позиция держится ночью целиком (exposure = |поз|),
  поэтому t0 ≡ база; t1 добавляет капитал, освобождённый в этот день.
Кэш с текущим ключом не собран — отказ (сборка — явно, sobrat=True), чтобы период не считался молча на устаревшем кэше."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "scripts" / "analysis", ROOT / "scripts" / "revers"):
    sys.path.insert(0, str(_p))
from fastbt import a_strogaya as AS, data, dev, fdr as FDR, s9, zona  # noqa: E402
import r1_a_cd as CD  # noqa: E402
import r1_a_plata as PL  # noqa: E402

VARIANTY = ("baza", "long", "short", "comm0", "div0", "fdr_t0", "fdr_t1")
_DVIZHKI: dict = {}          # движки процесса: (период, без дивидендов) → s9.Engine (только при кэше s9 по умолчанию)


def komissiya(period: str) -> float:
    """test1 — 0,04 % (AS.COMM), иначе s9.COMM — ровно как r1_a_cd.schitat."""
    return AS.COMM if period == "test1" else s9.COMM


def kesh_s9(period: str, sobrat: bool = False) -> dict:
    CD.ustanovit_period(period)
    if period == "test1" and not os.environ.get("FASTBT_Z2_R2FINAL"):
        raise SystemExit("test1: не задан FASTBT_Z2_R2FINAL (σ̂ семейства R2 final_d0_s2emb)")
    if not (s9.OUT / f"cache_{s9.cache_key()}.pkl").exists() and not sobrat:
        raise FileNotFoundError(f"кэш s9 {period} с ключом {s9.cache_key()} не собран (sobrat=True — собрать, 2 процесса)")
    return CD.kesh(period)


def lr_bez_dividendov(ticker: str, dates: np.ndarray) -> np.ndarray:
    """Дневная лог-доходность close→close по реальным ценам (с поправкой на корпсобытия), без прибавки дивиденда в день гэпа."""
    d = data.daily_df()
    x = d[d["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    c = x["close"].values.astype(np.float64)
    cfg = s9.period_cfg(dev.make_config(dev.load_design(s9.MAIN_YAML)[0], s9.SCOPE))
    if cfg.corp_actions:
        for _, ev in data.corp_actions(ticker).iterrows():
            c[(x["date"] < ev["date"]).values] *= float(ev["factor"])
    if not np.array_equal(pd.to_datetime(x["date"].values).values, dates):
        raise ValueError(f"{ticker}: даты ряда цен ≠ датам кэша s9")
    lr = np.zeros(len(c))
    lr[:-1] = np.log(c[1:] / np.maximum(c[:-1], 1e-12))
    return lr


def kesh_bez_dividendov(kesh: dict) -> dict:
    """Копия кэша s9: lr без дивидендов, позиции те же. Вне дней гэпа lr совпадает с кэшем (иначе — отказ: другая загрузка цен)."""
    out = {}
    for t, c in kesh.items():
        lr = lr_bez_dividendov(t, c["dates"])
        izm = np.flatnonzero(~np.isclose(lr, c["lr"], rtol=0, atol=1e-12))
        if len(izm) > len(c["dates"]) // 50:            # дивидендных гэпов у тикера единицы в год
            raise ValueError(f"{t}: lr без дивидендов расходится с кэшем в {len(izm)} днях — это не дивиденды")
        out[t] = {**c, "lr": lr}
    return out


def dvizhok(period: str, kesh: dict | None = None, bez_div: bool = False):
    """s9.Engine на кэше s9. Без явного kesh движок хранится в процессе (воркеры Pool строят свой — период аргументом)."""
    CD.ustanovit_period(period)
    kl = (period, bez_div)
    if kesh is None and kl in _DVIZHKI:
        return _DVIZHKI[kl]
    k = kesh_s9(period) if kesh is None else kesh
    eng = s9.Engine(kesh_bez_dividendov(k) if bez_div else k)
    if kesh is None:
        _DVIZHKI[kl] = eng
    return eng


def port_fdr(eng, pos: np.ndarray, comm: float, rezhim: str) -> pd.DataFrame:
    """Порт s9 со ставкой на свободный капитал по занятости на 18:50 (fdr.noch_iz_pos); стороны — арифметика Engine.port."""
    dp = np.diff(pos, prepend=0.0)
    dp[eng.first] = pos[eng.first]
    last = np.r_[eng.first[1:], True]
    sides = np.abs(dp) + np.where(last, np.abs(pos), 0.0)
    return eng.port_rows(pos * eng.lr_row, sides, np.abs(pos), comm=comm, noch=FDR.noch_iz_pos(pos, eng.first, rezhim))


def pozicii_variant(pos: np.ndarray, variant: str) -> np.ndarray:
    if variant == "long":
        return np.where(pos > 0, pos, 0.0)
    if variant == "short":
        return np.where(pos < 0, pos, 0.0)
    return pos


def ryad_i_pozicii(period: str, div: str | None = None, variant: str = "baza", kesh: dict | None = None):
    if variant not in VARIANTY:
        raise ValueError(f"variant {variant!r}: есть {VARIANTY}")
    if div is not None and (div == "NET") != bool(zona.div_net()):
        raise SystemExit(f"div={div!r} не совпал со средой (FASTBT_Z2_DIV, div_net={zona.div_net()})")
    eng = dvizhok(period, kesh, bez_div=(variant == "div0"))
    pos = pozicii_variant(eng.A_row, variant)
    comm = 0.0 if variant == "comm0" else komissiya(period)
    p = port_fdr(eng, pos, comm, variant[4:]) if variant.startswith("fdr_") else eng.port(pos, comm=comm)
    p = p.assign(baza=PL.baza_shorta(pos, eng.cal[eng.r["date"]], eng.r["date"], eng.cnt))
    zona.check(p.index.max(), "ряд A_zashch")                        # ≥ 2025 — отказ
    return p, {(t, s): eng.cache[t]["A"][s] for t, s in eng.pairs}
