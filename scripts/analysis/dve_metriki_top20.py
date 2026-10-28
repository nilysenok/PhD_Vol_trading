# -*- coding: utf-8 -*-
"""Две метрики на топ-20 точек каждой S1–S6 по проходу 1 (Ж-102 п. 2 (а, б), п. 3; Ж-103 п. 4). Справочно, train4 (2016–2019), потолок.

Тот же набор точек, что в диагностике #467 (prohod1_diag_vybor.vybor). Ряды — из ledger без пересчёта: excess из шарда `port`
минус 5 % × база платы за заём шорта из шарда `zaem` (как в tselevaya_train4). Только чтение.
Определения (Д7): Sharpe — дневной excess над КС, ddof=1, √252 (fastbt.s9.sharpe); mSR, skew, kurt — fastbt.metriki_e.
Запуск:  python3 scripts/analysis/dve_metriki_top20.py        (самопроверка:  ... --test)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastbt.metriki_e import asimmetriya, ekscess, msr, sharpe  # noqa: E402

LEDGER = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/docs/tables/progon_e1/_ledger_v2")
OUT = ROOT / "docs/tables/dve_metriki"
SHAPKA = "# разработка train4 (2016–2019), справочно, потолок: выбор и счёт на одних данных; в отбор не идёт; плата за шорт 5 %\n"
STAVKA, SIDS, K = 0.05, [f"S{i}" for i in range(1, 7)], 10
KOLONKI = ["sharpe", "msr", "razryv", "skew", "kurt", "sharpe_bez_hud", "msr_bez_hud", "sharpe_bez_luch", "msr_bez_luch",
           "sharpe_v_dele", "msr_v_dele"]


def _sh(r) -> float:
    return float(sharpe(r)) if len(r) > 2 else float("nan")


def _ms(r) -> float:
    return float(msr(r)) if len(r) > 2 else float("nan")


def metriki(r) -> dict:
    """Полный ряд, без K худших и без K лучших дней, дни с ненулевым рядом («Шарп в деле», Ж-103 п. 4)."""
    r = np.asarray(r, float)
    s = np.sort(r)
    bez_hud, bez_luch, v_dele = s[K:], s[:-K], r[r != 0.0]
    m, sh = _ms(r), _sh(r)
    return dict(dney=len(r), dney_v_dele=len(v_dele), sharpe=sh, msr=m, razryv=m - sh, skew=asimmetriya(r), kurt=ekscess(r),
                sharpe_bez_hud=_sh(bez_hud), msr_bez_hud=_ms(bez_hud), sharpe_bez_luch=_sh(bez_luch), msr_bez_luch=_ms(bez_luch),
                sharpe_v_dele=_sh(v_dele), msr_v_dele=_ms(v_dele))


def verdikt(sh: float, ms: float) -> str:
    """Правило Ж-102 п. 3. «mSR сильно ниже» понимаем как mSR ≤ Sharpe/2 (↺ порог не задан владельцем; записан здесь)."""
    if sh < 0 and ms < 0:
        return "нет информации"
    if sh > 0.5 and ms > 0:
        return "рабочая"
    if 0 <= sh <= 0.5 and ms <= sh / 2:
        return "проблема хвостов"
    if 0 <= sh <= 0.5:
        return "вне правила: Sharpe 0…0,5, mSR не ниже (хвосты не причина)"
    return "вне правила"


def svod(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in df.groupby("sid"):
        med = {c: float(np.nanmedian(g[c])) for c in KOLONKI + ["dney_v_dele"]}
        rows.append(dict(sid=sid, tochek=len(g), ustojchivyh=int(g.ustojchiva.sum()), msr_nan=int(g.msr.isna().sum()), **med,
                         verdikt=verdikt(med["sharpe"], med["msr"])))
    return pd.DataFrame(rows)


def schet() -> None:
    from fastbt import ledger_v2 as lm, zaem_v2
    from prohod1_diag_vybor import vybor
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    out = []
    for sid in SIDS:
        top, _ = vybor(LEDGER, sid)
        ch = lm.Chitatel(LEDGER, sid, sverka_kod=True)
        byi = {r["sobol_i"]: r for r in ch.rows() if r.get("istochnik") == "sobol1"}
        for _, p in top.iterrows():
            ref = byi[p.sobol_i]
            r = ch.matrica([ref])[:, 0] - STAVKA * zaem_v2.zaem_ryad(LEDGER, sid, ref)
            out.append(dict(sid=sid, sobol_i=int(p.sobol_i), ustojchiva=bool(p.ustojchiva), **metriki(r)))
        print(sid, "ok", flush=True)
    df = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "top20.csv").write_text(SHAPKA + df.to_csv(index=False), encoding="utf-8")
    sv = svod(df)
    (OUT / "top20_svod.csv").write_text(SHAPKA + sv.to_csv(index=False), encoding="utf-8")
    print(sv.round(3).to_string())


def samoproverka() -> None:
    rng = np.random.default_rng(1)
    r = np.zeros(300)
    r[::2] = rng.normal(0.001, 0.01, 150)
    m = metriki(r)
    assert m["dney_v_dele"] == 150 and m["sharpe_v_dele"] > m["sharpe"]          # нули занижают Sharpe в деле < полного
    r2 = rng.normal(0.0005, 0.01, 500)
    r2[0], r2[1] = 0.5, -0.5
    m2 = metriki(r2)
    assert m2["sharpe_bez_hud"] > m2["sharpe"] > m2["sharpe_bez_luch"] - 1 and m2["sharpe_bez_luch"] < m2["sharpe"]
    assert abs(m2["skew"]) < 0.5 and m2["kurt"] > 10                              # выбросы дают толстые хвосты
    assert np.isnan(_ms(np.array([0.0, 0.0])))                                    # пустая лента не падает
    assert verdikt(-1, -1) == "нет информации" and verdikt(1, 1) == "рабочая" and verdikt(0.3, 0.0) == "проблема хвостов"
    print("самопроверка ok")


if __name__ == "__main__":
    samoproverka() if "--test" in sys.argv else schet()
