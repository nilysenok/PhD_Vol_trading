# -*- coding: utf-8 -*-
"""Реверс раунд 1, лист R-д: сигналы S1–S6 против к2 и плацебо (перемешивания). Образец — scripts/analysis/prohod1_diagnostika.py (#467).

Запуск:  python3 scripts/revers/r1_d.py schet S3 --period train4 [--n-placebo N --n-k2 N]
         python3 scripts/revers/r1_d.py svod  --period test1
train4: те же точки (топ-20 ledger прохода 1, выбор на train4), те же 10 плацебо и 20 лагов к2, что в #467 —
        таблицы сверяются с образцом (docs/tables/progon_e1/diag_*) в тесте; пишутся в _sverka_train4/, не в итог.
test1:  2022–2024, ТЕ ЖЕ точки (перебора и выбора на test1 нет), 20 перемешиваний плацебо, 20 лагов к2.
        Нужна поддержка period="test1" в Evaluator (лист R-P); данные ≥ 2022 до снятия замка не читаются.
"""
from __future__ import annotations

import argparse
import contextlib
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from prohod1_diag_metriki import metriki, sdelki, terciles  # noqa: E402
from prohod1_diag_vybor import vybor  # noqa: E402

LEDGER = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/docs/tables/progon_e1/_ledger_v2")
PERIODS = {
    "train4": dict(godov=4, n_placebo=10, out="docs/tables/revers_r1/_sverka_train4",
                   shapka="# разработка train4 (2016–2019), справочно, потолок: выбор и счёт на одних данных; в отбор не идёт\n"),
    "test1": dict(godov=3, n_placebo=20, out="docs/tables/revers_r1",
                  shapka="# реверс раунд 1, 2022–2024, справочно, без выбора\n"),
}
N_K2 = 20
KUB_DIR = Path(__file__).resolve().parent / "kub_prohod1"


def na_god(m: dict, godov: int, godov0: int = 4) -> dict:
    """metriki считает на 4 года (GODA); пересчёт на длину периода: сделок в год и расчётный Шарп (ir·√(n/годов))."""
    if not m or godov == godov0:
        return m
    n = m["sdelok"]
    ir = m["ir_sdelki"]
    out = dict(m, sdelok_v_god=n / godov)
    out["sharpe_rasch"] = ir * (n / godov) ** 0.5 if ir == ir else float("nan")
    return out


def zapis(df: pd.DataFrame, name: str, cfg: dict) -> None:
    out = ROOT / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(cfg["shapka"] + df.to_csv(index=False), encoding="utf-8")


@contextlib.contextmanager
def modul_period(ps, period: str):
    """progon_sobol.PERIOD (читает fabrika_real) и KUB — на время расчёта; прежние значения возвращаются в finally (без глобального следа)."""
    old = ps.PERIOD, ps.KUB
    ps.PERIOD, ps.KUB = period, KUB_DIR / "kub_x.yaml"   # кубы прохода 1 (коммит 63906a32): ledger не восстанавливается из кубов прохода 2
    try:
        yield
    finally:
        ps.PERIOD, ps.KUB = old


def schet(sid: str, period: str, n_pl: int | None = None, n_k2: int = N_K2) -> None:
    from fastbt import progon_kontrol as pk, progon_sobol as ps
    with modul_period(ps, period):
        _schet(ps, pk, sid, period, n_pl, n_k2)


def _schet(ps, pk, sid: str, period: str, n_pl: int | None, n_k2: int) -> None:
    cfg, n_pl = PERIODS[period], n_pl if n_pl is not None else PERIODS[period]["n_placebo"]
    top, info = vybor(LEDGER, sid)                       # точки выбраны на train4 (ledger прохода 1); на test1 не перебираются
    print(sid, period, info, flush=True)
    ev = ps.fabrika_real(sid).evr
    lags = pk.k2_lags(20)[:: max(20 // n_k2, 1)][:n_k2]
    tochki, kontroli, terc = [], [], []
    for _, p in top.iterrows():
        shared = dict(sid=sid, sobol_i=p.sobol_i, ustojchiva=p.ustojchiva)
        runs = [("baza", "", {})] + [("placebo", f"placebo:{s}", {"sig_fn": pk.placebo_sig_fn(s)}) for s in range(n_pl)] \
            + [("k2", f"k2:lag{g}", {"sig_fn": pk.k2_sig_fn(g)}) for g in lags]
        for tip, nm, vz in runs:
            row, _, _ = ev.evaluate_v2(p.pt, period, vz)
            df = sdelki(ev) if row["status"] == "ok" else pd.DataFrame()
            m = na_god(metriki(row, df), cfg["godov"]) if row["status"] == "ok" else {}
            (tochki if tip == "baza" else kontroli).append({**shared, "kontrol": nm, **m, **({"msr_zaem": p.msr_vsego} if tip == "baza" else {})})
            if tip == "baza" and len(df):
                terc.append(terciles(df).assign(**shared))
        print(sid, int(p.sobol_i), "ok", flush=True)
    zapis(pd.DataFrame(tochki), f"diag_tochki_{sid}.csv", cfg)
    zapis(pd.DataFrame(kontroli), f"diag_kontrol_{sid}.csv", cfg)
    zapis(pd.concat(terc) if terc else pd.DataFrame(), f"diag_terciles_{sid}.csv", cfg)


def main(argv: list) -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["schet", "svod"])
    ap.add_argument("sid", nargs="?")
    ap.add_argument("--period", choices=sorted(PERIODS), required=True)
    ap.add_argument("--n-placebo", type=int)
    ap.add_argument("--n-k2", type=int, default=N_K2)
    a = ap.parse_args(argv)
    if a.cmd == "schet":
        schet(a.sid, a.period, a.n_placebo, a.n_k2)
    else:
        from prohod1_diag_svod import svod
        svod(ROOT / PERIODS[a.period]["out"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
