"""К-Ф4 (Ж-065 п. 3, контракт §1.7): паспорт A с ФДР внутри дня. Только вызов существующих модулей.
Строки: A, «A только Sk» (s9.Engine.port; ФДР — fdr.noch_iz_pos), P1 итерации 1 S1–S6 (Evaluator(fdr=…), в т.ч. выход лимиткой).
Колонки: база (fdr=None), +ФДР T+0, +ФДР T+1. Вклад ФДР = годовая сумма дневного fdr = Σ(total − r) при mg = 0
(mg ≠ 0 — остановка). Диагностика, train 2018–2019, комиссия 0,04 %, D·0,87; в ledger и реестр отбора не добавлено.
Запуск: FASTBT_MAIN=$PWD FASTBT_BARS1M=<D25/bars1m> python3 -m fastbt.pasport_fdr"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from fastbt import fdr as fdr_mod
from fastbt import metriki_e as me

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/tables/diag/pasport_A_fdr.csv"
COMM = 0.0004
YEARS = (2018, 2019)
NAMES = ["S1", "S2", "S3", "S4", "S5", "S6"]
STATUS = "разработка 2018–2019, после аудита, после #370/#398, потолок"
OGOVORKA = ("даты отсечки 2017–2021 — до правки #400 (не слит, ждёт владельца); эффект — несколько выплат MOEX, TATN, PHOR, NVTK")
COLS = ["строка", "колонка", "период", "sharpe", "msr", "вклад_ФДР_%_год", "Δ_ФДР_к_базе_%_год", "Δ_mSR_к_базе"]
KOLONKI = ("база", "+ФДР T+0", "+ФДР T+1")


def proverka_mg(df) -> None:
    """mg = max(0, used − 1)·…; при mg ≠ 0 «total − r» — не ФДР: вклад не считается (контракт §1.7)."""
    if float(np.max(df["exposure"].to_numpy())) > 1.0 + 1e-9:
        raise SystemExit("mg ≠ 0: экспозиция > 1, вклад ФДР как Σ(total − r) не определён")


def vklad_god(df, years=YEARS) -> float:
    """Σ_{d∈y}(total − r) · 100, усреднённая по годам (для года — сама годовая сумма), % годовых."""
    proverka_mg(df)
    d = (df["total"] - df["r"]).to_numpy()
    yr = np.asarray(df.index.year)
    return float(100 * sum(d[yr == y].sum() for y in years) / len(years))


def _ex(df, years):
    return df["excess"].to_numpy()[np.isin(np.asarray(df.index.year), years)]


def kolonki_pos(eng, pos, comm: float = COMM) -> dict:
    """База и T+0/T+1 для позиций pos (все сделки по close): те же r, gross, экспозиция, оборот — меняется только ставка."""
    base = eng.port(pos, comm=comm)
    out = {KOLONKI[0]: base}
    for name, rez in zip(KOLONKI[1:], fdr_mod.REZHIMY):
        noch = fdr_mod.noch_iz_pos(pos, eng.first, rez)
        noch_ = np.bincount(eng.r["date"], noch, len(eng.dates)) / eng.cnt
        out[name] = eng._finish(base["r"].to_numpy(), base["gross"].to_numpy(), base["exposure"].to_numpy(),
                                base["turnover"].to_numpy(), 1.0, noch_)
    return out


def g3(cols: dict, nazvanie: str) -> None:
    """Г3 / G2: у торговли по close «+ФДР T+0» совпадает с базой побайтно."""
    if cols[KOLONKI[0]]["excess"].to_numpy().tobytes() != cols[KOLONKI[1]]["excess"].to_numpy().tobytes():
        raise SystemExit(f"{nazvanie}: «+ФДР T+0» не совпадает с базой побайтно (Г3) — стоп")


def stroki(nazvanie: str, cols: dict) -> list:
    """Строки csv: по колонке и периоду (2018, 2019, train): Шарп, mSR, вклад, Δ к базе."""
    out = []
    base = cols[KOLONKI[0]]
    for kol, df in cols.items():
        for per, yrs in (("2018", (2018,)), ("2019", (2019,)), ("train", YEARS)):
            x, xb = _ex(df, yrs), _ex(base, yrs)
            v, vb = vklad_god(df, yrs), vklad_god(base, yrs)
            out.append([nazvanie, kol, per, me.sr_dnevnoy(x) * np.sqrt(252), me.msr(x), v, v - vb, me.msr(x) - me.msr(xb)])
    return out


def code() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def rows_A() -> tuple:
    from fastbt import s9, shag0
    from fastbt.sweep_val_baza import s9_engine
    eng = s9_engine()
    pos, sidr, out = np.asarray(eng.A_row, float), shag0.sid_row(eng), []
    for nm, p in [("A (все семейства)", pos)] + [(f"A только {s}", shag0.keep_sids(pos, sidr, [s])) for s in NAMES]:
        cols = kolonki_pos(eng, p)
        g3(cols, nm)
        out += stroki(nm, cols)
        print(nm, "готово", flush=True)
    return out, s9.cache_key()


def rows_P1() -> tuple:
    from fastbt import diag_zh066_run as run66
    from fastbt import etap1
    from fastbt.etap1_eval import YAKOR, Evaluator
    out, digs = [], []
    for sid in NAMES:
        pt = {**YAKOR, **run66.point(sid)}
        inp = etap1.build_inputs(sid)
        digs.append(f"{sid}:{inp.digest()}")
        cols = {k: Evaluator(sid, inp, etap1.base_point(sid), COMM, fdr=f).excess(pt, "train")[0]
                for k, f in zip(KOLONKI, (None, "t0", "t1"))}
        if pt.get("k_s") != "net" and pt.get("proverka") == "bary":                # выход лимиткой — только при лимитном входе
            cols["+выход лимиткой (T+0)"] = Evaluator(sid, inp, etap1.base_point(sid), COMM, fdr="t0").excess(
                {**pt, "vyhod": "lim"}, "train")[0]
        out += stroki(f"P1 {sid}", cols)
        print("P1", sid, "готово", flush=True)
    return out, "; ".join(digs)


def main() -> None:
    from fastbt import diag_zh066 as dz
    ra, ha = rows_A()
    rp, hp = rows_P1()
    meta = {"статус": STATUS, "оговорка": OGOVORKA, "коммит кода": code(), "хэш входов": f"A: {ha}; P1: {hp}",
            "определения": "комиссия 0,04 %, D·0,87; Шарп годовой (ddof=1), mSR — metriki_e; вклад ФДР — Σ(total − r)·100 за год, "
                           "для train — среднее по двум годам; mg = 0 проверено на каждой колонке; выход лимиткой — только P1 с лимитным входом"}
    dz.write_csv(OUT, meta, COLS, ra + rp)


if __name__ == "__main__":
    main()
