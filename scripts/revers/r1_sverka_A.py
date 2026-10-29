"""Реверс р.1 (Ж-124 п. 2): лестница A «как на предзащите» → A_zashch (A s9, ежегодный top-10) на 2022–2024. Ничего не подбирается.

  FASTBT_Z2_DAILY=… FASTBT_Z2_DIV=<таблица 2022–2024> FASTBT_MAIN=<основная копия> \
      python3 scripts/revers/r1_sverka_A.py pos <папка> [--runs R0,Q1,…] [--workers 6]
  python3 scripts/revers/r1_sverka_A.py pechat            — печатный ряд MSP_A_inst.csv по годам 2022–2024
  python3 scripts/revers/r1_sverka_A.py lestnica <папка> [--out CSV]   (код ≠ 0, если |остаток| ≥ 1e-3)
pos — позиции A (6 семей × 17 тикеров) текущим кодом в зоне Z2 (test1, прогрев 2021); переключатели — в процессе воркера:
  маска [1]: охрана ApproachA.year «train_end >= n» → «> n» (окно 2024 = весь 2024, бар 2025 не нужен и не читается);
  ov=False — окно отбора A без платы за выход на конце (#259, коммит 043247c): stats.window(..., comm=0);
  div: none — без дивидендов (dividendy.events пуст), NET — без таблицы 2022–2024 (zona), full — FASTBT_Z2_DIV.
Слой портфеля — sverka_A_predzashchita.portfolio, лестница — sverka_A_lestnica.ladder/check_rest (импорт, копии нет).
Печатный ряд и ставка печати читаются построчно до первой даты ≥ 2025 (hold-out не читается)."""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import sverka_A_lestnica as L  # noqa: E402
import sverka_A_predzashchita as S  # noqa: E402

YEARS = (2022, 2023, 2024)
HOLD = pd.Timestamp("2025-01-01")
SIDS = ("S1", "S2", "S3", "S4", "S5", "S6")
PECHAT = {2022: 1.68, 2023: 2.99, 2024: 1.56}      # FINAL_RESULTS_CORRECTED_V2.md:21-26 (A); 2022–2025 — 1,32 (:14)
OUT = ROOT / "docs" / "tables" / "revers_r1" / "sverka_predzashchita_A.csv"
A_CD = ROOT / "docs" / "tables" / "revers_r1" / "a_cd_test1.csv"
RUNS = {  # имя: (маска [1], комиссия отбора, корпсобытия, дивиденды, окна платят выход)
    "R0": (True, .004, False, "none", False), "Q1": (True, .0005, False, "none", False),
    "Q2": (True, .0005, False, "none", True), "Q3": (True, .0005, False, "NET", True),
    "Q4": (True, .0005, False, "full", True), "Q5": (True, .0005, True, "full", True),
    "Q6": (False, .0005, True, "full", True), "P1": (False, .004, False, "none", False)}
KEYS = ("mask1", "sel", "corp", "div", "ov")
BASE = dict(mask1=True, sel=.004, corp=False, div="none", ov=False, tf=False,
            comm=0.0, rate="pechat", fwd=False, exit_fee=False, lev_cal=False)
NE = "не измеряется переключением: "
STEPS = [  # (код, пункт владельца: изменение, изменение состояния | None, заметка)
    ("1а", "сетка и способ выбора: сетка сигнал × РМ со стопами (V4) → та же", None,
     NE + "A_zashch = fastbt/approach_a.py — та же сетка V4/V41 (c22a4b3; Σ сигнал×РМ 49 360 на тикер по S1–S6) и top-10 с голосованием, "
     "что дала печать (ворота 25.10: позиции = daily_positions_pre2026 1,000000, МСП 1,3153); «37 000» с типами MA — daily_upper_bound.py "
     "(6a7a4636, in-sample на 2022–2026), не генератор печати; zh047 (Ж-047) — сетка A_fix/этапа 1"),
    ("1б", "сетка и способ выбора: комиссия отбора 0,40 % → 0,05 % (main.yaml comm)", dict(sel=.0005), ""),
    ("1в", "сетка и способ выбора: окна отбора платят выход (#259, 043247c)", dict(ov=True), ""),
    ("1г", "сетка и способ выбора: фильтр оборота (main.yaml turnover_filter)", dict(tf=True), "в печатном A его не было"),
    ("1д", "сетка и способ выбора: критерий отбора — Шарп без ставки", None, NE + "одинаков: stats.window в обоих"),
    ("2", "состав МСП и веса: S1–S6 × 17, EW по имеющимся парам", None,
     NE + "те же 6 семей и 17 тикеров; Engine.port делит на число строк даты (= EW по парам, как печать)"),
    ("3а", "ФДР и вычет ставки: ставка печати (MSP_A_inst) → решения ЦБ (msp.rate_cbr)", dict(rate="cbr"), ""),
    ("3б", "ФДР и вычет ставки: Н2, дни [d, d_next) вместо (d_prev, d]", dict(fwd=True), ""),
    ("3в", "ФДР и вычет ставки: Н4, плата за плечо по календарным дням", dict(lev_cal=True), "экспозиция ≤ 1 — вклад 0 по построению"),
    ("4а", "дивиденды: нет → таблицы 2015–2021 (режим NET на 2022–2024)", dict(div="NET"), "меняет окно отбора (прогрев 2015–2021)"),
    ("4б", "дивиденды: + таблица 2022–2024 (FASTBT_Z2_DIV)", dict(div="full"), ""),
    ("4в", "корпсобытия: нет → data/corporate_actions.csv", dict(corp=True), "вне списка владельца, есть в A_zashch"),
    ("5а", "комиссия: ряд 0 → 0,04 %", dict(comm=.0004), ""),
    ("5б", "комиссия: Н5, выход позиции на конце периода", dict(exit_fee=True), ""),
    ("6", "период калибровки: окно до 1.01.Y+1 (маска [1]) → до 1.01.Y (маска [0])", dict(mask1=False), "утечка"),
    ("7", "остановка торгов 2022 (28.02–23.03)", None, NE + "баров нет в обоих рядах; куда ложится ставка за дни простоя — шаг 3б"),
    ("8", "обработка шортов: шорт разрешён, платы за заём нет", None, NE + "в обоих плата 0 (s9 Engine.port, печать)"),
]


def sharpe_po_godam(p: pd.DataFrame, col: str = "excess") -> tuple:
    x = p[col]
    return S.sharpe(x), {y: S.sharpe(x[x.index.year == y]) for y in YEARS}


def chitat_do_2025(path: Path) -> pd.DataFrame:
    """CSV по возрастанию даты в 1-й колонке: строки читаются до первой даты ≥ 2025, дальше файл не читается."""
    rows = []
    with open(path, encoding="utf-8") as f:
        head = f.readline().strip().split(",")
        prev = None
        for line in f:
            d = pd.Timestamp(line.split(",", 1)[0])
            if prev is not None and d <= prev:
                raise ValueError(f"{path.name}: даты не по возрастанию ({d.date()} после {prev.date()})")
            if d >= HOLD:
                break
            rows.append(line.strip().split(","))
            prev = d
    df = pd.DataFrame(rows, columns=head)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").astype(float)


def pechat(csv: Path | None = None) -> pd.DataFrame:
    """Печатный ряд МСП-A 2022–2024 и ставка печати — формула msp.rate_restored на усечённом ряду."""
    from fastbt.msp import MSP_CSV
    d = chitat_do_2025(csv or MSP_CSV / "MSP_A_inst.csv")
    cal = S.cal_days(d.index, False)
    d["kr"] = (d["total_ret"] - d["excess_ret"]) / cal * 365.0
    return d.rename(columns={"excess_ret": "excess"})


def stavka(mode: str) -> pd.Series:
    from fastbt import msp
    r = pechat()["kr"] if mode == "pechat" else msp.rate_cbr()
    return r[r.index < HOLD]


def patch_vorker(mask1: bool, div: str, ov: bool) -> None:
    """Переключатели в процессе воркера (spawn: состояние не наследуется, вызывается в каждом)."""
    import inspect
    from fastbt import approach_a, dividendy, stats, zona
    if getattr(approach_a, "_r1_sverka", None) is not None:          # воркер Pool берёт много задач одного прогона
        assert approach_a._r1_sverka == (mask1, div, ov), "воркер получил задачу другого прогона"
        return
    approach_a._r1_sverka = (mask1, div, ov)
    if div == "NET":
        os.environ["FASTBT_Z2_DIV"] = "NET"
    zona.set_zone("test1")
    if div == "none":
        dividendy.events = lambda t: pd.DataFrame({"last_cum_date": pd.Series(dtype="datetime64[ns]"), "dps_rub": pd.Series(dtype=float)})
    if not ov:
        approach_a.window = lambda st, nb, j, comm, bpy=252.0: stats.window(st, nb, j, 0.0, bpy)
    if mask1:
        ns = {}
        exec(compile(year_mask1_src(inspect.getsource(approach_a.ApproachA.year)), "year_mask1", "exec"), approach_a.__dict__, ns)
        approach_a.ApproachA.year = ns["year"]


def year_mask1_src(src: str) -> str:
    """Исходник ApproachA.year с охраной «train_end > n»: при маске [1] окно 2024 кончается на конце ряда Z2 (= n)."""
    if src.count("train_end >= u.n") != 1:
        raise ValueError("охрана ApproachA.year изменилась: переключатель маски [1] не применим")
    return "\n".join(x[4:] for x in src.replace("train_end >= u.n", "train_end > u.n").split("\n"))


def proverit_hold(df: pd.DataFrame) -> None:
    if len(df) and pd.Timestamp(df["date"].max()) >= HOLD:
        raise PermissionError("позиции с датой ≥ 2025 (hold-out)")


def _unit(j):
    t, sid, run = j
    mask1, sel, corp, div, ov = RUNS[run]
    os.environ["NUMBA_NUM_THREADS"] = "1"
    patch_vorker(mask1, div, ov)
    import dataclasses
    from fastbt import dev, turnover
    from fastbt import design as dsg
    from fastbt.approach_a import ApproachA
    from fastbt.data import load_ticker, year_bounds
    from fastbt.unit import make_unit
    d, _ = dev.load_design(ROOT / "fastbt" / "designs" / "final" / "main.yaml")
    cfg = dataclasses.replace(dev.make_config(d, "dev"), mask_old=mask1, comm=sel, corp_actions=corp, years_a=YEARS, val_start=2021)
    dsg.apply(cfg.design)
    td = load_ticker(t, corp)
    u = make_unit(t, sid, td)
    res = ApproachA(u, cfg).run()
    tf = turnover.apply(res, u, cfg, "A")
    out = []
    for Y in sorted(res):
        ys, ye = year_bounds(u.dates, Y)[0], year_bounds(u.dates, Y + 1)[0]
        out.append(pd.DataFrame(dict(date=pd.to_datetime(u.dates[ys:ye]), ticker=t, strategy=sid, position=res[Y][1][ys:ye],
                                     position_tf=tf[Y][1][ys:ye], lr=u.lr[ys:ye])))
    return pd.concat(out) if out else None


def produce(d: Path, runs, workers: int) -> None:
    import multiprocessing as mp
    from fastbt.data import TICKERS
    d.mkdir(parents=True, exist_ok=True)
    for run in runs:
        with mp.get_context("spawn").Pool(workers) as pool:
            parts = [x for x in pool.imap_unordered(_unit, [(t, s, run) for t in TICKERS for s in SIDS]) if x is not None]
        df = pd.concat(parts)
        proverit_hold(df)
        df.to_parquet(d / f"{run}.parquet")
        print(run, RUNS[run], "строк", len(df), "пар", df.groupby(["ticker", "strategy"]).ngroups, df["date"].min().date(), df["date"].max().date(), flush=True)


def run_name(st: dict):
    return next((n for n, v in RUNS.items() if v == tuple(st[k] for k in KEYS)), None)


def evaluate(d: Path, st: dict, cache: dict):
    n = run_name(st)
    if n is None:
        return None
    if n not in cache:
        cache[n] = pd.read_parquet(d / f"{n}.parquet")
    pos = cache[n].assign(position=lambda x: x["position_tf" if st["tf"] else "position"])
    p = S.portfolio(pos.assign(gross=pos["position"] * pos["lr"]), stavka(st["rate"]), st["comm"], "pairs",
                    st["fwd"], st["exit_fee"], st["lev_cal"])
    return (*sharpe_po_godam(p), n, p)


def a_zashch() -> tuple:
    """Итог лестницы — A (s9) из a_cd_test1.csv: (а) полные и по годам."""
    t = pd.read_csv(A_CD, comment="#")
    t = t[t.portfel.str.startswith("A (s9")].set_index("stroka")["sharpe"]
    return float(t["(а) полные"]), {y: float(t[f"по году {y}"]) for y in YEARS}


def stroka(code, text, res, prev, note):
    sh, y, n = (res[0], res[1], res[2]) if res else (np.nan, {k: np.nan for k in YEARS}, "")
    r = dict(shag=code, izmenenie=text, sharpe=sh, d_sharpe=np.nan if prev is None else sh - prev[0], run=n, zametka=note)
    for k in YEARS:
        r[f"sharpe_{k}"], r[f"d_{k}"] = y[k], (np.nan if prev is None else y[k] - prev[1][k])
    return r


def tablica(d: Path) -> tuple:
    cache = {}
    ev = lambda s: evaluate(d, s, cache)                                    # noqa: E731
    tgt, tgt_y = a_zashch()
    lad = L.ladder(lambda s: ev(s) or ev_none(s), BASE, [(c, t, ch, n) for c, t, ch, n in STEPS], tgt)
    first = lad["first"]
    rows, prev = [stroka("0", "A как на предзащите: маска [1], отбор 0,40 %, без ОВ, без фильтра оборота, без дивидендов и корпсобытий, "
                         "ряд без комиссии, ставка печати, дни (d_prev, d], EW по парам", first, None, "")], first
    for code, text, chg, note, res, _ in lad["steps"]:
        rows.append(stroka(code, text, res if chg else prev, prev, note))
        prev = res if chg else prev
    rest = dict(shag="к", izmenenie="остаток: A_zashch (a_cd_test1.csv, A s9) − последний шаг", sharpe=tgt, d_sharpe=tgt - prev[0],
                run="", zametka="цель |остаток| < 1e-3")
    for k in YEARS:
        rest[f"sharpe_{k}"], rest[f"d_{k}"] = tgt_y[k], tgt_y[k] - prev[1][k]
    rows.append(rest)
    big = max((s for s in lad["steps"] if s[2]), key=lambda s: abs(s[5]))
    e = ev(dict(BASE, **big[2]))
    if e:
        sg = "знак тот же" if np.sign(e[0] - first[0]) == np.sign(big[5]) else "знак меняется"
        rows.append(stroka(f"порядок: {big[0]} первым", "крупнейший шаг лестницы, поставленный первым", e, first, f"в лестнице {big[5]:+.4f}; {sg}"))
    else:
        rows.append(dict(shag=f"порядок: {big[0]} первым", izmenenie="прогон не посчитан", zametka="нет позиций для этого состояния"))
    rows += spravka(d, cache, first, lad["state"])
    return pd.DataFrame(rows), lad, tgt


def ev_none(st):
    raise KeyError(f"нет прогона позиций для состояния {tuple(st[k] for k in KEYS)}")


def spravka(d: Path, cache: dict, first, state: dict) -> list:
    """Печать против строки 0 (дневной ряд); остановка торгов 2022 — A_zashch без дня перед простоем (справочно)."""
    pe = pechat()
    sh, y = sharpe_po_godam(pe)
    out = [stroka("печать", "печатный ряд MSP_A_inst.csv на 2022–2024 (читается до 2025)", (sh, y, "MSP_A_inst"), None,
                  "печать по годам: " + " / ".join(f"{PECHAT[k]:+.2f}" for k in YEARS))]
    p0 = first[3]
    j = p0.index.intersection(pe.index)
    ce = np.corrcoef(p0.loc[j, "excess"], pe.loc[j, "excess"])[0, 1]
    cg = np.corrcoef(p0.loc[j, "r"], pe.loc[j, "trading_ret"])[0, 1]
    note = (f"дат печати {len(pe)}, строки 0 {len(p0)}, общих {len(j)}; corr excess {ce:.3f}, corr trading {cg:.3f}; "
            f"экспозиция {p0['exposure'].mean():.4f} против {pe['exposure'].mean():.4f}; "
            f"trading за год {p0['r'].mean() * 252:.4f} против {pe['trading_ret'].mean() * 252:.4f}")
    out.append(stroka("невоспроизведение строки 0", "строка 0 − печать (по тем же формулам Шарпа)", (first[0] - sh, {k: first[1][k] - y[k] for k in YEARS}, ""),
                      None, note))
    out += po_semyam(cache[first[2]])
    fin = evaluate(d, state, cache)[3]
    sh_dolya = {n: float(cache[n][c].clip(upper=0).abs().sum() / cache[n][c].abs().sum())
                for n, c in ((first[2], "position"), (run_name(state), "position_tf"))}
    out.append(dict(shag="справка: шорты", izmenenie="доля шорта в Σ|позиция| (строка 0, A_zashch)", run=",".join(sh_dolya),
                    zametka="; ".join(f"{n}: {v:.3f}" for n, v in sh_dolya.items()) + "; плата за заём 0 в обоих"))
    gap = fin.index[(fin.index >= "2022-02-24") & (fin.index <= "2022-03-25")]
    m = ~fin.index.isin(fin.index[(fin.index >= "2022-02-25") & (fin.index < "2022-03-24")])
    out.append(stroka("справка: остановка 2022", "A_zashch без дат 25.02–23.03.2022 (день перед простоем несёт ставку за простой)",
                      (*sharpe_po_godam(fin[m]), "Q6"), None, "даты около простоя: " + ", ".join(str(x.date()) for x in gap)))
    return out


def po_semyam(pos0: pd.DataFrame) -> list:
    """Строка 0 против печати по семье Sk: торговый Шарп (без ставки) ряда «среднее pos·lr по тикерам», корреляция по дням."""
    from fastbt.msp import MSP_CSV
    out = []
    for sid in SIDS:
        pe = chitat_do_2025(MSP_CSV / f"{sid}_A_inst.csv")["trading_ret"]
        g = pos0[pos0.strategy == sid]
        r0 = (g["position"] * g["lr"]).groupby(g["date"]).mean()
        j = r0.index.intersection(pe.index)
        y0 = {k: S.sharpe(r0[r0.index.year == k]) - S.sharpe(pe[pe.index.year == k]) for k in YEARS}
        out.append(stroka(f"печать {sid}", f"{sid}: торговый Шарп строки 0 − печать {sid}_A_inst.csv", (S.sharpe(r0) - S.sharpe(pe), y0, ""), None,
                          f"строка 0 {S.sharpe(r0):.4f}, печать {S.sharpe(pe):.4f}; corr {np.corrcoef(r0[j], pe[j])[0, 1]:.3f}; "
                          f"доля дней |Δ| > 1e-6: {(np.abs(r0[j] - pe[j]) > 1e-6).mean():.3f}"))
    return out


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def shapka(d: Path) -> str:
    kom = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    from fastbt.msp import KEY_RATE, MSP_CSV
    vh = [Path(os.environ[k]) for k in ("FASTBT_Z2_DAILY", "FASTBT_Z2_DIV") if os.environ.get(k, "NET") != "NET"] + [A_CD, KEY_RATE]
    vh += [ROOT / "data" / "corporate_actions.csv"] + [MSP_CSV / f"{n}_A_inst.csv" for n in ("MSP", *SIDS)]
    s = f"# статус: реверс р.1, 2022–2024, справочно, без выбора (Ж-124 п. 2); hold-out 2025+ не считается\n# коммит кода: {kom}\n"
    s += "".join(f"# вход: {p.name} sha256={sha(p)}\n" for p in vh)
    return s + "".join(f"# позиции: {n}.parquet sha256={sha(d / f'{n}.parquet')} {dict(zip(KEYS, RUNS[n]))}\n" for n in RUNS if (d / f"{n}.parquet").exists())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("pos")
    q.add_argument("dir", type=Path)
    q.add_argument("--runs", default=",".join(RUNS))
    q.add_argument("--workers", type=int, default=6)
    sub.add_parser("pechat")
    r = sub.add_parser("lestnica")
    r.add_argument("dir", type=Path)
    r.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    if a.cmd == "pos":
        return produce(a.dir, a.runs.split(","), a.workers)
    if a.cmd == "pechat":
        sh, y = sharpe_po_godam(pechat())
        print(f"печатный ряд 2022–2024: {sh:.4f}", {k: round(v, 4) for k, v in y.items()}, "печать:", PECHAT)
        return None
    df, lad, tgt = tablica(a.dir)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(shapka(a.dir) + df.to_csv(sep=";", decimal=",", index=False, float_format="%.4f"), encoding="utf-8")
    print(df[["shag", "run", "sharpe", "d_sharpe", "d_2022", "d_2023", "d_2024"]].to_string(float_format=lambda v: "%.4f" % v))
    L.check_rest(lad, tgt)
    return None


if __name__ == "__main__":
    main()
