"""Реверс р.1, Ж-124 п. 2 / Ж-125 п. 5: разложение шага 5а сверки C/D (позиции предзащиты → fastbt после аудита) на
2022–2024 переключателями по одному, для МСП A, C, D. Лестница цепная: P0 — fastbt в режиме ворот (--leak-mask-old) на барах Z2,
затем по одному изменению до дизайна main.yaml (final_test_main.parquet). Слой — тот же, что в r1_sverka_CD (шаги 1–4).
  FASTBT_MAIN=<своя копия> FASTBT_DATA_ROOT=<осн.>/data/fastbt/daily python3 scripts/revers/r1_sverka_5a.py progon P0,P1 [--workers 8]
  ... python3 scripts/revers/r1_sverka_5a.py svod [--out CSV]   (код ≠ 0 — ворота не пройдены)
Hold-out: бары Z2 ≤ 2024 (отказ иначе), σ̂ — фильтр < 2025 на чтении (fastbt.data._read_cut), позиции — фильтр до чтения."""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "revers"))
import r1_sverka_CD as CD  # noqa: E402
from fastbt.config import Config  # noqa: E402

M = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading")
Z2 = M / "data/regen/test1_2022_2024"
VP_ALIGNED = M / "moex_volatility/results/final/strategies/data/vpred_aligned.parquet"
VP_FINAL3 = M / "data/regen/sigma_pred_h1_final3_full.parquet"
DIV_Z2 = ROOT / "docs/issledovaniya/2026-10-29_dividendy_2022_2024.csv"
MAIN_YAML = ROOT / "fastbt/designs/final/main.yaml"
OUT = Path(os.environ.get("FASTBT_MAIN", ROOT)) / "data/fastbt/sverka_5a"
APS, TOL = ("A", "C", "D"), 1e-3
# fix_b_median, fix_rate, fix_sharpe — с P0: на позиции A/C/D не влияют (B не считается; ставка и Шарп — слой, он общий)
BASE = dataclasses.replace(Config.gate(), fix_c_fill=True, fix_b_median=True, fix_rate=True, fix_sharpe=True,
                           approaches="ACD", years_a=tuple(range(2022, 2027)))
DZ = ("horizons", "c_trend_hysteresis", "turnover_filter", "tf_min_hold", "tf_reentry")      # ключи дизайна main.yaml
# (шаг, изменение, поля Config, ключи дизайна, σ̂, дивиденды 2015–2021, дивиденды 2022–2024) — накопительно
SHAGI = [("P0", "бары и воспроизведение: предзащита → fastbt --leak-mask-old (маска [1], отбор 0,40 %, без корп. событий, C = A при "
          "пустом C) на барах Z2 18:50, σ̂ vpred_aligned < 2025, без дивидендов; 2024 = E0 (маске [1] нужен бар ≥ 2025)",
          {}, 0, "aligned", False, False),
         ("P1", "утечка окна: маска [1] → [0] (A до 31.12.Y−1, валидация C/D [2020; Y) вместо [2021; Y+1)); "
          "в 2024 — вместе с шагом P0 (бары и воспроизведение)", {"mask_old": False}, 0, "aligned", False, False),
         ("P2", "fix_c_s12_val: валидация C у S1/S2 по полным позициям A", {"fix_c_s12_val": True}, 0, "aligned", False, False),
         ("P3", "fix_d_units: σ в D в годовых единицах (·√252)", {"fix_d_units": True}, 0, "aligned", False, False),
         ("P4", "комиссия отбора 0,40 % → 0,05 % на сторону (С1)", {"comm": 0.0005}, 0, "aligned", False, False),
         ("P5", "корпоративные события (data/corporate_actions.csv, IRAO 2015)", {"corp_actions": True}, 0, "aligned", False, False),
         ("P6", "дивиденды 2015–2021 в барах и PnL отбора (Ж-057/059; в коде сейчас всегда)", {}, 0, "aligned", True, False),
         ("P7", "окна валидации C/D и фильтра оборота заморожены на 2018–2021 (scope test: val_start 2018, val_end 2021)",
          {"val_start": 2018, "val_end": 2021}, 0, "aligned", True, False),
         ("P8", "горизонт h: подбор h ∈ {h1, h5, h22} → фиксированный h1 (main.yaml horizons)", {}, 1, "aligned", True, False),
         ("P9", "гистерезис тренда C → [0] (main.yaml c_trend_hysteresis)", {}, 2, "aligned", True, False),
         ("P10", "фильтр оборота (main.yaml turnover_filter, tf_min_hold, tf_reentry)", {}, 5, "aligned", True, False),
         ("P11", "источник σ̂: vpred_aligned → sigma_pred_h1_final3_full (main.yaml vpred)", {}, 5, "final3", True, False)]
SPRAV = [("iR2", "справочно, вне 5а (это 5б): σ̂ final3_full → ch4_dev ≤ 2021 + R2 final_d0_s2emb (Z2)", "r2", True, False),
         ("iDIV", "справочно, вне 5а: дивиденды 2022–2024 NET → таблица 2026-10-29_dividendy_2022_2024.csv", "final3", True, True)]


def konfig(shag: str) -> tuple:
    """(Config, env, div_dev) шага; справочные шаги — от P11 с одним изменением."""
    from fastbt import design, dev
    sp = {s[0]: s for s in SPRAV}
    st = next(s for s in SHAGI if s[0] == ("P11" if shag in sp else shag))
    yml = dev.load_design(MAIN_YAML)[0]
    kw = {}
    for s in SHAGI[:SHAGI.index(st) + 1]:
        kw.update(s[2])
    cfg = dataclasses.replace(BASE, **kw, design=design.freeze({k: yml[k] for k in DZ[:st[3]]}))
    sig, div_dev, div_z2 = (sp[shag][2:] if shag in sp else st[4:])
    env = {"FASTBT_Z2_DAILY": str(Z2 / "daily/ohlcv_daily_1850.parquet"), "FASTBT_Z2_DIV": str(DIV_Z2) if div_z2 else "NET",
           "FASTBT_Z2_VPRED": str(VP_ALIGNED if sig == "aligned" else VP_FINAL3), "NUMBA_NUM_THREADS": "1",
           "FASTBT_CACHE": str(OUT / ("cache_div" if div_dev else "cache_bezdiv"))}
    if sig == "r2":
        env |= {"FASTBT_Z2_R2FINAL": str(Z2 / "regen/R2/final"), "FASTBT_DEV_SIGMA": str(M / "data/regen/ch4_dev")}
    return cfg, env, div_dev


def _init(div_dev: bool) -> None:
    """Воркер spawn: без дивидендов 2015–2021 — пустой список событий (как на предзащите и в final_test_main 26.10)."""
    if not div_dev:
        from fastbt import dividendy
        pusto = pd.DataFrame({"last_cum_date": pd.Series(dtype="datetime64[ns]"), "dps_rub": pd.Series(dtype=float)})
        dividendy.events = lambda ticker: pusto


def progon(shag: str, workers: int) -> Path:
    import multiprocessing as mp
    from fastbt import pipeline
    from fastbt.data import TICKERS
    cfg, env, div_dev = konfig(shag)
    os.environ.pop("FASTBT_Z2_R2FINAL", None)
    os.environ.update(env)
    cfg = pipeline.z2_config(cfg)
    jobs = [(t, ["S1", "S2", "S3", "S4", "S5", "S6"], cfg, "test1") for t in TICKERS]
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(div_dev,)) as pool:
        dfs = [r[1] for r in pool.imap_unordered(pipeline.run_ticker, jobs)]
    p = pd.concat(dfs, ignore_index=True)
    p = p[p["approach"].isin(APS) & p["test_year"].isin(CD.GODY) & (p["date"] < CD.GRAN)]
    OUT.mkdir(parents=True, exist_ok=True)
    p.to_parquet(OUT / f"{shag}.parquet", index=False)
    print(f"{shag}: {len(p)} строк → {OUT / (shag + '.parquet')}", flush=True)


def dopolnit(p: pd.DataFrame, e0: pd.DataFrame) -> pd.DataFrame:
    """P0: единицы (тикер, стратегия, год), которых нет в прогоне, — из E0. Маске [1] для 2024 нужен бар ≥ 2025
    (fastbt/approach_a.py year: train_end ≥ u.n → None), на Z2 его нет: 2024 шага P0 = предзащита."""
    k = ["ticker", "strategy", "test_year"]
    est = set(map(tuple, p[k].drop_duplicates().to_numpy()))
    return pd.concat([p, e0[[tuple(x) not in est for x in e0[k].to_numpy()]]], ignore_index=True)


def pozicii(ap: str, shag: str) -> pd.DataFrame:
    if shag == "E0":                                 # позиции предзащиты; C — с заполнением C = A (шаг 4 сверки C/D)
        return CD.zapolnit_a(CD.pozicii("C"), CD.pozicii("A"))[0] if ap == "C" else CD.pozicii(ap)
    p = CD.pozicii(ap, CD.FINAL_TEST if shag == "E5" else OUT / f"{shag}.parquet")
    return dopolnit(p, pozicii(ap, "E0")) if shag == "P0" else p


def dolya_izm(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Доля строк (дата, стратегия, тикер), где позиция изменилась (> 1e-9) или строки нет в одном из двух наборов."""
    k = ["date", "strategy", "ticker"]
    m = a[k + ["position"]].merge(b[k + ["position"]], on=k, how="outer")
    return float(((m["position_x"] - m["position_y"]).abs().fillna(1.0) > 1e-9).mean())


def lestnica(ap: str, rate) -> list:
    """[(шаг, изменение, Шарпы, доля изменённых позиций)] от E0 до E5; справочные — отдельно (не в сумме)."""
    sloy = lambda p: CD.sharpy(CD.msp.portfolio(p, CD.s_dnyami(rate, False), CD.COMM, agg="pairs")["excess"])  # noqa: E731
    imena = [("E0", "позиции предзащиты (daily_positions_pre2026, C = A при пропуске; = шаг 4 сверки C/D)")]
    imena += [(s[0], s[1]) for s in SHAGI]
    imena += [("E5", "код и бары: fastbt сейчас на Z2 (бары 18:50, код после 26.10) → final_test_main.parquet "
               "(бары v4 ohlcv_daily_full с корп. событиями, код 25b76aa, 26.10)")]
    pos = {s: pozicii(ap, s) for s, _ in imena}
    out = [(s, izm, sloy(pos[s]), None if i == 0 else dolya_izm(pos[imena[i - 1][0]], pos[s])) for i, (s, izm) in enumerate(imena)]
    sprav = [(s, izm, sloy(pozicii(ap, s)), dolya_izm(pos["P11"], pozicii(ap, s)), sloy(pos["P11"]))
             for s, izm, *_ in SPRAV if (OUT / f"{s}.parquet").exists()]
    return out, sprav


def tablica(ap: str, st: list, sprav: list) -> pd.DataFrame:
    kl, nm = ["ves", *CD.GODY], (lambda k: "" if k == "ves" else f"_{k}")        # noqa: E731
    rows = [dict(obekt=ap, shag=s, izmenenie=izm, dolya_izm_poz=dol, **{f"sharpe{nm(k)}": sh[k] for k in kl},
                 **{f"delta{nm(k)}": None if i == 0 else sh[k] - st[i - 1][2][k] for k in kl}) for i, (s, izm, sh, dol) in enumerate(st)]
    sum5a = {k: st[-1][2][k] - st[0][2][k] for k in kl}
    rows.append(dict(obekt=ap, shag="5а", izmenenie="итого 5а = E5 − E0 (= шаг 5а сверки C/D)", **{f"delta{nm(k)}": sum5a[k] for k in kl}))
    rows.append(dict(obekt=ap, shag="ост", izmenenie="остаток (5а − ΣΔ шагов)",
                     **{f"delta{nm(k)}": sum5a[k] - sum(r[f"delta{nm(k)}"] or 0.0 for r in rows[1:len(st)]) for k in kl}))
    rows += [dict(obekt=ap, shag="—", izmenenie=t, **{f"delta{nm(k)}": 0.0 for k in kl}) for t in (
        "fix_b_median: только подход B — в A/C/D 0 по построению (B не считается)",
        "fix_rate, fix_sharpe: только слой МСП/отчёт, на позиции не влияют — 0 по построению; слой одинаков на всех шагах")]
    for s, izm, sh, dol, base in sprav:
        rows.append(dict(obekt=ap, shag=s, izmenenie=izm, dolya_izm_poz=dol,
                         **{f"sharpe{nm(k)}": sh[k] for k in kl}, **{f"delta{nm(k)}": sh[k] - base[k] for k in kl}))
    return pd.DataFrame(rows)


def proverit(df: pd.DataFrame, ref: pd.DataFrame) -> list:
    """Ворота: |остаток| < TOL; 5а = шаг 5а сверки C/D (C, D); E0 = шаг 4 сверки (C, D); P11 — конфиг main.yaml."""
    osh = []
    for ap, g in df.groupby("obekt"):
        ost = g[g["shag"] == "ост"].iloc[0]
        osh += [f"{ap} {c}: остаток {ost[c]:.1e}" for c in g.columns if c.startswith("delta") and not abs(ost[c]) < TOL]
        if ap in ref["obekt"].values:
            r, s = ref[ref["obekt"] == ap].set_index("shag"), g.set_index("shag")
            for imya, d in (("5а", s.loc["5а", "delta"] - float(r.loc["5а", "delta"])), ("E0", s.loc["E0", "sharpe"] - float(r.loc["4", "sharpe"]))):
                osh += [f"{ap}: {imya} ≠ сверка C/D ({d:.1e})"] if not abs(d) < TOL else []
    from fastbt import dev, pipeline          # P11 = конфиг final_test_main: dev.make_config(main.yaml, test), val_start 2018
    z, ym = pipeline.z2_config, dataclasses.replace(dev.make_config(dev.load_design(MAIN_YAML)[0], "test"), val_start=2018, approaches="ACD")
    return osh + ([] if z(konfig("P11")[0]) == z(ym) else ["P11: конфиг ≠ main.yaml (scope test)"])


def shapka() -> str:
    import hashlib, subprocess  # noqa: E401
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()        # noqa: E731
    kom = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    vh = [CD.POS_PRE2026, CD.FINAL_TEST, CD.msp.KEY_RATE, Z2 / "daily/ohlcv_daily_1850.parquet", VP_ALIGNED, VP_FINAL3,
          DIV_Z2, MAIN_YAML, *sorted((Z2 / "regen/R2/final").glob("final_d0_s2emb_h*__t1.parquet"))]
    nm = lambda p: p.relative_to(ROOT) if ROOT in Path(p).parents else p        # noqa: E731
    return ("# статус: реверс р.1, 2022–2024; разложение шага 5а сверки C/D (Ж-124 п. 2, Ж-125 п. 5), справочно, без выбора\n"
            f"# коммит: {kom}\n# хэш файла целиком (механически); σ̂ и позиции — только строки < 2025-01-01; ch4_dev — {M / 'data/regen/ch4_dev'}\n"
            + "".join(f"# вход: {nm(Path(p))} sha256={sha(p)}\n" for p in vh))


def main(argv=None) -> int:
    a = argparse.ArgumentParser()
    a.add_argument("rezhim", choices=("progon", "svod"))
    a.add_argument("shagi", nargs="?", default="")
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--out", type=Path, default=CD.TAB / "sverka_5a.csv")
    a = a.parse_args(argv)
    if a.rezhim == "progon":
        return [progon(s, a.workers) for s in a.shagi.split(",")] and 0
    df = pd.concat([tablica(ap, *lestnica(ap, CD.msp.rate_cbr())) for ap in APS], ignore_index=True)
    osh = proverit(df, pd.read_csv(CD.TAB / "sverka_predzashchita_CD.csv", comment="#"))
    a.out.write_text(shapka() + df.to_csv(index=False), encoding="utf-8")
    print(df[["obekt", "shag", "sharpe", "delta", "delta_2022", "delta_2023", "delta_2024", "dolya_izm_poz"]].round(4).to_string())
    print("ворота: " + ("пройдены" if not osh else "НЕ ПРОЙДЕНЫ: " + "; ".join(osh)))
    return 1 if osh else 0


if __name__ == "__main__":
    raise SystemExit(main())
