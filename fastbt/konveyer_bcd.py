"""Конвейер B/C/D главы 4 (T3, Э3): сетки × лестница σ × (Шарп и метрики, к2, к3, Δ к A, p Ледуа–Вольфа).

    scripts/konveyer_bcd.sh <A-yaml>      (= python3 -m fastbt.konveyer_bcd <A-yaml>)

Выбора точек нет: только таблицы. Период — только train 2018–2019 (`dev_train` подменяет годы, обрезку σ и дизайн A).
Считает готовый код: к2/к3 — `fast_kontrol`, условия и Δ — `shag_c`, точки D/B — `s9.S9.d_points`/`b_weights`, p ЛВ —
`shag0.delta_p`; N, DSR, Холм — `konveyer_bcd_n`. Здесь: ступень const, лестница, шапки, ключ (Д5)."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import fast_kontrol as fk
from fastbt import metriki_e as me
from fastbt import s9, shag0
from fastbt import shag_c as C
from fastbt.konveyer_bcd_n import add_n, light_rows, sr_stats

ROOT = Path(__file__).resolve().parent.parent
SWEEP, OUT = ROOT / "fastbt" / "designs" / "sweep", ROOT / "docs" / "tables" / "konveyer_bcd" / "chernovik"
TRAIN, COMMS = (2018, 2019), (0.00035, 0.0004)
LADDER = ("const", "naive", "harj", "hyb")
SRC = {"const": {1: "const", 5: "const"}, "naive": {1: "naive_h1", 5: "naive_h1"},
       "harj": {1: "harj_h1", 5: "harj_h5"}, "hyb": {1: "hyb_h1", 5: "hyb_h5"}}
HLESS, SCREEN = ("const", "naive"), (0.10, 0.50)   # ряды без h (наивная h5 ≡ h1); экран доли закрытых дней масок D (#296)
HEAD = "черновик, текущий A, train 2018–2019, без выбора точек"


@contextmanager
def dev_train(a_yaml, years=TRAIN):
    """На время счёта: годы = train, σ обрезается по концу train (ничего после 31.12 последнего года), дизайн A — a_yaml."""
    assert max(years) <= 2019, "черновик T3 — только train 2018–2019"
    saved = (s9.YEARS, shag0.YEARS, s9.CUT, s9.MAIN_YAML)
    s9.YEARS = shag0.YEARS = tuple(years)
    s9.CUT, s9.MAIN_YAML = pd.Timestamp(f"{max(years)}-12-31"), Path(a_yaml)
    try:
        yield
    finally:
        s9.YEARS, shag0.YEARS, s9.CUT, s9.MAIN_YAML = saved


def sha(path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def cache_key(a_yaml, grid_paths, comm: float, years=TRAIN, series: str = "", code: str | None = None) -> str:
    """Д5: ключ = хэш всех входов и кода: yaml A, сетки, комиссия, период, дайджест рядов σ, код fastbt."""
    from fastbt.cache import code_hash
    parts = [sha(a_yaml), [sha(p) for p in sorted(map(str, grid_paths))], float(comm), list(years), series,
             code or code_hash()]
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def ensure_const(eng) -> None:
    """Ступень «константная σ» (sigma_e.const: среднее √RV за 252 дня до d−1) → eng.sig['const'] (до CUT)."""
    from fastbt import sigma_e
    assert eng.dates.max() <= s9.CUT, "строки движка вышли за train"
    df =sigma_e.load("const").query("date <= @s9.CUT")
    S = np.full((len(eng.tickers), eng.nmax), np.nan)
    for ti, t in enumerate(eng.tickers):
        S[ti, :eng.n_t[ti]] = s9.align(eng.cache[t]["dates"], df[df["ticker"] == t])
    eng.sig["const"] = S


def pts_c(eng, rung, h, grid):
    pts = [p for p in C.plan("naive" if rung in HLESS else rung, grid) if p["h"] == h]
    G = C.gates(eng, SRC[rung][h], grid, naive=rung in HLESS)
    return [(dict(family=p["family"], point=p["label"]), G[(p["family"], p["label"])], "mask") for p in pts]


def pts_d(eng, rung, h, grid):
    fams = s9.S9.d_points(types.SimpleNamespace(e=eng), SRC[rung][h])
    return [(dict(family=f, point=lab), G, kind) for f, pts in fams.items() if f != "D6" for lab, G, kind in pts]


def pts_b(eng, rung, h, grid):
    return [(dict(family="B", point=f"cap{cap:g}_band{int(round(band * 100))}"),
             s9.S9.b_weights(types.SimpleNamespace(e=eng), SRC[rung][h], cap, band), "scale")
            for cap in grid["cap"] for band in grid["band"]]


PTS = {"B": pts_b, "C": pts_c, "D": pts_d}


def eval_point(bases: dict, g, approach: str, reps: int) -> dict:
    """{комиссия: (строка, портфель)}: Шарп и метрики, к2, к3, Δ к A, p ЛВ; условие 3 (к2) — и у B (Ж-009)."""
    out = {}
    for c, b in bases.items():
        r = C.evaluate(b, g, reps)
        p = b.eng.port(b.A * g, comm=c)
        ex = p["excess"].values
        r.update(msr=me.msr(ex), sortino=me.sortino_e(ex), **sr_stats(ex))
        out[c] = (r, p)
    return out


def run_approach(eng, approach, grid, bases, reps, log=print):
    """Строки точек (все ступени лестницы, обе комиссии) и портфели для лестницы."""
    rows, ports = [], {}
    for rung in LADDER:
        for h in ((1,) if rung in HLESS else (1, 5)):
            for meta, G, kind in PTS[approach](eng, rung, h, grid):
                meta = dict(approach=approach, rung=rung, sigma=SRC[rung][h], h=0 if rung in HLESS else h, **meta)
                cc = eng.calib_closed(G) if approach == "D" and kind == "mask" else np.nan
                meta.update(calib_closed=cc, screened_in=bool(np.isnan(cc) or SCREEN[0] <= cc <= SCREEN[1]))
                if not meta["screened_in"]:
                    rows += light_rows(eng, bases, G, meta)      # выпавшая по экрану — тоже проба (N)
                    continue
                g = eng.g_row(G)
                if kind == "mask":
                    meta.update(eng.gate_stats(g))
                for c, (r, p) in eval_point(bases, g, approach, reps).items():
                    rows.append(dict(meta, comm_pct=round(c * 100, 6), **r))
                    ports[(approach, c, meta["family"], meta["point"], rung, meta["h"])] = p
            log(f"{approach} {rung} h{h}: {len(rows)} строк")
    return pd.DataFrame(rows), ports


def lestnitsa(ports: dict, reps: int) -> pd.DataFrame:
    """Лестница σ: на каждой точке сетки ступень к предыдущей из доступных (константная → наивная → HAR-J → гибрид)."""
    rows, chains = [], set()
    for ap, c, fam, pt, rung, h in ports:
        chains |= {(ap, c, fam, pt, hh) for hh in ((1, 5) if h == 0 else (h,))}
    for ap, c, fam, pt, hh in sorted(chains):
        have = [r for r in LADDER if (ap, c, fam, pt, r, 0 if r in HLESS else hh) in ports]
        for prev, rung in zip(have, have[1:]):
            p1, p0 = (ports[(ap, c, fam, pt, r, 0 if r in HLESS else hh)] for r in (rung, prev))
            d, pv = shag0.delta_p(p1, p0, reps)
            rows.append(dict(approach=ap, comm_pct=c * 100, family=fam, point=pt, h=hh, ot=prev, k=rung,
                             sharpe_ot=s9.sharpe(p0["excess"].values), sharpe_k=s9.sharpe(p1["excess"].values),
                             d_sharpe=d, p_lw=pv, **C.znaki(p1, p0)))
    return pd.DataFrame(rows)


def head_lines(pometka, a_yaml, grid_paths, extra=()) -> list:
    from fastbt.cache import code_hash
    g = ", ".join(f"{Path(p).name} {sha(p)[:12]}" for p in grid_paths)
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return [pometka, f"код: {git.stdout.strip() or 'git?'} · fastbt {code_hash()[:12]}",
            f"входы: A-yaml {Path(a_yaml).name} {sha(a_yaml)[:12]}; сетки {g}; σ (SHA256.txt ch4_dev) {sha(s9.CH4 / 'SHA256.txt')[:12]}",
            f"период {TRAIN[0]}–{TRAIN[1]}; комиссии {', '.join(f'{c * 100:g}' for c in COMMS)} %; к2: {fk.N_SHIFT} сдвигов, "
            f"seed {fk.SEED}; ЛВ блок 5; Δ = разность табличных Шарпов (ddof=1)", *extra]


def write_table(df, path, head) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(f"# {ln}\n" for ln in head))
    df.to_csv(path, mode="a", sep=";", decimal=",", index=False, float_format="%.4f")
    return path


def run(a_yaml, out=OUT, reps=10000, n_shift=fk.N_SHIFT, workers=2, c_grid=None, pometka=HEAD, log=print) -> list:
    from fastbt import fast_grid, fast_signals
    from fastbt import shag_c_tab as T
    grids = {"B": SWEEP / "B_setka_e3.yaml", "C": Path(c_grid or C.GRID), "D": SWEEP / "D_setka_e3.yaml"}
    cfg = {k: C.load_grid(v) for k, v in grids.items()}      # load_grid — yaml.safe_load
    sd = [T.sdvig_line(T.sdvig(s9.CH4 / "proverka_sdvig.csv", r)) for r in LADDER[1:]]
    if any("НЕ ПРОЙДЕНА" in x for x in sd):
        raise SystemExit("\n".join(sd))
    paths, head = [], head_lines(pometka, a_yaml, list(grids.values()), [f"сдвиг σ: {sd[0].splitlines()[0]}"])
    with dev_train(a_yaml):
        s9.set_scope("dev")
        s9.use_v2(C.COMM_SEL)
        s9.C_DOP = {"rvw": cfg["C"]["rvw"]}
        s9.OUT.mkdir(parents=True, exist_ok=True)
        cache = fast_signals.build(workers)
        with fast_grid.patched():
            eng = s9.Engine(cache)
            ensure_const(eng)
            bases = {c: C.Base(eng, c, n_shift) for c in COMMS}
            for ap in "BCD":
                df, ports = run_approach(eng, ap, cfg[ap], bases, reps, log)
                df = add_n(df, C.load_grid(SWEEP / f"{ap}_setka_e3.yaml")["tochek_razlichnyh"])
                paths += [write_table(df, Path(out) / f"{ap}_tochki.csv", head),
                          write_table(lestnitsa(ports, reps), Path(out) / f"{ap}_lestnitsa.csv", head)]
    return paths


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="конвейер B/C/D (train 2018–2019)")
    ap.add_argument("a_yaml")
    for k, v in (("out", str(OUT)), ("reps", 10000), ("n-shift", fk.N_SHIFT), ("workers", 2), ("c-grid", None), ("pometka", HEAD)):
        ap.add_argument(f"--{k}", default=v, type=type(v) if v is not None else str)   # c-grid: C_setka_e2.yaml после T2
    a = ap.parse_args(argv)
    for p in run(a.a_yaml, a.out, a.reps, a.n_shift, a.workers, a.c_grid, a.pometka):
        print(p)


if __name__ == "__main__":
    main()
