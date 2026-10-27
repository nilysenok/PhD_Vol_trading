"""FS: быстрый прогон сетки надстроек С9 (D/B/C по σ) — тот же код таблиц `fastbt.s9`, горячие
места заменены на время прогона (`patched()`); старый путь не меняется и остаётся эталоном.

Что заменяется:
- `Engine.select` — выбор точки по валидации для всей семьи одним numba-проходом: ядро
  `grid_val_sharpe` считает Шарп A·g на окне валидации 2017…Y−1 для всех точек × всех пар-лет
  (prange по парам-годам; арифметика — ровно `approach_cd.masked_eval` / `scaled_eval`, поэтому
  Шарпы и выбор совпадают побитно); частичная экспозиция (вес 0,5) — вид «scale»;
- `report.boot_delta` → `fast_boot.boot_delta` (индексы повторов один раз, Шарпы повторов —
  numba, побитно; кэш по ряду: эталоны A и C считаются один раз на сетку);
- `s9_stats.lw_sharpe_diff` → `fast_boot.lw_sharpe_diff` (повторы — numba);
- `s9_stats.past_rank / past_median / ema` → `fast_sigma` (предрасчёт рангов σ для всех окон
  одним проходом, кэш по содержимому ряда; ранг — по окну без текущего дня);
- `s9.load_cache` → `fast_signals.build` (A и C основного дизайна — один раз на
  тикер × стратегию, кэш по хэшу входов и кода, параллель по тикерам).

python3 -m fastbt.fast_grid run [--boot-reps 1000] [--sigma-h1 final3_full] [--workers 4]
        [--threads 4] [--tables DIR] [--c-dop [yaml]]
Только --scope dev (2018–2021): σ после 31.12.2021 не читается (fastbt.s9.CUT).
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit, prange


# ======================= выбор по валидации: вся семья одним проходом =======================

@njit(cache=True, parallel=True)
def grid_val_sharpe(A, lr, job_t, job_c, job_vs, job_ve, fam, scale, comm, out):
    """A[J, N] — позиция A пары-года j (полный ряд тикера, хвост 0); lr[T, N]; fam[K, C, T, N] —
    затворы/веса (C = 1 или 3 категории); scale=0 — маска (g > 0,5), 1 — вес (A·g).
    out[J, K] — Шарп на [vs, ve) как masked_eval / scaled_eval (порядок операций тот же)."""
    J = A.shape[0]
    K = fam.shape[0]
    k252 = math.sqrt(252.0)
    for j in prange(J):
        ti = job_t[j]
        ci = job_c[j]
        vs = job_vs[j]
        ve = job_ve[j]
        n_win = ve - vs
        act = np.empty(n_win, np.int64)
        na = 0
        for t in range(vs, ve):
            if A[j, t] != 0.0 or (t > vs and A[j, t - 1] != 0.0):
                act[na] = t
                na += 1
        for m in range(K):
            S = 0.0
            Q = 0.0
            for jj in range(na):
                t = act[jj]
                if scale == 0:
                    q = A[j, t] if fam[m, ci, ti, t] > 0.5 else 0.0
                    p = 0.0
                    if t > vs and fam[m, ci, ti, t - 1] > 0.5:
                        p = A[j, t - 1]
                else:
                    q = A[j, t] * fam[m, ci, ti, t]
                    p = A[j, t - 1] * fam[m, ci, ti, t - 1] if t > vs else 0.0
                net = q * lr[ti, t] - abs(q - p) * comm
                if t == ve - 1:     # ОВ: выход на конце окна отбора (как masked_eval)
                    net -= abs(q) * comm
                S += net
                Q += net * net
            var = max(Q - S * S / n_win, 0.0) / (n_win - 1)
            sd = math.sqrt(var)
            out[j, m] = 0.0 if sd < 1e-12 else (S / n_win) / sd * k252


def _first_best(sh, ok=None):
    best, bi = -999.0, -1
    for i in range(len(sh)):
        if sh[i] > best and (ok is None or ok[i]):
            best, bi = sh[i], i
    return bi


def _engine_cls():
    from fastbt import s9

    class FastEngine(s9.Engine):
        """s9.Engine с пакетным выбором по валидации (результат тот же)."""

        def __init__(self, cache):
            super().__init__(cache)
            # Ъ ~23:20: ставка «back» ((d_prev, d], rate_restored) — только печати v1;
            # быстрый слой — ставка ЦБ и дни [d, d_next)
            from fastbt import msp
            assert msp.rate_cbr().attrs.get("cal") != "back", "быстрый слой: ставка «back» запрещена"
            assert np.array_equal(self.cal, msp.cal_days(self.dates, forward=True)), \
                "быстрый слой: дни ставки не [d, d_next)"

        def _jobs(self):
            if getattr(self, "_fs_jobs", None) is None:
                jobs, Arows = [], []
                for pid, (t, sid) in enumerate(self.pairs):
                    ti = self.tickers.index(t)
                    c = self.cache[t]
                    for Y in s9.YEARS:
                        if Y not in c["A"][sid]:
                            continue
                        vs, ve = self.vwin[(ti, Y)]
                        jobs.append((pid, Y, ti, s9.CATS[sid], vs, ve))
                        a = np.zeros(self.nmax)
                        x = c["A"][sid][Y]
                        a[:len(x)] = x
                        Arows.append(a)
                self._fs_jobs = jobs
                self._fs_A = np.ascontiguousarray(np.stack(Arows))
                self._fs_lr = np.ascontiguousarray(self.lr)
            return self._fs_jobs

        def select(self, fam, kind="mask", screen=None, cat=False):
            jobs = self._jobs()
            K = fam.shape[0]
            F = fam if cat else fam[:, None]
            F = np.ascontiguousarray(F, np.float64)
            if F.shape[-1] < self.nmax:
                F = np.concatenate([F, np.ones(F.shape[:-1] + (self.nmax - F.shape[-1],))], -1)
            jt = np.array([j[2] for j in jobs], np.int64)
            jc = np.array([j[3] if cat else 0 for j in jobs], np.int64)
            jvs = np.array([j[4] for j in jobs], np.int64)
            jve = np.array([j[5] for j in jobs], np.int64)
            sh = np.zeros((len(jobs), K))
            grid_val_sharpe(self._fs_A, self._fs_lr, jt, jc, jvs, jve, F,
                            0 if kind == "mask" else 1, s9.COMM, sh)
            choice, cnt, cf_cache = {}, {}, {}
            for j, (pid, Y, ti, ci, vs, ve) in enumerate(jobs):
                ok = None
                if screen is not None:
                    key = (ti, ci if cat else 0, vs, ve)
                    if key not in cf_cache:
                        n = self.n_t[ti]
                        rows = fam[:, ci, ti, :n] if cat else fam[:, ti, :n]
                        cf_cache[key] = 1.0 - rows[:, vs:ve].mean(axis=1)
                    cf = cf_cache[key]
                    ok = (cf >= screen[0]) & (cf <= screen[1])
                    if not ok.any():
                        dist = np.maximum(screen[0] - cf, cf - screen[1])
                        ok = dist <= dist.min() + 1e-12
                k = _first_best(sh[j], ok)
                k = 0 if k < 0 else k
                choice[(pid, Y)] = k
                cnt[k] = cnt.get(k, 0) + 1
            g = np.ones(len(self.r["bar"]))
            pos = 0
            for pid, (t, sid) in enumerate(self.pairs):
                ti = self.tickers.index(t)
                for Y in s9.YEARS:
                    if (pid, Y) not in choice:
                        continue
                    ys, ye = self._slice(t, Y)
                    k = choice[(pid, Y)]
                    src = fam[k, s9.CATS[sid], ti] if cat else fam[k, ti]
                    g[pos:pos + ye - ys] = src[ys:ye]
                    pos += ye - ys
            return g, cnt

    return FastEngine


# ======================= подмена горячих мест на время прогона =======================

@contextlib.contextmanager
def patched(cache_loader=None):
    """Подменяет горячие функции старого пути быстрыми (результат тот же) и возвращает всё назад."""
    from fastbt import fast_boot, fast_sigma, report, s9, s9_stats
    saved = [(s9, "Engine", s9.Engine), (s9, "load_cache", s9.load_cache),
             (report, "boot_delta", report.boot_delta),
             (s9_stats, "lw_sharpe_diff", s9_stats.lw_sharpe_diff),
             (s9_stats, "past_rank", s9_stats.past_rank),
             (s9_stats, "past_median", s9_stats.past_median), (s9_stats, "ema", s9_stats.ema)]
    s9_stats._fs_orig = dict(past_rank=s9_stats.past_rank, past_median=s9_stats.past_median,
                             ema=s9_stats.ema)
    try:
        s9.Engine = _engine_cls()
        if cache_loader is not None:
            s9.load_cache = cache_loader
        report.boot_delta = fast_boot.boot_delta
        s9_stats.lw_sharpe_diff = fast_boot.lw_sharpe_diff
        s9_stats.past_rank = fast_sigma.past_rank
        s9_stats.past_median = fast_sigma.past_median
        s9_stats.ema = fast_sigma.ema
        yield
    finally:
        for mod, name, v in saved:
            setattr(mod, name, v)
        s9_stats.__dict__.pop("_fs_orig", None)


def run(reps: int, cache: dict, timings: dict | None = None) -> None:
    """Все эксперименты С9 (s9.run_all_experiments) на быстром пути; cache — словарь кэша С9."""
    from fastbt import fast_sigma, s9
    t0 = time.time()
    with patched(cache_loader=lambda: cache):
        eng = s9.Engine(cache)
        labs = [lab for lab in s9.SOURCES if lab in ("hyb_h1", "hyb_h5", "harj_h1", "naive_h1")]
        fast_sigma.precompute(eng, labs)
        if timings is not None:
            timings["sigma_precompute"] = time.time() - t0
        t1 = time.time()
        s9.run_all_experiments(reps)
        if timings is not None:
            timings["run"] = time.time() - t1


BENCH_SRCS = ("hyb_h1", "hyb_h5", "harj_h1", "naive_h1")


def bench(reps: int, cache: dict, fast: bool, srcs=BENCH_SRCS, lw: bool = True) -> dict:
    """Замер сетки в сотни конфигураций: D, B, C (с доп. точками, если загружены) на σ-источниках
    srcs; каждое Δ (к A, к C) — IID-бутстрап reps и, при lw, Ледуа–Вольф (блок 5, reps), как в
    раунде владельца. fast=False — старый путь, True — быстрый. Время — по слоям."""
    from fastbt import fast_sigma, s9, s9_stats
    tm = {k: 0.0 for k in ("sigma", "select", "port_metrics", "boot_iid", "boot_lw", "всего")}
    cnt = {"points": 0, "deltas": 0}
    ctx = patched() if fast else contextlib.nullcontext()
    t_all = time.time()
    with ctx:
        eng = s9.Engine(cache)
        x = s9.S9(eng, reps)
        if fast:
            t = time.time()
            fast_sigma.precompute(eng, srcs)
            tm["sigma"] += time.time() - t

        def timed(obj, name, key):
            f = getattr(obj, name)

            def w(*a, **k):
                t = time.time()
                try:
                    return f(*a, **k)
                finally:
                    tm[key] += time.time() - t
            setattr(obj, name, w)
            return f
        timed(eng, "select", "select")
        for nm in ("d_points", "c_points", "b_weights"):
            timed(x, nm, "sigma")
        boot0 = timed(x, "boot", "boot_iid")
        lwf = s9_stats.lw_sharpe_diff
        delta0 = x.delta

        def delta(r, key, p1, p0):
            delta0(r, key, p1, p0)
            cnt["deltas"] += 1
            if lw:
                t = time.time()
                r[f"p_lw_{key}"] = lwf(p1["excess"].values, p0["excess"].values, reps, 5)[2]
                tm["boot_lw"] += time.time() - t
        x.delta = delta
        dfs = []
        for src in srcs:
            dfs.append(x.run_d((src,))[0])
            dfs.append(x.run_b((src,))[0])
            dfs.append(x.run_c((src,))[0])
        del boot0
    tm["всего"] = time.time() - t_all
    tm["port_metrics"] = tm["всего"] - tm["sigma"] - tm["select"] - tm["boot_iid"] - tm["boot_lw"]
    import pandas as pd
    df = pd.concat(dfs, ignore_index=True)
    cnt["points"] = int(df["d_vs_A"].notna().sum())      # конфигурации с Δ к A
    return dict(tm, **cnt, frame=df)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["cache", "run", "all", "bench"])
    p.add_argument("--bench-path", choices=["old", "new", "both"], default="both")
    p.add_argument("--no-lw", action="store_true", help="bench: без Ледуа–Вольфа на каждое Δ")
    p.add_argument("--boot-reps", type=int, default=1000)
    p.add_argument("--sigma-h1", default="final3_full")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tables", default=None, help="папка таблиц (по умолчанию как у s9)")
    p.add_argument("--out", default=None, help="папка журнала run_log (по умолчанию как у s9)")
    p.add_argument("--c-dop", nargs="?", const="", default=None)
    # К (Ъ 27.10, 00:10): v2 и --c8v2 на быстром слое — как у fastbt.s9 run (без флагов — как было)
    p.add_argument("--sigma-set", choices=["old", "v2"], default="old")
    p.add_argument("--comm", type=float, default=None)
    p.add_argument("--c8v2", action="store_true")
    p.add_argument("--s9-cache", action="store_true",
                   help="кэш A/C из data/fastbt/s9 (s9.load_cache) вместо fast_signals.build")
    a = p.parse_args(argv)
    from fastbt import fast_signals, s9
    s9.set_scope("dev")
    if a.c8v2:
        s9.set_c8v2("dev", a.sigma_set, a.c_dop is not None)
    if a.sigma_set == "v2":
        s9.use_v2(a.comm)
    else:
        s9.use_sigma_h1(a.sigma_h1)
    if a.c_dop is not None:
        s9.load_c_dop(a.c_dop or None)
    if a.tables:
        s9.TABLES = Path(a.tables)
    # журнал run_log — не в папку С9 (её кэш и журнал не трогаем)
    from fastbt.cache import MAIN
    t0 = time.time()
    cache = s9.load_cache() if a.s9_cache else None
    s9.OUT = Path(a.out) if a.out else MAIN / "data" / "fastbt" / "fs"
    s9.OUT.mkdir(parents=True, exist_ok=True)
    if cache is None:
        cache = fast_signals.build(a.workers)
    print(f"cache (быстрый слой): {time.time() - t0:.1f} с", flush=True)
    if a.stage in ("run", "all"):
        tm = {}
        run(a.boot_reps, cache, tm)
        print(f"run (быстрый слой): σ-признаки {tm['sigma_precompute']:.1f} с, "
              f"таблицы {tm['run']:.1f} с", flush=True)
    if a.stage == "bench":
        import json
        res = {}
        paths = ("old", "new") if a.bench_path == "both" else (a.bench_path,)
        frames = {}
        for pth in paths:
            r = bench(a.boot_reps, cache, pth == "new", lw=not a.no_lw)
            frames[pth] = r.pop("frame")
            res[pth] = r
            print(pth, json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                   for k, v in r.items()}, ensure_ascii=False), flush=True)
            frames[pth].to_csv(s9.OUT / f"bench_{pth}.csv", sep=";", index=False,
                               float_format="%.10g")
        if len(frames) == 2:
            fo, fn = frames["old"], frames["new"]
            num = fo.select_dtypes("number").columns
            dmax = float(np.nanmax(np.abs(fo[num].values - fn[num].values))) if len(num) else 0.0
            print(f"bench: строк {len(fo)}, max|Δ| числовых столбцов старый/новый = {dmax:.3g}")
            res["max_abs_diff"] = dmax
        (s9.OUT / "bench.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    if "--threads" in sys.argv:
        os.environ.setdefault("NUMBA_NUM_THREADS", sys.argv[sys.argv.index("--threads") + 1])
    sys.exit(main())
