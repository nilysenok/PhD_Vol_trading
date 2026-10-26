"""FS: сигналы и позиции A / C основного дизайна для С9 — один раз на тикер × стратегию,
кэш по хэшу входов и кода, параллель по тикерам.

Повторяет `fastbt.s9._job` (кэш-стадию С9), но кэширует по частям:
- A[тикер, стратегия] (ансамбль A после фильтра оборота и до него) — ключ: цены тикера
  (`TickerData.digest`), yaml основного дизайна, область, код fastbt (без модулей таблиц и
  быстрого слоя). От σ A не зависит, поэтому новый σ-источник A не пересчитывает;
- C[тикер, стратегия, σ-источник] — ключ: ключ A + горизонт + байты выровненной σ.
Части лежат в памяти процесса и на диске (`<основная копия>/data/fastbt/fs_cache/`,
FASTBT_FS_CACHE — другая папка, FASTBT_NO_CACHE=1 — без диска). Расчёт — теми же функциями,
что `s9._job` (ApproachA, turnover.apply, approach_cd.run_c), поэтому кэш С9 совпадает побайтно.
Если `s9._job` изменился (хэш исходника не равен `JOB_SHA`), слой не гадает: считает
по-тикерно самим `s9._job` (тоже параллельно) и кэширует тикер целиком.
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import os
import pickle
import time
from pathlib import Path

import numpy as np

# sha1 исходника s9._job, под который написан _job_parts (иначе — запасной путь)
JOB_SHA = "e5d8c1700a6c"
_EXCLUDE = {"s9.py", "s9_show.py", "final_test.py", "report.py", "equiv.py", "bench.py",
            "export.py", "dev_rank.py", "decomp.py", "chapter4_a.py"}
_MEM: dict = {}
_CODE = None


def _cache_dir() -> Path:
    from fastbt.cache import MAIN
    return Path(os.environ.get("FASTBT_FS_CACHE", MAIN / "data" / "fastbt" / "fs_cache"))


def code_hash() -> str:
    """Хэш исходников, от которых зависят A и C (fastbt/**, кроме тестов, таблиц С9/теста и
    модулей fast_*), плюс исходники s9._job и s9.align."""
    global _CODE
    if _CODE is None:
        from fastbt import s9
        h = hashlib.sha1()
        base = Path(__file__).resolve().parent
        for p in sorted(base.rglob("*.py")):
            if "tests" in p.parts or p.name in _EXCLUDE or p.name.startswith("fast_"):
                continue
            h.update(p.name.encode() + p.read_bytes())
        h.update(inspect.getsource(s9._job).encode())
        h.update(inspect.getsource(s9.align).encode())
        h.update(inspect.getsource(_job_parts).encode())
        _CODE = h.hexdigest()
    return _CODE


def job_sha() -> str:
    from fastbt import s9
    return hashlib.sha1(inspect.getsource(s9._job).encode()).hexdigest()[:12]


def _key(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
        h.update(b"|")
    return h.hexdigest()[:24]


def _get(k):
    if k in _MEM:
        return _MEM[k]
    p = _cache_dir() / f"{k}.pkl"
    if os.environ.get("FASTBT_NO_CACHE", "") != "1" and p.exists():
        _MEM[k] = pickle.loads(p.read_bytes())
        return _MEM[k]
    return None


def _put(k, v):
    _MEM[k] = v
    if os.environ.get("FASTBT_NO_CACHE", "") == "1":
        return
    d = _cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{k}.tmp{os.getpid()}"
    tmp.write_bytes(pickle.dumps(v, protocol=4))
    os.replace(tmp, d / f"{k}.pkl")


def _ctx():
    from fastbt import s9
    return (s9.SCOPE, s9.VAL_START, s9.VAL_END, str(s9.CUT.date()), tuple(s9.YEARS),
            s9.MAIN_YAML.read_bytes())


def _job_parts(args):
    """Тот же расчёт, что s9._job, с кэшем частей: A по (тикер, стратегия), C по (…, σ)."""
    ticker, srcs, code = args[:3]
    os.environ["NUMBA_NUM_THREADS"] = "1"
    from fastbt import dev, masks, s9, turnover
    from fastbt import design as dsg
    from fastbt.approach_a import ApproachA
    from fastbt.approach_cd import run_c
    from fastbt.data import load_ticker
    from fastbt.unit import make_unit
    if len(args) > 3:
        s9.set_scope(*args[3])
    d, _ = dev.load_design(s9.MAIN_YAML)
    cfg = dev.make_config(d, s9.SCOPE)
    cfg5 = dev.make_config({**d, "horizons": ["h5"]}, s9.SCOPE)
    if s9.SCOPE != "dev":
        cfg = dataclasses.replace(cfg, val_start=s9.VAL_START, val_end=s9.VAL_END)
        cfg5 = dataclasses.replace(cfg5, val_start=s9.VAL_START, val_end=s9.VAL_END)
    dsg.apply(cfg.design)
    td = load_ticker(ticker, cfg.corp_actions)
    sig = {lab: (h, s9.align(td.dates, df)) for lab, (h, df) in srcs.items()}
    out = {"dates": td.dates, "lr": td.log_ret, "A": {}, "C": {}, "Cmeta": {}}
    ctx = _ctx()
    stat = {"A_hit": 0, "A_miss": 0, "C_hit": 0, "C_miss": 0}
    for sid in s9.SIDS:
        ka = _key("A", ticker, sid, td.digest(), code, *ctx)
        pa = _get(ka)
        u = None
        if pa is None:
            stat["A_miss"] += 1
            dsg.apply(cfg.design)
            u = make_unit(ticker, sid, td)
            a = ApproachA(u, cfg).run()
            a_tf = turnover.apply(a, u, cfg, "A")
            pa = {"A": {Y: v[1].astype(np.float64) for Y, v in a_tf.items()},
                  "A0": {Y: np.asarray(v[1], np.float64) for Y, v in a.items()}}
            _put(ka, pa)
        else:
            stat["A_hit"] += 1
        out["A"][sid] = pa["A"]
        if s9.SCOPE != "dev":
            out.setdefault("A0", {})[sid] = pa["A0"]
        a = {Y: (None, v) for Y, v in pa["A0"].items()}
        for lab, (h, arr) in sig.items():
            kc = _key("C", ka, lab, h, hashlib.sha1(np.ascontiguousarray(arr).tobytes()).digest())
            pc = _get(kc)
            if pc is None:
                stat["C_miss"] += 1
                if u is None:
                    dsg.apply(cfg.design)
                    u = make_unit(ticker, sid, td)
                c_cfg = cfg if h == "h1" else cfg5
                dsg.apply(c_cfg.design)
                sg = {k: np.full(td.n, np.nan) for k in ("h1", "h5", "h22")}
                sg[h] = arr
                u.td = dataclasses.replace(td, sigma=sg)
                masks._CACHE.clear()               # ключ кэша масок не включает σ
                c = turnover.apply(run_c(u, c_cfg, a), u, c_cfg, "C")
                pc = {"C": {Y: v[1].astype(np.float64) for Y, v in c.items() if Y in s9.YEARS},
                      "meta": {Y: str(v[0]) for Y, v in c.items() if Y in s9.YEARS}}
                _put(kc, pc)
                u.td = td
                masks._CACHE.clear()
            else:
                stat["C_hit"] += 1
            out["C"].setdefault(lab, {})[sid] = pc["C"]
            out["Cmeta"].setdefault(lab, {})[sid] = pc["meta"]
    return ticker, out, stat


def _job_whole(args):
    """Запасной путь: s9._job как есть, кэш тикера целиком."""
    from fastbt import s9
    ticker, srcs, code = args[:3]
    if len(args) > 3:
        s9.set_scope(*args[3])
    sh = b"".join(hashlib.sha1(np.ascontiguousarray(df["sigma"].values).tobytes()).digest()
                  + lab.encode() for lab, (h, df) in sorted(srcs.items()))
    k = _key("T", ticker, code, sh, *_ctx())
    v = _get(k)
    if v is None:
        _, v = s9._job((ticker, srcs) + tuple(args[3:]))
        _put(k, v)
        return ticker, v, {"T_miss": 1}
    return ticker, v, {"T_hit": 1}


def build(workers: int = 4, tickers=None, log=print) -> dict:
    """Словарь кэша С9 {тикер: {...}} (как s9.build_cache → load_cache)."""
    from fastbt import s9
    from fastbt.data import TICKERS
    os.environ["FASTBT_VPRED"] = str(s9.SOURCES["hyb_h1"][0])
    tickers = list(tickers or TICKERS)
    srcs = {lab: (s9.SOURCES[lab][2], s9.load_source(lab)) for lab in s9.SOURCES}
    code = code_hash()
    fn = _job_parts if job_sha() == JOB_SHA else _job_whole
    if fn is _job_whole:
        log(f"fast_signals: s9._job изменился ({job_sha()} ≠ {JOB_SHA}) — запасной путь")
    extra = () if s9.SCOPE == "dev" else ((s9.SCOPE, s9.VAL_START),)
    t0 = time.time()
    res, tot, jobs, tkey = {}, {}, [], {}
    for t in tickers:
        st = {lab: (h, df[df["ticker"] == t]) for lab, (h, df) in srcs.items()}
        sh = b"".join(lab.encode() + h.encode() +
                      hashlib.sha1(np.ascontiguousarray(df["sigma"].values).tobytes()).digest()
                      for lab, (h, df) in sorted(st.items()))
        tkey[t] = _key("mem", t, code, sh, fn.__name__, *_ctx())
        if tkey[t] in _MEM:                       # тикер уже в памяти процесса
            res[t] = _MEM[tkey[t]]
            tot["mem_hit"] = tot.get("mem_hit", 0) + 1
            continue
        jobs.append((t, st, code) + extra)
    if not jobs:
        it, pool = iter(()), None
    elif workers <= 1:
        it = map(fn, jobs)
        pool = None
    else:
        import multiprocessing as mp
        os.environ["NUMBA_NUM_THREADS"] = "1"
        pool = mp.get_context("spawn").Pool(min(workers, len(jobs)))
        it = pool.imap_unordered(fn, jobs)
    for tk, out, st in it:
        res[tk] = out
        _MEM[tkey[tk]] = out
        for k, v in st.items():
            tot[k] = tot.get(k, 0) + v
    if pool is not None:
        pool.close()
        pool.join()
    log(f"fast_signals: {len(res)} тикеров за {time.time() - t0:.1f} с, {tot}")
    return {t: res[t] for t in tickers}
