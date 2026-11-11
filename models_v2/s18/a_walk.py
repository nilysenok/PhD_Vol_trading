"""S-A-IMPL-3 (Ж-254; `S_A_SPEC.md`, разд. 5, 8, 9): помесячный проход вариантов A0 / A1 / A2 / A2p, слои X1 и W5-G над C_A, запись `data/regen/harj_struct/A/{a0,a1,a2,a2p}.parquet`.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Протокол: расширяющееся окно закрытых пар (t⁺ ≤ D), переоценка в первый торговый день месяца D (106 месяцев 2018-01 … 2026-11 по датам D_M: в D_M нет 2022-03; спецификация считала 107);
θ_H — при ≥ 250 закрытых парах (иначе нулевая точка = вложение); валидация — последние 126 закрытых пар, обучение — остальные; λ ∈ {1e-4, 1e-3, 1e-2} по QLIKE валидации ансамбля seeds (ничья — больший λ); затем подгонка на всех закрытых парах;
корректор (δw, δκ) обучается, если в обучающем пуле ≥ 120 пар с cfg8; seeds 0–4, итог — exp(среднее ln по seeds) отдельно для H_A и C_A. A0: ω, ψ, c не свободны, все seeds побитно одинаковы — считается seed 0.
Публичное API:
- константы `LAMS`, `VAL = 126`, `MIN_PAIRS = 250`, `MIN_CFG = 120`, `SEEDS`, `IDS` (вариант → id реестра), `OUTS` (вариант → имя parquet), `NAMES` (28 имён параметров), `A_DIR`.
- `closed_rows` (из `s13/v6_1_wf`): закрытые пары на D — t⁺ ≤ D (в пуле обучения и валидации); `refit_positions` — первые торговые дни месяцев.
- `fit_month(D, pool, days, var, seeds=SEEDS) -> dict(lam, n_train, corr, theta (28, среднее по seeds), lnH, lnC)`: переоценка на пуле (позиции пар в `D`) и прогноз для дней `days` (позиции в `D`).
- `walk_variant(var, D, x, positions=None, seeds=SEEDS, work=None) -> list[(a, b, dict)]`: все месяцы (или `positions` — позиции первых дней месяцев в окне M x); `work` — каталог возобновления (pickle по месяцу).
- `assemble(var, x, gap, parts) -> DataFrame`: колонки d, t_plus, rv, K1, H_A, C_A, X1_A, F, pi, beta_gap, refit_date, lam, n_train, corr, затем 28 колонок `NAMES` (среднее θ по seeds).
- `run(var, write=True) -> DataFrame`: отказ (SystemExit), если id нет в реестре или `a_reg.check()` не пуст; существующий выход с иным содержимым — отказ. Проверки — `a_walk_chk.py`.
Запуск: `python3 -W ignore -m models_v2.s18.a_walk --run A0|A1|A2|A2p | --check | --check-pit`."""
from __future__ import annotations

import hashlib
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models_v2.s13.v6_1_wf import closed_rows, refit_positions
from models_v2.s18 import a_feat as A
from models_v2.s18 import a_layers as L
from models_v2.s18 import a_model as M
from models_v2.s18 import a_reg as R

LAMS, VAL, MIN_PAIRS, MIN_CFG, SEEDS, FIRST = (1e-4, 1e-3, 1e-2), 126, 250, 120, (0, 1, 2, 3, 4), "2018-01-01"
IDS = {"A0": "S-A0", "A1": "S-A1", "A2": "S-A2", "A2p": "S-A2p"}
OUTS = {v: v.lower() + ".parquet" for v in IDS}
A_DIR = R.A_DIR
NAMES = (["d0", "d_s", "d_m", "d_l", "d_J", "om_s", "om_m", "om_l"] + ["psi_%s%d" % (k, i) for k in "sml" for i in (1, 2)] + ["c_%s%d" % (k, i) for k in "sml" for i in (1, 2, 3, 4)] + ["dw", "dk"])
assert len(NAMES) == M.NP


def n_cfg(D, sel) -> int:
    return int(D.has8[sel].sum())


def choose_lambda(D, pool, var: str, seeds) -> float:
    corr, Vt, Vv = n_cfg(D, pool[:-VAL]) >= MIN_CFG, D.view(pool[:-VAL]), D.view(pool[-VAL:])
    best = None
    for lam in LAMS:
        ln = np.mean([M.forward(torch.as_tensor(M.fit(Vt, var, lam, s, corr)), Vv, var)[1].numpy() for s in seeds], axis=0)
        q = float(M.qlike(Vv.rv, torch.as_tensor(ln)).mean())
        if best is None or q <= best[0]:
            best = (q, lam)
    return best[1]


def fit_month(D, pool, days, var: str, seeds=SEEDS) -> dict:
    seeds = (0,) if var == "A0" else tuple(seeds)
    pool = np.asarray(pool)
    if len(pool) < MIN_PAIRS:
        lam, corr, th = float("nan"), False, [np.zeros(M.NP)]
    else:
        lam, corr = choose_lambda(D, pool, var, seeds), n_cfg(D, pool) >= MIN_CFG
        V = D.view(pool)
        th = [M.fit(V, var, lam, s, corr) for s in seeds]
    Vd = D.view(days)
    fw = [M.forward(torch.as_tensor(t), Vd, var) for t in th]
    return dict(lam=lam, n_train=len(pool), corr=corr, theta=np.mean(th, axis=0), lnH=np.mean([f[0].numpy() for f in fw], axis=0), lnC=np.mean([f[1].numpy() for f in fw], axis=0))


def _day_pos(D, x) -> np.ndarray:
    p = pd.Series(np.arange(len(D.d)), index=pd.DatetimeIndex(D.d)).reindex(pd.DatetimeIndex(x["d"])).to_numpy()
    if np.isnan(p).any():
        raise SystemExit("дата окна M не в парах K1")
    return p.astype(int)


def code_key() -> str:
    h = hashlib.sha256()
    for n in ("a_feat", "a_model", "a_walk", "a_layers"):
        h.update((Path(__file__).parent / (n + ".py")).read_bytes())
    return h.hexdigest()[:12]


def walk_variant(var: str, D, x, positions=None, seeds=SEEDS, work=None) -> list:
    xd, dp = pd.DatetimeIndex(x["d"]), _day_pos(D, x)
    pos = refit_positions(xd, FIRST)
    out = []
    for a, b in zip(pos, pos[1:] + [len(xd)]):
        if positions is not None and a not in positions:
            continue
        f = None if work is None else Path(work) / ("m%03d.pkl" % a)
        if f is not None and f.exists():
            out.append((a, b, pickle.loads(f.read_bytes())))
            continue
        t0 = time.time()
        res = fit_month(D, closed_rows(D.tp, xd[a]), dp[a:b], var, seeds)
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(pickle.dumps(res))
        print("%s %s: пар %d, λ %s, %.0f с" % (var, xd[a].date(), res["n_train"], res["lam"], time.time() - t0), flush=True)
        out.append((a, b, res))
    return out


def assemble(var: str, x, gap, parts) -> pd.DataFrame:
    n = len(x)
    cols = dict(lnH=np.full(n, np.nan), lnC=np.full(n, np.nan), lam=np.full(n, np.nan), n_train=np.zeros(n, int), corr=np.zeros(n, bool))
    th, rd = np.full((n, M.NP), np.nan), np.full(n, np.datetime64("NaT"), "datetime64[ns]")
    for a, b, r in parts:
        for k in cols:
            cols[k][a:b] = r[k]
        th[a:b], rd[a:b] = r["theta"], np.datetime64(x["d"].iloc[a])
    lay = L.layers(x, np.exp(cols["lnC"]), gap)
    df = pd.DataFrame(dict(d=x["d"].to_numpy(), t_plus=x["t_plus"].to_numpy(), rv=x["rv"].to_numpy(float), K1=x["K1"].to_numpy(float), H_A=np.exp(cols["lnH"]), C_A=np.exp(cols["lnC"]),
                           X1_A=lay["X1_A"], F=lay["F"], pi=lay["pi"], beta_gap=lay["beta_gap"], refit_date=rd, lam=cols["lam"], n_train=cols["n_train"], corr=cols["corr"]))
    return pd.concat([df, pd.DataFrame(th, columns=NAMES)], axis=1)


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_exact=True)
        return True
    except AssertionError:
        return False


def run(var: str, write: bool = True, out: Path = A_DIR) -> pd.DataFrame:
    R.require(IDS[var])
    bad = R.check()
    if bad:
        raise SystemExit("реестр S-A не готов: %s" % "; ".join(bad))
    x, gap = L.inputs()
    D = M.load_data()
    parts = walk_variant(var, D, x, work=out / "_part" / ("%s_%s" % (var, code_key())))
    df = assemble(var, x, gap, parts)
    p = out / OUTS[var]
    if write:
        if p.exists() and not _same(pd.read_parquet(p), df):
            raise SystemExit("%s: существует и отличается от пересчёта — перезапись отказана" % p)
        tmp = out / (OUTS[var] + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(p)
    return df


if __name__ == "__main__":
    if "--run" in sys.argv:
        v = sys.argv[sys.argv.index("--run") + 1]
        t0 = time.time()
        d = run(v)
        print("%s: записано %d строк, %.0f с" % (v, len(d), time.time() - t0))
    elif "--check" in sys.argv or "--check-pit" in sys.argv:
        from models_v2.s18 import a_walk_chk as K
        b = K.check() if "--check" in sys.argv else K.check_pit()
        print("%s: %s" % ("--check" if "--check" in sys.argv else "--check-pit", "ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        raise SystemExit("укажите --run VAR, --check или --check-pit")
