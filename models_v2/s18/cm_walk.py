"""S-C-IMPL-3 (Ж-254; `S_C_SPEC.md`, разд. 3, 6, 7): помесячный проход вариантов CD / CK / CKA / C7 / CKL, слои X1 и W5-G над C_V, запись `data/regen/harj_struct/CM/{cd,ck,cka,c7,ckl}.parquet`.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Протокол: расширяющееся окно закрытых пар (t⁺ ≤ D, `x1_ya_walk.closed_pool`), переоценка в первый торговый день месяца D по датам D_M (106 месяцев; в D_M нет 2022-03);
включение — `x1_ya_walk.ACTIVATION` = 2018-07-02 (≥ 120 закрытых пар): до него 𝓜 = 1, C_V = cfg8 побитно (в том числе у C7); пока закрытых пар < 252 — λ = 1e-2, без валидации;
иначе последние 126 закрытых пар — валидация, λ ∈ {1e-4, 1e-3, 1e-2} по QLIKE суммы (ничья — больший λ), затем подгонка на всех закрытых парах, 5 перезапусков (итог — минимум штрафованной L). C7 не обучается.
Публичное API:
- `VAL = 126`, `MIN_VALID = 252`, `LAM_DEF = 1e-2`, `IDS` (вариант → id реестра), `OUTS` (вариант → имя parquet), `CM_DIR`, `FIRST` (первый месяц D_M).
- `choose_lambda(D, pool, var, seeds) -> float`; `fit_month(D, pool, days, var, seeds=SEEDS) -> dict(lam, n_train, seed, theta (6), spread, M, Fc, Fp, Fu)` (pool, days — позиции пар в `Data` D).
- `walk_variant(var, D, x, positions=None, seeds=SEEDS, work=None) -> list[(a, b, dict)]` (все 106 месяцев; до включения — нулевые параметры); `work` — каталог возобновления (pickle по месяцу).
- `assemble(var, x, gap, parts) -> DataFrame`: d, t_plus, rv, cfg8, M, C, F_c, F_p, F_u, X1_V, F, pi, beta_gap, refit_date, active, lam, n_train, seed_best, spread, затем 6 параметров `cm_model.NAMES[var]` (у C7 — без них).
- `run(var, write=True, out=CM_DIR) -> DataFrame`: отказ (SystemExit), если id нет в реестре или `cm_reg.check()` не пуст; существующий выход с иным содержимым — отказ. Проверки — `cm_walk_chk.py`.
Запуск: `python3 -W ignore -m models_v2.s18.cm_walk --run CD|CK|CKA|C7|CKL | --check | --check-pit`."""
from __future__ import annotations

import hashlib
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models_v2.s16 import x1_ya_walk as YA
from models_v2.s18 import cm_feat as CF
from models_v2.s18 import cm_layers as L
from models_v2.s18 import cm_model as M
from models_v2.s18 import cm_reg as R

torch.set_num_threads(1)
VAL, MIN_VALID, LAM_DEF = 126, 252, 1e-2
IDS = {"CD": "S-CD", "CK": "S-CK", "CKA": "S-CKA", "C7": "S-C7", "CKL": "S-CKL"}
OUTS = {v: v.lower() + ".parquet" for v in IDS}
CM_DIR = R.CM_DIR
ZERO = torch.zeros(M.NP, dtype=torch.float64)


def choose_lambda(D, pool, var: str, seeds=M.SEEDS) -> float:
    if len(pool) < MIN_VALID:
        return LAM_DEF
    Vt, Vv, best = D.view(pool[:-VAL]), D.view(pool[-VAL:]), None
    for lam in M.LAMS:
        th = torch.as_tensor(M.fit(Vt, var, lam, M.MU[var], seeds)["theta"])
        q = float(M.qlike_sum(th, Vv, var))
        if best is None or q <= best[0]:
            best = (q, lam)
    return best[1]


def _res(f: dict, **kw) -> dict:
    return dict(**kw, **{k: f[k].numpy() for k in ("M", "Fc", "Fp", "Fu")})


def zero_month(D, days) -> dict:
    nan = float("nan")
    return _res(M.forward(ZERO, D.view(days), "CK"), lam=nan, n_train=0, seed=nan, theta=np.zeros(M.NP), spread=nan)


def fit_month(D, pool, days, var: str, seeds=M.SEEDS) -> dict:
    V, pool, nan = D.view(days), np.asarray(pool), float("nan")
    if var == "C7":
        return _res(M.forward(ZERO, V, "C7"), lam=nan, n_train=len(pool), seed=nan, theta=np.full(M.NP, nan), spread=nan)
    lam = choose_lambda(D, pool, var, seeds)
    r = M.fit(D.view(pool), var, lam, M.MU[var], seeds)
    return _res(M.forward(torch.as_tensor(r["theta"]), V, var), lam=lam, n_train=len(pool), seed=r["seed"], theta=r["theta"], spread=float(np.ptp(r["losses"])))


def code_key() -> str:
    h = hashlib.sha256()
    for n in ("cm_feat", "cm_model", "cm_walk", "cm_layers"):
        h.update((Path(__file__).parent / (n + ".py")).read_bytes())
    return h.hexdigest()[:12]


def walk_variant(var: str, D, x, positions=None, seeds=M.SEEDS, work=None) -> list:
    xd, tp = pd.DatetimeIndex(x["d"]), pd.DatetimeIndex(x["t_plus"])
    YA.check_activation(xd, tp)
    pos, out = YA.month_starts(xd), []
    for a, b in zip(pos, pos[1:] + [len(xd)]):
        if positions is not None and a not in positions:
            continue
        f = None if work is None else Path(work) / ("m%03d.pkl" % a)
        if f is not None and f.exists():
            out.append((a, b, pickle.loads(f.read_bytes())))
            continue
        t0, days = time.time(), np.arange(a, b)
        res = zero_month(D, days) if xd[a] < YA.ACTIVATION else fit_month(D, YA.closed_pool(tp, xd, xd[a]), days, var, seeds)
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(pickle.dumps(res))
        print("%s %s: пар %d, λ %s, %.0f с" % (var, xd[a].date(), res["n_train"], res["lam"], time.time() - t0), flush=True)
        out.append((a, b, res))
    return out


def assemble(var: str, x, gap, parts) -> pd.DataFrame:
    n = len(x)
    col = {k: np.full(n, np.nan) for k in ("M", "Fc", "Fp", "Fu", "lam", "seed", "spread")}
    ntr, act, th = np.zeros(n, int), np.zeros(n, bool), np.full((n, M.NP), np.nan)
    rd = np.full(n, np.datetime64("NaT"), "datetime64[ns]")
    for a, b, r in parts:
        for k in col:
            col[k][a:b] = r[k]
        ntr[a:b], th[a:b] = r["n_train"], r["theta"]
        if pd.Timestamp(x["d"].iloc[a]) >= YA.ACTIVATION:
            act[a:b], rd[a:b] = True, np.datetime64(x["d"].iloc[a])
    c8 = x["C"].to_numpy(float)
    lay = L.layers(x, c8 * col["M"], gap)
    df = pd.DataFrame(dict(d=x["d"].to_numpy(), t_plus=x["t_plus"].to_numpy(), rv=x["rv"].to_numpy(float), cfg8=c8, M=col["M"], C=c8 * col["M"], F_c=col["Fc"], F_p=col["Fp"], F_u=col["Fu"],
                           X1_V=lay["X1_V"], F=lay["F"], pi=lay["pi"], beta_gap=lay["beta_gap"], refit_date=rd, active=act, lam=col["lam"], n_train=ntr, seed_best=col["seed"], spread=col["spread"]))
    if var == "C7":
        return df
    return pd.concat([df, pd.DataFrame(th, columns=list(M.NAMES[var]))], axis=1)


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_exact=True)
        return True
    except AssertionError:
        return False


def inputs(var: str, feat=None):
    x, gap = L.inputs()
    return x, gap, M.build_data(x, CF.load_feat() if feat is None else feat, lag=var == "CKL")


def run(var: str, write: bool = True, out: Path = CM_DIR) -> pd.DataFrame:
    R.require(IDS[var])
    bad = R.check()
    if bad:
        raise SystemExit("реестр S-C не готов: %s" % "; ".join(bad))
    x, gap, D = inputs(var)
    df = assemble(var, x, gap, walk_variant(var, D, x, work=out / "_part" / ("%s_%s" % (var, code_key()))))
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
        v, t0 = sys.argv[sys.argv.index("--run") + 1], time.time()
        print("%s: записано %d строк, %.0f с" % (v, len(run(v)), time.time() - t0))
    elif "--check" in sys.argv or "--check-pit" in sys.argv:
        from models_v2.s18 import cm_walk_chk as K
        b = K.check() if "--check" in sys.argv else K.check_pit()
        print("%s: %s" % ("--check" if "--check" in sys.argv else "--check-pit", "ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        raise SystemExit("укажите --run VAR, --check или --check-pit")
