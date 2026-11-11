"""S-Z-IMPL-2 (Ж-258; `docs/otchety/harj_struct/S_Z_SPEC.md`, разд. 2–3, 6, 8): помесячный проход S-RCKZ / S-RZ / S-RCKZs под процедурой R, слои X1 и W5-G над C_Z, запись `data/regen/harj_struct/Z/{rckz,rz,rckzs}.parquet`.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Исследовательский результат на раскрытой истории, не OOS. Код `r_*`, `cm_*`, `l_*`, `z_feat`, `z_model` не правится (импорт).
Протокол = S-RCK (`r_walk`): переоценка в первый торговый день месяца D, закрытый пул P (t⁺ ≤ D), |P| ≥ 630, сетка Λ = {θ = 0; 0,1; 0,01; 0,001}, блоки V₃, V₂, V₁ по 126, T_b — пары раньше V_b, допуск w ≥ 2 из 3, ничья — больший λ;
c, m — по T_b (финал — по P); G — обычная QLIKE уровня F при замороженных π, β. Голова на seed (5 голов), все z — от энкодера года D (класс № 146), μ, σ по T (обрезка ±3 после стандартизации), прогноз — «голова 0 + среднее отклонений».
S-RCKZs — z каждой даты заменено z сессии на 252 раньше (zmaps `zs`, без переноса); пары без z исключаются (Д4), дню прогноза без z — C := C_S-RCK (`z_ok` = 0).
Публичное API:
- `VARS` = {"rckz": ("CKZ", False), "rz": ("Z", False), "rckzs": ("CKZ", True)} (вариант модели, сдвиг), `IDS`, `OUTS`, `Z_DIR`; `load_z()` -> (zm, zs) из `zmaps.npz` (иначе считает `z_feat`).
- `fit_month(ctx, zy, P, days, c0, var, use_z=True, ref_c=None, zero=False) -> dict`: `use_z=False` — θ_Z удалён (путь C-K, 5 одинаковых голов), `zero=True` — z̃ ≡ 0 (диагностика Н3). `choose(...)`, `walk(ctx, zsrc, var, use_z, ref_c, work, positions)`, `assemble(ctx, parts)`, `run(key)`.
- Колонки выхода: как `R/rck.parquet` (θ_CK — среднее по головам) плюс `thz_0..3`, `thsd_<CK>`, `thsd_z0..3`, `n_clip`, `n_excl`, `z_ok`. Проверки — `z_walk_chk.py`.
Запуск: `python3 -W ignore -m models_v2.s18.z_walk --run rckz|rz|rckzs | --check-nest | --check-pit | --check | --synth-null`."""
from __future__ import annotations

import torch  # noqa: F401  (первым: конфликт OpenMP, класс № 152)

import hashlib
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as RW
from models_v2.s18 import r_walk_ctx as RC
from models_v2.s18 import z_feat as ZF
from models_v2.s18 import z_model as ZM
from models_v2.s18 import z_reg as ZR

torch.set_num_threads(1)
VARS = {"rckz": ("CKZ", False), "rz": ("Z", False), "rckzs": ("CKZ", True)}
IDS = {"rckz": "S-RCKZ", "rz": "S-RZ", "rckzs": "S-RCKZs"}
OUTS = {k: k + ".parquet" for k in IDS}
Z_DIR, SEEDS, RCK = ZF.Z_DIR, ZF.SEEDS, RW.R_DIR / "rck.parquet"


def load_z() -> tuple:
    if ZF.ZMAPS.exists():
        zm, zs, _ = ZF.unpack(ZF.ZMAPS)
        return zm, zs
    zg = ZF.zgrids()
    return ZF.zmaps(zg), ZF.zshift(zg)


def _zts(zy, T, zero: bool = False) -> list:
    """z̃ каждой головы (полная длина): μ, σ только по строкам T; zero — диагностика z̃ ≡ 0."""
    if zero:
        return [np.zeros((len(z), ZM.NZ)) for z in zy]
    return [ZM.ztil(z, *ZM.std_stats(z, T)) for z in zy]


def _mix(ls):
    return ls[0] + np.mean([l - ls[0] for l in ls], axis=0)      # голова 0 + среднее отклонений: равные головы дают голову 0 побитно


def fit_heads(V, zts, var: str, use_z: bool, lam: float, c: float, m: float) -> list:
    if not use_z:    # одна задача у всех голов ⇒ одинаковые θ (детерминированный L-BFGS-B): подгоняется один раз
        return [RL.fit_r(V, "CK", lam, c, m, "R")[0] if var == "CKZ" else np.zeros(0)] * len(zts)
    return [ZM.fit_head(V, zt, var, lam, c, m, "R") for zt in zts]


def head_lnc(ths, V, zts, var: str, use_z: bool) -> np.ndarray:
    if not use_z:
        return RL.lnc_of(torch.as_tensor(ths[0]), V, "CK").numpy() if var == "CKZ" else torch.log(V.c8).numpy()
    return ZM.predict_lnc(ths, V, zts, var)


def choose(ctx, zy, P, var: str, use_z: bool, zero: bool = False) -> tuple:
    G = {lam: [] for lam in RW.LAMS}
    for T, Vb in RW.blocks(P):     # обучение на T_b; валидация — обычная QLIKE уровня F на V_b
        c, m, zt = *RL.clip_stats(ctx.x0, T), _zts(zy, T, zero)
        VT, VV = ctx.view(T), ctx.view(Vb)
        base = float(RL.ell(ctx.rv[Vb], ctx.lnf0[Vb]).mean())
        for lam in RW.LAMS:
            ths = fit_heads(VT, [t[T] for t in zt], var, use_z, lam, c, m)
            lnf = RL.ln_f(torch.as_tensor(head_lnc(ths, VV, [t[Vb] for t in zt], var, use_z)), VV).numpy()
            G[lam].append(float(RL.ell(ctx.rv[Vb], lnf).mean()) - base)
    lam, w = RW.decide(G, "blocks")
    return lam, w, G


def _empty(P, c0, days) -> dict:
    nan = float("nan")
    return dict(active=False, lam=nan, w=(nan,) * 3, G=np.full((3, RW.NB), nan), c=nan, m=nan, n_train=len(P), theta=np.zeros(6), thz=np.zeros(4), thsd=np.zeros(10), n_clip=nan, n_excl=0,
                z_ok=np.ones(len(days), bool), C=np.asarray(c0, float).copy())


def _final(ctx, zy, P, days, var, use_z, lam, c, m, zero, ref) -> dict:
    zt = _zts(zy, P, zero)
    ths = fit_heads(ctx.view(P), [t[P] for t in zt], var, use_z, lam, c, m)
    zd = [t[days] for t in zt]
    ok = np.ones(len(days), bool) if not use_z else np.all([np.isfinite(t).all(1) for t in zd], axis=0)
    lnc = head_lnc(ths, ctx.view(days), zd, var, use_z)
    th = np.array(ths) if len(ths[0]) else np.zeros((len(ths), 0))
    ck = _mix(list(th[:, :6])) if var == "CKZ" else np.zeros(6)
    clip = float(np.mean([(np.abs(ZM.ztil_raw(z[P], *ZM.std_stats(z, P))) > ZM.CLIP).mean() for z in zy])) if use_z and not zero else float("nan")
    sd = np.zeros(10)
    if use_z:
        sd[10 - th.shape[1]:] = np.std(th, axis=0)     # θ_CK (6) и θ_Z (4); у S-RZ — только θ_Z
    return dict(theta=ck, thz=_mix(list(th[:, -4:])) if use_z else np.zeros(4), thsd=sd, n_clip=clip, z_ok=ok, C=np.where(ok, np.exp(lnc), ref))


def fit_month(ctx, zy, P, days, c0, var: str, use_z: bool = True, ref_c=None, zero: bool = False) -> dict:
    n0 = len(P)
    if use_z and not zero:     # пары без z (Д4) исключаются из пула, блоков, c, m, μ, σ
        P = P[np.all([np.isfinite(z[P]).all(1) for z in zy], axis=0)]
    out = _empty(P, c0, days)
    out["n_excl"] = n0 - len(P)
    if len(P) < RW.MIN_POOL:
        return out
    lam, w, G = choose(ctx, zy, P, var, use_z, zero)
    c, m = RL.clip_stats(ctx.x0, P)
    out.update(active=True, lam=lam, w=tuple(w[k] for k in RW.LAMS), c=c, m=m)
    for i, k in enumerate(RW.LAMS):
        out["G"][i, RW.NB - len(G[k]):] = G[k]
    if lam > 0.0:
        out.update(_final(ctx, zy, P, days, var, use_z, lam, c, m, zero, np.asarray(c0 if ref_c is None else ref_c, float)))
    return out


def code_key() -> str:
    h = hashlib.sha256()
    for n in ("r_loss", "r_walk", "z_model", "z_walk", "z_feat"):
        h.update((Path(__file__).parent / (n + ".py")).read_bytes())
    h.update(hashlib.sha256(ZF.ZMAPS.read_bytes()).digest() if ZF.ZMAPS.exists() else b"")
    return h.hexdigest()[:12]


def walk(ctx, zsrc, var: str, use_z: bool = True, ref_c=None, work=None, positions=None, zero: bool = False, quiet: bool = False) -> list:
    out, c8 = [], ctx.x["C"].to_numpy(float)
    for a, b in RW.month_bounds(ctx):
        if positions is not None and a not in positions:
            continue
        f = None if work is None else Path(work) / ("m%03d.pkl" % a)
        if f is not None and f.exists():
            out.append((a, b, pickle.loads(f.read_bytes())))
            continue
        t0, Y = time.time(), pd.Timestamp(ctx.x["d"].iloc[a]).year
        res = fit_month(ctx, [zsrc[Y][s] for s in SEEDS], RL.pool_of("CK", ctx.D, ctx.x, a), ctx.dp[a:b], c8[a:b], var, use_z, None if ref_c is None else ref_c[a:b], zero)
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(pickle.dumps(res))
        if not quiet:
            print("%s %s: пар %d, λ* %s, %.0f с" % (var, pd.Timestamp(ctx.x["d"].iloc[a]).date(), res["n_train"], res["lam"], time.time() - t0), flush=True)
        out.append((a, b, res))
    return out


def assemble(ctx, parts: list) -> pd.DataFrame:
    df, n = RW.assemble("rck", ctx, parts), len(ctx.x)
    ex = dict(thz=np.zeros((n, 4)), thsd=np.zeros((n, 10)), n_clip=np.full(n, np.nan), n_excl=np.zeros(n, int), z_ok=np.ones(n, int))
    for a, b, r in parts:
        ex["thz"][a:b], ex["thsd"][a:b], ex["n_clip"][a:b], ex["n_excl"][a:b], ex["z_ok"][a:b] = r["thz"], r["thsd"], r["n_clip"], r["n_excl"], np.asarray(r["z_ok"], int)
    sd = ["thsd_" + k for k in RW.names("CK")] + ["thsd_z%d" % j for j in range(4)]
    ex_df = pd.concat([pd.DataFrame(ex["thz"], columns=["thz_%d" % j for j in range(4)]), pd.DataFrame(ex["thsd"], columns=sd)], axis=1)
    return pd.concat([df, ex_df, pd.DataFrame({k: ex[k] for k in ("n_clip", "n_excl", "z_ok")})], axis=1)


def run(key: str, write: bool = True, out: Path = Z_DIR) -> pd.DataFrame:
    ZR.require(IDS[key])
    bad = ZR.check()
    if bad or not ZF.ZMAPS.exists():
        raise SystemExit("реестр S-Z не готов или нет zmaps.npz (порядок: регистрация → zmaps → проходы): %s" % "; ".join(bad))
    var, shift = VARS[key]
    ctx, (zm, zs) = RC.make_ctx("CK"), load_z()
    ref = pd.read_parquet(RCK)["C"].to_numpy(float) if var == "CKZ" else None
    df = assemble(ctx, walk(ctx, zs if shift else zm, var, True, ref, out / "_part" / ("%s_%s" % (key, code_key()))))
    p = out / OUTS[key]
    if write:
        if p.exists() and not RW._same(pd.read_parquet(p), df):
            raise SystemExit("%s: существует и отличается от пересчёта — перезапись отказана" % p)
        tmp = out / (OUTS[key] + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(p)
    return df


if __name__ == "__main__":
    if "--run" in sys.argv:
        v, t0 = sys.argv[sys.argv.index("--run") + 1], time.time()
        print("%s: записано %d строк, %.0f с" % (v, len(run(v)), time.time() - t0))
    else:
        from models_v2.s18 import z_walk_chk as K
        sys.exit(K.main(sys.argv[1:]))
