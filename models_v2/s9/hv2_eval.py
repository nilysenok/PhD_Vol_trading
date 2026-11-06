"""Ж-214, лист 2: общий контекст направлений V5/V6 — мир, замороженные контроли (A0 = HAR, XJ, LF, D0_drop_a из Ж-212), строка таблицы (QLIKE, crit_wf, Δ к каждому контролю,
лет лучше, DM), главный тест обрезки на реальных данных, правило V13 (с уточнениями У1/У2 GATES_V2), запись решений в CHECKPOINT.json."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s8 import hs_eval, hs_store
from models_v2.s9 import hv2_ledger as L, hv2_paths as P, hv2_world as W

CRIT, MIN_YEARS = hs_eval.CRIT_STEP, hs_eval.MIN_YEARS
D0_FULL_KEY, D0_DROP_KEY = "Ж-212:component:5d5e154cb6d9", "Ж-212:component:311098faa001"
TRUNC_AT = (450, 1000)    # позиции в сетке дат для обрезки на реальных данных


def context() -> dict:
    """Мир, df (d, target_date, rv, HAR, XJ, LF), контроли; стоп, если сетка/цель не совпадают с замороженной (same_grid, max_rel_rv)."""
    w = W.build(pd.read_parquet(P.SBER), pd.read_parquet(P.M10), pd.read_parquet(P.IMOEX))
    df = hs_store.frozen()
    same = bool((w.d == df["d"].to_numpy()).all() and (w.td == df["target_date"].to_numpy()).all())
    rel = float(np.abs(w.target / df["rv"].to_numpy() - 1).max())
    if not same or rel > 0:
        raise SystemExit("стоп: сетка/цель мира не совпадает с components (same_grid=%s, max_rel_rv=%g)" % (same, rel))
    d0 = hs_store.load("D")
    refs = {"A0": df["HAR"].to_numpy(float), "XJ": df["XJ"].to_numpy(float), "LF": df["LF"].to_numpy(float)}
    if "D0_drop_a" in d0.columns and (d0["d"].to_numpy() == df["d"].to_numpy()).all():
        refs["D0_drop_a"] = d0["D0_drop_a"].to_numpy(float)
    return {"w": w, "df": df, "refs": refs, "same_grid": same, "max_rel_rv": rel}


def row(ctx: dict, name: str, s2, extra: dict | None = None, ctrl: tuple = ("A0", "D0_drop_a", "XJ", "LF")) -> dict:
    """Строка: метрики и Δ к каждому контролю (суффикс _<контроль>): d_crit, d_qlike, d_qlike_no_240222, p_dm, years_better, sign_ok_2of3."""
    df, refs = ctx["df"], ctx["refs"]
    r = None
    for k in [c for c in ctrl if c in refs]:
        c = hs_eval.row(df, name, s2, refs[k])
        if r is None:
            r = {x: c[x] for x in c if not (x.startswith("d_") or x in ("p_dm", "years_better", "sign_ok_2of3", "crit_wf_ctrl"))}
        r.update({"%s_%s" % (x, k): c[x] for x in ("d_crit", "d_qlike", "d_qlike_no_240222", "p_dm", "years_better", "sign_ok_2of3")})
        if k == "A0":
            r.update({x: c[x] for x in c if x.startswith("d_G_") or x.startswith("d_P")})
    r.update(extra or {})
    r.update({"same_grid": ctx["same_grid"], "max_rel_rv": ctx["max_rel_rv"], "n": len(df)})
    return r


def trunc_diff(run_fn, w: W.World, i0: int) -> float:
    """Главный тест обрезки: σ̂² на дни <= i0 из исходного и из испорченного мира (все строки > i0 порчены); run_fn(world) -> σ̂² сетки (первый элемент кортежа допустим)."""
    def f(x):
        o = run_fn(W.sub(x, i0))
        return np.asarray(o[0] if isinstance(o, tuple) else o, float)
    a = f(w)
    return float(np.max(np.abs(a / f(W.corrupt(w, i0)) - 1)))


def trunc_real(ctx: dict, run_fn) -> float:
    w = ctx["w"]
    return max(trunc_diff(run_fn, w, int(w.el[k])) for k in TRUNC_AT)


def v13(rows: list, controls: tuple, spent: int, limit: int) -> dict:
    """V13 с У1/У2: лучшая (по crit_wf) строка минимального набора должна быть лучше КАЖДОГО контроля: Δcrit <= −0,0005 и лет лучше >= 6 из 10.
    Досрочно: израсходована половина бюджета направления и ни у одной строки Δcrit < 0 к контролю — стоп."""
    best = min(rows, key=lambda r: r["crit_wf"])
    chk = {k: {"d_crit": best["d_crit_%s" % k], "years_better": best["years_better_%s" % k],
               "ok": bool(best["d_crit_%s" % k] <= -CRIT and best["years_better_%s" % k] >= MIN_YEARS)} for k in controls}
    cont = all(v["ok"] for v in chk.values())
    neg = {k: any(r["d_crit_%s" % k] < 0 for r in rows) for k in controls}
    early = bool(spent * 2 >= limit and not all(neg.values()))
    return {"best": best["model"], "controls": chk, "continue": bool(cont and not early), "early_stop": early,
            "any_d_crit_negative": neg, "spent": spent, "limit": limit, "unused_trials": limit - spent,
            "rule": "лучшая crit_wf лучше каждого контроля: Δcrit_wf <= −0,0005 и лет лучше >= 6 из 10; досрочно — половина бюджета без Δ<0"}


def code_key(mods) -> str:
    """Д5: SHA-256 исходников модулей, M10-таблицы и замороженных контролей; смена любого входа — промах кэша."""
    import hashlib
    from pathlib import Path
    h = hashlib.sha256()
    for m in mods:
        h.update(Path(m.__file__).read_bytes())
    for f in (P.M10, P.COMP):
        h.update(f.read_bytes())
    return h.hexdigest()


def cached(direction: str, name: str, mods, compute, ctx: dict) -> dict:
    """Ряды конфигурации в `<direction>_series.parquet` (колонки `name`, `name__*`); ключ кода — в `<direction>_series_keys.json`. compute() -> {колонка: массив}."""
    path, kp = P.OUT / ("%s_series.parquet" % direction), P.OUT / ("%s_series_keys.json" % direction)
    keys = json.loads(kp.read_text()) if kp.exists() else {}
    df = pd.read_parquet(path) if path.exists() else ctx["df"][["d", "target_date", "rv"]].copy()
    key = code_key(mods)
    if keys.get(name) == key and name in df.columns:
        return {c: df[c].to_numpy(float) for c in df.columns if c == name or c.startswith(name + "__")}
    out = compute()
    for c, a in out.items():
        df[c] = np.asarray(a, float)
    P.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    keys[name] = key
    kp.write_text(json.dumps(keys, indent=1))
    return out


def decide(name: str, obj: dict) -> None:
    df = L.load()
    old = json.loads(L.CHECK.read_text()).get("decisions", {}) if L.CHECK.exists() else {}
    L.write_check(df, decisions={**old, name: obj})
