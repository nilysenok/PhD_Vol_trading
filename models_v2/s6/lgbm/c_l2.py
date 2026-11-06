"""C-L2: Optuna (TPE, seed 42, n_startup 8, ≤ 20 trials) по осям с эффектом из C-L1; иначе ABANDON. Пространство объявлено в GATES до счёта:
относительные оси (num_leaves, max_depth, min_data, lr, ff, l2) — u ∈ [−1; 1] (0 — база года, −1 — нижняя точка L1, +1 — верхняя, линейно по значению);
абсолютные: bag ∈ [0,6; 1,0], l1 ∈ [0; 5], gain ∈ [0; 10]. Воспроизводимость: кэш по trial -> повтор ничего не пересчитывает."""
from __future__ import annotations

import hashlib
import json

import optuna

from models_v2.s6.lgbm import c_hp, c_l0, c_l1, c_world
from models_v2.s6.lgbm.c_world import CItem

N_TRIALS, N_STARTUP, SEED = 20, 8, 42
REL = ("num_leaves", "max_depth", "min_data", "lr", "ff", "l2")
ABS = {"bag": (0.6, 1.0), "l1": (0.0, 5.0), "gain": (0.0, 10.0)}


def interp(h, axis: str, u: float):
    """Значения между базой года (u = 0) и нижней (u < 0) / верхней (u > 0) точкой L1 (|u| = 1); целые — округление.
    Ось max_depth при базе −1: u < −1/3 -> 3, u > 1/3 -> 6, иначе −1. Все поля, которые меняет точка оси (например num_leaves + max_depth), интерполируются."""
    h = c_hp.lift(h)
    lo, hi = c_hp.AXES[axis](h)
    t = lo if u < 0 else hi
    if axis == "max_depth" and h.max_depth == -1:
        return t if abs(u) > 1 / 3 else h
    kw = {}
    for f in ("num_leaves", "max_depth", "min_data", "lr", "ff", "l2", "bag", "l1", "gain"):
        v0, v1 = getattr(h, f), getattr(t, f)
        if v0 != v1:
            v = v0 + abs(u) * (v1 - v0)
            kw[f] = int(round(v)) if isinstance(v0, int) else round(float(v), 6)
    return c_hp._set(h, **kw)


def build(hp_y: dict, params: dict) -> dict:
    out = {}
    for Y, h in hp_y.items():
        for a, v in params.items():
            if a in REL:
                h = interp(h, a, v)
            else:
                h = c_hp._set(h, **{c_hp.FIELD[a]: v})
        out[Y] = h
    return out


def suggest(trial, axes: list) -> dict:
    p = {}
    for a in axes:
        if a in REL:
            p[a] = round(trial.suggest_float("u_" + a, -1.0, 1.0), 2)
        else:
            lo, hi = ABS[a]
            p[a] = round(trial.suggest_float(a, lo, hi), 2)
    return p


def stage(W: dict) -> None:
    ax = json.loads(c_l1.AXES_FILE.read_text())
    axes = ax["effect"]
    if not axes:
        (c_world.tab.REP / "C_L2_ABANDON.md").write_text("ABANDON C-L2: ни одна ось C-L1 не дала Δcrit_pen ≤ −%g; Optuna не запускалась (C-L1 таблица осей: C_L1_AXES.csv).\n" % c_l1.EFFECT)
        print("ABANDON C-L2: осей с эффектом нет", flush=True)
        return
    bname, spec, hp = c_l0.base_cfg(W)
    done = []

    def objective(trial):
        p = suggest(trial, axes)
        name = "C2_" + hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:8]
        it = CItem(name, spec, build(hp, p), "hp", "c_l2", "C-L2 Optuna(TPE seed 42) по осям %s: %s" % (axes, json.dumps(p)))
        c_world.run_batch(W, [it])
        done.append(name)
        return c_world.get(name)[1]["crit"]["crit_pen"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=N_STARTUP))
    st.optimize(objective, n_trials=N_TRIALS)
    names = list(dict.fromkeys(done))
    print("новых trials hp:", c_world.register(names), "уникальных конфигураций:", len(names), flush=True)
    df = c_world.rows(W, [bname] + names, bname)
    c_world.write(df, c_world.tab.REP / "C_L2_RESULTS.csv", "C-L2 Optuna: оси %s, %d trials Optuna, база %s; потолок" % (axes, N_TRIALS, bname))
    (c_world.DATA_C / "C_L2_best.json").write_text(json.dumps({"best_params": st.best_params, "best_value": st.best_value, "axes": axes}, indent=1))
    print("лучший C-L2:", st.best_params, "%.6f" % st.best_value, "CHECKPOINT:", c_world.checkpoint()["used_stream_c"], flush=True)
