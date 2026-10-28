# -*- coding: utf-8 -*-
"""
sweep_drivers.py — драйверы перебора и статистика отбора, независимые от данных проекта.

Контракт: пользователь передаёт функцию
    evaluate(params: dict, period: str) -> dict
которая возвращает хотя бы {"sharpe": float, "by_year": {год: sharpe}} и любые другие метрики.
Все функции чистые: без чтения файлов, без глобального состояния. Ничего не знают о стратегиях.

Содержимое:
  oat_sweep          — перебор по одному параметру (остальные фиксированы), с «торнадо» важности
  factorial_sweep    — полный факторный план по списку параметров
  plateau_score      — среднее целевой по соседям точки в сетке (выбор по плато, не по пику)
  pick_by_plateau    — топ-k кандидатов по плато
  objective_stable   — целевая «среднее годовых Шарпов − λ·σ по годам»
  psr, dsr           — Probabilistic / Deflated Sharpe Ratio (Bailey & López de Prado)
  expected_max_sharpe— E[max SR из N] для DSR
  pbo_cscv           — Probability of Backtest Overfitting по CSCV (Bailey, Borwein, LdP, Zhu 2017)
  optuna_search      — обёртка над Optuna TPE (если установлен), с той же целевой
  morris_screen      — упрощённый метод Морриса (элементарные эффекты) для ранжирования параметров
"""
from __future__ import annotations
import itertools, math, random
from typing import Callable, Dict, List, Any, Sequence, Tuple, Optional

Params = Dict[str, Any]
Evaluate = Callable[[Params, str], Dict[str, Any]]

# ------------------------------------------------------------------ целевые

def objective_stable(res: Dict[str, Any], lam: float = 0.5) -> float:
    """Среднее годовых Шарпов минус lam * их стандартное отклонение. Ищет плато во времени, не пик."""
    by = res.get("by_year") or {}
    vals = [float(v) for v in by.values() if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float(res.get("sharpe", float("nan")))
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0
    return m - lam * sd

# ------------------------------------------------------------------ OAT

def oat_sweep(evaluate: Evaluate, base: Params, grid: Dict[str, Sequence[Any]], period: str,
              objective: Callable[[Dict[str, Any]], float] = objective_stable) -> Dict[str, Any]:
    """Перебор по одному параметру. Возвращает строки и «торнадо»: размах целевой по каждому параметру."""
    rows: List[Dict[str, Any]] = []
    base_res = evaluate(dict(base), period); base_obj = objective(base_res)
    rows.append({"param": None, "value": None, **dict(base), "objective": base_obj, **_flat(base_res)})
    tornado = {}
    for p, values in grid.items():
        objs = []
        for v in values:
            prm = dict(base); prm[p] = v
            res = evaluate(prm, period); o = objective(res); objs.append(o)
            rows.append({"param": p, "value": v, **prm, "objective": o, **_flat(res)})
        tornado[p] = {"min": min(objs), "max": max(objs), "range": max(objs) - min(objs),
                      "best_value": values[objs.index(max(objs))], "n": len(values)}
    ranked = sorted(tornado.items(), key=lambda kv: -kv[1]["range"])
    return {"base_objective": base_obj, "rows": rows, "tornado": dict(ranked),
            "n_trials": 1 + sum(len(v) for v in grid.values())}

# ------------------------------------------------------------------ факторный

def factorial_sweep(evaluate: Evaluate, base: Params, grid: Dict[str, Sequence[Any]], period: str,
                    objective: Callable[[Dict[str, Any]], float] = objective_stable) -> Dict[str, Any]:
    keys = list(grid.keys()); rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        prm = dict(base); prm.update(dict(zip(keys, combo)))
        res = evaluate(prm, period); rows.append({**prm, "objective": objective(res), **_flat(res)})
    return {"rows": rows, "keys": keys, "grid": {k: list(grid[k]) for k in keys}, "n_trials": len(rows)}

# ------------------------------------------------------------------ плато

def plateau_score(rows: List[Dict[str, Any]], keys: List[str], grid: Dict[str, List[Any]],
                  point: Dict[str, Any], radius: int = 1, col: str = "objective") -> Tuple[float, int]:
    """Среднее целевой по соседям точки (±radius шагов по каждому ключу, включая саму точку). Возвращает (среднее, число соседей)."""
    idx = {k: grid[k].index(point[k]) for k in keys}
    lookup = {tuple(r[k] for k in keys): r[col] for r in rows}
    vals = []
    for offs in itertools.product(*[range(-radius, radius + 1)] * len(keys)):
        nb = {}
        ok = True
        for k, o in zip(keys, offs):
            j = idx[k] + o
            if j < 0 or j >= len(grid[k]): ok = False; break
            nb[k] = grid[k][j]
        if not ok: continue
        v = lookup.get(tuple(nb[k] for k in keys))
        if v is not None and not math.isnan(v): vals.append(v)
    return (sum(vals) / len(vals) if vals else float("nan")), len(vals)

def pick_by_plateau(fact: Dict[str, Any], k: int = 3, radius: int = 1, min_neighbors: int = 3) -> List[Dict[str, Any]]:
    """Топ-k точек по плато (а не по пику). Точки с числом соседей < min_neighbors (край сетки) помечаются."""
    rows, keys, grid = fact["rows"], fact["keys"], fact["grid"]
    scored = []
    for r in rows:
        pt = {kk: r[kk] for kk in keys}
        ps, n = plateau_score(rows, keys, grid, pt, radius)
        scored.append({**r, "plateau": ps, "neighbors": n, "edge": n < min_neighbors})
    scored.sort(key=lambda r: (-(r["plateau"] if not math.isnan(r["plateau"]) else -1e9)))
    return scored[:k]

# ------------------------------------------------------------------ PSR / DSR

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_ppf(p: float) -> float:
    # Acklam approximation, достаточная для DSR
    if p <= 0 or p >= 1: raise ValueError("p in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def psr(sr: float, sr_star: float, T: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Probabilistic Sharpe Ratio. sr, sr_star — в той же шкале, что и наблюдения (дневной SR для дневных T). kurt — обычный (не избыточный)."""
    if T <= 1: return float("nan")
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4.0 * sr * sr))
    return _norm_cdf((sr - sr_star) * math.sqrt(T - 1) / denom)

def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """E[max SR] среди N независимых проб с дисперсией SR var_sr (Bailey & LdP 2014)."""
    if n_trials <= 1 or var_sr <= 0: return 0.0
    g = 0.5772156649
    return math.sqrt(var_sr) * ((1 - g) * _norm_ppf(1 - 1.0 / n_trials) + g * _norm_ppf(1 - 1.0 / (n_trials * math.e)))

def dsr(sr: float, trials_sr: Sequence[float], T: int, skew: float = 0.0, kurt: float = 3.0) -> Dict[str, float]:
    """Deflated Sharpe Ratio: PSR относительно E[max SR из N]. trials_sr — SR всех проб (той же шкалы, что sr)."""
    n = len(trials_sr)
    if n > 1:
        m = sum(trials_sr) / n; var = sum((x - m) ** 2 for x in trials_sr) / (n - 1)
    else:
        var = 0.0
    sr_star = expected_max_sharpe(n, var)
    return {"n_trials": n, "var_sr": var, "sr_star": sr_star, "dsr": psr(sr, sr_star, T, skew, kurt)}

# ------------------------------------------------------------------ PBO (CSCV)

def pbo_cscv(perf_matrix: List[List[float]], n_blocks: int = 8, higher_is_better: bool = True) -> Dict[str, Any]:
    """
    Probability of Backtest Overfitting по CSCV.
    perf_matrix: T x N — метрика по времени (строки — периоды/дни, столбцы — конфигурации N).
    Делим T на n_blocks смежных блоков; для каждого разбиения блоков пополам (train/test):
      выбираем лучшую конфигурацию на train, смотрим её относительный ранг на test.
    PBO = доля разбиений, где лучшая на train оказалась в нижней половине на test.
    Возвращает pbo, число разбиений и распределение логитов.
    """
    T = len(perf_matrix); N = len(perf_matrix[0]) if T else 0
    if T < n_blocks or N < 2: return {"pbo": float("nan"), "n_splits": 0, "logits": []}
    edges = [round(i * T / n_blocks) for i in range(n_blocks + 1)]
    blocks = [list(range(edges[i], edges[i + 1])) for i in range(n_blocks)]
    def agg(rows_idx):
        out = [0.0] * N
        for i in rows_idx:
            r = perf_matrix[i]
            for j in range(N): out[j] += r[j]
        return [x / max(1, len(rows_idx)) for x in out]
    logits = []; below = 0; n_splits = 0
    for train_blocks in itertools.combinations(range(n_blocks), n_blocks // 2):
        test_blocks = [b for b in range(n_blocks) if b not in train_blocks]
        tr = [i for b in train_blocks for i in blocks[b]]; te = [i for b in test_blocks for i in blocks[b]]
        a_tr, a_te = agg(tr), agg(te)
        best = max(range(N), key=lambda j: a_tr[j] if higher_is_better else -a_tr[j])
        # относительный ранг лучшей на test (0..1), 1 = лучшая
        if higher_is_better:
            rank = sum(1 for j in range(N) if a_te[j] < a_te[best]) / (N - 1)
        else:
            rank = sum(1 for j in range(N) if a_te[j] > a_te[best]) / (N - 1)
        w = min(max(rank, 1e-6), 1 - 1e-6); logit = math.log(w / (1 - w))
        logits.append(logit); n_splits += 1
        if logit <= 0: below += 1
    return {"pbo": below / n_splits, "n_splits": n_splits, "logits": logits}

# ------------------------------------------------------------------ Моррис (упрощённый)

def morris_screen(evaluate: Evaluate, grid: Dict[str, List[Any]], period: str, n_paths: int = 10, seed: int = 20261027,
                  objective: Callable[[Dict[str, Any]], float] = objective_stable) -> Dict[str, Any]:
    """Элементарные эффекты: случайная стартовая точка сетки, меняем параметры по одному в случайном порядке.
    mu* — средний |эффект| (важность), sigma — разброс эффекта (взаимодействия/нелинейность)."""
    rnd = random.Random(seed); keys = list(grid.keys()); eff = {k: [] for k in keys}; n_eval = 0
    for _ in range(n_paths):
        pt = {k: rnd.choice(grid[k]) for k in keys}
        cur = objective(evaluate(dict(pt), period)); n_eval += 1
        for k in rnd.sample(keys, len(keys)):
            i = grid[k].index(pt[k]); j = i + 1 if i + 1 < len(grid[k]) else i - 1
            if j < 0: continue
            nxt = dict(pt); nxt[k] = grid[k][j]
            val = objective(evaluate(nxt, period)); n_eval += 1
            eff[k].append(val - cur); pt, cur = nxt, val
    out = {}
    for k in keys:
        e = eff[k]
        if not e: out[k] = {"mu_star": 0.0, "sigma": 0.0, "n": 0}; continue
        mu_star = sum(abs(x) for x in e) / len(e); m = sum(e) / len(e)
        sigma = (sum((x - m) ** 2 for x in e) / len(e)) ** 0.5
        out[k] = {"mu_star": mu_star, "sigma": sigma, "n": len(e)}
    ranked = dict(sorted(out.items(), key=lambda kv: -kv[1]["mu_star"]))
    return {"effects": ranked, "n_evaluations": n_eval}

# ------------------------------------------------------------------ Optuna

def optuna_search(evaluate: Evaluate, base: Params, space: Dict[str, Sequence[Any]], period: str, n_trials: int = 100,
                  seed: int = 20261027, objective: Callable[[Dict[str, Any]], float] = objective_stable,
                  n_jobs: int = 1) -> Dict[str, Any]:
    """TPE по категориальным сеткам (значения — из объявленной сетки, не непрерывные: так N остаётся честным)."""
    try:
        import optuna
    except ImportError:
        raise RuntimeError("optuna не установлен: pip install optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trials_log: List[Dict[str, Any]] = []
    def obj(trial):
        prm = dict(base)
        for k, vals in space.items():
            prm[k] = trial.suggest_categorical(k, list(vals))
        res = evaluate(prm, period); o = objective(res)
        trials_log.append({**prm, "objective": o, **_flat(res)})
        return o
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs)
    return {"best_params": study.best_params, "best_objective": study.best_value, "rows": trials_log, "n_trials": len(trials_log)}

# ------------------------------------------------------------------ служебное

def _flat(res: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in res.items():
        if k == "by_year" and isinstance(v, dict):
            for y, s in v.items(): out[f"sharpe_{y}"] = s
        elif isinstance(v, (int, float, str)) or v is None:
            out[k] = v
    return out


# ------------------------------------------------------------------ самопроверка
if __name__ == "__main__":
    # Синтетика: истинная целевая — плато вокруг (w=20, p=0.6), пик-шум в (w=5, p=0.9); проверяем, что выбор по плато не ведётся на шум.
    import statistics
    def truth(w, p): return -((w - 20) / 15) ** 2 - ((p - 0.6) / 0.3) ** 2
    def evaluate(params, period):
        w, p = params["w"], params["p"]; base = truth(w, p)
        spike = 2.5 if (w == 5 and p == 0.9) else 0.0
        by = {2018: base + spike + 0.05, 2019: base + spike - 0.05}
        return {"sharpe": base + spike, "by_year": by, "mdd": 0.1}
    grid = {"w": [5, 10, 15, 20, 25, 30, 35], "p": [0.3, 0.45, 0.6, 0.75, 0.9]}
    o = oat_sweep(evaluate, {"w": 20, "p": 0.6}, grid, "train")
    assert list(o["tornado"].keys())[0] in ("w", "p") and o["n_trials"] == 13
    f = factorial_sweep(evaluate, {}, grid, "train")
    top = pick_by_plateau(f, k=3)
    assert top[0]["w"] == 20 and top[0]["p"] == 0.6, top[0]      # плато, не шумовой пик (5, 0.9)
    peak = max(f["rows"], key=lambda r: r["objective"]); assert (peak["w"], peak["p"]) == (5, 0.9)  # а argmax — шум
    # DSR: при N=1 выше, чем при N=100 с тем же SR
    r1 = random.Random(1); d1 = dsr(0.08, [0.08], T=500); d100 = dsr(0.08, [r1.gauss(0, 0.05) for _ in range(100)], T=500)
    assert d1["dsr"] > d100["dsr"]
    # PBO: на случайных данных ≈ 0.5
    # на одном случайном наборе PBO сильно шумит (разбиения коррелированы), поэтому проверяем среднее по 12 наборам
    pbos = []
    for sd in range(12):
        rnd = random.Random(sd); M = [[rnd.gauss(0, 1) for _ in range(20)] for _ in range(400)]
        pbos.append(pbo_cscv(M, 8)["pbo"])
    pb = {"pbo": sum(pbos) / len(pbos)}; assert 0.35 < pb["pbo"] < 0.65, pb["pbo"]
    rnd = random.Random(99)
    # PBO: при настоящем сигнале в одной конфигурации — низко
    M2 = [[rnd.gauss(0.3 if j == 3 else 0, 1) for j in range(20)] for _ in range(400)]
    pb2 = pbo_cscv(M2, 8); assert pb2["pbo"] < 0.2, pb2["pbo"]
    m = morris_screen(evaluate, grid, "train", n_paths=6); assert set(m["effects"]) == {"w", "p"}
    print("sweep_drivers: самопроверка пройдена;", "OAT n =", o["n_trials"], "| плато →", (top[0]["w"], top[0]["p"]), "| PBO шум =", round(pb["pbo"], 2), "| PBO сигнал =", round(pb2["pbo"], 2), "| DSR N=1 %.2f, N=100 %.2f" % (d1["dsr"], d100["dsr"]))
