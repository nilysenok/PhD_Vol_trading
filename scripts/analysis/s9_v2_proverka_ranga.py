"""СD2, Ъ 23:10, п. 4: каким рангом считался отбор #243 — прямая проверка на данных (скрипт, не модуль).
Запуск: TMPDIR=<scratch> python3 <этот файл> <worktree>. Прогнан на базе 98a3fdc (до #246, ручка
c_rank_past ещё была): 1) ключ кэша #243 из дерева e16e220; 2) C основного дизайна (masks._c_masks)
в кэше #243 = пересчёт с c_rank_past true, ≠ с false; 3) ранг сетки C С9 (s9_stats.past_rank) =
перебор без s_t, будущее на ранг ≤ t не влияет. После #246 (ручки нет) шаг 2 «false» = «true»."""
import hashlib
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path(sys.argv[1])
sys.path.insert(0, str(WT))
os.environ["NUMBA_NUM_THREADS"] = "1"
from fastbt import s9  # noqa: E402
from fastbt.s9_stats import past_rank, minp  # noqa: E402

s9.use_v2()
out = []

# 1) ключ кэша #243 (7cb4dc4694) из дерева fastbt коммита e16e220 и main.yaml того же коммита
tmp = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")))
subprocess.run(f"git -C '{WT}' archive e16e220 fastbt | tar -x -C '{tmp}'", shell=True, check=True)
h = hashlib.sha1()
for p in sorted((tmp / "fastbt").rglob("*.py")):
    if "tests" in p.relative_to(tmp / "fastbt").parts:
        continue
    h.update(p.read_bytes())
ch = h.hexdigest()
yml = (tmp / "fastbt/designs/final/main.yaml").read_bytes()
k = hashlib.sha1(ch.encode())
k.update(yml)
for lab, (p, _, _) in s9.SOURCES.items():
    k.update(lab.encode() + hashlib.sha1(Path(p).read_bytes()).digest())
key = k.hexdigest()[:10]
out.append(f"ключ кэша из дерева e16e220: {key}; cache_latest_v2 = "
           f"{(s9.OUT / 'cache_latest_v2.txt').read_text().strip()}; "
           f"c_rank_past в main.yaml e16e220: {'c_rank_past: true' in yml.decode()}")

# 2) C основного дизайна в кэше #243 = пересчёт с c_rank_past true / ≠ c_rank_past false
cache = pickle.loads((s9.OUT / "cache_7cb4dc4694.pkl").read_bytes())
srcs_all = {lab: (s9.SOURCES[lab][2], s9.load_source(lab)) for lab in s9.SOURCES}
yaml_false = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))) / "main_false.yaml"
txt = s9.MAIN_YAML.read_text().replace("c_rank_past: true", "c_rank_past: false")
yaml_false.write_text(txt)
orig = s9.MAIN_YAML
for t in ("SBER", "GAZP" if "GAZP" in cache else "LKOH", "AFLT"):
    srcs = {lab: (hh, df[df["ticker"] == t]) for lab, (hh, df) in srcs_all.items()}
    res = {}
    for flag, path in (("true", orig), ("false", yaml_false)):
        s9.MAIN_YAML = path
        _, o = s9._job((t, srcs))
        res[flag] = o
    s9.MAIN_YAML = orig
    for flag, o in res.items():
        n_eq = n_all = 0
        maxd = 0.0
        for lab in s9.SOURCES:
            for sid in s9.SIDS:
                for Y in s9.YEARS:
                    a = cache[t]["C"][lab][sid].get(Y)
                    b = o["C"][lab][sid].get(Y)
                    if a is None or b is None:
                        continue
                    n_all += 1
                    eq = np.array_equal(a, b, equal_nan=True)
                    n_eq += eq
                    maxd = max(maxd, float(np.nanmax(np.abs(a - b))) if len(a) else 0.0)
        out.append(f"{t}: C u02 кэша #243 vs пересчёт c_rank_past={flag}: побайтно {n_eq}/{n_all}, "
                   f"max|Δпоз| {maxd:.3g}")
    # A не зависит от ручки
    eqA = all(np.array_equal(cache[t]["A"][sid][Y], res["true"]["A"][sid][Y])
              for sid in s9.SIDS for Y in cache[t]["A"][sid])
    out.append(f"{t}: A кэша = пересчёт: {eqA}")

# 3) ранг сетки C С9 (s9_stats.past_rank) в день t не включает s_t — перебором
cache_dates = {t: cache[t]["dates"] for t in cache}
for t in ("SBER", "LKOH", "AFLT"):
    for lab in ("hyb_h5", "harj_h5", "naive_h1"):
        df = s9.load_source(lab)
        S = s9.align(cache_dates[t], df[df["ticker"] == t])
        for W in (126, 252, 504):
            mp = s9._minp(W)
            R = past_rank(S.copy(), W, mp)
            ex = np.full(len(S), np.nan)
            inc = np.full(len(S), np.nan)
            for i in range(len(S)):
                v = S[i]
                if v != v:
                    continue
                w = S[max(0, i - W):i]
                w = w[~np.isnan(w)]
                if len(w) >= mp:
                    ex[i] = (w < v).mean()
                w2 = S[max(0, i - W + 1):i + 1]
                w2 = w2[~np.isnan(w2)]
                if len(w2) >= mp:
                    inc[i] = pd.Series(w2).rank(pct=True).iloc[-1]
            ok = np.array_equal(R, ex, equal_nan=True)
            diff_inc = int(np.nansum(np.abs(R - inc) > 1e-12))
            # возмущение будущего: s_{t+1..} → шум не меняет ранг ≤ t
            tt = len(S) // 2
            S2 = S.copy()
            S2[tt + 1:] = np.random.default_rng(0).uniform(0.001, 1.0, len(S) - tt - 1)
            R2 = past_rank(S2, W, mp)
            fut = np.array_equal(R[:tt + 1], R2[:tt + 1], equal_nan=True)
            out.append(f"{t} {lab} W{W}: ранг С9 = перебор без s_t: {ok}; "
                       f"дней, где ≠ ранг с s_t (rolling.rank): {diff_inc}; будущее не влияет: {fut}")
print("\n".join(out))
