"""Д17.2: признаки и цели на реализованном размахе (RRV) -> data/regen/features_pit_v3_rrv.parquet.

Вход: data/iss_check/rrv_1850.parquet (крейт moex_iss, `moex_iss range`): date, ticker, rrv_pk, rrv_gk, n_bars —
окно [09:50; 18:50), без баров карантина (функция пайплайна). Строки и все старые колонки — features_pit_v3 как есть.

Новые колонки для m ∈ {pk, gk}:
  rrv_{m}_d, rrv_{m}_w, rrv_{m}_m — ln среднего RRV за 1/5/22 торговых дня ряда RRV тикера, окно кончается днём t
      включительно (t−0…t−4, t−0…t−21; min_periods = 1, как add_har_lags у rv_w/rv_m в v3). Неположительное
      среднее (все бары дня плоские, H = L) -> NaN. Затем, как собственные признаки v3: строки матрицы без значения
      протягиваются по строкам тикера в порядке дат (только прошлое), до первого наблюдения — NaN.
  rrv_{m}_target_h1, rrv_{m}_target_mean_h5, rrv_{m}_target_mean_h22 — уровень RRV (не ln: как rv_target_*,
      логарифм берёт модель), оператор m0_pit_lib.add_targets: значение строки t+1 тикера и среднее строк t+1…t+h
      по строкам матрицы, без протяжки.
Запуск: python3 scripts/analysis/d172_rrv_features.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading")
RRV = MAIN / "data/iss_check/rrv_1850.parquet"
V3 = MAIN / "data/regen/features_pit_v3.parquet"
OUT = MAIN / "data/regen/features_pit_v3_rrv.parquet"
MEAS = ("pk", "gk")
FEAT = [f"rrv_{m}_{w}" for m in MEAS for w in ("d", "w", "m")]
TGT = [f"rrv_{m}_{t}" for m in MEAS for t in ("target_h1", "target_mean_h5", "target_mean_h22")]
NEW = FEAT + TGT


def load_rrv(path=RRV):
    r = pd.read_parquet(path)
    r["date"] = pd.to_datetime(r["date"])
    return r.sort_values(["ticker", "date"]).reset_index(drop=True)


def new_columns(keys, rrv):
    """keys — строки матрицы (date, ticker) в порядке матрицы; rrv — дневной ряд. Возвращает DataFrame NEW
    с тем же индексом, что keys."""
    r = rrv.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = r.groupby("ticker", sort=False)
    f = r[["date", "ticker"]].copy()
    for m in MEAS:
        x = r[f"rrv_{m}"]
        for w, n in (("d", 1), ("w", 5), ("m", 22)):
            mu = x.groupby(r["ticker"], sort=False).transform(lambda s: s.rolling(n, min_periods=1).mean())
            f[f"rrv_{m}_{w}"] = np.log(mu.where(mu > 0))
    del g
    k = keys[["date", "ticker"]].copy()
    k["_i"] = np.arange(len(k))
    k = k.merge(f, on=["date", "ticker"], how="left")
    k = k.merge(r[["date", "ticker", "rrv_pk", "rrv_gk"]], on=["date", "ticker"], how="left")
    k = k.sort_values(["ticker", "date", "_i"]).reset_index(drop=True)
    k[FEAT] = k.groupby("ticker", sort=False)[FEAT].ffill()
    for m in MEAS:
        x = k.groupby("ticker", sort=False)[f"rrv_{m}"]
        k[f"rrv_{m}_target_h1"] = x.shift(-1)
        for h in (5, 22):
            # среднее строк t+1…t+h явной суммой сдвигов: то же, что rolling у add_targets, но без накопления
            # округления скользящей суммы от дальнего будущего (иначе цель зависит от строк > t+h на уровне 1e-16)
            acc = x.shift(-1)
            for j in range(2, h + 1):
                acc = acc + x.shift(-j)
            k[f"rrv_{m}_target_mean_h{h}"] = acc / h
    k = k.sort_values("_i")
    k.index = keys.index
    return k[NEW]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    v3 = pd.read_parquet(V3)
    rrv = load_rrv()
    out = v3.copy()
    nc = new_columns(v3[["date", "ticker"]], rrv)
    for c in NEW:
        out[c] = nc[c].values
    bad = [c for c in v3.columns if not (out[c].dtype == v3[c].dtype and (
        out[c].equals(v3[c]) if out[c].dtype == object else out[c].to_numpy().tobytes() == v3[c].to_numpy().tobytes()))]
    assert not bad and len(out) == len(v3), bad
    out.to_parquet(OUT, index=False)
    back = pd.read_parquet(OUT)
    bad2 = [c for c in v3.columns if not (back[c].dtype == v3[c].dtype and (
        back[c].equals(v3[c]) if back[c].dtype == object else back[c].to_numpy().tobytes() == v3[c].to_numpy().tobytes()))]
    assert not bad2, bad2
    cfg = json.loads((MAIN / "data/regen/features_pit_v3.config.json").read_text())
    cfg["new_cols_rrv"] = {"features": FEAT, "targets": TGT}
    OUT.with_suffix(".config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    print(f"строк {len(out)}, колонок {out.shape[1]} (v3 {v3.shape[1]}), старые колонки побайтно равны: PASS")
    print("NaN по новым:", {c: round(float(out[c].isna().mean()), 5) for c in NEW})
    print("SHA-256", OUT.name, sha(OUT))
    print("SHA-256", OUT.with_suffix(".config.json").name, sha(OUT.with_suffix(".config.json")))


if __name__ == "__main__":
    main()
