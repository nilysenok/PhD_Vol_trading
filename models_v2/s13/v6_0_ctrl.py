"""V6-0, G1: замороженные входы SBER восстановлены побитно прежними: K1 (HAR-J PIT), K2 / K3 (`k23_SBER.parquet`), M = √(K2·K3), K8 = √(H·M), LGBM-D2 (панельный, только контроль). SHA файлов — против `CONTROLS.csv` V5;
K8 пересчитан против сохранённого `K8.parquet` V4 (побитно); QLIKE всей истории (4 знака); проверка сдвига (Д1) и мутация «прогноз сдвинут на +1 день»; ориентир 0,2081 — `v4_bench`. Переобучения нет."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_audit as AU, v4_bench, v4_fin_data as FD, v4_io as IO4
from models_v2.s13 import v6_0_io as IO, v6_0_q as Q

EXPECT4 = {"K1": 0.2337, "K2": 0.2113, "K3": 0.2117, "D2": 0.2244, "K8": 0.2165}
FILES = {"K1": IO4.path_of("H", "SBER"), "K2": IO4.path_of("K2", "SBER"), "K3": IO4.path_of("K3", "SBER"), "LGBM-D2": IO4.path_of("D2", "x")}
SER = ["K1", "K2", "K3", "M", "K8", "D2"]


def file_rows() -> pd.DataFrame:
    v5 = pd.read_csv(IO.V5REP / "CONTROLS.csv", comment="#")
    v5 = v5[v5["section"] == "file"].set_index("item")
    rows = []
    for k, p in FILES.items():
        now = IO.sha_file(p)
        ref = v5.loc["SBER:" + k, "sha256"]
        rows.append(dict(section="file", item="SBER:" + k, source=str(p.relative_to(IO.ROOT)), sha256=now, sha_v5=ref, ok=bool(now == ref)))
    return pd.DataFrame(rows)


def series_rows(w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SER:
        x = w[s].to_numpy(float)
        ok = np.isfinite(x) & (x > 0)
        q = Q.ql(w["rv"].to_numpy()[ok], x[ok])
        m = Q.wmask(w["t_plus"].to_numpy()[ok], "2017_2021")
        rows.append(dict(section="series", item=s, n=int(ok.sum()), first_d=w.index[ok].min().date(), last_d=w.index[ok].max().date(), first_t_plus=w["t_plus"][ok].min().date(),
                         last_t_plus=w["t_plus"][ok].max().date(), finite_share=float(ok.mean()), qlike_all=float(q.mean()), qlike_2017_2021=float(q[m].mean()),
                         expected_4dp=EXPECT4.get(s, np.nan), ok=bool(round(float(q.mean()), 4) == EXPECT4[s]) if s in EXPECT4 else True))
    return pd.DataFrame(rows)


def k8_row(w: pd.DataFrame) -> dict:
    """K8 пересчитан = сохранённый K8 V4 (`K8.parquet`, строки SBER) и = ряд V4:K8 из кадра, побитно."""
    sv = pd.read_parquet(FD.OUT / "K8.parquet", columns=["ticker", "date", "F"])
    sv = sv[sv["ticker"] == "SBER"].set_index("date")["F"].reindex(w.index)
    same = bool(np.array_equal(sv.to_numpy(float), w["K8"].to_numpy(float)))
    return dict(section="k8", item="K8 = √(H·√(K2·K3)) против K8.parquet V4 (SBER)", n=int(sv.notna().sum()), ok=same, value=float(np.nanmax(np.abs(sv.to_numpy(float) - w["K8"].to_numpy(float)))))


def shift_rows(w: pd.DataFrame) -> pd.DataFrame:
    """Проверка сдвига: настоящий ряд проходит, мутация «прогноз сдвинут на +1 день» не проходит (K1, K2, K3, M, K8, D2)."""
    rv = pd.Series(w["rv"].to_numpy(float))
    rows = []
    for s in SER:
        f = pd.Series(w[s].to_numpy(float))
        real = AU.shift_ok(rv, f)
        mut = AU.shift_ok(rv.iloc[:-1], f.shift(-1).iloc[:-1])
        rows.append(dict(section="shift", item=s, real_ok=bool(real), mutation_plus1_passes=bool(mut), ok=bool(real and not mut)))
    return pd.DataFrame(rows)


def ref_rows() -> tuple:
    _, b = v4_bench.run()
    rows = [dict(section="ref_0.2081", item=k, value=float(v) if not isinstance(v, bool) else float(v)) for k, v in b.items() if k in ("repro_diff", "n_ref", "n_common", "rv_mismatch", "t_plus_mismatch", "xj_vs_k2_maxdiff", "lf_vs_k3_maxdiff")]
    return pd.DataFrame(rows), b


def controls_table(w: pd.DataFrame) -> tuple:
    f, s, sh = file_rows(), series_rows(w), shift_rows(w)
    k8 = k8_row(w)
    rf, b = ref_rows()
    out = pd.concat([f, s, pd.DataFrame([k8]), sh, rf], ignore_index=True)
    res = dict(files_ok=bool(f["ok"].all()), series_ok=bool(s["ok"].all()), k8_bitwise=k8["ok"], shift_ok=bool(sh["ok"].all()), ref_ok=bool(b["repro_ok"] and b["n_common"] == 2271 and b["xj_vs_k2_maxdiff"] == 0.0 and b["lf_vs_k3_maxdiff"] == 0.0),
               qlike={r["item"]: r["qlike_all"] for _, r in s.iterrows()}, n_dates=int(s["n"].min()), ref=b)
    return out, res
