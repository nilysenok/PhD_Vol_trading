"""V5-0, G1: замороженные входы программы восстановлены побитно — H = K1, K2, K3, M = sqrt(K2·K3), D2, K8 = sqrt(H·M) для 17 бумаг. Только чтение: SHA файлов сверяются с `INVENTORY.csv` V4, K8 пересчитывается и сравнивается
с сохранённым `K8.parquet` V4 побитно, `crit_panel` — с ожидаемыми значениями и с `CHECKPOINT.json` V4, ориентир SBER 0,2081 — `v4_bench.run()` (чтение источника Ж-211 в памяти). Переобучения нет."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import v4_audit as AU, v4_bench, v4_fin_data as FD, v4_io as IO4, v4_rqc_calib as RQ
from models_v2.s12 import v5_0_io as IO

EXPECT = {"k1": 0.26220, "k2": 0.26821, "k3": 0.26820, "d2": 0.261451, "K8": 0.26098}
SER = ["k1", "k2", "k3", "M", "d2", "K8"]


def sha_rows() -> pd.DataFrame:
    """SHA-256 входных файлов сейчас против записанных в INVENTORY V4 (K1, K2, K3, LGBM-D2 по 17 бумагам)."""
    inv = IO4.read_csv("INVENTORY.csv")
    inv = inv[inv["model"] != "H∩K2∩K3"]
    now = [IO.sha_file(IO.ROOT / f) for f in inv["file"]]
    return pd.DataFrame(dict(section="file", item=inv["ticker"] + ":" + inv["model"], sha256=now, sha_v4=inv["sha256"].to_numpy(), ok=[a == b for a, b in zip(now, inv["sha256"])]))


def k8_rows(k: pd.DataFrame) -> tuple:
    """K8 = sqrt(H·sqrt(K2·K3)) пересчитан против сохранённого K8.parquet (поле F) на всех строках: побитное равенство."""
    sv = pd.read_parquet(FD.OUT / "K8.parquet")[["ticker", "date", "F"]]
    m = k[["ticker", "date", "k1", "k2", "k3"]].merge(sv, on=["ticker", "date"], how="outer", indicator=True)
    m["K8_new"] = RQ.k8(m["k1"], m["k2"], m["k3"])
    both = m[m["_merge"] == "both"]
    same = bool(np.array_equal(both["F"].to_numpy(float), both["K8_new"].to_numpy(float)))
    return same, float(np.abs(both["F"] - both["K8_new"]).max()), int(len(both)), int((m["_merge"] != "both").sum())


def coverage_rows(k: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, g in k.groupby("ticker"):
        rows.append(dict(section="ticker", item=t, n=len(g), **{"finite_" + c: float(np.isfinite(g[c]).mean()) for c in SER if c in g}))
    return pd.DataFrame(rows)


def crit_rows(k: pd.DataFrame) -> tuple:
    cols = FD.CTRL + FD.CFGS
    cp = S.crit_panel(k, cols).set_index("series")["crit_panel"]
    v4 = json.loads((IO.V4REP / "CHECKPOINT.json").read_text())["B_controls_crit_panel"]
    rows = [dict(section="crit", item=s, crit_panel=float(cp[s]), expected_5dp=e, ok=bool(round(float(cp[s]), 6 if s == "d2" else 5) == e), v4_checkpoint=v4.get(s, np.nan),
                 diff_v4=abs(float(cp[s]) - v4[s]) if s in v4 else np.nan) for s, e in EXPECT.items()]
    return pd.DataFrame(rows), cp


def shift_rows() -> pd.DataFrame:
    """Проверка сдвига (Д1) на каждой бумаге: настоящие ряды проходят, мутация «прогноз сдвинут на +1 день» (K2, K3) ловится."""
    rows = []
    for t in IO.T17:
        a = IO4.aligned(t)
        real = AU.shift_ok(a["rv"], a["k2"]) and AU.shift_ok(a["rv"], a["k3"])
        mut = AU.shift_ok(a["rv"].iloc[:-1], a["k2"].shift(-1).iloc[:-1]) or AU.shift_ok(a["rv"].iloc[:-1], a["k3"].shift(-1).iloc[:-1])
        rows.append(dict(section="shift", item=t, real_ok=bool(real), mutation_plus1_passes=bool(mut)))
    return pd.DataFrame(rows)


def sber_rows() -> tuple:
    _, b = v4_bench.run()
    rows = [dict(section="sber", item=k_, value=v) for k_, v in b.items() if k_ in ("repro_diff", "n_ref", "n_common", "rv_mismatch", "t_plus_mismatch", "xj_vs_k2_maxdiff", "lf_vs_k3_maxdiff", "repro_ok")]
    return pd.DataFrame(rows), b


def controls_table(k: pd.DataFrame) -> tuple:
    """-> (CONTROLS, резюме). Резюме: all_files_ok, k8_bitwise, crit_ok, shift_ok, sber_ok."""
    f, cov, sh = sha_rows(), coverage_rows(k), shift_rows()
    same, mx, nb, nout = k8_rows(k)
    cr, cp = crit_rows(k)
    sb, b = sber_rows()
    aud = IO4.read_csv("AUDIT_ALIGN.csv")
    k8r = pd.DataFrame([dict(section="k8", item="bitwise_vs_V4_K8.parquet", ok=same, value=mx, n=nb, note="строк вне пересечения: %d" % nout)])
    au = pd.DataFrame([dict(section="audit_v4", item="align_ok&oof_k2k3 (17 бумаг)", ok=bool(aud["align_ok"].all() and aud["oof_k2k3"].all() and len(aud) == 17))])
    out = pd.concat([f, cov, sh, k8r, cr, sb, au], ignore_index=True)
    sber_ok = bool(b["repro_ok"] and b["n_common"] == 2271 and abs(b["repro_diff"]) < 1e-4)
    res = dict(files_ok=bool(f["ok"].all()), n_files=len(f), k8_bitwise=same, crit_ok=bool(cr["ok"].all()), crit_v4_diff=float(cr["diff_v4"].max()), shift_ok=bool(sh["real_ok"].all() and not sh["mutation_plus1_passes"].any()),
               sber_ok=sber_ok, n_tickers=int(k["ticker"].nunique()), audit_ok=bool(au["ok"].iloc[0]), crit=cp, sber=b)
    return out, res
