"""X1-пилот, лист T0 (Ж-242 пп. 1–2): единая таблица QLIKE SBER, h = 1, по СОХРАНЁННЫМ прогнозам (обучения нет, прогнозы не пишутся).
Дневная потеря l = RV/F − ln(RV/F) − 1 (`models.metrics.qlike`, как в V6/V7/DE); QLIKE окна = среднее дневных l по датам окна на ОБЩИХ датах всех рядов; окна по году t⁺.
Δ — в % QLIKE БАЗЫ (колонка `base`, знаменатель `base_qlike`); 90 % интервал блоковый бутстрап (20, 2000, seed 0), DM лаг 5, Холм N = 738 / N_wide = 3181, MCS 90 % — реализации V6/V7 импортом.
Запуск: `python3 -W ignore -m models_v2.s16.x1_t0 [--write | --check]`; --check ничего не пишет."""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import models_v2.s16  # noqa: F401  (torch первым, до models_v2.s14)
from models_v2.s13 import v6_0_q as Q, v6_1_eval as EVX

ST = importlib.import_module("models_v2.s14.v7_1m_stat")
ROOT = Path(__file__).resolve().parents[2]
REG, REP = ROOT / "data/regen", ROOT / "docs/otchety/harj_x1"
N_HOLM, N_WIDE = 738, 3181
WINS = {"all": (0, 9999), "2018_2021": (2018, 2021), "2022_2025": (2022, 2025), "2026": (2026, 2026)}
YEARS = tuple(range(2018, 2027))
HYB = ("V6:A2", "V4:NTP2", "V7:cfg6", "V7:cfg8", "X1")
BASES = ("K1", "K2", "K3", "V4:NTP2")
HEAD = "# диагностика, не отбор; DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-242 пп. 1–2, лист T0 (SBER, h = 1; сохранённые прогнозы, обучения нет); l = RV/F − ln(RV/F) − 1, среднее по общим датам, окна по году t⁺\n"
SPEC = {"K1": ("HAR-J (K1)", "harj_panel_hybrid/har_SBER.parquet", {"target_date": "t_plus", "sigma2": "F"}),
        "K2": ("XGBoost (K2)", "harj_panel_hybrid/k23_SBER.parquet", {"date": "d", "rv_tplus": "rv", "sigma2_k2": "F"}),
        "K3": ("LightGBM (K3)", "harj_panel_hybrid/k23_SBER.parquet", {"date": "d", "rv_tplus": "rv", "sigma2_k3": "F"}),
        "V6:A2": ("V6 A2", "harj_hybrid_v6/A2.parquet", {}),
        "V4:NTP2": ("V4 NTP2", "harj_hybrid_v4/NTP2.parquet", {"date": "d"}),
        "V7:cfg6": ("V7 cfg6 (финалист по зарегистрированному правилу, конструкция B)", "harj_hybrid_v7/2/cfg6.parquet", {}),
        "V7:cfg8": ("V7 cfg8 (лучшая по QLIKE 2018–2021, выбрана ПОСЛЕ просмотра)", "harj_hybrid_v7/3/cfg8.parquet", {}),
        "X1": ("X1 (на базе cfg8)", "harj_deep/E1/x1.parquet", {})}


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def expected_sha() -> dict:
    c = pd.read_csv(ROOT / "docs/otchety/harj_hybrid_v6/cmp/CMP_SERIES.csv", comment="#")
    e = {k: c[c["series"] == k]["sha256_file"].iloc[0] for k in ("K1", "K2", "K3", "V6:A2", "V4:NTP2")}
    v7 = json.loads((ROOT / "docs/otchety/harj_hybrid_v7/CHECKPOINT.json").read_text(encoding="utf-8"))
    e["V7:cfg6"], e["V7:cfg8"] = v7["2"]["cfg6"]["sha256_file"], v7["3"]["cfg8"]["sha256_file"]
    e["X1"] = json.loads((ROOT / "docs/otchety/harj_deep/CHECKPOINT.json").read_text(encoding="utf-8"))["E1.x1"]["sha256"]
    return e


def load_all() -> tuple:
    """-> (кадры d, t_plus, rv, F без нечисловых строк, число сырых строк). Файлы сверяются по sha256 с CMP_SERIES / CHECKPOINT V7 / DE."""
    exp, fr, raw_n = expected_sha(), {}, {}
    for k, (_, rel, ren) in SPEC.items():
        p = REG / rel
        if sha_file(p) != exp[k]:
            raise SystemExit("%s: sha256 файла ≠ CMP_SERIES/CHECKPOINT" % k)
        raw = pd.read_parquet(p)
        raw = raw[raw["ticker"] == "SBER"] if "ticker" in raw else raw
        df = raw.rename(columns=ren)[["d", "t_plus", "rv", "F"]].copy()
        df["d"], df["t_plus"] = pd.to_datetime(df["d"]), pd.to_datetime(df["t_plus"])
        df = df[np.isfinite(df["rv"]) & np.isfinite(df["F"]) & (df["F"] > 0) & df["t_plus"].notna()].sort_values("d").reset_index(drop=True)
        if df["d"].duplicated().any():
            raise SystemExit("%s: повторные даты" % k)
        fr[k], raw_n[k] = df, len(raw)
    return fr, raw_n


def common(fr: dict) -> dict:
    """Пересечение дат d всех рядов; цель (rv) и t⁺ обязаны совпасть у всех."""
    idx = None
    for df in fr.values():
        idx = pd.DatetimeIndex(df["d"]) if idx is None else idx.intersection(pd.DatetimeIndex(df["d"]))
    al = {k: df.set_index("d").loc[idx].reset_index() for k, df in fr.items()}
    ref = al["K1"]
    for k, df in al.items():
        if not (df["t_plus"].equals(ref["t_plus"]) and np.allclose(df["rv"], ref["rv"], rtol=1e-12, atol=0)):
            raise SystemExit("%s: цель rv или t⁺ не совпадают с K1 на общих датах" % k)
    return al


def wmask(tp, lo: int, hi: int) -> np.ndarray:
    y = pd.DatetimeIndex(tp).year.to_numpy()
    return (y >= lo) & (y <= hi)


def win_q(rv, f, tp, lo: int, hi: int) -> float:
    """QLIKE окна = среднее ДНЕВНЫХ потерь по датам окна (не среднее показателей периодов)."""
    m = wmask(tp, lo, hi)
    return float(Q.ql(np.asarray(rv)[m], np.asarray(f)[m]).mean())


def qlike_table(fr: dict, al: dict, raw_n: dict) -> pd.DataFrame:
    ref = pd.DatetimeIndex(al["K1"]["d"])
    if not all(pd.DatetimeIndex(v["d"]).equals(ref) for v in al.values()):
        raise SystemExit("ряды таблицы заданы не на одних и тех же общих датах")
    mc = ST.mcs90(al["K1"]["rv"].to_numpy(float), {k: v["F"].to_numpy(float) for k, v in al.items()}, al["K1"]["t_plus"].to_numpy(), "all").set_index("series")
    rows = []
    for k, df in al.items():
        a, tp = fr[k], pd.DatetimeIndex(df["t_plus"])
        r = dict(series=k, label=SPEC[k][0], avail_first_d=str(a["d"].min().date()), avail_last_d=str(a["d"].max().date()), avail_first_tplus=str(a["t_plus"].min().date()),
                 avail_last_tplus=str(a["t_plus"].max().date()), avail_n=len(a), raw_rows=raw_n[k], common_first_d=str(df["d"].min().date()), common_last_d=str(df["d"].max().date()), has_2017=bool((a["t_plus"].dt.year == 2017).any()), n_2017=int((a["t_plus"].dt.year == 2017).sum()))
        for w, (lo, hi) in {**WINS, **{str(y): (y, y) for y in YEARS}}.items():
            m = wmask(tp, lo, hi)
            r["q_" + w], r["n_" + w] = win_q(df["rv"], df["F"], tp, lo, hi), int(m.sum())
            if w in WINS:
                r["first_" + w], r["last_" + w] = str(tp[m].min().date()), str(tp[m].max().date())
        r["mcs90_all"], r["mcs_p_all"] = bool(mc.at[k, "in_mcs"]), float(mc.at[k, "p"])
        rows.append(r)
    t = pd.DataFrame(rows)
    if t["n_all"].nunique() != 1 or t["first_all"].nunique() != 1 or t["last_all"].nunique() != 1:
        raise SystemExit("ряды таблицы заданы не на одних и тех же общих датах")
    return t


def stats_table(al: dict) -> pd.DataFrame:
    rv, tp = al["K1"]["rv"].to_numpy(float), pd.DatetimeIndex(al["K1"]["t_plus"])
    F, rows = {k: v["F"].to_numpy(float) for k, v in al.items()}, []
    for w, (lo, hi) in WINS.items():
        m = wmask(tp, lo, hi)
        for h in HYB:
            for b in (x for x in BASES if x != h):
                qh, qb = Q.ql(rv[m], F[h][m]), Q.ql(rv[m], F[b][m])
                bq = float(qb.mean())
                ci = EVX.delta_ci(qh, qb, bq, b=ST.B, block=ST.BLOCK, seed=ST.SEED)
                rows.append(dict(window=w, hybrid=h, base=b, base_qlike=bq, hybrid_qlike=float(qh.mean()), delta_q=float((qh - qb).mean()), delta_pct_of_base=ci["delta_pct"], lo90=ci["lo90"],
                                 hi90=ci["hi90"], p_one=EVX.dm_one(rv[m], F[b][m], F[h][m])["p_one"], n=int(m.sum()), first=str(tp[m].min().date()), last=str(tp[m].max().date())))
    t = pd.DataFrame(rows)
    t["p_holm_N"], t["p_holm_Nwide"] = np.nan, np.nan
    for w in WINS:
        k = (t["window"] == w) & t["p_one"].notna()
        t.loc[k, "p_holm_N"], t.loc[k, "p_holm_Nwide"] = EVX.holm_both(t.loc[k, "p_one"].to_numpy(), {"N": N_HOLM, "N_wide": N_WIDE})
    return t


def compute() -> tuple:
    from models_v2.s16 import x1_t0_doc as DOC
    fr, raw_n = load_all()
    al = common(fr)
    t, s = qlike_table(fr, al, raw_n), stats_table(al)
    return t, s, DOC.build(t, s)


def csv_text(df: pd.DataFrame) -> str:
    return HEAD + df.to_csv(index=False, lineterminator="\n")


def diff(want: pd.DataFrame, got_text: str, name: str) -> list:
    """Сверка таблицы с текстом CSV: колонки, строки, числа (rtol 1e-9)."""
    import io
    got = pd.read_csv(io.StringIO(got_text), comment="#")
    if list(got.columns) != list(want.columns) or len(got) != len(want):
        return ["%s: колонки или число строк" % name]
    bad = []
    for c in want.columns:
        if want[c].dtype.kind == "f":
            if not np.allclose(want[c].to_numpy(float), got[c].to_numpy(float), rtol=1e-9, atol=1e-12, equal_nan=True):
                bad.append("%s: %s" % (name, c))
        elif not (want[c].astype(str).reset_index(drop=True) == got[c].astype(str).reset_index(drop=True)).all():
            bad.append("%s: %s" % (name, c))
    return bad


def check(out: Path = REP) -> list:
    """Пересчёт из сохранённых прогнозов и сверка с файлами на диске; ничего не пишет."""
    t, s, md = compute()
    bad = []
    for nm, df in (("P0_TABLE.csv", t), ("P0_STATS.csv", s)):
        p = out / nm
        bad += diff(df, p.read_text(encoding="utf-8"), nm) if p.exists() else ["%s: нет файла" % nm]
    p = out / "P0_TABLE.md"
    return bad + ([] if p.exists() and p.read_text(encoding="utf-8") == md else ["P0_TABLE.md: текст ≠ генератору"])


def write(out: Path = REP) -> None:
    t, s, md = compute()
    out.mkdir(parents=True, exist_ok=True)
    (out / "P0_TABLE.csv").write_text(csv_text(t), encoding="utf-8")
    (out / "P0_STATS.csv").write_text(csv_text(s), encoding="utf-8")
    (out / "P0_TABLE.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    if "--check" in sys.argv:
        bad = check()
        print("T0 --check: %s" % ("расхождений нет" if not bad else "; ".join(bad)))
        sys.exit(1 if bad else 0)
    write()
    print("T0: записано в", REP)
