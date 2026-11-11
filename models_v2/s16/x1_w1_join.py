"""W1 (Ж-250 п. 1): склейка архива и продолжения 2026 по СОХРАНЁННЫМ прогнозам (обучения нет). Архив T0-рядов — те же файлы и sha256, что у T0 (`x1_t0.load_all`); ZB, ZB0 — `data/fwd/models/*.parquet`;
продолжение — `data/fwd/models/new/*.parquet` (t⁺ 2026-03-02…2026-11-10). Склейка по паре (d, t⁺); дубль d или t⁺ — остановка. RV дня t⁺ у всех рядов обязана совпасть на общих датах (иначе стоп с перечнем дат)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import models_v2.s16  # noqa: F401  (torch первым)
from models_v2.s16 import x1_t0 as T0

MODELS = T0.ROOT / "data/fwd/models"
NEW = MODELS / "new"
ARCH_END = pd.Timestamp("2026-02-27")
ALIAS = {"V6:A2": "A2", "V4:NTP2": "NTP2", "V7:cfg6": "cfg6", "V7:cfg8": "cfg8"}
FWD = {"K1": ("k1", {"target_date": "t_plus", "sigma2": "F"}), "K2": ("k23", {"date": "d", "sigma2_k2": "F"}), "K3": ("k23", {"date": "d", "sigma2_k3": "F"}),
       "X1": ("x1", {}), "cfg8": ("cfg8", {}), "ZB": ("zb", {}), "ZB0": ("zb0", {})}
RULE = {"K1": "МНК на всех парах t⁺ ≤ d, переоценка каждый день (протокол §3)",
        "K2": "дневная переобучка на расширяющемся окне пар t⁺ ≤ d, ранняя остановка И1, seed 42; И2 2026 — из метаданных (протокол §3)", "K3": "то же, что K2 (LightGBM; протокол §3)",
        "X1": "π каждый день парами t⁺ ≤ d; α помесячно по 126 закрытым парам (протокол §3)", "cfg8": "первый торговый день месяца: пары t⁺ ≤ d0, 5 seeds, z от энкодера года переоценки (протокол §3)",
        "ZB": "помесячно: пул t⁺ ≤ D, пары e_d = 1, L-BFGS-B (протокол §3)", "ZB0": "как ZB, колонка a = 0 в файле (протокол §3)"}
RELEASE = "прогноз дня d — по данным не позже закрытия d (обрезка begin ≤ d 18:40, `s17/fwd_src`); t⁺ — следующий канонический день (`calendar.parquet`, протокол §3)"


def read_fwd(path, ren: dict) -> pd.DataFrame:
    """Файл прогнозов -> d, t_plus, rv, F; нечисловые строки отброшены как в T0, повтор d — остановка."""
    df = pd.read_parquet(path).rename(columns=ren)[["d", "t_plus", "rv", "F"]].copy()
    df["d"], df["t_plus"] = pd.to_datetime(df["d"]), pd.to_datetime(df["t_plus"])
    df = df[np.isfinite(df["rv"]) & np.isfinite(df["F"]) & (df["F"] > 0) & df["t_plus"].notna()].sort_values("d").reset_index(drop=True)
    if df["d"].duplicated().any():
        raise SystemExit("%s: повторные даты" % path)
    return df


def archive_frames() -> tuple:
    """-> (кадры архива по именам W1, sha256 исходных файлов). T0-ряды — источники T0; ZB, ZB0 — fwd-архив."""
    fr, _ = T0.load_all()
    out = {ALIAS.get(k, k): v for k, v in fr.items()}
    sha = {ALIAS.get(k, k): T0.sha_file(T0.REG / T0.SPEC[k][1]) for k in fr}
    for k in ("ZB", "ZB0"):
        p = MODELS / (FWD[k][0] + ".parquet")
        out[k], sha[k] = read_fwd(p, FWD[k][1]), T0.sha_file(p)
    return out, sha


def new_frames() -> tuple:
    out, sha = {}, {}
    for k, (f, ren) in FWD.items():
        p = NEW / (f + ".parquet")
        out[k], sha[k] = read_fwd(p, ren), T0.sha_file(p)
    return out, sha


def stitch(a: pd.DataFrame, n: pd.DataFrame) -> pd.DataFrame:
    """Архив + продолжение: без дублей d и t⁺ (дубль — остановка, а не молчаливое удаление)."""
    a, n = a[a["t_plus"] <= ARCH_END].assign(part="archive"), n.assign(part="new")
    x = pd.concat([a, n], ignore_index=True).sort_values("d").reset_index(drop=True)
    if x["d"].duplicated().any() or x["t_plus"].duplicated().any():
        raise SystemExit("склейка: дубль d или t⁺ (даты: %s)" % sorted(set(x.loc[x["d"].duplicated(keep=False), "d"].dt.date.astype(str)) | set(x.loc[x["t_plus"].duplicated(keep=False), "t_plus"].dt.date.astype(str))))
    return x


def rv_cross_check(fr: dict, ref: str = "K1") -> None:
    """RV дня t⁺ одинакова во всех рядах на каждой общей дате, в том числе на продолжении; иначе — стоп с перечнем дат."""
    r = fr[ref].set_index("d")["rv"]
    bad = {}
    for k, df in fr.items():
        m = df.set_index("d")["rv"]
        idx = m.index.intersection(r.index)
        off = ~np.isclose(m.loc[idx].to_numpy(float), r.loc[idx].to_numpy(float), rtol=1e-12, atol=0)
        if off.any():
            bad[k] = [str(x.date()) for x in idx[off]]
    if bad:
        raise SystemExit("RV расходится с %s: %s" % (ref, bad))


def calendar_next() -> pd.Series:
    c = pd.read_parquet(MODELS / "calendar.parquet")
    return c.assign(date=pd.to_datetime(c["date"])).set_index("date")["t_plus"]


def ref_rv() -> pd.Series:
    """Независимый источник RV по дням: дневной ряд архива (daily_SBER) + запечатанная RV продолжения (замок открыт, Л6)."""
    from models_v2.s17.fwd_src import Src
    a = pd.read_parquet(T0.REG / "harj_panel_hybrid/daily_SBER.parquet")["rv"]
    s = Src().sealed_rv()["rv"]
    out = pd.concat([a, s[s.index > a.index.max()]])
    out.index = pd.to_datetime(out.index)
    return out


def join_rows(fr: dict, arch: dict, new: dict, sha_a: dict, sha_n: dict, fwd_arch: dict) -> pd.DataFrame:
    """Строка на ряд: дубли, стык, календарь d → t⁺, RV против общего ряда и независимого источника, сверка архивных строк с fwd-файлами, правило обновления."""
    nxt, ref, rows = calendar_next(), ref_rv(), []
    for k, x in fr.items():
        a, n = x[x["part"] == "archive"], x[x["part"] == "new"]
        la, fn = (a["t_plus"].max(), n["t_plus"].min()) if len(n) else (a["t_plus"].max(), pd.NaT)
        cal = nxt.reindex(x["d"]).to_numpy()
        rv_ref = ref.reindex(x["t_plus"]).to_numpy(float)
        fa = fwd_arch.get(k)
        diff = float("nan")
        if fa is not None:
            m = a.merge(fa, on="d", suffixes=("", "_f"))
            diff = float(max(np.abs(m["F"] - m["F_f"]).max(), np.abs(m["rv"] - m["rv_f"]).max())) if len(m) == len(a) else float("inf")
        rows.append(dict(series=k, archive_sha256=sha_a[k], new_sha256=sha_n.get(k, ""), n_archive=len(a), n_new=len(n), d_first=str(x["d"].min().date()), d_last=str(x["d"].max().date()),
                         tp_first=str(x["t_plus"].min().date()), tp_last=str(x["t_plus"].max().date()), dup_d=int(x["d"].duplicated().sum()), dup_tp=int(x["t_plus"].duplicated().sum()),
                         last_arch_tp=str(la.date()), first_new_tp=str(fn.date()) if len(n) else "", stitch_ok=bool(len(n) == 0 or nxt.get(la) == fn),
                         calendar_bad=int((pd.DatetimeIndex(cal) != pd.DatetimeIndex(x["t_plus"])).sum()), rv_vs_k1_bad=0,
                         rv_vs_daily_bad=int((np.isfinite(rv_ref) & ~np.isclose(x["rv"].to_numpy(float), rv_ref, rtol=1e-9, atol=0)).sum()),
                         rv_ref_missing=";".join(str(t.date()) for t in x.loc[~np.isfinite(rv_ref), "t_plus"]), archive_vs_fwd_maxdiff=diff,
                         update_rule=RULE.get(k, "только архив: продолжения нет (T0, справочно)"), release=RELEASE))
    return pd.DataFrame(rows)


def load_joined() -> tuple:
    """-> (склеенные кадры, строки W1_JOIN, sha256 входов). RV сверена между рядами (стоп при расхождении)."""
    arch, sha_a = archive_frames()
    new, sha_n = new_frames()
    fwd_arch = {k: read_fwd(MODELS / (f + ".parquet"), ren) for k, (f, ren) in FWD.items() if k not in ("ZB", "ZB0")}
    fr = {k: stitch(arch[k], new[k]) if k in new else arch[k].assign(part="archive") for k in arch}
    rv_cross_check(fr)
    return fr, join_rows(fr, arch, new, sha_a, sha_n, fwd_arch), (sha_a, sha_n)
