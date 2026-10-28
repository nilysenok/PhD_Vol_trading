"""Ряды МСП для рисунков главы 4 на разработке 2018–2021 (тест стратегий закрыт и не запускается).

Основной дизайн fastbt/designs/final/main.yaml (A/B/C/D одним прогоном, --scope dev: A 2017–2021,
B/C/D 2018–2021, валидация с 2017), σ_pred h=1 — финал главы 3 (final3 = D22_h1 или FINAL, если есть).
σ_pred обрезается по 31.12.2021 (как fastbt.s9: σ тестовых лет не читается пайплайном). Выход —
<MAIN>/data/fastbt/figures/: позиции dev_main_<tag>.parquet, обрезанный σ, ряды
dev_<tag>/perstrat_daily_returns (fastbt.export, годы 2018–2021). Журнал dev_log_strategies не
пишется: это не новый дизайн, а прогон зафиксированного кандидата для рисунков.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from figures_final_common import MAIN, ROOT

REGEN = MAIN / "data/regen"
OUT = MAIN / "data/fastbt/figures"
CUT = pd.Timestamp("2021-12-31")
YEARS = (2018, 2019, 2020, 2021)
SIGMA = {"final3": REGEN / "sigma_pred_h1_final3.parquet", "FINAL": REGEN / "sigma_pred_h1_FINAL.parquet"}
MAIN_YAML = ROOT / "fastbt/designs/final/main.yaml"
S9_REF = {"A": (-0.9262, "s9 final3 (run_log_final3.json; s9_sravnenie_sigma_final3.csv, A)"),
          "C": (-0.6144, "s9 final3 (s9_sravnenie_sigma_final3.csv, C u02 h1)")}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cut_sigma(src, tag):
    d = pd.read_parquet(src)
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= CUT].reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"sigma_pred_h1_{tag}_do2021.parquet"
    d.to_parquet(p, index=False)
    return p, d


def compare_sigma(tag_a, tag_b):
    """Совпадение σ_pred двух файлов на 2018–2021 (по паре дата-тикер)."""
    a, b = (pd.read_parquet(SIGMA[t]) for t in (tag_a, tag_b))
    for d in (a, b):
        d["date"] = pd.to_datetime(d["date"])
    m = a.merge(b, on=["date", "ticker"], how="outer", suffixes=("_a", "_b"), indicator=True)
    m = m[m.date.dt.year.between(2018, 2021)]
    both = m[m._merge == "both"]
    diff = (both.sigma_pred_h1_a - both.sigma_pred_h1_b).abs()
    return dict(rows_a=int((m._merge != "right_only").sum()), rows_b=int((m._merge != "left_only").sum()),
                only_one=int((m._merge != "both").sum()), max_abs=float(diff.max()) if len(diff) else np.nan)


def build(tag="final3", workers=4):
    """Позиции + ряды для tag; кэш по sha σ и кода fastbt. Возвращает (папка рядов, refs)."""
    src = SIGMA[tag]
    sp, sig = cut_sigma(src, tag)
    os.environ["FASTBT_VPRED"] = str(sp)            # до первого импорта fastbt.data
    sys.path.insert(0, str(ROOT))
    from fastbt import dev, export, msp
    from fastbt.cache import code_hash
    from fastbt.data import TICKERS, VPRED
    from fastbt.pipeline import run_all
    from fastbt.report import portfolios, summary
    assert Path(VPRED) == sp, f"fastbt.data.VPRED = {VPRED}, ожидался {sp}"
    key = hashlib.sha1((sha(sp) + code_hash() + MAIN_YAML.read_text()).encode()).hexdigest()[:10]
    pos_p, series = OUT / f"dev_main_{tag}.parquet", OUT / f"dev_{tag}" / "perstrat_daily_returns"
    meta_p = OUT / f"dev_main_{tag}.json"
    design, _ = dev.load_design(MAIN_YAML)
    design["vpred"] = str(sp)
    cfg = dev.make_config(design, "dev")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    if meta.get("key") == key and pos_p.exists():
        pos = pd.read_parquet(pos_p)
    else:
        pos, _ = run_all(cfg, design.get("tickers", TICKERS), design.get("sids", ["S1", "S2", "S3", "S4", "S5", "S6"]),
                         workers, log=lambda m: None)
        pos.to_parquet(pos_p, index=False)
    pos["date"] = pd.to_datetime(pos["date"])
    export.export(pos, series, msp.rate_cbr(), years=(YEARS[0], YEARS[-1]))
    ports = portfolios(pos, cfg, list(YEARS))
    _, deltas = summary(pos, cfg, list(YEARS), 1000)
    refs = dict(tag=tag, sigma=str(src), sigma_sha=sha(src), cut=str(sp), key=key,
                sharpe={ap: msp.sharpe(p["excess"].values) for ap, p in ports.items()},
                sharpe_years={ap: {y: msp.sharpe(p.loc[p.index.year == y, "excess"].values) for y in YEARS}
                              for ap, p in ports.items()},
                dsh={r["delta"].split("-")[0]: float(r["obs"]) for _, r in deltas.iterrows() if r["delta"].endswith("-A")},
                s9=S9_REF, sigma_rows=len(sig), sigma_max=str(sig.date.max().date()))
    from fastbt.final_test import ce
    refs["dce_C_g10"] = float((ce(ports["C"]["total"].values, 10) - ce(ports["A"]["total"].values, 10)) * 252 * 100)
    meta_p.write_text(json.dumps(dict(key=key, sigma_sha=refs["sigma_sha"], sharpe=refs["sharpe"]), ensure_ascii=False, indent=1))
    return series, refs
