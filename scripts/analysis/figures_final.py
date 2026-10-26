#!/usr/bin/env python3
"""Рисунки глав 3–4 на финальных рядах после теста главы 3 (замена PR #5 со старым σ_pred).

  python3 scripts/analysis/figures_final.py [--out figures/final] [--sigma auto|final3|FINAL] [--workers 4]

Глава 3 — тестовые прогнозы финала (<MAIN>/data/regen/test_locked/final3/final_h{1,5,22}_test.parquet),
сверка с docs/tables/table_3_*_test.csv; тест главы 3 не перезапускается.
Глава 4 — только разработка 2018–2021: основной дизайн стратегий на σ_pred h=1 финала
(final3 = data/regen/sigma_pred_h1_final3.parquet; auto — FINAL, если файл есть, с проверкой совпадения
с final3 на 2018–2021). Тест стратегий 2022–2026 закрыт: тестовые версии помечены «ждёт теста стратегий».
Пишет fig_<гл>_<N>_<имя>.png + .csv, sverka.csv, sverka.md, status.csv, sigma.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import figures_final_ch3 as c3  # noqa: E402
import figures_final_ch4 as c4  # noqa: E402
import figures_final_ch4b as c4b  # noqa: E402
import figures_final_dev as dv  # noqa: E402
from figures_final_common import ROOT, Ctx  # noqa: E402


def md_table(ctx):
    rows = ["| Рис. | Показатель | На рисунке | Эталон | Δ | Источник эталона | Флаг |", "|---|---|---|---|---|---|---|"]
    for c in ctx.checks:
        rows.append(f"| {c['fig']} | {c['item']} | {c['figure_value']:.4f} | {c['ref']:.4f} | {c['diff']:+.2e} | {c['src']} | "
                    f"{'РАСХОЖДЕНИЕ' if c['flag'] else 'ок'} |")
    return "\n".join(rows)


def sigma_check(ctx, D1):
    """σ_pred final3 = sqrt(rv_pred) теста h=1 (тот же финал D22_h1)."""
    s = pd.read_parquet(dv.SIGMA["final3"])
    s["date"] = pd.to_datetime(s["date"])
    m = s.merge(D1[["date", "ticker", "rv_pred"]], on=["date", "ticker"], how="inner")
    v = float(np.max(np.abs(m.sigma_pred_h1 - np.sqrt(m.rv_pred))))
    ctx.chk("σ", f"max|σ_pred final3 − √rv_pred теста h=1| ({len(m)} строк)", v, 0.0, "final_h1_test.parquet", 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "figures/final"))
    ap.add_argument("--sigma", choices=["auto", "final3", "FINAL"], default="auto")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    ctx = Ctx(a.out)
    D = c3.run(ctx)
    sigma_check(ctx, D[1])
    tag = a.sigma if a.sigma != "auto" else ("FINAL" if dv.SIGMA["FINAL"].exists() else "final3")
    info = {"tag": tag}
    if dv.SIGMA["FINAL"].exists():
        info["FINAL_vs_final3_2018_2021"] = dv.compare_sigma("FINAL", "final3")
        ctx.chk("σ", "max|σ_pred FINAL − final3| 2018–2021", info["FINAL_vs_final3_2018_2021"]["max_abs"], 0.0, "sigma_pred_h1_FINAL.parquet", 1e-9)
    series, refs = dv.build(tag, a.workers)
    info.update({k: refs[k] for k in ("sigma", "sigma_sha", "cut", "key", "sigma_rows", "sigma_max", "sharpe", "sharpe_years", "dsh")})
    ctx.series, ctx.refs = series, refs
    ctx.period = "разработка 2018–2021"
    ctx.footer = f"Разработка 2018–2021 (тест стратегий закрыт); main.yaml, σ_pred h=1 {tag} (D22_h1), sha256 {refs['sigma_sha'][:12]}…, σ ≤ 31.12.2021"
    for s in (c4.fig_4_0, c4.fig_4_1, c4.fig_4_2, c4.fig_4_3, c4.fig_4_4, c4.fig_4_5, c4.fig_4_6, c4.fig_4_7,
              c4.fig_4_8, c4b.fig_4_9, c4b.fig_4_10, c4b.fig_4_11_14, c4b.fig_4_12, c4b.fig_4_13, c4b.fig_4_15_16):
        s(ctx)
    out = Path(a.out)
    pd.DataFrame(ctx.checks).to_csv(out / "sverka.csv", index=False)
    (out / "sverka.md").write_text(md_table(ctx) + "\n", encoding="utf-8")
    pd.DataFrame(ctx.status, columns=["figure", "status", "period", "note"]).to_csv(out / "status.csv", index=False)
    (out / "sigma.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    bad = [c for c in ctx.checks if c["flag"]]
    print(f"\nсверка: {len(ctx.checks)} показателей, расхождений > допуска: {len(bad)}")
    for c in bad:
        print(f"  {c['fig']} {c['item']}: {c['figure_value']:.4f} против {c['ref']:.4f}")
    print("Шарп МСП 2018–2021:", {k: round(v, 3) for k, v in refs["sharpe"].items()})


if __name__ == "__main__":
    main()
