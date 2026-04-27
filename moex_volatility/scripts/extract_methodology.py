#!/usr/bin/env python3
"""extract_methodology.py — Extract dominant parameters for dissertation Methodology section."""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import ast

BASE = Path(__file__).resolve().parent.parent
SNAMES = {"S1":"S1_MeanRev","S2":"S2_Bollinger","S3":"S3_Donchian",
          "S4":"S4_Supertrend","S5":"S5_PivotPoints","S6":"S6_VWAP"}
SIDS = ["S1","S2","S3","S4","S5","S6"]
SID_MAP = {v:k for k,v in SNAMES.items()}


def parse_dict_or_kv(s):
    """Parse either {'k': v} or 'k=v k=v' format."""
    s = str(s).strip()
    if not s or s == 'nan': return {}
    try:
        d = ast.literal_eval(s)
        if isinstance(d, dict): return d
    except: pass
    d = {}
    for part in s.split():
        if "=" in part:
            k, v = part.split("=", 1)
            try: v = float(v)
            except: pass
            d[k] = v
    return d


def analyze_params(df, label):
    """Analyze parameter patterns for a set of upper bound results."""
    print(f"\n{'#'*80}")
    print(f"# {label}")
    print(f"{'#'*80}")

    for sid in SIDS:
        sname = SNAMES[sid]
        sub = df[df["strategy"] == sname].copy()
        if len(sub) == 0: continue

        print(f"\n{'='*70}")
        print(f"  {sname} — {label} ({len(sub)} тикеров)")
        print(f"{'='*70}")

        # Parse sig_params and filt_params
        sig_list = [parse_dict_or_kv(r) for r in sub.get("sig_params", [])]
        filt_list = [parse_dict_or_kv(r) for r in sub.get("filt_params", [])]

        # ── Signal parameters ──
        print(f"\n  Сигнальные параметры:")
        if sid in ("S1", "S2"):
            ma_types = [s.get("ma_type", "?") for s in sig_list if s.get("ma_type")]
            windows = [s.get("w", 0) for s in sig_list if s.get("w", 0) > 0]
            if ma_types:
                c = Counter(ma_types)
                print(f"    MA type: {c.most_common()}")
            if windows:
                print(f"    Window: median={np.median(windows):.0f}, "
                      f"range=[{min(windows)}, {max(windows)}], mean={np.mean(windows):.0f}")
            if sid == "S1":
                devs = [s.get("dev_val", 0) for s in sig_list if s.get("dev_val", 0) > 0]
                dev_types = [s.get("dev_type", "?") for s in sig_list if s.get("dev_type")]
                if devs: print(f"    Entry deviation: median={np.median(devs):.2f}, range=[{min(devs):.2f}, {max(devs):.2f}]")
                if dev_types: print(f"    Dev type: {Counter(dev_types).most_common()}")
            if sid == "S2":
                bs = [s.get("bs", 0) for s in sig_list if s.get("bs", 0) > 0]
                if bs: print(f"    Band std: median={np.median(bs):.2f}, range=[{min(bs):.2f}, {max(bs):.2f}]")
        elif sid == "S3":
            ws = [s.get("w", 0) for s in sig_list if s.get("w", 0) > 0]
            if ws: print(f"    Donchian window: median={np.median(ws):.0f}, range=[{min(ws)}, {max(ws)}]")
        elif sid == "S4":
            aps = [s.get("ap", 0) for s in sig_list if s.get("ap", 0) > 0]
            mults = [s.get("mult", 0) for s in sig_list if s.get("mult", 0) > 0]
            if aps: print(f"    ATR period: median={np.median(aps):.0f}, range=[{min(aps)}, {max(aps)}]")
            if mults: print(f"    Multiplier: median={np.median(mults):.2f}, range=[{min(mults):.2f}, {max(mults):.2f}]")
        elif sid == "S5":
            pts = [s.get("pt", "?") for s in sig_list if s.get("pt")]
            els = [s.get("el", 0) for s in sig_list if s.get("el", 0) > 0]
            if pts: print(f"    Pivot type: {Counter(pts).most_common()}")
            if els: print(f"    Entry level: {Counter(els).most_common()}")
        elif sid == "S6":
            vws = [s.get("vw", 0) for s in sig_list if s.get("vw", 0) > 0]
            dms = [s.get("dm", 0) for s in sig_list if s.get("dm", 0) > 0]
            if vws: print(f"    VWAP window: median={np.median(vws):.0f}, range=[{min(vws)}, {max(vws)}]")
            if dms: print(f"    Dev mult: median={np.median(dms):.2f}, range=[{min(dms):.2f}, {max(dms):.2f}]")

        # ── SL/TP ──
        print(f"\n  Risk Management:")
        sl_types = sub["sl_type"].value_counts().to_dict() if "sl_type" in sub.columns else {}
        tp_types = sub["tp_type"].value_counts().to_dict() if "tp_type" in sub.columns else {}
        sl_vals = sub["sl_val"].dropna().values if "sl_val" in sub.columns else []
        tp_vals = sub["tp_val"].dropna().values if "tp_val" in sub.columns else []
        mh_vals = sub["max_hold"].dropna().values if "max_hold" in sub.columns else (
                  sub["rm_mh"].dropna().values if "rm_mh" in sub.columns else [])
        be_vals = sub["be_atr"].dropna().values if "be_atr" in sub.columns else (
                  sub["rm_be"].dropna().values if "rm_be" in sub.columns else [])

        if sl_types: print(f"    SL type: {dict(sl_types)}")
        if len(sl_vals) > 0:
            sv = sl_vals[sl_vals > 0]
            if len(sv) > 0: print(f"    SL value: median={np.median(sv):.2f}, range=[{min(sv):.2f}, {max(sv):.2f}]")
        if tp_types: print(f"    TP type: {dict(tp_types)}")
        if len(tp_vals) > 0:
            tv = tp_vals[tp_vals > 0]
            if len(tv) > 0: print(f"    TP value: median={np.median(tv):.2f}, range=[{min(tv):.2f}, {max(tv):.2f}]")
        if len(mh_vals) > 0:
            mv = mh_vals[mh_vals > 0]
            if len(mv) > 0: print(f"    Max hold: median={np.median(mv):.0f}, range=[{min(mv):.0f}, {max(mv):.0f}]")
        if len(be_vals) > 0:
            bv = be_vals[be_vals > 0]
            if len(bv) > 0: print(f"    Break-even ATR: median={np.median(bv):.1f}, range=[{min(bv):.1f}, {max(bv):.1f}]")

        # ── Filters ──
        print(f"\n  Фильтры:")
        filter_keys = ["adx_th","atr_lo","vol_th","rsi_th","mom_n","dow"]
        for fk in filter_keys:
            vals = [f.get(fk, 0) for f in filt_list if f.get(fk, 0) > 0]
            if vals:
                print(f"    {fk}: used {len(vals)}/{len(filt_list)} тикеров, "
                      f"median={np.median(vals):.1f}, range=[{min(vals):.1f}, {max(vals):.1f}]")

        # ── Performance ──
        print(f"\n  Результаты:")
        exsh = sub["ex_sharpe"].values
        print(f"    ExSh: mean={np.mean(exsh):.2f}, median={np.median(exsh):.2f}, "
              f"range=[{min(exsh):.2f}, {max(exsh):.2f}]")
        if "exposure" in sub.columns:
            exp = sub["exposure"].dropna().values
            if len(exp) > 0: print(f"    Exposure: mean={np.mean(exp):.0f}%")
        if "wr" in sub.columns:
            wr = sub["wr"].dropna().values
            if len(wr) > 0: print(f"    Win rate: mean={np.mean(wr):.0f}%")
        if "tpy" in sub.columns:
            tpy = sub["tpy"].dropna().values
            if len(tpy) > 0: print(f"    Trades/year: mean={np.mean(tpy):.0f}")


def main():
    print("=" * 80)
    print("METHODOLOGY EXTRACTION — Parameter patterns for dissertation")
    print("=" * 80)

    # Daily
    d_path = BASE / "results/final/strategies/walkforward_v4/daily_a/upper_bound_params.csv"
    if d_path.exists():
        d_df = pd.read_csv(d_path)
        analyze_params(d_df, "ДНЕВНОЙ ТАЙМФРЕЙМ")
    else:
        print(f"  Daily CSV not found: {d_path}")

    # Hourly
    h_path = BASE / "results/final/strategies/walkforward_v4/hourly_a/hourly_upper_bound.csv"
    if h_path.exists():
        h_df = pd.read_csv(h_path)
        analyze_params(h_df, "ЧАСОВОЙ ТАЙМФРЕЙМ")
    else:
        print(f"  Hourly CSV not found: {h_path}")

    # ── Vol scaling (C) methodology ──
    print(f"\n{'#'*80}")
    print(f"# ПОДХОД C (Vol Scaling) — Методология")
    print(f"{'#'*80}")
    print(f"""
  Метод: Inverse Volatility Scaling (Barroso & Santa-Clara, 2015)

  Формула: position_C(t) = position_A(t) × (σ_target / σ_forecast(t))^power

  Параметры оптимизации:
    σ_forecast: rolling std(returns, window=20d/90h)
    σ_target: медиана σ на всём периоде
    power: 1.0 (линейное обратное масштабирование)
    min_position: 0.1 × base (не уменьшать ниже 10%)
    max_position: 2.0 × base (не увеличивать выше 200%)

  Эффект:
    - Высокая волатильность → уменьшение позиции → снижение риска
    - Низкая волатильность → увеличение позиции → рост exposure
    - Свободный капитал → FDR (ставка ЦБ / 365 × calendar_days)
    - Vol стабилизируется → Sharpe ratio растёт

  Статистическая значимость:
    - Paired bootstrap 10,000 итераций
    - p < 0.0001 для всех 12 комбинаций (6 стратегий × 2 таймфрейма)
    - 95% CI нижняя граница > 0 во всех случаях

  Теоретическая база:
    - Barroso, P., Santa-Clara, P. (2015). Momentum has its moments. JFE
    - Daniel, K., Moskowitz, T. (2016). Momentum crashes. JFE
    - Moreira, A., Muir, T. (2017). Volatility-managed portfolios. JF
    - Harvey, C.R. et al. (2018). The impact of volatility targeting. JFM
""")


if __name__ == "__main__":
    main()
